"""Capacity ladder rescored on an independent TEST split (2026-09-23).

Why: knee_s3_ladder.py's two learned probes (query85k, mlp1M) pick their
reported step on the 2,000-event evaluation split of rig_P1_Enar_S100.pt
and report R^2 on the same events (qprobe_p5.py:292-300,
knee_s3_ladder.py:147-158). This script adds a third, independent
2,000-event split and reports R^2 on it at the step the evaluation split
selected. Nothing else changes: same cache for fitting and evaluation,
same probes, same seeds, same selection rule (min over the 100-step
evaluations of the mean of the three per-target median |error| on the
EVALUATION split).

Imported unchanged (no edits to any existing file):
  knee_s3_ladder  -- mlp_fit constants, query_targets, CACHE_PATH, CSV_PATH
  qprobe_p5       -- BigProbe, _prep, _eval_preds, _r2, extract_tokens,
                     fit constants (Adam 2e-3, batch 128, 2500 steps)
  wp1_probe_driver-- encoder_from_ckpt, encode_meanpool, gen_probe_set
  rig_hardening_sweep -- make_cfg, gen_probe_set (to regenerate the
                     fitting split and recover its lnE mean/sd)
  wp1_metrics     -- wp1_metric_row (pooled-linear ridge, alpha = 1)
Local variants (below) of mlp_fit and fit_readability are line-for-line
copies of the originals plus one addition: at every evaluation the test
split is also scored. Scoring the test split consumes no random numbers,
so the training trajectory is the original one.

Test split (--build-test): 2,000 events, gaussian latents (mu = 0),
N = 100 spots, rendered by wp1_probe_driver.gen_probe_set with
PROBE_N_SPOTS = 100 and the config.yml of the 21x mu = 0 run
rigb_lejp_p1s100_nano_1; seed base 20260992 (not among the existing
probe-cache bases). ln E is standardised with the ORIGINAL fitting
split's mean and sd, recovered by regenerating that split (seed base
20260910) exactly as rig_hardening_sweep.py did; the regenerated
standardised y_tr must equal the cached y_tr to 1e-5 or nothing is
written. Saved as a NEW file rig_P1_Enar_S100_test.pt (refuses to
overwrite).

Runs: the 13 knee runs with ladder rows that are 21x nano (nano_1-3,
long_nano_1) or 4x p384_1-9, at the PRE/POST epochs recorded in
results/knee_s3_ladder.csv; probe seeds 1, 2; query85k and mlp1M, plus the
deterministic pooled-linear ridge on the same checkpoints.

Output: results/knee_s3_ladder_split.csv (new; append-mode, resumable by
(run, phase, probe, probe_seed)). Columns as specified, plus
r2_*_final_eval (the final-step R^2 on the evaluation split).

Usage (from analysis/; set the GPU):
  CUDA_VISIBLE_DEVICES=1 python knee_s3_ladder_split.py --build-test
  CUDA_VISIBLE_DEVICES=1 python knee_s3_ladder_split.py
  python knee_s3_ladder_split.py --report      # reads the CSVs only
"""

import sys

sys.dont_write_bytecode = True      # leave analysis/__pycache__ untouched

import argparse  # noqa: E402
import csv  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
sys.path.insert(0, _HERE)

CACHE_DIR = paths.CACHE
TEST_CACHE = os.path.join(CACHE_DIR, 'rig_P1_Enar_S100_test.pt')
TEST_SEED = 20260992
N_TEST = 2000
TEST_N_SPOTS = 100
CFG_RUN = 'rigb_lejp_p1s100_nano_1xxx2i533m'   # 21x mu=0 run; its config renders the test split
# probe-cache seed bases already in use
USED_SEED_BASES = ([20260817, 20260818, 20260901, 20260902, 20260910,
                    20260911, 20260930, 20260931]
                   + list(range(20260940, 20260946))
                   + list(range(20260980, 20260986)))
RUN_RE = re.compile(r'^rigb_lejp_p1s100_(nano_[123]|long_nano_1|p384_[1-9])xxx')
OUT_CSV = os.path.join(paths.CSV, 'knee_s3_ladder_split.csv')
COLS = ['run', 'phase', 'epoch', 'probe', 'probe_seed', 'best_step',
        'r2_z1_eval', 'r2_z2_eval', 'r2_E_eval',
        'r2_z1_test', 'r2_z2_test', 'r2_E_test',
        'r2_z1_final_test', 'r2_z2_final_test', 'r2_E_final_test',
        'r2_z1_final_eval', 'r2_z2_final_eval', 'r2_E_final_eval']
PROBES = ('pooled_linear', 'query85k', 'mlp1M')


def f6(x):
    return f'{float(x):.6f}'


def ladder_runs(ladder_csv):
    """(run, phase, epoch) for the 13 runs, from the existing ladder CSV."""
    seen = {}
    with open(ladder_csv) as fh:
        for r in csv.DictReader(fh):
            if RUN_RE.match(r['run']):
                seen[(r['run'], r['phase'])] = int(r['epoch'])
    runs = sorted({k[0] for k in seen})
    assert len(runs) == 13, runs
    out = []
    for run in runs:
        for phase in ('pre', 'post'):
            out.append((run, phase, seen[(run, phase)]))
    return out


# ------------------------------------------------------------ test split

def chunk_seed_family(seed_base, n_chunks=40, chunk_stride=7919):
    """Every seed gen_probe_set derives from a seed base (both builders use
    the same derivations): chunk/truth seed, z seed, long-stats seed, spot
    seed (_sim_seed(seed, 0)). 40 chunks covers 10,000 events."""
    fam = set()
    for c in range(n_chunks):
        s = (seed_base * 100003 + c * chunk_stride + 1) & 0x7FFFFFFF
        fam |= {s, (s * 2654435761 + 97) & 0x7FFFFFFF,
                (s * 7919 + 31337) & 0x7FFFFFFF,
                (s * 31337 + 0 + 1) & 0x7FFFFFFF}
    return fam


def build_test_split():
    import torch
    import yaml
    import wp1_probe_driver as drv
    import rig_hardening_sweep as RHS
    import knee_s3_ladder as L

    assert not os.path.exists(TEST_CACHE), f'refusing to overwrite {TEST_CACHE}'
    assert TEST_SEED not in USED_SEED_BASES
    fam_test = chunk_seed_family(TEST_SEED)
    for b in USED_SEED_BASES:
        inter = fam_test & chunk_seed_family(b)
        assert not inter, f'seed family of {TEST_SEED} meets {b}: {sorted(inter)[:5]}'
    print(f'seed families: {TEST_SEED} disjoint from all {len(USED_SEED_BASES)} '
          f'used bases (chunk, z, long-stats and spot seeds)', flush=True)

    cached = torch.load(L.CACHE_PATH, map_location='cpu', weights_only=True)
    print(f'cached {L.CACHE_PATH}: keys {sorted(cached)}; '
          f'imgs_tr {tuple(cached["imgs_tr"].shape)} imgs_ev '
          f'{tuple(cached["imgs_ev"].shape)}', flush=True)

    # 1. regenerate the fitting split exactly as rig_hardening_sweep did
    t0 = time.time()
    cfg_v = RHS.make_cfg('P1', 'Enar', 'S100')
    itr, ztr, etr = RHS.gen_probe_set(cfg_v, RHS.N_TRAIN, RHS.TRAIN_SEED,
                                      RHS.SPOTS['S100'])
    mu_e, sd_e = float(etr.mean()), float(etr.std()) + 1e-8   # RHS :145
    y_tr = torch.cat([ztr, ((etr - mu_e) / sd_e)[:, None]], dim=1)
    dy = float((y_tr - cached['y_tr']).abs().max())
    dimg = float((itr.float() - cached['imgs_tr'].float()).abs().max())
    print(f'fitting split regenerated ({time.time() - t0:.0f}s): '
          f'max|y_tr - cached| = {dy:.3e}; max|imgs_tr - cached| = {dimg:.3e}; '
          f'lnE mean {mu_e:.8f} sd {sd_e:.8f}', flush=True)
    assert dy < 1e-5, f'regenerated y_tr differs from the cache by {dy}'

    # 2. the test-split renderer (driver path, run config) must reproduce the
    #    cached EVALUATION split when given its seed base
    cfg = yaml.safe_load(open(os.path.join(drv.ROOT, CFG_RUN, 'config.yml')))
    drv.PROBE_N_SPOTS = TEST_N_SPOTS
    iev, zev, eev = drv.gen_probe_set(cfg, 'gaussian', None, 2000, RHS.EVAL_SEED)
    y_ev = torch.cat([zev, ((eev - mu_e) / sd_e)[:, None]], dim=1)
    dy_ev = float((y_ev - cached['y_ev']).abs().max())
    dimg_ev = float((iev.float() - cached['imgs_ev'].float()).abs().max())
    print(f'driver path + {CFG_RUN} config, seed {RHS.EVAL_SEED}: '
          f'max|y_ev - cached| = {dy_ev:.3e}; max|imgs_ev - cached| = '
          f'{dimg_ev:.3e}', flush=True)
    assert dy_ev < 1e-5, 'driver path does not reproduce the evaluation split'

    # 3. the test split
    t0 = time.time()
    ite, zte, ete = drv.gen_probe_set(cfg, 'gaussian', None, N_TEST, TEST_SEED)
    y_te = torch.cat([zte, ((ete - mu_e) / sd_e)[:, None]], dim=1)
    print(f'test split generated ({time.time() - t0:.0f}s): imgs '
          f'{tuple(ite.shape)} {ite.dtype}; y_test mean '
          f'{[round(float(v), 4) for v in y_te.mean(0)]} sd '
          f'{[round(float(v), 4) for v in y_te.std(0)]}', flush=True)
    assert not os.path.exists(TEST_CACHE)
    torch.save({'imgs_test': ite, 'y_test': y_te, 'lnE_mean': mu_e,
                'lnE_sd': sd_e, 'seed_base': TEST_SEED,
                'n_spots': TEST_N_SPOTS, 'cfg_run': CFG_RUN,
                'builder': 'knee_s3_ladder_split.py --build-test '
                           '(wp1_probe_driver.gen_probe_set)'}, TEST_CACHE)
    print(f'saved {TEST_CACHE}', flush=True)


# ----------------------------------------------------- local probe variants

def fit_readability_split(tok_tr, data_tr, tok_ev, data_ev, tok_te, data_te,
                          *, probe_seed):
    """qprobe_p5.fit_readability (kind='big') + test scoring at every
    evaluation. Selection unchanged: evaluation split only."""
    import torch
    import torch.nn.functional as F
    import qprobe_p5 as qp

    steps, batch, lr, eval_every = qp.MAX_STEPS, qp.BATCH, qp.LR, qp.EVAL_EVERY
    device = qp.DEV
    torch.manual_seed(int(probe_seed))
    g = torch.Generator(device='cpu').manual_seed(int(probe_seed))

    tr = qp._prep(data_tr, tok_tr, device)
    ev = qp._prep(data_ev, tok_ev, device)
    te = qp._prep(data_te, tok_te, device)

    token_dim = tok_tr.shape[-1]
    model = qp.BigProbe(token_dim).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    if token_dim == 96:
        assert n_par == qp.N_PARAMS_BIG, f'big probe is {n_par}p'
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    N = tr['mask'].shape[0]
    best = dict(score=float('inf'))
    final = None
    perm = torch.randperm(N, generator=g)
    ptr = 0
    model.train()
    first_loss, last_loss = None, None
    for s in range(steps):
        if ptr + batch > N:
            perm = torch.randperm(N, generator=g)
            ptr = 0
        sel = perm[ptr:ptr + batch].to(device)
        ptr += batch
        pred = model(tr['tok'][sel], tr['qf'][sel], tr['qpos'][sel])
        m = tr['mask'][sel]
        loss = F.mse_loss(pred[m], tr['t3'][sel][m])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if first_loss is None:
            first_loss = float(loss)
        last_loss = float(loss)
        if (s + 1) % eval_every == 0 or s == steps - 1:
            p, t = qp._eval_preds(model, ev)
            med = (p - t).abs().median(dim=0).values.cpu().numpy()
            score = float(med.mean())
            r2_ev = [float(v) for v in qp._r2(p, t)]
            pt, tt = qp._eval_preds(model, te)
            r2_te = [float(v) for v in qp._r2(pt, tt)]
            if score < best['score']:
                best = dict(score=score, step=s + 1,
                            med=[float(x) for x in med],
                            r2=r2_ev, r2_test=r2_te)
            final = dict(step=s + 1, r2=r2_ev, r2_test=r2_te)
    best['final'] = final
    best['first_loss'], best['last_loss'] = first_loss, last_loss
    best['n_params'] = int(n_par)
    return best


def mlp_fit_split(h_tr, y_tr, h_ev, y_ev, h_te, y_te, probe_seed):
    """knee_s3_ladder.mlp_fit + test scoring at every evaluation. Selection
    unchanged: evaluation split only."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import knee_s3_ladder as L

    DEV = L.DEV
    torch.manual_seed(int(probe_seed))
    g = torch.Generator(device='cpu').manual_seed(int(probe_seed))
    model = nn.Sequential(
        nn.Linear(96, 768), nn.ReLU(),
        nn.Linear(768, 768), nn.ReLU(),
        nn.Linear(768, 3)).to(DEV)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=L.MLP_LR)
    Xtr, Ytr = h_tr.to(DEV), y_tr.to(DEV)
    Xev, Yev = h_ev.to(DEV), y_ev.to(DEV)
    Xte, Yte = h_te.to(DEV), y_te.to(DEV)
    N = Xtr.shape[0]
    best = dict(score=float('inf'))
    final = None
    perm = torch.randperm(N, generator=g)
    ptr = 0
    first_loss = last_loss = None

    def r2_of(p, Y):
        ss_res = ((p - Y) ** 2).sum(0)
        ss_tot = ((Y - Y.mean(0, keepdim=True)) ** 2).sum(0) + 1e-12
        return [float(v) for v in (1.0 - ss_res / ss_tot)]

    model.train()
    for s in range(L.MLP_CAP):
        if ptr + L.MLP_BATCH > N:
            perm = torch.randperm(N, generator=g)
            ptr = 0
        sel = perm[ptr:ptr + L.MLP_BATCH].to(DEV)
        ptr += L.MLP_BATCH
        loss = F.mse_loss(model(Xtr[sel]), Ytr[sel])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if first_loss is None:
            first_loss = float(loss)
        last_loss = float(loss)
        if (s + 1) % L.MLP_EVAL_EVERY == 0 or s == L.MLP_CAP - 1:
            model.eval()
            with torch.no_grad():
                p = model(Xev)
                pt = model(Xte)
            model.train()
            med = (p - Yev).abs().median(dim=0).values
            score = float(med.mean())
            r2_ev, r2_te = r2_of(p, Yev), r2_of(pt, Yte)
            if score < best['score']:
                best = dict(score=score, step=s + 1, r2=r2_ev, r2_test=r2_te)
            final = dict(step=s + 1, r2=r2_ev, r2_test=r2_te)
    best['final'] = final
    best['first_loss'], best['last_loss'] = first_loss, last_loss
    best['n_params'] = n_par
    return best


# -------------------------------------------------------------------- main

def existing_keys():
    if not os.path.exists(OUT_CSV):
        return set()
    with open(OUT_CSV) as f:
        return {(r['run'], r['phase'], r['probe'], r['probe_seed'])
                for r in csv.DictReader(f)}


def append_row(**kw):
    row = {c: kw.get(c, '') for c in COLS}
    new = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS, lineterminator='\n')
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    import torch
    import yaml
    import wp1_probe_driver as drv
    import qprobe_p5 as qp
    import knee_s3_ladder as L
    from wp1_metrics import wp1_metric_row

    assert os.path.exists(TEST_CACHE), 'run --build-test first'
    print(f'GPU: {torch.cuda.get_device_name(0)} '
          f'(CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")})',
          flush=True)
    sets = torch.load(L.CACHE_PATH, map_location='cpu', weights_only=True)
    test = torch.load(TEST_CACHE, map_location='cpu', weights_only=True)
    y_tr, y_ev = sets['y_tr'].float(), sets['y_ev'].float()
    y_te = test['y_test'].float()
    qtgt = L.query_targets(sets)
    B = y_te.shape[0]
    qtgt_te = dict(mask=torch.ones(B, 1, dtype=torch.bool),
                   x=torch.zeros(B, 1), y=torch.zeros(B, 1),
                   t3=y_te[:, None, :])
    done = existing_keys()
    anomalies = []

    for run, phase, ep in ladder_runs(L.CSV_PATH):
        todo = [('pooled_linear', '0')] + [(p, str(s)) for p in ('query85k', 'mlp1M')
                                           for s in L.PROBE_SEEDS]
        if all((run, phase, p, s) in done for p, s in todo):
            print(f'[{run} {phase} ep{ep}] all rows present, skip', flush=True)
            continue
        run_dir = os.path.join(drv.ROOT, run)
        cfg = yaml.safe_load(open(os.path.join(run_dir, 'config.yml')))
        ck = os.path.join(run_dir, f'{cfg["logging"]["write_tag"]}-ep{ep}.pth.tar')
        assert os.path.exists(ck), ck
        enc = drv.encoder_from_ckpt(cfg, ck).eval()

        h_tr = drv.encode_meanpool(enc, sets['imgs_tr'], chunk=L.CHUNK)
        h_ev = drv.encode_meanpool(enc, sets['imgs_ev'], chunk=L.CHUNK)
        h_te = drv.encode_meanpool(enc, test['imgs_test'], chunk=L.CHUNK)
        if (run, phase, 'pooled_linear', '0') not in done:
            m_ev = wp1_metric_row(h_tr, y_tr, h_ev, y_ev)
            m_te = wp1_metric_row(h_tr, y_tr, h_te, y_te)
            append_row(run=run, phase=phase, epoch=ep, probe='pooled_linear',
                       probe_seed=0,
                       r2_z1_eval=f6(m_ev['ridge_r2_z1']),
                       r2_z2_eval=f6(m_ev['ridge_r2_z2']),
                       r2_E_eval=f6(m_ev['ridge_r2_E']),
                       r2_z1_test=f6(m_te['ridge_r2_z1']),
                       r2_z2_test=f6(m_te['ridge_r2_z2']),
                       r2_E_test=f6(m_te['ridge_r2_E']))
            print(f'[{run} {phase} ep{ep} pooled_linear] eval r2=('
                  f"{m_ev['ridge_r2_z1']:.4f},{m_ev['ridge_r2_z2']:.4f},"
                  f"{m_ev['ridge_r2_E']:.4f}) test r2=("
                  f"{m_te['ridge_r2_z1']:.4f},{m_te['ridge_r2_z2']:.4f},"
                  f"{m_te['ridge_r2_E']:.4f})", flush=True)

        tok_tr = qp.extract_tokens(enc, sets['imgs_tr'], chunk=128)
        tok_ev = qp.extract_tokens(enc, sets['imgs_ev'], chunk=128)
        tok_te = qp.extract_tokens(enc, test['imgs_test'], chunk=128)
        del enc
        torch.cuda.empty_cache()

        for seed in L.PROBE_SEEDS:
            if (run, phase, 'query85k', str(seed)) in done:
                continue
            t0 = time.time()
            best = fit_readability_split(tok_tr, qtgt['tr'], tok_ev, qtgt['ev'],
                                         tok_te, qtgt_te, probe_seed=seed)
            write_learned(run, phase, ep, 'query85k', seed, best, anomalies, t0)
        del tok_tr, tok_ev, tok_te
        torch.cuda.empty_cache()

        for seed in L.PROBE_SEEDS:
            if (run, phase, 'mlp1M', str(seed)) in done:
                continue
            t0 = time.time()
            best = mlp_fit_split(h_tr, y_tr, h_ev, y_ev, h_te, y_te, probe_seed=seed)
            write_learned(run, phase, ep, 'mlp1M', seed, best, anomalies, t0)

    if anomalies:
        print('\n=== anomalies ===')
        for a in anomalies:
            print('  ' + a)
    else:
        print('\nno anomalies (train loss decreased in every fit)')
    print(f'csv: {OUT_CSV}')


def write_learned(run, phase, ep, probe, seed, best, anomalies, t0):
    if best['last_loss'] >= best['first_loss'] * 0.999:     # the ladder's own check
        msg = (f'{run} {phase} {probe} s{seed}: train loss did not decrease '
               f'({best["first_loss"]:.4f} -> {best["last_loss"]:.4f}); row NOT written')
        anomalies.append(msg)
        print('WARNING ' + msg, flush=True)
        return
    r2, r2t, fin = best['r2'], best['r2_test'], best['final']
    append_row(run=run, phase=phase, epoch=ep, probe=probe, probe_seed=seed,
               best_step=best['step'],
               r2_z1_eval=f6(r2[0]), r2_z2_eval=f6(r2[1]), r2_E_eval=f6(r2[2]),
               r2_z1_test=f6(r2t[0]), r2_z2_test=f6(r2t[1]), r2_E_test=f6(r2t[2]),
               r2_z1_final_test=f6(fin['r2_test'][0]),
               r2_z2_final_test=f6(fin['r2_test'][1]),
               r2_E_final_test=f6(fin['r2_test'][2]),
               r2_z1_final_eval=f6(fin['r2'][0]),
               r2_z2_final_eval=f6(fin['r2'][1]),
               r2_E_final_eval=f6(fin['r2'][2]))
    print(f'[{run} {phase} ep{ep} {probe} s{seed}] best_step={best["step"]} '
          f'eval=({r2[0]:.4f},{r2[1]:.4f},{r2[2]:.4f}) '
          f'test=({r2t[0]:.4f},{r2t[1]:.4f},{r2t[2]:.4f}) '
          f'final_step={fin["step"]} final_test=({fin["r2_test"][0]:.4f},'
          f'{fin["r2_test"][1]:.4f},{fin["r2_test"][2]:.4f}) '
          f'({time.time() - t0:.0f}s)', flush=True)


# ------------------------------------------------------------------ report

def report():
    """Reproduction check against results/knee_s3_ladder.csv, per-row eval vs
    test, and the POST counts (learned probe at or below pooled linear,
    z-mean R^2 = mean of z1, z2; 2-seed mean as in the record's S3 tables,
    plus per seed and for E). Reads CSVs only."""
    from statistics import mean
    old = {}
    with open(os.path.join(paths.CSV, 'knee_s3_ladder.csv')) as fh:
        for r in csv.DictReader(fh):
            old[(r['run'], r['phase'], r['probe'], r['probe_seed'])] = r
    with open(OUT_CSV) as fh:
        new = list(csv.DictReader(fh))
    print(f'{OUT_CSV}: {len(new)} rows')

    print('\n== reproduction: r2_*_eval vs knee_s3_ladder.csv r2_* ==')
    worst, n_big, n_step_same, n_learned = {}, 0, 0, 0
    for r in new:
        o = old[(r['run'], r['phase'], r['probe'], r['probe_seed'])]
        d = max(abs(float(r[f'r2_{t}_eval']) - float(o[f'r2_{t}'])) for t in ('z1', 'z2', 'E'))
        worst[r['probe']] = max(worst.get(r['probe'], 0.0), d)
        if d > 1e-3:
            n_big += 1
            print(f"  |d| {d:.2e} > 1e-3: {r['run']} {r['phase']} {r['probe']} s{r['probe_seed']} "
                  f"best_step new {r['best_step']} old {o['best_step']}")
        if r['probe'] != 'pooled_linear':
            n_learned += 1
            n_step_same += r['best_step'] == o['best_step']
    for p, w in worst.items():
        print(f'  {p:14s} max |d| over rows and targets = {w:.2e}')
    print(f'  rows with |d| > 1e-3: {n_big}; learned-probe best_step identical: '
          f'{n_step_same}/{n_learned}')

    by = {}
    for r in new:
        by.setdefault((r['run'], r['phase'], r['probe']), []).append(r)

    def zm(r, split):
        return 0.5 * (float(r[f'r2_z1_{split}']) + float(r[f'r2_z2_{split}']))

    def val(run, phase, probe, split, target='z'):
        rs = by[(run, phase, probe)]
        if target == 'z':
            return mean(zm(r, split) for r in rs)
        return mean(float(r[f'r2_E_{split}']) for r in rs)

    def old_val(run, phase, probe, target='z'):
        rs = [o for k, o in old.items() if k[:3] == (run, phase, probe)]
        if target == 'z':
            return mean(0.5 * (float(o['r2_z1']) + float(o['r2_z2'])) for o in rs)
        return mean(float(o['r2_E']) for o in rs)

    runs = sorted({(r['run'], r['phase'], int(r['epoch'])) for r in new},
                  key=lambda x: (x[0], x[1] != 'pre'))
    print('\n== per run / phase: z-mean R^2 and R^2(E); learned probes = mean of 2 seeds ==')
    print(f'{"run":16s} {"ph":4s} {"ep":>3s} {"probe":14s} '
          f'{"z eval":>7s} {"z test":>7s} {"z finT":>7s} {"E eval":>7s} {"E test":>7s} {"E finT":>7s}')
    for run, phase, ep in runs:
        for p in PROBES:
            if (run, phase, p) not in by:
                continue
            fin = 'final_test' if p != 'pooled_linear' else 'test'
            print(f'{run.split("p1s100_")[1].split("xxx")[0]:16s} {phase:4s} {ep:3d} {p:14s} '
                  f'{val(run, phase, p, "eval"):7.4f} {val(run, phase, p, "test"):7.4f} '
                  f'{val(run, phase, p, fin):7.4f} '
                  f'{val(run, phase, p, "eval", "E"):7.4f} {val(run, phase, p, "test", "E"):7.4f} '
                  f'{val(run, phase, p, fin, "E"):7.4f}')

    print('\n== counts: learned probe <= pooled linear (2-seed mean; runs with '
          'both learned seeds present) ==')
    for phase in ('post', 'pre'):
        for target in ('z', 'E'):
            for p in ('query85k', 'mlp1M'):
                rl = [(r, ph) for r, ph, _ in runs
                      if ph == phase and len(by.get((r, ph, p), [])) == 2]
                n = len(rl)
                c_old = [r for r, ph in rl if old_val(r, ph, p, target)
                         > old_val(r, ph, 'pooled_linear', target)]
                c_ev = [r for r, ph in rl if val(r, ph, p, 'eval', target)
                        > val(r, ph, 'pooled_linear', 'eval', target)]
                c_te = [r for r, ph in rl if val(r, ph, p, 'test', target)
                        > val(r, ph, 'pooled_linear', 'test', target)]
                c_fi = [r for r, ph in rl if val(r, ph, p, 'final_test', target)
                        > val(r, ph, 'pooled_linear', 'test', target)]
                lab = lambda xs: (' (above: ' + ', '.join(
                    x.split('p1s100_')[1].split('xxx')[0] for x in xs) + ')') if xs else ''
                print(f'  {phase:4s} {target} {p:9s} existing eval {n - len(c_old):2d}/{n}{lab(c_old)} | '
                      f'rerun eval {n - len(c_ev):2d}/{n}{lab(c_ev)} | '
                      f'test, eval-selected step {n - len(c_te):2d}/{n}{lab(c_te)} | '
                      f'test, final step {n - len(c_fi):2d}/{n}{lab(c_fi)}')

    print('\n== counts per seed, POST (learned-probe fit at or below pooled linear) ==')
    for target in ('z', 'E'):
        for p in ('query85k', 'mlp1M'):
            for split, ref in (('eval', 'eval'), ('test', 'test'), ('final_test', 'test')):
                c, n, above = 0, 0, []
                for run, phase, _ in runs:
                    if phase != 'post' or (run, phase, p) not in by:
                        continue
                    pr = by[(run, phase, 'pooled_linear')][0]
                    pl = zm(pr, ref) if target == 'z' else float(pr[f'r2_E_{ref}'])
                    for r in by[(run, phase, p)]:
                        n += 1
                        v = zm(r, split) if target == 'z' else float(r[f'r2_E_{split}'])
                        if v <= pl:
                            c += 1
                        else:
                            above.append(f"{run.split('p1s100_')[1].split('xxx')[0]} s{r['probe_seed']}"
                                         f' {v:.4f}>{pl:.4f}')
                print(f'  {target} {p:9s} {split:10s} {c:2d}/{n}'
                      + (f"  above: {'; '.join(above)}" if above else ''))

    print('\n== test minus eval, z-mean R^2 (all rows) ==')
    for p in PROBES:
        d = [zm(r, 'test') - zm(r, 'eval') for r in new if r['probe'] == p]
        print(f'  {p:14s} n={len(d):2d} mean {mean(d):+.4f} min {min(d):+.4f} max {max(d):+.4f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--build-test', action='store_true')
    ap.add_argument('--report', action='store_true')
    a = ap.parse_args()
    if a.report:
        report()
    elif 'CUDA_VISIBLE_DEVICES' not in os.environ:
        raise SystemExit('set CUDA_VISIBLE_DEVICES explicitly (shared GPUs)')
    elif a.build_test:
        build_test_split()
    else:
        main()
