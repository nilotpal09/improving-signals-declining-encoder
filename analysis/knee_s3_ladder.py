"""S3: content loss or re-format? Probe-capacity ladder + per-token readout
at matched PRE/POST checkpoints of the 4 knee runs (study S3; definitions
written into the record before this ran, 2026-08-19).

Checkpoints (S1 R_A onsets; PRE = last grid epoch <= onset-20, POST = final):
  rigb_lejp_p1s100_nano_1xxx*      onset  90 -> PRE ep70,  POST ep150
  rigb_lejp_p1s100_nano_2xxx*      onset 110 -> PRE ep90,  POST ep150
  rigb_lejp_p1s100_nano_3xxx*      onset 100 -> PRE ep80,  POST ep150
  rigb_lejp_p1s100_long_nano_1xxx* onset 160 -> PRE ep140, POST ep300

Probe set: rig_P1_Enar_S100.pt (4000 tr / 2000 ev, y = (z1, z2, lnE_std)).
Single particle pinned at (0, 0); calo width/height 10 -> HALF_W = HALF_H = 5.

Ladder per (run, phase):
  1. pooled_linear — wp1_metric_row on encode_meanpool features (house
     ridge alpha=1 R^2 per target + pooled CCA). Deterministic, 1x.
  2. query85k     — the 85k truth-query probe (qprobe_p5
     conventions imported: BigProbe, fit_readability v2 protocol Adam 2e-3
     batch 128 <=2500 steps held-out best every 100), ONE query per event
     at the rig's fixed particle position (0, 0); raw fp16 unpooled tokens,
     no external LayerNorm. 2 probe seeds.
  3. mlp1M        — MLP 96->768->768->3 on pooled encode_meanpool features,
     Adam 1e-3, batch 256, <=4000 steps, held-out eval every 100 steps.
     2 probe seeds. Held-out-best criterion = the house one (min mean of
     the three median |error| values, as in fit_readability); R^2 reported
     at that step. (Stated: the literal 96->768->768->3 stack is 667,395
     params, not the ~1.2M quoted in the definition; the architecture
     string is what is pinned, so it is followed literally and the true
     count reported here.)
  4. per-token linear — per token position (256 of them), ridge alpha=1
     z-scored on train from that token's 96-d RAW features (same raw-token
     convention as the query probe) to the 3 targets; token_max = the
     position maximizing z-mean R^2 (its per-target values), token_median =
     per-target median over positions; plus token_flat = flattened
     all-token ridge (24576-d, alpha=1, dual/Gram form). Deterministic, 1x.

Output: results/knee_s3_ladder.csv, append-mode, idempotent by
(run, phase, probe, probe_seed). Final compact table + per-run
re-format-vs-content-loss classification printed.

Usage (from analysis/):
  CUDA_VISIBLE_DEVICES=1 python knee_s3_ladder.py
"""

import csv
import glob
import os
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
sys.path.insert(0, _HERE)

import wp1_probe_driver as drv          # noqa: E402
import qprobe_p5 as qp                  # noqa: E402  (BigProbe/fit protocol)
from wp1_metrics import wp1_metric_row  # noqa: E402

ROOT = drv.ROOT
CACHE_PATH = os.path.join(paths.CACHE, 'rig_P1_Enar_S100.pt')
CSV_PATH = os.path.join(paths.CSV, 'knee_s3_ladder.csv')
DEV = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CHUNK = 256

RUNS = [  # (pattern, PRE epoch, POST epoch)
    ('rigb_lejp_p1s100_nano_1xxx*', 70, 150),
    ('rigb_lejp_p1s100_nano_2xxx*', 90, 150),
    ('rigb_lejp_p1s100_nano_3xxx*', 80, 150),
    ('rigb_lejp_p1s100_long_nano_1xxx*', 140, 300),
]

PROBE_SEEDS = (1, 2)
Q_CAP = qp.MAX_STEPS                    # 2500
MLP_CAP, MLP_BATCH, MLP_LR, MLP_EVAL_EVERY = 4000, 256, 1e-3, 100

COLS = ['run', 'phase', 'epoch', 'probe', 'probe_seed',
        'r2_z1', 'r2_z2', 'r2_E', 'mean_cca', 'best_step', 'cap']


# ------------------------------------------------------------- bookkeeping

def existing_keys():
    if not os.path.exists(CSV_PATH):
        return set()
    with open(CSV_PATH) as f:
        return {(r['run'], r['phase'], r['probe'], r['probe_seed'])
                for r in csv.DictReader(f)}


def append_row(**kw):
    row = {c: kw.get(c, '') for c in COLS}
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS, lineterminator='\n')
        if new:
            w.writeheader()
        w.writerow(row)


def f6(x):
    return f'{float(x):.6f}'


# ----------------------------------------------------------- probe rungs

def mlp_fit(h_tr, y_tr, h_ev, y_ev, probe_seed):
    """96->768->768->3 MLP on pooled features; house held-out-best
    protocol (min mean of the three median |error| values)."""
    torch.manual_seed(int(probe_seed))
    g = torch.Generator(device='cpu').manual_seed(int(probe_seed))
    model = nn.Sequential(
        nn.Linear(96, 768), nn.ReLU(),
        nn.Linear(768, 768), nn.ReLU(),
        nn.Linear(768, 3)).to(DEV)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=MLP_LR)
    Xtr, Ytr = h_tr.to(DEV), y_tr.to(DEV)
    Xev, Yev = h_ev.to(DEV), y_ev.to(DEV)
    N = Xtr.shape[0]
    best = dict(score=float('inf'))
    perm = torch.randperm(N, generator=g)
    ptr = 0
    first_loss = last_loss = None
    model.train()
    for s in range(MLP_CAP):
        if ptr + MLP_BATCH > N:
            perm = torch.randperm(N, generator=g)
            ptr = 0
        sel = perm[ptr:ptr + MLP_BATCH].to(DEV)
        ptr += MLP_BATCH
        loss = F.mse_loss(model(Xtr[sel]), Ytr[sel])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if first_loss is None:
            first_loss = float(loss)
        last_loss = float(loss)
        if (s + 1) % MLP_EVAL_EVERY == 0 or s == MLP_CAP - 1:
            model.eval()
            with torch.no_grad():
                p = model(Xev)
            model.train()
            med = (p - Yev).abs().median(dim=0).values
            score = float(med.mean())
            if score < best['score']:
                ss_res = ((p - Yev) ** 2).sum(0)
                ss_tot = ((Yev - Yev.mean(0, keepdim=True)) ** 2).sum(0) + 1e-12
                best = dict(score=score, step=s + 1,
                            r2=[float(v) for v in (1.0 - ss_res / ss_tot)])
    best['first_loss'], best['last_loss'] = first_loss, last_loss
    best['n_params'] = n_par
    return best


def per_token_ridge(tok_tr, y_tr, tok_ev, y_ev, alpha=1.0):
    """256 independent 96-d ridges (alpha=1, z-scored on train), fp64 CPU.
    Returns r2 array (256, 3) held-out."""
    X = tok_tr.double().cpu()            # (Ntr, 256, 96)
    Xe = tok_ev.double().cpu()
    Y = y_tr.double()                    # (Ntr, 3)
    Ye = y_ev.double()
    Xm, Xs = X.mean(0), X.std(0) + 1e-8  # (256, 96)
    Ym, Ys = Y.mean(0), Y.std(0) + 1e-8
    Xn = ((X - Xm) / Xs).permute(1, 0, 2)        # (256, Ntr, 96)
    Xen = ((Xe - Xm) / Xs).permute(1, 0, 2)      # (256, Nev, 96)
    Yn, Yne = (Y - Ym) / Ys, (Ye - Ym) / Ys
    K = Xn.shape[-1]
    G = Xn.transpose(1, 2) @ Xn + alpha * torch.eye(K, dtype=Xn.dtype)
    B = Xn.transpose(1, 2) @ Yn                   # (256, 96, 3)
    W = torch.linalg.solve(G, B)
    P = Xen @ W                                   # (256, Nev, 3)
    ss_res = ((Yne[None] - P) ** 2).sum(1)
    ss_tot = ((Yne - Yne.mean(0)) ** 2).sum(0)[None] + 1e-12
    return (1.0 - ss_res / ss_tot).numpy()        # (256, 3)


def flat_ridge(tok_tr, y_tr, tok_ev, y_ev, alpha=1.0):
    """Flattened all-token ridge (24576-d, alpha=1, z-scored on train) in
    the dual/Gram form: W = Xn' (Xn Xn' + aI)^-1 Yn. fp64 CPU."""
    X = tok_tr.double().cpu().reshape(tok_tr.shape[0], -1)
    Xe = tok_ev.double().cpu().reshape(tok_ev.shape[0], -1)
    Y, Ye = y_tr.double(), y_ev.double()
    Xm, Xs = X.mean(0), X.std(0) + 1e-8
    Ym, Ys = Y.mean(0), Y.std(0) + 1e-8
    Xn, Xen = (X - Xm) / Xs, (Xe - Xm) / Xs
    Yn, Yne = (Y - Ym) / Ys, (Ye - Ym) / Ys
    N = Xn.shape[0]
    Kg = Xn @ Xn.T + alpha * torch.eye(N, dtype=Xn.dtype)
    A = torch.linalg.solve(Kg, Yn)                # (N, 3)
    P = Xen @ (Xn.T @ A)
    ss_res = ((Yne - P) ** 2).sum(0)
    ss_tot = ((Yne - Yne.mean(0)) ** 2).sum(0) + 1e-12
    return (1.0 - ss_res / ss_tot).tolist()


def query_targets(sets):
    """One query per event at the fixed particle position (0,0):
    t3 = (z1, z2, lnE_std) per event, mask all-true, P=1."""
    out = {}
    for split in ('tr', 'ev'):
        y = sets[f'y_{split}'].float()            # (B, 3)
        B = y.shape[0]
        out[split] = dict(mask=torch.ones(B, 1, dtype=torch.bool),
                          x=torch.zeros(B, 1), y=torch.zeros(B, 1),
                          t3=y[:, None, :])
    return out


# ----------------------------------------------------------------------- main

def main():
    sets = torch.load(CACHE_PATH, map_location='cpu', weights_only=True)
    y_tr, y_ev = sets['y_tr'].float(), sets['y_ev'].float()
    qtgt = query_targets(sets)
    done = existing_keys()
    anomalies = []

    for pat, pre_ep, post_ep in RUNS:
        run_dir = glob.glob(os.path.join(ROOT, pat))[0]
        run = os.path.basename(run_dir)
        cfg = yaml.safe_load(open(os.path.join(run_dir, 'config.yml')))
        tag = cfg['logging']['write_tag']
        for phase, ep in (('pre', pre_ep), ('post', post_ep)):
            todo = [('pooled_linear', '0'), ('token_max', '0'),
                    ('token_median', '0'), ('token_flat', '0')] + \
                   [(p, str(s)) for p in ('query85k', 'mlp1M')
                    for s in PROBE_SEEDS]
            if all((run, phase, p, s) in done for p, s in todo):
                print(f'[{run} {phase} ep{ep}] all rows present, skip',
                      flush=True)
                continue
            ck = os.path.join(run_dir, f'{tag}-ep{ep}.pth.tar')
            assert os.path.exists(ck), ck
            enc = drv.encoder_from_ckpt(cfg, ck).eval()

            # ---- 1. pooled linear (+ CCA)
            h_tr = drv.encode_meanpool(enc, sets['imgs_tr'], chunk=CHUNK)
            h_ev = drv.encode_meanpool(enc, sets['imgs_ev'], chunk=CHUNK)
            if (run, phase, 'pooled_linear', '0') not in done:
                m = wp1_metric_row(h_tr, y_tr, h_ev, y_ev)
                append_row(run=run, phase=phase, epoch=ep,
                           probe='pooled_linear', probe_seed=0,
                           r2_z1=f6(m['ridge_r2_z1']),
                           r2_z2=f6(m['ridge_r2_z2']),
                           r2_E=f6(m['ridge_r2_E']),
                           mean_cca=f6(m['mean_cca']))
                print(f"[{run} {phase} ep{ep} pooled_linear] "
                      f"r2=({m['ridge_r2_z1']:.3f},{m['ridge_r2_z2']:.3f},"
                      f"{m['ridge_r2_E']:.3f}) cca={m['mean_cca']:.4f} "
                      f"(c1={m['cca_per_component'][0]:.4f},"
                      f"c2={m['cca_per_component'][1]:.4f})", flush=True)

            # ---- raw token grids (query-probe convention: fp16, no LN)

            tok_tr = qp.extract_tokens(enc, sets['imgs_tr'], chunk=128)
            tok_ev = qp.extract_tokens(enc, sets['imgs_ev'], chunk=128)
            del enc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # ---- 4. per-token + flattened ridges (deterministic)
            if not all((run, phase, p, '0') in done
                       for p in ('token_max', 'token_median', 'token_flat')):
                t0 = time.time()
                r2_tok = per_token_ridge(tok_tr, y_tr, tok_ev, y_ev)
                zmean = 0.5 * (r2_tok[:, 0] + r2_tok[:, 1])
                imax = int(np.argmax(zmean))
                med = np.median(r2_tok, axis=0)
                zmed = float(np.median(zmean))
                r2_flat = flat_ridge(tok_tr, y_tr, tok_ev, y_ev)
                if (run, phase, 'token_max', '0') not in done:
                    append_row(run=run, phase=phase, epoch=ep,
                               probe='token_max', probe_seed=0,
                               r2_z1=f6(r2_tok[imax, 0]),
                               r2_z2=f6(r2_tok[imax, 1]),
                               r2_E=f6(r2_tok[imax, 2]))
                if (run, phase, 'token_median', '0') not in done:
                    append_row(run=run, phase=phase, epoch=ep,
                               probe='token_median', probe_seed=0,
                               r2_z1=f6(med[0]), r2_z2=f6(med[1]),
                               r2_E=f6(med[2]))
                if (run, phase, 'token_flat', '0') not in done:
                    append_row(run=run, phase=phase, epoch=ep,
                               probe='token_flat', probe_seed=0,
                               r2_z1=f6(r2_flat[0]), r2_z2=f6(r2_flat[1]),
                               r2_E=f6(r2_flat[2]))
                print(f'[{run} {phase} ep{ep} tokens] '
                      f'zmean max={zmean[imax]:.4f} (pos {imax}) '
                      f'median={zmed:.4f} flat_zmean='
                      f'{0.5 * (r2_flat[0] + r2_flat[1]):.4f} '
                      f'({time.time() - t0:.0f}s)', flush=True)

            # ---- 2. query probe 85k, one query/event at (0,0), 2 seeds
            for seed in PROBE_SEEDS:
                if (run, phase, 'query85k', str(seed)) in done:
                    continue
                t0 = time.time()
                best = qp.fit_readability(tok_tr, qtgt['tr'],
                                          tok_ev, qtgt['ev'],
                                          probe_seed=seed)
                if best['last_loss'] >= best['first_loss'] * 0.999:
                    msg = (f'{run} {phase} query85k s{seed}: train loss did '
                           f'not decrease ({best["first_loss"]:.4f} -> '
                           f'{best["last_loss"]:.4f}); row NOT written')
                    anomalies.append(msg)
                    print('WARNING ' + msg, flush=True)
                    continue
                r2 = best['r2']
                append_row(run=run, phase=phase, epoch=ep, probe='query85k',
                           probe_seed=seed, r2_z1=f6(r2[0]), r2_z2=f6(r2[1]),
                           r2_E=f6(r2[2]), best_step=best['step'], cap=Q_CAP)
                print(f'[{run} {phase} ep{ep} query85k s{seed}] '
                      f'r2=({r2[0]:.3f},{r2[1]:.3f},{r2[2]:.3f}) '
                      f'best_step={best["step"]}/{Q_CAP} '
                      f'({time.time() - t0:.0f}s)', flush=True)
            del tok_tr, tok_ev
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # ---- 3. MLP probe on pooled features, 2 seeds
            for seed in PROBE_SEEDS:
                if (run, phase, 'mlp1M', str(seed)) in done:
                    continue
                t0 = time.time()
                best = mlp_fit(h_tr, y_tr, h_ev, y_ev, probe_seed=seed)
                if best['last_loss'] >= best['first_loss'] * 0.999:
                    msg = (f'{run} {phase} mlp1M s{seed}: train loss did not '
                           f'decrease ({best["first_loss"]:.4f} -> '
                           f'{best["last_loss"]:.4f}); row NOT written')
                    anomalies.append(msg)
                    print('WARNING ' + msg, flush=True)
                    continue
                r2 = best['r2']
                append_row(run=run, phase=phase, epoch=ep, probe='mlp1M',
                           probe_seed=seed, r2_z1=f6(r2[0]), r2_z2=f6(r2[1]),
                           r2_E=f6(r2[2]), best_step=best['step'],
                           cap=MLP_CAP)
                print(f'[{run} {phase} ep{ep} mlp1M s{seed}] '
                      f'r2=({r2[0]:.3f},{r2[1]:.3f},{r2[2]:.3f}) '
                      f'best_step={best["step"]}/{MLP_CAP} '
                      f'n_params={best["n_params"]} '
                      f'({time.time() - t0:.0f}s)', flush=True)

    # ------------------------------------------------------------ summary
    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))
    by = {}
    for r in rows:
        by.setdefault((r['run'], r['phase'], r['probe']), []).append(r)

    def zmean_of(run, phase, probe):
        rs = by.get((run, phase, probe))
        if not rs:
            return None, []
        vals = [0.5 * (float(r['r2_z1']) + float(r['r2_z2'])) for r in rs]
        return float(np.mean(vals)), vals

    print('\n=== S3 ladder — z-mean R^2 (2-seed probes: mean over seeds) ===')
    print(f"{'run':<38}{'phase':<7}{'pooled_lin':>11}{'query85k':>10}"
          f"{'mlp1M':>10}{'token_flat':>11}")
    summ = {}
    for pat, pre_ep, post_ep in RUNS:
        run = os.path.basename(glob.glob(os.path.join(ROOT, pat))[0])
        for phase in ('pre', 'post'):
            vals = {}
            for probe in ('pooled_linear', 'query85k', 'mlp1M', 'token_flat'):
                v, seeds = zmean_of(run, phase, probe)
                vals[probe] = v
                if len(seeds) >= 2 and max(seeds) - min(seeds) > 0.05:
                    anomalies.append(
                        f'{run} {phase} {probe}: probe seeds disagree '
                        f'{max(seeds) - min(seeds):.3f} ({seeds})')
            summ[(run, phase)] = vals
            fmt = lambda v, w: f'{v:>{w}.4f}' if v is not None else f'{"-":>{w}}'
            print(f'{run:<38}{phase:<7}'
                  f'{fmt(vals["pooled_linear"], 11)}'
                  f'{fmt(vals["query85k"], 10)}{fmt(vals["mlp1M"], 10)}'
                  f'{fmt(vals["token_flat"], 11)}')

    # best_step-vs-cap convergence report
    print('\nconvergence (best_step/cap):')
    for r in rows:
        if r['probe'] in ('query85k', 'mlp1M') and r['best_step']:
            bs, cap = int(r['best_step']), int(r['cap'])
            note = '  <-- AT CAP (possibly not converged)' if bs >= cap else ''
            print(f"  {r['run']} {r['phase']} {r['probe']} s{r['probe_seed']}: "
                  f'{bs}/{cap}{note}')
            if bs >= cap:
                anomalies.append(f"{r['run']} {r['phase']} {r['probe']} "
                                 f"s{r['probe_seed']}: best_step at cap")

    print('\n=== per-run classification (numbers only) ===')
    for pat, pre_ep, post_ep in RUNS:
        run = os.path.basename(glob.glob(os.path.join(ROOT, pat))[0])
        pre, post = summ[(run, 'pre')], summ[(run, 'post')]
        parts = []
        for probe in ('pooled_linear', 'query85k', 'mlp1M', 'token_flat'):
            if pre[probe] is None or post[probe] is None:
                parts.append(f'{probe}: n/a')
            else:
                parts.append(f'{probe}: {pre[probe]:.3f}->{post[probe]:.3f} '
                             f'(gap {pre[probe] - post[probe]:+.3f})')
        print(f'{run}\n  ' + '\n  '.join(parts))
        gaps = {p: (pre[p] - post[p])
                for p in ('pooled_linear', 'query85k', 'mlp1M', 'token_flat')
                if pre[p] is not None and post[p] is not None}
        if 'pooled_linear' in gaps:
            hi = [p for p in ('query85k', 'mlp1M', 'token_flat')
                  if p in gaps]
            if hi and all(gaps[p] <= 0.5 * gaps['pooled_linear']
                          for p in hi):
                verdict = ('RE-FORMAT signature (higher-capacity gaps <= '
                           'half the pooled-linear gap)')
            elif hi and all(gaps[p] >= 0.5 * gaps['pooled_linear']
                            for p in hi):
                verdict = ('CONTENT-LOSS signature (drop persists at all '
                           'capacities)')
            else:
                verdict = 'MIXED (some capacities recover, some do not)'
            print(f'  -> {verdict}')

    if anomalies:
        print('\n=== anomalies ===')
        for a in anomalies:
            print('  ' + a)
    else:
        print('\nno anomalies (all fits converged below cap; '
              'seed spread <= 0.05)')
    print(f'\ncsv: {CSV_PATH}')


if __name__ == '__main__':
    main()
