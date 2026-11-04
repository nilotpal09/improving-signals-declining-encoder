"""F7 disambiguation — truth-anchored per-particle query probe on the P5 rig.

Question: is LeJEPA's pooled-linear deficit on the multiparticle rig
(P5_Enar_S100: escaped I-JEPA 0.94 > MAE 0.87 > LeJEPA 0.79 mean CCA,
floor 0.75) a READOUT artifact (z-information spatially localized per
particle, destroyed by mean-pooling) or genuinely absent z-information?

Instrument: the companion study's truth-query probe (probe_model.QueryProbe,
capacity 'big', 85,189 params) — one query per TRUE particle from its known
(x, y) via Fourier features (N_FOURIER=5), cross-attention onto the UNPOOLED
(B, 256, 96) token grid with the learnable-strength locality prior
(sigma=0.25), no query-to-query attention. Fit protocol = the companion
study's v2_common.fit_readability: Adam lr 2e-3, batch 128, 2500 steps, held-out
eval every 100 steps, report the held-out-best checkpoint (best = min mean
of the three median |error| values).

Import policy: probe_model is imported read-only from analysis/probe_model.py
(it is pure torch, no side effects). The thin v2_common pieces (BigProbe 3-output
head, probe_query_feats, _prep/_eval_preds/fit_readability) are transcribed
below VERBATIM in logic — importing v2_common would drag in _sp_common,
which prepends an older I-JEPA code tree to sys.path and collides with
the `src` package (train/src/) used by the renderer.

Token convention (matches the companion study's _sp_common.extract_tokens): raw encoder
output tokens, stored fp16 on device, NO external LayerNorm — the probe's
own first op is tok_norm = LayerNorm(96). The POOLED-LINEAR comparison
readout uses the house pooling instead (encode_meanpool: LN -> mean ->
ridge alpha=1.0 z-scored, wp1_metrics.ridge_r2), so the per-particle vs
pooled comparison lives in one script on identical events.

Data: P5_Enar_S100 rig (N_range [5,5], x/y_range [-4,4],
E_range [20000,40000], N_spots 100), every particle sharing the event's
z=(z1,z2); gaussian z; 4000 train / 2000 eval; seed bases 20260940/20260941
(disjoint from all previous probe-set bases). Per-particle targets:
(z1, z2) = the event z repeated per particle, plus standardized
per-particle lnE (train-split stats). Cached to
CACHE_DIR/qprobe_P5S100_gauss.pt with per-particle truth (p_x, p_y, p_E).

Usage (from analysis/):
  CUDA_VISIBLE_DEVICES=1 python qprobe_p5.py
"""

import argparse
import csv
import math
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

import wp1_probe_driver as drv  # noqa: E402  (puts train/ on sys.path)
from src.lejepa_data import sample_z  # noqa: E402
from wp1_metrics import ridge_r2  # noqa: E402

import probe_model as PM  # noqa: E402  (pure torch)

# companion-study determinism conventions (math SDP backend: deterministic backward)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

CACHE_DIR = paths.CACHE
CACHE_PATH = os.path.join(CACHE_DIR, 'qprobe_P5S100_gauss.pt')
CSV_PATH = os.path.join(paths.CSV, 'qprobe_p5.csv')
ROOT = drv.ROOT

N_TRAIN, N_EVAL = 4000, 2000
TRAIN_SEED, EVAL_SEED = 20260940, 20260941
CHUNK = 256
N_SPOTS = 100
HALF_W, HALF_H = 5.0, 5.0            # calo width/height 10 cm (companion-study default)
PROBE_SEEDS = (1, 2)
DEV = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# fit protocol pinned from the companion study's v2_common
BATCH, LR, MAX_STEPS, EVAL_EVERY = 128, 2e-3, 2500, 100
N_PARAMS_BIG = 85189

ENCODERS = {
    'lejepa':      'rigb_lejp_p5s100_nano_1xxxtjsuqj',
    'mae':         'rigb_mae_p5s100_nano_1xxxwonepd5',
    'ijepa':       'rigb_ijp_p5s100_nano_1xxx0c638u6',
    'random_init': None,             # torch.manual_seed(0) fresh vit_nano
}

COLS = ['encoder', 'probe_seed', 'readout', 'r2_z1', 'r2_z2', 'r2_E',
        'best_step', 'sec']


def make_cfg():
    """P5_Enar_S100 variant config (identical to rig_hardening_sweep)."""
    return {
        'data': {
            'calo_cfg': {
                'N_cells_x': 64, 'N_cells_y': 64, 'N_cells_z': 4,
                'N_spots_per_layer': N_SPOTS,
                'Z': 82, 'cell_e_threshold': 0.0,
                'depth': 20.0, 'height': 10.0, 'width': 10.0,
            },
            'generator_cfg': {
                'kind': 'gun', 'pad_value': float('nan'),
                'E_range': [20000.0, 40000.0], 'N_range': [5, 5],
                'x_range': [-4.0, 4.0], 'y_range': [-4.0, 4.0],
            },
            'norm': {'mode': 'log1p_scale', 'scale': 5500.0},
        },
        'meta': {'model_name': 'vit_nano'},
        'mask': {'patch_size': 4},
    }


# ----------------------------------------------------------------- probe set

def gen_probe_set(cfg, n_events, seed_base):
    """Mirrors rig_hardening_sweep.gen_probe_set (same seed derivations,
    gaussian dial) but keeps the PER-PARTICLE truth (p_x, p_y, p_E)."""
    coll = drv._renderer(cfg)
    imgs, zs, xs, ys, Es = [], [], [], [], []
    n_chunks = (n_events + CHUNK - 1) // CHUNK
    for c in range(n_chunks):
        n = min(CHUNK, n_events - c * CHUNK)
        seed = (seed_base * 100003 + c * 7919 + 1) & 0x7FFFFFFF
        p_E, p_x, p_y = coll._generate_truth(n, seed)
        z = sample_z(n, dist='gaussian', mu=None,
                     generator=torch.Generator().manual_seed(
                         (seed * 2654435761 + 97) & 0x7FFFFFFF))
        lp = coll._lp_from_z(p_E, (seed * 7919 + 31337) & 0x7FFFFFFF, z)
        img = coll.norm.forward(coll._simulate_once(
            p_E, p_x, p_y, coll._sim_seed(seed, 0),
            n_spots_override=N_SPOTS, long_params_override=lp))
        imgs.append(img.detach().to('cpu', torch.float16))
        zs.append(z)
        xs.append(p_x.detach().cpu().float())
        ys.append(p_y.detach().cpu().float())
        Es.append(p_E.detach().cpu().float())
    return (torch.cat(imgs), torch.cat(zs), torch.cat(xs), torch.cat(ys),
            torch.cat(Es))


def load_probe_sets():
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(CACHE_PATH):
        return torch.load(CACHE_PATH, map_location='cpu', weights_only=True)
    cfg = make_cfg()
    t0 = time.time()
    itr, ztr, xtr, ytr, Etr = gen_probe_set(cfg, N_TRAIN, TRAIN_SEED)
    iev, zev, xev, yev, Eev = gen_probe_set(cfg, N_EVAL, EVAL_SEED)
    d = {'imgs_tr': itr, 'z_tr': ztr, 'p_x_tr': xtr, 'p_y_tr': ytr,
         'p_E_tr': Etr,
         'imgs_ev': iev, 'z_ev': zev, 'p_x_ev': xev, 'p_y_ev': yev,
         'p_E_ev': Eev,
         'sec_gen': time.time() - t0}
    torch.save(d, CACHE_PATH)
    return d


def build_targets(sets):
    """Per-particle t3 = (z1, z2, lnE_std) with train-split lnE stats, plus
    pooled event-level y = (z1, z2, lnE_total_std) (rig-sweep convention)."""
    out = {}
    mask_tr = torch.isfinite(sets['p_E_tr'])
    mask_ev = torch.isfinite(sets['p_E_ev'])
    lnE_tr = torch.where(mask_tr, sets['p_E_tr'], torch.ones_like(
        sets['p_E_tr'])).log()
    lnE_ev = torch.where(mask_ev, sets['p_E_ev'], torch.ones_like(
        sets['p_E_ev'])).log()
    mu = float(lnE_tr[mask_tr].mean())
    sd = float(lnE_tr[mask_tr].std()) + 1e-8
    P = sets['p_E_tr'].shape[1]
    for split, mask, lnE in (('tr', mask_tr, lnE_tr), ('ev', mask_ev, lnE_ev)):
        z = sets[f'z_{split}']                                  # (B, 2)
        zrep = z[:, None, :].expand(-1, P, -1)                  # (B, P, 2)
        t3 = torch.cat([zrep, ((lnE - mu) / sd)[..., None]], dim=-1)
        lnEtot = torch.log(torch.where(
            mask, sets[f'p_E_{split}'],
            torch.zeros_like(lnE)).double().sum(1)).float()
        out[split] = dict(mask=mask, x=sets[f'p_x_{split}'].nan_to_num(0.0),
                          y=sets[f'p_y_{split}'].nan_to_num(0.0),
                          t3=t3, z=z, lnEtot=lnEtot)
    mu_t = float(out['tr']['lnEtot'].mean())
    sd_t = float(out['tr']['lnEtot'].std()) + 1e-8
    for split in ('tr', 'ev'):
        d = out[split]
        d['y_pooled'] = torch.cat(
            [d['z'], ((d['lnEtot'] - mu_t) / sd_t)[:, None]], dim=1)
    return out


# ------------------------------------------------- probe (from v2_common)

class BigProbe(PM.QueryProbe):
    """v2_common.BigProbe transcribed: the house truth-query probe
    (probe_model.QueryProbe, capacity 'big') with a 3-output head:
    85,189 parameters, 22-dim position-only queries, Fourier token
    positions, learnable locality prior sigma=0.25."""

    def __init__(self, token_dim=96, n_feat=22, n_out=3):
        super().__init__(token_dim, n_feat, capacity='big')
        d = PM.CAPACITY['big']['d_model']
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_out))


def probe_query_feats(x, y, half_w=HALF_W, half_h=HALF_H):
    """v2_common.probe_query_feats, 'big' branch (position only)."""
    qf = PM.query_features(x, y, x, half_w, half_h, 0.0, 1.0,
                           use_energy=False)
    qp = torch.stack([(x / half_w).clamp(-2, 2),
                      (y / half_h).clamp(-2, 2)], dim=-1)
    return qf, qp


def _prep(data, tok, device):
    m = data['mask'].to(device)
    qf, qp = probe_query_feats(data['x'].to(device), data['y'].to(device))
    return dict(mask=m, qf=qf, qpos=qp, t3=data['t3'].to(device), tok=tok)


@torch.no_grad()
def _eval_preds(model, ev, chunk=256):
    model.eval()
    P, T = [], []
    for i in range(0, ev['mask'].shape[0], chunk):
        sl = slice(i, i + chunk)
        p = model(ev['tok'][sl], ev['qf'][sl], ev['qpos'][sl])
        m = ev['mask'][sl]
        P.append(p[m].float())
        T.append(ev['t3'][sl][m].float())
    model.train()
    return torch.cat(P), torch.cat(T)


def _r2(pred, truth):
    ss_res = ((pred - truth) ** 2).sum(0)
    ss_tot = ((truth - truth.mean(0, keepdim=True)) ** 2).sum(0) + 1e-12
    return (1.0 - ss_res / ss_tot).cpu().numpy()


def fit_readability(tok_tr, data_tr, tok_ev, data_ev, *, probe_seed,
                    steps=MAX_STEPS, batch=BATCH, lr=LR,
                    eval_every=EVAL_EVERY, device=DEV, verbose=False):
    """v2_common.fit_readability transcribed (kind='big' path only): held-out
    eval every `eval_every` steps; the reported checkpoint is the held-out
    minimum of the mean of the three median |error| values."""
    torch.manual_seed(int(probe_seed))
    g = torch.Generator(device='cpu').manual_seed(int(probe_seed))

    tr = _prep(data_tr, tok_tr, device)
    ev = _prep(data_ev, tok_ev, device)

    token_dim = tok_tr.shape[-1]
    model = BigProbe(token_dim).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    if token_dim == 96:
        assert n_par == N_PARAMS_BIG, f'big probe is {n_par}p'
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    N = tr['mask'].shape[0]
    curve = []
    best = dict(score=float('inf'))
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
            p, t = _eval_preds(model, ev)
            med = (p - t).abs().median(dim=0).values.cpu().numpy()
            curve.append((s + 1, *[float(x) for x in med]))
            score = float(med.mean())
            if score < best['score']:
                best = dict(score=score, step=s + 1,
                            med=[float(x) for x in med],
                            r2=[float(v) for v in _r2(p, t)])
            if verbose:
                print(f'      step {s+1:5d} loss {float(loss):.4f}  '
                      f'z1 {med[0]:.4f} z2 {med[1]:.4f} E {med[2]:.4f}',
                      flush=True)
    best['curve'] = np.array(curve, dtype=np.float32)
    best['first_loss'], best['last_loss'] = first_loss, last_loss
    best['n_params'] = int(n_par)
    return best


# ---------------------------------------------------------------- encoders

@torch.no_grad()
def extract_tokens(enc, imgs_cpu, chunk=128, out_dtype=torch.float16):
    """_sp_common.extract_tokens convention: raw unpooled (B, 256, 96) token
    grid, fp16, kept on device, NO external LayerNorm (the probe's tok_norm
    LayerNorms internally)."""
    enc.eval()
    out = []
    for i in range(0, imgs_cpu.size(0), chunk):
        x = imgs_cpu[i:i + chunk].to(DEV, non_blocking=True).float()
        out.append(enc(x).to(out_dtype))
        del x
    return torch.cat(out, dim=0)


def get_encoder(name):
    if name == 'random_init':
        torch.manual_seed(0)
        return drv.build_encoder_from_cfg(make_cfg()).eval()
    run_dir = os.path.join(ROOT, ENCODERS[name])
    cfg = yaml.safe_load(open(os.path.join(run_dir, 'config.yml')))
    tag = cfg['logging']['write_tag']
    ck = os.path.join(run_dir, f'{tag}-ep150.pth.tar')
    assert os.path.exists(ck), ck
    return drv.encoder_from_ckpt(cfg, ck).eval()


# -------------------------------------------------------------- bookkeeping

def existing_keys():
    if not os.path.exists(CSV_PATH):
        return set()
    with open(CSV_PATH) as f:
        return {(r['encoder'], r['probe_seed'], r['readout'])
                for r in csv.DictReader(f)}


def append_row(row):
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS, lineterminator='\n')
        if new:
            w.writeheader()
        w.writerow(row)


# ----------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoders', default='all')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    names = list(ENCODERS) if args.encoders == 'all' \
        else args.encoders.split(',')
    for n in names:
        assert n in ENCODERS, n

    sets = load_probe_sets()
    print(f"probe set: {sets['imgs_tr'].shape[0]} train / "
          f"{sets['imgs_ev'].shape[0]} eval  (gen {sets.get('sec_gen', 0):.0f}s)",
          flush=True)
    tgt = build_targets(sets)
    done = existing_keys()

    results = {}
    for name in names:
        enc = get_encoder(name)
        tok_tr = extract_tokens(enc, sets['imgs_tr'])
        tok_ev = extract_tokens(enc, sets['imgs_ev'])

        # pooled-linear readout (house pooling: LN -> mean -> ridge a=1.0)
        if (name, '0', 'pooled_linear') not in done:
            t0 = time.time()
            h_tr = drv.encode_meanpool(enc, sets['imgs_tr'], chunk=CHUNK)
            h_ev = drv.encode_meanpool(enc, sets['imgs_ev'], chunk=CHUNK)
            r2 = ridge_r2(h_tr, tgt['tr']['y_pooled'],
                          h_ev, tgt['ev']['y_pooled'])
            append_row({'encoder': name, 'probe_seed': 0,
                        'readout': 'pooled_linear',
                        'r2_z1': f'{r2[0]:.6f}', 'r2_z2': f'{r2[1]:.6f}',
                        'r2_E': f'{r2[2]:.6f}', 'best_step': '',
                        'sec': f'{time.time() - t0:.1f}'})
            print(f'[{name} pooled_linear] r2=({r2[0]:.3f},{r2[1]:.3f},'
                  f'{r2[2]:.3f})', flush=True)
        del enc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # query-probe readout, two probe seeds
        for seed in PROBE_SEEDS:
            if (name, str(seed), 'query') in done:
                continue
            t0 = time.time()
            best = fit_readability(tok_tr, tgt['tr'], tok_ev, tgt['ev'],
                                   probe_seed=seed, verbose=args.verbose)
            if best['last_loss'] >= best['first_loss'] * 0.999:
                print(f'WARNING [{name} s{seed}]: train loss did not '
                      f'decrease ({best["first_loss"]:.4f} -> '
                      f'{best["last_loss"]:.4f}) — NOT writing a row; debug.',
                      flush=True)
                continue
            r2 = best['r2']
            append_row({'encoder': name, 'probe_seed': seed,
                        'readout': 'query',
                        'r2_z1': f'{r2[0]:.6f}', 'r2_z2': f'{r2[1]:.6f}',
                        'r2_E': f'{r2[2]:.6f}', 'best_step': best['step'],
                        'sec': f'{time.time() - t0:.1f}'})
            print(f'[{name} query s{seed}] r2=({r2[0]:.3f},{r2[1]:.3f},'
                  f'{r2[2]:.3f}) med=({best["med"][0]:.3f},'
                  f'{best["med"][1]:.3f},{best["med"][2]:.3f}) '
                  f'best_step={best["step"]} sec={time.time() - t0:.0f}',
                  flush=True)
        del tok_tr, tok_ev
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------ final table
    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))
    by = {}
    for r in rows:
        by[(r['encoder'], r['probe_seed'], r['readout'])] = r
    print('\nquery vs pooled readout — r2_z = mean(r2_z1, r2_z2)')
    hdr = (f"{'encoder':<12}{'query s1':>10}{'query s2':>10}"
           f"{'pooled':>10}{'q_r2_E s1':>11}{'pooled_E':>10}")
    print(hdr)
    for name in ENCODERS:
        def rz(key):
            r = by.get(key)
            if r is None:
                return None
            return 0.5 * (float(r['r2_z1']) + float(r['r2_z2']))
        q1, q2 = rz((name, '1', 'query')), rz((name, '2', 'query'))
        pl = rz((name, '0', 'pooled_linear'))
        qe = by.get((name, '1', 'query'))
        pe = by.get((name, '0', 'pooled_linear'))
        fmt = lambda v: f'{v:>10.4f}' if v is not None else f'{"-":>10}'
        fe = lambda r, w: (f'{float(r["r2_E"]):>{w}.4f}' if r is not None
                           else f'{"-":>{w}}')
        print(f'{name:<12}{fmt(q1)}{fmt(q2)}{fmt(pl)}{fe(qe, 11)}{fe(pe, 10)}')

    # sanity gates
    p_rand = by.get(('random_init', '0', 'pooled_linear'))
    if p_rand is not None:
        v = 0.5 * (float(p_rand['r2_z1']) + float(p_rand['r2_z2']))
        band = 'OK' if 0.55 <= v <= 0.75 else 'OUT OF BAND (expected ~0.6-0.7)'
        print(f'\ngate (a) random_init pooled r2_z = {v:.4f}  {band}')
        q_rand = [x for x in (by.get(('random_init', '1', 'query')),
                              by.get(('random_init', '2', 'query')))
                  if x is not None]
        if q_rand:
            qv = np.mean([0.5 * (float(r['r2_z1']) + float(r['r2_z2']))
                          for r in q_rand])
            print(f'gate (b) random_init query r2_z = {qv:.4f} '
                  f'{"OK (query > pooled)" if qv > v else "FAILED (query <= pooled)"}')


if __name__ == '__main__':
    main()
