"""The LiDAR audit at smaller regulariser delta, both groupings, all runs
(2026-09-24).

Why. lidar() (analysis/lidar_metric.py, imported by
knee_lidar.py) adds an ABSOLUTE delta * I, delta = 1e-4, to the within-group
covariance S_w. The LiDAR paper (arXiv 2312.04000, eq. 2) calls delta "a small
positive constant" and gives no value. On our pooled features delta is
comparable to the within-group variance along the leading whitened
direction. The standing audit (CUSUM timely first alarms 18 of 27 with shape
redrawn in a group, 1 of 27 with shape shared) was computed at delta = 1e-4
only. This script repeats the whole audit at other delta.

Material. All 58 runs of results/knee_lidar.csv, every checkpoint with an N = 100
row (2,114), both N = 100 replica sets (knee_lidar_N100.pt = shape redrawn,
knee_lidar_sharedshape_N100.pt = shape shared). Features: knee_lidar.lidar_of's
path verbatim (encoder tokens -> LayerNorm -> token mean, chunks of 64,
float32), encoder from wp1_probe_driver.encoder_from_ckpt (which also reads
the MAE checkpoints: knee_lidar_sharedshape.py computed its 58-run series,
MAE included, through this same path and reproduced knee_lidar.csv rows of
MAE s1 and s7 with it).

Stages
  --features  (GPU) replica-set features. The 19 event-free LeJEPA runs are
              REUSED from replica_feats/ in the work folder (written by
              knee_lidar_killtest.py --features; checked by --gate). The other
              39 runs go to RFEAT_NEW under probe_caches/ in the same
              format (a new directory, so that scripts that glob
              replica_feats/*.pt keep seeing exactly the 19 files they were
              written for). float32, refuses to overwrite, .partial then
              os.replace, resumable by run.
  --gate      (GPU) G0 file checks and re-extraction of sampled checkpoints;
              G1 LiDAR at delta = 1e-4 (float32 on the GPU, the standing
              pipeline) against results/knee_lidar.csv and
              results/knee_lidar_sharedshape.csv at every checkpoint; G2 the
              standing audit at delta = 1e-4 against the recorded counts and
              results/knee_lidar_calib_sensitivity_v2.csv cell for cell; G3 lambda1
              against results/knee_lidar_spectrum_decomp.csv and the verifier's
              knee_stats.pkl at delta = 1e-4, 1e-6, 1e-8, 0. Prints only.
  --run       (GPU for the float32 rows; the sweep itself is float64 on the
              CPU) the sweep and the audit. Writes results/knee_lidar_delta_sweep.csv
              and results/knee_lidar_delta_sweep_audit.csv (refuse to overwrite;
              .partial then os.replace).
  --report    prints every table from the two CSVs.

Delta labels (knee_lidar_delta_sweep.csv, column delta_label)
  f32gpu_1e-4  lidar() exactly as the standing pipeline: float32 on the GPU,
               delta = 1e-4. Reproduces the standing CSVs (G1).
  1e-4 ... 1e-8, 0
               lidar() (imported unchanged) on the same features cast to
               float64, CPU. float64 because float32 departs from float64 by
               percent-level amounts at delta <= 1e-6 (see the precision
               check). delta = 0 uses lidar()'s own eigenvalue floor
               (ew.clamp(min=1e-12)); it is reported but flagged: S_w is
               singular along the all-ones direction (LayerNorm) and, at late
               checkpoints, along a few more directions.
  0_noones     diagnostic for the delta = 0 flag: delta = 0 after projecting
               the features onto the complement of the all-ones direction.
  rel_a, rel_b scale-relative regulariser delta = eps * tr(S_w) / p, with eps
               set per grouping so that delta equals 1e-4 (rel_a) or 1e-6
               (rel_b) at the median over the 677 checkpoints of the 19
               event-free LeJEPA runs.
lambda1 and q come from knee_lidar_spectrum_decomp's mats / whitener / decomp
(its eigh path), with that module's constant DELTA set to the label's delta
for the call and restored afterwards (the functions themselves unchanged).
delta_share = delta / (x' S_w x + delta), x the unit leading generalised
eigenvector: the regulariser's share of the leading whitened direction's
denominator (the verifier's quantity). n_sw_below = number of raw S_w
eigenvalues below delta (below lidar()'s floor 1e-12 when delta = 0).

Audit (per label and grouping; machinery imported unchanged):
  knee_lidar_replica_seed.counts_from_S  standing thresholds (1.1 x max over
      each campaign's four mu = 0.97 controls, recalibrated on the series at
      the same delta), the four tests of knee_s2_monitors.TESTS, knee alarms,
      timely (|first alarm - onset| <= 10), held-out LeJEPA /11, MAE /10,
      ViT-Tiny flat /1, any of four.
  knee_labelfree_calib_sensitivity.calib_rows  calibrations A, LOO, B-flat,
      H-300ep-only with the onset-window ratios.
  knee_labelfree_audit.s2a  S2a local-deviation rank at onset and placebo
      (knee_s2_monitors.median, the upper median, as the audit prints).
  knee_lidar_spectrum_decomp.build_plan  the PRE / onset / POST checkpoints
      of the 27 knee runs and PRE / POST of the 19 event-free runs.
Second ways (table 'check' of the audit CSV): LiDAR and lambda1 recomputed
with scipy.linalg.eigh(S_b, S_w + delta I) in numpy (every checkpoint, every
label with delta > 0); CUSUM counts recomputed with an independently written
CUSUM; eigvalsh path (lidar()) against the eigh path (decomp).

Usage (from analysis/):
  CUDA_VISIBLE_DEVICES=1 python -B knee_lidar_delta_sweep.py --features
  CUDA_VISIBLE_DEVICES=1 python -B knee_lidar_delta_sweep.py --gate
  CUDA_VISIBLE_DEVICES=1 python -B knee_lidar_delta_sweep.py --run
  python -B knee_lidar_delta_sweep.py --report
"""
import sys

sys.dont_write_bytecode = True      # leave analysis/__pycache__ untouched

import argparse  # noqa: E402
import csv  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import pickle  # noqa: E402
import time  # noqa: E402
from collections import OrderedDict, defaultdict  # noqa: E402
from statistics import median as smedian  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
sys.path.insert(0, _HERE)

CACHE = paths.CACHE
RFEAT_REUSED = os.path.join(paths.WORK, 'replica_feats')        # knee_lidar_killtest.py --features (19 runs, plus the runs added 2026-10-13)
RFEAT_NEW = os.path.join(CACHE, 'replica_feats_all_denom_2026-10-13')       # this script, the other 39 runs (plus the runs added 2026-10-13)
SET_FILE = OrderedDict((('redrawn', 'knee_lidar_N100.pt'),
                        ('shared', 'knee_lidar_sharedshape_N100.pt')))
SET_SHA = {'redrawn': 'ab09b74c2745232b', 'shared': '36058dcf14b36323'}
GROUPINGS = tuple(SET_FILE)
PROTO = {'redrawn': 'existing', 'shared': 'shared'}          # labels in the v2 calibration CSV
LID_CSV = {'redrawn': os.path.join(paths.CSV, 'knee_lidar.csv'),
           'shared': os.path.join(paths.CSV, 'knee_lidar_sharedshape.csv')}
DECOMP_CSV = os.path.join(paths.CSV, 'knee_lidar_spectrum_decomp.csv')
CALIB_CSV = os.path.join(paths.CSV, 'knee_lidar_calib_sensitivity_v2.csv')
_NOTES = os.environ.get('MONITOR_NOTES', os.path.join(paths.REPO, 'notes'))   # independent re-implementation (not released)
VERIFIER_STATS = os.path.join(_NOTES, 'verifier_2026-09-24', 'knee_stats.pkl')
VERIFIER_FEATS = os.path.join(_NOTES, 'verifier_2026-09-24', 'knee_feats.pkl')
OUT = os.path.join(paths.CSV, 'knee_lidar_delta_sweep.csv')
OUT_AUDIT = os.path.join(paths.CSV, 'knee_lidar_delta_sweep_audit.csv')
N_SPOTS, M, K = 100, 64, 8
FLOOR = 1e-12                                                  # lidar()'s ew.clamp(min=1e-12)
ABS = OrderedDict((('1e-4', 1e-4), ('1e-5', 1e-5), ('1e-6', 1e-6), ('1e-7', 1e-7),
                   ('1e-8', 1e-8), ('0', 0.0)))
REL = OrderedDict((('rel_a', 1e-4), ('rel_b', 1e-6)))
F32 = 'f32gpu_1e-4'
NOONES = '0_noones'
LABELS = [F32] + list(ABS) + [NOONES] + list(REL)
SWEEP_COLS = ['run', 'epoch', 'grouping', 'delta_label', 'delta_value', 'lidar', 'lambda1', 'q',
              'delta_share', 'n_sw_below', 'trace_sw_over_p']
AUDIT_COLS = ['delta_label', 'grouping', 'table', 'population', 'phase', 'campaign', 'test',
              'calibration', 'quantity', 'value']

# the record at delta = 1e-4 (N = 100): per test (knee 21x alarms, knee 4x alarms,
# held-out LeJEPA /11, MAE /10, ViT-Tiny flat /1); CUSUM timely 21x, 4x; any of four.
# S2a medians are knee_s2_monitors.median (the upper median), as printed in
# the record (lidar rows); an earlier report printed 0.158 for redrawn 21x, the
# ordinary median of the two middle printed ranks 0.148, 0.167.
RECORD = {
    'redrawn': {'CUSUM': (10, 10, 1, 1, 0), 'PageHinkley': (9, 11, 2, 2, 1),
                'EWMA': (10, 10, 2, 0, 0), 'slope11': (11, 11, 2, 1, 1),
                'timely': (10, 8), 'any': (3, 3, 1), 'thr': (26.0734, 48.6835),
                's2a': {'21x': (0.167, 0.481), '4x': (0.148, 0.481)}},
    'shared': {'CUSUM': (0, 3, 2, 0, 0), 'PageHinkley': (2, 1, 0, 0, 0),
               'EWMA': (1, 1, 2, 0, 0), 'slope11': (0, 2, 0, 0, 0),
               'timely': (0, 1), 'any': (2, 0, 0), 'thr': (45.9742, 19.5307),
               's2a': {'21x': (0.593, 0.481), '4x': (0.222, 0.407)}},
}


def short(run):
    return run.split('xxx')[0]


def need_gpu():
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        raise SystemExit('set CUDA_VISIBLE_DEVICES explicitly (shared GPUs)')


def plan():
    """short run -> (full run name, sorted N = 100 epochs), results/knee_lidar.csv order."""
    grid, full = OrderedDict(), {}
    with open(LID_CSV['redrawn']) as fh:
        for r in csv.DictReader(fh):
            if r['n_spots'] == str(N_SPOTS):
                s = short(r['run'])
                full[s] = r['run']
                grid.setdefault(s, set()).add(int(r['epoch']))
    out = OrderedDict((s, (full[s], sorted(e))) for s, e in grid.items())
    assert len(out) == 61 and sum(len(v[1]) for v in out.values()) == 2228  # runs added 2026-10-13: + 3 x 38
    return out


def runs19():
    import knee_lidar_audit as AUD
    import knee_lidar_audit_sharedshape as A2
    return AUD.CTRL_21 + AUD.CTRL_4 + A2.HELD_21 + A2.HELD_4


def feat_path(s, full):
    return os.path.join(RFEAT_REUSED if s in runs19() else RFEAT_NEW, f'{full}.pt')


def load_feats(s, full, eps):
    import torch
    R = torch.load(feat_path(s, full), map_location='cpu', weights_only=True)
    assert R['run'] == full and list(R['epochs']) == list(eps), s
    return R


def csv_values():
    """(grouping, short, epoch) -> (float, 6-decimal string) from the standing CSVs."""
    out = {}
    for g, p in LID_CSV.items():
        with open(p) as fh:
            for r in csv.DictReader(fh):
                if r['n_spots'] == str(N_SPOTS):
                    out[(g, short(r['run']), int(r['epoch']))] = (float(r['lidar']), r['lidar'])
    return out


def f6(x):
    return f'{x:.6f}'


# ================================================================ features

def feats_fn():
    import torch
    import torch.nn.functional as Fn
    import knee_lidar as KL

    @torch.no_grad()
    def feats_of(enc, x):
        """knee_lidar.lidar_of's feature path verbatim, returning (M*K, D)."""
        out = []
        enc.eval()
        for i in range(0, x.size(0), 64):
            h = enc(x[i:i + 64].to(KL.DEV, torch.float32)).float()
            h = Fn.layer_norm(h, (h.size(-1),))
            out.append(h.mean(dim=1))
        return torch.cat(out, dim=0)
    return feats_of


def load_sets():
    import torch
    import knee_lidar_sharedshape as KLS
    imgs = {g: torch.load(os.path.join(CACHE, f), map_location='cpu', weights_only=True)
            for g, f in SET_FILE.items()}
    for g in imgs:
        d = KLS.set_digest(imgs[g])
        assert d == SET_SHA[g], (g, d)
        print(f'{g}: {SET_FILE[g]} {tuple(imgs[g].shape)} {imgs[g].dtype} sha256 {d}', flush=True)
    return imgs


def encoder(full, ep):
    import yaml
    import knee_lidar as KL
    cfg = yaml.safe_load(open(os.path.join(KL.drv.ROOT, full, 'config.yml')))
    tag = cfg['logging']['write_tag']
    ck = os.path.join(KL.drv.ROOT, full, f'{tag}-ep{ep}.pth.tar')
    assert os.path.exists(ck), ck
    return KL.drv.encoder_from_ckpt(cfg, ck), tag


def features():
    import torch
    import knee_lidar as KL
    need_gpu()
    print(f'GPU: {torch.cuda.get_device_name(0)} '
          f'(CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")})', flush=True)
    imgs = load_sets()
    feats_of = feats_fn()
    os.makedirs(RFEAT_NEW, exist_ok=True)
    P, R19 = plan(), runs19()
    t_all = time.time()
    for s, (full, eps) in P.items():
        out = feat_path(s, full)
        if s in R19:
            assert os.path.exists(out), f'reused file missing: {out}'
            print(f'{s:38s} reused from replica_feats/ (checked by --gate)', flush=True)
            continue
        if os.path.exists(out):
            print(f'{s:38s} exists, skip', flush=True)
            continue
        t0 = time.time()
        feats = {g: [] for g in SET_FILE}
        lid = {g: [] for g in SET_FILE}
        tag = None
        for ep in eps:
            enc, tag = encoder(full, ep)
            for g in SET_FILE:
                f = feats_of(enc, imgs[g])
                lid[g].append(KL.lidar(f.reshape(M, K, -1)))
                feats[g].append(f.float().cpu())
            del enc
        obj = {'run': full, 'epochs': list(eps), 'write_tag': tag, 'sets': dict(SET_FILE),
               'set_sha256': dict(SET_SHA), 'script': 'knee_lidar_delta_sweep.py --features'}
        for g in SET_FILE:
            obj[g] = torch.stack(feats[g])                       # (n_ckpt, 512, D) float32
            obj[f'lidar_{g}'] = lid[g]
        assert not os.path.exists(out), f'refusing to overwrite {out}'
        torch.save(obj, out + '.partial')
        os.replace(out + '.partial', out)
        print(f'{s:38s} {len(eps)} ckpts D={obj["redrawn"].shape[-1]} '
              f'({time.time() - t0:.0f}s; total {time.time() - t_all:.0f}s)', flush=True)
    print('DONE features')


# ================================================================ linear algebra

_Q = {}


def noones_basis(D):
    """Orthonormal basis (D x D-1, float64) of the complement of the all-ones direction."""
    import torch
    if D not in _Q:
        A = torch.cat([torch.ones(D, 1, dtype=torch.float64) / math.sqrt(D),
                       torch.eye(D, dtype=torch.float64)], 1)
        Q, _ = torch.linalg.qr(A)
        Qc = Q[:, 1:D].contiguous()
        assert float((torch.ones(D, dtype=torch.float64) @ Qc).abs().max()) < 1e-12
        _Q[D] = Qc
    return _Q[D]


def trace_over_p(F3):
    """tr(raw S_w) / p of an (M, K, D) block, lidar()'s S_w construction."""
    import knee_lidar_spectrum_decomp as SD
    _, Sw = SD.mats(F3)
    return float(Sw.double().trace()) / F3.shape[-1]


def spectrum(F3, delta):
    """lambda1, q, delta's share of the leading direction's denominator, number of raw
    S_w eigenvalues below delta (below lidar()'s floor when delta = 0), and LiDAR from
    the same eigh path. knee_lidar_spectrum_decomp's mats / whitener / decomp; its
    module constant DELTA is set to delta for the whitener call only."""
    import torch
    import knee_lidar_spectrum_decomp as SD
    Sb, Sw_raw = SD.mats(F3)
    old = SD.DELTA
    SD.DELTA = delta
    try:
        W, _ = SD.whitener(Sw_raw)
    finally:
        SD.DELTA = old
    Mw = W @ Sb @ W
    lam, V = torch.linalg.eigh((Mw + Mw.T) / 2)
    lam = lam.clamp(min=0)
    d = SD.decomp(lam)
    x = W @ V[:, -1]
    x = x / x.norm()
    vx = float(x @ Sw_raw @ x)
    share = delta / (vx + delta) if delta > 0 else 0.0
    ew = torch.linalg.eigvalsh(Sw_raw)
    nb = int((ew < (delta if delta > 0 else FLOOR)).sum())
    return dict(lambda1=d['lambda1'], q=d['q'], share=share, n_below=nb,
                lidar_eigh=SD.lidar_from_lam(lam))


def second_way(X64np, delta):
    """LiDAR and lambda1 from the definition with scipy's generalised symmetric
    eigensolver (Cholesky of S_w + delta I): numpy float64, no torch."""
    import numpy as np
    from scipy.linalg import eigh
    D = X64np.shape[1]
    X = X64np.reshape(M, K, D)
    mc = X.mean(1)
    mu = mc.mean(0)
    Sb = (mc - mu).T @ (mc - mu) / (M - 1)
    Xw = (X - mc[:, None, :]).reshape(-1, D)
    Sw = Xw.T @ Xw / (M * (K - 1))
    lam = eigh(Sb, Sw + delta * np.eye(D), eigvals_only=True)
    lam = np.clip(lam, 0, None)
    p = lam / lam.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum())), float(lam.max())


# ================================================================ audit pieces

def audit_counts(S):
    """knee_lidar_replica_seed.counts_from_S, flattened: per test the knee alarms and
    timely first alarms per campaign, the held-out / MAE / ViT-Tiny flat false alarms,
    any of four, thresholds."""
    import knee_lidar_audit_sharedshape as A2
    import knee_lidar_replica_seed as RS
    c = RS.counts_from_S(S)
    out = {'thr': c['thr']}
    for t in A2.TESTS:
        for camp in ('21x', '4x'):
            n, tot, d = c[('knee', camp, t)]
            out[(t, camp)] = (n, tot, sum(abs(x) <= 10 for x in d), sorted(d))
        for pop in ('heldout', 'mae', 'tiny'):
            fired, n = c[(pop, t)]
            out[(t, pop)] = (len(fired), n, [(A2.short(r), c['sc'][r][t][2]) for r in fired])
    for pop in ('heldout', 'mae', 'tiny'):
        fired, n = c[(pop, 'any')]
        out[('any', pop)] = (len(fired), n, [A2.short(r) for r in fired])
    return out


def indep_cusum(S):
    """CUSUM counts written from the definitions (not imported): increments standardised
    on epochs 21-60, statistic from epoch 21, k = 0.5, threshold 1.1 x max over each
    campaign's four mu = 0.97 controls, 4x runs scored against 4x thresholds."""
    import knee_lidar_audit as AUD
    import knee_lidar_audit_sharedshape as A2
    ctrl = {'21x': ['rigd_lejp_p1s100_mu097_nano_1', 'rigd_lejp_p1s100_mu097_nano_2',
                    'rigd_lejp_p1s100_mu097_nano_3', 'rigd_lejp_p1s100_mu097_long_nano_1'],
            '4x': [f'rigd_lejp_p1s100_mu097_p384_{i}' for i in (1, 2, 3, 4)]}

    def trace(run):
        eps, xs = S[run]
        v = [xs[i] - xs[i - 1] for i in range(1, len(xs))]
        e2 = eps[1:]
        cal = [x for e, x in zip(e2, v) if 21 <= e <= 60]
        mu = sum(cal) / len(cal)
        sd = math.sqrt(sum((x - mu) ** 2 for x in cal) / len(cal)) or 1e-12
        sp = sn = 0.0
        out = []
        for e, x in zip(e2, v):
            if e <= 20:
                continue
            z = (x - mu) / sd
            sp, sn = max(0.0, sp + z - 0.5), max(0.0, sn - z - 0.5)
            out.append((e, max(sp, sn)))
        return out
    camp = lambda r: '4x' if '_p384_' in r else '21x'   # noqa: E731
    thr = {c: 1.1 * max(max(v for _, v in trace(r)) for r in rs) for c, rs in ctrl.items()}
    first = lambda r: next((e for e, v in trace(r) if v > thr[camp(r)]), None)  # noqa: E731
    res = {'thr': thr}
    for c, knee in (('21x', AUD.KNEE_21), ('4x', AUD.KNEE_4)):
        al = {k: first(k) for k in knee}
        res[c] = (sum(a is not None for a in al.values()),
                  sum(a is not None and abs(a - knee[k]) <= 10 for k, a in al.items()))
        ratios = []
        for k, o in knee.items():
            w = [v for e, v in trace(k) if o - 10 <= e <= o + 10]
            ratios.append(max(w) / thr[c])
        res[c + '_ratio_median'] = smedian(ratios)
    for pop, runs in (('heldout', A2.HELD_21 + A2.HELD_4), ('mae', A2.MAE), ('tiny', A2.TINY_FLAT)):
        res[pop] = sum(first(r) is not None for r in runs)
    return res


def s2a_rows(S):
    import knee_lidar_audit as AUD
    import knee_labelfree_audit as LA
    import knee_s2_monitors as S2
    out = {}
    for camp, knee in (('21x', AUD.KNEE_21), ('4x', AUD.KNEE_4)):
        at, pl = LA.s2a(S, knee)
        out[camp] = (S2.median(at), len(at), S2.median(pl), len(pl))
    return out


def pct(v, p):
    s = sorted(v)
    return s[min(len(s) - 1, max(0, int(math.ceil(p / 100 * len(s))) - 1))]


def read_sweep(path=OUT):
    """label -> grouping -> short -> {epoch: row dict (floats)}."""
    D = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    with open(path) as fh:
        for r in csv.DictReader(fh):
            D[r['delta_label']][r['grouping']][short(r['run'])][int(r['epoch'])] = dict(
                lidar=float(r['lidar']), lambda1=float(r['lambda1']), q=float(r['q']),
                delta=float(r['delta_value']), share=float(r['delta_share']),
                n_below=int(r['n_sw_below']), trp=float(r['trace_sw_over_p']))
    return D


def series(Dg, key='lidar'):
    return {s: (sorted(v), [v[e][key] for e in sorted(v)]) for s, v in Dg.items()}


def audit_label(Dg, label, g, emit):
    """Every audit table for one (label, grouping) series; emit(**fields) records a row."""
    import knee_lidar_audit as AUD
    import knee_lidar_audit_sharedshape as A2
    import knee_lidar_spectrum_decomp as SD
    import knee_labelfree_calib_sensitivity as LCS
    import knee_s2_monitors as S2
    S = series(Dg)
    base = dict(delta_label=label, grouping=g)

    def E(**kw):
        emit(dict(base, **kw))

    # ---- standing audit, four tests
    c = audit_counts(S)
    for t in A2.TESTS:
        for camp in ('21x', '4x'):
            n, tot, ti, d = c[(t, camp)]
            E(table='standing', population='knee', campaign=camp, test=t, quantity='threshold',
              value=f'{c["thr"][camp][t]:.4f}')
            E(table='standing', population='knee', campaign=camp, test=t, quantity='alarms',
              value=f'{n}/{tot}')
            E(table='standing', population='knee', campaign=camp, test=t, quantity='timely',
              value=f'{ti}/{tot}')
            E(table='standing', population='knee', campaign=camp, test=t,
              quantity='alarm_minus_onset', value=' '.join(f'{x:+d}' for x in d))
        for pop in ('heldout', 'mae', 'tiny'):
            n, tot, fired = c[(t, pop)]
            E(table='standing', population=pop, test=t, quantity='alarms', value=f'{n}/{tot}')
            E(table='standing', population=pop, test=t, quantity='runs',
              value=' '.join(f'{r}@{e}' for r, e in fired))
    for pop in ('heldout', 'mae', 'tiny'):
        n, tot, fired = c[('any', pop)]
        E(table='standing', population=pop, test='any', quantity='alarms', value=f'{n}/{tot}')
        E(table='standing', population=pop, test='any', quantity='runs', value=' '.join(fired))

    # ---- calibration rows (A, LOO, B-flat, H-300ep-only)
    crows = LCS.calib_rows(S, PROTO[g])
    for row in crows:
        for q in ('n_cal', 'thr', 'knee_alarms', 'near_onset', 'heldout_fa', 'excluded_ctrl_fa',
                  'mae_tiny', 'onset_window_ratio_median', 'onset_window_ratio_max',
                  'frac_runs_window_above_thr'):
            E(table='calib', campaign=row['campaign'], calibration=row['calibration'],
              quantity=q, value=str(row[q]))

    # ---- S2a
    for camp, (a, na, p, npl) in s2a_rows(S).items():
        E(table='s2a', campaign=camp, quantity='knee_median', value=f'{a:.3f}')
        E(table='s2a', campaign=camp, quantity='knee_n', value=str(na))
        E(table='s2a', campaign=camp, quantity='placebo_median', value=f'{p:.3f}')
        E(table='s2a', campaign=camp, quantity='placebo_n', value=str(npl))

    # ---- PRE -> POST (knee: also onset), LiDAR and lambda1; share and n_below
    P, _ = SD.build_plan()
    acc = defaultdict(list)
    for full, s, role, camp, onset, cks in P:
        ph = dict(cks)
        v = {p: Dg[s][e] for p, e in ph.items()}
        pop = 'knee' if role == 'knee' else 'eventfree'
        acc[(pop, 'dL')].append(v['post']['lidar'] - v['pre']['lidar'])
        acc[(pop, 'l1pre')].append(v['pre']['lambda1'])
        acc[(pop, 'l1post')].append(v['post']['lambda1'])
        acc[(pop, 'Lpre')].append(v['pre']['lidar'])
        acc[(pop, 'Lpost')].append(v['post']['lidar'])
        if pop == 'knee':
            acc[(pop, 'dLon')].append(v['onset']['lidar'] - v['pre']['lidar'])
            acc[(pop, 'Lon')].append(v['onset']['lidar'])
        for p, row in v.items():
            acc[(pop, 'share', p)].append(row['share'])
            acc[(pop, 'nb', p)].append(row['n_below'])
    for pop, n in (('knee', 28), ('eventfree', 21)):  # runs added 2026-10-13: + nano_11; + nano_10, nano_12
        dL, a, b = acc[(pop, 'dL')], acc[(pop, 'l1pre')], acc[(pop, 'l1post')]
        assert len(dL) == n
        E(table='prepost', population=pop, quantity='lidar_pre_median', value=f'{smedian(acc[(pop, "Lpre")]):.4f}')
        E(table='prepost', population=pop, quantity='lidar_post_median', value=f'{smedian(acc[(pop, "Lpost")]):.4f}')
        E(table='prepost', population=pop, quantity='lidar_change_median', value=f'{smedian(dL):+.4f}')
        E(table='prepost', population=pop, quantity='lidar_change_min', value=f'{min(dL):+.4f}')
        E(table='prepost', population=pop, quantity='lidar_change_max', value=f'{max(dL):+.4f}')
        E(table='prepost', population=pop, quantity='lidar_rises', value=f'{sum(x > 0 for x in dL)}/{n}')
        E(table='prepost', population=pop, quantity='lambda1_pre_median', value=f'{smedian(a):.4g}')
        E(table='prepost', population=pop, quantity='lambda1_post_median', value=f'{smedian(b):.4g}')
        E(table='prepost', population=pop, quantity='lambda1_falls', value=f'{sum(y < x for x, y in zip(a, b))}/{n}')
        E(table='prepost', population=pop, quantity='lambda1_ratio_median',
          value=f'{smedian(y / x for x, y in zip(a, b)):.4f}')
        if pop == 'knee':
            dO = acc[(pop, 'dLon')]
            E(table='prepost', population=pop, quantity='lidar_onset_median', value=f'{smedian(acc[(pop, "Lon")]):.4f}')
            E(table='prepost', population=pop, quantity='lidar_pre_to_onset_median', value=f'{smedian(dO):+.4f}')
            E(table='prepost', population=pop, quantity='lidar_pre_to_onset_rises', value=f'{sum(x > 0 for x in dO)}/{n}')
        for p in (('pre', 'onset', 'post') if pop == 'knee' else ('pre', 'post')):
            sh, nb = acc[(pop, 'share', p)], acc[(pop, 'nb', p)]
            E(table='share', population=pop, phase=p, quantity='share_median', value=f'{smedian(sh):.4f}')
            E(table='share', population=pop, phase=p, quantity='share_min', value=f'{min(sh):.4f}')
            E(table='share', population=pop, phase=p, quantity='share_max', value=f'{max(sh):.4f}')
            E(table='share', population=pop, phase=p, quantity='n_below_median', value=f'{smedian(nb):g}')
    # event-free, every checkpoint at epoch >= 30 (the verifier's third population)
    ef = runs19()
    sh = [Dg[s][e]['share'] for s in ef for e in Dg[s] if e >= 30]
    nb = [Dg[s][e]['n_below'] for s in ef for e in Dg[s] if e >= 30]
    E(table='share', population='eventfree', phase='all_ep30', quantity='share_median', value=f'{smedian(sh):.4f}')
    E(table='share', population='eventfree', phase='all_ep30', quantity='share_min', value=f'{min(sh):.4f}')
    E(table='share', population='eventfree', phase='all_ep30', quantity='share_max', value=f'{max(sh):.4f}')
    E(table='share', population='eventfree', phase='all_ep30', quantity='n_below_median', value=f'{smedian(nb):g}')
    E(table='share', population='eventfree', phase='all_ep30', quantity='n_checkpoints', value=str(len(sh)))

    # ---- step noise: S2.local_dev's step (consecutive checkpoints, epoch > 20)
    for pop, runs in (('eventfree', ef), ('mae', A2.MAE), ('knee', list(AUD.KNEE_21) + list(AUD.KNEE_4))):
        st, lv, rel, cal = [], [], [], []
        for r in runs:
            eps, xs = S[r]
            for i in range(1, len(eps)):
                if eps[i] > S2.WARM:
                    st.append(abs(xs[i] - xs[i - 1]))
                    lv.append(xs[i])
                    rel.append(abs(xs[i] - xs[i - 1]) / xs[i])
            v = [xs[i] - xs[i - 1] for i in range(1, len(xs))]
            cw = [x for e, x in zip(eps[1:], v) if 21 <= e <= 60]
            m = sum(cw) / len(cw)
            cal.append(math.sqrt(sum((x - m) ** 2 for x in cw) / len(cw)))
        E(table='steps', population=pop, quantity='abs_step_median', value=f'{smedian(st):.4f}')
        E(table='steps', population=pop, quantity='abs_step_p90', value=f'{pct(st, 90):.4f}')
        E(table='steps', population=pop, quantity='level_median', value=f'{smedian(lv):.4f}')
        E(table='steps', population=pop, quantity='rel_step_median', value=f'{smedian(rel):.4f}')
        E(table='steps', population=pop, quantity='calib_increment_sd_median', value=f'{smedian(cal):.4f}')
        E(table='steps', population=pop, quantity='n_steps', value=str(len(st)))

    # ---- independent CUSUM
    ic = indep_cusum(S)
    agree = (all(abs(ic['thr'][cc] - c['thr'][cc]['CUSUM']) < 1e-9 for cc in ('21x', '4x'))
             and all(ic[cc] == (c[('CUSUM', cc)][0], c[('CUSUM', cc)][2]) for cc in ('21x', '4x'))
             and all(ic[p] == c[('CUSUM', p)][0] for p in ('heldout', 'mae', 'tiny'))
             and all(round(ic[r['campaign'] + '_ratio_median'], 3) == r['onset_window_ratio_median']
                     for r in crows if r['calibration'] == 'A'))
    E(table='check', quantity='indep_cusum_agrees', value=str(agree))
    return c, ic


# ================================================================ gate

def gate():
    import numpy as np
    import torch
    import knee_lidar as KL
    import knee_lidar_audit as AUD
    import knee_lidar_audit_sharedshape as A2
    import knee_lidar_spectrum_decomp as SD
    import knee_labelfree_calib_sensitivity as LCS
    need_gpu()
    torch.set_num_threads(1)
    print(f'GPU: {torch.cuda.get_device_name(0)} '
          f'(CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")})', flush=True)
    P, R19 = plan(), runs19()
    cv = csv_values()
    imgs = load_sets()
    feats_of = feats_fn()

    # ---- G0 files + re-extraction of sampled checkpoints
    print('\n== G0: feature files ==')
    n_ck, worst_re, n_re = 0, 0.0, 0
    for s, (full, eps) in P.items():
        R = load_feats(s, full, eps)
        for g in GROUPINGS:
            assert R[g].dtype == torch.float32 and tuple(R[g].shape[:2]) == (len(eps), 512), (s, g)
            assert R['set_sha256'][g] == SET_SHA[g], (s, g)
        exp_script = 'knee_lidar_killtest.py --features' if s in R19 else 'knee_lidar_delta_sweep.py --features'
        assert R['script'] == exp_script, (s, R['script'])
        n_ck += len(eps)
        js = [len(eps) - 1] + ([0] if s in R19 else [])
        for j in js:
            enc, _ = encoder(full, eps[j])
            for g in GROUPINGS:
                f = feats_of(enc, imgs[g]).float().cpu()
                worst_re = max(worst_re, float((f - R[g][j]).abs().max()))
                n_re += 1
            del enc
    print(f'  58 runs, {n_ck} checkpoints; dtype float32, shape (n_ckpt, 512, D), set sha256 and '
          f'writer script as expected in every file')
    print(f'  re-extracted {n_re} (checkpoint, set) blocks (last checkpoint of every run, first '
          f'checkpoint of the 19 reused runs): max |re-extracted - stored| = {worst_re:.2e}')

    # ---- G1 LiDAR float32 GPU at delta = 1e-4 against the CSVs
    print('\n== G1: lidar() at delta = 1e-4, float32 on the GPU, from the stored features ==')
    S32 = {g: {} for g in GROUPINGS}
    S64 = {g: {} for g in GROUPINGS}
    Scsv = {g: {} for g in GROUPINGS}
    worst = {g: [0.0, 0.0, 0, 0.0, 0.0, 0.0] for g in GROUPINGS}   # abs, rel, str mismatches, ext, f64 abs, f64 rel
    feats_cache = {}
    for s, (full, eps) in P.items():
        R = load_feats(s, full, eps)
        feats_cache[s] = {g: R[g] for g in GROUPINGS}
        for g in GROUPINGS:
            v32, v64, vc = [], [], []
            for j, ep in enumerate(eps):
                X = R[g][j]
                L = KL.lidar(X.to(KL.DEV).reshape(M, K, -1))
                L64 = KL.lidar(X.double().reshape(M, K, -1), delta=1e-4)
                c, cstr = cv[(g, s, ep)]
                w = worst[g]
                w[0] = max(w[0], abs(L - c))
                w[1] = max(w[1], abs(L / c - 1))
                w[2] += f6(L) != cstr
                w[3] = max(w[3], abs(L - R[f'lidar_{g}'][j]))
                w[4] = max(w[4], abs(L64 - c))
                w[5] = max(w[5], abs(L64 / c - 1))
                v32.append(float(f6(L)))
                v64.append(float(f6(L64)))
                vc.append(c)
            S32[g][s] = (list(eps), v32)
            S64[g][s] = (list(eps), v64)
            Scsv[g][s] = (list(eps), vc)
    for g in GROUPINGS:
        w = worst[g]
        print(f'  {g:8s} {n_ck} checkpoints vs {os.path.basename(LID_CSV[g])}: max |d| {w[0]:.2e}, '
              f'max relative |d| {w[1]:.2e}, 6-decimal strings differing {w[2]}; vs the value stored at '
              f'extraction max |d| {w[3]:.2e}')
        print(f'  {g:8s} the same in float64 on the CPU (the sweep\'s precision): max |d| {w[4]:.2e}, '
              f'max relative |d| {w[5]:.2e}')

    # ---- G2 the standing audit at delta = 1e-4
    print('\n== G2: the standing audit at delta = 1e-4 against the record ==')
    ref = list(csv.DictReader(open(CALIB_CSV)))
    for tag, SS in (('standing CSVs', Scsv), ('float32 GPU recomputed (6 dp)', S32),
                    ('float64 CPU recomputed (6 dp)', S64)):
        for g in GROUPINGS:
            c = audit_counts(SS[g])
            rec = RECORD[g]
            bad = []
            for t in A2.TESTS:
                got = (c[(t, '21x')][0], c[(t, '4x')][0], c[(t, 'heldout')][0], c[(t, 'mae')][0],
                       c[(t, 'tiny')][0])
                if got != rec[t]:
                    bad.append(f'{t} {got} vs {rec[t]}')
            got_t = (c[('CUSUM', '21x')][2], c[('CUSUM', '4x')][2])
            if got_t != rec['timely']:
                bad.append(f'timely {got_t} vs {rec["timely"]}')
            got_a = tuple(c[('any', p)][0] for p in ('heldout', 'mae', 'tiny'))
            if got_a != rec['any']:
                bad.append(f'any {got_a} vs {rec["any"]}')
            thr = (round(c['thr']['21x']['CUSUM'], 4), round(c['thr']['4x']['CUSUM'], 4))
            s2 = s2a_rows(SS[g])
            s2g = {cc: (round(s2[cc][0], 3), round(s2[cc][2], 3)) for cc in s2}
            if s2g != rec['s2a']:
                bad.append(f's2a {s2g} vs {rec["s2a"]}')
            mine = LCS.calib_rows(SS[g], PROTO[g])
            refg = [r for r in ref if r['protocol'] == PROTO[g]]
            assert len(mine) == len(refg)
            ncell = sum(str(a[col]) != v for a, b in zip(mine, refg) for col, v in b.items())
            print(f'  {tag:30s} {g:8s}: CUSUM knee {c[("CUSUM", "21x")][0]}/12 {c[("CUSUM", "4x")][0]}/15, '
                  f'timely {got_t[0]} + {got_t[1]}, held-out {c[("CUSUM", "heldout")][0]}/11, MAE '
                  f'{c[("CUSUM", "mae")][0]}/10, tiny {c[("CUSUM", "tiny")][0]}/1; thresholds {thr} (record '
                  f'{rec["thr"]}); differences from the record in the four tests / timely / any / S2a: '
                  f'{bad if bad else "none"}; calibration rows vs v2 CSV: {len(mine)} rows, {ncell} differing cells')

    # ---- G3 lambda1 against the decomposition CSV; the verifier's delta values
    print('\n== G3a: lambda1 at delta = 1e-4 against csv/knee_lidar_spectrum_decomp.csv ==')
    dec = {}
    with open(DECOMP_CSV) as fh:
        for r in csv.DictReader(fh):
            if r['variant'] == 'raw' and r['term'] in ('redrawn|redrawn', 'shared|shared'):
                dec[(r['term'].split('|')[0], short(r['run']), int(r['epoch']))] = float(r['lambda1'])
    for g in GROUPINGS:
        w32 = w64 = 0.0
        n = 0
        for (gg, s, ep), l1 in dec.items():
            if gg != g:
                continue
            j = P[s][1].index(ep)
            X = feats_cache[s][g][j]
            a = spectrum(X.to(KL.DEV).reshape(M, K, -1), 1e-4)['lambda1']
            b = spectrum(X.double().reshape(M, K, -1), 1e-4)['lambda1']
            w32, w64 = max(w32, abs(a / l1 - 1)), max(w64, abs(b / l1 - 1))
            n += 1
        print(f'  {g:8s} {n} checkpoints (27 knee PRE/onset/POST, 19 event-free PRE/POST): max relative '
              f'|d| float32 GPU {w32:.2e}, float64 CPU {w64:.2e}')

    print('\n== G3b: the verifier\'s knee_stats.pkl (knee PRE / onset / POST, CPU-extracted features) ==')
    if not os.path.exists(VERIFIER_STATS):
        print(f'  {VERIFIER_STATS} not found; skipped')
        print('\nGATE DONE')
        return
    st = pickle.load(open(VERIFIER_STATS, 'rb'))
    DL = (1e-4, 1e-6, 1e-8, 0.0)
    mine = {}
    for (s, ep, g) in st:
        j = P[s][1].index(ep)
        F3 = feats_cache[s][g][j].double().reshape(M, K, -1)
        mine[(s, ep, g)] = {d: (KL.lidar(F3, delta=d), spectrum(F3, d)['lambda1']) for d in DL}
    print(f'  {len(st)} (run, epoch, grouping) entries')
    for d in DL:
        for g in GROUPINGS:
            ks = [k for k in st if k[2] == g]
            rl = max(abs(mine[k][d][0] / st[k]['L'][d][0] - 1) for k in ks)
            r1 = max(abs(mine[k][d][1] / st[k]['L'][d][1] - 1) for k in ks)
            ml = smedian(abs(mine[k][d][0] / st[k]['L'][d][0] - 1) for k in ks)
            print(f'  delta {d:g} {g:8s}: mine (GPU features) vs verifier: LiDAR max rel |d| {rl:.2e} '
                  f'(median {ml:.1e}), lambda1 max rel |d| {r1:.2e}')
    # PRE -> POST counts, mine vs the verifier's
    ph = {}
    with open(DECOMP_CSV) as fh:
        for r in csv.DictReader(fh):
            if r['role'] == 'knee' and r['term'] == 'redrawn|redrawn' and r['variant'] == 'raw':
                ph[(short(r['run']), r['phase'])] = int(r['epoch'])
    runs = sorted({s for s, _ in ph})
    for d in DL:
        for g in GROUPINGS:
            cells = []
            for who, src in (('mine', lambda k: mine[k][d]), ('verifier', lambda k: st[k]['L'][d])):
                up = sum(src((s, ph[(s, 'post')], g))[0] > src((s, ph[(s, 'pre')], g))[0] for s in runs)
                fa = sum(src((s, ph[(s, 'post')], g))[1] < src((s, ph[(s, 'pre')], g))[1] for s in runs)
                cells.append(f'{who}: LiDAR rises {up}/27, lambda1 falls {fa}/27')
            print(f'  delta {d:g} {g:8s} PRE -> POST: ' + '; '.join(cells))
    if os.path.exists(VERIFIER_FEATS):
        kf = pickle.load(open(VERIFIER_FEATS, 'rb'))
        wf = wl = w1 = 0.0
        for (s, ep), rec in kf.items():
            j = P[s][1].index(ep)
            for g in GROUPINGS:
                wf = max(wf, float((rec[g].float() - feats_cache[s][g][j]).abs().max()))
                F3 = rec[g].double().reshape(M, K, -1)
                for d in DL:
                    wl = max(wl, abs(KL.lidar(F3, delta=d) / st[(s, ep, g)]['L'][d][0] - 1))
                    w1 = max(w1, abs(spectrum(F3, d)['lambda1'] / st[(s, ep, g)]['L'][d][1] - 1))
        print(f'  on the verifier\'s own CPU features ({len(kf)} checkpoints, both groupings, 4 deltas): '
              f'this script\'s LiDAR vs the verifier\'s max rel |d| {wl:.2e}, lambda1 {w1:.2e}; '
              f'max |CPU feature - GPU feature| {wf:.2e}')
    else:
        print(f'  {VERIFIER_FEATS} not found; the formula check on the verifier\'s own features skipped')
    print('\nGATE DONE')


# ================================================================ run

def run():
    import numpy as np
    import torch
    import knee_lidar as KL
    need_gpu()
    torch.set_num_threads(1)
    for p in (OUT, OUT_AUDIT):
        assert not os.path.exists(p), f'refusing to overwrite {p}'
    print(f'GPU: {torch.cuda.get_device_name(0)} '
          f'(CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")})', flush=True)
    P, R19 = plan(), runs19()
    t_all = time.time()

    # ---- eps of the relative convention: median tr(S_w)/p over the 19 event-free runs
    trp = {g: [] for g in GROUPINGS}
    for s in R19:
        full, eps = P[s]
        R = load_feats(s, full, eps)
        for g in GROUPINGS:
            for j in range(len(eps)):
                trp[g].append(trace_over_p(R[g][j].double().reshape(M, K, -1)))
    med = {g: smedian(v) for g, v in trp.items()}
    eps_rel = {g: {lab: tgt / med[g] for lab, tgt in REL.items()} for g in GROUPINGS}
    print(f'relative convention: median tr(S_w)/p over {len(trp["redrawn"])} event-free checkpoints '
          + ', '.join(f'{g} {med[g]:.4e}' for g in GROUPINGS) + '; eps '
          + ', '.join(f'{g} {lab} {eps_rel[g][lab]:.4e}' for g in GROUPINGS for lab in REL), flush=True)

    rows = []
    prec = defaultdict(list)          # (label, g) -> relative |f32 GPU - f64|
    chk = defaultdict(float)          # worst second-way / path differences
    for s, (full, eps) in P.items():
        t0 = time.time()
        R = load_feats(s, full, eps)
        for j, ep in enumerate(eps):
            for g in GROUPINGS:
                X = R[g][j]
                D = X.shape[1]
                Fg = X.to(KL.DEV).reshape(M, K, -1)
                F64 = X.double().reshape(M, K, -1)
                tp = trace_over_p(F64)
                Xnp = X.double().numpy()

                def emit(label, dval, L, sp):
                    rows.append(dict(run=full, epoch=ep, grouping=g, delta_label=label,
                                     delta_value=f'{dval:.6e}', lidar=f6(L),
                                     lambda1=f'{sp["lambda1"]:.6e}', q=f6(sp['q']),
                                     delta_share=f6(sp['share']), n_sw_below=sp['n_below'],
                                     trace_sw_over_p=f'{tp:.6e}'))
                # standing pipeline: float32 on the GPU
                L32 = KL.lidar(Fg)
                emit(F32, 1e-4, L32, spectrum(Fg, 1e-4))
                todo = [(lab, d) for lab, d in ABS.items()] + [(lab, eps_rel[g][lab] * tp) for lab in REL]
                for lab, d in todo:
                    L = KL.lidar(F64, delta=d)
                    sp = spectrum(F64, d)
                    emit(lab, d, L, sp)
                    chk[('eigh_vs_eigvalsh', lab)] = max(chk[('eigh_vs_eigvalsh', lab)],
                                                        abs(sp['lidar_eigh'] - L) / L)
                    if lab in ABS:
                        prec[(lab, g)].append(abs(KL.lidar(Fg, delta=d) / L - 1))
                    if d > 0:
                        L2, l12 = second_way(Xnp, d)
                        chk[('second_way_lidar', lab)] = max(chk[('second_way_lidar', lab)],
                                                            abs(float(f6(L)) - L2) / L2)
                        chk[('second_way_lambda1', lab)] = max(chk[('second_way_lambda1', lab)],
                                                              abs(sp['lambda1'] / l12 - 1))
                Fp = (X.double() @ noones_basis(D)).reshape(M, K, -1)
                emit(NOONES, 0.0, KL.lidar(Fp, delta=0.0), spectrum(Fp, 0.0))
        print(f'{s:38s} {len(eps)} ckpts ({time.time() - t0:.0f}s; total {time.time() - t_all:.0f}s)',
              flush=True)

    assert len(rows) == 2228 * 2 * len(LABELS), len(rows)  # includes the runs added 2026-10-13
    with open(OUT + '.partial', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=SWEEP_COLS, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    assert not os.path.exists(OUT)
    os.replace(OUT + '.partial', OUT)
    print(f'wrote {OUT}: {len(rows)} rows', flush=True)

    # ---- the audit, from the CSV just written
    arows = []
    emit = arows.append
    for g in GROUPINGS:
        emit(dict(grouping=g, table='rel_eps', quantity='median_trace_sw_over_p_eventfree',
                  value=f'{med[g]:.6e}'))
        for lab in REL:
            emit(dict(grouping=g, table='rel_eps', delta_label=lab, quantity='eps',
                      value=f'{eps_rel[g][lab]:.6e}'))
    for (lab, g), v in sorted(prec.items()):
        emit(dict(delta_label=lab, grouping=g, table='check', quantity='f32gpu_vs_f64_rel_max',
                  value=f'{max(v):.3e}'))
        emit(dict(delta_label=lab, grouping=g, table='check', quantity='f32gpu_vs_f64_rel_median',
                  value=f'{smedian(v):.3e}'))
    for (q, lab), v in sorted(chk.items()):
        emit(dict(delta_label=lab, table='check', quantity=q + '_rel_max', value=f'{v:.3e}'))
    D = read_sweep(OUT)
    for lab in LABELS:
        for g in GROUPINGS:
            audit_label(D[lab][g], lab, g, emit)
    for r in arows:
        for c in AUDIT_COLS:
            r.setdefault(c, '')
    with open(OUT_AUDIT + '.partial', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=AUDIT_COLS, lineterminator='\n')
        w.writeheader()
        w.writerows(arows)
    assert not os.path.exists(OUT_AUDIT)
    os.replace(OUT_AUDIT + '.partial', OUT_AUDIT)
    print(f'wrote {OUT_AUDIT}: {len(arows)} rows ({time.time() - t_all:.0f}s)', flush=True)
    report()


# ================================================================ report

def report():
    A = defaultdict(dict)
    with open(OUT_AUDIT) as fh:
        for r in csv.DictReader(fh):
            A[(r['delta_label'], r['grouping'], r['table'], r['population'], r['phase'],
               r['campaign'], r['test'], r['calibration'])][r['quantity']] = r['value']

    def get(lab, g, table, q, population='', phase='', campaign='', test='', calibration=''):
        return A[(lab, g, table, population, phase, campaign, test, calibration)][q]

    def frac(x):
        a, b = x.split('/')
        return int(a), int(b)

    def dlabel(lab):
        return {'f32gpu_1e-4': '1e-4 (float32 GPU, standing)', '0': '0 (floor 1e-12; flagged)',
                '0_noones': '0, all-ones direction removed', 'rel_a': 'relative, 1e-4 at median',
                'rel_b': 'relative, 1e-6 at median'}.get(lab, lab)

    print('## Headline: CUSUM, standing calibration (1.1 x max over the four mu = 0.97 controls, '
          'recalibrated at each delta)\n')
    print('| delta | grouping | timely first alarms /27 (21x + 4x) | any alarm /27 | held-out LeJEPA /11 | '
          'MAE /10 | ViT-Tiny flat /1 | onset-window ratio median 21x / 4x | thresholds 21x / 4x |')
    print('|---|---|---|---|---|---|---|---|---|')
    for lab in LABELS:
        for g in GROUPINGS:
            t21 = frac(get(lab, g, 'standing', 'timely', 'knee', campaign='21x', test='CUSUM'))
            t4 = frac(get(lab, g, 'standing', 'timely', 'knee', campaign='4x', test='CUSUM'))
            a21 = frac(get(lab, g, 'standing', 'alarms', 'knee', campaign='21x', test='CUSUM'))
            a4 = frac(get(lab, g, 'standing', 'alarms', 'knee', campaign='4x', test='CUSUM'))
            r21 = get(lab, g, 'calib', 'onset_window_ratio_median', campaign='21x', calibration='A')
            r4 = get(lab, g, 'calib', 'onset_window_ratio_median', campaign='4x', calibration='A')
            print(f'| {dlabel(lab)} | {g} | {t21[0] + t4[0]} ({t21[0]} + {t4[0]}) | {a21[0] + a4[0]} '
                  f'({a21[0]} + {a4[0]}) | {get(lab, g, "standing", "alarms", "heldout", test="CUSUM")} '
                  f'| {get(lab, g, "standing", "alarms", "mae", test="CUSUM")} | '
                  f'{get(lab, g, "standing", "alarms", "tiny", test="CUSUM")} | {r21} / {r4} | '
                  f'{float(get(lab, g, "standing", "threshold", "knee", campaign="21x", test="CUSUM")):.2f} / '
                  f'{float(get(lab, g, "standing", "threshold", "knee", campaign="4x", test="CUSUM")):.2f} |')

    print('\n## Who alarms: CUSUM held-out and MAE runs, alarm minus onset of knee alarms\n')
    for lab in LABELS:
        for g in GROUPINGS:
            ho = get(lab, g, 'standing', 'runs', 'heldout', test='CUSUM')
            ma = get(lab, g, 'standing', 'runs', 'mae', test='CUSUM')
            d21 = get(lab, g, 'standing', 'alarm_minus_onset', 'knee', campaign='21x', test='CUSUM')
            d4 = get(lab, g, 'standing', 'alarm_minus_onset', 'knee', campaign='4x', test='CUSUM')
            print(f'- {dlabel(lab)}, {g}: held-out [{ho}] MAE [{ma}]; knee 21x [{d21}]; 4x [{d4}]')

    print('\n## The other three tests: knee alarms (timely) 21x, 4x | held-out, MAE, ViT-Tiny flat\n')
    print('| delta | grouping | Page-Hinkley | EWMA | slope11 | any of four: held-out, MAE, tiny |')
    print('|---|---|---|---|---|---|')
    for lab in LABELS:
        for g in GROUPINGS:
            cells = []
            for t in ('PageHinkley', 'EWMA', 'slope11'):
                k = [f'{get(lab, g, "standing", "alarms", "knee", campaign=c, test=t)} '
                     f'({frac(get(lab, g, "standing", "timely", "knee", campaign=c, test=t))[0]})'
                     for c in ('21x', '4x')]
                f = [get(lab, g, 'standing', 'alarms', p, test=t) for p in ('heldout', 'mae', 'tiny')]
                cells.append(', '.join(k) + ' \\| ' + ', '.join(f))
            an = ', '.join(get(lab, g, 'standing', 'alarms', p, test='any') for p in ('heldout', 'mae', 'tiny'))
            print(f'| {dlabel(lab)} | {g} | ' + ' | '.join(cells) + f' | {an} |')

    print('\n## Calibration rows (CUSUM): threshold, knee alarms, within one grid step, held-out '
          'false alarms (7 at 21x, 4 at 4x, less the calibrator), excluded controls, MAE/tiny, '
          'onset-window ratio median / max, runs with window above threshold\n')
    cal_order = {}
    for key in A:
        if key[2] == 'calib':
            cal_order.setdefault((key[5], key[7]), len(cal_order))
    for lab in LABELS:
        for g in GROUPINGS:
            print(f'\n### {dlabel(lab)}, {g}\n')
            print('| campaign | calibration | thr | knee | near onset | held-out FA | excl. ctrl FA | MAE / tiny | '
                  'ratio median | ratio max | window > thr |')
            print('|---|---|---|---|---|---|---|---|---|---|---|')
            for (camp, cal) in sorted(cal_order, key=cal_order.get):
                a = A[(lab, g, 'calib', '', '', camp, '', cal)]
                print(f'| {camp} | {cal} | {float(a["thr"]):.3f} | {a["knee_alarms"]} | {a["near_onset"]} | '
                      f'{a["heldout_fa"]} | {a["excluded_ctrl_fa"]} | {a["mae_tiny"]} | '
                      f'{a["onset_window_ratio_median"]} | {a["onset_window_ratio_max"]} | '
                      f'{a["frac_runs_window_above_thr"]} |')

    print('\n## S2a local-deviation rank at onset (median over knee runs) / placebo median\n')
    print('| delta | redrawn 21x | redrawn 4x | shared 21x | shared 4x |')
    print('|---|---|---|---|---|')
    for lab in LABELS:
        cells = []
        for g in GROUPINGS:
            for c in ('21x', '4x'):
                cells.append(f'{get(lab, g, "s2a", "knee_median", campaign=c)} / '
                             f'{get(lab, g, "s2a", "placebo_median", campaign=c)}')
        print(f'| {dlabel(lab)} | ' + ' | '.join(cells) + ' |')

    print('\n## PRE -> POST: LiDAR and lambda1 (27 knee runs; 19 event-free runs)\n')
    print('| delta | grouping | population | LiDAR PRE / (onset) / POST median | change median (range) | '
          'rises | onset - PRE median, rises | lambda1 PRE -> POST median | falls | POST/PRE median |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for lab in LABELS:
        for g in GROUPINGS:
            for pop in ('knee', 'eventfree'):
                a = A[(lab, g, 'prepost', pop, '', '', '', '')]
                on = (f' / {a["lidar_onset_median"]}' if pop == 'knee' else '')
                oc = (f'{a["lidar_pre_to_onset_median"]}, {a["lidar_pre_to_onset_rises"]}'
                      if pop == 'knee' else '-')
                print(f'| {dlabel(lab)} | {g} | {pop} | {a["lidar_pre_median"]}{on} / {a["lidar_post_median"]} | '
                      f'{a["lidar_change_median"]} ({a["lidar_change_min"]} to {a["lidar_change_max"]}) | '
                      f'{a["lidar_rises"]} | {oc} | {a["lambda1_pre_median"]} -> {a["lambda1_post_median"]} | '
                      f'{a["lambda1_falls"]} | {a["lambda1_ratio_median"]} |')

    print('\n## delta\'s share of the leading whitened direction\'s denominator, median [min, max]; '
          'median number of raw S_w eigenvalues below delta (of 96)\n')
    print('| delta | grouping | knee PRE | knee onset | knee POST | event-free PRE | event-free POST | '
          'event-free all ep >= 30 |')
    print('|---|---|---|---|---|---|---|---|')
    for lab in LABELS:
        if lab in ('0', NOONES):
            continue
        for g in GROUPINGS:
            cells = []
            for pop, ph in (('knee', 'pre'), ('knee', 'onset'), ('knee', 'post'), ('eventfree', 'pre'),
                            ('eventfree', 'post'), ('eventfree', 'all_ep30')):
                a = A[(lab, g, 'share', pop, ph, '', '', '')]
                cells.append(f'{float(a["share_median"]):.2f} [{float(a["share_min"]):.2f}, '
                             f'{float(a["share_max"]):.2f}]; {a["n_below_median"]}')
            print(f'| {dlabel(lab)} | {g} | ' + ' | '.join(cells) + ' |')
    print('\nFor delta = 0, the count is of raw S_w eigenvalues below lidar()\'s floor 1e-12:')
    for lab in ('0', NOONES):
        for g in GROUPINGS:
            cells = []
            for pop, ph in (('knee', 'pre'), ('knee', 'post'), ('eventfree', 'pre'), ('eventfree', 'post'),
                            ('eventfree', 'all_ep30')):
                cells.append(f'{pop} {ph} {A[(lab, g, "share", pop, ph, "", "", "")]["n_below_median"]}')
            print(f'- {dlabel(lab)}, {g}: ' + '; '.join(cells))

    print('\n## Step noise (consecutive checkpoints, epoch > 20): median |step|, 90th pct, median level, '
          'median |step|/level, median sd of increments in the CUSUM calibration window (21-60)\n')
    print('| delta | grouping | event-free (19) | MAE (10) | knee (27) |')
    print('|---|---|---|---|---|')
    for lab in LABELS:
        for g in GROUPINGS:
            cells = []
            for pop in ('eventfree', 'mae', 'knee'):
                a = A[(lab, g, 'steps', pop, '', '', '', '')]
                cells.append(f'{a["abs_step_median"]}, {a["abs_step_p90"]}, {a["level_median"]}, '
                             f'{a["rel_step_median"]}, {a["calib_increment_sd_median"]}')
            print(f'| {dlabel(lab)} | {g} | ' + ' | '.join(cells) + ' |')

    print('\n## Checks\n')
    for g in GROUPINGS:
        print(f'- relative convention, {g}: median tr(S_w)/p over event-free checkpoints '
              f'{get("", g, "rel_eps", "median_trace_sw_over_p_eventfree")}; eps '
              + ', '.join(f'{lab} {get(lab, g, "rel_eps", "eps")}' for lab in REL))
    for lab in ABS:
        print(f'- float32 GPU lidar() vs float64, delta {lab}: '
              + '; '.join(f'{g} max rel {get(lab, g, "check", "f32gpu_vs_f64_rel_max")}, median '
                          f'{get(lab, g, "check", "f32gpu_vs_f64_rel_median")}' for g in GROUPINGS))
    for lab in list(ABS) + list(REL):
        cells = [f'{q}: {A[(lab, "", "check", "", "", "", "", "")].get(q + "_rel_max", "-")}'
                 for q in ('second_way_lidar', 'second_way_lambda1', 'eigh_vs_eigvalsh')]
        print(f'- delta {lab}: ' + '; '.join(cells))
    bad = [(lab, g) for lab in LABELS for g in GROUPINGS
           if A[(lab, g, 'check', '', '', '', '', '')]['indep_cusum_agrees'] != 'True']
    print(f'- independently written CUSUM agrees with the imported machinery on thresholds, knee '
          f'alarms, timely, held-out, MAE and tiny counts: {"all " + str(len(LABELS) * 2) + " series" if not bad else "NOT for " + str(bad)}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', action='store_true')
    ap.add_argument('--gate', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--report', action='store_true')
    a = ap.parse_args()
    if a.features:
        features()
    elif a.gate:
        gate()
    elif a.run:
        run()
    elif a.report:
        report()
    else:
        ap.print_help()
