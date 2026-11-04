"""Kill test for the redrawn-grouping LiDAR statistic (2026-09-24).

Question: does the LiDAR statistic on groups that share energy and position
but redraw shower shape ("redrawn") also move for a readability loss that
leaves energy alone? No training: a synthetic readability loss is built in
feature space on event-free LeJEPA runs, with a positive control.

Material: the 19 event-free LeJEPA runs (8 mu = 0.97 controls,
knee_lidar_audit.CTRL_21 + CTRL_4; 11 held-out runs,
knee_lidar_audit_sharedshape.HELD_21 + HELD_4) at every N = 100 checkpoint
in results/knee_lidar.csv.

Stages
  --features  (GPU) pooled features of the two cached N = 100 replica sets
              (knee_lidar_N100.pt = shape redrawn, knee_lidar_sharedshape_N100.pt
              = shape shared), exactly knee_lidar.lidar_of's path (tokens ->
              LayerNorm -> token mean, chunks of 64, fp32), one file per run in
              paths.WORK/replica_feats/ (float32; refuses to overwrite;
              resumable by run). LiDAR is also computed on the GPU at extraction
              and stored in the file.
  --gate      Gate 0 and the checks the design rests on: LiDAR from the saved
              features against results/knee_lidar.csv and results/knee_lidar_sharedshape.csv
              at every checkpoint; readability from labelfree_feats against
              results/knee_labelfree_readability.csv; the audit code on the real
              knee series against the recorded counts and onset-window ratios;
              lambda1 against results/knee_lidar_spectrum_decomp.csv; the rescaling
              invariance check (G5, final checkpoint of each of the 8 controls,
              factors 0.1 and 10) and its delta dependence (G6).
  --run       (GPU) the synthetic-event series for the arms null / energy /
              shape, 3 noise seeds, both groupings, and the dose-response table.
              Writes results/knee_lidar_killtest.csv and results/knee_lidar_killtest_dose.csv
              (refuse to overwrite; .partial then os.replace). Per-run parts go to
              paths.WORK/replica_feats/killtest_parts/ (resumable by run).
  --report    prints every table of the report from the two CSVs.

Perturbation. LiDAR (generalised eigenvalues of between- vs within-group
covariance, regulariser delta = 1e-4) and least-squares R^2 are invariant to
invertible linear maps up to their regularisers, so a readability loss is made
by adding independent Gaussian noise along chosen feature directions: every
sample (fitting split, evaluation split, every replica of every group) gets
its own draw. The same directions and noise scales apply to every input of a
checkpoint (one "degraded encoder").

Directions, per checkpoint, from the fitting split only, in the geometry of
the house ridge (wp1_metrics.ridge_r2: features z-scored on the fitting split,
W = (Xn'Xn + I)^-1 Xn'Yn). With A = Xn'Xn + I and read-outs w_z1, w_z2, w_E
(columns of W), noise along v changes A to about A + s^2 N v v', and the
fitted signal of target k falls by an amount proportional to (v' w_k)^2
(Sherman-Morrison). So a direction selective for k is one orthogonal to the
other read-outs; the one that removes the most of k per unit of added noise is
v = A a, where a is w_k orthogonalised in the A-metric against the other
read-outs (a' A w_j = 0 for j != k, hence v' w_j = 0).
  energy arm: v_E = A a_E, a_E = w_E orthogonalised against {w_z1, w_z2}
  shape arm:  v_1 = A a_1, a_1 = w_z1 orthogonalised against {w_E, w_z2};
              v_2 = A a_2, a_2 = w_z2 orthogonalised against {w_E, w_z1};
              two independent noise terms with their own scales.
In raw feature units the direction is s * v (s = the fitting split's feature
sd). Selectivity is not assumed: every reported R^2 change is measured with
wp1_metrics.ridge_r2 (refit on the perturbed fitting split, scored on the
perturbed evaluation split).

Dose. Template = the median over the 27 knee runs of R^2(E)(PRE) - R^2(E)(onset
+ k), PRE = last grid epoch <= onset - 20, k on the 10-epoch grid
(results/knee_labelfree_readability.csv; onsets knee_lidar_audit.KNEE_21 / KNEE_4),
linearly interpolated in k. Synthetic onset = the campaign's median knee onset
(115 at 21x, 90 at 4x). Checkpoints before the synthetic onset are left
unperturbed; at a checkpoint e >= onset the noise scale is set (root finding on
the measured evaluation-split drop, per noise seed) so that the target
factor's ridge R^2 falls by template(e - onset) relative to the same
checkpoint unperturbed: R^2(E) for the energy arm; each of R^2(z1) and R^2(z2)
by that same absolute amount for the shape arm.

Audit: standing CUSUM thresholds from the UNPERTURBED controls
(knee_lidar_audit_sharedshape.thresholds on the null series), scoring with
knee_lidar_audit_sharedshape.score, timely = |first alarm - synthetic onset|
<= 10 (knee_lidar_replica_seed.counts_from_S's convention), onset-window
ratio = max CUSUM in [onset - 10, onset + 10] / threshold
(knee_lidar_calib_sensitivity.py's definition, transcribed: that module
asserts at import that its own output does not exist).

Usage (from analysis/, always -B):
  CUDA_VISIBLE_DEVICES=1 python -B knee_lidar_killtest.py --features
  CUDA_VISIBLE_DEVICES=1 python -B knee_lidar_killtest.py --gate
  CUDA_VISIBLE_DEVICES=1 python -B knee_lidar_killtest.py --run
  python -B knee_lidar_killtest.py --report
"""
import sys

sys.dont_write_bytecode = True      # leave analysis/__pycache__ untouched

import argparse  # noqa: E402
import csv  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from collections import OrderedDict, defaultdict  # noqa: E402
from statistics import median  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
sys.path.insert(0, _HERE)

CACHE = paths.CACHE
RFEAT_DIR = os.path.join(paths.WORK, 'replica_feats')  # denom: 19 symlinks + new
PARTS_DIR = os.path.join(RFEAT_DIR, 'killtest_parts')
LFEAT_DIR = os.path.join(paths.WORK, 'labelfree_feats')  # denom: 58 symlinks + 3 new
SET_FILE = OrderedDict((('redrawn', 'knee_lidar_N100.pt'),
                        ('shared', 'knee_lidar_sharedshape_N100.pt')))
LID_CSV = {'redrawn': os.path.join(paths.CSV, 'knee_lidar.csv'),
           'shared': os.path.join(paths.CSV, 'knee_lidar_sharedshape.csv')}
READ_CSV = os.path.join(paths.CSV, 'knee_labelfree_readability.csv')
DECOMP_CSV = os.path.join(paths.CSV, 'knee_lidar_spectrum_decomp.csv')
CALIB_CSV = os.path.join(paths.CSV, 'knee_lidar_calib_sensitivity_v2.csv')
OUT = os.path.join(paths.CSV, 'knee_lidar_killtest.csv')
OUT_DOSE = os.path.join(paths.CSV, 'knee_lidar_killtest_dose.csv')
N_SPOTS = 100
GROUPINGS = tuple(SET_FILE)


def short(run):
    return run.split('xxx')[0]


def runs19():
    import knee_lidar_audit as AUD
    import knee_lidar_audit_sharedshape as A2
    r = AUD.CTRL_21 + AUD.CTRL_4 + A2.HELD_21 + A2.HELD_4
    new = ('rigb_lejp_p1s100_nano_10', 'rigb_lejp_p1s100_nano_12')   # denom 2026-10-13: appended last so the
    return [x for x in r if x not in new] + [x for x in r if x in new]  # original 19 keep their run index (noise keys)


def plan():
    """short run -> (full run name, sorted N = 100 epochs) for the 19 runs."""
    grid, full = defaultdict(set), {}
    with open(LID_CSV['redrawn']) as fh:
        for r in csv.DictReader(fh):
            if r['n_spots'] == str(N_SPOTS):
                s = short(r['run'])
                full[s] = r['run']
                grid[s].add(int(r['epoch']))
    out = OrderedDict()
    for s in runs19():
        out[s] = (full[s], sorted(grid[s]))
    assert len(out) == 21  # denom 2026-10-13: 19 + nano_10, nano_12
    return out


def rfeat_path(full_run):
    return os.path.join(RFEAT_DIR, f'{full_run}.pt')


def need_gpu():
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        raise SystemExit('set CUDA_VISIBLE_DEVICES explicitly (shared GPUs)')


# ================================================================ features

def features():
    """Replica-set features, knee_lidar.lidar_of's path, saved float32."""
    import torch
    import torch.nn.functional as F
    import yaml
    import knee_lidar as KL
    import knee_lidar_sharedshape as KLS

    print(f'GPU: {torch.cuda.get_device_name(0)} '
          f'(CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")})', flush=True)
    os.makedirs(RFEAT_DIR, exist_ok=True)
    imgs = {k: torch.load(os.path.join(KL.CACHE, f), map_location='cpu', weights_only=True)
            for k, f in SET_FILE.items()}
    digest = {k: KLS.set_digest(v) for k, v in imgs.items()}
    for k in imgs:
        print(f'{k}: {SET_FILE[k]} {tuple(imgs[k].shape)} {imgs[k].dtype} sha256 {digest[k]}',
              flush=True)

    @torch.no_grad()
    def feats_of(enc, x):
        """knee_lidar.lidar_of's feature path verbatim, returning (M*K, D)."""
        out = []
        enc.eval()
        for i in range(0, x.size(0), 64):
            h = enc(x[i:i + 64].to(KL.DEV, torch.float32)).float()
            h = F.layer_norm(h, (h.size(-1),))
            out.append(h.mean(dim=1))
        return torch.cat(out, dim=0)

    P = plan()
    t_all = time.time()
    for s, (full, eps) in P.items():
        out = rfeat_path(full)
        if os.path.exists(out):
            print(f'{s}: exists, skip', flush=True)
            continue
        t0 = time.time()
        cfg = yaml.safe_load(open(os.path.join(KL.drv.ROOT, full, 'config.yml')))
        tag = cfg['logging']['write_tag']
        feats = {k: [] for k in SET_FILE}
        lid = {k: [] for k in SET_FILE}
        for ep in eps:
            ck = os.path.join(KL.drv.ROOT, full, f'{tag}-ep{ep}.pth.tar')
            assert os.path.exists(ck), ck
            enc = KL.drv.encoder_from_ckpt(cfg, ck)
            for k in SET_FILE:
                f = feats_of(enc, imgs[k])
                lid[k].append(KL.lidar(f.reshape(KL.M, KL.K, -1)))
                feats[k].append(f.float().cpu())
            del enc
        obj = {'run': full, 'epochs': list(eps), 'write_tag': tag,
               'sets': dict(SET_FILE), 'set_sha256': digest,
               'script': 'knee_lidar_killtest.py --features'}
        for k in SET_FILE:
            obj[k] = torch.stack(feats[k])                       # (n_ckpt, 512, D) float32
            obj[f'lidar_{k}'] = lid[k]
        assert not os.path.exists(out), f'refusing to overwrite {out}'
        torch.save(obj, out + '.partial')
        os.replace(out + '.partial', out)
        print(f'{s:38s} {len(eps)} ckpts D={obj["redrawn"].shape[-1]} '
              f'({time.time() - t0:.0f}s; total {time.time() - t_all:.0f}s)', flush=True)
    print('DONE features')


# ================================================================ shared pieces

ONSET_SYN = {'21x': 115, '4x': 90}       # campaigns' median knee onsets
SEEDS = (0, 1, 2)
ARMS = ('null', 'energy', 'shape')
ARM_ID = {'null': 0, 'energy': 1, 'shape': 2}
DOSES = (0.0, 0.05, 0.10, 0.18, 0.30)
FACTORS = ('z1', 'z2', 'E')              # ridge_r2 target order (cache y columns)
SHAPE_TOL = 2e-4                         # alternation stop: both shape drops within this of target
XTOL = 1e-6                              # brentq tolerance on the noise scale (raw feature units)
SERIES_COLS = ['run', 'campaign', 'role', 'epoch', 'onset_syn', 'k_since_onset', 'arm', 'seed',
               'grouping', 'dose_target', 'sigma_1', 'sigma_2', 'reachable',
               'r2_z1_base', 'r2_z2_base', 'r2_E_base', 'dR2_z1', 'dR2_z2', 'dR2_E',
               'lidar', 'lambda1', 'q']
DOSE_COLS = ['run', 'campaign', 'epoch', 'arm', 'seed', 'dose_target', 'sigma_1', 'sigma_2',
             'reachable', 'r2_z1_base', 'r2_z2_base', 'r2_E_base', 'dR2_z1', 'dR2_z2', 'dR2_E',
             'lidar_redrawn', 'lidar_shared', 'lambda1_redrawn', 'lambda1_shared',
             'q_redrawn', 'q_shared']
DIR_COLS = ['run', 'campaign', 'epoch', 'k_since_onset',
            'orth_share_E', 'orth_share_z1', 'orth_share_z2',
            'leak_E', 'leak_z1', 'leak_z2', 'cos_uE_uz1', 'cos_uE_uz2', 'cos_uz1_uz2',
            'sd_fit_uE', 'sd_fit_uz1', 'sd_fit_uz2',
            'varw_redrawn_uE', 'varw_redrawn_uz1', 'varw_redrawn_uz2',
            'varw_shared_uE', 'varw_shared_uz1', 'varw_shared_uz2']


def campaign(s):
    return '4x' if '_p384_' in s else '21x'


def role(s):
    import knee_lidar_audit as AUD
    return 'control' if s in AUD.CTRL_21 + AUD.CTRL_4 else 'heldout'


def out_names(onset21):
    """Output paths; a non-default 21x synthetic onset gets its own files."""
    if onset21 == ONSET_SYN['21x']:
        return OUT, OUT_DOSE, OUT.replace('.csv', '_dirs.csv'), PARTS_DIR
    tag = f'_onset21x{onset21}'
    return (OUT.replace('.csv', f'{tag}.csv'), OUT_DOSE.replace('.csv', f'{tag}.csv'),
            OUT.replace('.csv', f'_dirs{tag}.csv'), PARTS_DIR + tag)


def load_readability():
    rd = {}
    with open(READ_CSV) as fh:
        for r in csv.DictReader(fh):
            rd[(short(r['run']), int(r['epoch']))] = (float(r['ridge_r2_z1']),
                                                      float(r['ridge_r2_z2']),
                                                      float(r['ridge_r2_E']))
    return rd


def template(rd=None, factor=2):
    """k -> (median over the 27 knee runs of R2(PRE) - R2(onset + k), n runs),
    PRE = last grid epoch <= onset - 20; factor 2 = E (0 = z1, 1 = z2)."""
    import knee_lidar_audit as AUD
    rd = rd or load_readability()
    knee = dict(AUD.KNEE_21)
    knee.update(AUD.KNEE_4)
    eps_of = defaultdict(list)
    for (r, e) in rd:
        eps_of[r].append(e)
    rel = defaultdict(list)
    for k, o in knee.items():
        eps = sorted(eps_of[k])
        pre = max(e for e in eps if e <= o - 20)
        for e in eps:
            if e >= o - 20:
                rel[e - o].append(rd[(k, pre)][factor] - rd[(k, e)][factor])
    return OrderedDict((k, (median(v), len(v))) for k, v in sorted(rel.items()))


def template_at(T, k):
    ks = [x for x in T if x >= 0]
    assert 0 <= k <= ks[-1], k
    if k in T:
        return T[k][0]
    lo = max(x for x in ks if x < k)
    hi = min(x for x in ks if x > k)
    return T[lo][0] + (T[hi][0] - T[lo][0]) * (k - lo) / (hi - lo)


def directions(Xtr, ytr):
    """Selective noise directions from the fitting split, in the house ridge's
    geometry (wp1_metrics.ridge_r2: z-scored features and targets, alpha = 1).
    Returns raw-space unit directions {'E', 'z1', 'z2'} and diagnostics."""
    import torch
    X, Y = Xtr.double(), ytr.double()
    m, s = X.mean(0), X.std(0) + 1e-8
    Xn = (X - m) / s
    Yn = (Y - Y.mean(0)) / (Y.std(0) + 1e-8)
    D = X.shape[1]
    A = Xn.T @ Xn + torch.eye(D, dtype=X.dtype)
    C = Xn.T @ Yn
    W = torch.linalg.solve(A, C)                       # read-outs, columns z1, z2, E

    def orth(k, others):
        wk, Wo = W[:, k], W[:, others]
        return wk - Wo @ torch.linalg.solve(Wo.T @ A @ Wo, Wo.T @ A @ wk)

    idx = {'z1': 0, 'z2': 1, 'E': 2}
    u, diag = {}, {}
    for name in ('E', 'z1', 'z2'):
        k = idx[name]
        a = orth(k, [j for j in range(3) if j != k])
        v = A @ a
        ur = s * v                                     # standardised -> raw feature units
        u[name] = ur / ur.norm()
        diag[f'orth_share_{name}'] = float(a @ A @ a) / float(W[:, k] @ C[:, k])
        diag[f'leak_{name}'] = (max(abs(float(v @ W[:, j])) for j in range(3) if j != k)
                                / abs(float(v @ W[:, k])))
        diag[f'sd_fit_u{name}'] = float((X @ u[name]).std())
    diag['cos_uE_uz1'] = float(u['E'] @ u['z1'])
    diag['cos_uE_uz2'] = float(u['E'] @ u['z2'])
    diag['cos_uz1_uz2'] = float(u['z1'] @ u['z2'])
    return u, diag


def arm_dirs(arm, u):
    return {'energy': [u['E']], 'shape': [u['z1'], u['z2']]}[arm]


def normals(n, m, *key):
    """Fixed-seed standard normals (float64), keyed by integers."""
    import numpy as np
    import torch
    g = np.random.default_rng(np.random.SeedSequence([int(x) for x in key]))
    return torch.from_numpy(g.standard_normal((n, m)))


def perturb(X64, eps, sig, us):
    out = X64.clone()
    for j, (sg, u) in enumerate(zip(sig, us)):
        if sg:
            out += sg * eps[:, j:j + 1] * u[None, :]
    return out


def drops_fn(Xtr, ytr, Xev, yev, base, etr, eev, us):
    """sigma list -> measured drops [dR2_z1, dR2_z2, dR2_E] (base - perturbed),
    wp1_metrics.ridge_r2 refit on the perturbed fitting split, scored on the
    perturbed evaluation split."""
    from wp1_metrics import ridge_r2

    def f(sig):
        r = ridge_r2(perturb(Xtr, etr, sig, us), ytr, perturb(Xev, eev, sig, us), yev)
        return [b - x for b, x in zip(base, r)]
    return f


def solve1(g, guess=0.0):
    """Root of g(x) = measured drop - target for x >= 0 (g(0) = -target < 0)."""
    from scipy.optimize import brentq
    hi = max(2.0 * guess, 0.25)
    while g(hi) < 0:
        hi *= 4.0
        if hi > 1e3:
            return hi, False
    lo = 0.0
    if guess > 0 and g(0.5 * guess) < 0:
        lo = 0.5 * guess
    return brentq(g, lo, hi, xtol=XTOL), True


def calibrate(arm, d, f):
    """Noise scales so that the target factor's measured drop equals d.
    energy: R2(E); shape: each of R2(z1) and R2(z2). Returns (sig, drops, ok)."""
    n = 1 if arm == 'energy' else 2
    if d == 0:
        return [0.0] * n, [0.0, 0.0, 0.0], True
    if arm == 'energy':
        sg, ok = solve1(lambda x: f([x])[2] - d)
        return [sg], f([sg]), ok
    s1 = s2 = 0.0
    ok = False
    for _ in range(12):
        s1, ok1 = solve1(lambda x: f([x, s2])[0] - d, s1)
        s2, ok2 = solve1(lambda x: f([s1, x])[1] - d, s2)
        dd = f([s1, s2])
        ok = ok1 and ok2 and abs(dd[0] - d) < SHAPE_TOL and abs(dd[1] - d) < SHAPE_TOL
        if ok:
            break
    return [s1, s2], dd, ok


def lid_stats(F512):
    """(LiDAR as knee_lidar.lidar, lambda1, q) of a (512, D) float32 feature
    block on the GPU; lambda1 and q from the whitened spectrum built as
    knee_lidar_spectrum_decomp.term_stats builds it (eigh)."""
    import torch
    import knee_lidar as KL
    import knee_lidar_spectrum_decomp as SD
    F3 = F512.to(KL.DEV, torch.float32).reshape(KL.M, KL.K, -1)
    L = KL.lidar(F3)
    Sb, Sw = SD.mats(F3)
    Wh, _ = SD.whitener(Sw)
    Mw = Wh @ Sb @ Wh
    lam, _ = torch.linalg.eigh((Mw + Mw.T) / 2)
    d = SD.decomp(lam.clamp(min=0))
    return L, d['lambda1'], d['q']


def varw_along(F512, u):
    """Within-group variance of a (512, D) block along unit u (lidar()'s S_w, no delta)."""
    import knee_lidar as KL
    import knee_lidar_spectrum_decomp as SD
    _, Sw = SD.mats(F512.double().reshape(KL.M, KL.K, -1))
    return float(u @ Sw @ u)


def load_run(full, cache_store):
    import torch
    F = torch.load(os.path.join(LFEAT_DIR, f'{full}.pt'), map_location='cpu', weights_only=True)
    if F['cache'] not in cache_store:
        c = torch.load(os.path.join(CACHE, F['cache']), map_location='cpu', weights_only=True)
        cache_store[F['cache']] = (c['y_tr'].double(), c['y_ev'].double())
    R = torch.load(rfeat_path(full), map_location='cpu', weights_only=True)
    assert list(F['epochs']) == list(R['epochs']), full
    return F, cache_store[F['cache']], R


def f6(x):
    return f'{x:.6f}'


# ================================================================ audit helpers

def read_series(path):
    """{short run: (epochs, values)} at N = 100 from a (run, epoch, n_spots, lidar) CSV."""
    import knee_lidar_audit as AUD
    AUD.LID = path
    return AUD.load(N_SPOTS)


def audit_runs(S, thr, onsets):
    """Per run in `onsets`: first alarm per test (knee_lidar_audit_sharedshape.score
    against thresholds `thr`), timely (|first CUSUM alarm - onset| <= 10), and the
    onset-window ratio max CUSUM in [onset - 10, onset + 10] / threshold."""
    import knee_lidar_audit_sharedshape as A2
    import knee_s2_monitors as S2
    sc = A2.score({r: S[r] for r in onsets}, thr)
    fn, on_inc = S2.TESTS['CUSUM']
    out = {}
    for r, o in onsets.items():
        tr = fn(S2.stdz(*S[r], on_inc))
        th = thr[A2.campaign(r)]['CUSUM']
        win = [v for e, v in tr if o - 10 <= e <= o + 10]
        first = sc[r]['CUSUM'][2]
        out[r] = dict(first=first, timely=first is not None and abs(first - o) <= 10,
                      ratio=(max(win) / th) if win else float('nan'),
                      tests={t: sc[r][t][2] for t in A2.TESTS})
    return out


# ================================================================ gate

def gate():
    import torch
    import knee_lidar as KL
    import knee_lidar_audit as AUD
    import knee_lidar_audit_sharedshape as A2
    from wp1_metrics import ridge_r2, wp1_metric_row

    print(f'GPU: {torch.cuda.get_device_name(0)} '
          f'(CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")})', flush=True)
    P = plan()
    csvS = {g: read_series(LID_CSV[g]) for g in GROUPINGS}
    csv_val = {}
    for g in GROUPINGS:
        with open(LID_CSV[g]) as fh:
            for r in csv.DictReader(fh):
                if r['n_spots'] == str(N_SPOTS):
                    csv_val[(g, short(r['run']), int(r['epoch']))] = float(r['lidar'])
    read_rows = {}
    with open(READ_CSV) as fh:
        for r in csv.DictReader(fh):
            read_rows[(short(r['run']), int(r['epoch']))] = r

    # ---- G1 LiDAR from the saved replica features; G2 readability from labelfree_feats
    print('\n== G1: LiDAR recomputed on the GPU from the saved replica features ==')
    d_csv = {g: 0.0 for g in GROUPINGS}
    d_ext = {g: 0.0 for g in GROUPINGS}
    n_ck = 0
    recS = {g: {} for g in GROUPINGS}
    lam = {g: {} for g in GROUPINGS}
    d_read = defaultdict(float)
    d_fn = 0.0
    cache_store = {}
    for s, (full, eps) in P.items():
        F, (ytr, yev), R = load_run(full, cache_store)
        assert list(R['epochs']) == eps, s
        for g in GROUPINGS:
            assert R[g].dtype == torch.float32 and R[g].shape[:2] == (len(eps), 512), (s, g)
            vals = []
            for j, ep in enumerate(eps):
                L, l1, _ = lid_stats(R[g][j])
                vals.append(L)
                lam[g][(s, ep)] = l1
                d_csv[g] = max(d_csv[g], abs(L - csv_val[(g, s, ep)]))
                d_ext[g] = max(d_ext[g], abs(L - R[f'lidar_{g}'][j]))
            recS[g][s] = (list(eps), vals)
        for j, ep in enumerate(eps):
            m = wp1_metric_row(F['feats_tr'][j], ytr, F['feats_ev'][j], yev)
            rr = read_rows[(s, ep)]
            for col in ('ridge_r2_z1', 'ridge_r2_z2', 'ridge_r2_E', 'mean_cca'):
                d_read[col] = max(d_read[col], abs(m[col] - float(rr[col])))
            r2 = ridge_r2(F['feats_tr'][j], ytr, F['feats_ev'][j], yev)
            d_fn = max(d_fn, max(abs(a - m[f'ridge_r2_{n}']) for a, n in zip(r2, FACTORS)))
        n_ck += len(eps)
    for g in GROUPINGS:
        print(f'  {g:8s} {n_ck} checkpoints (19 runs): max |recomputed - {os.path.basename(LID_CSV[g])}| '
              f'= {d_csv[g]:.2e}; max |recomputed - value stored at extraction| = {d_ext[g]:.2e}')
    print(f'\n== G2: readability recomputed with wp1_metrics.wp1_metric_row from labelfree_feats '
          f'({n_ck} checkpoints) against {os.path.basename(READ_CSV)} ==')
    for col, v in d_read.items():
        print(f'  {col:12s} max |d| = {v:.2e}')
    print(f'  wp1_metrics.ridge_r2 (used in the calibration loop) vs wp1_metric_row ridge values: '
          f'max |d| = {d_fn:.2e}')

    # ---- G3 thresholds and the audit code
    print('\n== G3: standing thresholds and the audit code ==')
    for g in GROUPINGS:
        t_csv, t_rec = A2.thresholds(csvS[g]), A2.thresholds(recS[g])
        print(f'  {g:8s} CUSUM thresholds from the CSV series 21x {t_csv["21x"]["CUSUM"]:.4f} '
              f'4x {t_csv["4x"]["CUSUM"]:.4f}; from the recomputed null series 21x '
              f'{t_rec["21x"]["CUSUM"]:.4f} 4x {t_rec["4x"]["CUSUM"]:.4f}; max |diff| over the '
              f'four tests {max(abs(t_csv[c][t] - t_rec[c][t]) for c in t_csv for t in A2.TESTS):.2e}')
    calib = {}
    with open(CALIB_CSV) as fh:
        for r in csv.DictReader(fh):
            if r['calibration'] == 'A':
                calib[(r['protocol'], r['campaign'])] = r
    for g, proto in (('redrawn', 'existing'), ('shared', 'shared')):
        thr = A2.thresholds(csvS[g])
        for camp, knee in (('21x', AUD.KNEE_21), ('4x', AUD.KNEE_4)):
            res = audit_runs(csvS[g], thr, knee)
            n_al = sum(v['first'] is not None for v in res.values())
            n_ti = sum(v['timely'] for v in res.values())
            rat = [v['ratio'] for v in res.values()]
            c = calib[(proto, camp)]
            print(f'  real knee runs, {g:8s} {camp:3s}: CUSUM alarms {n_al}/{len(res)} (record '
                  f'{c["knee_alarms"]}), timely {n_ti}/{len(res)} (record {c["near_onset"]}); '
                  f'onset-window ratio median {median(rat):.3f} max {max(rat):.3f} (record '
                  f'{c["onset_window_ratio_median"]}, {c["onset_window_ratio_max"]})')

    # ---- G4 lambda1 against the decomposition CSV
    print('\n== G4: lambda1 (whitened spectrum, eigh path) against knee_lidar_spectrum_decomp.csv ==')
    worst = {g: 0.0 for g in GROUPINGS}
    n4 = {g: 0 for g in GROUPINGS}
    with open(DECOMP_CSV) as fh:
        for r in csv.DictReader(fh):
            if r['variant'] != 'raw' or r['role'] == 'knee':
                continue
            for g in GROUPINGS:
                if r['term'] == f'{g}|{g}':
                    key = (short(r['run']), int(r['epoch']))
                    worst[g] = max(worst[g], abs(lam[g][key] / float(r['lambda1']) - 1))
                    n4[g] += 1
    for g in GROUPINGS:
        print(f'  {g:8s} {n4[g]} event-free checkpoints: max relative |d| = {worst[g]:.2e}')

    # ---- G5 invariance to rescaling one direction
    print('\n== G5: rescaling one feature direction by 0.1 and by 10 (an invertible linear map) ==')
    print('  per control, final checkpoint. Directions: the energy direction u_E and the shape '
          'directions u_z1, u_z2 of the design (fitting split), and a random unit direction '
          '(orthogonal to the all-ones vector, which the LayerNorm-pooled features have no variance along).')
    gen = torch.Generator().manual_seed(20260924)
    for s in AUD.CTRL_21 + AUD.CTRL_4:
        full, eps = P[s]
        F, (ytr, yev), R = load_run(full, cache_store)
        j = len(eps) - 1
        Xtr, Xev = F['feats_tr'][j].double(), F['feats_ev'][j].double()
        base = ridge_r2(Xtr, ytr, Xev, yev)
        u, _ = directions(Xtr, ytr)
        rnd = torch.randn(Xtr.shape[1], generator=gen, dtype=torch.float64)
        rnd = rnd - rnd.mean()
        u['random'] = rnd / rnd.norm()
        L0 = {g: lid_stats(R[g][j]) for g in GROUPINGS}
        print(f'  {s.split("p1s100_")[1]} ep{eps[j]}: unperturbed R2 z1/z2/E '
              f'{base[0]:.4f}/{base[1]:.4f}/{base[2]:.4f}; LiDAR redrawn {L0["redrawn"][0]:.4f} '
              f'(lambda1 {L0["redrawn"][1]:.2f}), shared {L0["shared"][0]:.4f} (lambda1 {L0["shared"][1]:.2f})')
        for name in ('E', 'z1', 'z2', 'random'):
            vw = {g: varw_along(R[g][j], u[name]) for g in GROUPINGS}
            for c in (0.1, 10.0):
                M = torch.eye(Xtr.shape[1], dtype=torch.float64) - (1 - c) * torch.outer(u[name], u[name])
                r2 = ridge_r2(Xtr @ M, ytr, Xev @ M, yev)
                L = {g: lid_stats((R[g][j].double() @ M).float()) for g in GROUPINGS}
                print(f'    u_{name:6s} x{c:<4g}: dR2 z1 {r2[0] - base[0]:+.5f} z2 {r2[1] - base[1]:+.5f} '
                      f'E {r2[2] - base[2]:+.5f} | LiDAR redrawn {L["redrawn"][0]:.4f} '
                      f'({L["redrawn"][0] - L0["redrawn"][0]:+.4f}; lambda1 {L["redrawn"][1]:.2f}), shared '
                      f'{L["shared"][0]:.4f} ({L["shared"][0] - L0["shared"][0]:+.4f}; lambda1 '
                      f'{L["shared"][1]:.2f}) | within-group var along u before rescaling: redrawn '
                      f'{vw["redrawn"]:.2e}, shared {vw["shared"]:.2e} (delta = 1e-4)')

    # ---- G6 is the rescaling response carried by the regulariser delta?
    print('\n== G6: LiDAR (knee_lidar.lidar, float64 here) before and after rescaling u_E by 0.1, at '
          'delta = 1e-4 (the standing value), 1e-6 and 1e-8. For an invertible map M the generalised '
          'eigenvalues of (M Sb M, M Sw M) equal those of (Sb, Sw), so without delta the map changes nothing. ==')
    for s in AUD.CTRL_21 + AUD.CTRL_4:
        full, eps = P[s]
        F, (ytr, yev), R = load_run(full, cache_store)
        j = len(eps) - 1
        u, _ = directions(F['feats_tr'][j].double(), ytr)
        M = torch.eye(u['E'].shape[0], dtype=torch.float64) - 0.9 * torch.outer(u['E'], u['E'])
        cells = []
        for g in GROUPINGS:
            X = R[g][j].double().to(KL.DEV)
            Xm = X @ M.to(KL.DEV)
            for dl in (1e-4, 1e-6, 1e-8):
                a = KL.lidar(X.reshape(KL.M, KL.K, -1), delta=dl)
                b = KL.lidar(Xm.reshape(KL.M, KL.K, -1), delta=dl)
                cells.append(f'{g} d={dl:.0e}: {a:.4f} -> {b:.4f} ({b - a:+.4f})')
        print(f'  {s.split("p1s100_")[1]} ep{eps[j]}: ' + '; '.join(cells))
    print('\nGATE DONE')


# ================================================================ run

def run(onset21):
    import torch
    import knee_lidar_audit as AUD
    from wp1_metrics import ridge_r2

    onset_syn = {'21x': onset21, '4x': ONSET_SYN['4x']}
    out_s, out_d, out_dir, parts = out_names(onset21)
    for p in (out_s, out_d, out_dir):
        assert not os.path.exists(p), f'refusing to overwrite {p}'
    os.makedirs(parts, exist_ok=True)
    print(f'GPU: {torch.cuda.get_device_name(0)} '
          f'(CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")}); synthetic onsets '
          f'{onset_syn}; parts in {parts}', flush=True)
    T = template()
    print('template (median R2(E) drop from PRE vs epochs since onset, n runs): ' +
          ' '.join(f'{k:+d}:{v:.4f}({n})' for k, (v, n) in T.items()), flush=True)
    P = plan()
    cache_store = {}
    t_all = time.time()
    for ri, (s, (full, eps)) in enumerate(P.items()):
        pp = {k: os.path.join(parts, f'{full}.{k}.csv') for k in ('series', 'dose', 'dirs')}
        if all(os.path.exists(p) for p in pp.values()):
            print(f'{s}: parts exist, skip', flush=True)
            continue
        t0 = time.time()
        F, (ytr, yev), R = load_run(full, cache_store)
        camp, rl = campaign(s), role(s)
        onset = onset_syn[camp]
        rows, drows, dirrows = [], [], []
        n_unreach = 0
        for j, ep in enumerate(eps):
            Xtr, Xev = F['feats_tr'][j].double(), F['feats_ev'][j].double()
            base = ridge_r2(Xtr, ytr, Xev, yev)
            null = {g: lid_stats(R[g][j]) for g in GROUPINGS}
            post = ep >= onset
            k = ep - onset if post else ''
            common = dict(run=full, campaign=camp, role=rl, epoch=ep, onset_syn=onset,
                          k_since_onset=k, r2_z1_base=f6(base[0]), r2_z2_base=f6(base[1]),
                          r2_E_base=f6(base[2]))

            def emit(arm, seed, dose, sig, dd, ok, stats):
                for g in GROUPINGS:
                    L, l1, q = stats[g]
                    rows.append(dict(common, arm=arm, seed=seed, grouping=g, dose_target=f6(dose),
                                     sigma_1=f'{sig[0]:.6e}' if sig else '',
                                     sigma_2=f'{sig[1]:.6e}' if len(sig) > 1 else '',
                                     reachable=int(ok), dR2_z1=f'{dd[0]:.6f}', dR2_z2=f'{dd[1]:.6f}',
                                     dR2_E=f'{dd[2]:.6f}', lidar=f6(L), lambda1=f'{l1:.6e}',
                                     q=f6(q)))
            emit('null', 0, 0.0, [], [0.0, 0.0, 0.0], True, null)
            if post:
                d = template_at(T, k)
                u, diag = directions(Xtr, ytr)
                vw = {(g, n): varw_along(R[g][j], u[n]) for g in GROUPINGS for n in ('E', 'z1', 'z2')}
                dirrows.append(dict(run=full, campaign=camp, epoch=ep, k_since_onset=k,
                                    **{c: f'{diag[c]:.6e}' for c in DIR_COLS if c in diag},
                                    **{f'varw_{g}_u{n}': f'{vw[(g, n)]:.6e}' for (g, n) in vw}))
            for arm in ('energy', 'shape'):
                for seed in SEEDS:
                    if not post:
                        emit(arm, seed, 0.0, [0.0] * (1 if arm == 'energy' else 2),
                             [0.0, 0.0, 0.0], True, null)
                        continue
                    key = (seed, ri, ep, ARM_ID[arm])
                    etr = normals(Xtr.shape[0], 2, *key, 0)
                    eev = normals(Xev.shape[0], 2, *key, 1)
                    us = arm_dirs(arm, u)
                    f = drops_fn(Xtr, ytr, Xev, yev, base, etr, eev, us)
                    sig, dd, ok = calibrate(arm, d, f)
                    n_unreach += not ok
                    stats = {}
                    for gi, g in enumerate(GROUPINGS):
                        er = normals(512, 2, *key, 2 + gi)
                        stats[g] = lid_stats(perturb(R[g][j].double(), er, sig, us).float())
                    emit(arm, seed, d, sig, dd, ok, stats)
        # dose-response: final checkpoint of each control
        if rl == 'control':
            j = len(eps) - 1
            ep = eps[j]
            Xtr, Xev = F['feats_tr'][j].double(), F['feats_ev'][j].double()
            base = ridge_r2(Xtr, ytr, Xev, yev)
            u, _ = directions(Xtr, ytr)
            for arm in ('energy', 'shape'):
                us = arm_dirs(arm, u)
                for seed in SEEDS:
                    key = (seed, ri, ep, 10 + ARM_ID[arm])
                    etr = normals(Xtr.shape[0], 2, *key, 0)
                    eev = normals(Xev.shape[0], 2, *key, 1)
                    er = {g: normals(512, 2, *key, 2 + gi) for gi, g in enumerate(GROUPINGS)}
                    f = drops_fn(Xtr, ytr, Xev, yev, base, etr, eev, us)
                    for d in DOSES:
                        sig, dd, ok = calibrate(arm, d, f)
                        n_unreach += not ok
                        st = {g: (lid_stats(R[g][j]) if d == 0 else
                                  lid_stats(perturb(R[g][j].double(), er[g], sig, us).float()))
                              for g in GROUPINGS}
                        drows.append(dict(run=full, campaign=camp, epoch=ep, arm=arm, seed=seed,
                                          dose_target=f6(d), sigma_1=f'{sig[0]:.6e}',
                                          sigma_2=f'{sig[1]:.6e}' if len(sig) > 1 else '',
                                          reachable=int(ok), r2_z1_base=f6(base[0]),
                                          r2_z2_base=f6(base[1]), r2_E_base=f6(base[2]),
                                          dR2_z1=f'{dd[0]:.6f}', dR2_z2=f'{dd[1]:.6f}',
                                          dR2_E=f'{dd[2]:.6f}',
                                          **{f'{c}_{g}': (f6(st[g][i]) if c != 'lambda1'
                                                          else f'{st[g][i]:.6e}')
                                             for g in GROUPINGS
                                             for i, c in enumerate(('lidar', 'lambda1', 'q'))}))
        for kind, rr, cols in (('series', rows, SERIES_COLS), ('dose', drows, DOSE_COLS),
                               ('dirs', dirrows, DIR_COLS)):
            p = pp[kind]
            assert not os.path.exists(p), f'refusing to overwrite {p}'
            with open(p + '.partial', 'w', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=cols, lineterminator='\n')
                w.writeheader()
                w.writerows(rr)
            os.replace(p + '.partial', p)
        print(f'{s:38s} {rl:8s} {len(eps)} ckpts, {len(rows)} series rows, {len(drows)} dose rows, '
              f'unreachable/unconverged {n_unreach} ({time.time() - t0:.0f}s; total '
              f'{time.time() - t_all:.0f}s)', flush=True)
    # merge
    for kind, dst, cols in (('series', out_s, SERIES_COLS), ('dose', out_d, DOSE_COLS),
                            ('dirs', out_dir, DIR_COLS)):
        allrows = []
        for s, (full, _) in P.items():
            with open(os.path.join(parts, f'{full}.{kind}.csv')) as fh:
                allrows += list(csv.DictReader(fh))
        assert not os.path.exists(dst), f'refusing to overwrite {dst}'
        with open(dst + '.partial', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=cols, lineterminator='\n')
            w.writeheader()
            w.writerows(allrows)
        os.replace(dst + '.partial', dst)
        print(f'wrote {dst}: {len(allrows)} rows', flush=True)
    print('DONE run')


# ================================================================ report

def pct(v, p):
    s = sorted(v)
    if not s:
        return float('nan')
    i = min(len(s) - 1, max(0, int(math.ceil(p / 100 * len(s))) - 1))
    return s[i]


def report(onset21):
    import knee_lidar_audit as AUD
    import knee_lidar_audit_sharedshape as A2
    import knee_s2_monitors as S2

    onset_syn = {'21x': onset21, '4x': ONSET_SYN['4x']}
    out_s, out_d, out_dir, _ = out_names(onset21)
    rows = list(csv.DictReader(open(out_s)))
    drows = list(csv.DictReader(open(out_d)))
    dirrows = list(csv.DictReader(open(out_dir)))
    fl = lambda r, k: float(r[k])                                   # noqa: E731
    runs = list(plan())
    onsets = {s: onset_syn[campaign(s)] for s in runs}
    rd = load_readability()
    print(f'{out_s}: {len(rows)} rows; {out_d}: {len(drows)} rows; {out_dir}: {len(dirrows)} rows')
    print(f'synthetic onsets {onset_syn}')

    # ---------------------------------------------------------- template and context
    print('\n## Template: median over the 27 knee runs of R2(PRE) - R2(onset + k) (n runs)')
    Ts = {n: template(rd, i) for i, n in enumerate(FACTORS)}
    print('| k | ' + ' | '.join(f'R2({n}) drop' for n in ('E', 'z1', 'z2')) + ' | n |')
    print('|---|---|---|---|---|')
    for k in Ts['E']:
        print(f'| {k:+d} | {Ts["E"][k][0]:.4f} | {Ts["z1"][k][0]:.4f} | {Ts["z2"][k][0]:.4f} | {Ts["E"][k][1]} |')
    knee = dict(AUD.KNEE_21)
    knee.update(AUD.KNEE_4)
    eps_of = defaultdict(list)
    for (r, e) in rd:
        eps_of[r].append(e)
    print('\nReal event, 27 knee runs, R2 at PRE (last grid epoch <= onset - 20) and POST (final), medians:')
    for i, n in enumerate(FACTORS):
        pre, post = [], []
        for k, o in knee.items():
            eps = sorted(eps_of[k])
            p = max(e for e in eps if e <= o - 20)
            pre.append(rd[(k, p)][i])
            post.append(rd[(k, max(eps))][i])
        print(f'  R2({n}): PRE {median(pre):.3f}, POST {median(post):.3f}, drop median '
              f'{median(a - b for a, b in zip(pre, post)):.3f} (range '
              f'{min(a - b for a, b in zip(pre, post)):.3f} to {max(a - b for a, b in zip(pre, post)):.3f})')

    print('\n## Checkpoint-to-checkpoint variation of R2 in the 19 event-free runs (unperturbed)')
    for lo in (30, 90):
        cells = []
        for i, n in enumerate(FACTORS):
            d1, d2 = [], []
            for s in runs:
                eps = sorted(e for e in eps_of[s] if e >= lo)
                for a, b in zip(eps, eps[1:]):
                    if b - a == 10:
                        d1.append(abs(rd[(s, b)][i] - rd[(s, a)][i]))
                for a, b, c in zip(eps, eps[1:], eps[2:]):
                    if b - a == 10 and c - b == 10:
                        d2.append(abs(rd[(s, b)][i] - 0.5 * (rd[(s, a)][i] + rd[(s, c)][i])))
            cells.append(f'R2({n}): |step| median {median(d1):.4f}, 90th pct {pct(d1, 90):.4f}, max '
                         f'{max(d1):.4f} (n {len(d1)}); |deviation from neighbours\' mean| median '
                         f'{median(d2):.4f}, 90th pct {pct(d2, 90):.4f}')
        print(f'  epochs >= {lo}: ' + '; '.join(cells))

    post_keys = sorted({(short(r['run']), int(r['epoch'])) for r in rows if r['k_since_onset'] != ''})
    rz1 = [rd[k][0] / rd[k][2] for k in post_keys]
    rz2 = [rd[k][1] / rd[k][2] for k in post_keys]
    print(f'\n## Levels at the {len(post_keys)} post-onset checkpoints: R2(z1)/R2(E) median {median(rz1):.4f} '
          f'(range {min(rz1):.4f}-{max(rz1):.4f}); R2(z2)/R2(E) median {median(rz2):.4f} '
          f'(range {min(rz2):.4f}-{max(rz2):.4f}). A shape arm matched on the fraction of explained '
          f'variance lost would use drop x R2(z)/R2(E): at a drop of 0.18 that differs from the '
          f'absolute-drop arm by median {0.18 * (1 - median(rz1)):.4f} (z1) and {0.18 * (1 - median(rz2)):.4f} '
          f'(z2), at most {0.18 * (1 - min(rz2)):.4f}.')

    # ---------------------------------------------------------- directions
    print('\n## Direction diagnostics (post-onset checkpoints, fitting split)')
    for c in DIR_COLS[4:]:
        v = [fl(r, c) for r in dirrows]
        print(f'  {c:18s} median {median(v):+.4e}  min {min(v):+.4e}  max {max(v):+.4e}')

    # ---------------------------------------------------------- selectivity
    print('\n## Selectivity in the synthetic-event series (post-onset rows, 3 seeds; one row per '
          'checkpoint and seed, the redrawn copy)')
    post = [r for r in rows if r['k_since_onset'] != '' and r['grouping'] == 'redrawn']
    for arm, tgt, off in (('energy', ['dR2_E'], ['dR2_z1', 'dR2_z2']),
                          ('shape', ['dR2_z1', 'dR2_z2'], ['dR2_E'])):
        sel = [r for r in post if r['arm'] == arm]
        err = [abs(fl(r, t) - fl(r, 'dose_target')) for r in sel for t in tgt]
        nre = sum(r['reachable'] == '0' for r in sel)
        cells = [f'{arm}: {len(sel)} rows, target doses {min(fl(r, "dose_target") for r in sel):.4f}'
                 f'-{max(fl(r, "dose_target") for r in sel):.4f}; max |achieved - target| '
                 f'{max(err):.1e}; unreachable/unconverged {nre}']
        for c in off:
            v = [fl(r, c) for r in sel]
            cells.append(f'{c}: median {median(v):+.5f}, 95th pct of |.| {pct([abs(x) for x in v], 95):.5f}, '
                         f'max |.| {max(abs(x) for x in v):.5f}')
        print('  ' + '; '.join(cells))

    # ---------------------------------------------------------- the audit
    series = defaultdict(lambda: defaultdict(dict))        # (arm, seed, grouping) -> run -> ep -> v
    lam1 = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        key = (r['arm'], int(r['seed']), r['grouping'])
        series[key][short(r['run'])][int(r['epoch'])] = fl(r, 'lidar')
        lam1[key][short(r['run'])][int(r['epoch'])] = fl(r, 'lambda1')

    def S_of(key):
        return {k: (sorted(v), [v[e] for e in sorted(v)]) for k, v in series[key].items()}

    print('\n## LiDAR in each arm minus the null arm at the same checkpoint (post-onset rows, 3 seeds), '
          'and the null series\' own checkpoint-to-checkpoint steps for scale')
    for g in GROUPINGS:
        steps = []
        for s in runs:
            ser = series[('null', 0, g)][s]
            eps = sorted(e for e in ser if e >= 90)
            steps += [abs(ser[b] - ser[a]) for a, b in zip(eps, eps[1:]) if b - a == 10]
        print(f'  null {g:8s}: |LiDAR step| between neighbouring checkpoints (epochs >= 90), median '
              f'{median(steps):.4f}, 90th pct {pct(steps, 90):.4f}, max {max(steps):.4f} (n {len(steps)})')
        for arm in ('energy', 'shape'):
            dd, rel = [], []
            for r in rows:
                if r['arm'] == arm and r['grouping'] == g and r['k_since_onset'] != '':
                    n0 = series[('null', 0, g)][short(r['run'])][int(r['epoch'])]
                    dd.append(fl(r, 'lidar') - n0)
                    rel.append(fl(r, 'lidar') / n0 - 1)
            print(f'  {arm:6s} {g:8s}: LiDAR - null median {median(dd):+.4f} (min {min(dd):+.4f}, max '
                  f'{max(dd):+.4f}); relative median {median(rel):+.4f} (min {min(rel):+.4f}, max '
                  f'{max(rel):+.4f}); n {len(dd)}')

    thr = {g: A2.thresholds(S_of(('null', 0, g))) for g in GROUPINGS}
    print('\n## Synthetic-event audit (CUSUM, standing thresholds from the unperturbed controls)')
    for g in GROUPINGS:
        print(f'  thresholds {g}: 21x {thr[g]["21x"]["CUSUM"]:.4f}, 4x {thr[g]["4x"]["CUSUM"]:.4f}')
    pre_ep = {}
    for s in runs:
        eps = sorted(series[('null', 0, 'redrawn')][s])
        pre_ep[s] = (max(e for e in eps if e <= onsets[s] - 20), max(eps))
    print(f'  PRE/POST epochs: 21x {sorted({v for s, v in pre_ep.items() if campaign(s) == "21x"})}, '
          f'4x {sorted({v for s, v in pre_ep.items() if campaign(s) == "4x"})}')
    audit_res = {}
    print('\n| arm | grouping | seed | timely first alarms (21x + 4x) | any alarm | alarm - onset '
          '(first alarms) | onset-window ratio median / max | LiDAR PRE -> POST: median change '
          '(range), rises | lambda1 PRE -> POST: median PRE, median POST, falls, median POST/PRE |')
    print('|---|---|---|---|---|---|---|---|---|')
    for arm in ARMS:
        for g in GROUPINGS:
            for seed in (SEEDS if arm != 'null' else (0,)):
                key = (arm, seed, g)
                S = S_of(key)
                res = audit_runs(S, thr[g], onsets)
                audit_res[key] = res
                t21 = sum(res[s]['timely'] for s in runs if campaign(s) == '21x')
                t4 = sum(res[s]['timely'] for s in runs if campaign(s) == '4x')
                anyn = sum(res[s]['first'] is not None for s in runs)
                al = sorted(res[s]['first'] - onsets[s] for s in runs if res[s]['first'] is not None)
                rat = [res[s]['ratio'] for s in runs]
                dl = [series[key][s][pre_ep[s][1]] - series[key][s][pre_ep[s][0]] for s in runs]
                l_pre = [lam1[key][s][pre_ep[s][0]] for s in runs]
                l_post = [lam1[key][s][pre_ep[s][1]] for s in runs]
                print(f'| {arm} | {g} | {seed} | {t21 + t4} / 19 ({t21} + {t4}) | {anyn} / 19 | '
                      f'{" ".join(f"{x:+d}" for x in al) or "-"} | {median(rat):.3f} / {max(rat):.3f} | '
                      f'{median(dl):+.3f} ({min(dl):+.3f} to {max(dl):+.3f}), {sum(x > 0 for x in dl)}/19 | '
                      f'{median(l_pre):.2f}, {median(l_post):.2f}, {sum(b < a for a, b in zip(l_pre, l_post))}/19, '
                      f'{median(b / a for a, b in zip(l_pre, l_post)):.3f} |')

    print('\n### By campaign: timely first alarms, any alarm, onset-window ratio median (seeds 0, 1, 2)')
    print('| arm | grouping | 21x (11 runs, onset %d) | 4x (8 runs, onset %d) |' % (onset_syn['21x'],
                                                                             onset_syn['4x']))
    print('|---|---|---|---|')
    for arm in ARMS:
        for g in GROUPINGS:
            cells = []
            for camp in ('21x', '4x'):
                rr = [s for s in runs if campaign(s) == camp]
                per = []
                for seed in (SEEDS if arm != 'null' else (0,)):
                    res = audit_res[(arm, seed, g)]
                    per.append(f'{sum(res[s]["timely"] for s in rr)}/{len(rr)}, '
                               f'{sum(res[s]["first"] is not None for s in rr)}/{len(rr)}, '
                               f'{median(res[s]["ratio"] for s in rr):.3f}')
                cells.append('; '.join(per))
            print(f'| {arm} | {g} | ' + ' | '.join(cells) + ' |')

    print('\n### The other three tests: timely first alarms / any alarm, out of 19 (seeds 0, 1, 2)')
    print('| arm | grouping | ' + ' | '.join(t for t in S2.TESTS if t != 'CUSUM') + ' |')
    print('|---|---|---|---|---|')
    for arm in ARMS:
        for g in GROUPINGS:
            cells = []
            for t in S2.TESTS:
                if t == 'CUSUM':
                    continue
                per = []
                for seed in (SEEDS if arm != 'null' else (0,)):
                    res = audit_res[(arm, seed, g)]
                    fa = [res[s]['tests'][t] for s in runs]
                    ti = sum(a is not None and abs(a - onsets[s]) <= 10 for a, s in zip(fa, runs))
                    an = sum(a is not None for a in fa)
                    per.append(f'{ti}/{an}')
                cells.append(', '.join(per))
            print(f'| {arm} | {g} | ' + ' | '.join(cells) + ' |')

    print('\n### Per run, CUSUM first alarm epoch (seed 0) and onset-window ratio')
    print('| run | campaign | onset | ' + ' | '.join(f'{arm} {g}' for arm in ARMS for g in GROUPINGS) + ' |')
    print('|---|---|---|' + '---|' * 6)
    for s in runs:
        cells = []
        for arm in ARMS:
            for g in GROUPINGS:
                res = audit_res[(arm, 0, g)][s]
                cells.append(f'{("@" + str(res["first"])) if res["first"] is not None else "-"} '
                             f'({res["ratio"]:.2f})')
        print(f'| {s.split("p1s100_")[1]} | {campaign(s)} | {onsets[s]} | ' + ' | '.join(cells) + ' |')

    # ---------------------------------------------------------- real event reference
    print('\n## Real event reference (27 knee runs): LiDAR PRE -> POST from the CSV series, lambda1 from '
          'knee_lidar_spectrum_decomp.csv')
    for g in GROUPINGS:
        S = read_series(LID_CSV[g])
        dl = []
        for k, o in knee.items():
            eps, xs = S[k]
            d = dict(zip(eps, xs))
            p = max(e for e in eps if e <= o - 20)
            dl.append(d[max(eps)] - d[p])
        lp, lq = {}, {}
        with open(DECOMP_CSV) as fh:
            for r in csv.DictReader(fh):
                if r['role'] == 'knee' and r['variant'] == 'raw' and r['term'] == f'{g}|{g}':
                    if r['phase'] == 'pre':
                        lp[r['run']] = fl(r, 'lambda1')
                    elif r['phase'] == 'post':
                        lq[r['run']] = fl(r, 'lambda1')
        ks = sorted(lp)
        print(f'  {g:8s}: LiDAR change median {median(dl):+.3f} ({min(dl):+.3f} to {max(dl):+.3f}), '
              f'rises {sum(x > 0 for x in dl)}/27; lambda1 PRE median {median(lp[k] for k in ks):.2f}, POST '
              f'median {median(lq[k] for k in ks):.2f}, falls {sum(lq[k] < lp[k] for k in ks)}/27, '
              f'median POST/PRE {median(lq[k] / lp[k] for k in ks):.3f}')

    # ---------------------------------------------------------- dose-response
    print('\n## Dose-response at the final checkpoint of each control (8 controls x 3 seeds; medians, '
          'with min-max over the 24)')
    print('| arm | target drop | achieved dR2 z1 / z2 / E (median) | max |off-target dR2| | LiDAR redrawn | '
          'LiDAR shared | lambda1 redrawn | lambda1 shared | unreachable |')
    print('|---|---|---|---|---|---|---|---|---|')
    for arm in ('energy', 'shape'):
        off = ['dR2_z1', 'dR2_z2'] if arm == 'energy' else ['dR2_E']
        for d in DOSES:
            sel = [r for r in drows if r['arm'] == arm and abs(fl(r, 'dose_target') - d) < 1e-9]
            mm = lambda c: (f'{median(fl(r, c) for r in sel):.3f} ({min(fl(r, c) for r in sel):.3f}-'   # noqa: E731
                            f'{max(fl(r, c) for r in sel):.3f})')
            print(f'| {arm} | {d:.2f} | {median(fl(r, "dR2_z1") for r in sel):.4f} / '
                  f'{median(fl(r, "dR2_z2") for r in sel):.4f} / {median(fl(r, "dR2_E") for r in sel):.4f} | '
                  f'{max(abs(fl(r, c)) for r in sel for c in off):.5f} | {mm("lidar_redrawn")} | '
                  f'{mm("lidar_shared")} | {mm("lambda1_redrawn")} | {mm("lambda1_shared")} | '
                  f'{sum(r["reachable"] == "0" for r in sel)} |')
    print('\n### Dose-response per control (seed-median LiDAR redrawn / shared)')
    print('| control | arm | ' + ' | '.join(f'{d:.2f}' for d in DOSES) + ' |')
    print('|---|---|' + '---|' * len(DOSES))
    for s in AUD.CTRL_21 + AUD.CTRL_4:
        for arm in ('energy', 'shape'):
            cells = []
            for d in DOSES:
                sel = [r for r in drows if short(r['run']) == s and r['arm'] == arm
                       and abs(fl(r, 'dose_target') - d) < 1e-9]
                cells.append(f'{median(fl(r, "lidar_redrawn") for r in sel):.2f} / '
                             f'{median(fl(r, "lidar_shared") for r in sel):.2f}')
            print(f'| {s.split("p1s100_")[1]} | {arm} | ' + ' | '.join(cells) + ' |')

    # ---------------------------------------------------------- seed spread
    print('\n## Noise-seed spread (series): per run, max - min over the 3 seeds')
    for arm in ('energy', 'shape'):
        for g in GROUPINGS:
            rng_post, rng_peak = [], []
            for s in runs:
                vals = [series[(arm, sd, g)][s][pre_ep[s][1]] for sd in SEEDS]
                rng_post.append(max(vals) - min(vals))
                pk = []
                for sd in SEEDS:
                    res = audit_res[(arm, sd, g)][s]
                    pk.append(res['ratio'])
                rng_peak.append(max(pk) - min(pk))
            cnt = [sum(audit_res[(arm, sd, g)][s]['timely'] for s in runs) for sd in SEEDS]
            print(f'  {arm:6s} {g:8s}: LiDAR at POST, seed range median {median(rng_post):.4f}, max '
                  f'{max(rng_post):.4f}; onset-window ratio seed range median {median(rng_peak):.4f}, max '
                  f'{max(rng_peak):.4f}; timely first alarms per seed {cnt}')


# ================================================================== main

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', action='store_true')
    ap.add_argument('--gate', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--onset21', type=int, default=ONSET_SYN['21x'],
                    help='21x synthetic onset (default 115, the campaign median)')
    a = ap.parse_args()
    if a.features:
        need_gpu()
        features()
    elif a.gate:
        need_gpu()
        gate()
    elif a.run:
        need_gpu()
        run(a.onset21)
    elif a.report:
        report(a.onset21)
    else:
        ap.print_help()
