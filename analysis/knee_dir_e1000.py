"""Direction battery on the six 1000-epoch mu=0.9 falls (3 cosine,
3 constant-LR). Definitions printed in the record before this ran
(2026-08-24).

Construction IMPORTED from knee_t2_direction (ridge_resid, feats, W_KNEE)
rather than reimplemented, so these rows are produced by exactly the
machinery the T2 battery fixed in advance used. Adds two columns that
script does not compute: lambda share and PRE contamination.

NO READINGS EMITTED -- raw values only, per the binding no-labels rule.

STANDING DISCRIMINATOR (agreed 2026-08-24). cosine s1 turned out to hold
TWO falls with different directions: event A canonical (|cos| 0.997),
event B nearly orthogonal (|cos| 0.070) and NOT noise (lam 88%,
dR2 0.96). n=1, and "real second mechanism" vs "degenerated run" is
undistinguished. The named discriminator is the same two-window
decomposition on any other run that falls to comparable depth. Run it
with, e.g.:

    python knee_dir_e1000.py --run rigb_lejp_p1s100_p384_ou09_e1000_1xxx \\
        --cache rig_P1_Enar_S100.pt --windows 70:300,300:1000

--windows takes explicit PRE:POST pairs and bypasses the R_A anchor, so
a two-event run can be split without waiting on a rule that fires on
whichever event happens to come first.
"""
import argparse
import csv
import glob
import os
import sys

import torch
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
sys.path.insert(0, _HERE)
import wp1_probe_driver as drv  # noqa: E402
from knee_s1_onsets import onset_A  # noqa: E402
from knee_t2_direction import W_KNEE, feats, ridge_resid  # noqa: E402

CACHE = paths.CACHE
TRAJ = os.path.join(paths.CSV, 'knee_trajectories.csv')
CACHE_NAME = 'rigd_p1s100_mu09.pt'
RUNS = [(f'cosine  s{s}', f'rigd_lejp_p1s100_mu09_p384_e1000_{s}xxx')
        for s in (1, 2, 3)]
RUNS += [(f'constLR s{s}', f'rigd_lejp_p1s100_mu09_p384_e1000_constlr_{s}xxx')
         for s in (1, 2, 3)]


def direction(d, cfg, sets, pre, post, wbar):
    """The S7(iv) construction on one explicit window. Returns the raw
    numbers only; no reading is derived here or anywhere downstream."""
    e_pre, Yne = ridge_resid(*feats(d, cfg, sets, pre))
    e_post, _ = ridge_resid(*feats(d, cfg, sets, post))
    Se_pre = e_pre.T @ e_pre / e_pre.shape[0]
    Se_post = e_post.T @ e_post / e_post.shape[0]
    Sy = Yne.T @ Yne / Yne.shape[0] + 1e-12 * torch.eye(3, dtype=Yne.dtype)
    Li = torch.linalg.inv(torch.linalg.cholesky(Sy))
    Mm = Li @ (Se_post - Se_pre) @ Li.T
    lam, V = torch.linalg.eigh((Mm + Mm.T) / 2)
    w = Li.T @ V[:, -1]
    w = w / w.norm()
    if w[0] < 0:
        w = -w
    dr2 = float(w @ (Se_post - Se_pre) @ w / (w @ Sy @ w))
    return (w, dr2, abs(float(w @ wbar)),
            float(lam.abs().max() / lam.abs().sum()))


def wbar_of(wk):
    v = torch.stack(list(wk.values())).mean(0)
    return v / v.norm()


def explicit(run_prefix, cache_name, windows):
    """Two-window (or n-window) decomposition on one run, the standing
    discriminator. PRE:POST pairs are given, not inferred from R_A."""
    wk = {k: torch.tensor(v, dtype=torch.float64) for k, v in W_KNEE.items()}
    wk = {k: v / v.norm() for k, v in wk.items()}
    wbar = wbar_of(wk)
    hits = glob.glob(os.path.join(drv.ROOT, run_prefix + '*'))
    if not hits:
        print(f'no run dir matching {run_prefix}')
        return
    d = hits[0]
    cfg = yaml.safe_load(open(os.path.join(d, 'config.yml')))
    sets = torch.load(os.path.join(CACHE, cache_name), map_location='cpu',
                      weights_only=True)
    print(f'{os.path.basename(d)}   cache {cache_name}')
    print(f'{"window":16s} {"w_z1":>6s} {"w_z2":>6s} {"w_E":>6s} '
          f'{"dR2":>7s} {"|cos|":>7s} {"lam%":>6s}')
    for pre, post in windows:
        w, dr2, cosv, ls = direction(d, cfg, sets, pre, post, wbar)
        print(f'{f"{pre}->{post}":16s} {float(w[0]):6.3f} {float(w[1]):6.3f} '
              f'{float(w[2]):6.3f} {dr2:7.3f} {cosv:7.4f} {100 * ls:5.1f}%')
    print('raw values only; no readings emitted (no-labels rule).')


def main():
    wk = {k: torch.tensor(v, dtype=torch.float64) for k, v in W_KNEE.items()}
    wk = {k: v / v.norm() for k, v in wk.items()}
    wbar = torch.stack(list(wk.values())).mean(0)
    wbar = wbar / wbar.norm()

    traj = {}
    for r in csv.DictReader(open(TRAJ)):
        traj.setdefault(r['run'].split('xxx')[0], {})[int(r['epoch'])] = \
            float(r['mean_cca'])

    sets = torch.load(os.path.join(CACHE, CACHE_NAME), map_location='cpu',
                      weights_only=True)

    print(f'{"run":13s} {"R_A":>4s} {"PRE":>4s} {"POST":>5s} {"w_z1":>6s} '
          f'{"w_z2":>6s} {"w_E":>6s} {"dR2":>7s} {"|cos|":>7s} {"lam%":>6s} '
          f'{"contam":>7s}   |cos| to the four mu=0 vectors')
    for label, prefix in RUNS:
        hits = glob.glob(os.path.join(drv.ROOT, prefix + '*'))
        if not hits:
            print(f'{label:13s} no run dir')
            continue
        d = hits[0]
        key = os.path.basename(d).split('xxx')[0]
        t = traj.get(key)
        if not t:
            print(f'{label:13s} no CCA trajectory')
            continue
        on, _ = onset_A(t)
        grid = sorted(t)
        if on is None:
            print(f'{label:13s} no R_A onset -> no PRE anchor defined')
            continue
        pre = max((e for e in grid if e <= on - 20), default=None)
        post = max(grid)
        if pre is None:
            print(f'{label:13s} onset {on} too early for a PRE')
            continue

        cfg = yaml.safe_load(open(os.path.join(d, 'config.yml')))
        e_pre, Yne = ridge_resid(*feats(d, cfg, sets, pre))
        e_post, _ = ridge_resid(*feats(d, cfg, sets, post))
        Se_pre = e_pre.T @ e_pre / e_pre.shape[0]
        Se_post = e_post.T @ e_post / e_post.shape[0]
        Sy = Yne.T @ Yne / Yne.shape[0] + 1e-12 * torch.eye(3, dtype=Yne.dtype)
        Li = torch.linalg.inv(torch.linalg.cholesky(Sy))
        Mm = Li @ (Se_post - Se_pre) @ Li.T
        lam, V = torch.linalg.eigh((Mm + Mm.T) / 2)
        w = Li.T @ V[:, -1]
        w = w / w.norm()
        if w[0] < 0:
            w = -w
        dr2 = float(w @ (Se_post - Se_pre) @ w / (w @ Sy @ w))
        cosv = abs(float(w @ wbar))
        lam_share = float(lam.abs().max() / lam.abs().sum())

        pk = max(t, key=lambda e: t[e])
        total = t[pk] - t[post]
        contam = (t[pk] - t[pre]) / total if total > 0 else float('nan')

        per = ' '.join(f'{abs(float(w @ v)):.3f}' for v in wk.values())
        print(f'{label:13s} {on:4d} {pre:4d} {post:5d} {float(w[0]):6.3f} '
              f'{float(w[1]):6.3f} {float(w[2]):6.3f} {dr2:7.3f} '
              f'{cosv:7.4f} {100 * lam_share:5.1f}% {contam:7.3f}   {per}')

    print(f'\nw-bar = ({float(wbar[0]):.3f}, {float(wbar[1]):.3f}, '
          f'{float(wbar[2]):.3f}); no readings emitted (no-labels rule).')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', help='run dir prefix; omit for the six-run table')
    ap.add_argument('--cache', default=CACHE_NAME,
                    help='probe cache basename; MUST match the run\'s dial '
                         'point/rung (levels do not travel across caches)')
    ap.add_argument('--windows', default='',
                    help='comma-separated PRE:POST pairs, e.g. 70:300,300:1000')
    a = ap.parse_args()
    if a.run:
        wins = [tuple(int(x) for x in p.split(':'))
                for p in a.windows.split(',') if p]
        if not wins:
            ap.error('--run needs --windows PRE:POST[,PRE:POST...]')
        explicit(a.run, a.cache, wins)
    else:
        main()
