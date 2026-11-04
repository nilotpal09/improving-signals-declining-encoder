"""Consolidated dying-direction sweep over every scored knee run, into
one CSV. Built 2026-08-25 because the ~21 direction measurements on
record live scattered in written notes with no machine-readable
source, and a figure going to an external reader must not be built by
scraping prose.

Construction IMPORTED from knee_t2_direction (ridge_resid, feats,
W_KNEE) -- identical machinery to the battery fixed in advance. Anchor is
the printed rule: PRE = last grid epoch <= R_A - 20, POST = last.

Each run is scored on its OWN dial-matched cache (levels do not travel
across caches).

Writes results/knee_dir_sweep.csv. Idempotent by run. Raw values only --
no readings, no labels.
"""
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
from knee_s1_onsets import onset_A, onset_B  # noqa: E402
from knee_t2_direction import W_KNEE, feats, ridge_resid  # noqa: E402

CACHE = paths.CACHE
TRAJ = os.path.join(paths.CSV, 'knee_trajectories.csv')
OUT = os.path.join(paths.CSV, 'knee_dir_sweep.csv')
COLS = ['run', 'cache', 'onset_rule', 'onset', 'pre', 'post', 'w_z1', 'w_z2',
        'w_E', 'dR2', 'cos_wbar', 'lam_share', 'pre_contam', 'drop', 'peak_ep']


def cache_for(run):
    """Dial-matched cache, by run name. mu=0 runs use the narrow-E set."""
    for tag, name in (('mu097', 'rigd_p1s100_mu097.pt'),
                      ('mu09', 'rigd_p1s100_mu09.pt'),
                      ('mu07', 'rigd_p1s100_mu07.pt'),
                      ('mu04', 'rigd_p1s100_mu04.pt')):
        if tag in run:
            return name
    for tag, name in (('p1s30_', 'rig_P1_Enar_S30_tda.pt'),
                      ('p1s300_', 'rig_P1_Enar_S300_tda.pt'),
                      ('p1s1000_', 'rig_P1_Enar_S1000_tda.pt')):
        if tag in run:
            return name
    return 'rig_P1_Enar_S100.pt'


def main():
    wk = {k: torch.tensor(v, dtype=torch.float64) for k, v in W_KNEE.items()}
    wk = {k: v / v.norm() for k, v in wk.items()}
    wbar = torch.stack(list(wk.values())).mean(0)
    wbar = wbar / wbar.norm()

    traj = {}
    for r in csv.DictReader(open(TRAJ)):
        traj.setdefault(r['run'].split('xxx')[0], {})[int(r['epoch'])] = \
            float(r['mean_cca'])

    done = set()
    if os.path.exists(OUT):
        done = {r['run'] for r in csv.DictReader(open(OUT))}
    new = not os.path.exists(OUT)
    fout = open(OUT, 'a', newline='')
    w_csv = csv.DictWriter(fout, fieldnames=COLS, lineterminator='\n')
    if new:
        w_csv.writeheader()

    sets_cache = {}
    for run in sorted(traj):
        if run in done:
            continue
        t = traj[run]
        if max(t) < 300:
            continue                       # still training
        on, rule = onset_A(t)[0], 'R_A'
        if on is None:
            on, rule = onset_B(t)[0], 'R_B'
        if on is None:
            continue                       # no onset -> no anchor, skipped
        grid = sorted(t)
        pre = max((e for e in grid if e <= on - 20), default=None)
        post = max(grid)
        if pre is None:
            continue
        hits = glob.glob(os.path.join(drv.ROOT, run + 'xxx*'))
        if not hits:
            continue
        d = hits[0]
        cname = cache_for(run)
        if cname not in sets_cache:
            p = os.path.join(CACHE, cname)
            if not os.path.exists(p):
                print(f'{run}: cache {cname} missing, skipped', flush=True)
                continue
            sets_cache[cname] = torch.load(p, map_location='cpu',
                                           weights_only=True)
        sets = sets_cache[cname]
        cfg = yaml.safe_load(open(os.path.join(d, 'config.yml')))
        try:
            e_pre, Yne = ridge_resid(*feats(d, cfg, sets, pre))
            e_post, _ = ridge_resid(*feats(d, cfg, sets, post))
        except Exception as exc:                       # noqa: BLE001
            print(f'{run}: {type(exc).__name__} {exc}', flush=True)
            continue
        Sp = e_pre.T @ e_pre / e_pre.shape[0]
        Sq = e_post.T @ e_post / e_post.shape[0]
        Sy = Yne.T @ Yne / Yne.shape[0] + 1e-12 * torch.eye(3, dtype=Yne.dtype)
        Li = torch.linalg.inv(torch.linalg.cholesky(Sy))
        Mm = Li @ (Sq - Sp) @ Li.T
        lam, V = torch.linalg.eigh((Mm + Mm.T) / 2)
        wv = Li.T @ V[:, -1]
        wv = wv / wv.norm()
        if wv[0] < 0:
            wv = -wv
        pk = max(t, key=lambda e: t[e])
        total = t[pk] - t[post]
        w_csv.writerow({
            'run': run, 'cache': cname, 'onset_rule': rule, 'onset': on,
            'pre': pre, 'post': post,
            'w_z1': f'{float(wv[0]):.6f}', 'w_z2': f'{float(wv[1]):.6f}',
            'w_E': f'{float(wv[2]):.6f}',
            'dR2': f'{float(wv @ (Sq - Sp) @ wv / (wv @ Sy @ wv)):.6f}',
            'cos_wbar': f'{abs(float(wv @ wbar)):.6f}',
            'lam_share': f'{float(lam.abs().max() / lam.abs().sum()):.6f}',
            'pre_contam': f'{(t[pk] - t[pre]) / total:.6f}'
            if total > 0 else 'nan',
            'drop': f'{total:.6f}', 'peak_ep': pk})
        fout.flush()
        print(f'{run:44s} {rule} on{on:<5d} cos {abs(float(wv @ wbar)):.4f} '
              f'lam {float(lam.abs().max() / lam.abs().sum()):.3f}', flush=True)
    fout.close()
    print('DONE knee_dir_sweep')


if __name__ == '__main__':
    main()
