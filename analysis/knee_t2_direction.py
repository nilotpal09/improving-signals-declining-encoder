"""Cross-mu dying-direction stability, exactly as fixed in advance in the
record before any T2 data was examined (2026-08-19).

For every T2 run (mu 0.4 / 0.7 / 0.9) that has an S1 R_A onset: rebuild the
S7(iv) dying direction by the identical construction on that point's own
dist-matched cache, then report dR2(w) and |cos| against the four printed
mu=0 vectors and their mean. Runs with no onset, or not finished, are
listed and skipped. Readings are the ones fixed in advance; nothing else.
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
from knee_s1_onsets import onset_A  # noqa: E402

CACHE = paths.CACHE
TRAJ = os.path.join(paths.CSV, 'knee_trajectories.csv')
W_KNEE = {'knee seed 1': (0.511, -0.662, 0.549),
          'knee seed 2': (0.511, -0.641, 0.573),
          'knee seed 3': (0.520, -0.632, 0.575),
          'knee long': (0.513, -0.643, 0.569)}
POINTS = {'mu04': 'rigd_p1s100_mu04.pt', 'mu07': 'rigd_p1s100_mu07.pt',
          'mu09': 'rigd_p1s100_mu09.pt'}


def ridge_resid(h_tr, y_tr, h_ev, y_ev, alpha=1.0):
    h_tr, y_tr, h_ev, y_ev = (t.double() for t in (h_tr, y_tr, h_ev, y_ev))
    Xm, Xs = h_tr.mean(0), h_tr.std(0) + 1e-8
    Ym, Ys = y_tr.mean(0), y_tr.std(0) + 1e-8
    Xn, Yn = (h_tr - Xm) / Xs, (y_tr - Ym) / Ys
    K = Xn.shape[1]
    W = torch.linalg.solve(Xn.T @ Xn + alpha * torch.eye(K, dtype=Xn.dtype),
                           Xn.T @ Yn)
    Yne = (y_ev - Ym) / Ys
    return Yne - ((h_ev - Xm) / Xs) @ W, Yne


def feats(run_dir, cfg, sets, ep):
    enc = drv.encoder_from_ckpt(cfg, os.path.join(run_dir,
                                                  f'jepa-ep{ep}.pth.tar'))
    return (drv.encode_meanpool(enc, sets['imgs_tr']), sets['y_tr'],
            drv.encode_meanpool(enc, sets['imgs_ev']), sets['y_ev'])


def main():
    wk = {k: torch.tensor(v, dtype=torch.float64) for k, v in W_KNEE.items()}
    wk = {k: v / v.norm() for k, v in wk.items()}
    wbar = torch.stack(list(wk.values())).mean(0)
    wbar = wbar / wbar.norm()

    traj = {}
    for r in csv.DictReader(open(TRAJ)):
        traj.setdefault(r['run'], {})[int(r['epoch'])] = float(r['mean_cca'])

    print('readings fixed in advance: support = |cos| >= 0.95 AND dR2 >= 0.2; '
          'differs = |cos| <= 0.8 AND dR2 >= 0.2; else inconclusive\n')
    print(f'{"run":36s} {"onset":>5s} {"PRE":>4s} {"POST":>5s} '
          f'{"w_z1":>6s} {"w_z2":>6s} {"w_E":>6s} {"dR2":>6s} '
          f'{"|cos wbar|":>10s}  reading')
    skipped = []
    for pt, cache_name in POINTS.items():
        for seed in (1, 2, 3):
            hits = glob.glob(os.path.join(
                drv.ROOT, f'rigd_lejp_p1s100_{pt}_nano_{seed}xxx*'))
            if not hits:
                skipped.append((f'{pt}_s{seed}', 'no run dir'))
                continue
            d = hits[0]
            run = os.path.basename(d)
            if run not in traj:
                skipped.append((f'{pt}_s{seed}',
                                'no CCA trajectory (run knee_s1_onsets.py)'))
                continue
            on, _ = onset_A(traj[run])
            if on is None:
                print(f'{run.split("xxx")[0]:36s} {"none":>5s} '
                      + ' ' * 40 + 'no onset -> no direction to compare')
                continue
            grid = sorted(traj[run])
            pre = max((e for e in grid if e <= on - 20), default=None)
            post = max(grid)
            if pre is None:
                skipped.append((f'{pt}_s{seed}', f'onset {on} too early for a PRE'))
                continue
            cfg = yaml.safe_load(open(os.path.join(d, 'config.yml')))
            sets = torch.load(os.path.join(CACHE, cache_name), map_location='cpu',
                              weights_only=True)
            e_pre, Yne = ridge_resid(*feats(d, cfg, sets, pre))
            e_post, _ = ridge_resid(*feats(d, cfg, sets, post))
            Se_pre = e_pre.T @ e_pre / e_pre.shape[0]
            Se_post = e_post.T @ e_post / e_post.shape[0]
            Sy = Yne.T @ Yne / Yne.shape[0] + 1e-12 * torch.eye(3, dtype=Yne.dtype)
            Li = torch.linalg.inv(torch.linalg.cholesky(Sy))
            Mm = Li @ (Se_post - Se_pre) @ Li.T
            _, V = torch.linalg.eigh((Mm + Mm.T) / 2)
            w = Li.T @ V[:, -1]
            w = w / w.norm()
            if w[0] < 0:
                w = -w
            dr2 = float(w @ (Se_post - Se_pre) @ w / (w @ Sy @ w))
            cosv = abs(float(w @ wbar))
            if cosv >= 0.95 and dr2 >= 0.2:
                reading = 'DIRECTION-STABILITY SUPPORT'
            elif cosv <= 0.8 and dr2 >= 0.2:
                reading = 'DIRECTION DIFFERS ACROSS MU'
            else:
                reading = 'inconclusive'
            per = ' '.join(f'{abs(float(w @ v)):.3f}' for v in wk.values())
            print(f'{run.split("xxx")[0]:36s} {on:5d} {pre:4d} {post:5d} '
                  f'{float(w[0]):6.3f} {float(w[1]):6.3f} {float(w[2]):6.3f} '
                  f'{dr2:6.3f} {cosv:10.4f}  {reading}')
            print(f'{"":36s} |cos| to the four printed vectors: {per}')

    for name, why in skipped:
        print(f'  PENDING/SKIPPED {name:12s} {why}')


if __name__ == '__main__':
    main()
