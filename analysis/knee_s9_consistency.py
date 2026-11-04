"""S9: S5-vs-S7 consistency check, per the formula and rule written in the
record (2026-08-19, before computation).

Measured  dR2_E = D_EE / Sy_EE, D = Se_post - Se_pre.
Predicted (rank-one hypothesis, exact form)
          dR2_E^pred = dR2(w) * (e_E' Sy w)^2 / [(w' Sy w) * Sy_EE]
          reduced form (Sy ~ I): dR2(w) * w_E^2
Rank-one diagnostic: eigenvalues of Sy^-1/2 D Sy^-1/2.
Tolerance: 500-resample eval-split bootstrap (seed 20260960), 90% interval
of (measured - predicted).
"""
import glob
import os
import sys

import torch
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
sys.path.insert(0, _HERE)
import wp1_probe_driver as drv  # noqa: E402

CACHE = paths.CACHE
SETS = 'rig_P1_Enar_S100.pt'
N_BOOT, BOOT_SEED = 500, 20260960
RUNS = {  # pattern -> (PRE, POST), the S3/S7 checkpoints, unchanged
    'rigb_lejp_p1s100_nano_1xxx*': (70, 150),
    'rigb_lejp_p1s100_nano_2xxx*': (90, 150),
    'rigb_lejp_p1s100_nano_3xxx*': (80, 150),
    'rigb_lejp_p1s100_long_nano_1xxx*': (140, 300),
}


def resid(h_tr, y_tr, h_ev, y_ev, alpha=1.0):
    """Eval residuals + standardized eval targets, house ridge."""
    h_tr, y_tr, h_ev, y_ev = (t.double() for t in (h_tr, y_tr, h_ev, y_ev))
    Xm, Xs = h_tr.mean(0), h_tr.std(0) + 1e-8
    Ym, Ys = y_tr.mean(0), y_tr.std(0) + 1e-8
    Xn, Yn = (h_tr - Xm) / Xs, (y_tr - Ym) / Ys
    K = Xn.shape[1]
    W = torch.linalg.solve(Xn.T @ Xn + alpha * torch.eye(K, dtype=Xn.dtype),
                           Xn.T @ Yn)
    Yne = (y_ev - Ym) / Ys
    return Yne - ((h_ev - Xm) / Xs) @ W, Yne


def quantities(e_pre, e_post, Yne):
    """(measured, predicted_exact, predicted_reduced, w, dr2_w, eigs)."""
    n = e_pre.shape[0]
    Se_pre = e_pre.T @ e_pre / n
    Se_post = e_post.T @ e_post / n
    D = Se_post - Se_pre
    Sy = Yne.T @ Yne / n + 1e-12 * torch.eye(3, dtype=Yne.dtype)
    Li = torch.linalg.inv(torch.linalg.cholesky(Sy))
    M = Li @ D @ Li.T
    M = (M + M.T) / 2
    lam, V = torch.linalg.eigh(M)
    w = Li.T @ V[:, -1]
    w = w / w.norm()
    if w[0] < 0:
        w = -w
    dr2_w = float(w @ D @ w / (w @ Sy @ w))
    e_E = torch.tensor([0., 0., 1.], dtype=Yne.dtype)
    measured = float(D[2, 2] / Sy[2, 2])
    pred_exact = float(dr2_w * (e_E @ Sy @ w) ** 2
                       / ((w @ Sy @ w) * Sy[2, 2]))
    pred_red = float(dr2_w * w[2] ** 2)
    return measured, pred_exact, pred_red, w, dr2_w, lam.flip(0)


def main():
    sets = torch.load(os.path.join(CACHE, SETS), map_location='cpu',
                      weights_only=True)
    y_tr, y_ev = sets['y_tr'], sets['y_ev']

    print(f'{"run":30s} {"dR2(w)":>7s} {"w_E":>6s} {"pred_ex":>8s} {"pred_red":>9s} '
          f'{"measured":>9s} {"meas-pred":>10s} {"boot 90% CI":>18s} '
          f'{"lam1 share":>10s}')
    tally_match, tally_extra = 0, 0
    rows = []
    for pat, (pre, post) in RUNS.items():
        d = glob.glob(os.path.join(drv.ROOT, pat))[0]
        cfg = yaml.safe_load(open(os.path.join(d, 'config.yml')))
        E = {}
        for ph, ep in (('pre', pre), ('post', post)):
            enc = drv.encoder_from_ckpt(cfg, os.path.join(d, f'jepa-ep{ep}.pth.tar'))
            E[ph] = (drv.encode_meanpool(enc, sets['imgs_tr']),
                     drv.encode_meanpool(enc, sets['imgs_ev']))
            del enc
        e_pre, Yne = resid(E['pre'][0], y_tr, E['pre'][1], y_ev)
        e_post, _ = resid(E['post'][0], y_tr, E['post'][1], y_ev)
        meas, pex, pred, w, dr2_w, lam = quantities(e_pre, e_post, Yne)
        share = float(lam[0].abs() / lam.abs().sum())

        g = torch.Generator().manual_seed(BOOT_SEED)
        n = e_pre.shape[0]
        diffs = []
        for _ in range(N_BOOT):
            idx = torch.randint(0, n, (n,), generator=g)
            m2, p2, _, _, _, _ = quantities(e_pre[idx], e_post[idx], Yne[idx])
            diffs.append(m2 - p2)
        diffs = torch.tensor(diffs, dtype=torch.float64)
        lo, hi = (float(diffs.quantile(0.05)), float(diffs.quantile(0.95)))
        contains0 = lo <= 0.0 <= hi
        if contains0 and share >= 0.9:
            tally_match += 1
        if ((meas - pex) > 0 and not contains0) or share < 0.9:
            tally_extra += 1
        tag = pat.split('xxx')[0]
        print(f'{tag:30s} {dr2_w:7.3f} {float(w[2]):6.3f} {pex:8.4f} {pred:9.4f} '
              f'{meas:9.4f} {meas - pex:10.4f} '
              f'{f"[{lo:+.4f},{hi:+.4f}]":>18s} {share:10.3f}')
        rows.append((tag, lam, share))

    print('\nfull eigenvalues of the whitened change Sy^-1/2 (Se_post-Se_pre) Sy^-1/2:')
    for tag, lam, share in rows:
        print(f'{tag:30s} ' + ' '.join(f'{float(x):+8.4f}' for x in lam)
              + f'   lam1 share {share:.3f}')

    n_runs = len(RUNS)
    print(f'\nrule tally over {n_runs} runs: "fully one direction" met in '
          f'{tally_match}, "something else dying" met in {tally_extra}')
    if tally_match > n_runs / 2:
        print('VERDICT: FULLY DESCRIBED BY ONE DIRECTION\'S DEATH')
    elif tally_extra > n_runs / 2:
        print('VERDICT: SOMETHING ELSE IS ALSO DYING')
    else:
        print('VERDICT: AMBIGUOUS (reported as such, per the printed rule)')


if __name__ == '__main__':
    main()
