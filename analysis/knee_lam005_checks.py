"""Checks on the lambda=0.05 runs that have an onset (2026-10-04): do the
loss, RankMe and direction behave as in the lambda=0.02 declines?

Runs: rigb_lejp_p1s100_p384_lam005_{1..10}; an onset is R_A, else R_B
(knee_s1_onsets, unchanged), on the mean-CCA trajectories in
csv/knee_trajectories.csv (seeds 1-3) and csv/knee_traj_lam005_ext.csv
(seeds 4-10). Pre-onset = last grid epoch <= onset - 20; final = last.

Quantities, each with the definition the paper's lambda=0.02 numbers used:
  loss drift   within-run Spearman of own-direction quality (-loss) against
               epoch over checkpoints > 20, before onset (epoch < onset) and
               after (epoch >= onset); >= 4 checkpoints. Loss at grid epoch N
               = mean of the 'loss' rows of lejepa_diag_r0.csv at logged epoch
               N - 1 (knee_loss_census.series, imported; alignment as in
               knee_labelfree_selection.loss_series). Spearman imported from
               knee_lidar_vs_readability.
  RankMe       published RankMe (src/utils/logging.effective_rank, imported)
               on the uncentred 4,000-event fitting split of
               rig_P1_Enar_S100.pt, encoded as knee_rank_estimators_prepost
               does, at pre-onset and final.
  direction    exactly knee_dir_sweep's construction (ridge_resid, feats,
               W_KNEE imported from knee_t2_direction): leading generalized
               eigenvector of the residual second-moment change, |cos| with
               the reference direction, rank-one share, directional dR2.

Writes csv/knee_lam005_checks.csv (new file). Reads only existing files.
"""
import sys

sys.dont_write_bytecode = True

import csv  # noqa: E402
import glob  # noqa: E402
import os  # noqa: E402

import torch  # noqa: E402
import yaml  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
sys.path.insert(0, _HERE)
import wp1_probe_driver as drv  # noqa: E402
import knee_loss_census as LC  # noqa: E402
from knee_lidar_vs_readability import spearman  # noqa: E402
from knee_s1_onsets import onset_A, onset_B  # noqa: E402
from knee_t2_direction import W_KNEE, feats, ridge_resid  # noqa: E402
from src.utils.logging import effective_rank  # noqa: E402

CACHE = os.path.join(paths.CACHE, 'rig_P1_Enar_S100.pt')
TRAJ = [os.path.join(paths.CSV, 'knee_trajectories.csv'),
        os.path.join(paths.CSV, 'knee_traj_lam005_ext.csv')]
OUT = os.path.join(paths.CSV, 'knee_lam005_checks.csv')
WARM, MIN_N = 20, 4


def drift(eps, xs, lo=None, hi=None):
    pairs = [(e, -x) for e, x in zip(eps, xs)
             if e > WARM and (lo is None or e >= lo) and (hi is None or e < hi)]
    if len(pairs) < MIN_N:
        return None, len(pairs)
    return spearman([p[0] for p in pairs], [p[1] for p in pairs]), len(pairs)


def main():
    assert not os.path.exists(OUT), OUT
    traj = {}
    for p in TRAJ:
        for r in csv.DictReader(open(p)):
            if 'p384_lam005_' in r['run']:
                traj.setdefault(r['run'], {})[int(r['epoch'])] = float(r['mean_cca'])
    wk = {k: torch.tensor(v, dtype=torch.float64) for k, v in W_KNEE.items()}
    wk = {k: v / v.norm() for k, v in wk.items()}
    wbar = torch.stack(list(wk.values())).mean(0)
    wbar = wbar / wbar.norm()
    sets = torch.load(CACHE, map_location='cpu', weights_only=True)
    rows = []
    for run in sorted(traj, key=lambda r: int(r.split('lam005_')[1].split('xxx')[0])):
        t = traj[run]
        on, rule = onset_A(t)[0], 'R_A'
        if on is None:
            on, rule = onset_B(t)[0], 'R_B'
        if on is None:
            continue
        grid = sorted(t)
        pre = max(e for e in grid if e <= on - 20)
        post = max(grid)
        hits = glob.glob(os.path.join(drv.ROOT, run))
        assert len(hits) == 1, (run, hits)
        d = hits[0]
        cfg = yaml.safe_load(open(os.path.join(d, 'config.yml')))
        # loss drift, census reader, grid epoch N -> logged epoch N - 1
        leps, lval = LC.series(run, 'lejepa')
        by = dict(zip(leps, lval))
        lgrid = [e for e in grid if (e - 1) in by]
        lxs = [by[e - 1] for e in lgrid]
        rb, nb = drift(lgrid, lxs, hi=on)
        ra, na = drift(lgrid, lxs, lo=on)
        # RankMe, published formula, uncentred fitting split
        rk = {}
        for ph, ep in (('pre', pre), ('post', post)):
            enc = drv.encoder_from_ckpt(cfg, os.path.join(d, f'jepa-ep{ep}.pth.tar'))
            X = drv.encode_meanpool(enc, sets['imgs_tr']).double()
            rk[ph] = float(effective_rank(X))
        # direction, knee_dir_sweep construction
        e_pre, Yne = ridge_resid(*feats(d, cfg, sets, pre))
        e_post, _ = ridge_resid(*feats(d, cfg, sets, post))
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
        row = {'run': run, 'onset_rule': rule, 'onset': on, 'pre': pre, 'post': post,
               'drop': f'{t[pk] - t[post]:.6f}',
               'loss_pre_ep': f'{by[pre - 1]:.6f}', 'loss_post_ep': f'{by[post - 1]:.6f}',
               'loss_drift_before': 'nan' if rb is None else f'{rb:.4f}', 'n_before': nb,
               'loss_drift_after': 'nan' if ra is None else f'{ra:.4f}', 'n_after': na,
               'rankme_pre': f"{rk['pre']:.6f}", 'rankme_post': f"{rk['post']:.6f}",
               'w_z1': f'{float(wv[0]):.6f}', 'w_z2': f'{float(wv[1]):.6f}',
               'w_E': f'{float(wv[2]):.6f}',
               'cos_wbar': f'{abs(float(wv @ wbar)):.6f}',
               'lam_share': f'{float(lam.abs().max() / lam.abs().sum()):.6f}',
               'dR2': f'{float(wv @ (Sq - Sp) @ wv / (wv @ Sy @ wv)):.6f}'}
        rows.append(row)
        print(row, flush=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
