"""P2: the direction battery's NULL -- identical machinery run on
EVENT-FREE runs (2026-09-02; the statistical control).

Construction IMPORTED (knee_t2_direction: ridge_resid, feats, W_KNEE;
knee_dir_sweep: cache_for) so these rows are produced by exactly the
machinery that produced the knee rows. Nothing reimplemented.

MATCHING RULE, written down before computing. Flat runs have no
onset, so no PRE can come from their own data without circularity. Each
flat run is measured over THREE windows whose PRE epochs are taken from
its campaign's KNEE cohort -- that cohort's median, min and max PRE --
with POST = the flat run's final checkpoint, matching the knee
convention's POST=final. Anchors never depend on the run being tested,
so the rule cannot be tuned by what it finds. Three windows give a
per-run spread rather than a single point.

Raw values only, no verdict labels.
"""
import csv
import glob
import os
import statistics
import sys

import torch
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
sys.path.insert(0, _HERE)
import wp1_probe_driver as drv  # noqa: E402
from knee_dir_sweep import cache_for  # noqa: E402
from knee_t2_direction import W_KNEE, feats, ridge_resid  # noqa: E402

CACHE = paths.CACHE
SWEEP = os.path.join(paths.CSV, 'knee_dir_sweep.csv')
OUT = os.path.join(paths.CSV, 'knee_dir_null.csv')

# Event-free runs, taken from what the record ALREADY calls a control or
# reports as flat -- not selected by a fresh threshold invented here.
FLAT_21 = ([f'rigd_lejp_p1s100_mu097_nano_{s}' for s in (1, 2, 3)]
           + ['rigd_lejp_p1s100_mu097_long_nano_1', 'rigb_lejp_p1s100_nano_5'])
FLAT_4 = ([f'rigd_lejp_p1s100_mu097_p384_{s}' for s in (1, 2, 3, 4)]
          + ['rigb_lejp_p1s100_p384_10']
          + [f'rigb_lejp_p1s100_p384_ou0{k}_{s}' for k in ('5', '9')
             for s in (1, 2, 3)]
          + [f'rigb_lejp_p1s100_p384_ou09_e1000_{s}' for s in (1, 2, 3)]
          + [f'rigd_lejp_p1s100_mu097_p384_ou09_{s}' for s in (1, 2, 3)]
          + [f'rigb_lejp_p1s300_p384_{s}' for s in (1, 2, 3)]
          + [f'rigb_lejp_p1s1000_p384_{s}' for s in (1, 2, 3)])


def knee_pre_anchors():
    """median / min / max PRE epoch of the KNEE runs, per campaign."""
    pres = {'21x': [], '4x': []}
    with open(SWEEP) as fh:
        for r in csv.DictReader(fh):
            camp = '21x' if 'nano' in r['run'] else '4x'
            pres[camp].append(int(r['pre']))
    return {c: (int(statistics.median(v)), min(v), max(v))
            for c, v in pres.items() if v}


def main():
    wk = {k: torch.tensor(v, dtype=torch.float64) for k, v in W_KNEE.items()}
    wk = {k: v / v.norm() for k, v in wk.items()}
    wbar = torch.stack(list(wk.values())).mean(0)
    wbar = wbar / wbar.norm()

    anchors = knee_pre_anchors()
    print('PRE anchors from each campaign\'s KNEE cohort (median, min, max):')
    for c, a in anchors.items():
        print(f'  {c}: {a}')

    done = set()
    if os.path.exists(OUT):
        done = {(r['run'], int(r['pre'])) for r in csv.DictReader(open(OUT))}
    new = not os.path.exists(OUT)
    fh = open(OUT, 'a', newline='')
    w = csv.writer(fh, lineterminator='\n')
    if new:
        w.writerow(['run', 'campaign', 'cache', 'pre', 'post',
                    'w_z1', 'w_z2', 'w_E', 'dR2', 'cos_wbar', 'lam_share'])

    sets_cache = {}
    for camp, runs in (('21x', FLAT_21), ('4x', FLAT_4)):
        med, lo, hi = anchors[camp]
        for run in runs:
            hits = glob.glob(os.path.join(drv.ROOT, run + 'xxx*'))
            if not hits:
                print(f'  skip (no dir): {run}', flush=True)
                continue
            d = hits[0]
            cfg = yaml.safe_load(open(os.path.join(d, 'config.yml')))
            tag = cfg['logging']['write_tag']
            cks = {int(p.split('-ep')[-1].split('.')[0]): p for p in
                   glob.glob(os.path.join(d, f'{tag}-ep*.pth.tar'))}
            if not cks:
                print(f'  skip (no ckpts): {run}', flush=True)
                continue
            post = max(cks)
            cname = cache_for(run)
            if cname not in sets_cache:
                p = os.path.join(CACHE, cname)
                if not os.path.exists(p):
                    print(f'  skip (cache {cname} missing): {run}', flush=True)
                    continue
                sets_cache[cname] = torch.load(p, map_location='cpu',
                                               weights_only=True)
            sets = sets_cache[cname]
            for pre_want in sorted({med, lo, hi}):
                # nearest available checkpoint at or below the anchor
                cands = [e for e in cks if e <= pre_want and e > 20]
                if not cands:
                    continue
                pre = max(cands)
                if (run, pre) in done or pre >= post:
                    continue
                e_pre, Yne = ridge_resid(*feats(d, cfg, sets, pre))
                e_post, _ = ridge_resid(*feats(d, cfg, sets, post))
                Sp = e_pre.T @ e_pre / e_pre.shape[0]
                Sq = e_post.T @ e_post / e_post.shape[0]
                Sy = (Yne.T @ Yne / Yne.shape[0]
                      + 1e-12 * torch.eye(3, dtype=Yne.dtype))
                Li = torch.linalg.inv(torch.linalg.cholesky(Sy))
                Mm = Li @ (Sq - Sp) @ Li.T
                lam, V = torch.linalg.eigh((Mm + Mm.T) / 2)
                wv = Li.T @ V[:, -1]
                wv = wv / wv.norm()
                if wv[0] < 0:
                    wv = -wv
                dr2 = float(wv @ (Sq - Sp) @ wv / (wv @ Sy @ wv))
                cosv = abs(float(wv @ wbar))
                w.writerow([run, camp, cname, pre, post,
                            f'{float(wv[0]):.6f}', f'{float(wv[1]):.6f}',
                            f'{float(wv[2]):.6f}', f'{dr2:.6f}',
                            f'{cosv:.6f}',
                            f'{float(lam.abs().max() / lam.abs().sum()):.6f}'])
                fh.flush()
            print(f'{run:44s} done', flush=True)
    fh.close()
    print('DONE knee_dir_null')


if __name__ == '__main__':
    main()
