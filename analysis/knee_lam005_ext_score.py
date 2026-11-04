"""Dense CCA trajectories for the lambda=0.05 arm extension, seeds 4-10
(launched 2026-09-29).

Same scoring as the arm's seeds 1-3 in results/knee_trajectories.csv: probe cache
rig_P1_Enar_S100.pt, encode_meanpool + wp1_metric_row, one row per saved
checkpoint. Writes a NEW file, results/knee_traj_lam005_ext.csv, and never touches
knee_trajectories.csv. Idempotent via its own CSV.

  --check   re-score seed 2 of the original arm at a few epochs and print them
            beside the stored knee_trajectories.csv values (reproduction check;
            writes nothing).
  --seeds   which extension seeds to score (default 4..10).
"""
import argparse
import csv
import glob
import os
import sys
import time

import torch
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
sys.path.insert(0, _HERE)
import wp1_probe_driver as drv  # noqa: E402
from wp1_metrics import wp1_metric_row  # noqa: E402

ROOT = drv.ROOT
CACHE = os.path.join(paths.CACHE, 'rig_P1_Enar_S100.pt')
OUT = os.path.join(paths.CSV, 'knee_traj_lam005_ext.csv')
STORED = os.path.join(paths.CSV, 'knee_trajectories.csv')
COLS = ['run', 'epoch', 'mean_cca', 'cca_c1', 'cca_c2', 'orth_err',
        'ridge_r2_z1', 'ridge_r2_z2', 'ridge_r2_E']


def ckpts(d, cfg):
    tag = cfg['logging']['write_tag']
    return sorted(glob.glob(os.path.join(d, f'{tag}-ep*.pth.tar')),
                  key=lambda p: int(p.split('-ep')[-1].split('.')[0]))


def score(cfg, ck, sets):
    enc = drv.encoder_from_ckpt(cfg, ck)
    return wp1_metric_row(drv.encode_meanpool(enc, sets['imgs_tr']),
                          sets['y_tr'],
                          drv.encode_meanpool(enc, sets['imgs_ev']),
                          sets['y_ev'])


def check(sets):
    d = glob.glob(os.path.join(ROOT, 'rigb_lejp_p1s100_p384_lam005_2xxx*'))[0]
    run = os.path.basename(d)
    cfg = yaml.safe_load(open(os.path.join(d, 'config.yml')))
    stored = {int(r['epoch']): float(r['mean_cca'])
              for r in csv.DictReader(open(STORED)) if r['run'] == run}
    for ck in ckpts(d, cfg):
        ep = int(ck.split('-ep')[-1].split('.')[0])
        if ep not in (2, 80, 150, 300):
            continue
        t0 = time.time()
        m = score(cfg, ck, sets)
        print(f'{run} ep{ep}: rescored {m["mean_cca"]:.6f}  stored '
              f'{stored.get(ep, float("nan")):.6f}  ({time.time() - t0:.0f}s)',
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(4, 11)))
    a = ap.parse_args()
    sets = torch.load(CACHE, map_location='cpu', weights_only=True)
    if a.check:
        check(sets)
        return
    done = set()
    if os.path.exists(OUT):
        done = {(r['run'], int(r['epoch'])) for r in csv.DictReader(open(OUT))}
    new = not os.path.exists(OUT)
    fout = open(OUT, 'a', newline='')
    w = csv.writer(fout, lineterminator='\n')
    if new:
        w.writerow(COLS)
    for s in a.seeds:
        hits = glob.glob(os.path.join(ROOT, f'rigb_lejp_p1s100_p384_lam005_{s}xxx*'))
        if not hits:
            print(f'seed {s}: no run dir', flush=True)
            continue
        d = hits[0]
        run = os.path.basename(d)
        cfg = yaml.safe_load(open(os.path.join(d, 'config.yml')))
        for ck in ckpts(d, cfg):
            ep = int(ck.split('-ep')[-1].split('.')[0])
            if (run, ep) in done:
                continue
            m = score(cfg, ck, sets)
            w.writerow([run, ep, f"{m['mean_cca']:.6f}",
                        f"{m['cca_per_component'][0]:.6f}",
                        f"{m['cca_per_component'][1]:.6f}",
                        f"{m['orth_err']:.6f}",
                        f"{m['ridge_r2_z1']:.6f}", f"{m['ridge_r2_z2']:.6f}",
                        f"{m['ridge_r2_E']:.6f}"])
            fout.flush()
        print(f'seed {s}: done', flush=True)
    fout.close()


if __name__ == '__main__':
    main()
