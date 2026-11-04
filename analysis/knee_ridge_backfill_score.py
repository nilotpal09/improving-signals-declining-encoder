"""Per-factor ridge R2 for the 39 trajectory runs that have only mean CCA on
record (results/knee_trajectories.csv rows with no ridge columns), so that every
scored run has R2(z1), R2(z2), R2(log E) at every saved checkpoint
(readability = held-out ridge R2 of the named factors; onset
rule to be redefined on their average).

Same scoring as knee_seedext_score.py: encode_meanpool + wp1_metric_row, one
row per saved checkpoint, on the condition's cache from knee_dir_sweep.cache_for
(imported, unchanged). Writes a NEW file, results/knee_traj_ridge_backfill.csv;
knee_trajectories.csv is only read. Every scored row is compared with its
stored mean CCA (reproduction check on all rows, printed per run).
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
from knee_dir_sweep import cache_for  # noqa: E402
from wp1_metrics import wp1_metric_row  # noqa: E402

CACHE = paths.CACHE
CSV = os.path.join(paths.CSV)
STORED = os.path.join(CSV, 'knee_trajectories.csv')
RIDGE_FILES = ('knee_traj_full.csv', 'knee_traj_seedext.csv',
               'knee_traj_lam005_ext.csv', 'knee_labelfree_readability.csv')
OUT = os.path.join(CSV, 'knee_traj_ridge_backfill.csv')
COLS = ['run', 'epoch', 'cache', 'mean_cca', 'cca_c1', 'cca_c2', 'cca_c3',
        'orth_err', 'ridge_r2_z1', 'ridge_r2_z2', 'ridge_r2_E']
_sets = {}


def sets_for(cname):
    if cname not in _sets:
        _sets[cname] = torch.load(os.path.join(CACHE, cname), map_location='cpu',
                                  weights_only=True)
    return _sets[cname]


def ckpts(d, cfg):
    tag = cfg['logging']['write_tag']
    return sorted(glob.glob(os.path.join(d, f'{tag}-ep*.pth.tar')),
                  key=lambda p: int(p.split('-ep')[-1].split('.')[0]))


def todo_runs():
    stored = {}
    for r in csv.DictReader(open(STORED)):
        stored.setdefault(r['run'], {})[int(r['epoch'])] = float(r['mean_cca'])
    has_ridge = set()
    for f in RIDGE_FILES:
        for r in csv.DictReader(open(os.path.join(CSV, f))):
            if r.get('ridge_r2_z1'):
                has_ridge.add(r['run'])
    return {k: v for k, v in stored.items() if k not in has_ridge}


def main():
    runs = todo_runs()
    print(f'{len(runs)} runs without ridge R2', flush=True)
    done = set()
    if os.path.exists(OUT):
        done = {(r['run'], int(r['epoch'])) for r in csv.DictReader(open(OUT))}
    new = not os.path.exists(OUT)
    fout = open(OUT, 'a', newline='')
    w = csv.writer(fout, lineterminator='\n')
    if new:
        w.writerow(COLS)
    for run in sorted(runs):
        d = os.path.join(drv.ROOT, run)
        cname = cache_for(run.split('xxx')[0])
        cfg = yaml.safe_load(open(os.path.join(d, 'config.yml')))
        worst, n = 0.0, 0
        for ck in ckpts(d, cfg):
            ep = int(ck.split('-ep')[-1].split('.')[0])
            if (run, ep) in done:
                continue
            sets = sets_for(cname)
            enc = drv.encoder_from_ckpt(cfg, ck)
            m = wp1_metric_row(drv.encode_meanpool(enc, sets['imgs_tr']), sets['y_tr'],
                               drv.encode_meanpool(enc, sets['imgs_ev']), sets['y_ev'])
            c = m['cca_per_component']
            w.writerow([run, ep, cname, f"{m['mean_cca']:.6f}", f'{c[0]:.6f}',
                        f'{c[1]:.6f}', f'{c[2]:.6f}', f"{m['orth_err']:.6f}",
                        f"{m['ridge_r2_z1']:.6f}", f"{m['ridge_r2_z2']:.6f}",
                        f"{m['ridge_r2_E']:.6f}"])
            fout.flush()
            if ep in runs[run]:
                worst = max(worst, abs(m['mean_cca'] - runs[run][ep]))
            n += 1
        print(f'{run}: {n} checkpoints ({cname}), max |mean CCA - stored| = '
              f'{worst:.2e}', flush=True)
    fout.close()


if __name__ == '__main__':
    main()
