"""Dense CCA trajectories for the six-seed extension (tiers 2 and 3, seeds
4-6, submitted 2026-10-04).

Same scoring as each condition's seeds 1-3 in results/knee_trajectories.csv:
encode_meanpool + wp1_metric_row, one row per saved checkpoint, on the
condition's cache as chosen by knee_dir_sweep.cache_for (imported, unchanged):
mu runs on rigd_p1s100_mu0{4,7,9}.pt, N runs on rig_P1_Enar_S{30,300}_tda.pt,
refresh runs on rig_P1_Enar_S100.pt. Writes a NEW file,
results/knee_traj_seedext.csv; knee_trajectories.csv is only read.

  --check   re-score seed 1 of every condition at its first and last
            checkpoint and compare with the stored rows (writes nothing).
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
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
STORED = os.path.join(paths.CSV, 'knee_trajectories.csv')
OUT = os.path.join(paths.CSV, 'knee_traj_seedext.csv')
PREFIXES = (['rigb_lejp_p1s100_p384_ou09_e1000'] +
            [f'rigd_lejp_p1s100_mu0{m}_{w}' for m in (4, 7, 9) for w in ('p384', 'nano')] +
            ['rigb_lejp_p1s30_p384', 'rigb_lejp_p1s300_p384'])
COLS = ['run', 'epoch', 'cache', 'mean_cca', 'cca_c1', 'cca_c2', 'orth_err',
        'ridge_r2_z1', 'ridge_r2_z2', 'ridge_r2_E']
_sets = {}


def sets_for(cname):
    if cname not in _sets:
        _sets[cname] = torch.load(os.path.join(CACHE, cname), map_location='cpu',
                                  weights_only=True)
    return _sets[cname]


def run_dir(prefix, s):
    hits = glob.glob(os.path.join(drv.ROOT, f'{prefix}_{s}xxx*'))
    assert len(hits) == 1, (prefix, s, hits)
    return hits[0]


def ckpts(d, cfg):
    tag = cfg['logging']['write_tag']
    return sorted(glob.glob(os.path.join(d, f'{tag}-ep*.pth.tar')),
                  key=lambda p: int(p.split('-ep')[-1].split('.')[0]))


def score(cfg, ck, sets):
    enc = drv.encoder_from_ckpt(cfg, ck)
    return wp1_metric_row(drv.encode_meanpool(enc, sets['imgs_tr']), sets['y_tr'],
                          drv.encode_meanpool(enc, sets['imgs_ev']), sets['y_ev'])


def check():
    stored = {}
    for r in csv.DictReader(open(STORED)):
        stored[(r['run'], int(r['epoch']))] = float(r['mean_cca'])
    worst = 0.0
    for p in PREFIXES:
        d = run_dir(p, 1)
        run = os.path.basename(d)
        cname = cache_for(run.split('xxx')[0])
        cfg = yaml.safe_load(open(os.path.join(d, 'config.yml')))
        cks = ckpts(d, cfg)
        for ck in (cks[0], cks[-1]):
            ep = int(ck.split('-ep')[-1].split('.')[0])
            m = score(cfg, ck, sets_for(cname))['mean_cca']
            st = stored[(run, ep)]
            worst = max(worst, abs(m - st))
            print(f'{run:44s} {cname:26s} ep{ep:<5d} rescored {m:.6f} stored {st:.6f}',
                  flush=True)
    print(f'max |rescored - stored| = {worst:.2e}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    if a.check:
        check()
        return
    done = set()
    if os.path.exists(OUT):
        done = {(r['run'], int(r['epoch'])) for r in csv.DictReader(open(OUT))}
    new = not os.path.exists(OUT)
    fout = open(OUT, 'a', newline='')
    w = csv.writer(fout, lineterminator='\n')
    if new:
        w.writerow(COLS)
    for p in PREFIXES:
        for s in (4, 5, 6):
            d = run_dir(p, s)
            run = os.path.basename(d)
            cname = cache_for(run.split('xxx')[0])
            cfg = yaml.safe_load(open(os.path.join(d, 'config.yml')))
            for ck in ckpts(d, cfg):
                ep = int(ck.split('-ep')[-1].split('.')[0])
                if (run, ep) in done:
                    continue
                m = score(cfg, ck, sets_for(cname))
                w.writerow([run, ep, cname, f"{m['mean_cca']:.6f}",
                            f"{m['cca_per_component'][0]:.6f}",
                            f"{m['cca_per_component'][1]:.6f}",
                            f"{m['orth_err']:.6f}", f"{m['ridge_r2_z1']:.6f}",
                            f"{m['ridge_r2_z2']:.6f}", f"{m['ridge_r2_E']:.6f}"])
                fout.flush()
            print(f'{run}: done ({cname})', flush=True)
    fout.close()


if __name__ == '__main__':
    main()
