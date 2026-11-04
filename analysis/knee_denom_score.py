"""Per-checkpoint scores for the 16 denominator-completion runs (2026-10-12;
the runs are listed in PLAN below, their configs are in configs/).

Same scoring as knee_seedext_score.py and knee_ridge_backfill_score.py:
encode_meanpool + wp1_metric_row, one row per saved checkpoint, on the
condition's cache from knee_dir_sweep.cache_for (imported, unchanged).
Writes a NEW file, results/knee_traj_denom.csv (appends, skips rows already
there, so it can be rerun while runs are still training). A checkpoint
written in the last 5 minutes is skipped, so a file still being saved is
never read.

  --check   re-score one existing run of each condition at its first and
            last checkpoint and compare ridge R2 (z1, z2, log E) with the
            values knee_onsets_avgr2.load() uses (writes nothing).
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import csv  # noqa: E402
import glob  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
import yaml  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
sys.path.insert(0, _HERE)
import wp1_probe_driver as drv  # noqa: E402
from knee_dir_sweep import cache_for  # noqa: E402
from wp1_metrics import wp1_metric_row  # noqa: E402

CACHE = paths.CACHE
OUT = os.path.join(paths.CSV, 'knee_traj_denom.csv')
# (prefix, new seeds, an existing run of the same condition for --check)
PLAN = [('rigb_lejp_p1s100_nano', (10, 11, 12), 'rigb_lejp_p1s100_nano_4'),
        ('rigd_lejp_p1s100_mu097_nano', (4, 5, 6, 7, 8), 'rigd_lejp_p1s100_mu097_long_nano_1'),
        ('rigd_lejp_p1s100_mu097_p384', (5, 6), 'rigd_lejp_p1s100_mu097_p384_1'),
        ('rigb_lejp_p1s100_p384_ou09', (4, 5, 6), 'rigb_lejp_p1s100_p384_ou09_1'),
        ('rigb_lejp_p1s100_p384_ou05', (4, 5, 6), 'rigb_lejp_p1s100_p384_ou05_1')]
COLS = ['run', 'epoch', 'cache', 'mean_cca', 'cca_c1', 'cca_c2', 'cca_c3',
        'orth_err', 'ridge_r2_z1', 'ridge_r2_z2', 'ridge_r2_E']
SETTLE_S = 300
_sets = {}


def sets_for(cname):
    if cname not in _sets:
        _sets[cname] = torch.load(os.path.join(CACHE, cname), map_location='cpu',
                                  weights_only=True)
    return _sets[cname]


def run_dir(name):
    hits = glob.glob(os.path.join(drv.ROOT, f'{name}xxx*'))
    assert len(hits) <= 1, (name, hits)
    return hits[0] if hits else None


def ckpts(d, cfg):
    tag = cfg['logging']['write_tag']
    return sorted(glob.glob(os.path.join(d, f'{tag}-ep*.pth.tar')),
                  key=lambda p: int(p.split('-ep')[-1].split('.')[0]))


def epoch_of(ck):
    return int(ck.split('-ep')[-1].split('.')[0])


def score(cfg, ck, sets):
    enc = drv.encoder_from_ckpt(cfg, ck)
    return wp1_metric_row(drv.encode_meanpool(enc, sets['imgs_tr']), sets['y_tr'],
                          drv.encode_meanpool(enc, sets['imgs_ev']), sets['y_ev'])


def check():
    from knee_onsets_avgr2 import load
    R, _ = load()
    worst = 0.0
    for _, _, ref in PLAN:
        d = run_dir(ref)
        cname = cache_for(ref)
        cfg = yaml.safe_load(open(os.path.join(d, 'config.yml')))
        cks = ckpts(d, cfg)
        for ck in (cks[0], cks[-1]):
            ep = epoch_of(ck)
            m = score(cfg, ck, sets_for(cname))
            new = (m['ridge_r2_z1'], m['ridge_r2_z2'], m['ridge_r2_E'])
            st = R[ref][ep]
            dev = max(abs(a - b) for a, b in zip(new, st))
            worst = max(worst, dev)
            print(f'{ref:40s} {cname:24s} ep{ep:<4d} rescored '
                  + ' '.join(f'{v:.6f}' for v in new) + '  stored '
                  + ' '.join(f'{v:.6f}' for v in st) + f'  max dev {dev:.1e}', flush=True)
    print(f'max |rescored - stored| ridge R2 = {worst:.2e}')


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
    now = time.time()
    for prefix, seeds, _ in PLAN:
        for s in seeds:
            d = run_dir(f'{prefix}_{s}')
            if d is None:
                print(f'{prefix}_{s}: no run dir yet', flush=True)
                continue
            run = os.path.basename(d)
            cname = cache_for(run.split('xxx')[0])
            cfg = yaml.safe_load(open(os.path.join(d, 'config.yml')))
            n = 0
            cks = ckpts(d, cfg)
            for ck in cks:
                ep = epoch_of(ck)
                if (run, ep) in done or now - os.path.getmtime(ck) < SETTLE_S:
                    continue
                m = score(cfg, ck, sets_for(cname))
                c = m['cca_per_component']
                w.writerow([run, ep, cname, f"{m['mean_cca']:.6f}", f'{c[0]:.6f}',
                            f'{c[1]:.6f}', f'{c[2]:.6f}', f"{m['orth_err']:.6f}",
                            f"{m['ridge_r2_z1']:.6f}", f"{m['ridge_r2_z2']:.6f}",
                            f"{m['ridge_r2_E']:.6f}"])
                fout.flush()
                n += 1
            last = epoch_of(cks[-1]) if cks else None
            print(f'{run}: {n} new checkpoints scored, latest saved epoch {last} ({cname})',
                  flush=True)
    fout.close()


if __name__ == '__main__':
    main()
