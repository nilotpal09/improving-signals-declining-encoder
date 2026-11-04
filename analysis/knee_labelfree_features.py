"""Pooled features for the label-free metric pass (2026-09-23). Reads
checkpoints, writes new files only; no training.

Feature path: exactly knee_s7_mech.py's (wp1_probe_driver.encoder_from_ckpt
on the checkpoint named by the run's write_tag, then
wp1_probe_driver.encode_meanpool: encoder tokens -> LayerNorm over the
feature channels -> mean over tokens, fp32). Events: the run's own
dial-matched probe cache (knee_dir_sweep.cache_for), fitting split (4,000
events) AND evaluation split (2,000 events); the evaluation split is saved
so readability (ridge R^2, CCA) can be computed at every checkpoint of
every run without a GPU.

Checkpoints: every (run, epoch) with an N = 100 row in results/knee_lidar.csv
(58 runs, 2,114 checkpoints).

Storage: float32, not fp16. fp16 keeps ~3 significant digits; its rounding
noise (variance ~1e-8 per entry at these feature magnitudes) would sit on
top of the smallest covariance eigenvalues, which the condition number and
the alpha-ReQ fit read. The fp16 effect is measured in the metric tests.

Output: one file per run,
  paths.WORK/labelfree_feats/<run>.pt
  {'run', 'cache', 'epochs' (list), 'feats_tr' (n_ckpt, 4000, D) float32,
   'feats_ev' (n_ckpt, 2000, D) float32, 'write_tag', 'script'}
Refuses to overwrite; skips runs whose file exists (resumable by run).

Usage (from analysis/):
  CUDA_VISIBLE_DEVICES=0 python knee_labelfree_features.py --part 0 --nparts 2
  CUDA_VISIBLE_DEVICES=1 python knee_labelfree_features.py --part 1 --nparts 2
"""
import sys

sys.dont_write_bytecode = True      # leave analysis/__pycache__ untouched

import argparse  # noqa: E402
import csv  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from collections import OrderedDict  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
sys.path.insert(0, _HERE)

CACHE_DIR = paths.CACHE
FEAT_DIR = os.path.join(paths.WORK, 'labelfree_feats')  # denom: 58 symlinks + 3 new
SRC = os.path.join(paths.CSV, 'knee_lidar.csv')


def plan():
    runs = OrderedDict()
    with open(SRC) as fh:
        for r in csv.DictReader(fh):
            if r['n_spots'] == '100':
                runs.setdefault(r['run'], set()).add(int(r['epoch']))
    return OrderedDict((k, sorted(v)) for k, v in runs.items())


def feat_path(run):
    return os.path.join(FEAT_DIR, f'{run}.pt')


def main():
    import torch
    import yaml
    import wp1_probe_driver as drv
    import knee_dir_sweep as DS

    ap = argparse.ArgumentParser()
    ap.add_argument('--part', type=int, default=0)
    ap.add_argument('--nparts', type=int, default=1)
    a = ap.parse_args()
    P = plan()
    assert len(P) == 61, len(P)  # denom 2026-10-13: 58 + nano_10-12
    runs = [r for i, r in enumerate(P) if i % a.nparts == a.part]
    os.makedirs(FEAT_DIR, exist_ok=True)
    print(f'GPU {torch.cuda.get_device_name(0)}; part {a.part}/{a.nparts}: {len(runs)} runs, '
          f'{sum(len(P[r]) for r in runs)} checkpoints', flush=True)
    caches = {}
    t_all = time.time()
    for run in runs:
        out = feat_path(run)
        if os.path.exists(out):
            print(f'{run}: exists, skip', flush=True)
            continue
        t0 = time.time()
        short = run.split('xxx')[0]
        cname = DS.cache_for(short)
        if cname not in caches:
            caches[cname] = torch.load(os.path.join(CACHE_DIR, cname), map_location='cpu',
                                       weights_only=True)
        sets = caches[cname]
        cfg = yaml.safe_load(open(os.path.join(drv.ROOT, run, 'config.yml')))
        tag = cfg['logging']['write_tag']
        ftr, fev = [], []
        for ep in P[run]:
            enc = drv.encoder_from_ckpt(cfg, os.path.join(drv.ROOT, run, f'{tag}-ep{ep}.pth.tar'))
            ftr.append(drv.encode_meanpool(enc, sets['imgs_tr']).float())
            fev.append(drv.encode_meanpool(enc, sets['imgs_ev']).float())
            del enc
        obj = {'run': run, 'cache': cname, 'epochs': list(P[run]), 'write_tag': tag,
               'feats_tr': torch.stack(ftr), 'feats_ev': torch.stack(fev),
               'script': 'knee_labelfree_features.py'}
        assert not os.path.exists(out)
        torch.save(obj, out + '.partial')
        os.replace(out + '.partial', out)
        print(f'{short:38s} {len(P[run])} ckpts D={obj["feats_tr"].shape[-1]} '
              f'({time.time() - t0:.0f}s; total {time.time() - t_all:.0f}s)', flush=True)
    print('DONE features part', a.part)


if __name__ == '__main__':
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        raise SystemExit('set CUDA_VISIBLE_DEVICES explicitly (shared GPUs)')
    main()
