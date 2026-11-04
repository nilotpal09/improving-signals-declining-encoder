"""LiDAR monitor trajectories on all knee-study runs, both campaigns.

Implementation IMPORTED from the monitors paper's own code
(copied verbatim to analysis/lidar_metric.py): lidar() and its feature path are
used as-is, not reimplemented. Replica convention is theirs too — M=64
truths x K=8 replicas, generate_replicas(normalize=True) with FULL
stochasticity (no fixed_spot_seed), seed = 10007*ps + n with ps=0.

Two spot rungs, both reported (noted in the record before this
ran): N=100 rig-matched primary, N=1000 their literal LIDAR_N constant.

Warmup region included so the series is complete. Output:
results/knee_lidar.csv, idempotent by (run, epoch, n_spots).
"""
import csv
import glob
import os
import sys

import torch
import torch.nn.functional as F
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
_COND = paths.TRAIN
sys.path.insert(0, _HERE)
sys.path.insert(0, _COND)

import wp1_probe_driver as drv  # noqa: E402
from calo_collator import make_collator  # noqa: E402

from lidar_metric import lidar  # noqa: E402  (the monitors paper's own lidar())

CACHE = paths.CACHE
OUT = os.path.join(paths.CSV, 'knee_lidar.csv')
M, K = 64, 8                      # V.M_TRUTHS, V.K_REPLICAS
RUNGS = (100, 1000)               # rig-matched primary, their LIDAR_N
DEV = 'cuda:0'

RUNS = []
RUNS += [f'rigb_lejp_p1s100_nano_{s}xxx*' for s in range(1, 10)]
RUNS += ['rigb_lejp_p1s100_long_nano_1xxx*']
RUNS += [f'rigd_lejp_p1s100_mu0{m}_nano_{s}xxx*' for m in ('4', '7', '9') for s in (1, 2, 3)]
RUNS += [f'rigd_lejp_p1s100_mu097_nano_{s}xxx*' for s in (1, 2, 3)]
RUNS += ['rigd_lejp_p1s100_mu097_long_nano_1xxx*']
RUNS += [f'rigb_mae_p1s100_long_nano_{s}xxx*' for s in (1, 2, 3)]
RUNS += [f'rigb_lejp_p1s100_tiny_{s}xxx*' for s in (1, 2)]
RUNS += [f'rigb_lejp_p1s100_p384_{s}xxx*' for s in range(1, 11)]
RUNS += [f'rigd_lejp_p1s100_mu0{m}_p384_{s}xxx*' for m in ('4', '7', '9') for s in (1, 2, 3)]
RUNS += [f'rigd_lejp_p1s100_mu097_p384_{s}xxx*' for s in (1, 2, 3, 4)]
RUNS += [f'rigb_lejp_p1s100_nano_{s}xxx*' for s in (10, 11, 12)]  # runs added 2026-10-13


def replica_set(cfg, n_spots):
    """Their convention: full stochasticity, seed = 10007*0 + n."""
    path = os.path.join(CACHE, f'knee_lidar_N{n_spots}.pt')
    if os.path.exists(path):
        return torch.load(path, map_location='cpu', weights_only=True)
    ccfg = {k: v for k, v in cfg['data']['calo_cfg'].items()
            if k not in ('lejepa_latents', 'paired_fixed_long')}
    coll = make_collator(ccfg, dict(cfg['data']['generator_cfg']), n_spots,
                         cfg['data'].get('norm') or {})
    _, reps = coll.generate_replicas(batch_size=M, n_replicas=K,
                                     seed=10007 * 0 + n_spots, normalize=True)
    imgs = reps.reshape(M * K, *reps.shape[2:]).to('cpu', torch.float16)
    torch.save(imgs, path)
    del coll
    torch.cuda.empty_cache()
    return imgs


@torch.no_grad()
def lidar_of(enc, imgs, chunk=64):
    """Their monitors() feature path: tokens -> LayerNorm -> mean over tokens."""
    out = []
    enc.eval()
    for i in range(0, imgs.size(0), chunk):
        h = enc(imgs[i:i + chunk].to(DEV, torch.float32)).float()
        h = F.layer_norm(h, (h.size(-1),))
        out.append(h.mean(dim=1))
    feats_event = torch.cat(out, dim=0)
    return lidar(feats_event.reshape(M, K, -1))


def main():
    done = set()
    if os.path.exists(OUT):
        done = {(r['run'], int(r['epoch']), int(r['n_spots']))
                for r in csv.DictReader(open(OUT))}
    new = not os.path.exists(OUT)
    fout = open(OUT, 'a', newline='')
    w = csv.writer(fout, lineterminator='\n')
    if new:
        w.writerow(['run', 'epoch', 'n_spots', 'lidar'])

    cfg0 = yaml.safe_load(open(os.path.join(
        glob.glob(os.path.join(drv.ROOT, RUNS[0]))[0], 'config.yml')))
    IMGS = {n: replica_set(cfg0, n) for n in RUNGS}

    for pat in RUNS:
        hits = glob.glob(os.path.join(drv.ROOT, pat))
        if not hits:
            print(f'skip (no dir): {pat}', flush=True)
            continue
        d = hits[0]
        run = os.path.basename(d)
        cfg = yaml.safe_load(open(os.path.join(d, 'config.yml')))
        tag = cfg['logging']['write_tag']
        cks = sorted(glob.glob(os.path.join(d, f'{tag}-ep*.pth.tar')),
                     key=lambda p: int(p.split('-ep')[-1].split('.')[0]))
        for ck in cks:
            ep = int(ck.split('-ep')[-1].split('.')[0])
            todo = [n for n in RUNGS if (run, ep, n) not in done]
            if not todo:
                continue
            enc = drv.encoder_from_ckpt(cfg, ck)
            for n in todo:
                w.writerow([run, ep, n, f'{lidar_of(enc, IMGS[n]):.6f}'])
            fout.flush()
            del enc
        print(f'{run.split("xxx")[0]:38s} done ({len(cks)} ckpts)', flush=True)
    fout.close()
    print('DONE knee_lidar')


if __name__ == '__main__':
    main()
