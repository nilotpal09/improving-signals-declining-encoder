"""Score the hard-rig dial-endpoint runs (rigd_*: mu=0.97 on P5_S100 /
P1_S100) on RIG- AND DIST-MATCHED probe sets (bimodal mu=0.97, generated
here, cached). Also scores the matched random-init floor once per rig.
Appends to csv/rigd_scores.csv. Idempotent. Run whenever checkpoints exist.
"""
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

CACHE = paths.CACHE
OUT = os.path.join(paths.CSV, 'rigd_scores.csv')
COLS = ['run', 'epoch', 'encoder_key', 'rig', 'mean_cca', 'cca_c1', 'cca_c2',
        'orth_err', 'ridge_r2_z1', 'ridge_r2_z2', 'ridge_r2_E', 'timestamp']


def rig_sets(cfg, rig):
    """4k/2k probe sets at the run's OWN rig geometry and dist (mu=0.97)."""
    path = os.path.join(CACHE, f'rigd_{rig}_mu097.pt')
    if os.path.exists(path):
        return torch.load(path, map_location='cpu', weights_only=True)
    old = drv.PROBE_N_SPOTS
    drv.PROBE_N_SPOTS = cfg['data']['calo_cfg']['N_spots_per_layer']
    try:
        itr, ztr, etr = drv.gen_probe_set(cfg, 'bimodal', 0.97, 4000, 20260930)
        iev, zev, eev = drv.gen_probe_set(cfg, 'bimodal', 0.97, 2000, 20260931)
    finally:
        drv.PROBE_N_SPOTS = old
    # multi-particle rigs: the driver returns per-PARTICLE lnE; the event
    # target is log total energy (logsumexp over the event's particles)
    if etr.numel() != ztr.shape[0]:
        etr = torch.logsumexp(etr.view(ztr.shape[0], -1), dim=1)
        eev = torch.logsumexp(eev.view(zev.shape[0], -1), dim=1)
    mu_e, sd_e = float(etr.mean()), float(etr.std()) + 1e-8
    d = {'imgs_tr': itr, 'imgs_ev': iev,
         'y_tr': torch.cat([ztr, ((etr - mu_e) / sd_e)[:, None]], dim=1),
         'y_ev': torch.cat([zev, ((eev - mu_e) / sd_e)[:, None]], dim=1)}
    torch.save(d, path)
    return d


def emit(run, ep, key, rig, m):
    new = not os.path.exists(OUT)
    with open(OUT, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS, lineterminator='\n')
        if new:
            w.writeheader()
        w.writerow({'run': run, 'epoch': ep, 'encoder_key': key, 'rig': rig,
                    'mean_cca': f"{m['mean_cca']:.6f}",
                    'cca_c1': f"{m['cca_per_component'][0]:.6f}",
                    'cca_c2': f"{m['cca_per_component'][1]:.6f}",
                    'orth_err': f"{m['orth_err']:.6f}",
                    'ridge_r2_z1': f"{m['ridge_r2_z1']:.6f}",
                    'ridge_r2_z2': f"{m['ridge_r2_z2']:.6f}",
                    'ridge_r2_E': f"{m['ridge_r2_E']:.6f}",
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')})
    print(f"{run.split('xxx')[0]:26s} ep{ep:>3} {key:14s} "
          f"cca={m['mean_cca']:.4f} orth={m['orth_err']:.3f} "
          f"r2z=({m['ridge_r2_z1']:.3f},{m['ridge_r2_z2']:.3f})")


done = set()
if os.path.exists(OUT):
    done = {(r['run'], r['epoch'], r['encoder_key'])
            for r in csv.DictReader(open(OUT))}

for d in sorted(glob.glob(os.path.join(drv.ROOT, 'rigd_*_nano_1xxx*'))):
    run = os.path.basename(d)
    rig = run.split('xxx')[0].split('_')[2]
    cfg = yaml.safe_load(open(os.path.join(d, 'config.yml')))
    tag = cfg['logging']['write_tag']
    sets = rig_sets(cfg, rig)
    if (f'floor_{rig}', '0', 'random_init') not in done:
        torch.manual_seed(0)
        enc0 = drv.build_encoder_from_cfg(cfg)
        m = wp1_metric_row(drv.encode_meanpool(enc0, sets['imgs_tr']),
                           sets['y_tr'],
                           drv.encode_meanpool(enc0, sets['imgs_ev']),
                           sets['y_ev'])
        emit(f'floor_{rig}', '0', 'random_init', rig, m)
        done.add((f'floor_{rig}', '0', 'random_init'))
    cks = sorted(glob.glob(os.path.join(d, f'{tag}-ep*.pth.tar')),
                 key=lambda p: int(p.split('-ep')[-1].split('.')[0]))
    if not cks:
        continue
    ck = cks[-1]
    ep = str(int(ck.split('-ep')[-1].split('.')[0]))
    if (run, ep, 'target_encoder') in done:
        continue
    enc = drv.encoder_from_ckpt(cfg, ck)
    m = wp1_metric_row(drv.encode_meanpool(enc, sets['imgs_tr']),
                       sets['y_tr'],
                       drv.encode_meanpool(enc, sets['imgs_ev']),
                       sets['y_ev'])
    emit(run, ep, 'target_encoder', rig, m)
