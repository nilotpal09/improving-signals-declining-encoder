"""Probe caches for the element-count runs (N = 30, 300, 1000 elements per layer):
rig_P1_Enar_S{N}_tda.pt, each matched to its rung's element count.

The original builder was not kept. This rebuilds the caches from the seed bases
recorded when they were made (N=30 20260980/81, N=300 20260982/83, N=1000
20260984/85), with the construction of knee_t2_caches.py: 4000 training and 2000
evaluation events from wp1_probe_driver.gen_probe_set, Gaussian latents, the element
count from the rung's own run config; targets (z1, z2, standardized log E).
The rebuilt files are byte-identical to the released ones. Idempotent: existing
files are kept.
"""
import os
import sys

import torch
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
sys.path.insert(0, _HERE)
import wp1_probe_driver as drv  # noqa: E402

RUNGS = {30: ('rigb_lejp_p1s30_p384_1', 20260980, 20260981),
         300: ('rigb_lejp_p1s300_p384_1', 20260982, 20260983),
         1000: ('rigb_lejp_p1s1000_p384_1', 20260984, 20260985)}


def build(n):
    path = os.path.join(paths.CACHE, f'rig_P1_Enar_S{n}_tda.pt')
    if os.path.exists(path):
        print(f'N={n}: exists, skipping')
        return
    run, s_tr, s_ev = RUNGS[n]
    cfg = yaml.safe_load(open(os.path.join(paths.REPO, 'configs', f'{run}.yaml')))
    assert cfg['data']['calo_cfg']['N_spots_per_layer'] == n, run
    old = drv.PROBE_N_SPOTS
    drv.PROBE_N_SPOTS = n
    try:
        itr, ztr, etr = drv.gen_probe_set(cfg, 'gaussian', None, 4000, s_tr)
        iev, zev, eev = drv.gen_probe_set(cfg, 'gaussian', None, 2000, s_ev)
    finally:
        drv.PROBE_N_SPOTS = old
    mu_e, sd_e = float(etr.mean()), float(etr.std()) + 1e-8
    torch.save({'imgs_tr': itr, 'imgs_ev': iev,
                'y_tr': torch.cat([ztr, ((etr - mu_e) / sd_e)[:, None]], dim=1),
                'y_ev': torch.cat([zev, ((eev - mu_e) / sd_e)[:, None]], dim=1)}, path)
    print(f'N={n}: seeds {s_tr}/{s_ev} -> {os.path.basename(path)}')


if __name__ == '__main__':
    for n in RUNGS:
        build(n)
