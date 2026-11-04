"""T2 probe caches: hard-rig (P1_S100) dist-matched probe sets for the
mu-boundary points 0.4 / 0.7 / 0.9, following the score_rigd.py::rig_sets
pattern exactly (4000/2000, bimodal at the point's mu, N_spots from the
run's own config). Seed bases are new per point and recorded:
mu04 20260940/41, mu07 20260942/43, mu09 20260944/45. Cached as
rigd_p1s100_mu{04,07,09}.pt. Idempotent."""
import glob
import os
import sys

import torch
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
sys.path.insert(0, _HERE)
import wp1_probe_driver as drv  # noqa: E402

CACHE = paths.CACHE
POINTS = {'mu04': (0.4, 20260940, 20260941),
          'mu07': (0.7, 20260942, 20260943),
          'mu09': (0.9, 20260944, 20260945)}

for tag, (mu, s_tr, s_ev) in POINTS.items():
    path = os.path.join(CACHE, f'rigd_p1s100_{tag}.pt')
    if os.path.exists(path):
        print(f'{tag}: exists, skipping')
        continue
    d = glob.glob(os.path.join(drv.ROOT,
                               f'rigd_lejp_p1s100_{tag}_nano_1xxx*'))[0]
    cfg = yaml.safe_load(open(os.path.join(d, 'config.yml')))
    old = drv.PROBE_N_SPOTS
    n_spots = cfg['data']['calo_cfg']['N_spots_per_layer']
    drv.PROBE_N_SPOTS = n_spots
    try:
        itr, ztr, etr = drv.gen_probe_set(cfg, 'bimodal', mu, 4000, s_tr)
        iev, zev, eev = drv.gen_probe_set(cfg, 'bimodal', mu, 2000, s_ev)
    finally:
        drv.PROBE_N_SPOTS = old
    mu_e, sd_e = float(etr.mean()), float(etr.std()) + 1e-8
    torch.save({'imgs_tr': itr, 'imgs_ev': iev,
                'y_tr': torch.cat([ztr, ((etr - mu_e) / sd_e)[:, None]], dim=1),
                'y_ev': torch.cat([zev, ((eev - mu_e) / sd_e)[:, None]], dim=1)},
               path)
    print(f'{tag}: mu={mu} N_spots={n_spots} seeds {s_tr}/{s_ev} '
          f'-> {os.path.basename(path)}  z_std={float(ztr.std()):.4f} '
          f'z_mean={float(ztr.mean()):+.4f}')
