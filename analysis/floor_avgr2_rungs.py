"""Untrained-encoder floor on each element-count rung's own evaluation set, in
the readability measure of 2026-10-07 (ridge R2 of z1, z2, log E and their
average), for Figure 3(c) (the figures use average R2).

House floor convention (nspots_floor_sweep.py): torch.manual_seed(s) for
s in (0, 1), drv.build_encoder_from_cfg, encode_meanpool, wp1_metric_row.
The encoder geometry comes from each rung's own run config (seed 1 run). Caches
from knee_dir_sweep.cache_for (imported, unchanged): N = 30, 300, 1000 use the
rung caches the pre-measured floors 0.5738 / 0.8103 / 0.9111 were quoted on;
N = 100 uses rig_P1_Enar_S100.pt, the cache its runs are scored on (the
installed figure's N = 100 floor came from a separately generated set,
nspots_sweep.csv). Printed check: mean CCA against the quoted floors.
Writes a NEW file, results/floor_avgr2_rungs.csv (refuses to overwrite).
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
OUT = os.path.join(paths.CSV, 'floor_avgr2_rungs.csv')
RUNG_RUN = {30: 'rigb_lejp_p1s30_p384_1', 100: 'rigb_lejp_p1s100_p384_1',
            300: 'rigb_lejp_p1s300_p384_1', 1000: 'rigb_lejp_p1s1000_p384_1'}
QUOTED = {30: 0.5738, 300: 0.8103, 1000: 0.9111}
COLS = ['n_spots', 'cache', 'seed', 'mean_cca', 'ridge_r2_z1', 'ridge_r2_z2', 'ridge_r2_E', 'avg_r2']


def main():
    assert not os.path.exists(OUT), f'refusing to overwrite {OUT}'
    rows = []
    for n, run in RUNG_RUN.items():
        d = glob.glob(os.path.join(drv.ROOT, run + 'xxx*'))
        assert len(d) == 1, (run, d)
        cfg = yaml.safe_load(open(os.path.join(d[0], 'config.yml')))
        cname = cache_for(run)
        sets = torch.load(os.path.join(CACHE, cname), map_location='cpu', weights_only=True)
        for s in (0, 1):
            torch.manual_seed(s)
            enc = drv.build_encoder_from_cfg(cfg)
            m = wp1_metric_row(drv.encode_meanpool(enc, sets['imgs_tr']), sets['y_tr'],
                               drv.encode_meanpool(enc, sets['imgs_ev']), sets['y_ev'])
            r2 = [m['ridge_r2_z1'], m['ridge_r2_z2'], m['ridge_r2_E']]
            rows.append([n, cname, s, m['mean_cca']] + r2 + [sum(r2) / 3])
            print(f'N={n:4d} {cname:26s} seed {s}: mean CCA {m["mean_cca"]:.4f}  R2 z1 {r2[0]:.4f} '
                  f'z2 {r2[1]:.4f} logE {r2[2]:.4f}  avg {sum(r2) / 3:.4f}', flush=True)
        mc = [r[3] for r in rows if r[0] == n]
        if n in QUOTED:
            print(f'   quoted floor {QUOTED[n]:.4f}; seed 0 {mc[0]:.4f}, seed mean {sum(mc) / 2:.4f}')
    with open(OUT, 'w', newline='') as fh:
        w = csv.writer(fh, lineterminator='\n')
        w.writerow(COLS)
        for r in rows:
            w.writerow([r[0], r[1], r[2]] + [f'{v:.6f}' for v in r[3:]])
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
