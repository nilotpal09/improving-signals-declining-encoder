"""S7(i): full wp1_metric_row at every grid checkpoint of the 8 knee-study
runs against the rig-matched probe caches (S1 pairing, unchanged). Feeds
S5 (ridge_r2_E trajectory), S7 per-dimension CCA + orth trajectories.
Writes csv/knee_traj_full.csv. Idempotent via the CSV."""
import csv
import glob
import os
import sys

import torch
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
sys.path.insert(0, _HERE)
import wp1_probe_driver as drv  # noqa: E402
from wp1_metrics import wp1_metric_row  # noqa: E402

CACHE = paths.CACHE
OUT = os.path.join(paths.CSV, 'knee_traj_full.csv')
RUNS = {
    'rigb_lejp_p1s100_nano_1xxx*': 'rig_P1_Enar_S100.pt',
    'rigb_lejp_p1s100_nano_2xxx*': 'rig_P1_Enar_S100.pt',
    'rigb_lejp_p1s100_nano_3xxx*': 'rig_P1_Enar_S100.pt',
    'rigb_lejp_p1s100_long_nano_1xxx*': 'rig_P1_Enar_S100.pt',
    'rigd_lejp_p1s100_mu097_nano_1xxx*': 'rigd_p1s100_mu097.pt',
    'rigd_lejp_p1s100_mu097_nano_2xxx*': 'rigd_p1s100_mu097.pt',
    'rigd_lejp_p1s100_mu097_nano_3xxx*': 'rigd_p1s100_mu097.pt',
    'rigd_lejp_p1s100_mu097_long_nano_1xxx*': 'rigd_p1s100_mu097.pt',
}
COLS = ['run', 'epoch', 'mean_cca', 'cca_c1', 'cca_c2', 'cca_c3',
        'orth_err', 'orth_scale', 'ridge_r2_z1', 'ridge_r2_z2', 'ridge_r2_E']

def main():
    done = set()
    if os.path.exists(OUT):
        done = {(r['run'], int(r['epoch'])) for r in csv.DictReader(open(OUT))}
    new = not os.path.exists(OUT)
    fout = open(OUT, 'a', newline='')
    w = csv.writer(fout, lineterminator='\n')
    if new:
        w.writerow(COLS)

    for pat, cache in RUNS.items():
        d = glob.glob(os.path.join(drv.ROOT, pat))[0]
        run = os.path.basename(d)
        cfg = yaml.safe_load(open(os.path.join(d, 'config.yml')))
        sets = torch.load(os.path.join(CACHE, cache), map_location='cpu',
                          weights_only=True)
        cks = sorted(glob.glob(os.path.join(d, 'jepa-ep*.pth.tar')),
                     key=lambda p: int(p.split('-ep')[-1].split('.')[0]))
        for ck in cks:
            ep = int(ck.split('-ep')[-1].split('.')[0])
            if (run, ep) in done:
                continue
            enc = drv.encoder_from_ckpt(cfg, ck)
            m = wp1_metric_row(drv.encode_meanpool(enc, sets['imgs_tr']),
                               sets['y_tr'],
                               drv.encode_meanpool(enc, sets['imgs_ev']),
                               sets['y_ev'])
            w.writerow([run, ep, f"{m['mean_cca']:.6f}",
                        f"{m['cca_per_component'][0]:.6f}",
                        f"{m['cca_per_component'][1]:.6f}",
                        f"{m['cca_per_component'][2]:.6f}",
                        f"{m['orth_err']:.6f}", f"{m['orth_scale']:.6e}",
                        f"{m['ridge_r2_z1']:.6f}", f"{m['ridge_r2_z2']:.6f}",
                        f"{m['ridge_r2_E']:.6f}"])
            fout.flush()
            print(f"{run.split('xxx')[0]:36s} ep{ep:>3} cca3="
                  f"{m['cca_per_component'][2]:.4f} r2=({m['ridge_r2_z1']:.3f},"
                  f"{m['ridge_r2_z2']:.3f},{m['ridge_r2_E']:.3f})", flush=True)
    fout.close()
    print('DONE knee_traj_full')


if __name__ == '__main__':
    main()
