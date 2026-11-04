"""S1: dense CCA trajectories for the 8 knee-study runs + onset under the
two pre-printed rules (definitions of 2026-08-19). Writes
results/knee_trajectories.csv (run, epoch, mean_cca, cca_c1, cca_c2, orth_err)
and prints the per-run onset table. Idempotent via the CSV."""
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

ROOT = drv.ROOT
CACHE = paths.CACHE
OUT = os.path.join(paths.CSV, 'knee_trajectories.csv')
RUNS = {  # pattern -> probe cache
    'rigb_lejp_p1s100_nano_1xxx*': 'rig_P1_Enar_S100.pt',
    'rigb_lejp_p1s100_nano_2xxx*': 'rig_P1_Enar_S100.pt',
    'rigb_lejp_p1s100_nano_3xxx*': 'rig_P1_Enar_S100.pt',
    'rigb_lejp_p1s100_long_nano_1xxx*': 'rig_P1_Enar_S100.pt',
    'rigd_lejp_p1s100_mu097_nano_1xxx*': 'rigd_p1s100_mu097.pt',
    'rigd_lejp_p1s100_mu097_nano_2xxx*': 'rigd_p1s100_mu097.pt',
    'rigd_lejp_p1s100_mu097_nano_3xxx*': 'rigd_p1s100_mu097.pt',
    'rigd_lejp_p1s100_mu097_long_nano_1xxx*': 'rigd_p1s100_mu097.pt',
}
# Knee-study trainings (T1 seeds 4-9, T2 mu boundary, T3 MAE control). Same
# rules, same caches convention; patterns with no run dir yet are skipped, so
# this is safe to run while the fleet is still in the queue.
RUNS.update({f'rigb_lejp_p1s100_nano_{s}xxx*': 'rig_P1_Enar_S100.pt'
             for s in range(4, 10)})
RUNS.update({f'rigd_lejp_p1s100_{m}_nano_{s}xxx*': f'rigd_p1s100_{m}.pt'
             for m in ('mu04', 'mu07', 'mu09') for s in (1, 2, 3)})
RUNS.update({f'rigb_mae_p1s100_long_nano_{s}xxx*': 'rig_P1_Enar_S100.pt'
             for s in (1, 2, 3)})

def onset_A(t):
    # Guard for runs still in flight: the rule is unchanged, but a run with
    # no grid epoch in the [20, 70] reference window has no threshold to
    # define, so it reports "insufficient data" instead of crashing.
    ref = sorted(e for e in t if 20 <= e <= 70)
    if not ref or not any(e > 70 for e in t):
        return None, float('nan')
    vals = sorted(t[e] for e in ref)
    med = vals[len(vals) // 2]
    mad = sorted(abs(v - med) for v in vals)[len(vals) // 2]
    thr = med - max(5 * mad, 0.02)
    eps = sorted(e for e in t if e > 70)
    for i, e in enumerate(eps):
        if t[e] < thr and all(t[e2] < thr for e2 in eps[i:]):
            return e, thr
    return None, thr


def onset_B(t):
    eps = sorted(t)
    best, arg = 0.0, None
    for a, b in zip(eps, eps[1:]):
        drop = t[a] - t[b]
        if drop > best:
            best, arg = drop, b
    return (arg, best) if best > 0.03 else (None, best)



def main():
    done = set()
    if os.path.exists(OUT):
        done = {(r['run'], int(r['epoch'])) for r in csv.DictReader(open(OUT))}
    new = not os.path.exists(OUT)
    fout = open(OUT, 'a', newline='')
    w = csv.writer(fout, lineterminator='\n')
    if new:
        w.writerow(['run', 'epoch', 'mean_cca', 'cca_c1', 'cca_c2', 'orth_err'])

    traj = {}
    for pat, cache in RUNS.items():
        hits = glob.glob(os.path.join(ROOT, pat))
        if not hits:
            print(f'skip (no run dir yet): {pat}')
            continue
        d = hits[0]
        run = os.path.basename(d)
        cfg = yaml.safe_load(open(os.path.join(d, 'config.yml')))
        sets = torch.load(os.path.join(CACHE, cache), map_location='cpu',
                          weights_only=True)
        tag = cfg['logging']['write_tag']      # 'mae' on the T3 control arm
        cks = sorted(glob.glob(os.path.join(d, f'{tag}-ep*.pth.tar')),
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
                        f"{m['orth_err']:.6f}"])
            fout.flush()
    fout.close()

    for r in csv.DictReader(open(OUT)):
        traj.setdefault(r['run'], {})[int(r['epoch'])] = float(r['mean_cca'])



    print(f"{'run':44s} {'R_A':>6s} {'thr':>7s} {'R_B':>6s} {'maxdrop':>8s} agree")
    for run, t in sorted(traj.items()):
        a, thr = onset_A(t)
        b, drop = onset_B(t)
        if thr != thr:            # NaN: run still in flight, no reference window
            print(f"{run.split('xxx')[0]:44s} {'--':>6s} {'--':>7s} {'--':>6s} "
                  f"{'--':>8s} incomplete ({len(t)} ckpts, max ep {max(t)})")
            continue
        agree = ('YES' if a is not None and b is not None and abs(a - b) <= 10
                 else ('n/a' if a is None and b is None else 'NO'))
        print(f"{run.split('xxx')[0]:44s} {str(a):>6s} {thr:7.3f} {str(b):>6s} "
              f"{drop:8.3f} {agree}")


if __name__ == '__main__':
    main()
