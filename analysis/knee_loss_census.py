"""Loss-anomaly census over all 51 runs, both campaigns.

Definitions fixed in the record before computing (2026-08-22):
  series   x(e) = log(per-epoch mean training loss), warmup e <= 20 masked
  spike    |r(e)| > 5 * MAD(r), r(e) = x(e) - median(x over [e-5,e+5] \ {e})
  level    final-epoch loss
  co-loc   for knee runs, min |spike epoch - onset|; for non-knee runs the
           same counts/densities, which are the control.
No cut, no verdict: counts, epochs, distances, densities, levels.

Writes results/knee_loss_census.csv and prints the table.
"""
import csv
import glob
import math
import os
import statistics
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
ROOT = paths.RUNS
OUT = os.path.join(paths.CSV, 'knee_loss_census.csv')
WARM, HALF, K = 20, 5, 5.0

# (label, dir prefix, loss file kind, onset, onset source) ; onset None = no knee
RUNS = []
_k21 = {1: (90, 'R_A'), 2: (110, 'R_A'), 3: (100, 'R_A')}
for s, (o, src) in _k21.items():
    RUNS.append((f'21x mu=0 s{s}', f'rigb_lejp_p1s100_nano_{s}xxx', 'lejepa', o, src))
RUNS.append(('21x mu=0 long', 'rigb_lejp_p1s100_long_nano_1xxx', 'lejepa', 160, 'R_A'))
for s, o, src in ((4, 80, 'R_A'), (5, None, None), (6, 70, 'R_B'), (7, 170, 'R_A'),
                  (8, 110, 'R_A'), (9, 120, 'R_A')):
    RUNS.append((f'21x mu=0 s{s}', f'rigb_lejp_p1s100_nano_{s}xxx', 'lejepa', o, src))
for m, tag, seeds in (('mu04', '21x mu=0.4', ((1, None, None), (2, None, None), (3, 130, 'R_A'))),
                      ('mu07', '21x mu=0.7', ((1, 140, 'R_A'), (2, 120, 'R_A'), (3, None, None))),
                      ('mu09', '21x mu=0.9', ((1, None, None), (2, None, None), (3, None, None)))):
    for s, o, src in seeds:
        RUNS.append((f'{tag} s{s}', f'rigd_lejp_p1s100_{m}_nano_{s}xxx', 'lejepa', o, src))
for s in (1, 2, 3):
    RUNS.append((f'21x mu=0.97 s{s}', f'rigd_lejp_p1s100_mu097_nano_{s}xxx', 'lejepa', None, None))
RUNS.append(('21x mu=0.97 long', 'rigd_lejp_p1s100_mu097_long_nano_1xxx', 'lejepa', None, None))
for s in (1, 2, 3):
    RUNS.append((f'21x MAE s{s}', f'rigb_mae_p1s100_long_nano_{s}xxx', 'mae', None, None))
for s in (1, 2):
    RUNS.append((f'21x tiny s{s}', f'rigb_lejp_p1s100_tiny_{s}xxx', 'lejepa',
                 90 if s == 2 else None, 'R_A' if s == 2 else None))
_k4 = {1: (50, 'R_B'), 2: (140, 'R_A'), 3: (90, 'R_A'), 4: (140, 'R_A'), 5: (60, 'R_B'),
       6: (70, 'R_B'), 7: (80, 'R_A'), 8: (90, 'R_A'), 9: (120, 'R_A'), 10: (None, None)}
for s, (o, src) in _k4.items():
    RUNS.append((f'4x mu=0 s{s}', f'rigb_lejp_p1s100_p384_{s}xxx', 'lejepa', o, src))
for m, tag, seeds in (('mu04', '4x mu=0.4', ((1, 80, 'R_A'), (2, 80, 'R_A'), (3, 80, 'R_A'))),
                      ('mu07', '4x mu=0.7', ((1, 90, 'R_A'), (2, None, None), (3, 90, 'R_A'))),
                      ('mu09', '4x mu=0.9', ((1, None, None), (2, None, None), (3, 130, 'R_B')))):
    for s, o, src in seeds:
        RUNS.append((f'{tag} s{s}', f'rigd_lejp_p1s100_{m}_p384_{s}xxx', 'lejepa', o, src))
for s in (1, 2, 3, 4):
    RUNS.append((f'4x mu=0.97 s{s}', f'rigd_lejp_p1s100_mu097_p384_{s}xxx', 'lejepa', None, None))


def series(prefix, kind):
    hits = glob.glob(os.path.join(ROOT, prefix + '*'))
    if not hits:
        return None
    fn = os.path.join(hits[0], 'mae_r0.csv' if kind == 'mae' else 'lejepa_diag_r0.csv')
    by = defaultdict(list)
    with open(fn) as fh:
        for r in csv.DictReader(fh):
            try:
                by[int(r['epoch'])].append(float(r['loss']))
            except (ValueError, KeyError):
                continue
    eps = sorted(by)
    return eps, [sum(by[e]) / len(by[e]) for e in eps]


def spikes(eps, loss):
    x = {e: math.log(v) for e, v in zip(eps, loss) if v > 0}
    ok = [e for e in sorted(x) if e > WARM]
    res = {}
    for e in ok:
        win = [x[k] for k in x if abs(k - e) <= HALF and k != e]
        if len(win) >= 4:
            res[e] = x[e] - statistics.median(win)
    if not res:
        return [], 0.0
    med = statistics.median(res.values())
    mad = statistics.median([abs(v - med) for v in res.values()]) or 1e-12
    return sorted(e for e, v in res.items() if abs(v - med) > K * mad), mad


def main():
    rows = []
    for label, prefix, kind, onset, src in RUNS:
        s = series(prefix, kind)
        if s is None:
            print(f'MISSING {label} ({prefix})')
            continue
        eps, loss = s
        sp, mad = spikes(eps, loss)
        span = max(eps) - WARM
        dens = 100.0 * len(sp) / span if span > 0 else float('nan')
        near = min((abs(e - onset) for e in sp), default=None) if onset else None
        rows.append(dict(run=label, onset='' if onset is None else onset,
                         onset_src=src or '', n_spikes=len(sp),
                         density_per100=round(dens, 2),
                         nearest_spike_to_onset='' if near is None else near,
                         final_loss=round(loss[-1], 5), max_ep=max(eps),
                         spike_epochs=' '.join(str(e) for e in sp)))

    with open(OUT, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)

    print(f"{'run':20s} {'onset':>6s} {'src':>4s} {'spikes':>7s} {'/100ep':>7s} "
          f"{'nearest':>8s} {'final loss':>11s}")
    for r in rows:
        print(f"{r['run']:20s} {str(r['onset']):>6s} {r['onset_src']:>4s} "
              f"{r['n_spikes']:>7d} {r['density_per100']:>7.2f} "
              f"{str(r['nearest_spike_to_onset']):>8s} {r['final_loss']:>11.5f}")

    kn = [r for r in rows if r['onset'] != '']
    nk = [r for r in rows if r['onset'] == '']
    print(f'\nknee runs n={len(kn)}: spike density per 100 ep, sorted: '
          + ' '.join(f"{r['density_per100']:.2f}" for r in sorted(kn, key=lambda r: r['density_per100'])))
    print(f'non-knee n={len(nk)}: spike density per 100 ep, sorted: '
          + ' '.join(f"{r['density_per100']:.2f}" for r in sorted(nk, key=lambda r: r['density_per100'])))
    d = [r['nearest_spike_to_onset'] for r in kn if r['nearest_spike_to_onset'] != '']
    print(f'nearest-spike-to-onset over {len(d)} knee runs with >=1 spike, sorted: '
          + ' '.join(str(x) for x in sorted(d)))
    print(f'knee runs with NO spike anywhere: {sum(1 for r in kn if r["n_spikes"] == 0)}/{len(kn)}')
    print(f'csv: {OUT}')


if __name__ == '__main__':
    main()
