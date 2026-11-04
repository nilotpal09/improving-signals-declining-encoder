"""Per-epoch training loss of the ten default runs (width 384, mu = 0, N = 100,
seeds 1-10), the loss traces of the paper's Figures 1 and 2.

Each run's training log (lejepa_diag_r0.csv in its run folder) logs the loss
every 10 iterations; the per-epoch value is the mean over that epoch's logged
rows. This is the computation make_results_figs.py used for the 2026-09-26
figure manifest; the figure scripts now read this CSV instead of the manifest.
Writes csv/knee_default_loss.csv (run, epoch, loss, n_rows). Floats are
written with full precision so the figures reproduce exactly.
"""
import csv
import glob
import os
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
STORAGE = paths.RUNS
RUNS = [f'rigb_lejp_p1s100_p384_{s}' for s in range(1, 11)]
OUT = os.path.join(paths.CSV, 'knee_default_loss.csv')


def main():
    rows = []
    for k in RUNS:
        matches = sorted(glob.glob(os.path.join(STORAGE, k + 'xxx*')))
        assert len(matches) == 1, (k, matches)
        by = defaultdict(list)
        with open(os.path.join(matches[0], 'lejepa_diag_r0.csv')) as f:
            for r in csv.DictReader(f):
                try:
                    by[int(r['epoch'])].append(float(r['loss']))
                except (ValueError, KeyError):
                    continue
        for e in sorted(by):
            rows.append(dict(run=k, epoch=e, loss=repr(sum(by[e]) / len(by[e])),
                             n_rows=len(by[e])))
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['run', 'epoch', 'loss', 'n_rows'])
        w.writeheader()
        w.writerows(rows)
    print(f'{len(RUNS)} runs, {len(rows)} rows; wrote {OUT}')


if __name__ == '__main__':
    main()
