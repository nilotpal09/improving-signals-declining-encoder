"""Per-epoch mean of the every-iteration training loss, for every run.

knee_labelfree_selection.loss_all_iterations uses it for the sensitivity series
"loss [all iterations, N-1]". It is computed from the every-iteration logs
(jepa_r0.csv for LeJEPA, mae_r0.csv for MAE). The LeJEPA ones are not in the Zenodo
record (0.9 GB), so this table holds the per-epoch means instead; values are written
with full float precision, and the mean is taken exactly as before (sum / count over
the rows of each logged epoch, in file order).
Writes results/loss_all_iterations.csv (run, epoch, loss_mean).
"""
import csv
import glob
import os
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402

OUT = os.path.join(paths.CSV, 'loss_all_iterations.csv')


def main():
    assert not os.path.exists(OUT), f'refusing to overwrite {OUT}'
    rows = []
    for r in csv.DictReader(open(os.path.join(paths.CSV, 'runs.csv'))):
        d = os.path.join(paths.RUNS, r['run_folder'])
        fn = os.path.join(d, 'mae_r0.csv' if r['objective'] == 'MAE' else 'jepa_r0.csv')
        by = defaultdict(list)
        with open(fn) as fh:
            for x in csv.DictReader(fh):
                by[int(x['epoch'])].append(float(x['loss']))
        for e in sorted(by):
            rows.append(dict(run=r['run'], epoch=e, loss_mean=repr(sum(by[e]) / len(by[e]))))
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['run', 'epoch', 'loss_mean'], lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {OUT}: {len(rows)} rows')


if __name__ == '__main__':
    main()
