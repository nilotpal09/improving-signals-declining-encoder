"""Training time of the LeJEPA width-384, 300-epoch runs, for the paper's
"Width-384 runs of 300 epochs took three to seven GPU-hours" (training
appendix). The training logs do not record wall time, so it is measured
from checkpoint file times: hours from the epoch-2 checkpoint to the
epoch-300 checkpoint, scaled by 300/298 to cover the whole run. Run list:
results/runs.csv; checkpoint times: results/checkpoint_times.csv (the file
times of the original checkpoints; the Zenodo MANIFEST.csv gives the same
times in its file_time_utc column).
Writes results/knee_run_hours.csv, one row per run.
"""
import csv
import os
import statistics

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402

RUNS = os.path.join(paths.CSV, 'runs.csv')
TIMES = os.path.join(paths.CSV, 'checkpoint_times.csv')
OUT = os.path.join(paths.CSV, 'knee_run_hours.csv')
FIRST, LAST = 2, 300


def main():
    t = {(r['run'], r['checkpoint_tag']): float(r['mtime_unix']) for r in csv.DictReader(open(TIMES))}
    rows = []
    for r in csv.DictReader(open(RUNS)):
        if r['width'] != '384' or r['epochs'] != '300':
            continue
        if r['objective'] == 'MAE':
            continue
        k0, k1 = (r['run'], f'ep{FIRST}'), (r['run'], f'ep{LAST}')
        if k0 not in t or k1 not in t:
            rows.append(dict(run=r['run'], hours='', note='checkpoint missing'))
            continue
        h = (t[k1] - t[k0]) / 3600.0
        rows.append(dict(run=r['run'], hours=round(h * LAST / (LAST - FIRST), 2),
                         note=''))
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['run', 'hours', 'note'])
        w.writeheader()
        w.writerows(rows)
    hs = [x['hours'] for x in rows if x['hours'] != '']
    print(f'{len(hs)} runs timed, {len(rows) - len(hs)} missing; '
          f'min {min(hs):.2f} h, median {statistics.median(hs):.2f} h, '
          f'max {max(hs):.2f} h')
    print('wrote', OUT)


if __name__ == '__main__':
    main()
