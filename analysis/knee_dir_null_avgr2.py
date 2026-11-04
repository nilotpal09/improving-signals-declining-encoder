"""Direction null at the average-R2 anchors (2026-10-10).

Companion of knee_dir_sweep_avgr2.py. knee_dir_null.py takes each flat run's PRE epochs from its
campaign's decline cohort (median, min and max PRE of results/knee_dir_sweep.csv, old mean-CCA rules);
here the same code runs unchanged (its main() is reused) with two inputs swapped:
  SWEEP  -> results/knee_dir_sweep_avgr2.csv (PRE at the vetted average-R2 onsets, 55 declines);
  W_KNEE -> the four original width-2048 runs' directions recomputed at the new anchors, so cos_wbar is
            against wbar_new (|cos(wbar_old, wbar_new)| = 1.00000 in knee_dir_sweep_avgr2's log).
Flat runs: knee_dir_null's FLAT_21 / FLAT_4 lists; all 28 are non-declines under the average-R2 rule.
Inference only. Writes results/knee_dir_null_avgr2.csv (refuses to overwrite).
"""
import sys

sys.dont_write_bytecode = True

import csv  # noqa: E402
import os  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
sys.path.insert(0, _HERE)
import knee_dir_null as N  # noqa: E402

REF = {'knee seed 1': 'rigb_lejp_p1s100_nano_1', 'knee seed 2': 'rigb_lejp_p1s100_nano_2',
       'knee seed 3': 'rigb_lejp_p1s100_nano_3', 'knee long': 'rigb_lejp_p1s100_long_nano_1'}

N.SWEEP = os.path.join(paths.CSV, 'knee_dir_sweep_avgr2.csv')
N.OUT = os.path.join(paths.CSV, 'knee_dir_null_avgr2.csv')

if __name__ == '__main__':
    assert not os.path.exists(N.OUT), f'refusing to overwrite {N.OUT}'
    rows = {r['run']: r for r in csv.DictReader(open(N.SWEEP))}
    N.W_KNEE = {k: tuple(float(rows[run][c]) for c in ('w_z1', 'w_z2', 'w_E')) for k, run in REF.items()}
    N.main()
