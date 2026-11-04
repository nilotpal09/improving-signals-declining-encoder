"""Direction null at the average-R2 anchors, VARIANT B (2026-10-13).

Copy of knee_dir_null_avgr2.py with two changes (the status quo output
results/knee_dir_null_avgr2.csv is left as is):
  * the denominator runs without a decline join the FLAT lists in the existing
    pattern: width 2048 (FLAT_21) += rigb nano_10, nano_12, rigd mu097_nano_4-8;
    width 384 (FLAT_4) += rigb p384_ou09_4-6, p384_ou05_4-6, rigd mu097_p384_5-6;
  * output results/knee_dir_null_avgr2_varB.csv (refuses to overwrite).
Each added run is checked to be a non-decline in results/knee_onsets_avgr2.csv.
Inference only.
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
ADD_21 = ([f'rigb_lejp_p1s100_nano_{s}' for s in (10, 12)]
          + [f'rigd_lejp_p1s100_mu097_nano_{s}' for s in (4, 5, 6, 7, 8)])
ADD_4 = ([f'rigb_lejp_p1s100_p384_ou0{k}_{s}' for k in ('9', '5') for s in (4, 5, 6)]
         + [f'rigd_lejp_p1s100_mu097_p384_{s}' for s in (5, 6)])

N.SWEEP = os.path.join(paths.CSV, 'knee_dir_sweep_avgr2.csv')
N.OUT = os.path.join(paths.CSV, 'knee_dir_null_avgr2_varB.csv')

if __name__ == '__main__':
    assert not os.path.exists(N.OUT), f'refusing to overwrite {N.OUT}'
    ons = {r['run']: r for r in csv.DictReader(open(os.path.join(paths.CSV, 'knee_onsets_avgr2.csv')))}
    for k in ADD_21 + ADD_4:
        assert ons[k]['decline'] == '0', k
        assert k not in N.FLAT_21 + N.FLAT_4, k
    N.FLAT_21 = N.FLAT_21 + ADD_21
    N.FLAT_4 = N.FLAT_4 + ADD_4
    print(f'FLAT_21 {len(N.FLAT_21)} runs, FLAT_4 {len(N.FLAT_4)} runs', flush=True)
    rows = {r['run']: r for r in csv.DictReader(open(N.SWEEP))}
    N.W_KNEE = {k: tuple(float(rows[run][c]) for c in ('w_z1', 'w_z2', 'w_E')) for k, run in REF.items()}
    N.main()
