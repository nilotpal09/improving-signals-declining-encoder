"""The chronological detector study (knee_detector_chrono.py, unchanged) rerun
with the onsets of the average-R2 rule (results/knee_onsets_avgr2.csv).
Nothing else changes: same series, detectors, thresholds,
horizons, controls, held-out runs and timely window (|alarm - onset| <= 10).

How: the onset tables knee_lidar_audit.KNEE_21 / KNEE_4 and the copies made
at import (knee_detector_chrono.KNEE, which is REAL.knee;
knee_labelfree_audit.KNEE; knee_labelfree_selection.KNEE) are updated in
place, and the output paths are redirected to new files. The record script
and its CSVs are not touched.

  --onsets new   new onsets -> results/knee_detector_chrono_avgr2_*.csv
  --onsets old   stored onsets, outputs to --outdir (reproduction check: the
                 summary must equal results/knee_detector_chrono_summary.csv)
  --report       print the record script's report from the new files
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import csv  # noqa: E402
import os  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import knee_detector_chrono as DC  # noqa: E402

ONS = os.path.join(DC.CSV_DIR, 'knee_onsets_avgr2.csv')


def swap_onsets():
    new = {r['run']: int(r['onset']) for r in csv.DictReader(open(ONS)) if r['decline'] == '1'}
    old = dict(DC.KNEE)
    for d in (DC.AUD.KNEE_21, DC.AUD.KNEE_4, DC.KNEE, DC.LFA.KNEE, DC.SEL.KNEE):
        for k in d:
            d[k] = new[k]
    assert DC.REAL.knee is DC.KNEE
    return old, dict(DC.KNEE)


def redirect(outdir, tag):
    for k, p in DC.OUT.items():
        DC.OUT[k] = os.path.join(outdir, os.path.basename(p).replace('knee_detector_chrono_', tag))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onsets', choices=('new', 'old'))
    ap.add_argument('--outdir', default=DC.CSV_DIR)
    ap.add_argument('--report', action='store_true')
    a = ap.parse_args()
    if a.report or a.onsets == 'new':
        old, new = swap_onsets()
        redirect(DC.CSV_DIR if a.report else a.outdir, 'knee_detector_chrono_avgr2_')
        if a.report:
            DC.report()
            return
        moved = {k: (old[k], new[k]) for k in old if old[k] != new[k]}
        print(f'onsets changed in {len(moved)} of {len(old)} runs: {moved}')
        print(f'main comparison set (eligible in all three matched designs): {len(DC.main21_set())} runs')
    else:
        assert a.outdir != DC.CSV_DIR, 'old-onset check must write outside results/'
        redirect(a.outdir, 'repro_knee_detector_chrono_')
    DC.run()


if __name__ == '__main__':
    main()
