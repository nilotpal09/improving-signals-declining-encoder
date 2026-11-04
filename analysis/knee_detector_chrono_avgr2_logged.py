"""The two Figure 3 signals the chronological audit lacked, added at the vetted
onsets (2026-10-11): SIGReg (training-log column lejepa_sigreg) and feature
standard deviation (feat_std_event).

Everything else is the vetted-onset audit (knee_detector_chrono_avgr2.py, which
runs knee_detector_chrono.py with results/knee_onsets_avgr2.csv): same detectors,
thresholds, horizons, controls, held-out runs, onsets and timely window. Only
the series list is replaced by these two.

Series: each run's lejepa_diag_r0.csv, the mean over the rows of each logged
epoch, with checkpoint epoch N taking logged epoch N - 1 on the loss grid. This
is the loss series' own mapping (knee_labelfree_selection.loss_series) and the
one Figure 3 uses (knee_response_auc.log_series); the loader asserts that the
loss read the same way equals the audit's loss series exactly. MAE has no
LeJEPA log, so its rows are n/a.

Outputs results/knee_detector_chrono_avgr2_logged_{runs,summary,fits,paths,manifest};
no existing file is touched (the record script refuses to overwrite).
"""
import sys

sys.dont_write_bytecode = True

import csv  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
from collections import defaultdict  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import knee_detector_chrono as DC  # noqa: E402
import knee_detector_chrono_avgr2 as A  # noqa: E402
import knee_labelfree_metrics  # noqa: E402,F401  (manifest_rows hashes this module)

COLS = {'log lejepa_sigreg': 'lejepa_sigreg', 'log feat_std_event': 'feat_std_event'}


def series_specs():
    return [dict(name=n, family='training log', role='', designs=DC.ALL4,
                 source=f'lejepa_diag_r0.csv column {c}, mean per logged epoch; checkpoint epoch N <- logged '
                        f'epoch N - 1 on the loss grid', g1='')
            for n, c in COLS.items()]


def load_all_series():
    grid = DC.loss_grid()
    loss = DC.SEL.loss_series(grid)
    out = {n: {} for n in COLS}
    for run in DC.SEL.ALL_RUNS:
        if run in DC.SEL.MAE:
            continue
        fn = os.path.join(DC.SEL.run_dir(run), 'lejepa_diag_r0.csv')
        by = defaultdict(lambda: defaultdict(list))
        with open(fn) as fh:
            for r in csv.DictReader(fh):
                for c in list(COLS.values()) + ['loss']:
                    try:
                        by[c][int(r['epoch'])].append(float(r[c]))
                    except (ValueError, KeyError):
                        pass
        m = {c: {e: sum(v) / len(v) for e, v in d.items()} for c, d in by.items()}
        g = grid[run]
        assert [m['loss'][e - 1] for e in g] == loss[run][1], ('loss mapping differs', run)
        for n, c in COLS.items():
            out[n][run] = (g, [m.get(c, {}).get(e - 1, math.nan) for e in g])
    return out


def main():
    A.swap_onsets()
    A.redirect(DC.CSV_DIR, 'knee_detector_chrono_avgr2_logged_')
    DC.series_specs = series_specs
    DC.load_all_series = load_all_series
    DC.run()


if __name__ == '__main__':
    main()
