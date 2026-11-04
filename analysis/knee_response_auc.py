"""Rule-free response test: the decline onset is already known, so this checks
directly how the monitors respond around it. No detector,
threshold, calibration window or horizon cut.

For a decline run d with onset o_d and pre-onset checkpoint p_d = o_d - 20
(results/knee_onsets_avgr2.csv), a signal x and an offset k:
    change_d(k)    = x_d(o_d + k) - x_d(p_d)
For each run n without a decline in the same campaign (width), over the SAME
epochs:
    change_n(d, k) = x_n(o_d + k) - x_n(p_d)
AUC(k) = fraction of pairs (d, n) with change_d(k) > change_n(d, k), ties 1/2.
0.5: the signal's change around the onset does not tell decline runs from runs
without one; near 0 or 1: it does. Pairs where either run lacks a value are
dropped (21x series end at epoch 150). Interval: 2000 bootstrap resamples of
runs, within group and campaign, seed 0.

Runs: the alarm audit's LeJEPA cohort (27 decline runs; 8 mu = 0.97 controls
and 11 held-out runs without a decline). Series: knee_detector_chrono's
load_all_series (unchanged); the training-log columns of lejepa_diag_r0.csv
(per-epoch mean, grid epoch N <- logged epoch N - 1, the mapping the audit uses
for loss; checked by reproducing its loss series exactly); and readability
(average and per-factor ridge R2) as the positive control, since the onset is
defined on it. Writes results/knee_response_auc.csv (refuses to overwrite).
"""
import sys

sys.dont_write_bytecode = True

import csv  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
from collections import defaultdict  # noqa: E402
from statistics import median  # noqa: E402

import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
sys.path.insert(0, _HERE)
import knee_detector_chrono as DC  # noqa: E402
import knee_labelfree_selection as SEL  # noqa: E402
from knee_onsets_avgr2 import load as load_readability  # noqa: E402

CSV = os.path.join(paths.CSV)
ONS = os.path.join(CSV, 'knee_onsets_avgr2.csv')
OUT = os.path.join(CSV, 'knee_response_auc.csv')
OFFSETS = (-10, 0, 10, 20, 30, 40, 50)
PRE_GAP = 20
N_BOOT = 2000
LOG_COLS = ('loss', 'lejepa_inv', 'lejepa_sigreg', 'sigreg_eval', 'feat_std_event',
            'rankme_tokens', 'rankme_event')


def log_series(runs, grid):
    """Training-log columns on each run's checkpoint grid, loss mapping (N <- N - 1)."""
    out = defaultdict(dict)
    for run in runs:
        fn = os.path.join(SEL.run_dir(run), 'lejepa_diag_r0.csv')
        by = defaultdict(lambda: defaultdict(list))
        for r in csv.DictReader(open(fn)):
            for c in LOG_COLS:
                try:
                    by[c][int(r['epoch'])].append(float(r[c]))
                except (ValueError, KeyError):
                    pass
        for c in LOG_COLS:
            m = {e: sum(v) / len(v) for e, v in by[c].items()}
            out[f'log {c}'][run] = {e: m[e - 1] for e in grid[run] if e - 1 in m}
    return out


def all_series(runs):
    S = {}
    for name, d in DC.load_all_series().items():
        S[name] = {r: dict(zip(*d[r])) for r in runs}
    grid = {r: sorted(S['loss'][r]) for r in runs}
    L = log_series(runs, grid)
    worst = max(abs(L['log loss'][r][e] - S['loss'][r][e]) for r in runs for e in grid[r])
    print(f'check: training-log loss vs audit loss series, max |diff| = {worst:.2e}')
    assert worst == 0.0
    del L['log loss']
    S.update(L)
    R, _ = load_readability()
    for i, f in enumerate(('z1', 'z2', 'E')):
        S[f'readability R2 {f}'] = {r: {e: v[i] for e, v in R[r].items()} for r in runs}
    S['readability R2 average'] = {r: {e: sum(v) / 3 for e, v in R[r].items()} for r in runs}
    return S


def wins(x, dec, nod, ons, k):
    """Per campaign: W[i, j] in {0, .5, 1} or nan, with the changes behind it."""
    W = np.full((len(dec), len(nod)), np.nan)
    cd, cn = [], []
    for i, d in enumerate(dec):
        o = ons[d]
        a, b = o - PRE_GAP, o + k
        xd = x[d]
        if a not in xd or b not in xd or not (math.isfinite(xd[a]) and math.isfinite(xd[b])):
            continue
        dd = xd[b] - xd[a]
        cd.append(dd)
        for j, n in enumerate(nod):
            xn = x[n]
            if a not in xn or b not in xn or not (math.isfinite(xn[a]) and math.isfinite(xn[b])):
                continue
            dn = xn[b] - xn[a]
            cn.append(dn)
            W[i, j] = 1.0 if dd > dn else (0.5 if dd == dn else 0.0)
    return W, cd, cn


def auc_of(mats):
    s = sum(np.nansum(W) for W in mats)
    n = sum(np.count_nonzero(~np.isnan(W)) for W in mats)
    return (s / n if n else float('nan')), n


def main():
    assert not os.path.exists(OUT), f'refusing to overwrite {OUT}'
    ons = {r['run']: int(r['onset']) for r in csv.DictReader(open(ONS)) if r['decline'] == '1'}
    dec = {c: [r for r in DC.KNEE if DC.A2.campaign(r) == c] for c in DC.CAMPS}
    nod = {'21x': DC.AUD.CTRL_21 + DC.A2.HELD_21, '4x': DC.AUD.CTRL_4 + DC.A2.HELD_4}
    for c in DC.CAMPS:
        assert all(r in ons for r in dec[c]) and not any(r in ons for r in nod[c]), c
    runs = [r for c in DC.CAMPS for r in dec[c] + nod[c]]
    S = all_series(runs)
    rng = np.random.default_rng(0)
    rows = []
    for name, x in S.items():
        for k in OFFSETS:
            per = {c: wins(x, dec[c], nod[c], ons, k) for c in DC.CAMPS}
            boot = []
            for _ in range(N_BOOT):
                mats = []
                for c in DC.CAMPS:
                    W = per[c][0]
                    i = rng.integers(0, W.shape[0], W.shape[0])
                    j = rng.integers(0, W.shape[1], W.shape[1])
                    mats.append(W[np.ix_(i, j)])
                boot.append(auc_of(mats)[0])
            for scope, cs in (('pooled', DC.CAMPS), ('21x', ('21x',)), ('4x', ('4x',))):
                a, n = auc_of([per[c][0] for c in cs])
                if scope == 'pooled':
                    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
                else:
                    lo = hi = float('nan')
                cd = [v for c in cs for v in per[c][1]]
                cn = [v for c in cs for v in per[c][2]]
                nd = sum(int(np.any(~np.isnan(per[c][0]), axis=1).sum()) for c in cs)
                nn = sum(int(np.any(~np.isnan(per[c][0]), axis=0).sum()) for c in cs)
                rows.append(dict(series=name, scope=scope, offset=k, n_decline=nd, n_nodecline=nn,
                                 n_pairs=n, auc=a, ci_lo=lo, ci_hi=hi,
                                 median_change_decline=median(cd) if cd else float('nan'),
                                 median_change_nodecline=median(cn) if cn else float('nan')))
    cols = list(rows[0])
    with open(OUT, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow({c: (f'{v:.6g}' if isinstance(v, float) else v) for c, v in r.items()})
    print(f'wrote {OUT}: {len(rows)} rows, {len(S)} series x {len(OFFSETS)} offsets x 3 scopes')


if __name__ == '__main__':
    main()
