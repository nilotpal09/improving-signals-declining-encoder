"""Readability in average ridge R2 for the main-text figures (convention:
readability = held-out ridge R2 of z1, z2, log E; figures show their average;
onsets from the average-R2 rule).

Sources (read only, results/): per-checkpoint R2 via
knee_onsets_avgr2.load; peak, final, drop and onset per run from
knee_onsets_avgr2.csv; untrained-encoder floors (seed 0, the convention that
reproduces the quoted mean-CCA floors) from floor_avgr2_rungs.csv.

check(man) reproduces every mean-CCA value the installed Figure 3 plotted from
the run lists below (drop = max over all checkpoints - final, as installed),
so the run lists are the ones behind each plotted point.

2026-10-19: the 16 denominator-completion runs join the lists below. Data now
come from the 147-run onset table and the new runs' readability; check(man)
runs on the original lists (DENOM excluded), whose values the 147-run tables
leave unchanged.
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBES = ROOT / 'analysis'
sys.path.insert(0, str(PROBES))
from knee_onsets_avgr2 import load  # noqa: E402

R3, MCCA = load()
AVG = {k: {e: sum(v) / 3 for e, v in t.items()} for k, t in R3.items()}
ONS = {r['run']: r for r in csv.DictReader(open(ROOT / 'results/knee_onsets_avgr2.csv'))}
FLOOR = {int(r['n_spots']): float(r['avg_r2'])
         for r in csv.DictReader(open(ROOT / 'results/floor_avgr2_rungs.csv')) if r['seed'] == '0'}

# Figures 1 and 2 traces (2026-10-20): read from the CSVs below instead
# of the 2026-09-26 figure manifest (JSON); the values are identical.
LOSS_CSV = ROOT / 'results/knee_default_loss.csv'   # knee_default_loss.py
_cache = {}


def _base(run):
    return run.split('xxx')[0]


def loss_trace(k):
    """(epochs, per-epoch mean training loss) of a default run."""
    if 'loss' not in _cache:
        t = {}
        for r in csv.DictReader(open(LOSS_CSV)):
            t.setdefault(r['run'], {})[int(r['epoch'])] = float(r['loss'])
        _cache['loss'] = t
    t = _cache['loss'][k]
    return sorted(t), [t[e] for e in sorted(t)]


def rankme_trace(k):
    """(checkpoint epochs, uncentred RankMe) from knee_labelfree_metrics.csv."""
    if 'rank' not in _cache:
        t = {}
        for r in csv.DictReader(open(ROOT / 'results/knee_labelfree_metrics.csv')):
            if (r['metric'], r['variant']) == ('rankme', 'pub_uncentred'):
                t.setdefault(_base(r['run']), {})[int(r['epoch'])] = float(r['value'])
        _cache['rank'] = t
    t = _cache['rank'][k]
    return sorted(t), [t[e] for e in sorted(t)]


def factor_traces(k):
    """(checkpoint epochs, {ridge_r2_z1, ridge_r2_z2, ridge_r2_E}) from knee_traj_full.csv."""
    rows = sorted((r for r in csv.DictReader(open(ROOT / 'results/knee_traj_full.csv'))
                   if _base(r['run']) == k), key=lambda r: int(r['epoch']))
    keys = ('ridge_r2_z1', 'ridge_r2_z2', 'ridge_r2_E')
    return [int(r['epoch']) for r in rows], {c: [float(r[c]) for r in rows] for c in keys}

L = 'rigb_lejp_p1s100_'
D = 'rigd_lejp_p1s100_'
REFRESH = {(1.0, 300): [f'{L}p384_{s}' for s in range(1, 11)],
           (0.9, 300): [f'{L}p384_ou09_{s}' for s in range(1, 7)],
           (0.5, 300): [f'{L}p384_ou05_{s}' for s in range(1, 7)],
           (0.9, 1000): [f'{L}p384_ou09_e1000_{s}' for s in range(1, 7)]}
OCCURRENCE = {(0.0, 384): [f'{L}p384_{s}' for s in range(1, 11)],
              (0.0, 2048): [f'{L}nano_{s}' for s in range(4, 10)] + [f'{L}long_nano_1']
              + [f'{L}nano_{s}' for s in (10, 11, 12)],
              (0.97, 384): [f'{D}mu097_p384_{s}' for s in range(1, 7)],
              (0.97, 2048): [f'{D}mu097_long_nano_1'] + [f'{D}mu097_nano_{s}' for s in range(4, 9)]}
for m in ('04', '07', '09'):
    OCCURRENCE[(int(m) / 10, 384)] = [f'{D}mu{m}_p384_{s}' for s in range(1, 7)]
    OCCURRENCE[(int(m) / 10, 2048)] = [f'{D}mu{m}_nano_{s}' for s in range(1, 7)]
RUNGS = {30: [f'rigb_lejp_p1s30_p384_{s}' for s in range(1, 7)],
         100: [f'{L}p384_{s}' for s in range(1, 11)],
         300: [f'rigb_lejp_p1s300_p384_{s}' for s in range(1, 7)],
         1000: [f'rigb_lejp_p1s1000_p384_{s}' for s in (1, 2, 3)]}
DENOM = ({f'{L}nano_{s}' for s in (10, 11, 12)} | {f'{L}p384_ou0{k}_{s}' for k in '95' for s in (4, 5, 6)}
         | {f'{D}mu097_nano_{s}' for s in range(4, 9)} | {f'{D}mu097_p384_{s}' for s in (5, 6)})
assert len(DENOM) == 16 and DENOM <= set(ONS)
TEN = [f'{L}p384_{s}' for s in range(1, 11)]          # Figure 2, width 384, mu = 0
NINE = [k for k in TEN if ONS[k]['decline'] == '1']   # Figure 1(b)


def onset(k):
    return int(ONS[k]['onset'])


def drop(k):
    return float(ONS[k]['drop'])


def peak_final(k):
    return float(ONS[k]['avg_peak']), float(ONS[k]['avg_final'])


def _old_drop(k):
    t = MCCA[k]
    return round(max(t.values()) - t[max(t)], 4)


def check(man):
    cm = man['fig_results_comparisons']
    bad = []
    for r in cm['refresh']:
        runs = [k for k in REFRESH[(r['rho'], r['epochs'])] if k not in DENOM]
        if sorted(_old_drop(k) for k in runs) != sorted(r['drops']):
            bad.append(('refresh', r['rho'], r['epochs']))
    for r in cm['occurrence']:
        runs = [k for k in OCCURRENCE[(r['mu'], r['width'])] if k not in DENOM]
        if sorted(_old_drop(k) for k in runs) != sorted(r['drops']):
            bad.append(('occurrence', r['mu'], r['width']))
    for r in cm['graininess']:
        t = MCCA[r['run']]
        if abs(max(t.values()) - r['peak']) > 1e-6 or abs(t[max(t)] - r['final']) > 1e-6:
            bad.append(('graininess', r['run']))
    assert NINE == [k for k in TEN if k != f'{L}p384_10'], NINE
    n = len(cm['refresh']) + len(cm['occurrence']) + len(cm['graininess'])
    print(f'  avgr2_data check: run lists reproduce the installed mean-CCA values: '
          f'{n - len(bad)}/{n} entries' + (f'; MISMATCH {bad}' if bad else ''))
    assert not bad
