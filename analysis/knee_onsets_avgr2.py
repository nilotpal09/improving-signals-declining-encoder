"""Decline label, onset and per-factor sizes on average ridge R2
(readability = held-out ridge R2 of z1, z2, log E; figures show
their average; onset rule replaced and defined on the average).

Rule (vetted 2026-10-08 on the marked sheet of all 131 runs):
  * avg(e) = (R2_z1 + R2_z2 + R2_logE) / 3 at every saved checkpoint e.
  * Peak = checkpoint with the largest avg among epochs >= 20; final = last.
  * Drop = avg(peak) - avg(final). Decline if drop >= 0.07 (middle of the empty
    gap between 0.044 and 0.094; any value in the gap gives the same split).
  * Onset = first checkpoint after the peak with avg < avg(peak) - drop / 5.
  * Pre-onset checkpoint = last checkpoint at least 20 epochs before onset.
  * Dip = largest fall from the best value so far (epochs >= 20); trough = where
    it bottoms out. Descriptive only.
Per-factor drops use the same two checkpoints (peak, final), so their mean is
the drop. Old mean-CCA onsets (knee_s1_onsets R_A, R_B) are carried for
comparison only.

Inputs (read only): results/knee_trajectories.csv (mean CCA) and the per-factor
files knee_traj_full, knee_traj_seedext, knee_traj_lam005_ext,
knee_labelfree_readability, knee_traj_ridge_backfill. Where a run appears in
several, the file with the most checkpoints is used. Writes a NEW file,
results/knee_onsets_avgr2.csv (refuses to overwrite).
"""
import sys

sys.dont_write_bytecode = True

import csv  # noqa: E402
import os  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
sys.path.insert(0, _HERE)
from knee_s1_onsets import onset_A, onset_B  # noqa: E402

CSV = os.path.join(paths.CSV)
RIDGE = ('knee_traj_full.csv', 'knee_traj_seedext.csv', 'knee_traj_lam005_ext.csv',
         'knee_labelfree_readability.csv', 'knee_traj_ridge_backfill.csv',
         'knee_traj_denom.csv')  # the 16 denominator runs
OUT = os.path.join(CSV, 'knee_onsets_avgr2.csv')
DELTA, FRAC, PRE_GAP = 0.07, 0.2, 20
F = ('z1', 'z2', 'E')


def key(run):
    return run.split('xxx')[0]


def load():
    R = {}
    for f in RIDGE:
        tmp = {}
        for r in csv.DictReader(open(os.path.join(CSV, f))):
            if r.get('ridge_r2_z1'):
                tmp.setdefault(key(r['run']), {})[int(r['epoch'])] = \
                    tuple(float(r[f'ridge_r2_{x}']) for x in F)
        for k, v in tmp.items():
            if k not in R or len(v) > len(R[k]):
                R[k] = v
    M = {}
    for r in csv.DictReader(open(os.path.join(CSV, 'knee_trajectories.csv'))):
        M.setdefault(key(r['run']), {})[int(r['epoch'])] = float(r['mean_cca'])
    for f in RIDGE:
        for r in csv.DictReader(open(os.path.join(CSV, f))):
            M.setdefault(key(r['run']), {}).setdefault(int(r['epoch']), float(r['mean_cca']))
    return R, M


def label(r3):
    avg = {e: sum(v) / 3 for e, v in r3.items()}
    g = [e for e in sorted(avg) if e >= 20]
    pk = max(g, key=lambda e: avg[e])
    fin = g[-1]
    drop = avg[pk] - avg[fin]
    best, dip, tr = -1.0, 0.0, None
    for e in g:
        best = max(best, avg[e])
        if best - avg[e] > dip:
            dip, tr = best - avg[e], e
    dec = drop >= DELTA
    on = pre = None
    if dec:
        on = next(e for e in g if e > pk and avg[e] < avg[pk] - FRAC * drop)
        pre = max(e for e in sorted(avg) if e <= on - PRE_GAP)
    return avg, pk, fin, drop, dip, tr, dec, on, pre


def main():
    assert not os.path.exists(OUT), f'{OUT} exists; refusing to overwrite'
    R, M = load()
    cols = (['run', 'last_epoch', 'peak_epoch', 'avg_peak', 'avg_final', 'drop', 'decline',
             'onset', 'pre_onset_epoch', 'dip', 'trough_epoch']
            + [f'drop_{x}' for x in F] + [f'r2_{x}_peak' for x in F] + [f'r2_{x}_final' for x in F]
            + ['old_RA_meancca', 'old_RB_meancca'])
    w = csv.writer(open(OUT, 'w', newline=''), lineterminator='\n')
    w.writerow(cols)
    for k in sorted(R):
        avg, pk, fin, drop, dip, tr, dec, on, pre = label(R[k])
        a, _ = onset_A(M[k])
        b, _ = onset_B(M[k])
        w.writerow([k, fin, pk, f'{avg[pk]:.6f}', f'{avg[fin]:.6f}', f'{drop:.6f}', int(dec),
                    on if on is not None else '', pre if pre is not None else '',
                    f'{dip:.6f}', tr if tr is not None else '']
                   + [f'{R[k][pk][i] - R[k][fin][i]:.6f}' for i in range(3)]
                   + [f'{R[k][pk][i]:.6f}' for i in range(3)]
                   + [f'{R[k][fin][i]:.6f}' for i in range(3)]
                   + [a if a is not None else '', b if b is not None else ''])
    print(f'wrote {OUT}: {len(R)} runs')


if __name__ == '__main__':
    main()
