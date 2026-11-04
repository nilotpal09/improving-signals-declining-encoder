"""S2a/S2b audit on the LiDAR series, per the rules printed in the record
before computation (2026-08-22).

Statistic functions are imported from knee_s2_monitors (local_dev, rank_at,
stdz, cusum, page_hinkley, ewma, wslope) rather than reimplemented, so the
LiDAR series is audited by exactly the machinery every other monitor was.
Thresholds are calibrated PER CAMPAIGN on that campaign's own four mu=0.97
controls, 1.1 x max, as printed.

Usage: python knee_lidar_audit.py [--rung 100|1000]
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
sys.path.insert(0, _HERE)
import knee_s2_monitors as S2  # noqa: E402

LID = os.path.join(paths.CSV, 'knee_lidar.csv')

KNEE_21 = {'rigb_lejp_p1s100_nano_1': 90, 'rigb_lejp_p1s100_nano_2': 110,
           'rigb_lejp_p1s100_nano_3': 100, 'rigb_lejp_p1s100_long_nano_1': 160,
           'rigb_lejp_p1s100_nano_4': 80, 'rigb_lejp_p1s100_nano_6': 70,
           'rigb_lejp_p1s100_nano_7': 170, 'rigb_lejp_p1s100_nano_8': 110,
           'rigb_lejp_p1s100_nano_9': 120,
           'rigd_lejp_p1s100_mu04_nano_3': 130,
           'rigd_lejp_p1s100_mu07_nano_1': 140, 'rigd_lejp_p1s100_mu07_nano_2': 120}
KNEE_21['rigb_lejp_p1s100_nano_11'] = 90  # denom 2026-10-13 (onset from the new onset table)
CTRL_21 = [f'rigd_lejp_p1s100_mu097_nano_{s}' for s in (1, 2, 3)] + \
          ['rigd_lejp_p1s100_mu097_long_nano_1']
KNEE_4 = {f'rigb_lejp_p1s100_p384_{s}': o for s, o in
          ((1, 50), (2, 140), (3, 90), (4, 140), (5, 60), (6, 70),
           (7, 80), (8, 90), (9, 120))}
KNEE_4.update({f'rigd_lejp_p1s100_mu04_p384_{s}': 80 for s in (1, 2, 3)})
KNEE_4.update({'rigd_lejp_p1s100_mu07_p384_1': 90,
               'rigd_lejp_p1s100_mu07_p384_3': 90,
               'rigd_lejp_p1s100_mu09_p384_3': 130})
CTRL_4 = [f'rigd_lejp_p1s100_mu097_p384_{s}' for s in (1, 2, 3, 4)]


def load(rung):
    out = defaultdict(dict)
    with open(LID) as fh:
        for r in csv.DictReader(fh):
            if int(r['n_spots']) != rung:
                continue
            out[r['run'].split('xxx')[0]][int(r['epoch'])] = float(r['lidar'])
    return {k: (sorted(v), [v[e] for e in sorted(v)]) for k, v in out.items()}


def audit(name, knee, ctrl, S):
    have_k = {k: o for k, o in knee.items() if k in S}
    have_c = [c for c in ctrl if c in S]
    if not have_k or not have_c:
        print(f'\n[{name}] incomplete: {len(have_k)}/{len(knee)} knee, '
              f'{len(have_c)}/{len(ctrl)} controls present — skipped')
        return
    print(f'\n=== {name}: LiDAR audit ({len(have_k)} knee, {len(have_c)} controls) ===')

    print('S2a local-deviation rank fraction at onset (small = standout):')
    placebo = []
    for k, o in have_k.items():
        D = S2.local_dev(*S[k])
        r = S2.rank_at(D, o)
        print(f'  {k.split("p1s100_")[-1]:22s} {r:.3f}' if r else
              f'  {k.split("p1s100_")[-1]:22s}   --')
        for k2, o2 in have_k.items():
            if k2 != k:
                r2 = S2.rank_at(D, o2)
                if r2:
                    placebo.append(r2)
    print(f'  placebo median (each run at the OTHER onsets): '
          f'{S2.median(placebo):.3f}  (n={len(placebo)})')

    print('S2b sequential tests (thresholds = 1.1 x max over this campaign\'s controls):')
    for tname, (fn, on_inc) in S2.TESTS.items():
        thr = 0.0
        for c in have_c:
            thr = max(thr, max(v for _, v in fn(S2.stdz(*S[c], on_inc))))
        thr *= 1.1
        alarms, aligns = 0, []
        for k, o in have_k.items():
            trace = fn(S2.stdz(*S[k], on_inc))
            hit = next((e for e, v in trace if v > thr), None)
            if hit is not None:
                alarms += 1
                aligns.append(f'{hit - o:+d}')
        print(f'  {tname:12s} thr {thr:9.4f}   alarms {alarms}/{len(have_k)}   '
              f'ctrl 0/{len(have_c)}   alarm-onset: {" ".join(aligns) if aligns else "--"}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--rung', type=int, default=100)
    a = ap.parse_args()
    S = load(a.rung)
    print(f'LiDAR rung N={a.rung} spots; {len(S)} runs with series')
    audit(f'21x  N={a.rung}', KNEE_21, CTRL_21, S)
    audit(f'4x   N={a.rung}', KNEE_4, CTRL_4, S)
