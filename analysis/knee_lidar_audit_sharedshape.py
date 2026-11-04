"""S2a/S2b LiDAR audit on the SHARED-SHAPE series (2026-09-23).

Machinery unchanged: knee_lidar_audit is imported and only its input path
AUD.LID is pointed at results/knee_lidar_sharedshape.csv (written by
knee_lidar_sharedshape.py). Thresholds are recalibrated on the same four
mu = 0.97 controls per campaign, 1.1 x max, by AUD.audit itself; the four
tests are knee_s2_monitors.TESTS.

Added printout (no new statistic):
  * per run in the CSV and per test: peak statistic, the threshold it is
    scored against, first alarm epoch. Thresholds: 4x runs (p384) against
    the 4x controls; every other run (21x nano, MAE, ViT-Tiny) against the
    21x controls, as knee_lidar_mae.py and the 2026-09-05 audit did.
  * counts per population: knee runs (AUD.KNEE_21 / AUD.KNEE_4, with the
    alarm - onset distances), the 11 held-out LeJEPA runs (7 at 21x, 4 at
    4x, as defined in the 2026-09-05 audit), the 10
    MAE runs, the ViT-Tiny flat run (tiny_1). tiny_2 (a knee run) is
    printed but, as in the record, not counted.
  * LiDAR level at the PRE/POST anchors of the knee runs (PRE = last grid
    epoch <= onset - 20, POST = final checkpoint, as in the record), for
    this series and for the existing results/knee_lidar.csv series.

Usage:
  python knee_lidar_audit_sharedshape.py                       # shared shape
  python knee_lidar_audit_sharedshape.py --csv ../results/knee_lidar.csv   # same
      printout on the existing series (read only), for the side-by-side
"""
import sys

sys.dont_write_bytecode = True      # leave analysis/__pycache__ untouched

import argparse  # noqa: E402
import os  # noqa: E402
from statistics import median  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
sys.path.insert(0, _HERE)
import knee_lidar_audit as AUD  # noqa: E402
import knee_s2_monitors as S2  # noqa: E402

NEW = os.path.join(paths.CSV, 'knee_lidar_sharedshape.csv')
OLD = os.path.join(paths.CSV, 'knee_lidar.csv')

HELD_21 = ['rigb_lejp_p1s100_nano_5', 'rigd_lejp_p1s100_mu04_nano_1',
           'rigd_lejp_p1s100_mu04_nano_2', 'rigd_lejp_p1s100_mu07_nano_3',
           'rigd_lejp_p1s100_mu09_nano_1', 'rigd_lejp_p1s100_mu09_nano_2',
           'rigd_lejp_p1s100_mu09_nano_3']
HELD_21 += ['rigb_lejp_p1s100_nano_10', 'rigb_lejp_p1s100_nano_12']  # denominator runs (2026-10-13; no decline)
HELD_4 = ['rigb_lejp_p1s100_p384_10', 'rigd_lejp_p1s100_mu07_p384_2',
          'rigd_lejp_p1s100_mu09_p384_1', 'rigd_lejp_p1s100_mu09_p384_2']
MAE = [f'rigb_mae_p1s100_long_nano_{s}' for s in range(1, 11)]
TINY_FLAT = ['rigb_lejp_p1s100_tiny_1']
TINY_KNEE = ['rigb_lejp_p1s100_tiny_2']
TESTS = list(S2.TESTS)


def campaign(run):
    return '4x' if '_p384_' in run else '21x'


def role(run):
    if run in AUD.KNEE_21 or run in AUD.KNEE_4:
        return 'knee'
    if run in AUD.CTRL_21 or run in AUD.CTRL_4:
        return 'control'
    if run in HELD_21 or run in HELD_4:
        return 'heldout'
    if run in MAE:
        return 'mae'
    if run in TINY_FLAT:
        return 'tiny_flat'
    if run in TINY_KNEE:
        return 'tiny_knee'
    return 'other'


def thresholds(S):
    """1.1 x max over each campaign's four mu=0.97 controls (the rule in
    knee_lidar_audit.audit, recomputed here so every run can be scored)."""
    thr = {}
    for camp, ctrl in (('21x', AUD.CTRL_21), ('4x', AUD.CTRL_4)):
        have = [c for c in ctrl if c in S]
        assert len(have) == 4, (camp, have)
        thr[camp] = {}
        for t, (fn, on_inc) in S2.TESTS.items():
            m = 0.0
            for c in have:
                m = max(m, max(v for _, v in fn(S2.stdz(*S[c], on_inc))))
            thr[camp][t] = 1.1 * m
    return thr


def score(S, thr):
    """run -> test -> (peak, threshold, first alarm epoch or None)."""
    out = {}
    for run in S:
        out[run] = {}
        th = thr[campaign(run)]
        for t, (fn, on_inc) in S2.TESTS.items():
            trace = fn(S2.stdz(*S[run], on_inc))
            peak = max(v for _, v in trace)
            hit = next((e for e, v in trace if v > th[t]), None)
            out[run][t] = (peak, th[t], hit)
    return out


def short(run):
    return run.replace('rigb_lejp_p1s100_', '').replace('rigd_lejp_p1s100_', '') \
              .replace('rigb_mae_p1s100_long_nano_', 'mae_')


def report(rung, S, thr, sc):
    print(f'\n##### N={rung}: per-run peak / threshold / first alarm epoch')
    print('thresholds: ' + '; '.join(
        f'{c} ' + ' '.join(f'{t} {thr[c][t]:.4f}' for t in TESTS) for c in thr))
    order = {'knee': 0, 'control': 1, 'heldout': 2, 'mae': 3, 'tiny_flat': 4,
             'tiny_knee': 5, 'other': 6}
    runs = sorted(S, key=lambda r: (order[role(r)], campaign(r), r))
    print(f'{"run":24s} {"role":9s} {"thr":4s} ' + ' '.join(
        f'{t + " peak/alarm":>22s}' for t in TESTS) + '  onset')
    for r in runs:
        onset = AUD.KNEE_21.get(r, AUD.KNEE_4.get(r))
        cells = []
        for t in TESTS:
            pk, th, hit = sc[r][t]
            cells.append(f'{pk:11.3f}/{("@" + str(hit)) if hit is not None else "-":>9s}')
        print(f'{short(r):24s} {role(r):9s} {campaign(r):4s} ' + ' '.join(
            f'{c:>22s}' for c in cells) + (f'  {onset}' if onset else ''))

    print(f'\n##### N={rung}: counts')
    for camp, knee in (('21x', AUD.KNEE_21), ('4x', AUD.KNEE_4)):
        have = [k for k in knee if k in S]
        for t in TESTS:
            d = [sc[k][t][2] - knee[k] for k in have if sc[k][t][2] is not None]
            print(f'  knee {camp:3s} {t:12s} {len(d):2d}/{len(have):<2d} '
                  f'alarm-onset median {median(d) if d else "-"}  '
                  f'[{" ".join(f"{x:+d}" for x in sorted(d))}]')
    for t in TESTS:
        d = [sc[k][t][2] - o for knee in (AUD.KNEE_21, AUD.KNEE_4)
             for k, o in knee.items() if k in S and sc[k][t][2] is not None]
        print(f'  knee both {t:12s} alarm-onset median over {len(d)} alarms: '
              f'{median(d) if d else "-"}')
    pops = [('heldout LeJEPA 21x', HELD_21), ('heldout LeJEPA 4x', HELD_4),
            ('heldout LeJEPA all', HELD_21 + HELD_4), ('MAE', MAE),
            ('ViT-Tiny flat', TINY_FLAT),
            ('pooled heldout+MAE+tiny', HELD_21 + HELD_4 + MAE + TINY_FLAT)]
    for name, pop in pops:
        have = [r for r in pop if r in S]
        parts = []
        for t in TESTS:
            fired = [r for r in have if sc[r][t][2] is not None]
            parts.append(f'{t} {len(fired)}/{len(have)}'
                         + (f' ({", ".join(short(r) + "@" + str(sc[r][t][2]) for r in fired)})'
                            if fired else ''))
        anyn = sum(any(sc[r][t][2] is not None for t in TESTS) for r in have)
        print(f'  {name:24s} ' + '; '.join(parts) + f'; any of four {anyn}/{len(have)}')
    ctrl_alarms = sum(sc[c][t][2] is not None for c in AUD.CTRL_21 + AUD.CTRL_4
                      if c in S for t in TESTS)
    print(f'  controls: alarms over 8 runs x 4 tests = {ctrl_alarms} '
          f'(0 by construction)')


def levels(rung):
    """LiDAR at the PRE/POST anchors of every knee run, both series."""
    AUD.LID = OLD
    So = AUD.load(rung)
    AUD.LID = NEW
    Sn = AUD.load(rung)
    print(f'\n##### N={rung}: LiDAR level at PRE (last grid epoch <= onset-20), '
          f'onset, POST (final); existing | shared shape')
    up = {'old': 0, 'new': 0}
    n = 0
    for camp, knee in (('21x', AUD.KNEE_21), ('4x', AUD.KNEE_4)):
        for k, o in knee.items():
            if k not in So or k not in Sn:
                continue
            vals = {}
            for tag, S in (('old', So), ('new', Sn)):
                eps, xs = S[k]
                d = dict(zip(eps, xs))
                pre = max(e for e in eps if e <= o - 20)
                vals[tag] = (pre, d[pre], d.get(o, float('nan')), max(eps), d[max(eps)])
                up[tag] += d[max(eps)] > d[pre]
            n += 1
            (pe, a0, a1, fe, a2), (_, b0, b1, _, b2) = vals['old'], vals['new']
            print(f'  {camp:3s} {short(k):22s} onset {o:3d} PRE ep{pe:<3d} POST ep{fe:<3d}  '
                  f'existing {a0:7.3f} {a1:7.3f} {a2:7.3f} ({a2 - a0:+7.3f})  |  '
                  f'shared {b0:7.3f} {b1:7.3f} {b2:7.3f} ({b2 - b0:+7.3f})')
    print(f'  POST > PRE: existing {up["old"]}/{n}, shared shape {up["new"]}/{n}')
    AUD.LID = NEW


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=NEW)
    a = ap.parse_args()
    AUD.LID = os.path.abspath(a.csv)
    print(f'series: {AUD.LID}')
    for rung in (100, 1000):
        S = AUD.load(rung)
        print(f'\n======== LiDAR rung N={rung}: {len(S)} runs with series ========')
        AUD.audit(f'21x  N={rung}', AUD.KNEE_21, AUD.CTRL_21, S)
        AUD.audit(f'4x   N={rung}', AUD.KNEE_4, AUD.CTRL_4, S)
        thr = thresholds(S)
        report(rung, S, thr, score(S, thr))
    if os.path.abspath(a.csv) == NEW:
        for rung in (100, 1000):
            levels(rung)


if __name__ == '__main__':
    main()
