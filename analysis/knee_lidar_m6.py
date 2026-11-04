"""M6-style threshold-transfer sensitivity ON THE LIDAR SERIES.

M6 (2026-08-23) ran this fragility test on loss / SIGReg-train /
SIGReg-eval only. The LiDAR headline never got it. Same design, same
imported statistics (knee_s2_monitors: stdz, TESTS), same run sets as
the LiDAR audit (knee_lidar_audit: KNEE_21/CTRL_21, KNEE_4/CTRL_4) --
nothing reimplemented.

Two calibration sets per campaign, 1.1 x max convention in both:
  A. the campaign's four mu=0.97 controls   (the standing convention)
  B. the campaign's single FLAT mu=0 run    (21x: seed 5; 4x: seed 10)

Only ONE flat mu=0 run exists per campaign, so a 1.1x-max over it has no
spread to absorb. As in M6 this is SENSITIVITY ONLY, explicitly not a
verdict.

Usage: python knee_lidar_m6.py [--rung 100]
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import knee_lidar_audit as AUD  # noqa: E402
import knee_s2_monitors as S2  # noqa: E402

FLAT = {'21x': 'rigb_lejp_p1s100_nano_5', '4x': 'rigb_lejp_p1s100_p384_10'}


def thr_over(names, fn, on_inc, S):
    """1.1 x max-over-calibration-set, the standing convention."""
    t = 0.0
    for c in names:
        t = max(t, max(v for _, v in fn(S2.stdz(*S[c], on_inc))))
    return 1.1 * t


def alarms(knee, fn, on_inc, S, thr):
    n = 0
    for k in knee:
        if k not in S:
            continue
        if any(v > thr for _, v in fn(S2.stdz(*S[k], on_inc))):
            n += 1
    return n


def run(tag, knee, ctrl, S):
    have_k = [k for k in knee if k in S]
    have_c = [c for c in ctrl if c in S]
    flat = FLAT[tag]
    if flat not in S:
        print(f'\n[{tag}] flat run {flat} has no LiDAR series -- skipped')
        return
    print(f'\n=== {tag} LiDAR threshold transfer ===')
    print(f'calibration A: {len(have_c)} mu=0.97 controls | '
          f'calibration B: 1 flat mu=0 run ({flat.split("p1s100_")[-1]})')
    # Control false-alarm counts are printed beside the knee counts. A
    # lower threshold mechanically yields more alarms, so a knee count is
    # uninterpretable without the rate the same threshold produces on runs
    # with no event. Under calibration B the flat run itself is the
    # calibrator, so it is excluded from its own false-alarm column.
    print(f'{"test":8s} {"thrA(mu097 x%d)" % len(have_c):>16s} '
          f'{"thrB(flat x1)":>14s} {"knee A":>8s} {"knee B":>8s} '
          f'{"ctrlFA A":>9s} {"ctrlFA B":>9s}   (knee n=%d, ctrl n=%d)'
          % (len(have_k), len(have_c)))
    wanted = [t for t in S2.TESTS if t.upper() in ('CUSUM', 'EWMA')]
    assert len(wanted) == 2, f'expected CUSUM and EWMA, got {wanted}'
    for tname in wanted:                  # M6 covered CUSUM and EWMA only
        fn, on_inc = S2.TESTS[tname]
        ta = thr_over(have_c, fn, on_inc, S)
        tb = thr_over([flat], fn, on_inc, S)
        print(f'{tname:8s} {ta:16.4f} {tb:14.4f} '
              f'{alarms(have_k, fn, on_inc, S, ta):8d} '
              f'{alarms(have_k, fn, on_inc, S, tb):8d} '
              f'{alarms(have_c, fn, on_inc, S, ta):9d} '
              f'{alarms(have_c, fn, on_inc, S, tb):9d}')
    print('SENSITIVITY ONLY -- explicitly not a verdict (M6 framing).')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--rung', type=int, default=100)
    a = ap.parse_args()
    S = AUD.load(a.rung)
    print(f'LiDAR rung N={a.rung}; {len(S)} runs with series')
    run('21x', AUD.KNEE_21, AUD.CTRL_21, S)
    run('4x', AUD.KNEE_4, AUD.CTRL_4, S)
