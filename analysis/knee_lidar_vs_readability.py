"""Does the LiDAR level track energy readability across checkpoints?
(2026-09-23; reads existing CSVs only, no GPU, writes nothing.)

Question from the monitor session: with the profile shape redrawn inside a
replica group (csv/knee_lidar.csv), the groups differ only in energy (the rig
fixes the particle position), and shape plus element noise is within-group
scatter. In a linear-Gaussian picture LiDAR is then set by one number, the
energy signal-to-scatter ratio s_E = var(e) b_e' S_w^-1 b_e, which equals
R2_E / (1 - R2_E) for the best linear read-out of energy from a single
render with shape random: the probe's own R^2(E). That picture predicts
LiDAR falls as R^2(E) rises and rises as it falls, the same way before and
after the event, unless the within-group bulk (how many feature directions
carry within-group scatter) also changes.

Inputs (read only): csv/knee_traj_full.csv (ridge R^2 per checkpoint on the
run's own probe cache), csv/knee_lidar.csv and csv/knee_lidar_sharedshape.csv
(N = 100). Runs: the knee runs present in both (onsets from
knee_lidar_audit), plus the event-free runs present in both.

Printed: rank correlations of LiDAR with R^2(E), z-mean R^2 and mean CCA over
post-warm-up checkpoints; LiDAR at matched R^2(E) before training settles
(epochs <= 20) and after the onset.
"""
import sys

sys.dont_write_bytecode = True

import csv  # noqa: E402
import os  # noqa: E402
from statistics import median  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
sys.path.insert(0, _HERE)
import knee_lidar_audit as AUD  # noqa: E402

TRAJ = os.path.join(paths.CSV, 'knee_traj_full.csv')
LID = {'existing': os.path.join(paths.CSV, 'knee_lidar.csv'),
       'shared': os.path.join(paths.CSV, 'knee_lidar_sharedshape.csv')}


def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = 0.5 * (i + j) + 1
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den


def load():
    traj = {}
    with open(TRAJ) as fh:
        for r in csv.DictReader(fh):
            traj[(r['run'].split('xxx')[0], int(r['epoch']))] = (
                float(r['ridge_r2_E']),
                0.5 * (float(r['ridge_r2_z1']) + float(r['ridge_r2_z2'])),
                float(r['mean_cca']))
    lid = {}
    for tag, p in LID.items():
        with open(p) as fh:
            for r in csv.DictReader(fh):
                if r['n_spots'] == '100':
                    lid[(tag, r['run'].split('xxx')[0], int(r['epoch']))] = float(r['lidar'])
    return traj, lid


def main():
    traj, lid = load()
    onset = dict(AUD.KNEE_21)
    onset.update(AUD.KNEE_4)
    runs = sorted({k[0] for k in traj})
    knee = [r for r in runs if r in onset]
    free = [r for r in runs if r not in onset]
    print(f'runs with both series: {len(knee)} knee, {len(free)} event-free')
    print('knee: ' + ', '.join(r.replace('rigb_lejp_p1s100_', '').replace('rigd_lejp_p1s100_', '')
                               for r in knee))

    def rows(rs, tag, lo=21, hi=10 ** 9, phase=None):
        out = []
        for r in rs:
            for (rr, e), (r2e, r2z, cca) in traj.items():
                if rr != r or not (lo <= e <= hi) or (tag, r, e) not in lid:
                    continue
                if phase == 'pre' and e >= onset[r]:
                    continue
                if phase == 'post' and e < onset[r]:
                    continue
                out.append((r, e, r2e, r2z, cca, lid[(tag, r, e)]))
        return out

    print('\n== rank correlation of LiDAR with readability, checkpoints e > 20 ==')
    for tag in ('existing', 'shared'):
        for name, rs in (('knee', knee), ('event-free', free)):
            d = rows(rs, tag)
            L = [x[5] for x in d]
            within = []
            for r in rs:
                dr = [x for x in d if x[0] == r]
                if len(dr) >= 5:
                    within.append(spearman([x[2] for x in dr], [x[5] for x in dr]))
            print(f'  {tag:8s} {name:10s} n={len(d):4d}  pooled rho(LiDAR, R2_E) '
                  f'{spearman([x[2] for x in d], L):+.3f}  (z-mean R2 '
                  f'{spearman([x[3] for x in d], L):+.3f}, CCA '
                  f'{spearman([x[4] for x in d], L):+.3f});  within-run rho(LiDAR, R2_E): '
                  f'median {median(within):+.3f}, negative in '
                  f'{sum(w < 0 for w in within)}/{len(within)}')

    print('\n== matched energy readability: early checkpoints (e <= 20) vs after the onset ==')
    print('   per knee run: R2_E range and LiDAR (existing) in each set; the early set is before')
    print('   any onset, so a pure function of R2_E would give the same LiDAR at the same R2_E')
    for r in knee:
        early = [(e, traj[(r, e)][0], lid[('existing', r, e)]) for e in range(2, 21)
                 if (r, e) in traj and ('existing', r, e) in lid]
        post = [(e, traj[(r, e)][0], lid[('existing', r, e)]) for (rr, e) in traj
                if rr == r and e >= onset[r] and ('existing', r, e) in lid]
        if not early or not post:
            continue
        lo = max(min(x[1] for x in early), min(x[1] for x in post))
        hi = min(max(x[1] for x in early), max(x[1] for x in post))
        ea = [x for x in early if lo <= x[1] <= hi]
        pa = [x for x in post if lo <= x[1] <= hi]
        tag = r.replace('rigb_lejp_p1s100_', '').replace('rigd_lejp_p1s100_', '')
        if ea and pa:
            print(f'  {tag:14s} overlap R2_E {lo:.3f}-{hi:.3f}: early n={len(ea)} LiDAR median '
                  f'{median(x[2] for x in ea):.2f} | post n={len(pa)} LiDAR median '
                  f'{median(x[2] for x in pa):.2f}')
        else:
            print(f'  {tag:14s} no early checkpoint inside the post-onset R2_E range: early '
                  f'{min(x[1] for x in early):.3f}-{max(x[1] for x in early):.3f}, post '
                  f'{min(x[1] for x in post):.3f}-{max(x[1] for x in post):.3f}')
    drift_by_mu()


def drift_by_mu():
    """Raw change (median over grid epochs > final - 30 minus median over
    epochs 21-60; the 2026-09-05 audit's section 4b statistic) of the
    event-free runs, grouped by training mu, both series, N = 100."""
    import re
    import knee_lidar_audit_sharedshape as A2

    def change(S, run):
        eps, xs = S[run]
        d = dict(zip(eps, xs))
        fin = max(eps)
        return (median([d[e] for e in eps if e > fin - 30])
                - median([d[e] for e in eps if 21 <= e <= 60]))

    def mu(run):
        m = re.search(r'_mu0(\d+)_', run)
        return {'4': 0.4, '7': 0.7, '9': 0.9, '97': 0.97}[m.group(1)] if m else 0.0

    free = A2.HELD_21 + A2.HELD_4 + AUD.CTRL_21 + AUD.CTRL_4
    out = {}
    for tag, p in LID.items():
        AUD.LID = p
        S = AUD.load(100)
        for r in free:
            out.setdefault(r, {})[tag] = change(S, r)
    print('\n== event-free raw LiDAR change by training mu, N = 100: existing -> shared shape ==')
    for m in (0.0, 0.4, 0.7, 0.9, 0.97):
        rs = [r for r in free if mu(r) == m]
        print(f'  mu={m:4.2f}: ' + '; '.join(
            f"{A2.short(r)} {out[r]['existing']:+.2f} -> {out[r]['shared']:+.2f}" for r in rs))


if __name__ == '__main__':
    main()
