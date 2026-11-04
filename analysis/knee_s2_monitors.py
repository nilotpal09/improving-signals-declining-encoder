"""S2: monitor audit at the knee, per the definitions printed in the record
(2026-08-19). Inputs: per-epoch monitor series from each run's
lejepa_diag_r0.csv; onsets from S1 (rule R_A). Outputs: printed tables
(a) local-deviation rank at onset + placebo + power, (b) sequential-test
discrimination with control-calibrated thresholds + alarm alignment.
"""
import csv
import glob
import os
import paths  # noqa: E402

ROOT = paths.RUNS
MONS = ['loss', 'lejepa_sigreg', 'sigreg_eval', 'feat_std_event', 'rankme_event']
KNEE = {  # pattern -> S1 R_A onset
    'rigb_lejp_p1s100_nano_1xxx*': 90,
    'rigb_lejp_p1s100_nano_2xxx*': 110,
    'rigb_lejp_p1s100_nano_3xxx*': 100,
    'rigb_lejp_p1s100_long_nano_1xxx*': 160,
}
CTRL = ['rigd_lejp_p1s100_mu097_nano_1xxx*', 'rigd_lejp_p1s100_mu097_nano_2xxx*',
        'rigd_lejp_p1s100_mu097_nano_3xxx*', 'rigd_lejp_p1s100_mu097_long_nano_1xxx*']
WARM = 20


def series(pat):
    d = glob.glob(os.path.join(ROOT, pat))[0]
    by = {}
    for r in csv.DictReader(open(os.path.join(d, 'lejepa_diag_r0.csv'))):
        e = int(r['epoch'])
        for m in MONS:
            by.setdefault(m, {}).setdefault(e, []).append(float(r[m]))
    out = {}
    for m in MONS:
        eps = sorted(by[m])
        out[m] = (eps, [sum(by[m][e]) / len(by[m][e]) for e in eps])
    return out


def median(v):
    s = sorted(v)
    return s[len(s) // 2]


def local_dev(eps, xs):
    """D(e) normalized by median routine step, epochs > WARM, edges excluded."""
    idx = [i for i in range(1, len(eps) - 1) if eps[i] > WARM]
    steps = [abs(xs[i] - xs[i - 1]) for i in range(1, len(eps)) if eps[i] > WARM]
    norm = median(steps) or 1e-12
    D = {eps[i]: abs(xs[i] - 0.5 * (xs[i - 1] + xs[i + 1])) / norm for i in idx}
    return D


def rank_at(D, onset, w=3):
    win = [D[e] for e in D if onset - w <= e <= onset + w]
    if not win:
        return None
    peak = max(win)
    allv = sorted(D.values(), reverse=True)
    return (allv.index(peak) + 1) / len(allv)   # non-strict enough: first match


# ---------------- S2b
def stdz(eps, xs, on_increments):
    v = ([xs[i] - xs[i - 1] for i in range(1, len(eps))] if on_increments
         else xs[1:])
    e2 = eps[1:]
    cal = [x for e, x in zip(e2, v) if 21 <= e <= 60]
    mu = sum(cal) / len(cal)
    sd = (sum((x - mu) ** 2 for x in cal) / len(cal)) ** 0.5 or 1e-12
    return [(e, (x - mu) / sd) for e, x in zip(e2, v) if e > WARM]


def cusum(z):
    sp = sn = 0.0
    out = []
    for e, v in z:
        sp = max(0.0, sp + v - 0.5)
        sn = max(0.0, sn - v - 0.5)
        out.append((e, max(sp, sn)))
    return out


def page_hinkley(z):
    mT = 0.0
    mn = mx = 0.0
    out = []
    for e, v in z:
        mT += v - 0.5
        mn = min(mn, mT)
        mx = max(mx, mT)
        out.append((e, max(mT - mn, mx - mT)))
    return out


def ewma(z, lam=0.2):
    s = 0.0
    out = []
    for e, v in z:
        s = lam * v + (1 - lam) * s
        out.append((e, abs(s)))
    return out


def wslope(z, w=11):
    out = []
    vs = [v for _, v in z]
    es = [e for e, _ in z]
    for i in range(w, len(vs)):
        ys = vs[i - w:i]
        xs = list(range(w))
        mx = sum(xs) / w
        my = sum(ys) / w
        num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        den = sum((a - mx) ** 2 for a in xs)
        out.append((es[i], abs(num / den)))
    return out


TESTS = {'CUSUM': (cusum, True), 'PageHinkley': (page_hinkley, False),
         'EWMA': (ewma, True), 'slope11': (wslope, False)}


def s2a():
    # ---------------- S2a
    print('== S2a local-deviation rank fraction at onset (1/N = top of run) ==')
    print(f"{'run':12s} " + ' '.join(f'{m:>14s}' for m in MONS))
    placebo = {m: [] for m in MONS}
    for pat, onset in KNEE.items():
        s = series(pat)
        tag = pat.split('xxx')[0].replace('rigb_lejp_p1s100_', '').replace('_nano', '')
        row = f'{tag:12s} '
        for m in MONS:
            D = local_dev(*s[m])
            r = rank_at(D, onset)
            row += f'{r:14.3f}' if r else f'{"--":>14s}'
            for pat2, o2 in KNEE.items():
                if pat2 != pat:
                    r2 = rank_at(D, o2)
                    if r2:
                        placebo[m].append(r2)
        print(row)
    print('placebo (each run at the OTHER runs\' onsets), median rank fraction:')
    print(' '.join(f'{m}={median(placebo[m]):.3f}' for m in MONS))

    print('power check (2x routine step injected at ep120, control runs):')
    for pat in CTRL[:2]:
        s = series(pat)
        tag = 'ctrl_' + pat.split('xxx')[0].rsplit('_', 1)[-1]
        row = f'{tag:12s} '
        for m in MONS:
            eps, xs = s[m]
            stepn = median([abs(xs[i] - xs[i - 1]) for i in range(1, len(eps))
                            if eps[i] > WARM])
            xs2 = [x + (2 * stepn if e == 120 else 0) for e, x in zip(eps, xs)]
            D = local_dev(eps, xs2)
            r = rank_at(D, 120, w=1)
            row += f'{r:14.3f}' if r else f'{"--":>14s}'
        print(row)



def s2b():
    print('\n== S2b sequential tests (thresholds = 1.1 x max over 4 controls) ==')
    print(f"{'test':12s} {'monitor':>15s} {'knee alarms':>12s} {'ctrl alarms':>12s} "
          f"{'alarm-onset (per knee run)':>30s}")
    for tname, (fn, on_inc) in TESTS.items():
        for m in MONS:
            thr = 0.0
            for pat in CTRL:
                eps, xs = series(pat)[m]
                trace = fn(stdz(eps, xs, on_inc))
                thr = max(thr, max(v for _, v in trace))
            thr *= 1.1
            kn_alarm, aligns = 0, []
            for pat, onset in KNEE.items():
                eps, xs = series(pat)[m]
                trace = fn(stdz(eps, xs, on_inc))
                hit = next((e for e, v in trace if v > thr), None)
                if hit is not None:
                    kn_alarm += 1
                    aligns.append(f'{hit - onset:+d}')
                else:
                    aligns.append('--')
            print(f'{tname:12s} {m:>15s} {kn_alarm:>10d}/4 {0:>10d}/4 '
                  f'{" ".join(a for a in aligns):>30s}')


if __name__ == '__main__':
    s2a()
    s2b()
