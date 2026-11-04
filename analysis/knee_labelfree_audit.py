"""Audit of the label-free metrics (2026-09-23). Reads
results/knee_labelfree_metrics.csv and the stored
pooled features; writes one new CSV (readability at every checkpoint) and
prints the audit. No GPU, no training.

Machinery imported unchanged:
  * knee_s2_monitors: local_dev, rank_at, median (S2a), TESTS = CUSUM,
    Page-Hinkley, EWMA, slope11 on stdz series (S2b).
  * knee_lidar_audit: KNEE_21 / KNEE_4 (onsets), CTRL_21 / CTRL_4 (the four
    mu = 0.97 controls per campaign); S2a loop transcribed from its audit()
    only to collect the numbers it prints.
  * knee_lidar_audit_sharedshape: thresholds (1.1 x max over each campaign's
    four controls), score, populations HELD_21 + HELD_4 (11 held-out
    LeJEPA), MAE (10), TINY_FLAT.
  * knee_lidar_replica_seed.counts_from_S: the per-population counts used in
    today's items 1 to 3 (alarms within one grid step = |alarm - onset| <= 10).
  * knee_rank_estimators_prepost.ladder_epochs: PRE and POST epochs of the 13
    capacity-tally runs (item 3b).
  * knee_lidar_vs_readability.spearman: the rank correlation used for LiDAR.

Readability at every checkpoint: wp1_metrics.wp1_metric_row on the stored
fitting-split features (fit) and evaluation-split features (report), with
the run's own dial-matched cache targets, exactly as knee_s7_trajfull.py
computed results/knee_traj_full.csv. Written once to
results/knee_labelfree_readability.csv (refuses to overwrite); checked against
knee_traj_full.csv on the keys the two share.

Rank correlation, as for LiDAR (knee_lidar_vs_readability.py): Spearman over
checkpoints with epoch > 20, pooled over a group's checkpoints and within
each run (median over runs, count negative). Groups: event-free LeJEPA runs
(8 mu = 0.97 controls, 11 held-out, ViT-Tiny flat), MAE (10), knee runs (27).

LiDAR rows (existing and shared-shape series, N = 100) go through the same
code as a check against today's reported counts.

kappa sensitivity: kappa recomputed from the stored features with the zero
tolerance at 1x, 10x (the pass's) and 100x the unit eps32 * rms(stored X) *
(sqrt N + sqrt d), each series through the same CUSUM counts; the 10x
series is checked against the pass's values.

Usage:
  python knee_labelfree_audit.py --readability      # once; writes the CSV + gate
  python knee_labelfree_audit.py --gate             # the readability gate again
  python knee_labelfree_audit.py --lidar-only       # audit code on the LiDAR rows only
  python knee_labelfree_audit.py [--md out.md]      # the audit
"""
import sys

sys.dont_write_bytecode = True      # leave analysis/__pycache__ untouched

import argparse  # noqa: E402
import csv  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
from collections import defaultdict  # noqa: E402
from statistics import median as smedian  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
sys.path.insert(0, _HERE)
import knee_lidar_audit as AUD  # noqa: E402
import knee_s2_monitors as S2  # noqa: E402
import knee_lidar_audit_sharedshape as A2  # noqa: E402
import knee_lidar_replica_seed as RS  # noqa: E402
import knee_lidar_vs_readability as LVR  # noqa: E402

MET = os.path.join(paths.CSV, 'knee_labelfree_metrics.csv')
READ = os.path.join(paths.CSV, 'knee_labelfree_readability.csv')
TRAJ = os.path.join(paths.CSV, 'knee_traj_full.csv')
LIDAR = {'lidar redrawn': os.path.join(paths.CSV, 'knee_lidar.csv'),
         'lidar shared': os.path.join(paths.CSV, 'knee_lidar_sharedshape.csv')}
CACHE_DIR = paths.CACHE
FEAT_DIR = os.path.join(paths.WORK, 'labelfree_feats')  # denominator runs: 58 symlinks + 3 new
READ_COLS = ['run', 'epoch', 'mean_cca', 'cca_c1', 'cca_c2', 'cca_c3', 'orth_err', 'orth_scale',
             'ridge_r2_z1', 'ridge_r2_z2', 'ridge_r2_E']

KNEE = dict(AUD.KNEE_21)
KNEE.update(AUD.KNEE_4)
FREE = AUD.CTRL_21 + AUD.CTRL_4 + A2.HELD_21 + A2.HELD_4 + A2.TINY_FLAT
GROUPS = [('event-free LeJEPA', FREE), ('MAE', A2.MAE), ('knee', list(KNEE))]
TAG = {  # (metric, variant) -> P = as published, C = as the authors' code, S = survey
    ('alpha_req', 'centred'): 'P S', ('alpha_req', 'code_wls'): 'C',
    ('rankme', 'pub_uncentred'): 'P S',
    ('ne_sum', 'centred'): 'P S', ('kappa', 'uncentred'): 'P S',
    ('self_cluster', 'sq_uncentred'): 'P (its constants)',
    ('self_cluster', 'lit_uncentred'): 'P (as printed), S',
    ('dse', 'code'): 'C S', ('dse', 'paper'): 'P form, sigma 10',
    ('dse', 'paper_codebw'): 'P form, code width',
    ('id_twonn', 'fit'): 'P C', ('id_twonn', 'mle'): 'S (as written)'}


# ============================================================ readability

def build_readability():
    import torch
    from wp1_metrics import wp1_metric_row
    assert not os.path.exists(READ), f'refusing to overwrite {READ}'
    caches, rows = {}, []
    files = sorted(f for f in os.listdir(FEAT_DIR) if f.endswith('.pt'))
    assert len(files) == 61, len(files)  # denominator runs (2026-10-13): 58 + nano_10-12
    for f in files:
        F = torch.load(os.path.join(FEAT_DIR, f), map_location='cpu', weights_only=True)
        if F['cache'] not in caches:
            c = torch.load(os.path.join(CACHE_DIR, F['cache']), map_location='cpu', weights_only=True)
            caches[F['cache']] = (c['y_tr'], c['y_ev'])
        y_tr, y_ev = caches[F['cache']]
        for i, ep in enumerate(F['epochs']):
            m = wp1_metric_row(F['feats_tr'][i], y_tr, F['feats_ev'][i], y_ev)
            rows.append([F['run'], ep, f"{m['mean_cca']:.6f}",
                         f"{m['cca_per_component'][0]:.6f}", f"{m['cca_per_component'][1]:.6f}",
                         f"{m['cca_per_component'][2]:.6f}", f"{m['orth_err']:.6f}",
                         f"{m['orth_scale']:.6e}", f"{m['ridge_r2_z1']:.6f}",
                         f"{m['ridge_r2_z2']:.6f}", f"{m['ridge_r2_E']:.6f}"])
        print(f"{F['run'].split('xxx')[0]:38s} {len(F['epochs'])} ckpts", flush=True)
    with open(READ + '.partial', 'w', newline='') as fh:
        w = csv.writer(fh, lineterminator='\n')
        w.writerow(READ_COLS)
        w.writerows(rows)
    os.replace(READ + '.partial', READ)
    print(f'wrote {READ}: {len(rows)} rows')
    gate_readability()


def gate_readability():
    new = {(r['run'], int(r['epoch'])): r for r in csv.DictReader(open(READ))}
    old = {(r['run'], int(r['epoch'])): r for r in csv.DictReader(open(TRAJ))}
    keys = sorted(set(new) & set(old))
    print(f'\nreadability gate: {len(keys)} (run, epoch) keys shared with knee_traj_full.csv '
          f'({len({k[0] for k in keys})} runs)')
    for c in READ_COLS[2:]:
        d = max(abs(float(new[k][c]) - float(old[k][c])) for k in keys)
        print(f'  {c:12s} max |new - knee_traj_full| = {d:.2e}')


def load_readability():
    out = {}
    for r in csv.DictReader(open(READ)):
        out[(r['run'].split('xxx')[0], int(r['epoch']))] = (float(r['ridge_r2_E']), float(r['mean_cca']))
    return out


# ============================================================ series

def load_metrics():
    by = defaultdict(lambda: defaultdict(dict))
    with open(MET) as fh:
        for r in csv.DictReader(fh):
            by[(r['metric'], r['variant'])][r['run'].split('xxx')[0]][int(r['epoch'])] = float(r['value'])
    out = {}
    for mv, runs in by.items():
        out[mv] = {k: (sorted(d), [d[e] for e in sorted(d)]) for k, d in runs.items()}
    return out


def load_lidar(path):
    AUD.LID = path
    return AUD.load(100)


def s2a(S, knee):
    """The S2a loop of knee_lidar_audit.audit, collecting what it prints."""
    at, placebo = [], []
    for k, o in knee.items():
        if k not in S:
            continue
        D = S2.local_dev(*S[k])
        r = S2.rank_at(D, o)
        if r:
            at.append(r)
        for k2, o2 in knee.items():
            if k2 != k and k2 in S:
                r2 = S2.rank_at(D, o2)
                if r2:
                    placebo.append(r2)
    return at, placebo


def prepost(S):
    import knee_rank_estimators_prepost as KR
    out = []
    for run, (pre, post) in KR.ladder_epochs().items():
        k = run.split('xxx')[0]
        eps, xs = S[k]
        d = dict(zip(eps, xs))
        out.append((k, d[pre], d[post]))
    return out


def kappa_sensitivity(M):
    """kappa at zero tolerances of 1x, 10x (the pass) and 100x the unit
    eps32 * rms(stored X) * (sqrt N + sqrt d), from the stored features; the
    10x series must equal the pass's values. Returns {(variant, mult): S} and
    the per-multiplier counts of checkpoints with n_excluded != 1."""
    import numpy as np
    import torch
    from knee_labelfree_metrics import kappa_tol
    out, nonone = defaultdict(dict), defaultdict(int)
    worst = 0.0
    for f in sorted(os.listdir(FEAT_DIR)):
        if not f.endswith('.pt'):
            continue
        F = torch.load(os.path.join(FEAT_DIR, f), map_location='cpu', weights_only=True)
        run = F['run'].split('xxx')[0]
        for i, ep in enumerate(F['epochs']):
            X = F['feats_tr'][i].double().numpy()
            unit = kappa_tol(X, 1.0)
            for v, Xv in (('uncentred', X), ('centred', X - X.mean(0))):
                s = np.linalg.svd(Xv, compute_uv=False)
                for mult in (1.0, 10.0, 100.0):
                    keep = s > mult * unit
                    out[(v, mult)].setdefault(run, {})[ep] = float(s[0] / s[keep].min())
                    nonone[(v, mult)] += int((~keep).sum() != 1)
                ref = M[('kappa', v)][run][1][M[('kappa', v)][run][0].index(ep)]
                worst = max(worst, abs(out[(v, 10.0)][run][ep] - ref) / ref)
    print(f'kappa sensitivity: max relative |recomputed 10x - pass value| = {worst:.1e} (pass CSV has 10 digits)')
    S = {k: {r: (sorted(d), [d[e] for e in sorted(d)]) for r, d in runs.items()} for k, runs in out.items()}
    return S, nonone


def correlations(S, rd):
    res = {}
    for gname, runs in GROUPS:
        for qi, qname in ((0, 'R2E'), (1, 'CCA')):
            pooled_m, pooled_q, within = [], [], []
            for r in runs:
                eps, xs = S[r]
                pairs = [(x, rd[(r, e)][qi]) for e, x in zip(eps, xs) if e > 20 and (r, e) in rd]
                pooled_m += [p[0] for p in pairs]
                pooled_q += [p[1] for p in pairs]
                if len(pairs) >= 5:
                    within.append(LVR.spearman([p[1] for p in pairs], [p[0] for p in pairs]))
            res[(gname, qname)] = (LVR.spearman(pooled_q, pooled_m), within, len(pooled_m))
    return res


# ============================================================ report

def fmt_pop(fired, n):
    return f'{len(fired)} / {n}' + (f' ({", ".join(A2.short(r) for r in fired)})' if fired else '')


def audit(md_path=None, lidar_only=False):
    series = []
    if not lidar_only:
        from knee_labelfree_metrics import VARIANTS
        order = {(m, v): i for i, (m, v, *_) in enumerate(VARIANTS)}
        M = load_metrics()
        assert set(M) == set(order), sorted(set(M) ^ set(order))
        series = [(f'{m} {v}', TAG.get((m, v), ''), M[(m, v)]) for (m, v) in sorted(M, key=order.get)]
    rd = load_readability()
    series += [(name, 'ref', load_lidar(p)) for name, p in LIDAR.items()]
    md = []

    # sanity on the series
    for name, _, S in series:
        assert len(S) == 58, (name, len(S))
        bad = sum(1 for eps, xs in S.values() for x in xs if not math.isfinite(x))
        if bad:
            print(f'!! {name}: {bad} non-finite values')

    md.append('### CUSUM\n')
    md.append('| metric, variant | P/C/S | threshold 21x / 4x | knee 21x | knee 4x | within one grid step '
              '(21x + 4x) | held-out LeJEPA | MAE | ViT-Tiny flat |')
    md.append('|---|---|---|---|---|---|---|---|---|')
    other = {t: [] for t in A2.TESTS if t != 'CUSUM'}
    anyrow = []
    s2arows = []
    for name, tag, S in series:
        c = RS.counts_from_S(S)
        thr = c['thr']
        k21, n21, d21 = c[('knee', '21x', 'CUSUM')]
        k4, n4, d4 = c[('knee', '4x', 'CUSUM')]
        near = f'{sum(abs(x) <= 10 for x in d21)} + {sum(abs(x) <= 10 for x in d4)}'
        md.append(f'| {name} | {tag} | {thr["21x"]["CUSUM"]:.2f} / {thr["4x"]["CUSUM"]:.2f} | '
                  f'{k21} / {n21} | {k4} / {n4} | {near} | {fmt_pop(*c[("heldout", "CUSUM")])} | '
                  f'{fmt_pop(*c[("mae", "CUSUM")])} | {fmt_pop(*c[("tiny", "CUSUM")])} |')
        for t in other:
            a21, m21, e21 = c[('knee', '21x', t)]
            a4, m4, e4 = c[('knee', '4x', t)]
            other[t].append(
                f'| {name} | {a21}/{m21}, {a4}/{m4} (near {sum(abs(x) <= 10 for x in e21)} + '
                f'{sum(abs(x) <= 10 for x in e4)}) | {len(c[("heldout", t)][0])}/{c[("heldout", t)][1]}, '
                f'{len(c[("mae", t)][0])}/{c[("mae", t)][1]}, {len(c[("tiny", t)][0])}/{c[("tiny", t)][1]} |')
        anyrow.append(f'| {name} | {fmt_pop(*c[("heldout", "any")])} | {fmt_pop(*c[("mae", "any")])} | '
                      f'{fmt_pop(*c[("tiny", "any")])} |')
        cells = []
        for camp, knee in (('21x', AUD.KNEE_21), ('4x', AUD.KNEE_4)):
            at, pl = s2a(S, knee)
            cells.append((f'{S2.median(at):.3f}' if at else '-') + f' (n={len(at)}) | '
                         + (f'{S2.median(pl):.3f}' if pl else '-') + f' (n={len(pl)})')
        s2arows.append(f'| {name} | ' + ' | '.join(cells) + ' |')
        print(f'{name:32s} CUSUM knee {k21}/{n21} {k4}/{n4} near {near}; alarm-onset 21x '
              f'{sorted(d21)} 4x {sorted(d4)}', flush=True)
    for t, rows in other.items():
        md.append(f'\n### {t}: knee 21x, knee 4x (within one grid step) | held-out LeJEPA, MAE, ViT-Tiny flat\n')
        md.append('| metric, variant | knee | false alarms |')
        md.append('|---|---|---|')
        md += rows
    md.append('\n### Any of the four tests, false-alarm populations\n')
    md.append('| metric, variant | held-out LeJEPA | MAE | ViT-Tiny flat |')
    md.append('|---|---|---|---|')
    md += anyrow
    md.append('\n### S2a: local-deviation rank fraction at onset (small = the onset step stands out), '
              'median over knee runs, and the placebo median (each run at the other runs\' onsets)\n')
    md.append('| metric, variant | 21x at onset | 21x placebo | 4x at onset | 4x placebo |')
    md.append('|---|---|---|---|---|')
    md += s2arows

    md.append('\n### PRE to POST on the 13 capacity-tally runs (ladder epochs, item 3b)\n')
    md.append('| metric, variant | PRE median | POST median | rose / fell | median change (range) | '
              'median % change (range) |')
    md.append('|---|---|---|---|---|---|')
    for name, tag, S in series:
        pp = prepost(S)
        ch = [b - a for _, a, b in pp]
        pos = all(a > 0 for _, a, _ in pp)
        pct = [100 * (b - a) / a for _, a, b in pp] if pos else None
        md.append(f'| {name} | {smedian(a for _, a, _ in pp):.4g} | {smedian(b for _, _, b in pp):.4g} | '
                  f'{sum(c > 0 for c in ch)} / {sum(c < 0 for c in ch)} | {smedian(ch):+.3g} '
                  f'({min(ch):+.3g} to {max(ch):+.3g}) | '
                  + (f'{smedian(pct):+.1f} ({min(pct):+.1f} to {max(pct):+.1f})' if pct else 'n/a') + ' |')

    md.append('\n### Rank correlation with readability, checkpoints with epoch > 20 '
              '(pooled rho; within-run median rho, runs negative / runs)\n')
    md.append('| metric, variant | event-free LeJEPA: R2(E) | event-free LeJEPA: mean CCA | MAE: R2(E) | '
              'MAE: mean CCA | knee: R2(E) | knee: mean CCA |')
    md.append('|---|---|---|---|---|---|---|')
    for name, tag, S in series:
        cr = correlations(S, rd)
        cells = []
        for g, _ in GROUPS:
            for q in ('R2E', 'CCA'):
                pooled, within, n = cr[(g, q)]
                cells.append(f'{pooled:+.2f}; {smedian(within):+.2f}, {sum(w < 0 for w in within)}/{len(within)}')
        md.append(f'| {name} | ' + ' | '.join(cells) + ' |')

    if not lidar_only:
        KS, nonone = kappa_sensitivity(M)
        md.append('\n### kappa: zero tolerance 1x, 10x (the pass) and 100x the unit eps32 * rms(stored X) * '
                  '(sqrt N + sqrt d)\n')
        md.append('| variant | tolerance | checkpoints with n_excluded other than 1 (of 2114) | CUSUM knee 21x | '
                  'knee 4x | within one grid step | held-out LeJEPA | MAE | ViT-Tiny flat | any of four: '
                  'held-out, MAE, ViT-Tiny |')
        md.append('|---|---|---|---|---|---|---|---|---|---|')
        for v in ('uncentred', 'centred'):
            for mult in (1.0, 10.0, 100.0):
                c = RS.counts_from_S(KS[(v, mult)])
                k21, n21, d21 = c[('knee', '21x', 'CUSUM')]
                k4, n4, d4 = c[('knee', '4x', 'CUSUM')]
                md.append(f'| {v} | {mult:g}x | {nonone[(v, mult)]} | {k21} / {n21} | {k4} / {n4} | '
                          f'{sum(abs(x) <= 10 for x in d21)} + {sum(abs(x) <= 10 for x in d4)} | '
                          f'{fmt_pop(*c[("heldout", "CUSUM")])} | {fmt_pop(*c[("mae", "CUSUM")])} | '
                          f'{fmt_pop(*c[("tiny", "CUSUM")])} | {len(c[("heldout", "any")][0])}/11, '
                          f'{len(c[("mae", "any")][0])}/10, {len(c[("tiny", "any")][0])}/1 |')

    text = '\n'.join(md)
    print('\n' + text)
    if md_path:
        with open(md_path, 'w') as fh:
            fh.write(text + '\n')
        print('wrote', md_path)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--readability', action='store_true')
    ap.add_argument('--gate', action='store_true')
    ap.add_argument('--lidar-only', action='store_true')
    ap.add_argument('--md', default=None)
    a = ap.parse_args()
    if a.readability:
        build_readability()
    elif a.gate:
        gate_readability()
    else:
        audit(a.md, lidar_only=a.lidar_only)
