"""Audit of the projector-output LiDAR and RankMe series (2026-09-24), next to
the pooled-representation series of the same runs. CPU only; reads CSVs;
writes nothing except an optional markdown file (--md, must not exist).

Series (N = 100 replica sets for LiDAR; 4,000-event fitting split for RankMe):
  projector   results/knee_proj_lidar.csv (grouping shared / redrawn x BN eval /
              batch) and results/knee_proj_rankme.csv (pub_uncentred, pub_centred,
              sq_centred x BN eval / batch), from knee_proj_lidar_rankme.py.
              48 LeJEPA runs; the 10 MAE runs have no projector.
  pooled      results/knee_lidar.csv, results/knee_lidar_sharedshape.csv (N = 100),
              results/knee_labelfree_metrics.csv (RankMe rows); restricted to the
              same 48 runs for the like-for-like rows. The 58-run MAE counts
              are quoted beside them for reference only.

Machinery imported unchanged (through knee_labelfree_audit, which imports
knee_lidar_audit, knee_s2_monitors, knee_lidar_audit_sharedshape,
knee_lidar_replica_seed, knee_lidar_vs_readability):
  * knee_lidar_replica_seed.counts_from_S: thresholds = 1.1 x max over each
    campaign's four mu = 0.97 controls, the four tests (CUSUM, Page-Hinkley,
    EWMA, slope11), per-population counts; within one grid step = |alarm -
    onset| <= 10.
  * knee_labelfree_audit.s2a (local-deviation rank at onset and placebo),
    .prepost (the 13 capacity-tally runs, knee_rank_estimators_prepost
    .ladder_epochs), .load_readability (results/knee_labelfree_readability.csv),
    .load_lidar, .load_metrics, .fmt_pop.
  * knee_lidar_replica_seed.level_changes (PRE -> POST of the 27 knee runs;
    event-free raw change), only for populations present.

MAE handling (checked, 2026-09-24): counts_from_S, thresholds, score, s2a,
prepost and level_changes filter runs by presence in the series, so an absent
MAE population gives a zero denominator, printed here as n/a.
knee_labelfree_audit.correlations indexes S[r] for every MAE run (KeyError)
and knee_labelfree_audit.audit asserts 58 runs, so neither is called on a
48-run series; correlations_no_mae below is the loop body of
knee_labelfree_audit.correlations, copied, run over the groups present, and
--selftest checks it returns exactly what the original returns on the 58-run
series.

Usage (from analysis/; run with -B):
  python -B knee_proj_audit.py --selftest     # reproduce the published pooled rows first
  python -B knee_proj_audit.py [--md out.md]
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
import knee_labelfree_audit as LFA  # noqa: E402

AUD, S2, A2, RS, LVR = LFA.AUD, LFA.S2, LFA.A2, LFA.RS, LFA.LVR

PROJ_LIDAR = os.path.join(paths.CSV, 'knee_proj_lidar.csv')
PROJ_RANKME = os.path.join(paths.CSV, 'knee_proj_rankme.csv')
SENS_CSV = os.path.join(paths.CSV, 'knee_proj_lidar_delta.csv')        # knee_proj_lidar_rankme.py --check
SHARE_CSV = os.path.join(paths.CSV, 'knee_proj_lidar_delta_share.csv')  # knee_proj_lidar_rankme.py --delta-share
REPORT = os.path.join(os.environ.get('MONITOR_NOTES', os.path.join(paths.REPO, 'notes')), 'labelfree_metrics_2026-09-23.md')
RANKME_VARIANTS = ('pub_uncentred', 'pub_centred', 'sq_centred')
BN_MODES = ('eval', 'batch')
NA = 'n/a'


# ------------------------------------------------------------------ series

def _series(rows):
    """[(run, epoch, value)] -> {short run: (sorted epochs, values)}, the shape
    knee_lidar_audit.load returns (short = run.split('xxx')[0])."""
    by = defaultdict(dict)
    for run, ep, v in rows:
        k = run.split('xxx')[0]
        assert ep not in by[k], (k, ep)
        by[k][ep] = v
    return {k: (sorted(d), [d[e] for e in sorted(d)]) for k, d in by.items()}


def load_proj():
    lid, rk = defaultdict(list), defaultdict(list)
    with open(PROJ_LIDAR) as fh:
        for r in csv.DictReader(fh):
            lid[(r['grouping'], r['bn_mode'])].append((r['run'], int(r['epoch']), float(r['lidar'])))
    with open(PROJ_RANKME) as fh:
        for r in csv.DictReader(fh):
            rk[(r['variant'], r['bn_mode'])].append((r['run'], int(r['epoch']), float(r['value'])))
    return ({k: _series(v) for k, v in lid.items()}, {k: _series(v) for k, v in rk.items()})


def load_pooled():
    """58-run pooled series: LiDAR redrawn / shared (N = 100) and the three
    RankMe variants, via knee_labelfree_audit's own loaders."""
    out = {('lidar', 'redrawn'): LFA.load_lidar(LFA.LIDAR['lidar redrawn']),
           ('lidar', 'shared'): LFA.load_lidar(LFA.LIDAR['lidar shared'])}
    M = LFA.load_metrics()
    for v in RANKME_VARIANTS:
        out[('rankme', v)] = M[('rankme', v)]
    for k, S in out.items():
        assert len(S) == 61, (k, len(S))  # denominator runs: 58 + nano_10-12, as in knee_labelfree_audit
    return out


def drop_mae(S):
    return {k: v for k, v in S.items() if k not in A2.MAE}


# ------------------------------------------------------------------ statistics

def correlations_no_mae(S, rd):
    """knee_labelfree_audit.correlations' loop body, copied unchanged, over the
    groups whose runs are in S; a group absent from S -> None (n/a)."""
    res = {}
    for gname, runs in LFA.GROUPS:
        present = [r in S for r in runs]
        if not all(present):
            assert not any(present), (gname, 'partially present')
            res[(gname, 'R2E')] = res[(gname, 'CCA')] = None
            continue
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


def pop(c, name, t):
    fired, n = c[(name, t)]
    return NA if n == 0 else LFA.fmt_pop(fired, n)


def pop_short(c, name, t):
    fired, n = c[(name, t)]
    return NA if n == 0 else f'{len(fired)}/{n}'


def cells(S, rd):
    """Every audit quantity for one series, formatted as knee_labelfree_audit
    .audit formats it (so the pooled rows can be compared with the report)."""
    out = {}
    c = RS.counts_from_S(S)
    out['c'] = c
    thr = c['thr']
    k21, n21, d21 = c[('knee', '21x', 'CUSUM')]
    k4, n4, d4 = c[('knee', '4x', 'CUSUM')]
    near21, near4 = sum(abs(x) <= 10 for x in d21), sum(abs(x) <= 10 for x in d4)
    out['near'] = (near21, near4)
    out['cusum'] = [f'{thr["21x"]["CUSUM"]:.2f} / {thr["4x"]["CUSUM"]:.2f}', f'{k21} / {n21}',
                    f'{k4} / {n4}', f'{near21} + {near4}', pop(c, 'heldout', 'CUSUM'),
                    pop(c, 'mae', 'CUSUM'), pop(c, 'tiny', 'CUSUM')]
    out['cusum_d'] = (sorted(d21), sorted(d4))
    for t in A2.TESTS:
        if t == 'CUSUM':
            continue
        a21, m21, e21 = c[('knee', '21x', t)]
        a4, m4, e4 = c[('knee', '4x', t)]
        out[t] = [f'{a21}/{m21}, {a4}/{m4} (near {sum(abs(x) <= 10 for x in e21)} + '
                  f'{sum(abs(x) <= 10 for x in e4)})',
                  f'{pop_short(c, "heldout", t)}, {pop_short(c, "mae", t)}, {pop_short(c, "tiny", t)}']
        out[t + '_near'] = sum(abs(x) <= 10 for x in e21) + sum(abs(x) <= 10 for x in e4)
    out['any'] = [pop(c, 'heldout', 'any'), pop(c, 'mae', 'any'), pop(c, 'tiny', 'any')]
    s = []
    for camp, knee in (('21x', AUD.KNEE_21), ('4x', AUD.KNEE_4)):
        at, pl = LFA.s2a(S, knee)
        s.append((f'{S2.median(at):.3f}' if at else '-') + f' (n={len(at)})')
        s.append((f'{S2.median(pl):.3f}' if pl else '-') + f' (n={len(pl)})')
    out['s2a'] = s
    pp = LFA.prepost(S)
    ch = [b - a for _, a, b in pp]
    pos = all(a > 0 for _, a, _ in pp)
    pct = [100 * (b - a) / a for _, a, b in pp] if pos else None
    out['prepost'] = [f'{smedian(a for _, a, _ in pp):.4g}', f'{smedian(b for _, _, b in pp):.4g}',
                      f'{sum(x > 0 for x in ch)} / {sum(x < 0 for x in ch)}',
                      f'{smedian(ch):+.3g} ({min(ch):+.3g} to {max(ch):+.3g})',
                      (f'{smedian(pct):+.1f} ({min(pct):+.1f} to {max(pct):+.1f})' if pct else 'n/a')]
    cr = correlations_no_mae(S, rd)
    corr = []
    for g, _ in LFA.GROUPS:
        for q in ('R2E', 'CCA'):
            if cr[(g, q)] is None:
                corr.append(NA)
                continue
            pooled, within, n = cr[(g, q)]
            corr.append(f'{pooled:+.2f}; {smedian(within):+.2f}, {sum(w < 0 for w in within)}/{len(within)}')
    out['corr'] = corr
    out['corr_raw'] = cr
    L = RS_level_changes(S)
    out['level'] = L
    return out


def RS_level_changes(S):
    """knee_lidar_replica_seed.level_changes, called unchanged; it filters the
    event-free populations by presence, so MAE comes back as an empty list."""
    return RS.level_changes(S)


# ------------------------------------------------------------------ selftest

def report_rows():
    """{section prefix: {row name: [cells after the name]}} from the label-free
    report's audit tables."""
    want = ('### 3.1', '### 3.2', '### 3.3', '### 3.4', '### 3.5', '### 3.7', '### 4.1', '### 4.2')
    out, sec = {}, None
    for line in open(REPORT):
        if line.startswith('#'):
            sec = next((w for w in want if line.startswith(w)), None)
            continue
        if sec and line.startswith('| ') and not line.startswith('|---'):
            parts = [p.strip() for p in line.strip().strip('|').split('|')]
            out.setdefault(sec, {})[parts[0]] = parts[1:]
    return out


def selftest():
    rd = LFA.load_readability()
    P58 = load_pooled()
    R = report_rows()
    names = {('lidar', 'redrawn'): 'lidar redrawn', ('lidar', 'shared'): 'lidar shared',
             ('rankme', 'pub_uncentred'): 'rankme pub_uncentred',
             ('rankme', 'pub_centred'): 'rankme pub_centred', ('rankme', 'sq_centred'): 'rankme sq_centred'}
    n_ok = n_all = 0
    for key, S in P58.items():
        nm = names[key]
        C = cells(S, rd)
        mine = {'### 3.1': C['cusum'], '### 3.2': C['PageHinkley'], '### 3.3': C['EWMA'],
                '### 3.4': C['slope11'], '### 3.5': C['any'], '### 3.7': C['s2a'],
                '### 4.1': C['prepost'], '### 4.2': C['corr']}
        for sec, got in mine.items():
            ref = R[sec][nm]
            if sec == '### 3.1':
                ref = ref[1:]                        # drop the P/C/S tag cell
            ok = ref == got
            n_ok += ok
            n_all += 1
            if not ok:
                print(f'MISMATCH {nm} {sec}:\n   report {ref}\n   mine   {got}')
        # the correlation wrapper returns exactly what the original returns (non-MAE groups)
        orig = LFA.correlations(S, rd)
        wrap = correlations_no_mae(drop_mae(S), rd)
        same = all(orig[k] == wrap[k] for k in orig if k[0] != 'MAE') and \
            all(wrap[k] is None for k in wrap if k[0] == 'MAE')
        full = correlations_no_mae(S, rd)
        same_full = all(orig[k] == full[k] for k in orig)
        print(f'{nm:22s}: report rows reproduced {sum(1 for s in mine if (R[s][nm][1:] if s == "### 3.1" else R[s][nm]) == mine[s])}/8; '
              f'wrapper on 48 runs == knee_labelfree_audit.correlations on 58 (non-MAE groups) {same}; '
              f'wrapper on 58 == original on 58 (all groups) {same_full}')
        assert same and same_full
        # the 48-run restriction changes nothing but the MAE population
        c58, c48 = RS.counts_from_S(S), RS.counts_from_S(drop_mae(S))
        diff = [k for k in c58 if k not in ('S', 'sc') and not (isinstance(k, tuple) and k[0] == 'mae')
                and c58[k] != c48[k]]
        assert not diff, (nm, diff)
        assert c48[('mae', 'CUSUM')][1] == 0
    print(f'\nSELFTEST: {n_ok}/{n_all} published audit rows reproduced cell for cell; 48-run counts equal '
          f'58-run counts outside the MAE population for all 5 pooled series')
    assert n_ok == n_all


# ------------------------------------------------------------------ report

def build_rows():
    rd = LFA.load_readability()
    PL, PR = load_proj()
    for k, S in list(PL.items()) + list(PR.items()):
        assert len(S) == 48, (k, len(S))
        bad = sum(1 for eps, xs in S.values() for x in xs if not math.isfinite(x))
        assert bad == 0, (k, bad)
        assert not any(r in S for r in A2.MAE)
    P58 = load_pooled()
    rows = []   # (label, family, layer, C, C58-or-None)
    for g in ('shared', 'redrawn'):
        for b in BN_MODES:
            rows.append((f'LiDAR {g}, projector, BN {b}', f'lidar {g}', f'projector {b}', cells(PL[(g, b)], rd), None))
        rows.append((f'LiDAR {g}, pooled (same 48 runs)', f'lidar {g}', 'pooled',
                     cells(drop_mae(P58[('lidar', g)]), rd), cells(P58[('lidar', g)], rd)))
    for v in RANKME_VARIANTS:
        for b in BN_MODES:
            rows.append((f'RankMe {v}, projector, BN {b}', f'rankme {v}', f'projector {b}', cells(PR[(v, b)], rd), None))
        rows.append((f'RankMe {v}, pooled (same 48 runs)', f'rankme {v}', 'pooled',
                     cells(drop_mae(P58[('rankme', v)]), rd), cells(P58[('rankme', v)], rd)))
    return rows


def mae_cell(C, C58, key):
    """n/a on 48-run series; the 58-run pooled count quoted for reference."""
    if C58 is None:
        return 'n/a (no projector)'
    fired, n = C58['c'][key]
    return f'n/a; 58-run series {len(fired)} / {n}'


def render(rows):
    md = []
    md.append('### Headline: CUSUM first alarms\n')
    md.append('| series | knee runs with a first alarm within one grid step of onset (of 27; 21x + 4x) | '
              'held-out LeJEPA false alarms (of 11) | ViT-Tiny flat (of 1) | MAE (of 10) |')
    md.append('|---|---|---|---|---|')
    for lab, fam, lay, C, C58 in rows:
        n21, n4 = C['near']
        ho, hn = C['c'][('heldout', 'CUSUM')]
        to, tn = C['c'][('tiny', 'CUSUM')]
        md.append(f'| {lab} | {n21 + n4} ({n21} + {n4}) | {len(ho)} / {hn} | {len(to)} / {tn} | '
                  f'{mae_cell(C, C58, ("mae", "CUSUM"))} |')

    md.append('\n### CUSUM (thresholds 1.1 x max over each campaign\'s four mu = 0.97 controls)\n')
    md.append('| series | threshold 21x / 4x | knee 21x | knee 4x | within one grid step (21x + 4x) | '
              'held-out LeJEPA | MAE | ViT-Tiny flat |')
    md.append('|---|---|---|---|---|---|---|---|')
    for lab, fam, lay, C, C58 in rows:
        cc = list(C['cusum'])
        cc[5] = mae_cell(C, C58, ('mae', 'CUSUM'))
        md.append(f'| {lab} | ' + ' | '.join(cc) + ' |')
    md.append('\nCUSUM alarm minus onset (epochs), knee runs that alarm: 21x | 4x\n')
    md.append('| series | 21x | 4x |')
    md.append('|---|---|---|')
    for lab, fam, lay, C, C58 in rows:
        d21, d4 = C['cusum_d']
        md.append(f'| {lab} | {" ".join(f"{x:+d}" for x in d21) or "-"} | {" ".join(f"{x:+d}" for x in d4) or "-"} |')

    for t in A2.TESTS:
        if t == 'CUSUM':
            continue
        md.append(f'\n### {t}: knee 21x, knee 4x (within one grid step) | false alarms: held-out LeJEPA, MAE, '
                  f'ViT-Tiny flat\n')
        md.append('| series | knee | false alarms |')
        md.append('|---|---|---|')
        for lab, fam, lay, C, C58 in rows:
            fa = C[t][1]
            if C58 is not None:
                fa += f' (58-run MAE {pop_short(C58["c"], "mae", t)})'
            md.append(f'| {lab} | {C[t][0]} | {fa} |')

    md.append('\n### Any of the four tests, false-alarm populations\n')
    md.append('| series | held-out LeJEPA | MAE | ViT-Tiny flat |')
    md.append('|---|---|---|---|')
    for lab, fam, lay, C, C58 in rows:
        a = list(C['any'])
        a[1] = mae_cell(C, C58, ('mae', 'any'))
        md.append(f'| {lab} | ' + ' | '.join(a) + ' |')

    md.append('\n### S2a: local-deviation rank fraction at onset (small = the onset step stands out), median over '
              'knee runs, and the placebo median (each run at the other runs\' onsets)\n')
    md.append('| series | 21x at onset | 21x placebo | 4x at onset | 4x placebo |')
    md.append('|---|---|---|---|---|')
    for lab, fam, lay, C, C58 in rows:
        md.append(f'| {lab} | ' + ' | '.join(C['s2a']) + ' |')

    md.append('\n### PRE to POST on the 13 capacity-tally runs (ladder epochs)\n')
    md.append('| series | PRE median | POST median | rose / fell | median change (range) | median % change (range) |')
    md.append('|---|---|---|---|---|---|')
    for lab, fam, lay, C, C58 in rows:
        md.append(f'| {lab} | ' + ' | '.join(C['prepost']) + ' |')

    md.append('\n### Level change, all 27 knee runs (PRE = last grid epoch <= onset - 20, POST = final) and '
              'event-free raw change (median of epochs > final - 30 minus median of epochs 21-60)\n')
    md.append('| series | knee 21x: median (range), rises | knee 4x: median (range), rises | 21x controls | '
              '4x controls | 21x held-out | 4x held-out | ViT-Tiny flat |')
    md.append('|---|---|---|---|---|---|---|---|')
    for lab, fam, lay, C, C58 in rows:
        L = C['level']
        cc = []
        for name in ('knee 21x', 'knee 4x'):
            v = L[name]
            cc.append(f'{smedian(v):+.3g} ({min(v):+.3g} to {max(v):+.3g}), {sum(x > 0 for x in v)}/{len(v)}')
        for name in ('21x controls', '4x controls', '21x held-out', '4x held-out', 'ViT-Tiny flat'):
            v = L[name]
            cc.append(f'{smedian(v):+.3g}' + (f' ({min(v):+.3g} to {max(v):+.3g})' if len(v) > 1 else ''))
        md.append(f'| {lab} | ' + ' | '.join(cc) + ' |')

    md.append('\n### Rank correlation with readability (csv/knee_labelfree_readability.csv, pooled-feature '
              'R2(E) and mean CCA), checkpoints with epoch > 20: pooled rho; within-run median rho, runs '
              'negative / runs\n')
    md.append('| series | event-free LeJEPA: R2(E) | event-free LeJEPA: mean CCA | MAE | knee: R2(E) | '
              'knee: mean CCA |')
    md.append('|---|---|---|---|---|---|')
    for lab, fam, lay, C, C58 in rows:
        cr = C['corr']                   # order: free R2E, free CCA, MAE R2E, MAE CCA, knee R2E, knee CCA
        mae = 'n/a (no projector)' if C58 is None else f'n/a; 58-run R2(E) {C58["corr"][2]}'
        md.append(f'| {lab} | {cr[0]} | {cr[1]} | {mae} | {cr[4]} | {cr[5]} |')
    if os.path.exists(SENS_CSV):
        md += sensitivity()
    if os.path.exists(SHARE_CSV):
        md += share_table()
    return '\n'.join(md)


def share_table():
    """Summary of SHARE_CSV: within-group variance along lidar()'s leading
    whitened direction in units of delta (1e-4), and delta's share of that
    direction's denominator."""
    rows = list(csv.DictReader(open(SHARE_CSV)))
    md = ['\n### Projector LiDAR, delta = 1e-4: within-group variance along the leading whitened direction '
          '(units of delta) and delta\'s share of that direction\'s denominator; median (range)\n',
          '| grouping, BN | knee PRE (27): variance / delta | knee PRE: delta share | knee POST (27): variance / '
          'delta | knee POST: delta share | controls final (8): variance / delta | controls final: delta share |',
          '|---|---|---|---|---|---|---|']
    for g in ('shared', 'redrawn'):
        for b in BN_MODES:
            cc = []
            for role, ph in (('knee', 'pre'), ('knee', 'post'), ('control', 'final')):
                rs = [r for r in rows if r['grouping'] == g and r['bn_mode'] == b and r['role'] == role
                      and r['phase'] == ph]
                s = [float(r['sw_lead_over_delta']) for r in rs]
                sh = [float(r['delta_share']) for r in rs]
                cc.append(f'{smedian(s):.3g} ({min(s):.3g} to {max(s):.3g})')
                cc.append(f'{smedian(sh):.3f} ({min(sh):.3f} to {max(sh):.3f})')
            md.append(f'| {g}, {b} | ' + ' | '.join(cc) + ' |')
    return md


def sensitivity():
    """Projector LiDAR recomputed from the stored embeddings in float64 on the
    CPU with lidar()'s delta = 1e-4 (as used) and 1e-5, 1e-3, through the same
    audit, beside the GPU float32 series of the main CSV."""
    rd = LFA.load_readability()
    PL, _ = load_proj()
    by = defaultdict(list)
    with open(SENS_CSV) as fh:
        rdr = csv.DictReader(fh)
        cols = [c for c in rdr.fieldnames if c.startswith('lidar_f64_delta')]
        for r in rdr:
            for c in cols:
                by[(r['grouping'], r['bn_mode'], c)].append((r['run'], int(r['epoch']), float(r[c])))
    md = ['\n### Projector LiDAR: float32 on the GPU (the CSV) vs float64 on the CPU, and lidar()\'s delta\n',
          '| grouping, BN | computation | timely CUSUM (of 27; 21x + 4x) | CUSUM knee alarms 21x, 4x | '
          'held-out LeJEPA (of 11) | ViT-Tiny flat | S2a at onset 21x, 4x | PRE/POST rose / fell |',
          '|---|---|---|---|---|---|---|---|']
    for g in ('shared', 'redrawn'):
        for b in BN_MODES:
            items = [('GPU float32, delta 1e-4 (CSV)', PL[(g, b)])]
            items += [(f'CPU float64, delta {c[len("lidar_f64_delta"):]}', _series(by[(g, b, c)]))
                      for c in sorted(cols, key=lambda c: -float(c[len('lidar_f64_delta'):]))]
            for lab, S in items:
                assert len(S) == 48
                C = cells(S, rd)
                n21, n4 = C['near']
                k21 = C['c'][('knee', '21x', 'CUSUM')]
                k4 = C['c'][('knee', '4x', 'CUSUM')]
                ho, hn = C['c'][('heldout', 'CUSUM')]
                to, tn = C['c'][('tiny', 'CUSUM')]
                md.append(f'| {g}, {b} | {lab} | {n21 + n4} ({n21} + {n4}) | {k21[0]}/{k21[1]}, {k4[0]}/{k4[1]} | '
                          f'{len(ho)} / {hn} | {len(to)} / {tn} | {C["s2a"][0]}, {C["s2a"][2]} | '
                          f'{C["prepost"][2]} |')
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--md', default=None)
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    text = render(build_rows())
    print(text)
    if a.md:
        assert not os.path.exists(a.md), f'refusing to overwrite {a.md}'
        with open(a.md, 'w') as fh:
            fh.write(text + '\n')
        print('wrote', a.md)


if __name__ == '__main__':
    main()
