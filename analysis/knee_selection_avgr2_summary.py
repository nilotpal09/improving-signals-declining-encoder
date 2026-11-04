"""Checkpoint-selection summary on the new readability measure and onsets
(2026-10-10). Readability = held-out ridge R2 of z1, z2 and log E, and their
average; onsets from the average-R2 rule (results/knee_onsets_avgr2.csv).

Nothing is recomputed that the existing files already hold:
  picks, old reference epochs, R2(E) cost   results/knee_labelfree_selection_v2.csv
  per-factor costs, new reference epochs    results/knee_selection_perfactor.csv
  new decline labels and onsets             results/knee_onsets_avgr2.csv
This script joins them, checks them, and summarises per series and
population. Writes a NEW file, results/knee_selection_avgr2_summary.csv (refuses
to overwrite), and prints the paper's numbers.

Gates (printed):
  G1  the summary code, fed the OLD reference epochs (v2 column ref_epoch),
      reproduces every at/after count, pick = final count and R2(E) cost
      median (max) in tables 3.1, 3.2, 3.3 and 5 of the internal report
      selection_2026-09-24.md (parsed from the file; read from MONITOR_NOTES,
      not released; skipped when absent).
  G2  per-factor costs of knee_selection_perfactor.csv recomputed from the
      trajectories of knee_onsets_avgr2.load() (an independent loader over the
      five ridge files): max |diff|; candidate grids equal v2's.
  G3  new reference epochs recomputed from knee_onsets_avgr2.csv (own onset
      for decline runs; campaign median new onset for the others): equal to
      knee_selection_perfactor.csv's ref_epoch_new on every row.
  G4  the 27 v2 'knee' runs are exactly the new-rule decline runs among the 57.

Populations: decline runs (27, v2 group 'knee'); LeJEPA runs without a
decline (20, 'event-free LeJEPA'); MAE (10); and the 30 runs without a decline
pooled (20 + 10), as knee_selection_perfactor.py printed them.
"""
import sys

sys.dont_write_bytecode = True

import csv  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
from collections import defaultdict  # noqa: E402
from statistics import median  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
sys.path.insert(0, _HERE)
import knee_onsets_avgr2 as KO  # noqa: E402

CSV = os.path.join(paths.CSV)
V2 = os.path.join(CSV, 'knee_labelfree_selection_v2.csv')
PF = os.path.join(CSV, 'knee_selection_perfactor.csv')
ONS = os.path.join(CSV, 'knee_onsets_avgr2.csv')
MD = os.path.join(os.environ.get('MONITOR_NOTES', os.path.join(paths.REPO, 'notes')), 'selection_2026-09-24.md')
OUT = os.path.join(CSV, 'knee_selection_avgr2_summary.csv')
V2_ORIG = os.path.join(CSV, 'orig_copies', 'knee_labelfree_selection_v2.csv')  # the recorded v2 table (57 runs)
WARM = 20
F = ('z1', 'z2', 'E')
POPS = (('decline', ('knee',)), ('no decline, LeJEPA', ('event-free LeJEPA',)), ('MAE', ('MAE',)),
        ('no decline, LeJEPA + MAE', ('event-free LeJEPA', 'MAE')))
PAPER = [('Training loss', 'loss'), ('RankMe pub_uncentred', 'rankme pub_uncentred'),
         ('LiDAR, training-style (shared)', 'lidar shared'), ('LiDAR, shape-resampled (redrawn)', 'lidar redrawn'),
         ('alpha-ReQ centred', 'alpha_req centred'), ('First candidate', 'rule: first checkpoint > 20'),
         ('Epoch 60', 'rule: epoch 60'), ('Final checkpoint', 'rule: final checkpoint')]


def load():
    v2 = list(csv.DictReader(open(V2)))
    pf = {(r['series'], r['run']): r for r in csv.DictReader(open(PF))}
    ons = {r['run']: r for r in csv.DictReader(open(ONS))}
    assert len(v2) == len(pf) == 3120, (len(v2), len(pf))  # 60 runs x 52 series, with the runs added 2026-10-13 (was 57 x 52)
    rows = []
    for r in v2:
        p = pf[(r['series'], r['run'])]
        assert int(p['pick_epoch']) == int(r['pick_epoch']) and p['group'] == r['group']
        rows.append(dict(series=r['series'], kind=r['kind'], run=r['run'], group=r['group'],
                         campaign=r['campaign'], pick=int(r['pick_epoch']), final=int(r['final_epoch']),
                         ref_old=float(r['ref_epoch']), ref_new=float(p['ref_epoch_new']),
                         r2e_cost=float(r['r2e_cost']),
                         cost_z1=float(p['cost_z1']), cost_z2=float(p['cost_z2']),
                         cost_E=float(p['cost_E']), cost_avg=float(p['cost_avg'])))
    return rows, ons


def summarise(rs, ref):
    c = lambda k: [r[k] for r in rs]  # noqa: E731
    out = dict(n=len(rs), at_after=sum(r['pick'] >= r[ref] for r in rs),
               pick_final=sum(r['pick'] == r['final'] for r in rs),
               pick_minus_ref_median=median(r['pick'] - r[ref] for r in rs))
    for k in ('r2e_cost', 'cost_avg', 'cost_z1', 'cost_z2', 'cost_E'):
        out[f'{k}_median'] = median(c(k))
        out[f'{k}_max'] = max(c(k))
    return out


# ------------------------------------------------------------------ gates

def parse_md():
    """{(table, series): (at_after, n, pick_final_or_None, cost_med, cost_max)} from tables 3.1-3.3 and 5."""
    txt = open(MD).read()
    secs = {'knee': ('### 3.1', '### 3.2'), 'event-free LeJEPA': ('### 3.2', '### 3.3'),
            'MAE': ('### 3.3', '**Facts'), 'reversed': ('## 5.', '- **With the direction reversed')}
    out = {}
    for tab, (a, b) in secs.items():
        block = txt[txt.index(a):txt.index(b, txt.index(a))]
        for line in block.splitlines():
            cells = [x.strip() for x in line.strip().strip('|').split('|')]
            if len(cells) < 6 or not re.match(r'^\d+ / \d+$', cells[2]):
                continue
            at, n = (int(x) for x in cells[2].split(' / '))
            if tab == 'reversed':
                fin = int(cells[3].split(' / ')[0])
                m = re.match(r'([\d.]+) \(([\d.]+)\)', cells[4])
            else:
                fin = int(cells[4].split(' / ')[0])
                m = re.match(r'([\d.]+) \(([\d.]+)\)', cells[5])
            out[(tab, cells[0])] = (at, n, fin, m.group(1), m.group(2))
    return out


def gate1(rows):
    if not os.path.exists(MD):
        print(f'gate 1 skipped: {MD} (internal report) not present')
        return
    # the report's tables cover the original 57 runs (before the runs added 2026-10-13) at the record's old reference epochs
    rec = {(r['series'], r['run']): float(r['ref_epoch']) for r in csv.DictReader(open(V2_ORIG))}
    rows = [dict(r, ref_old=rec[(r['series'], r['run'])]) for r in rows if (r['series'], r['run']) in rec]
    md = parse_md()
    by = defaultdict(list)
    for r in rows:
        by[(r['series'], r['group'])].append(r)
    bad, n = [], 0
    for (tab, s), (at, nn, fin, cm, cx) in md.items():
        if tab == 'reversed':
            rs = by[(s + ' [reversed]', 'knee')]
        else:
            rs = by[(s, tab)]
        g = summarise(rs, 'ref_old')
        got = (g['at_after'], g['n'], g['pick_final'], f"{g['r2e_cost_median']:.3f}", f"{g['r2e_cost_max']:.3f}")
        n += 1
        if got != (at, nn, fin, cm, cx):
            bad.append((tab, s, got, (at, nn, fin, cm, cx)))
    print(f'G1: {n} table rows of selection_2026-09-24.md (3.1, 3.2, 3.3, 5) checked with OLD reference epochs: '
          f'{n - len(bad)} reproduce exactly (at/after, n, pick = final, R2(E) cost median, max); {len(bad)} differ')
    for b in bad:
        print('   differs:', b)
    assert n == 29 * 3 + 21 and not bad
    return n


def gate2(rows):
    R, _ = KO.load()
    rd = {}
    for r in csv.DictReader(open(os.path.join(CSV, 'knee_labelfree_readability.csv'))):
        rd[(r['run'].split('xxx')[0], int(r['epoch']))] = tuple(float(r[f'ridge_r2_{x}']) for x in F)
    worst_rd = max(abs(a - b) for (k, e), v in rd.items() for a, b in zip(v, R[k][e]))
    v2 = {(r['series'], r['run']): r for r in csv.DictReader(open(V2))}
    worst = 0.0
    for r in rows:
        t = R[r['run']]
        cand = [e for e in sorted(t) if e > WARM]
        v = v2[(r['series'], r['run'])]
        assert (len(cand), cand[0], cand[-1]) == (int(v['n_cand']), int(v['first_epoch']), int(v['final_epoch']))
        for i, x in enumerate(F):
            worst = max(worst, abs((max(t[e][i] for e in cand) - t[r['pick']][i]) - r[f'cost_{x}']))
        avg = {e: sum(t[e]) / 3 for e in cand}
        worst = max(worst, abs((max(avg.values()) - avg[r['pick']]) - r['cost_avg']))
        worst = max(worst, abs(r['cost_E'] - r['r2e_cost']))
    print(f'G2: ridge R2 in knee_labelfree_readability.csv vs knee_onsets_avgr2.load(): max |diff| {worst_rd:.1e}; '
          f'per-factor and average costs recomputed from knee_onsets_avgr2.load(), and cost_E vs v2 r2e_cost: '
          f'max |diff| {worst:.1e} over {len(rows)} rows (CSV precision 1e-6)')
    assert worst_rd < 1e-6 and worst < 2e-6
    return worst_rd, worst


def gate3_4(rows, ons):
    camp = defaultdict(list)
    for k in {r['run'] for r in rows if r['group'] == 'knee'}:
        camp['4x' if '_p384_' in k else '21x'].append(int(ons[k]['onset']))
    pseudo = {c: median(v) for c, v in camp.items()}
    mism = 0
    for r in rows:
        dec = ons[r['run']]['decline'] == '1'
        assert dec == (r['group'] == 'knee'), r['run']
        ref = int(ons[r['run']]['onset']) if dec else pseudo[r['campaign']]
        mism += ref != r['ref_new']
    print(f'G3: new reference epochs recomputed from knee_onsets_avgr2.csv: {mism} of {len(rows)} rows differ from '
          f'knee_selection_perfactor.csv ref_epoch_new; campaign medians of new onsets (pseudo-onsets) {pseudo} '
          f'(old: 115 at 21x, 90 at 4x)')
    print('G4: the 27 v2 decline runs are exactly the new-rule decline runs among the 57 (asserted per run)')
    assert mism == 0
    return pseudo


# ------------------------------------------------------------------ main

def main():
    assert not os.path.exists(OUT), f'refusing to overwrite {OUT}'
    rows, ons = load()
    gate1(rows)
    gate2(rows)
    pseudo = gate3_4(rows, ons)

    by = defaultdict(list)
    for r in rows:
        by[r['series']].append(r)
    order = list(dict.fromkeys(r['series'] for r in rows))
    kinds = {r['series']: r['kind'] for r in rows}
    out = []
    for s in order:
        for pop, groups in POPS:
            rs = [r for r in by[s] if r['group'] in groups]
            old, new = summarise(rs, 'ref_old'), summarise(rs, 'ref_new')
            row = dict(series=s, kind=kinds[s], population=pop, n=new['n'],
                       at_after_old=old['at_after'], at_after_new=new['at_after'], pick_final=new['pick_final'],
                       pick_minus_ref_median_old=old['pick_minus_ref_median'],
                       pick_minus_ref_median_new=new['pick_minus_ref_median'])
            for k in ('cost_avg', 'cost_z1', 'cost_z2', 'cost_E'):
                row[f'{k}_median'] = new[f'{k}_median']
                row[f'{k}_max'] = new[f'{k}_max']
            out.append(row)
    cols = list(out[0])
    part = OUT + '.partial'
    assert not os.path.exists(part)
    with open(part, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator='\n')
        w.writeheader()
        for o in out:
            w.writerow({c: (f'{v:.6f}' if isinstance(v, float) and c.startswith('cost') else
                            (f'{v:g}' if isinstance(v, float) else v)) for c, v in o.items()})
    os.replace(part, OUT)
    print(f'wrote {OUT}: {len(out)} rows')

    S = {(o['series'], o['population']): o for o in out}

    def line(s, pop):
        o = S[(s, pop)]
        return (f"{o['at_after_old']:>2d}->{o['at_after_new']:>2d}/{o['n']:<2d} final {o['pick_final']:>2d} | "
                f"avg {o['cost_avg_median']:.3f} ({o['cost_avg_max']:.3f})  z1 {o['cost_z1_median']:.3f}  "
                f"z2 {o['cost_z2_median']:.3f}  logE {o['cost_E_median']:.3f} ({o['cost_E_max']:.3f})")

    print('\nmedian cost (max) at the pick vs the best candidate > 20; at/after onset OLD->NEW reference')
    for pop, _ in POPS:
        print(f'\n## {pop}')
        for s in order:
            if kinds[s] in ('metric', 'lidar', 'loss', 'oracle', 'reference', 'rule'):
                print(f'{s:32s} {line(s, pop)}')

    print('\n## paper table (sections/table_selection_main.tex): at/after onset NEW | decline cost | '
          'no-decline LeJEPA cost; cost in average ridge R2 [log-E R2 in brackets]')
    for name, s in PAPER:
        k, f = S[(s, 'decline')], S[(s, 'no decline, LeJEPA')]
        print(f"{name:34s} {k['at_after_new']}/{k['n']} (old {k['at_after_old']}/{k['n']}) | "
              f"{k['cost_avg_median']:.3f} [{k['cost_E_median']:.3f}] | {f['cost_avg_median']:.3f} "
              f"[{f['cost_E_median']:.3f}]")

    lf = [s for s in order if kinds[s] == 'metric' and not s.startswith('alpha_req')]
    lid = [s for s in order if kinds[s] == 'lidar']
    for tag, ss in (('18 label-free variants other than alpha-ReQ', lf), ('the same + 2 LiDAR', lf + lid)):
        a_o = [S[(s, 'decline')]['at_after_old'] for s in ss]
        a_n = [S[(s, 'decline')]['at_after_new'] for s in ss]
        c_n = [S[(s, 'decline')]['cost_avg_median'] for s in ss]
        c_e = [S[(s, 'decline')]['cost_E_median'] for s in ss]
        print(f'\n{tag}: at/after onset old {min(a_o)}-{max(a_o)} of 27, new {min(a_n)}-{max(a_n)} of 27; '
              f'cost median avg R2 {min(c_n):.3f}-{max(c_n):.3f}, log-E R2 {min(c_e):.3f}-{max(c_e):.3f}')
        print('   new counts: ' + ', '.join(f"{s} {S[(s, 'decline')]['at_after_new']}" for s in ss))
    for s in [x for x in order if x.startswith('alpha_req') and kinds[x] == 'metric']:
        o = S[(s, 'decline')]
        print(f"{s}: before onset in {o['n'] - o['at_after_new']} of 27 (old {o['n'] - o['at_after_old']})")
    rev = [s for s in order if kinds[s] == 'metric-reversed']
    d_old = [abs(S[(s, 'decline')]['at_after_old'] - S[(s.replace(' [reversed]', ''), 'decline')]['at_after_old'])
             for s in rev]
    d_new = [abs(S[(s, 'decline')]['at_after_new'] - S[(s.replace(' [reversed]', ''), 'decline')]['at_after_new'])
             for s in rev]
    print(f'reversing a label-free variant moves its decline-run at/after count by: old {min(d_old)}-{max(d_old)}, '
          f'new {min(d_new)}-{max(d_new)} runs')
    print(f'pseudo-onsets used for runs without a decline: {pseudo}')


if __name__ == '__main__':
    main()
