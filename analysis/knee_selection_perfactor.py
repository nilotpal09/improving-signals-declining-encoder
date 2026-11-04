"""Checkpoint-selection cost per factor, on the new readability measure
(2026-10-07/08: readability = ridge R2 of z1, z2, log E; sizes and costs per
factor; onsets from the average-R2 rule, results/knee_onsets_avgr2.csv).

Picks are read unchanged from results/knee_labelfree_selection_v2.csv (series,
run, pick_epoch; same candidates: grid checkpoints with epoch > 20, ties to the
earliest). For each pick and factor f in (z1, z2, E):
    cost_f = max over candidates of R2_f  -  R2_f at the pick,
and cost_avg the same on the average of the three. Readability at every
checkpoint from results/knee_labelfree_readability.csv (the file the v2 costs used).
Check first: cost_E must equal the stored r2e_cost on every row.

Onset-based columns, recomputed with the new onsets: decline runs use their
own onset; the other runs use the median new onset of the decline runs in
their campaign (the v1 pseudo-onset construction, knee_labelfree_selection.py
lines 19-21, with new onsets). Writes a NEW file,
results/knee_selection_perfactor.csv (refuses to overwrite), and prints medians.
"""
import sys

sys.dont_write_bytecode = True

import csv  # noqa: E402
import os  # noqa: E402
from collections import defaultdict  # noqa: E402
from statistics import median  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
CSV = os.path.join(paths.CSV)
SEL = os.path.join(CSV, 'knee_labelfree_selection_v2.csv')
READ = os.path.join(CSV, 'knee_labelfree_readability.csv')
ONS = os.path.join(CSV, 'knee_onsets_avgr2.csv')
OUT = os.path.join(CSV, 'knee_selection_perfactor.csv')
WARM = 20
F = ('z1', 'z2', 'E')
SHOW = ['loss', 'rankme pub_uncentred', 'lidar redrawn', 'lidar shared', 'alpha_req centred',
        'rule: final checkpoint', 'rule: first checkpoint > 20', 'rule: epoch 60']


def main():
    assert not os.path.exists(OUT), f'refusing to overwrite {OUT}'
    rd = defaultdict(dict)
    for r in csv.DictReader(open(READ)):
        rd[r['run'].split('xxx')[0]][int(r['epoch'])] = tuple(float(r[f'ridge_r2_{x}']) for x in F)
    ons = {r['run']: r for r in csv.DictReader(open(ONS))}
    sel = list(csv.DictReader(open(SEL)))

    # pseudo-onset: median new onset of the decline runs in each campaign
    camp = defaultdict(list)
    seen = set()
    for r in sel:
        if r['group'] == 'knee' and r['run'] not in seen:
            seen.add(r['run'])
            camp[r['campaign']].append(int(ons[r['run']]['onset']))
    pseudo = {c: median(v) for c, v in camp.items()}
    print('pseudo-onset per campaign (new):', pseudo)

    rows, worst = [], 0.0
    for r in sel:
        k = r['run']
        t = rd[k]
        cand = [e for e in sorted(t) if e > WARM]
        pe = int(r['pick_epoch'])
        assert pe in cand, (k, pe)
        out = {'series': r['series'], 'kind': r['kind'], 'run': k, 'group': r['group'],
               'campaign': r['campaign'], 'pick_epoch': pe}
        for i, x in enumerate(F):
            out[f'cost_{x}'] = max(t[e][i] for e in cand) - t[pe][i]
        avg = {e: sum(t[e]) / 3 for e in cand}
        out['cost_avg'] = max(avg.values()) - avg[pe]
        worst = max(worst, abs(out['cost_E'] - float(r['r2e_cost'])))
        is_dec = ons[k]['decline'] == '1'
        assert is_dec == (r['group'] == 'knee'), k
        ref = int(ons[k]['onset']) if is_dec else pseudo[r['campaign']]
        out['ref_epoch_new'] = ref
        out['pick_at_or_after_ref_new'] = int(pe >= ref)
        rows.append(out)
    print(f'check: max |cost_E - stored r2e_cost| = {worst:.2e} over {len(rows)} rows')
    assert worst < 1e-5

    cols = list(rows[0])
    with open(OUT, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator='\n')
        w.writeheader()
        for o in rows:
            w.writerow({c: (f'{v:.6f}' if isinstance(v, float) else v) for c, v in o.items()})
    print(f'wrote {OUT}: {len(rows)} rows')

    by = defaultdict(list)
    for o in rows:
        by[(o['series'], o['group'])].append(o)
    print('\nmedian cost (max) per factor; decline runs (n=27) | runs without a decline (n=30: 20 LeJEPA + 10 MAE)')
    print(f"{'series':30s} {'z1':>13s} {'z2':>13s} {'logE':>13s} {'avg':>13s} | {'z1':>6s} {'z2':>6s} {'logE':>6s} {'avg':>6s} | pick>=onset (decline)")
    for s in SHOW:
        k = by[(s, 'knee')]
        f = by[(s, 'event-free LeJEPA')] + by[(s, 'MAE')]
        def m(rs, c):
            return f"{median(o[c] for o in rs):.3f} ({max(o[c] for o in rs):.3f})"
        def mf(rs, c):
            return f"{median(o[c] for o in rs):.3f}"
        print(f"{s:30s} {m(k,'cost_z1'):>13s} {m(k,'cost_z2'):>13s} {m(k,'cost_E'):>13s} {m(k,'cost_avg'):>13s} | "
              f"{mf(f,'cost_z1'):>6s} {mf(f,'cost_z2'):>6s} {mf(f,'cost_E'):>6s} {mf(f,'cost_avg'):>6s} | "
              f"{sum(o['pick_at_or_after_ref_new'] for o in k)}/{len(k)}")


if __name__ == '__main__':
    main()
