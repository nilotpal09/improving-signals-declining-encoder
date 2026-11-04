"""Loss trend before and after onset, with the onsets of the average-R2 rule
(2026-10-10). Recomputes the paper's "loss's fitted trend was decreasing in
25/25 pre-onset and 27/27 post-onset windows" (Appendix E, macros
\\lossTrendPre / \\lossTrendPost; source: the earlier selection analysis,
section 4, loss row).

The calculation is v1's, imported and unchanged (knee_labelfree_selection):
  series  the training loss as v1 reads it (V1.loss_series: census log,
          checkpoint N takes logged epoch N - 1), on the readability grid;
          also the every-iteration sensitivity series (source='all').
  trend   V1.drift(eps, loss, 'lower', lo, hi): within-run Spearman of
          minus the loss against epoch over the candidates (epoch > 20) in the
          window, if it holds >= 4 checkpoints (V1.MIN_RHO); rho > 0 means the
          loss decreases. Windows: before = epoch < reference, after =
          epoch >= reference.
Only the reference epoch changes: decline runs use their own new onset
(results/knee_onsets_avgr2.csv); runs without a decline use the campaign median of
the new onsets of the decline runs (110 at 21x, 70 at 4x), as
knee_selection_perfactor.py does.

Gate: with the OLD reference epochs (V1.ref_epoch), the census-series rho and
window sizes equal the 'loss' rows of results/knee_labelfree_selection_v2.csv for
all 57 runs (max |diff|), and the decline-run counts are 25/25 and 27/27.

Writes a NEW file, results/knee_loss_trend_avgr2.csv (refuses to overwrite):
one row per (series, run) with old and new reference, window sizes and rho.
"""
import sys

sys.dont_write_bytecode = True

import csv  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
from statistics import median  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
sys.path.insert(0, _HERE)
import knee_labelfree_selection as V1  # noqa: E402

CSV = os.path.join(paths.CSV)
V2 = os.path.join(CSV, 'knee_labelfree_selection_v2.csv')
ONS = os.path.join(CSV, 'knee_onsets_avgr2.csv')
OUT = os.path.join(CSV, 'knee_loss_trend_avgr2.csv')
V2_ORIG = os.path.join(CSV, 'orig_copies', 'knee_labelfree_selection_v2.csv')  # the recorded v2 table (57 runs)
SERIES = (('loss', 'census', 1), ('loss [all iterations, N-1]', 'all', 1))


def new_refs():
    ons = {r['run']: r for r in csv.DictReader(open(ONS))}
    dec = {k: int(ons[k]['onset']) for k in V1.ALL_RUNS if ons[k]['decline'] == '1'}
    assert set(dec) == set(V1.KNEE), 'new decline set differs from the 27 knee runs'
    camp = {}
    for k, o in dec.items():
        camp.setdefault(V1.A2.campaign(k), []).append(o)
    pseudo = {c: median(v) for c, v in camp.items()}
    return {k: (dec[k] if k in dec else pseudo[V1.A2.campaign(k)]) for k in V1.ALL_RUNS}, pseudo


def trend(eps, xs, ref):
    b, nb = V1.drift(eps, xs, 'lower', None, ref)
    a, na = V1.drift(eps, xs, 'lower', ref, None)
    return b, nb, a, na


def count(vals):
    v = [x for x in vals if x is not None and not math.isnan(x)]
    return sum(x > 0 for x in v), len(v), (median(v) if v else float('nan'))


def main():
    assert not os.path.exists(OUT), f'refusing to overwrite {OUT}'
    rd = V1.LA.load_readability()
    grid = {r: sorted(e for (rr, e) in rd if rr == r) for r in V1.ALL_RUNS}
    ref_new, pseudo = new_refs()
    v2 = {r['run']: r for r in csv.DictReader(open(V2)) if r['series'] == 'loss'}

    rows = []
    for name, src, shift in SERIES:
        S = V1.loss_series(grid, source=src, shift=shift)
        for run in V1.ALL_RUNS:
            eps, xs = S[run]
            ro = V1.ref_epoch(run)
            bo, nbo, ao, nao = trend(eps, xs, ro)
            bn, nbn, an, nan_ = trend(eps, xs, ref_new[run])
            rows.append(dict(series=name, run=run, group=V1.group_of(run), campaign=V1.A2.campaign(run),
                             ref_old=ro, n_before_old=nbo, rho_before_old=bo, n_after_old=nao, rho_after_old=ao,
                             ref_new=ref_new[run], n_before_new=nbn, rho_before_new=bn, n_after_new=nan_,
                             rho_after_new=an))

    # gate: old references reproduce the v2 'loss' rows
    worst, nmis = 0.0, 0
    for r in rows:
        if r['series'] != 'loss':
            continue
        v = v2[r['run']]
        for key in ('before', 'after'):
            got, n = r[f'rho_{key}_old'], r[f'n_{key}_old']
            exp = float(v[f'rho_{key}']) if v[f'rho_{key}'] != '' else None
            nmis += int(v[f'n_{key}']) != n
            if (got is None) != (exp is None):
                nmis += 1
            elif got is not None:
                worst = max(worst, abs(got - exp))
    orig27 = {r['run'] for r in csv.DictReader(open(V2_ORIG)) if r['group'] == 'knee'}  # the recorded counts are over these
    knee = [r for r in rows if r['series'] == 'loss' and r['group'] == 'knee' and r['run'] in orig27]
    pb, nb, _ = count([r['rho_before_old'] for r in knee])
    pa, na, _ = count([r['rho_after_old'] for r in knee])
    print(f'GATE: old references vs v2 loss rows (57 runs): max |rho diff| {worst:.1e} (CSV precision 1e-6), '
          f'window-size or definedness mismatches {nmis}; decline runs decreasing before {pb}/{nb}, after {pa}/{na} '
          f'(paper: 25/25 and 27/27)')
    assert worst < 1e-6 and nmis == 0 and (pb, nb, pa, na) == (25, 25, 27, 27)

    cols = list(rows[0])
    part = OUT + '.partial'
    assert not os.path.exists(part)
    with open(part, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow({c: ('' if v is None else (f'{v:.6f}' if isinstance(v, float) and c.startswith('rho')
                                                  else (f'{v:g}' if isinstance(v, float) else v)))
                        for c, v in r.items()})
    os.replace(part, OUT)
    print(f'wrote {OUT}: {len(rows)} rows')

    print(f'new pseudo-onsets for runs without a decline: {pseudo} (old: {V1.PSEUDO})')
    free19 = [r for r in V1.FREE if r not in V1.A2.TINY_FLAT]
    for name, _, _ in SERIES:
        print(f'\n## {name}: runs with decreasing loss (rho > 0) / runs with >= 4 checkpoints in the window; median rho')
        for gname, runs in (('decline (27)', list(V1.KNEE)), ('no decline LeJEPA, matched window (19, no ViT-Tiny)',
                                                              free19)):
            rs = [r for r in rows if r['series'] == name and r['run'] in runs]
            for ver in ('old', 'new'):
                b = count([r[f'rho_before_{ver}'] for r in rs])
                a = count([r[f'rho_after_{ver}'] for r in rs])
                print(f'  {gname:52s} {ver}: before {b[0]}/{b[1]} (median {b[2]:+.2f}), after {a[0]}/{a[1]} '
                      f'(median {a[2]:+.2f})')
        short = sorted((r['run'], r['ref_new'], r['n_before_new']) for r in rows
                       if r['series'] == name and r['group'] == 'knee' and r['rho_before_new'] is None)
        print(f'  decline runs with < 4 candidates before the new onset ({len(short)}): '
              + ', '.join(f'{k.replace("rigb_lejp_p1s100_", "").replace("rigd_lejp_p1s100_", "")} (onset {o})'
                          for k, o, _ in short))
        neg = [(r['run'], w) for r in rows if r['series'] == name and r['group'] == 'knee'
               for w in ('before', 'after') if r[f'rho_{w}_new'] is not None and r[f'rho_{w}_new'] <= 0]
        print(f'  decline-run windows with rho <= 0 under the new onsets: {neg if neg else "none"}')


if __name__ == '__main__':
    main()
