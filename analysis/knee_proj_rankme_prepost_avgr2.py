"""Projector RankMe PRE/POST on the 13 capacity-tally runs, re-anchored at the
average-R2 pre-onset epoch (2026-10-10). CPU only.

Source numbers: the earlier projector report (the file REPORT names; not
included, so gate 1 runs only where it is present), section 3, table "PRE to
POST on the 13 capacity-tally runs (ladder epochs)", made by
knee_proj_bn_revision.py --tables -> knee_proj_audit.build_rows/cells, which
take PRE/POST through knee_labelfree_audit.prepost ->
knee_rank_estimators_prepost.ladder_epochs (results/knee_s3_ladder.csv). The
paper quotes the row "RankMe pub_uncentred, projector, BN batch (primary)":
PRE median 15.94, POST median 15.99 (whole-set batch-statistics BatchNorm on
the 4,000-event fitting split).

Nothing is re-implemented. Series: knee_proj_audit.load_proj() reads
results/knee_proj_rankme.csv (every checkpoint of the 48 LeJEPA runs, so the new
PRE epochs are already there; no encoding needed). The PRE/POST cells come
from knee_proj_audit.cells(S, rd)['prepost'], unchanged. The only change is
knee_rank_estimators_prepost.ladder_epochs -> the average-R2 anchors
(knee_rank_estimators_prepost_avgr2.patch: PRE = pre_onset_epoch of
results/knee_onsets_avgr2.csv, POST = final checkpoint, unchanged).

Gates / checks:
  1. Old anchors: cells()['prepost'] must reproduce the report's section-3
     rows cell for cell for the six projector RankMe rows (3 variants x BN
     batch / eval) and the three pooled RankMe rows.
  2. New anchors: runs whose PRE did not move have identical PRE and POST
     values to the old-anchor values (same CSV lookup; max |diff| printed).
  3. The CSV values at old PRE, new PRE and POST recomputed from the stored
     projector embeddings (proj_feats/<run>.pt, emb['fit_<bn>']) with the
     same imported functions (knee_labelfree_metrics.rankme_pub,
     rankme_sq_centred); max relative |diff| printed (CSV keeps 10 digits).
  4. Cross-check (if results/knee_rank_estimators_prepost_avgr2.csv exists): the
     pooled RankMe series of results/knee_labelfree_metrics.csv (stored pooled
     features) at the new anchors vs that CSV's re-encoded values
     (rankme_pub_uncentred, rankme_pub_centred, rankme_sq_centred).

Output: results/knee_proj_rankme_prepost_avgr2.csv (new; refuses to overwrite):
one row per run x variant x BN mode with old and new PRE epoch and value,
POST epoch and value, and the recomputed values.

Usage (from analysis/):
  CUDA_VISIBLE_DEVICES= python -B knee_proj_rankme_prepost_avgr2.py
"""
import sys

sys.dont_write_bytecode = True      # leave analysis/__pycache__ untouched

import csv  # noqa: E402
import os  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
sys.path.insert(0, _HERE)
import knee_proj_audit as KPA  # noqa: E402
import knee_rank_estimators_prepost as KR  # noqa: E402
import knee_rank_estimators_prepost_avgr2 as KRA  # noqa: E402

LFA = KPA.LFA
REPORT = os.path.join(os.environ.get('MONITOR_NOTES', os.path.join(paths.REPO, 'notes')), 'projector_lidar_rankme_2026-09-24.md')
SECTION = '### PRE to POST on the 13 capacity-tally runs (ladder epochs)'
PROJ_FEATS = os.path.join(paths.WORK, 'proj_feats')
RANK_AVGR2 = KRA.OUT                        # results/knee_rank_estimators_prepost_avgr2.csv
OUT = os.path.join(paths.CSV, 'knee_proj_rankme_prepost_avgr2.csv')
COLS = ['run', 'label', 'variant', 'bn_mode', 'old_pre_ep', 'pre_ep', 'post_ep',
        'old_pre_value', 'pre_value', 'post_value',
        'old_pre_recomputed', 'pre_recomputed', 'post_recomputed']


def report_rows():
    """{row label: [cells]} of the report's section-3 PRE/POST table."""
    out, on = {}, False
    for line in open(REPORT):
        if line.startswith('#'):
            on = line.strip() == SECTION
            continue
        if on and line.startswith('| ') and not line.startswith('|---'):
            parts = [p.strip() for p in line.strip().strip('|').split('|')]
            out[parts[0]] = parts[1:]
    return out


def series():
    """[(label in the report, S)] for the 6 projector and 3 pooled RankMe rows."""
    _, PR = KPA.load_proj()
    P58 = KPA.load_pooled()
    out = []
    for v in KPA.RANKME_VARIANTS:
        for b in ('batch', 'eval'):
            tag = 'BN batch (primary)' if b == 'batch' else 'BN eval (sensitivity)'
            out.append((f'RankMe {v}, projector, {tag}', ('proj', v, b), PR[(v, b)]))
        out.append((f'RankMe {v}, pooled (same 48 runs)', ('pooled', v, None),
                    KPA.drop_mae(P58[('rankme', v)])))
    return out


def recompute(old, new):
    """{(short, variant, bn, epoch): value} from the stored projector
    embeddings at the old PRE, new PRE and POST epochs."""
    import torch
    import knee_labelfree_metrics as LF
    out = {}
    for run in old:
        short = run.split('xxx')[0]
        O = torch.load(os.path.join(PROJ_FEATS, f'{run}.pt'), map_location='cpu', weights_only=True)
        eps = O['epochs']
        for ep in sorted({old[run][0], new[run][0], new[run][1]}):
            i = eps.index(ep)
            for b in KPA.BN_MODES:
                X = O['emb'][f'fit_{b}'][i].numpy()
                out[(short, 'pub_uncentred', b, ep)] = LF.rankme_pub(X, False)
                out[(short, 'pub_centred', b, ep)] = LF.rankme_pub(X, True)
                out[(short, 'sq_centred', b, ep)] = LF.rankme_sq_centred(X)
        del O
    return out


def main():
    assert not os.path.exists(OUT), f'refusing to overwrite {OUT}'
    rd = LFA.load_readability()
    R = report_rows() if os.path.exists(REPORT) else None   # internal report, gate 1 only
    SER = series()
    old = KR.ladder_epochs()                         # original reader, old anchors

    # ---- 1. old anchors reproduce the report
    print('GATE 1: old anchors (knee_s3_ladder) vs the report\'s section-3 PRE/POST rows'
          + ('' if R is not None else ' -- skipped, report not present'))
    old_cells, old_pp = {}, {}
    n_ok = 0
    for lab, key, S in SER:
        C = KPA.cells(S, rd)['prepost']
        old_cells[lab], old_pp[key] = C, LFA.prepost(S)
        ok = R is None or C == R[lab]
        n_ok += ok
        print(f'  {"ok " if ok else "BAD"} {lab:52s} {" | ".join(C)}' + ('' if ok else f'\n      report {R[lab]}'))
    print(f'  => {n_ok}/{len(SER)} rows identical cell for cell')
    assert n_ok == len(SER)

    # ---- new anchors
    KRA.patch()
    new = KR.ladder_epochs()
    assert list(new) == list(old)
    new_cells, new_pp = {}, {}
    for lab, key, S in SER:
        new_cells[lab] = KPA.cells(S, rd)['prepost']
        new_pp[key] = LFA.prepost(S)

    # ---- 2. unchanged-PRE runs: identical values
    worst = 0.0
    n_same = sum(old[r][0] == new[r][0] for r in old)
    for key in old_pp:
        for (k0, a0, b0), (k1, a1, b1), run in zip(old_pp[key], new_pp[key], old):
            assert k0 == k1 == run.split('xxx')[0]
            worst = max(worst, abs(b1 - b0))
            if old[run][0] == new[run][0]:
                worst = max(worst, abs(a1 - a0))
    print(f'\nGATE 2: {n_same} runs with unchanged PRE: max |diff| new vs old over their PRE and '
          f'all POST values, 9 series: {worst:.2e}')

    # ---- 3. recompute from stored embeddings
    REC = recompute(old, new)
    rel = 0.0
    for (lab, key, S) in SER:
        if key[0] != 'proj':
            continue
        _, v, b = key
        for run in old:
            short = run.split('xxx')[0]
            d = dict(zip(*S[short]))
            for ep in {old[run][0], new[run][0], new[run][1]}:
                rel = max(rel, abs(REC[(short, v, b, ep)] - d[ep]) / abs(d[ep]))
    print(f'\nCHECK 3: projector RankMe recomputed from proj_feats embeddings at old PRE, new PRE, '
          f'POST (13 runs x 3 variants x 2 BN modes): max relative |diff| vs CSV {rel:.2e}')

    # ---- 4. cross-check pooled series vs the re-encoded rank-estimator CSV
    if os.path.exists(RANK_AVGR2):
        rk = KRA.read_rows(RANK_AVGR2)
        m = {'pub_uncentred': 'rankme_pub_uncentred', 'pub_centred': 'rankme_pub_centred',
             'sq_centred': 'rankme_sq_centred'}
        rel4 = 0.0
        for (lab, key, S) in SER:
            if key[0] != 'pooled':
                continue
            for (short, a, b), run in zip(new_pp[key], old):
                r = rk[run]
                assert int(r['pre_ep']) == new[run][0] and int(r['post_ep']) == new[run][1]
                for got, col in ((a, f'{m[key[1]]}_pre'), (b, f'{m[key[1]]}_post')):
                    rel4 = max(rel4, abs(got - float(r[col])) / abs(got))
        print(f'\nCHECK 4: pooled RankMe (csv/knee_labelfree_metrics.csv, stored features) at the avgr2 '
              f'anchors vs re-encoded csv/{os.path.basename(RANK_AVGR2)}, 13 runs x 3 variants x PRE/POST: '
              f'max relative |diff| {rel4:.2e} (that CSV prints 6 decimals)')
    else:
        print(f'\nCHECK 4 skipped: {RANK_AVGR2} not present')

    # ---- table
    print('\nPRE to POST on the 13 capacity-tally runs: PRE median | POST median | rose / fell | '
          'median change (range) | median % change (range)')
    for lab, key, S in SER:
        print(f'  {lab}\n     old (ladder) : {" | ".join(old_cells[lab])}\n     new (avgr2)  : '
              f'{" | ".join(new_cells[lab])}')

    # ---- CSV
    rows = []
    for lab, key, S in SER:
        if key[0] != 'proj':
            continue
        _, v, b = key
        for (short, a0, _), (_, a1, b1), run in zip(old_pp[key], new_pp[key], old):
            rows.append([run, KR.label(run), v, b, old[run][0], new[run][0], new[run][1],
                         f'{a0:.10g}', f'{a1:.10g}', f'{b1:.10g}',
                         f'{REC[(short, v, b, old[run][0])]:.10g}', f'{REC[(short, v, b, new[run][0])]:.10g}',
                         f'{REC[(short, v, b, new[run][1])]:.10g}'])
    assert not os.path.exists(OUT), f'refusing to overwrite {OUT}'
    with open(OUT + '.partial', 'x', newline='') as fh:
        w = csv.writer(fh, lineterminator='\n')
        w.writerow(COLS)
        w.writerows(rows)
    os.replace(OUT + '.partial', OUT)
    print(f'\nwrote {OUT}: {len(rows)} rows')


if __name__ == '__main__':
    main()
