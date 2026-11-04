"""Rank estimators at PRE/POST re-anchored at the average-R2 pre-onset epoch
(2026-10-10; re-anchor of knee_rank_estimators_prepost.py).

Why: knee_rank_estimators_prepost.py took PRE from results/knee_s3_ladder.csv
(the old mean-CCA onset rule). The paper now times declines by the vetted
average-ridge-R2 rule, results/knee_onsets_avgr2.csv. PRE =
pre_onset_epoch there (last grid epoch <= onset - 20); POST = the final
checkpoint, unchanged (asserted equal to the ladder POST).

Nothing is re-implemented. knee_rank_estimators_prepost is imported and run
unchanged (its main(): same 13 runs, same 4,000-event fitting split of
rig_P1_Enar_S100.pt through wp1_probe_driver.encode_meanpool, same five
quantities); only two module attributes are replaced before it runs:
  KR.ladder_epochs -> avgr2_epochs (below): same run list and POST, PRE from
                      results/knee_onsets_avgr2.csv
  KR.OUT           -> results/knee_rank_estimators_prepost_avgr2.csv
KR.main() still prints its check against the record's old-anchor values;
for the six runs whose PRE moved those differences are expected.

New-PRE runs (old -> new): long_nano_1 140->130, nano_1 70->60, p384_2
120->110, p384_3 70->50, p384_7 60->40, p384_8 70->30. Unchanged: nano_2,
nano_3, p384_1, p384_4, p384_5, p384_6, p384_9.

Gate: for every run, the POST columns must reproduce
results/knee_rank_estimators_prepost.csv; for the seven runs whose PRE did not
move, the PRE and percent-change columns must too (max |diff| printed).

Output: results/knee_rank_estimators_prepost_avgr2.csv (new; refuses to
overwrite, checked before any encoding).

Usage (from analysis/; set the GPU):
  CUDA_VISIBLE_DEVICES=1 python -B knee_rank_estimators_prepost_avgr2.py
"""
import sys

sys.dont_write_bytecode = True      # leave analysis/__pycache__ untouched

import csv  # noqa: E402
import os  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
sys.path.insert(0, _HERE)
import knee_rank_estimators_prepost as KR  # noqa: E402

ONSETS_CSV = os.path.join(paths.CSV, 'knee_onsets_avgr2.csv')
OLD_CSV = KR.OUT                                   # results/knee_rank_estimators_prepost.csv
OUT = os.path.join(paths.CSV, 'knee_rank_estimators_prepost_avgr2.csv')
_LADDER_OLD = KR.ladder_epochs                     # the original (old-anchor) reader

# cross-check of the anchors against the expected table (label: new PRE)
EXPECTED_PRE = {'long_nano_1': 130, 'nano_1': 60, 'nano_2': 90, 'nano_3': 80,
                'p384_1': 30, 'p384_2': 110, 'p384_3': 50, 'p384_4': 120,
                'p384_5': 40, 'p384_6': 50, 'p384_7': 40, 'p384_8': 30,
                'p384_9': 100}


def avgr2_epochs():
    """{run: (PRE, POST)} for the same 13 runs as KR.ladder_epochs, with PRE =
    pre_onset_epoch of results/knee_onsets_avgr2.csv and POST = last_epoch there,
    asserted equal to the ladder POST. Checks every checkpoint file exists."""
    import wp1_probe_driver as drv
    ons = {}
    with open(ONSETS_CSV) as fh:
        for r in csv.DictReader(fh):
            ons[r['run'].split('xxx')[0]] = r
    out = {}
    for run, (_, post_old) in _LADDER_OLD().items():
        r = ons[run.split('xxx')[0]]
        assert r['decline'] == '1', run
        pre, post = int(r['pre_onset_epoch']), int(r['last_epoch'])
        assert post == post_old, (run, post, post_old)
        assert pre == EXPECTED_PRE[KR.label(run)], (run, pre)
        for ep in (pre, post):
            p = os.path.join(drv.ROOT, run, f'jepa-ep{ep}.pth.tar')
            assert os.path.exists(p), p
        out[run] = (pre, post)
    assert len(out) == 13
    return out


def patch():
    """Point knee_rank_estimators_prepost (and everything that imports it) at
    the average-R2 anchors."""
    KR.ladder_epochs = avgr2_epochs


def read_rows(path):
    with open(path) as fh:
        return {r['run']: r for r in csv.DictReader(fh)}


def gate(new_path, old_path, quants):
    """POST columns for all runs; PRE and pct columns where PRE did not move.
    Returns (max |diff| over value columns, max |diff| over pct columns,
    number of unchanged-PRE runs)."""
    new, old = read_rows(new_path), read_rows(old_path)
    assert set(new) == set(old), 'run sets differ'
    worst_val = worst_pct = 0.0
    n_same = 0
    print(f'\nGATE against {os.path.basename(old_path)}:')
    for run in sorted(new):
        a, b = new[run], old[run]
        assert a['post_ep'] == b['post_ep'], run
        same_pre = a['pre_ep'] == b['pre_ep']
        n_same += same_pre
        cols = [f'{q}_post' for q in quants]
        if same_pre:
            cols += [f'{q}_pre' for q in quants]
        d = max(abs(float(a[c]) - float(b[c])) for c in cols)
        worst_val = max(worst_val, d)
        msg = f'  {KR.label(run):12s} PRE {b["pre_ep"]:>3s}->{a["pre_ep"]:>3s} ' \
              f'{"(same)" if same_pre else "(moved)"}  max|diff| over ' \
              f'{"PRE+POST" if same_pre else "POST"} values {d:.2e}'
        if same_pre and f'{quants[0]}_pct' in a:
            dp = max(abs(float(a[f'{q}_pct']) - float(b[f'{q}_pct'])) for q in quants)
            worst_pct = max(worst_pct, dp)
            msg += f'; pct {dp:.2e}'
        print(msg)
    print(f'  => worst |diff| values {worst_val:.2e} (CSV prints 6 decimals); '
          f'pct {worst_pct:.2e} (2 decimals); {n_same} runs with unchanged PRE')
    return worst_val, worst_pct, n_same


def tally(path):
    rows = list(read_rows(path).values())
    out = {}
    for q in KR.QUANTS:
        fall = sorted(KR.label(r['run']) for r in rows if float(r[f'{q}_pct']) <= 0)
        out[q] = (len(rows) - len(fall), fall)
    joint = sorted(KR.label(r['run']) for r in rows
                   if not (float(r['rankme_sq_centred_pct']) > 0 and float(r['pr_centred_pct']) > 0))
    out['joint_sq_and_pr'] = (len(rows) - len(joint), joint)
    return out


def main():
    assert not os.path.exists(OUT), f'refusing to overwrite {OUT}'
    patch()
    KR.OUT = OUT
    print('anchors (avgr2): ' + ', '.join(
        f'{KR.label(r)} {p}/{q}' for r, (p, q) in KR.ladder_epochs().items()))
    print('NOTE: the "check against the record" block printed by KR.main() compares with the '
          'record\'s OLD-anchor values; differences are expected for the six runs whose PRE moved.\n',
          flush=True)
    KR.main()
    gate(OUT, OLD_CSV, KR.QUANTS)
    t_old, t_new = tally(OLD_CSV), tally(OUT)
    print('\nrises PRE->POST out of 13, old anchors | avgr2 anchors (runs that did not rise):')
    for q in list(KR.QUANTS) + ['joint_sq_and_pr']:
        (n0, f0), (n1, f1) = t_old[q], t_new[q]
        print(f'  {q:28s} {n0:2d}/13 {",".join(f0) or "-":28s} | {n1:2d}/13 {",".join(f1) or "-"}')


if __name__ == '__main__':
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        raise SystemExit('set CUDA_VISIBLE_DEVICES explicitly (shared GPUs)')
    main()
