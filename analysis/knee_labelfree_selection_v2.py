"""Within-run checkpoint selection, version 2 (2026-09-24): the condition
number kappa scored in the survey's direction, higher is better. Supersedes
knee_labelfree_selection.py (v1) and results/knee_labelfree_selection.csv, which
stay on disk unchanged as the superseded first version.

Why: v1 scored kappa (both variants) as lower-is-better, from Tsitsulin et
al.'s "stability" reading and the survey's sec. 3.1 remark on conditioning.
But the survey (arXiv 2608.23182v1) states a preferred direction for every
metric: Tables 8 to 16, caption, "Arrows indicate the preferred direction:
up (higher is better), down (lower is better), ->1 (closer to 1 is better)",
and kappa is marked "(up)" in all 20 occurrences (e.g. Table 11);
sec. 4.2: "kappa shows a statistically significant,
mild-to-moderate positive correlation" with OLS accuracy (which the survey
attributes to the OLS solver's conditioning). Tsitsulin et al. 2023 state no
quality direction for kappa (only "stability" and a perturbation bound); their
Fig. 1 and 2 'Cond. Num.' axes run from 1e-6 to 1e-2, which suggests the
plotted quantity is 1/kappa, and their Table 4 gives +0.53 and -0.43. So the
direction with a source is higher. The survey's arrows for the other six
metrics equal v1's directions: alpha-ReQ ->1, RankMe up, NE-Sum up,
Self-Cluster down, DSE up, ID down (20 occurrences each); --test checks this.

Everything else is v1, imported and unchanged (definitions, loaders, loss
alignment, rules, the report). The only change is v1.DIRECTION['kappa'] =
'higher' (and its flag), so the kappa rows of v2 equal v1's kappa
'[reversed]' rows, and v2's kappa '[reversed]' rows equal v1's kappa rows;
--test checks exactly that against the v1 CSV.

Usage (from analysis/):
  python -B knee_labelfree_selection_v2.py --test     # checks against the v1 CSV; writes nothing
  python -B knee_labelfree_selection_v2.py            # writes results/knee_labelfree_selection_v2.csv, prints tables
  python -B knee_labelfree_selection_v2.py --report   # tables from the v2 CSV
"""
import sys

sys.dont_write_bytecode = True      # leave analysis/__pycache__ untouched

import argparse  # noqa: E402
import csv  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
from statistics import median  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
sys.path.insert(0, _HERE)
import knee_labelfree_selection as V1  # noqa: E402

OUT = os.path.join(paths.CSV, 'knee_labelfree_selection_v2.csv')
V1_CSV = V1.OUT
KAPPA_FLAG = 'survey arrow up; its correlation attributed by the survey to the OLS solver'
SURVEY_ARROWS = {'alpha_req': 'near1', 'rankme': 'higher', 'ne_sum': 'higher', 'kappa': 'higher',
                 'self_cluster': 'lower', 'dse': 'higher', 'id_twonn': 'lower'}
KAPPA_SERIES = {'kappa uncentred', 'kappa centred', 'kappa uncentred [reversed]', 'kappa centred [reversed]'}

V1.DIRECTION['kappa'] = 'higher'
V1.FLAG['kappa'] = KAPPA_FLAG


def compute():
    return V1.compute()


def write_csv(rows):
    assert not os.path.exists(OUT), f'refusing to overwrite {OUT}'
    part = OUT + '.partial'
    assert not os.path.exists(part), f'refusing to overwrite {part}'
    with open(part, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=V1.COLS, lineterminator='\n')
        w.writeheader()
        w.writerows(V1.to_strings(rows))
    os.replace(part, OUT)
    print(f'wrote {OUT}: {len(rows)} rows')


def _sign(v):
    return '+' if v > 0 else '-'


def extra_report(rows, out=print):
    """Additions for v2: whole-run sign agreement next to the matched-window
    one, R^2(E)'s own pattern, and the knee headline table."""
    by = {}
    for r in rows:
        by.setdefault((r['series'], r['group']), []).append(r)

    def med(rs, key):
        return median(r[key] for r in rs if r[key] is not None and not math.isnan(r[key]))

    metric = [s for s in dict.fromkeys(r['series'] for r in rows)
              if by[(s, 'knee')][0]['kind'] == 'metric']
    free19 = lambda s: [r for r in by[(s, 'event-free LeJEPA')] if r['run'] not in V1.A2.TINY_FLAT]  # noqa: E731
    whole_same = [s for s in metric
                  if _sign(med(by[(s, 'knee')], 'rho_whole')) == _sign(med(by[(s, 'event-free LeJEPA')], 'rho_whole'))]
    win_same = [s for s in metric
                if (_sign(med(by[(s, 'knee')], 'rho_before')), _sign(med(by[(s, 'knee')], 'rho_after')))
                == (_sign(med(free19(s), 'rho_before')), _sign(med(free19(s), 'rho_after')))]
    out('\n## v2 additions\n')
    out(f'Same sign of median drift, knee vs event-free: matched window (before and after) {len(win_same)} of '
        f'{len(metric)}; whole run {len(whole_same)} of {len(metric)}. Whole run differing: '
        + ', '.join(s for s in metric if s not in whole_same) + '.')
    for s in ('readability R2(E)', 'readability mean CCA', 'loss'):
        k, f = by[(s, 'knee')], free19(s)
        out(f'{s}: knee before {med(k, "rho_before"):+.2f}, after {med(k, "rho_after"):+.2f}; event-free matched '
            f'before {med(f, "rho_before"):+.2f}, after {med(f, "rho_after"):+.2f}; whole run knee '
            f'{med(k, "rho_whole"):+.2f}, event-free {med(by[(s, "event-free LeJEPA")], "rho_whole"):+.2f}')
    out('\n### Knee headline (27 runs)\n')
    out('| series | direction | pick at/after onset | pick = first ckpt > 20 | R2(E) cost median (max) |')
    out('|---|---|---|---|---|')
    for s in ('alpha_req centred', 'alpha_req code_wls', 'alpha_req uncentred', 'kappa uncentred', 'kappa centred',
              'loss', 'lidar redrawn', 'lidar shared', 'rule: first checkpoint > 20', 'rule: final checkpoint',
              'rule: epoch 60', 'readability R2(E)'):
        rs = by[(s, 'knee')]
        c = [r['r2e_cost'] for r in rs]
        out(f'| {s} | {V1.SHOW_DIR[rs[0]["direction"]]} | {sum(r["pick_epoch"] >= r["ref_epoch"] for r in rs)} / 27 | '
            f'{sum(r["pick_epoch"] == r["first_epoch"] for r in rs)} / 27 | {median(c):.3f} ({max(c):.3f}) |')


def report(rows):
    def out(line):
        print(line.replace('csv/knee_labelfree_selection.csv', 'csv/knee_labelfree_selection_v2.csv'))
    V1.report(rows, out=out)
    extra_report(rows)


def test():
    T = V1.T
    print('=== directions: every metric scored in the survey\'s arrow direction ===')
    T.check('v2 DIRECTION equals the survey arrows (Tables 8-16)', dict(V1.DIRECTION), SURVEY_ARROWS)
    fresh = V1.to_strings(compute())
    with open(V1_CSV) as fh:
        v1 = list(csv.DictReader(fh))
    print('=== v2 against the v1 CSV ===')
    T.check('row count v2 = v1', len(fresh), len(v1))
    key = lambda r: (r['series'], r['run'])  # noqa: E731
    f_by, v_by = {key(r): r for r in fresh}, {key(r): r for r in v1}
    T.check('same (series, run) keys', set(f_by) == set(v_by), True)
    diff = [k for k in f_by if f_by[k] != v_by[k]]
    T.check('rows that differ from v1: 4 kappa series x 57 runs', len(diff), 4 * 57)
    T.check('every differing row is a kappa row', {k[0] for k in diff} <= KAPPA_SERIES, True)
    T.check('non-kappa rows identical to v1 (all fields)',
            all(f_by[k] == v_by[k] for k in f_by if k[0] not in KAPPA_SERIES), True)
    ign = {'series', 'kind', 'source_flag'}
    for s in ('kappa uncentred', 'kappa centred'):
        same = all({c: f_by[(s, r)][c] for c in V1.COLS if c not in ign}
                   == {c: v_by[(s + ' [reversed]', r)][c] for c in V1.COLS if c not in ign} for r in V1.ALL_RUNS)
        T.check(f'v2 "{s}" = v1 "{s} [reversed]" (all fields but name, kind, flag)', same, True)
        same_r = all({c: f_by[(s + ' [reversed]', r)][c] for c in V1.COLS if c not in ign}
                     == {c: v_by[(s, r)][c] for c in V1.COLS if c not in ign} for r in V1.ALL_RUNS)
        T.check(f'v2 "{s} [reversed]" = v1 "{s}" (all fields but name, kind, flag)', same_r, True)
        T.check(f'v2 "{s}" direction', {f_by[(s, r)]['direction'] for r in V1.ALL_RUNS}, {'higher'})
    if os.path.exists(OUT):
        print('=== the v2 CSV on disk equals a fresh recomputation ===')
        with open(OUT) as fh:
            disk = list(csv.DictReader(fh))
        T.check('v2 CSV rows identical to recomputation (all fields)', disk == fresh, True)
    print(f'\n{T.n_pass + T.n_fail} checks: {T.n_pass} PASS, {T.n_fail} FAIL')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', action='store_true')
    ap.add_argument('--report', action='store_true')
    a = ap.parse_args()
    if a.test:
        test()
    elif a.report:
        report(V1.read_csv(OUT))
    else:
        assert not os.path.exists(OUT), f'refusing to overwrite {OUT}'
        write_csv(compute())
        report(V1.read_csv(OUT))


if __name__ == '__main__':
    main()
