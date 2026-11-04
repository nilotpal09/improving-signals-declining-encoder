"""Chronological detector study, implementing the detector protocol v3 (2026-09-24)
exactly as written.
CPU only; reads existing CSVs and run logs; writes five new files (refuses to overwrite).

DESIGNS (protocol sections 1 and 2)
  continuity        the standing detector (knee_s2_monitors.stdz + TESTS, thresholds 1.1 x max over each
                    campaign's four mu = 0.97 controls) at each run's original endpoint. Gate G1.
  standing_matched  the same detector, every run and control truncated before anything else at the
                    matched horizon: epoch 150 (21x campaign; MAE and ViT-Tiny count as 21x) or 300 (4x).
  D2_matched        stdz's standing constants (calibration epochs 30-60, increments for CUSUM and EWMA,
                    levels for Page-Hinkley and slope11); the test functions receive inputs from epoch 70
                    only, with fresh state. MAE (10) and ViT-Tiny flat (1) are scored with the 21x
                    thresholds, truncated at 150, as transfer checks.
  D1_matched        control-referenced. LeJEPA nano runs only (27 knee, 8 controls, 11 held-out).
                    increment(e) = x(e) - x(e - 10), e >= 30. Fit F(C) on a set C of controls:
                    r_C(e) = midpoint median over C of the increments at e; res_c(e) = inc_c(e) -
                    r_{C minus c}(e); s_C = 1.4826 x midpoint median over all (c, e) residuals, equal
                    weight, of |res - midpoint median(res)|; s_C zero, non-finite or not computable ->
                    the fit fails (no floor). Monitored run: z(e) = (inc(e) - r_C(e)) / s_C with C = all 4
                    controls. Control j (threshold setting): fitted with F(C minus j), scored with its own
                    increments. CUSUM and EWMA take z; Page-Hinkley and slope11 take L(e) = z(30) + ... +
                    z(e). Any failed fit -> that series' D1 threshold in that campaign is unavailable.
  The four test functions are knee_s2_monitors.cusum / ewma / page_hinkley / wslope, unchanged.

RULES (sections 3 to 5)
  complete finite input on the declared grid (first value needed to the horizon), else the run is
  unscorable for that design with the reason; a missing or non-finite control value makes the
  threshold unavailable. Alarm = first epoch with statistic > threshold (strict), no reset. Timely =
  |first alarm - onset| <= 10. Per knee run: evaluation status, onset-window coverage (full / partial /
  none over the grid epochs onset - 10, onset, onset + 10), first-alarm status (early / timely / late /
  no alarm, or "no result" with its reason), population membership. Window ratio = max statistic in
  [onset - 10, onset + 10] / threshold: empty window n/a, 0/0 undefined, positive/0 infinite.
  Eligible event: whole window inside the horizon and inside the test's output period (CUSUM, EWMA,
  Page-Hinkley: from the first input; slope11: 110 epochs later).

SERIES (section 6): the 21 label-free variants (results/knee_labelfree_metrics.csv, via
  knee_labelfree_audit.load_metrics); training loss (knee_labelfree_selection.loss_series, census
  series, checkpoint epoch N <- logged epoch N - 1); pooled LiDAR at delta 1e-4 (results/knee_lidar.csv,
  results/knee_lidar_sharedshape.csv, n_spots = 100, via knee_labelfree_audit.load_lidar); pooled LiDAR at
  delta 1e-6 and 1e-8, both groupings (results/knee_lidar_delta_sweep.csv, float64; standing_matched and
  D1 only); projector LiDAR (2 groupings) and projector RankMe (3 variants), BN batch (primary) and
  eval (sensitivity) (knee_proj_audit.load_proj). The projector has no MAE rows (n/a).

Usage (from analysis/):
  python -B knee_detector_chrono.py --test     # gate G2, deterministic fixtures; writes nothing
  python -B knee_detector_chrono.py --gate     # gate G1, continuity mode vs the standing record; writes nothing
  python -B knee_detector_chrono.py --run      # writes results/knee_detector_chrono_{runs,summary,fits,manifest}.csv
                                               #   and results/knee_detector_chrono_paths.csv.gz
  python -B knee_detector_chrono.py --report   # tables in the protocol's reporting order, from the CSVs
"""
import sys

sys.dont_write_bytecode = True      # leave analysis/__pycache__ untouched

import argparse  # noqa: E402
import csv  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
sys.path.insert(0, _HERE)
import knee_s2_monitors as S2  # noqa: E402
import knee_lidar_audit as AUD  # noqa: E402
import knee_lidar_audit_sharedshape as A2  # noqa: E402
import knee_lidar_replica_seed as RS  # noqa: E402
import knee_labelfree_audit as LFA  # noqa: E402
import knee_labelfree_selection as SEL  # noqa: E402
import knee_proj_audit as PA  # noqa: E402

REPO = paths.REPO
MON = os.environ.get('MONITOR_NOTES', os.path.join(REPO, 'notes'))   # internal reports (not released)
PROTOCOL = os.path.join(MON, 'detector_protocol_v3_2026-09-24.md')
PROTOCOL_REVISION = 'v3 (detector_protocol_v3_2026-09-24.md)'
CSV_DIR = os.path.join(paths.CSV)
OUT = {'runs': os.path.join(CSV_DIR, 'knee_detector_chrono_runs.csv'),
       'summary': os.path.join(CSV_DIR, 'knee_detector_chrono_summary.csv'),
       'fits': os.path.join(CSV_DIR, 'knee_detector_chrono_fits.csv'),
       'paths': os.path.join(CSV_DIR, 'knee_detector_chrono_paths.csv.gz'),
       'manifest': os.path.join(CSV_DIR, 'knee_detector_chrono_manifest.csv')}
SWEEP_CSV = os.path.join(CSV_DIR, 'knee_lidar_delta_sweep.csv')
REPORT_LF = os.path.join(MON, 'labelfree_metrics_2026-09-23.md')
REPORT_PROJ = os.path.join(MON, 'projector_lidar_rankme_2026-09-24.md')
REPORT_SHARED = os.path.join(MON, 'lidar_sharedshape_2026-09-23.md')

# ---------------------------------------------------------------- fixed constants
TEST_ORDER = ['CUSUM', 'EWMA', 'PageHinkley', 'slope11']
MAIN_TESTS = ['CUSUM', 'EWMA', 'PageHinkley']
DESIGNS = ['continuity', 'standing_matched', 'D2_matched', 'D1_matched']
MATCHED = ['standing_matched', 'D2_matched', 'D1_matched']
ALL4 = list(DESIGNS)
HORIZON = {'21x': 150, '4x': 300}
STEP = 10
FIRST_INPUT = {'continuity': 30, 'standing_matched': 30, 'D2_matched': 70, 'D1_matched': 30}
SLOPE_W = 11                       # knee_s2_monitors.wslope default window
WARM = S2.WARM                     # 20
THR_MULT = 1.1
TIMELY = 10
MAD_K = 1.4826
CAMPS = ('21x', '4x')
TEST_FNS = {t: fn for t, (fn, _) in S2.TESTS.items()}     # indirection only so --test can record inputs
ON_INC = {t: inc for t, (_, inc) in S2.TESTS.items()}     # CUSUM, EWMA: True; PageHinkley, slope11: False
VALID_KNEE = ('early', 'timely', 'late', 'no alarm')
VALID_OTHER = ('alarm', 'no alarm')

KNEE = dict(AUD.KNEE_21)
KNEE.update(AUD.KNEE_4)


def short(run):
    return A2.short(run)


def first_output_rule(design, test):
    """Epoch of the first statistic output on the 10-epoch grid (section 5)."""
    return FIRST_INPUT[design] + (SLOPE_W * STEP if test == 'slope11' else 0)


# ---------------------------------------------------------------- cohorts
class Cohort:
    """Named runs, onsets and campaign mapping. REAL is the protocol's cohort; --test builds fixtures."""

    def __init__(self, knee, ctrl, held, mae, tiny, campaign):
        self.knee, self.ctrl, self.held, self.mae, self.tiny = knee, ctrl, held, mae, tiny
        self.campaign = campaign

    def role(self, run):
        if run in self.knee:
            return 'knee'
        if any(run in v for v in self.ctrl.values()):
            return 'control'
        if any(run in v for v in self.held.values()):
            return 'heldout'
        if run in self.mae:
            return 'mae'
        if run in self.tiny:
            return 'tiny_flat'
        return 'other'

    def scope(self, design):
        """[(run, role, in_scope_note)]: note '' = scored; otherwise the n/a reason."""
        out = [(r, 'knee', '') for r in self.knee]
        out += [(r, 'control', '') for c in CAMPS for r in self.ctrl[c]]
        out += [(r, 'heldout', '') for c in CAMPS for r in self.held[c]]
        if design in ('continuity', 'D2_matched'):
            out += [(r, 'mae', '') for r in self.mae] + [(r, 'tiny_flat', '') for r in self.tiny]
        elif design == 'D1_matched':
            na = 'n/a: no same-family controls (protocol 2.3)'
            out += [(r, 'mae', na) for r in self.mae] + [(r, 'tiny_flat', na) for r in self.tiny]
        return out


REAL = Cohort(knee=KNEE, ctrl={'21x': list(AUD.CTRL_21), '4x': list(AUD.CTRL_4)},
              held={'21x': list(A2.HELD_21), '4x': list(A2.HELD_4)}, mae=list(A2.MAE),
              tiny=list(A2.TINY_FLAT), campaign=A2.campaign)


# ---------------------------------------------------------------- arithmetic
def mid_median(vals):
    """Midpoint median: for an even count, the average of the two middle values (protocol 2.3).
    Infinite values sort last; an empty list raises."""
    s = sorted(vals)
    n = len(s)
    if n == 0:
        raise ValueError('median of an empty list')
    if n % 2:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def check_complete(S_run, first, H):
    """(ok, reason). Every grid epoch first, first + 10, ..., H present and finite, and no epoch in
    (20, H] off the 10-epoch grid (it would change which values stdz treats as adjacent)."""
    if S_run is None:
        return False, 'run absent from series'
    eps, xs = S_run
    d = dict(zip(eps, xs))
    for e in range(first, H + 1, STEP):
        if e not in d:
            return False, f'missing value at epoch {e}'
        if not math.isfinite(d[e]):
            return False, f'non-finite value at epoch {e}'
    off = [e for e in eps if WARM < e <= H and e % STEP]
    if off:
        return False, f'off-grid epoch(s) {off}'
    return True, ''


def truncate(S_run, H):
    eps, xs = S_run
    keep = [(e, x) for e, x in zip(eps, xs) if e <= H]
    return [e for e, _ in keep], [x for _, x in keep]


def stdz_constants(te, tx, on_inc):
    """stdz's calibration constants, the same lines as knee_s2_monitors.stdz; checked against its
    output on every run (standing_inputs asserts the reconstruction)."""
    v = [tx[i] - tx[i - 1] for i in range(1, len(te))] if on_inc else tx[1:]
    e2 = te[1:]
    cal = [x for e, x in zip(e2, v) if 21 <= e <= 60]
    mu = sum(cal) / len(cal)
    sd_raw = (sum((x - mu) ** 2 for x in cal) / len(cal)) ** 0.5
    return dict(mu=mu, sd_raw=sd_raw, sd=sd_raw or 1e-12, n_cal=len(cal),
                cal_epochs=[e for e in e2 if 21 <= e <= 60], values=dict(zip(e2, v)))


def standing_inputs(S_run, test, H, first_input):
    """Standing standardisation (knee_s2_monitors.stdz, unchanged) on the series truncated at H, then
    only inputs with epoch >= first_input are passed on (D2: 70). Returns (inputs, constants)."""
    on_inc = ON_INC[test]
    te, tx = truncate(S_run, H)
    z = S2.stdz(te, tx, on_inc)
    k = stdz_constants(te, tx, on_inc)
    for e, v in z:
        assert v == (k['values'][e] - k['mu']) / k['sd'], ('stdz reconstruction', e)
    return [(e, v) for e, v in z if e >= first_input], k


def run_test(test, inputs):
    return TEST_FNS[test](inputs)


def increments(S_run, H):
    d = dict(zip(*S_run))
    return {e: d[e] - d[e - STEP] for e in range(30, H + 1, STEP)}


def d1_fit(incs, members, epochs):
    """F(C) of protocol 2.3. incs: control -> {epoch: increment}."""
    f = dict(members=list(members), epochs=list(epochs), r={}, inner={}, res=[], res_median=None,
             mad=None, s=None, ok=False, reason='')
    if len(members) < 2:
        f['reason'] = f'cannot be computed: {len(members)} control(s)'
        return f
    f['r'] = {e: mid_median([incs[c][e] for c in members]) for e in epochs}
    res = []
    for c in members:
        others = [o for o in members if o != c]
        rc = {e: mid_median([incs[o][e] for o in others]) for e in epochs}
        f['inner'][c] = rc
        res += [incs[c][e] - rc[e] for e in epochs]
    f['res'] = res
    med = mid_median(res)
    mad = mid_median([abs(x - med) for x in res])
    s = MAD_K * mad
    f['res_median'], f['mad'], f['s'] = med, mad, s
    if not math.isfinite(s):
        f['reason'] = 'scale non-finite'
    elif s == 0:
        f['reason'] = 'scale zero'
    else:
        f['ok'] = True
    return f


def cumulate(z):
    L, out = 0.0, []
    for e, v in z:
        L += v
        out.append((e, L))
    return out


def d1_inputs(z, test):
    return z if ON_INC[test] else cumulate(z)


def fit_label(members, allc):
    if len(members) == len(allc):
        return 'F(C)'
    miss = [c for c in allc if c not in members]
    return 'F(C-' + '-'.join(short(m) for m in miss) + ')'


def d1_campaign(S, controls, H):
    """All five fits for one series and campaign. Returns dict with fits (key 'all' or control name),
    incs, epochs, ok_all (every fit ok), fail (reason string or '')."""
    epochs = list(range(30, H + 1, STEP))
    out = dict(controls=list(controls), H=H, epochs=epochs, fits={}, incs={}, ok_all=False, fail='')
    bad = []
    for c in controls:
        ok, why = check_complete(S.get(c), WARM, H)
        if not ok:
            bad.append(f'{short(c)} {why}')
    if bad:
        why = 'control input incomplete: ' + '; '.join(bad)
        for key in ['all'] + list(controls):
            out['fits'][key] = dict(members=[c for c in controls if c != key], epochs=epochs, r={}, inner={},
                                    res=[], res_median=None, mad=None, s=None, ok=False,
                                    reason='cannot be computed: ' + why)
        out['fits']['all']['members'] = list(controls)
        out['fail'] = why
        return out
    incs = {c: increments(S[c], H) for c in controls}
    out['incs'] = incs
    out['fits']['all'] = d1_fit(incs, controls, epochs)
    for j in controls:
        out['fits'][j] = d1_fit(incs, [c for c in controls if c != j], epochs)
    failed = [(k, f['reason']) for k, f in out['fits'].items() if not f['ok']]
    if failed:
        out['fail'] = 'fit failed: ' + '; '.join(
            f'{fit_label(out["fits"][k]["members"], controls)} {why}' for k, why in failed)
    else:
        out['ok_all'] = True
    return out


def threshold_from(ctrl_paths, controls):
    """ctrl_paths: control -> (status_ok, reason, path). 1.1 x max over the controls' statistics
    (the standing rule of knee_lidar_audit_sharedshape.thresholds). Returns a dict."""
    bad = [f'{short(c)}: {ctrl_paths[c][1]}' for c in controls if not ctrl_paths[c][0]]
    if bad:
        return dict(status='undefined', value=None, reason='; '.join(bad), setter=None, peaks={})
    paths = {c: ctrl_paths[c][2] for c in controls}
    nonempty = [c for c in controls if paths[c]]
    if not nonempty:
        return dict(status='undefined', value=None, reason='no control produced output', setter=None,
                    peaks={c: (None, None) for c in controls})
    if len(nonempty) != len(controls):
        raise RuntimeError('some control paths empty and others not: not covered by the protocol')
    m = 0.0
    peaks = {}
    for c in controls:
        pk = max(v for _, v in paths[c])
        peaks[c] = (pk, next(e for e, v in paths[c] if v == pk))
        m = max(m, pk)
    setter = next(c for c in controls if peaks[c][0] == m)
    return dict(status='defined', value=THR_MULT * m, reason='', setter=setter, peaks=peaks)


def first_alarm(path, thr):
    return next((e for e, v in path if v > thr), None)


def knee_status(a, onset):
    if a is None:
        return 'no alarm'
    d = a - onset
    if abs(d) <= TIMELY:
        return 'timely'
    return 'early' if d < -TIMELY else 'late'


def window_epochs(onset):
    return [onset - TIMELY, onset, onset + TIMELY]


def coverage(path, onset):
    have = {e for e, _ in path}
    n = sum(e in have for e in window_epochs(onset))
    return 'full' if n == 3 else ('none' if n == 0 else 'partial')


def window_ratio(path, onset, thr):
    """(ratio or None, status). status: finite / infinite / undefined (0/0) / n/a (empty window)."""
    vals = [v for e, v in path if onset - TIMELY <= e <= onset + TIMELY]
    if not vals:
        return None, 'n/a (empty window)', None
    mx = max(vals)
    if thr > 0:
        return mx / thr, 'finite', mx
    if mx == 0:
        return None, 'undefined (0/0)', mx
    return math.inf, 'infinite', mx


def ratio_summary(statuses_values):
    """[(status, value)] -> dict of counts and the median / min / max over finite and infinite values
    (infinity kept and sorted last; undefined and n/a counted and left out)."""
    cnt = defaultdict(int)
    vals = []
    for st, v in statuses_values:
        cnt[st] += 1
        if st in ('finite', 'infinite'):
            vals.append(v)
    out = dict(n_finite=cnt['finite'], n_infinite=cnt['infinite'], n_undefined=cnt['undefined (0/0)'],
               n_empty=cnt['n/a (empty window)'], n_noresult=sum(n for k, n in cnt.items() if k.startswith('no result')),
               median=None, min=None, max=None)
    if vals:
        out['median'], out['min'], out['max'] = mid_median(vals), min(vals), max(vals)
    return out


def eligible(design, test, onset, H_run):
    fo = first_output_rule(design, test)
    return onset - TIMELY >= fo and onset + TIMELY <= H_run


def any_output_in_window(design, test, onset, H_run):
    fo = first_output_rule(design, test)
    return any(fo <= e <= H_run for e in window_epochs(onset))


# ---------------------------------------------------------------- evaluation of one series and design
def horizon_for(design, S_run, camp, horizons=HORIZON):
    if design == 'continuity':
        return max(S_run[0]) if S_run is not None else None
    return horizons[camp]


def main21_set(cohort=REAL):
    """Intersection of the three matched designs' eligible sets for CUSUM, EWMA, Page-Hinkley."""
    out = []
    for run, o in cohort.knee.items():
        H = HORIZON[cohort.campaign(run)]
        if all(eligible(d, 'CUSUM', o, H) for d in MATCHED):
            out.append(run)
    return out


def populations(design, test, run, onset, H_run, cohort=REAL):
    """Comparison populations (section 5) a knee run belongs to, for this design and test."""
    pops = []
    if design == 'continuity':
        pops.append('continuity27')
        if test == 'slope11':
            if any_output_in_window(design, test, onset, H_run):
                pops.append('slope11_any_output')
            if eligible(design, test, onset, H_run):
                pops.append('slope11_full_window')
        return pops
    if test == 'slope11':
        return ['knee_all']
    m21 = main21_set(cohort)
    if run in m21:
        pops.append('main21')
    if design in ('standing_matched', 'D1_matched') and eligible(design, test, onset, H_run):
        pops.append('matched25')
    return pops


def evaluate(S, design, cohort=REAL, keep_paths=True, horizons=HORIZON):
    """Scores every run in the design's scope. Returns dict(recs, thr, fits, paths, d1)."""
    recs, fits, paths = [], [], []
    thr = {}
    d1 = {}
    ctrl_paths = {}      # (camp, test) -> control -> (ok, reason, path)
    run_paths = {}       # (run, test) -> (ok, reason, inputs, path, H, input_kind)
    stdk = {}            # (run, on_inc) -> constants (standing designs)
    scope = cohort.scope(design)

    def score_standing(run, test, camp):
        S_run = S.get(run)
        H = horizon_for(design, S_run, camp, horizons)
        first_needed = WARM if ON_INC[test] else 30
        if S_run is None:
            return False, 'run absent from series', [], [], H, ''
        ok, why = check_complete(S_run, first_needed, H)
        if not ok:
            return False, why, [], [], H, ''
        inputs, k = standing_inputs(S_run, test, H, FIRST_INPUT[design])
        stdk[(run, ON_INC[test])] = k
        return True, '', inputs, run_test(test, inputs), H, ('stdz increments' if ON_INC[test] else 'stdz levels')

    if design == 'D1_matched':
        for camp in CAMPS:
            d1[camp] = d1_campaign(S, cohort.ctrl[camp], horizons[camp])

    def score_d1(run, test, camp, as_control):
        H = horizons[camp]
        dc = d1[camp]
        S_run = S.get(run)
        if S_run is None:
            return False, 'run absent from series', [], [], H, ''
        ok, why = check_complete(S_run, WARM, H)
        if not ok:
            return False, why, [], [], H, ''
        F = dc['fits'][run] if as_control else dc['fits']['all']
        if not F['ok']:
            lab = fit_label(F['members'], dc['controls'])
            return False, f'{lab} failed: {F["reason"]}', [], [], H, ''
        inc = increments(S_run, H)
        z = [(e, (inc[e] - F['r'][e]) / F['s']) for e in dc['epochs']]
        inputs = d1_inputs(z, test)
        return True, '', inputs, run_test(test, inputs), H, ('z' if ON_INC[test] else 'L')

    # thresholds
    for camp in CAMPS:
        for test in TEST_ORDER:
            cp = {}
            for c in cohort.ctrl[camp]:
                if design == 'D1_matched':
                    ok, why, inp, p, H, kind = score_d1(c, test, camp, True)
                    if not ok and d1[camp]['fail'] and 'control input incomplete' in d1[camp]['fail']:
                        why = d1[camp]['fail']
                else:
                    ok, why, inp, p, H, kind = score_standing(c, test, camp)
                cp[c] = (ok, why, p)
                run_paths[(c, test)] = (ok, why, inp, p, H, kind)
            t = threshold_from(cp, cohort.ctrl[camp])
            if design == 'D1_matched' and t['status'] == 'defined' and not d1[camp]['ok_all']:
                t = dict(status='undefined', value=None, reason=d1[camp]['fail'], setter=None, peaks=t['peaks'])
            if design == 'D1_matched' and t['status'] == 'undefined' and d1[camp]['fail'] and not t['reason'].startswith(d1[camp]['fail']):
                t['reason'] = d1[camp]['fail'] + ('; ' + t['reason'] if t['reason'] else '')
            thr[(camp, test)] = t
            ctrl_paths[(camp, test)] = cp

    # every run in scope
    for run, rl, na in scope:
        camp = cohort.campaign(run)
        onset = cohort.knee.get(run)
        for test in TEST_ORDER:
            base = dict(design=design, test=test, campaign=camp, run=run, run_short=short(run), role=rl,
                        onset=onset)
            if na or (S.get(run) is None and rl in ('mae', 'tiny_flat')):
                note = na or 'n/a: series does not exist for this model'
                base.update(horizon=None, eval_status='n/a', status_reason=note, path_status='', n_outputs=None,
                            first_output_epoch=None, last_output_epoch=None, threshold=None,
                            threshold_status='', threshold_reason='', peak=None, peak_epoch=None,
                            first_alarm_epoch=None, alarm_minus_onset=None, alarm_status='n/a',
                            result_reason=note, coverage='', window_max=None, window_ratio=None,
                            window_ratio_status='', eligible=None, populations='')
                recs.append(base)
                continue
            if (run, test) in run_paths:
                ok, why, inputs, path, H, kind = run_paths[(run, test)]
            elif design == 'D1_matched':
                ok, why, inputs, path, H, kind = score_d1(run, test, camp, False)
            else:
                ok, why, inputs, path, H, kind = score_standing(run, test, camp)
            t = thr[(camp, test)]
            base.update(horizon=H, eval_status='scorable' if ok else 'unscorable', status_reason=why,
                        path_status=('nonempty' if path else 'empty') if ok else '',
                        n_outputs=len(path) if ok else None,
                        first_output_epoch=path[0][0] if path else None,
                        last_output_epoch=path[-1][0] if path else None,
                        threshold=t['value'], threshold_status=t['status'], threshold_reason=t['reason'])
            if path:
                pk = max(v for _, v in path)
                base.update(peak=pk, peak_epoch=next(e for e, v in path if v == pk))
            else:
                base.update(peak=None, peak_epoch=None)
            reasons = []
            if not ok:
                reasons.append('unscorable')
            else:
                if not path:
                    reasons.append('empty path')
                if t['status'] != 'defined':
                    reasons.append('undefined threshold')
            a = first_alarm(path, t['value']) if not reasons else None
            base['first_alarm_epoch'] = a
            base['alarm_minus_onset'] = (a - onset) if (a is not None and onset is not None) else None
            if reasons:
                base['alarm_status'] = 'no result'
                base['result_reason'] = '; '.join(reasons)
            elif rl == 'knee':
                base['alarm_status'] = knee_status(a, onset)
                base['result_reason'] = ''
            else:
                base['alarm_status'] = 'alarm' if a is not None else 'no alarm'
                base['result_reason'] = ''
            if rl == 'knee':
                base['coverage'] = coverage(path, onset) if ok else 'n/a (unscorable)'
                base['eligible'] = int(eligible(design, test, onset, H)) if H is not None else None
                base['populations'] = ';'.join(populations(design, test, run, onset, H, cohort)) if H else ''
                if not ok:
                    base.update(window_max=None, window_ratio=None, window_ratio_status='no result (unscorable)')
                elif t['status'] != 'defined':
                    wv = [v for e, v in path if onset - TIMELY <= e <= onset + TIMELY]
                    base.update(window_max=max(wv) if wv else None, window_ratio=None,
                                window_ratio_status='no result (undefined threshold)' if wv else 'n/a (empty window)')
                else:
                    r, st, mx = window_ratio(path, onset, t['value'])
                    base.update(window_max=mx, window_ratio=r, window_ratio_status=st)
            else:
                base.update(coverage='', window_max=None, window_ratio=None, window_ratio_status='',
                            eligible=None, populations='')
            recs.append(base)
            if keep_paths and ok:
                paths.append(dict(design=design, test=test, run=run, campaign=camp, role=rl, input_kind=kind,
                                  epochs=[e for e, _ in inputs], inputs=[v for _, v in inputs],
                                  stat_epochs=[e for e, _ in path], stats=[v for _, v in path]))

    # fits
    for camp in CAMPS:
        for test in TEST_ORDER:
            t = thr[(camp, test)]
            fits.append(dict(design=design, campaign=camp, fit='threshold', run='', members='', quantity=f'threshold {test}',
                             epoch='', value=t['value'], detail=(f'status {t["status"]}; ' + (
                                 f'set by {short(t["setter"])} (peak {t["peaks"][t["setter"]][0]!r} at epoch '
                                 f'{t["peaks"][t["setter"]][1]})' if t['status'] == 'defined' else t['reason']))))
            for c, (pk, pe) in t['peaks'].items():
                fits.append(dict(design=design, campaign=camp, fit='threshold', run=c, members='',
                                 quantity=f'control peak {test}', epoch=pe if pe is not None else '', value=pk,
                                 detail=''))
    if design == 'D1_matched':
        for camp in CAMPS:
            dc = d1[camp]
            for key, F in dc['fits'].items():
                lab = fit_label(F['members'], dc['controls'])
                mem = ' '.join(short(m) for m in F['members'])
                fits.append(dict(design=design, campaign=camp, fit=lab, run='', members=mem, quantity='status',
                                 epoch='', value='', detail='ok' if F['ok'] else 'failed: ' + F['reason']))
                for q in ('s', 'mad', 'res_median'):
                    fits.append(dict(design=design, campaign=camp, fit=lab, run='', members=mem,
                                     quantity={'s': 'scale s', 'mad': 'MAD', 'res_median': 'residual median'}[q],
                                     epoch='', value=F[q], detail=''))
                fits.append(dict(design=design, campaign=camp, fit=lab, run='', members=mem,
                                 quantity='n residuals', epoch='', value=len(F['res']), detail=''))
                for e in sorted(F['r']):
                    fits.append(dict(design=design, campaign=camp, fit=lab, run='', members=mem,
                                     quantity='reference r(e)', epoch=e, value=F['r'][e], detail=''))
                for c, rc in F['inner'].items():
                    inner_mem = ' '.join(short(m) for m in F['members'] if m != c)
                    for e in sorted(rc):
                        fits.append(dict(design=design, campaign=camp, fit=lab, run=c, members=inner_mem,
                                         quantity='inner reference for residual of run', epoch=e, value=rc[e],
                                         detail=''))
    else:
        for (run, inc), k in sorted(stdk.items()):
            lab = 'stdz increments' if inc else 'stdz levels'
            for q in ('mu', 'sd_raw', 'sd', 'n_cal'):
                fits.append(dict(design=design, campaign=cohort.campaign(run), fit=lab, run=run, members='',
                                 quantity={'mu': 'mean', 'sd_raw': 'sd (divisor n)', 'sd': 'sd used (floor 1e-12 '
                                           'if zero, as stdz)', 'n_cal': 'calibration values'}[q],
                                 epoch='', value=k[q],
                                 detail='calibration epochs ' + ' '.join(str(e) for e in k['cal_epochs'])
                                 if q == 'n_cal' else ''))
    return dict(recs=recs, thr=thr, fits=fits, paths=paths, d1=d1)


# ---------------------------------------------------------------- summaries
SUM_COLS = ['series', 'family', 'series_role', 'design', 'test', 'campaign', 'population', 'quantity', 'count',
            'scorable', 'planned', 'value', 'detail']


def _camp_members(recs, camp):
    return recs if camp == 'total' else [r for r in recs if r['campaign'] == camp]


def summarize(meta, design, ev, cohort=REAL):
    rows = []

    def add(test, camp, pop, q, count='', scorable='', planned='', value='', detail=''):
        rows.append(dict(series=meta['name'], family=meta['family'], series_role=meta['role'], design=design,
                         test=test, campaign=camp, population=pop, quantity=q, count=count, scorable=scorable,
                         planned=planned, value=value, detail=detail))

    for test in TEST_ORDER:
        R = [r for r in ev['recs'] if r['test'] == test]
        for camp in CAMPS:
            t = ev['thr'][(camp, test)]
            if t['status'] == 'defined':
                add(test, camp, 'controls', 'threshold', value=repr(t['value']),
                    detail=f'set by {short(t["setter"])} (peak {t["peaks"][t["setter"]][0]!r} at epoch '
                           f'{t["peaks"][t["setter"]][1]}); control peaks: ' + ', '.join(
                               f'{short(c)} {pk!r}@{pe}' for c, (pk, pe) in t['peaks'].items()))
            else:
                add(test, camp, 'controls', 'threshold', value='undefined', detail=t['reason'])
        for camp in CAMPS + ('total',):
            C = [r for r in _camp_members(R, camp) if r['role'] == 'control']
            add(test, camp, 'controls', 'control_alarms', count=sum(r['alarm_status'] == 'alarm' for r in C),
                scorable=sum(r['alarm_status'] in VALID_OTHER for r in C), planned=len(C),
                detail=', '.join(f'{r["run_short"]}@{r["first_alarm_epoch"]}' for r in C if r['alarm_status'] == 'alarm'))
        # knee populations
        pops = sorted({p for r in R if r['role'] == 'knee' for p in r['populations'].split(';') if p})
        order = ['main21', 'matched25', 'continuity27', 'slope11_any_output', 'slope11_full_window', 'knee_all']
        pops = [p for p in order if p in pops]
        if design != 'continuity' and test == 'slope11':
            add(test, 'total', 'matched eligible', 'timely', value='n/a',
                detail='no event has its full onset window after the first slope11 output (protocol 5)')
        for pop in pops:
            for camp in CAMPS + ('total',):
                M = [r for r in _camp_members(R, camp) if r['role'] == 'knee' and pop in r['populations'].split(';')]
                valid = [r for r in M if r['alarm_status'] in VALID_KNEE]
                timely = [r for r in valid if r['alarm_status'] == 'timely']
                diff = [f'{r["run_short"]}: {r["result_reason"]}' for r in M if r['alarm_status'] not in VALID_KNEE]
                add(test, camp, pop, 'timely', count=len(timely), scorable=len(valid), planned=len(M),
                    detail=('timely: ' + ', '.join(f'{r["run_short"]}@{r["first_alarm_epoch"]}' for r in timely)
                            if timely else '') + (('; no result: ' + '; '.join(diff)) if diff else ''))
                add(test, camp, pop, 'any_alarm', count=sum(r['alarm_status'] in ('early', 'timely', 'late') for r in valid),
                    scorable=len(valid), planned=len(M))
                for st in ('early', 'timely', 'late', 'no alarm', 'no result'):
                    add(test, camp, pop, f'alarm_status:{st}', count=sum(r['alarm_status'] == st for r in M),
                        planned=len(M))
                for rr in sorted({r['result_reason'] for r in M if r['alarm_status'] == 'no result'}):
                    add(test, camp, pop, f'no_result:{rr}', count=sum(r['result_reason'] == rr for r in M
                                                                      if r['alarm_status'] == 'no result'),
                        planned=len(M))
                for cv in ('full', 'partial', 'none', 'n/a (unscorable)'):
                    add(test, camp, pop, f'coverage:{cv}', count=sum(r['coverage'] == cv for r in M), planned=len(M))
                for es in ('scorable', 'unscorable'):
                    add(test, camp, pop, f'eval_status:{es}', count=sum(r['eval_status'] == es for r in M),
                        planned=len(M),
                        detail='; '.join(f'{r["run_short"]}: {r["status_reason"]}' for r in M
                                         if r['eval_status'] == es and es == 'unscorable'))
                d = sorted(r['alarm_minus_onset'] for r in valid if r['alarm_minus_onset'] is not None)
                add(test, camp, pop, 'alarm_minus_onset', count=len(d), scorable=len(valid), planned=len(M),
                    value=' '.join(f'{x:+d}' for x in d))
                rs = ratio_summary([(r['window_ratio_status'], r['window_ratio']) for r in M])
                add(test, camp, pop, 'window_ratio', count=rs['n_finite'], planned=len(M),
                    value='' if rs['median'] is None else repr(rs['median']),
                    detail=f'finite {rs["n_finite"]}; infinite {rs["n_infinite"]}; undefined {rs["n_undefined"]}; '
                           f'empty window {rs["n_empty"]}; no result {rs["n_noresult"]}; '
                           f'min {rs["min"]!r}; max {rs["max"]!r}')
        # false-alarm populations
        for pop, groups in (('heldout', cohort.held), ('mae', {'21x': cohort.mae}), ('tiny_flat', {'21x': cohort.tiny})):
            if pop != 'heldout' and design not in ('continuity', 'D2_matched', 'D1_matched'):
                continue
            for camp in CAMPS + ('total',):
                if camp == 'total':
                    names = [r for c in CAMPS for r in groups.get(c, [])]
                else:
                    names = groups.get(camp, [])
                if not names:
                    continue
                M = [r for r in R if r['run'] in names]
                if not M:
                    continue
                if all(r['alarm_status'] == 'n/a' for r in M):
                    add(test, camp, pop, 'false_alarms', value='n/a', planned=len(names), detail=M[0]['result_reason'])
                    continue
                valid = [r for r in M if r['alarm_status'] in VALID_OTHER]
                fired = [r for r in valid if r['alarm_status'] == 'alarm']
                diff = sorted({r['result_reason'] for r in M if r['alarm_status'] not in VALID_OTHER})
                add(test, camp, pop, 'false_alarms', count=len(fired), scorable=len(valid), planned=len(names),
                    detail=', '.join(f'{r["run_short"]}@{r["first_alarm_epoch"]}' for r in fired)
                    + (('; no result: ' + '; '.join(
                        f'{sum(r["result_reason"] == d for r in M)} {d}' for d in diff)) if diff else ''))
    return rows


def paired_rows(meta, recs_by_design):
    """Paired comparisons (section 3): timely counts on the intersection of knee runs with a valid
    result in every design compared. main21: the three matched designs; matched25: standing and D1."""
    rows = []
    for pop, designs in (('main21', MATCHED), ('matched25', ['standing_matched', 'D1_matched'])):
        if not all(d in recs_by_design for d in designs):
            continue
        for test in MAIN_TESTS:
            per = {d: {r['run']: r for r in recs_by_design[d] if r['test'] == test and r['role'] == 'knee'
                       and pop in r['populations'].split(';')} for d in designs}
            base = set(per[designs[0]])
            for d in designs[1:]:
                assert set(per[d]) == base, (meta['name'], pop, test, d)
            inter = sorted(r for r in base if all(per[d][r]['alarm_status'] in VALID_KNEE for d in designs))
            for d in designs:
                for camp in CAMPS + ('total',):
                    mem = [r for r in inter if camp == 'total' or per[d][r]['campaign'] == camp]
                    planned = [r for r in base if camp == 'total' or per[d][r]['campaign'] == camp]
                    rows.append(dict(series=meta['name'], family=meta['family'], series_role=meta['role'], design=d,
                                     test=test, campaign=camp, population=pop, quantity='timely_paired',
                                     count=sum(per[d][r]['alarm_status'] == 'timely' for r in mem),
                                     scorable=len(mem), planned=len(planned), value='',
                                     detail='paired over ' + ', '.join(designs)))
    return rows


# ---------------------------------------------------------------- series registry and loaders
def series_specs():
    from knee_labelfree_metrics import VARIANTS
    specs = []
    for m, v, *_ in VARIANTS:
        specs.append(dict(name=f'{m} {v}', family='label-free', role='', designs=ALL4,
                          source=f'csv/knee_labelfree_metrics.csv: metric={m}, variant={v}, column value '
                                 f'(knee_labelfree_audit.load_metrics)', g1='labelfree'))
    specs.append(dict(name='loss', family='loss', role='', designs=ALL4,
                      source='knee_labelfree_selection.loss_series (knee_loss_census.series; checkpoint epoch N '
                             '<- logged epoch N - 1) on the csv/knee_labelfree_readability.csv grid', g1='independent'))
    for g, p in (('redrawn', 'knee_lidar.csv'), ('shared', 'knee_lidar_sharedshape.csv')):
        specs.append(dict(name=f'pooled lidar {g} 1e-4', family='pooled LiDAR', role='reference', designs=ALL4,
                          source=f'csv/{p}: n_spots = 100, column lidar (float32 series; '
                                 f'knee_labelfree_audit.load_lidar)', g1='labelfree'))
    for g in ('redrawn', 'shared'):
        for d in ('1e-6', '1e-8'):
            specs.append(dict(name=f'pooled lidar {g} {d}', family='pooled LiDAR', role='sensitivity',
                              designs=['standing_matched', 'D1_matched'],
                              source=f'csv/knee_lidar_delta_sweep.csv: grouping={g}, delta_label={d}, column lidar '
                                     f'(float64)', g1=''))
    for g in ('redrawn', 'shared'):
        for b in ('batch', 'eval'):
            specs.append(dict(name=f'projector lidar {g} {b}', family='projector LiDAR',
                              role='primary' if b == 'batch' else 'sensitivity', designs=ALL4,
                              source=f'csv/knee_proj_lidar.csv: grouping={g}, bn_mode={b}, column lidar '
                                     f'(knee_proj_audit.load_proj)', g1='projector'))
    for v in ('pub_uncentred', 'pub_centred', 'sq_centred'):
        for b in ('batch', 'eval'):
            specs.append(dict(name=f'projector rankme {v} {b}', family='projector RankMe',
                              role='primary' if b == 'batch' else 'sensitivity', designs=ALL4,
                              source=f'csv/knee_proj_rankme.csv: variant={v}, bn_mode={b}, column value '
                                     f'(knee_proj_audit.load_proj)', g1='projector'))
    return specs


def read_sweep(groupings=('redrawn', 'shared'), labels=('1e-6', '1e-8')):
    D = defaultdict(lambda: defaultdict(dict))
    with open(SWEEP_CSV) as fh:
        for r in csv.DictReader(fh):
            if r['grouping'] in groupings and r['delta_label'] in labels:
                k = (r['grouping'], r['delta_label'])
                run = r['run'].split('xxx')[0]
                e = int(r['epoch'])
                assert e not in D[k][run], ('duplicate', k, run, e)
                D[k][run][e] = float(r['lidar'])
    return {k: {run: (sorted(d), [d[e] for e in sorted(d)]) for run, d in v.items()} for k, v in D.items()}


def loss_grid():
    rd = LFA.load_readability()
    return {r: sorted(e for (rr, e) in rd if rr == r) for r in SEL.ALL_RUNS}


def load_all_series():
    out = {}
    M = LFA.load_metrics()
    for s in series_specs():
        if s['family'] == 'label-free':
            m, v = s['name'].split(' ')
            out[s['name']] = M[(m, v)]
    out['loss'] = SEL.loss_series(loss_grid())
    lid0 = AUD.LID
    out['pooled lidar redrawn 1e-4'] = LFA.load_lidar(LFA.LIDAR['lidar redrawn'])
    out['pooled lidar shared 1e-4'] = LFA.load_lidar(LFA.LIDAR['lidar shared'])
    AUD.LID = lid0
    sw = read_sweep()
    for g in ('redrawn', 'shared'):
        for d in ('1e-6', '1e-8'):
            out[f'pooled lidar {g} {d}'] = sw[(g, d)]
    PL, PR = PA.load_proj()
    for g in ('redrawn', 'shared'):
        for b in ('batch', 'eval'):
            out[f'projector lidar {g} {b}'] = PL[(g, b)]
    for v in ('pub_uncentred', 'pub_centred', 'sq_centred'):
        for b in ('batch', 'eval'):
            out[f'projector rankme {v} {b}'] = PR[(v, b)]
    assert set(out) == {s['name'] for s in series_specs()}
    return out


# ---------------------------------------------------------------- checker
class T:
    n_pass = 0
    n_fail = 0
    fails = []

    @classmethod
    def check(cls, name, got, expect, tol=None, quiet=False):
        if tol is None:
            ok = got == expect
        else:
            ok = (got is not None and expect is not None and
                  (got == expect or abs(got - expect) <= tol * max(1.0, abs(expect))))
        if ok:
            cls.n_pass += 1
        else:
            cls.n_fail += 1
            cls.fails.append(name)
        if not ok or not quiet:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + ('' if ok and quiet else f': got {got!r} expect {expect!r}')
                  + (f' (tol {tol})' if tol is not None else ''), flush=True)
        return ok


# ================================================================ G2: deterministic fixtures
def grid(H):
    return list(range(2, 21, 2)) + list(range(30, H + 1, STEP))


def fx_series(fn, H):
    eps = grid(H)
    return (eps, [float(fn(e)) for e in eps])


def fixture_cohort(knee=None, ctrl21=None, ctrl4=None, held21=None, held4=None, camp=None):
    return Cohort(knee=knee or {}, ctrl={'21x': ctrl21 or [], '4x': ctrl4 or []},
                  held={'21x': held21 or [], '4x': held4 or []}, mae=[], tiny=[],
                  campaign=camp or (lambda r: '4x' if r.startswith('F4') else '21x'))


def test_g2():
    print('=== G2.1 median: midpoint on even counts (knee_s2_monitors.median is upper-middle; not used in D1) ===')
    T.check('mid_median([4, 1, 3, 2]) = (2 + 3) / 2', mid_median([4, 1, 3, 2]), 2.5)
    T.check('mid_median([1, 3, 2]) = 2', mid_median([1, 3, 2]), 2)
    T.check('mid_median([2, 8]) = 5 (two controls: their mean)', mid_median([2, 8]), 5.0)
    T.check('mid_median([7]) = 7', mid_median([7]), 7)
    T.check('knee_s2_monitors.median([1, 2, 3, 4]) = 3 (upper middle; differs)', S2.median([1, 2, 3, 4]), 3)
    T.check('mid_median([0.5, inf]) = inf (infinity sorted last)', mid_median([0.5, math.inf]), math.inf)
    T.check('mid_median([0.5, 1.0, inf]) = 1.0', mid_median([0.5, 1.0, math.inf]), 1.0)

    print('=== G2.2 Page-Hinkley: a zero input gives n/2 after n steps ===')
    ok = all(S2.page_hinkley([(i, 0.0) for i in range(n)])[-1][1] == n / 2 for n in range(1, 41))
    T.check('page_hinkley on n zeros, n = 1..40: last statistic = n/2', ok, True)
    T.check('page_hinkley on 13 zeros (21x matched, epochs 30-150): 6.5', S2.page_hinkley([(e, 0.0) for e in range(30, 151, 10)])[-1][1], 6.5)

    print('=== G2.3 slope11 first output: 140 (inputs from 30), 180 (inputs from 70) ===')
    S = {'F4a': fx_series(lambda e: math.sin(e / 17.0) + 0.01 * e, 300),
         'F21a': fx_series(lambda e: math.cos(e / 13.0), 300)}
    for design, fo in (('continuity', 140), ('standing_matched', 140), ('D2_matched', 180)):
        inp, _ = standing_inputs(S['F4a'], 'slope11', 300, FIRST_INPUT[design])
        p = run_test('slope11', inp)
        T.check(f'{design}: slope11 first output epoch (4x, horizon 300)', p[0][0], fo)
        T.check(f'{design}: first_output_rule', first_output_rule(design, 'slope11'), fo)
    inp, _ = standing_inputs(S['F21a'], 'slope11', 150, 70)
    T.check('D2_matched at horizon 150: slope11 path empty (inputs 70-150 = 9 < 12)', run_test('slope11', inp), [])
    z = [(e, 0.1 * e) for e in range(30, 151, 10)]
    T.check('D1: slope11 on L from 30 first output 140', run_test('slope11', cumulate(z))[0][0], 140)
    for t in MAIN_TESTS:
        T.check(f'{t}: first output = first input (30 standing/D1, 70 D2)',
                (first_output_rule('standing_matched', t), first_output_rule('D2_matched', t),
                 first_output_rule('D1_matched', t)), (30, 70, 30))

    print('=== G2.4 D2 restart: the test functions receive no input before epoch 70; state starts fresh ===')
    # levels x(20)=0, x(30)=1, x(40)=2, x(50)=1, x(60)=0, then 0: calibration increments 1, 1, -1, -1 ->
    # mean 0, sd 1 (divisor n), so z(e) = increment. Standing CUSUM: 30 .5, 40 1.0, 50 .5, 60 1.0 (sn),
    # 70: sn = max(0, 1.0 - 0 - .5) = .5. D2 CUSUM at 70: fresh state, z = 0 -> 0.
    # Standing EWMA: .2, .36, .088, -.1296, 70: .8 x -.1296 = -.10368 -> .10368. D2 EWMA at 70: 0.
    lv = {30: 1.0, 40: 2.0, 50: 1.0, 60: 0.0}
    S4 = {'F21r': fx_series(lambda e: lv.get(e, 0.0), 300)}
    got = {}
    for design in ('standing_matched', 'D2_matched'):
        for t in ('CUSUM', 'EWMA'):
            inp, k = standing_inputs(S4['F21r'], t, 150, FIRST_INPUT[design])
            got[(design, t)] = dict(run_test(t, inp))[70]
    T.check('stdz constants of the fixture: mean 0, sd 1', (stdz_constants(*truncate(S4['F21r'], 150), True)['mu'],
                                                          stdz_constants(*truncate(S4['F21r'], 150), True)['sd']), (0.0, 1.0))
    T.check('standing CUSUM at 70 = 0.5 (state carried from 30-60)', got[('standing_matched', 'CUSUM')], 0.5)
    T.check('D2 CUSUM at 70 = 0.0 (fresh state)', got[('D2_matched', 'CUSUM')], 0.0)
    T.check('standing EWMA at 70 = 0.10368', got[('standing_matched', 'EWMA')], 0.10368, tol=1e-12)
    T.check('D2 EWMA at 70 = 0.0', got[('D2_matched', 'EWMA')], 0.0)
    seen = defaultdict(list)
    orig = dict(TEST_FNS)
    try:
        for t in TEST_ORDER:
            def rec(z, _t=t):
                seen[_t].extend(e for e, _ in z)
                return orig[_t](z)
            TEST_FNS[t] = rec
        co = fixture_cohort(knee={'F21k': 100}, ctrl21=['F21c1', 'F21c2', 'F21c3', 'F21c4'])
        SS = {n: fx_series(lambda e, i=i: math.sin(e / (7.0 + i)) + 0.02 * i * e, 300)
              for i, n in enumerate(['F21k', 'F21c1', 'F21c2', 'F21c3', 'F21c4'])}
        seen.clear()
        evaluate(SS, 'D2_matched', co, keep_paths=False)
        T.check('D2: the earliest input passed to each of the four test functions is epoch 70',
                {t: min(v) for t, v in seen.items() if v}, {t: 70 for t in TEST_ORDER})
        T.check('D2 at 21x horizon 150: slope11 received inputs from 70 (9 of them per run), produced no output',
                sorted(set(seen['slope11'])), list(range(70, 151, 10)))
        seen.clear()
        evaluate(SS, 'standing_matched', co, keep_paths=False)
        T.check('standing_matched: inputs start at 30 (contrast)', {t: min(v) for t, v in seen.items()},
                {t: 30 for t in TEST_ORDER})
    finally:
        TEST_FNS.clear()
        TEST_FNS.update(orig)
    ev = evaluate(SS, 'D2_matched', co)
    T.check('D2 21x slope11 threshold undefined (no control produced output)',
            (ev['thr'][('21x', 'slope11')]['status'], ev['thr'][('21x', 'slope11')]['reason']),
            ('undefined', 'no control produced output'))
    rk = [r for r in ev['recs'] if r['run'] == 'F21k' and r['test'] == 'slope11'][0]
    T.check('D2 21x slope11 knee run: no result (empty path; undefined threshold), not "no alarm"',
            (rk['alarm_status'], rk['result_reason'], rk['path_status']), ('no result', 'empty path; undefined threshold', 'empty'))

    print('=== G2.5 nested exclusion: altering control j leaves F(C minus j) unchanged; j is rescored ===')
    # increments at epochs 30, 40 (x(20) = 0): A (1, 2), B (3, 4), C (5, 8), D (7, 0).
    inc = {'A': (1.0, 2.0), 'B': (3.0, 4.0), 'C': (5.0, 8.0), 'D': (7.0, 0.0)}

    def lv_from(i30, i40):
        return lambda e: 0.0 if e <= 20 else (i30 if e == 30 else i30 + i40)
    SN = {n: fx_series(lv_from(*v), 40) for n, v in inc.items()}
    dc = d1_campaign(SN, ['A', 'B', 'C', 'D'], 40)
    Fa, FA_, FB_ = dc['fits']['all'], dc['fits']['A'], dc['fits']['B']
    # F(C): r(30) = median(1,3,5,7) = 4; r(40) = median(0,2,4,8) = 3. Inner references (median of 3):
    # minus A (5, 4), minus B (5, 2), minus C (3, 2), minus D (3, 4). Residuals A (-4,-2) B (-2,2) C (2,6)
    # D (4,-4); median 0; |res| sorted 2 2 2 2 4 4 4 6 -> MAD 3; s = 1.4826 x 3.
    T.check('F(C): r = {30: 4, 40: 3}', Fa['r'], {30: 4.0, 40: 3.0})
    T.check('F(C): residual median 0, MAD 3, s = 1.4826 x 3', (Fa['res_median'], Fa['mad'], Fa['s']), (0.0, 3.0, 1.4826 * 3))
    T.check('F(C): inner references (3 controls each)', {c: Fa['inner'][c] for c in 'ABCD'},
            {'A': {30: 5.0, 40: 4.0}, 'B': {30: 5.0, 40: 2.0}, 'C': {30: 3.0, 40: 2.0}, 'D': {30: 3.0, 40: 4.0}})
    # F(C minus A): r = median(B,C,D) = (5, 4); inner (mean of 2): B<-(C,D) (6, 4), C<-(B,D) (5, 2),
    # D<-(B,C) (4, 6); residuals B (-3, 0) C (0, 6) D (3, -6); median 0; |res| 0 0 3 3 6 6 -> MAD 3.
    T.check('F(C-A): r = {30: 5, 40: 4}; inner references are 2-control means',
            (FA_['r'], {c: FA_['inner'][c] for c in 'BCD'}),
            ({30: 5.0, 40: 4.0}, {'B': {30: 6.0, 40: 4.0}, 'C': {30: 5.0, 40: 2.0}, 'D': {30: 4.0, 40: 6.0}}))
    T.check('F(C-A): residual median 0, MAD 3, s = 1.4826 x 3', (FA_['res_median'], FA_['mad'], FA_['s']),
            (0.0, 3.0, 1.4826 * 3))
    # F(C minus B): r = median(A,C,D) = (5, 2); inner A<-(C,D) (6, 4), C<-(A,D) (4, 1), D<-(A,C) (3, 5);
    # residuals A (-5,-2) C (1,7) D (4,-5); sorted -5 -5 -2 1 4 7 -> median -0.5; |res + .5| sorted
    # 1.5 1.5 4.5 4.5 4.5 7.5 -> MAD 4.5.
    T.check('F(C-B): r = {30: 5, 40: 2}, residual median -0.5, MAD 4.5, s = 1.4826 x 4.5',
            (FB_['r'], FB_['res_median'], FB_['mad'], FB_['s']), ({30: 5.0, 40: 2.0}, -0.5, 4.5, 1.4826 * 4.5))
    co = fixture_cohort(ctrl21=['A', 'B', 'C', 'D'])

    def ctrl_z(S, j):
        d = d1_campaign(S, ['A', 'B', 'C', 'D'], 40)
        F = d['fits'][j]
        i = increments(S[j], 40)
        return [(e, (i[e] - F['r'][e]) / F['s']) for e in (30, 40)], d
    zA, _ = ctrl_z(SN, 'A')
    zB, _ = ctrl_z(SN, 'B')
    T.check('control A scored against F(C-A): z = ((1-5)/s, (2-4)/s), s = 1.4826 x 3',
            zA, [(30, (1.0 - 5.0) / (1.4826 * 3)), (40, (2.0 - 4.0) / (1.4826 * 3))])
    T.check('control B scored against F(C-B): z = ((3-5)/s, (4-2)/s), s = 1.4826 x 4.5',
            zB, [(30, (3.0 - 5.0) / (1.4826 * 4.5)), (40, (4.0 - 2.0) / (1.4826 * 4.5))])
    SN2 = dict(SN)
    SN2['A'] = fx_series(lv_from(100.0, -50.0), 40)
    zA2, dc2 = ctrl_z(SN2, 'A')
    F2 = dc2['fits']['A']
    T.check('altered A: F(C-A) reference unchanged', F2['r'], FA_['r'])
    T.check('altered A: F(C-A) scale unchanged', F2['s'], FA_['s'])
    T.check('altered A: A rescored with its own altered increments: z = (95/s, -54/s)',
            zA2, [(30, (100.0 - 5.0) / (1.4826 * 3)), (40, (-50.0 - 4.0) / (1.4826 * 3))])
    T.check('altered A: F(C-B) changes legitimately (A is a member): r = {30: median(100,5,7)=7, 40: median(-50,8,0)=0}',
            dc2['fits']['B']['r'], {30: 7.0, 40: 0.0})
    # Through evaluate() (horizon 40): control traces, CUSUM (slack 0.5), threshold.
    # F(C-C) = F(A,B,D): r = (median(1,3,7), median(2,4,0)) = (3, 2); residual median -0.5, MAD 3 -> s = 1.4826 x 3;
    #   z_C = ((5-3)/s, (8-2)/s) -> CUSUM 0 at 30, (8-2)/s - 0.5 = 0.849 at 40.
    # F(C-D) = F(A,B,C): r = (3, 4); s = 1.4826 x 3; z_D = (4/s, -4/s) -> CUSUM peak 4/s - 0.5 = 0.399.
    # z_A (above) -> CUSUM peak 4/s - 0.5 = 0.399 at 30; z_B = (-2/6.67, 2/6.67) -> CUSUM 0.
    # Threshold = 1.1 x ((8-2)/(1.4826 x 3) - 0.5), set by C at epoch 40.
    # Altered A (100, -50): z_A = (95/s, -54/s) -> CUSUM 95/s - 0.5 = 20.86 at 30; the fits of B, C, D now
    #   contain A and give z of order 0.1 (scales 57.8, 58.9, ...), CUSUM 0. Threshold = 1.1 x (95/s - 0.5), set by A.
    coN = fixture_cohort(ctrl21=['A', 'B', 'C', 'D'])
    hz = {'21x': 40, '4x': 40}
    evN = evaluate(SN, 'D1_matched', coN, horizons=hz)
    evN2 = evaluate(SN2, 'D1_matched', coN, horizons=hz)
    PN = {(p['run'], p['test']): p for p in evN['paths']}
    PN2 = {(p['run'], p['test']): p for p in evN2['paths']}
    T.check('evaluate (D1): control A is scored with its z against F(C-A)', PN[('A', 'CUSUM')]['inputs'],
            [v for _, v in zA])
    T.check('evaluate (D1): control B is scored with its z against F(C-B)', PN[('B', 'CUSUM')]['inputs'],
            [v for _, v in zB])
    s3 = 1.4826 * 3
    tN = evN['thr'][('21x', 'CUSUM')]
    T.check('evaluate (D1): CUSUM threshold 1.1 x ((8-2)/s - 0.5), set by C at epoch 40',
            (tN['value'], tN['setter'], tN['peaks']['C'][1]), (1.1 * ((8.0 - 2.0) / s3 - 0.5), 'C', 40), tol=None)
    T.check('evaluate (D1), altered A: A rescored (95/s, -54/s)', PN2[('A', 'CUSUM')]['inputs'], [v for _, v in zA2])
    tN2 = evN2['thr'][('21x', 'CUSUM')]
    T.check('evaluate (D1), altered A: threshold changes to 1.1 x (95/s - 0.5), set by A at epoch 30',
            (tN2['value'], tN2['setter'], tN2['peaks']['A'][1]), (1.1 * ((100.0 - 5.0) / s3 - 0.5), 'A', 30))
    T.check('evaluate (D1): controls never alarm against their own threshold (CUSUM, EWMA, Page-Hinkley; by construction)',
            [r['alarm_status'] for ev_ in (evN, evN2) for r in ev_['recs'] if r['role'] == 'control' and r['test'] != 'slope11'],
            ['no alarm'] * 24)
    T.check('evaluate (D1) at horizon 40: slope11 has no output (2 inputs), threshold undefined',
            evN['thr'][('21x', 'slope11')]['status'], 'undefined')

    print('=== G2.6 prefix invariance (D1): changing a monitored run after epoch e leaves z and statistics up to e unchanged ===')
    names = ['F21c1', 'F21c2', 'F21c3', 'F21c4']
    base = {n: fx_series(lambda e, i=i: 0.3 * math.sin(e / (9.0 + i)) + 0.004 * e * (1 + 0.1 * i), 300)
            for i, n in enumerate(names)}
    base['F21k'] = fx_series(lambda e: 0.3 * math.sin(e / 8.0) + (0.5 if e >= 100 else 0.0), 300)
    co = fixture_cohort(knee={'F21k': 100}, ctrl21=names)
    ev0 = evaluate(base, 'D1_matched', co)
    P0 = {(p['run'], p['test']): p for p in ev0['paths']}
    for e0 in (30, 60, 90, 140):
        alt = dict(base)
        eps, xs = base['F21k']
        alt['F21k'] = (eps, [x if e <= e0 else x + 7.0 * math.cos(e) for e, x in zip(eps, xs)])
        ev1 = evaluate(alt, 'D1_matched', co)
        P1 = {(p['run'], p['test']): p for p in ev1['paths']}
        same = True
        for t in TEST_ORDER:
            a, b = P0[('F21k', t)], P1[('F21k', t)]
            ia = [v for e, v in zip(a['epochs'], a['inputs']) if e <= e0]
            ib = [v for e, v in zip(b['epochs'], b['inputs']) if e <= e0]
            sa = [v for e, v in zip(a['stat_epochs'], a['stats']) if e <= e0]
            sb = [v for e, v in zip(b['stat_epochs'], b['stats']) if e <= e0]
            same &= (ia == ib and sa == sb)
        changed_after = P0[('F21k', 'CUSUM')]['inputs'] != P1[('F21k', 'CUSUM')]['inputs']
        T.check(f'alter F21k after epoch {e0}: z, L and all four statistics up to {e0} identical; later inputs differ',
                (same, changed_after), (True, True))

    print('=== G2.7 truncation: no value after epoch 150 enters any 21x fit, nested fits included ===')
    alt = dict(base)
    for n in names + ['F21k']:
        eps, xs = base[n]
        alt[n] = (eps, [x if e <= 150 else (float('nan') if e % 20 else 1e9) for e, x in zip(eps, xs)])
    for design in ('standing_matched', 'D2_matched', 'D1_matched'):
        a = evaluate(base, design, co)
        b = evaluate(alt, design, co)
        T.check(f'{design}: thresholds identical when every value after 150 is replaced (1e9 / NaN)',
                {k: v['value'] for k, v in a['thr'].items() if k[0] == '21x'},
                {k: v['value'] for k, v in b['thr'].items() if k[0] == '21x'})
        T.check(f'{design}: every run scorable with values after 150 non-finite',
                all(r['eval_status'] == 'scorable' for r in b['recs']), True)
        if design == 'D1_matched':
            same = all(a['d1']['21x']['fits'][k]['r'] == b['d1']['21x']['fits'][k]['r'] and
                       a['d1']['21x']['fits'][k]['s'] == b['d1']['21x']['fits'][k]['s'] and
                       a['d1']['21x']['fits'][k]['inner'] == b['d1']['21x']['fits'][k]['inner']
                       for k in a['d1']['21x']['fits'])
            T.check('D1: all five 21x fits (references, inner references, scales) identical', same, True)
            T.check('D1: fit epochs are 30..150', a['d1']['21x']['epochs'], list(range(30, 151, 10)))
            T.check('D1: largest epoch in any 21x path is 150', max(max(p['stat_epochs']) for p in a['paths'] if p['stat_epochs']), 150)

    print('=== G2.8 failures: zero scale, missing value, missing control value ===')
    ident = fx_series(lambda e: 0.01 * e, 300)
    SZ = {n: ident for n in names}
    SZ['F21k'] = base['F21k']
    ez = evaluate(SZ, 'D1_matched', co)
    T.check('identical controls: F(C) fails with "scale zero"', (ez['d1']['21x']['fits']['all']['ok'],
                                                                ez['d1']['21x']['fits']['all']['reason']), (False, 'scale zero'))
    rk = [r for r in ez['recs'] if r['run'] == 'F21k' and r['test'] == 'CUSUM'][0]
    T.check('zero scale: monitored run unscorable with the reason, "no result" (not "no alarm")',
            (rk['eval_status'], rk['status_reason'], rk['alarm_status'], rk['result_reason']),
            ('unscorable', 'F(C) failed: scale zero', 'no result', 'unscorable'))
    T.check('zero scale: D1 threshold undefined for every test', {t: ez['thr'][('21x', t)]['status'] for t in TEST_ORDER},
            {t: 'undefined' for t in TEST_ORDER})
    for kind, val in (('missing', None), ('non-finite', float('nan'))):
        SM = dict(base)
        eps, xs = base['F21k']
        if val is None:
            SM['F21k'] = ([e for e in eps if e != 90], [x for e, x in zip(eps, xs) if e != 90])
        else:
            SM['F21k'] = (eps, [val if e == 90 else x for e, x in zip(eps, xs)])
        for design in ('standing_matched', 'D2_matched', 'D1_matched'):
            em = evaluate(SM, design, co)
            st = {(r['eval_status'], r['status_reason'], r['alarm_status']) for r in em['recs'] if r['run'] == 'F21k'}
            T.check(f'{kind} value at 90, {design}: monitored run unscorable, "no result", every test',
                    st, {('unscorable', f'{kind} value at epoch 90', 'no result')})
    SC = dict(base)
    eps, xs = base['F21c3']
    SC['F21c3'] = ([e for e in eps if e != 100], [x for e, x in zip(eps, xs) if e != 100])
    for design in ('standing_matched', 'D2_matched', 'D1_matched'):
        ec = evaluate(SC, design, co)
        T.check(f'missing control value (F21c3 epoch 100), {design}: threshold undefined for every test',
                {t: ec['thr'][('21x', t)]['status'] for t in TEST_ORDER}, {t: 'undefined' for t in TEST_ORDER})
        T.check(f'missing control value, {design}: reason names the control',
                all('F21c3' in ec['thr'][('21x', t)]['reason'] for t in TEST_ORDER), True)
        rk = [r for r in ec['recs'] if r['run'] == 'F21k' and r['test'] == 'CUSUM'][0]
        T.check(f'missing control value, {design}: knee run has no result (not "no alarm")',
                rk['alarm_status'], 'no result')

    print('=== G2.9 alarm rule (strict >, first alarm, no reset) and timing classes ===')
    p = [(80, 1.0), (90, 2.0), (100, 3.0), (110, 0.0), (120, 5.0)]
    T.check('first alarm with threshold 2.0 is 100 (2.0 is not > 2.0)', first_alarm(p, 2.0), 100)
    T.check('threshold 5.0: no alarm (5.0 not > 5.0)', first_alarm(p, 5.0), None)
    T.check('onset 100: alarm 90 timely, 110 timely, 80 early, 120 late, none no alarm',
            [knee_status(a, 100) for a in (90, 110, 80, 120, None)], ['timely', 'timely', 'early', 'late', 'no alarm'])

    print('=== G2.10 thresholds: 1.1 x max over the controls; the control that sets it ===')
    cp = {'c1': (True, '', [(30, 1.0), (40, 4.0)]), 'c2': (True, '', [(30, 2.0), (40, 3.0)]),
          'c3': (True, '', [(30, 0.0)]), 'c4': (True, '', [(30, 4.0), (40, 0.5)])}
    t = threshold_from(cp, ['c1', 'c2', 'c3', 'c4'])
    T.check('threshold = 1.1 x 4.0, set by c1 (first of the tied controls), peak epoch 40',
            (t['value'], t['setter'], t['peaks']['c1']), (1.1 * 4.0, 'c1', (4.0, 40)))
    t0 = threshold_from({c: (True, '', [(30, 0.0), (40, 0.0)]) for c in 'ab'}, ['a', 'b'])
    T.check('all-zero control statistics: threshold 0', t0['value'], 0.0)

    print('=== G2.11 window ratio: finite, infinite (positive / 0), undefined (0 / 0), empty window ===')
    T.check('max 1 in [90, 110], threshold 2 -> 0.5 finite', window_ratio([(90, 1.0), (100, 0.5)], 100, 2.0)[:2], (0.5, 'finite'))
    T.check('threshold 0, window max 2 -> infinite', window_ratio([(100, 2.0)], 100, 0.0)[:2], (math.inf, 'infinite'))
    T.check('threshold 0, window max 0 -> undefined', window_ratio([(100, 0.0)], 100, 0.0)[:2], (None, 'undefined (0/0)'))
    T.check('outputs only from 180, onset 100 -> n/a', window_ratio([(180, 3.0)], 100, 1.0)[:2], (None, 'n/a (empty window)'))
    rs = ratio_summary([('finite', 0.5), ('infinite', math.inf), ('finite', 1.0), ('undefined (0/0)', None),
                        ('n/a (empty window)', None), ('no result (unscorable)', None)])
    T.check('summary: finite 2, infinite 1, undefined 1, empty 1, no result 1; median over {0.5, 1.0, inf} = 1.0, max inf',
            (rs['n_finite'], rs['n_infinite'], rs['n_undefined'], rs['n_empty'], rs['n_noresult'], rs['median'], rs['max']),
            (2, 1, 1, 1, 1, 1.0, math.inf))

    print('=== G2.12 onset-window coverage ===')
    p70 = [(e, 0.0) for e in range(70, 301, 10)]
    T.check('outputs from 70: onset 50 none, 70 partial, 80 full', [coverage(p70, o) for o in (50, 70, 80)],
            ['none', 'partial', 'full'])
    p140 = [(e, 0.0) for e in range(140, 151, 10)]
    T.check('slope11 at horizon 150 (outputs 140, 150): onset 130 partial, 140 partial, 120 none',
            [coverage(p140, o) for o in (130, 140, 120)], ['partial', 'partial', 'none'])

    print('=== G2.13 L(e) = [x(e) - x(20) - sum_{u=30..e} r(u)] / s on a complete grid ===')
    dcb = d1_campaign(base, names, 150)
    F = dcb['fits']['all']
    d = dict(zip(*base['F21k']))
    inc_k = increments(base['F21k'], 150)
    L = cumulate([(e, (inc_k[e] - F['r'][e]) / F['s']) for e in range(30, 151, 10)])
    worst = max(abs(Lv - (d[e] - d[20] - sum(F['r'][u] for u in range(30, e + 1, 10))) / F['s'])
                for e, Lv in L)
    T.check('max |L - identity| over epochs 30-150 (float rounding only)', worst < 1e-12, True)

    print('=== G2.14 protocol eligibility, from the onset table (knee_lidar_audit.KNEE_21 / KNEE_4) ===')

    def elig(design, test):
        out = {'21x': [], '4x': []}
        for run, o in KNEE.items():
            c = A2.campaign(run)
            H = (300 if run not in ('rigb_lejp_p1s100_nano_1', 'rigb_lejp_p1s100_nano_2', 'rigb_lejp_p1s100_nano_3')
                 else 150) if design == 'continuity' else HORIZON[c]
            if eligible(design, test, o, H):
                out[c].append(run)
        return out
    for design, exp in (('continuity', (12, 15)), ('standing_matched', (10, 15)), ('D1_matched', (10, 15)),
                        ('D2_matched', (9, 12))):
        for t in MAIN_TESTS:
            e = elig(design, t)
            T.check(f'{design} {t}: eligible 21x, 4x', (len(e['21x']), len(e['4x'])), exp, quiet=True)
    T.check('matched 25: excluded by name', sorted(set(KNEE) - set(sum(elig('standing_matched', 'CUSUM').values(), []))),
            ['rigb_lejp_p1s100_long_nano_1', 'rigb_lejp_p1s100_nano_7'])
    T.check('D2 21: excluded by name', sorted(set(KNEE) - set(sum(elig('D2_matched', 'CUSUM').values(), []))),
            sorted(['rigb_lejp_p1s100_long_nano_1', 'rigb_lejp_p1s100_nano_7', 'rigb_lejp_p1s100_p384_1',
                    'rigb_lejp_p1s100_p384_5', 'rigb_lejp_p1s100_p384_6', 'rigb_lejp_p1s100_nano_6']))
    T.check('main 21 = D2 eligible set = intersection of the three matched sets', sorted(main21_set()),
            sorted(sum(elig('D2_matched', 'CUSUM').values(), [])))
    for design in MATCHED:
        e = elig(design, 'slope11')
        T.check(f'{design}: slope11 eligible events', len(e['21x']) + len(e['4x']), 0)
    anyo = [r for r, o in KNEE.items() if any_output_in_window(
        'continuity', 'slope11', o, 150 if r in ('rigb_lejp_p1s100_nano_1', 'rigb_lejp_p1s100_nano_2',
                                                 'rigb_lejp_p1s100_nano_3') else 300)]
    T.check('continuity slope11: 7 events with at least one output in the window (review v2 table)',
            sorted(short(r) for r in anyo), sorted(['mu04_nano_3', 'mu07_nano_1', 'long_nano_1', 'nano_7',
                                                    'mu09_p384_3', 'p384_2', 'p384_4']))
    full = elig('continuity', 'slope11')
    T.check('continuity slope11: 2 full windows (onsets 160, 170)', sorted(short(r) for r in full['21x'] + full['4x']),
            ['long_nano_1', 'nano_7'])


# ================================================================ G1: continuity reproduces the standing record
def continuity_counts(meta, S):
    ev = evaluate(S, 'continuity', keep_paths=False)
    return ev, summarize(meta, 'continuity', ev)


def harness_view(ev):
    """Harness continuity results in counts_from_S terms."""
    out = {'thr': {c: {t: ev['thr'][(c, t)]['value'] for t in TEST_ORDER} for c in CAMPS}}
    rec = {(r['run'], r['test']): r for r in ev['recs']}
    for camp, knee in (('21x', AUD.KNEE_21), ('4x', AUD.KNEE_4)):
        for t in TEST_ORDER:
            rs = [rec[(k, t)] for k in knee if (k, t) in rec]
            d = [r['first_alarm_epoch'] - r['onset'] for r in rs if r['first_alarm_epoch'] is not None]
            out[('knee', camp, t)] = (len(d), len(rs), d)
            out[('timely', camp, t)] = sum(r['alarm_status'] == 'timely' for r in rs)
    for name, pop in (('heldout', A2.HELD_21 + A2.HELD_4), ('mae', A2.MAE), ('tiny', A2.TINY_FLAT)):
        have = [r for r in pop if (r, 'CUSUM') in rec and rec[(r, 'CUSUM')]['eval_status'] == 'scorable']
        for t in TEST_ORDER:
            out[(name, t)] = ([r for r in have if rec[(r, t)]['first_alarm_epoch'] is not None], len(have))
        out[(name, 'any')] = ([r for r in have if any(rec[(r, t)]['first_alarm_epoch'] is not None for t in TEST_ORDER)],
                              len(have))
    out['rec'] = rec
    return out


def md_tables(path):
    """Every markdown table in a file: [(heading line, header cells, [row cells])]. A table is a run of
    consecutive lines starting with '|'; its first line is the header, separator lines are skipped."""
    out, heading, cur = [], None, None
    for line in open(path):
        line = line.rstrip('\n')
        if line.startswith('#'):
            heading, cur = line, None
            continue
        if line.startswith('|'):
            cells = [p.strip() for p in line.strip().strip('|').split('|')]
            if cur is None:
                cur = (heading, cells, [])
                out.append(cur)
            elif not set(line.replace('|', '').strip()) - set('-: '):
                continue
            else:
                cur[2].append(cells)
        else:
            cur = None
    return out


def md_table(tables, heading_prefix, header0=None, index=0):
    """{row label: [cells after the label]} of the index-th table under a heading with this prefix
    (optionally whose first header cell is header0); rows with a repeated label are kept as a list."""
    hits = [t for t in tables if t[0] and t[0].startswith(heading_prefix) and (header0 is None or t[1][0] == header0)]
    assert len(hits) > index, (heading_prefix, header0, len(hits))
    out = {}
    for cells in hits[index][2]:
        out.setdefault(cells[0], []).append(cells[1:])
    return {k: (v[0] if len(v) == 1 else v) for k, v in out.items()}


def fmt_pop(fired, n):
    return LFA.fmt_pop(fired, n)


def g1_series(meta, S, lf_rows, pj_rows):
    ev, summ = continuity_counts(meta, S)
    H = harness_view(ev)
    c = RS.counts_from_S(S)
    nm = meta['name']
    # --- 1. against the existing code, live (knee_lidar_replica_seed.counts_from_S)
    T.check(f'{nm}: thresholds, 4 tests x 2 campaigns, equal to counts_from_S (exact)', H['thr'], c['thr'], quiet=True)
    mism = [(r, t) for (r, t), rec in H['rec'].items()
            if rec['eval_status'] == 'scorable' and (rec['first_alarm_epoch'] != c['sc'][r][t][2] or rec['peak'] != c['sc'][r][t][0])]
    T.check(f'{nm}: every scored run x test: first alarm epoch and peak equal to knee_lidar_audit_sharedshape.score',
            mism, [], quiet=True)
    for camp in CAMPS:
        for t in TEST_ORDER:
            n, tot, d = c[('knee', camp, t)]
            hn, htot, hd = H[('knee', camp, t)]
            T.check(f'{nm}: knee {camp} {t}: alarms, runs, alarm - onset', (hn, htot, sorted(hd)), (n, tot, sorted(d)), quiet=True)
            T.check(f'{nm}: knee {camp} {t}: timely', H[('timely', camp, t)], sum(abs(x) <= 10 for x in d), quiet=True)
    for pop in ('heldout', 'mae', 'tiny'):
        for t in TEST_ORDER + ['any']:
            T.check(f'{nm}: {pop} {t}: fired runs and denominator', H[(pop, t)], c[(pop, t)], quiet=True)
    # the summary rows used by the report carry the same counts
    sm = {(r['test'], r['campaign'], r['population'], r['quantity']): r for r in summ}
    for camp in CAMPS:
        for t in TEST_ORDER:
            n, tot, d = c[('knee', camp, t)]
            r = sm[(t, camp, 'continuity27', 'timely')]
            T.check(f'{nm}: summary row continuity27 {camp} {t} timely / scorable / planned',
                    (r['count'], r['scorable'], r['planned']), (sum(abs(x) <= 10 for x in d), tot, tot), quiet=True)
            r = sm[(t, camp, 'continuity27', 'any_alarm')]
            T.check(f'{nm}: summary row continuity27 {camp} {t} any alarm', r['count'], n, quiet=True)
    for t in TEST_ORDER:
        fired, n = c[('heldout', t)]
        r = sm[(t, 'total', 'heldout', 'false_alarms')]
        T.check(f'{nm}: summary row held-out {t}', (r['count'], r['scorable']), (len(fired), n), quiet=True)
    # --- 2. against the report tables
    if meta['g1'] == 'labelfree':
        key = {'pooled lidar redrawn 1e-4': 'lidar redrawn', 'pooled lidar shared 1e-4': 'lidar shared'}.get(nm, nm)
        k21, n21, d21 = H[('knee', '21x', 'CUSUM')]
        k4, n4, d4 = H[('knee', '4x', 'CUSUM')]
        mine = [f'{H["thr"]["21x"]["CUSUM"]:.2f} / {H["thr"]["4x"]["CUSUM"]:.2f}', f'{k21} / {n21}', f'{k4} / {n4}',
                f'{H[("timely", "21x", "CUSUM")]} + {H[("timely", "4x", "CUSUM")]}', fmt_pop(*H[('heldout', 'CUSUM')]),
                fmt_pop(*H[('mae', 'CUSUM')]), fmt_pop(*H[('tiny', 'CUSUM')])]
        T.check(f'{nm}: label-free report 3.1 (CUSUM) row, cell for cell', mine, lf_rows['### 3.1'][key][1:], quiet=True)
        for sec, t in (('### 3.2', 'PageHinkley'), ('### 3.3', 'EWMA'), ('### 3.4', 'slope11')):
            a21, m21, _ = H[('knee', '21x', t)]
            a4, m4, _ = H[('knee', '4x', t)]
            mine = [f'{a21}/{m21}, {a4}/{m4} (near {H[("timely", "21x", t)]} + {H[("timely", "4x", t)]})',
                    f'{len(H[("heldout", t)][0])}/{H[("heldout", t)][1]}, {len(H[("mae", t)][0])}/{H[("mae", t)][1]}, '
                    f'{len(H[("tiny", t)][0])}/{H[("tiny", t)][1]}']
            T.check(f'{nm}: label-free report {sec[4:]} ({t}) row', mine, lf_rows[sec][key], quiet=True)
        mine = [fmt_pop(*H[('heldout', 'any')]), fmt_pop(*H[('mae', 'any')]), fmt_pop(*H[('tiny', 'any')])]
        T.check(f'{nm}: label-free report 3.5 (any of four) row', mine, lf_rows['### 3.5'][key], quiet=True)
    if meta['g1'] == 'projector':
        kind, what, bn = nm.split(' ')[1], nm.split(' ')[2], nm.split(' ')[3]
        lab = (f'LiDAR {what}' if kind == 'lidar' else f'RankMe {what}') + f', projector, BN {bn}'
        labf = lab + (' (primary)' if bn == 'batch' else ' (sensitivity)')
        na = 'n/a (no projector)'
        k21, n21, d21 = H[('knee', '21x', 'CUSUM')]
        k4, n4, d4 = H[('knee', '4x', 'CUSUM')]
        ti21, ti4 = H[('timely', '21x', 'CUSUM')], H[('timely', '4x', 'CUSUM')]
        mine = [f'{H["thr"]["21x"]["CUSUM"]:.2f} / {H["thr"]["4x"]["CUSUM"]:.2f}', f'{k21} / {n21}', f'{k4} / {n4}',
                f'{ti21} + {ti4}', fmt_pop(*H[('heldout', 'CUSUM')]), na, fmt_pop(*H[('tiny', 'CUSUM')])]
        T.check(f'{nm}: projector report CUSUM row', mine, pj_rows['### CUSUM'][labf], quiet=True)
        ho, hn = H[('heldout', 'CUSUM')]
        to, tn = H[('tiny', 'CUSUM')]
        T.check(f'{nm}: projector report headline row', [f'{ti21 + ti4} ({ti21} + {ti4})', f'{len(ho)} / {hn}',
                                                         f'{len(to)} / {tn}', na], pj_rows['### Headline'][labf], quiet=True)
        T.check(f'{nm}: projector report CUSUM alarm minus onset row',
                [' '.join(f'{x:+d}' for x in sorted(d21)) or '-', ' '.join(f'{x:+d}' for x in sorted(d4)) or '-'],
                pj_rows['CUSUM alarm minus onset'][labf], quiet=True)
        for sec, t in (('### PageHinkley', 'PageHinkley'), ('### EWMA', 'EWMA'), ('### slope11', 'slope11')):
            a21, m21, _ = H[('knee', '21x', t)]
            a4, m4, _ = H[('knee', '4x', t)]
            mine = [f'{a21}/{m21}, {a4}/{m4} (near {H[("timely", "21x", t)]} + {H[("timely", "4x", t)]})',
                    f'{len(H[("heldout", t)][0])}/{H[("heldout", t)][1]}, n/a, {len(H[("tiny", t)][0])}/{H[("tiny", t)][1]}']
            T.check(f'{nm}: projector report {t} row', mine, pj_rows[sec][labf], quiet=True)
        T.check(f'{nm}: projector report any-of-four row',
                [fmt_pop(*H[('heldout', 'any')]), na, fmt_pop(*H[('tiny', 'any')])], pj_rows['### Any of the four'][labf],
                quiet=True)
        ctab = pj_rows['### Which control'][lab]
        by_camp = {r[0]: r for r in ctab}
        for camp in CAMPS:
            t = ev['thr'][(camp, 'CUSUM')]
            pk, pe = t['peaks'][t['setter']]
            nxt = max(v for cc, (v, _) in t['peaks'].items() if cc != t['setter'])
            k = stdz_constants(*truncate(S[t['setter']], max(S[t['setter']][0])), True)
            mine = [camp, f'{t["value"]:.2f}', short(t['setter']), f'{pk:.1f} (ep{pe})', f'{nxt:.1f}',
                    f'{k["n_cal"]}, {k["sd_raw"]:.3f}']
            T.check(f'{nm}: projector report control-setting row {camp}', mine, by_camp.get(camp), quiet=True)
    return ev, H


def g1_shared_appendix(H):
    """REPORT_SHARED, Appendix A, N = 100 block: thresholds (4 decimals), per-run peak
    (3 decimals) and first alarm epoch, all four tests."""
    lines = open(REPORT_SHARED).read().split('\n')
    i0 = next(i for i, l in enumerate(lines) if l.startswith('##### N=100: per-run peak'))
    thr_line = lines[i0 + 1]
    m = re.findall(r'(21x|4x) CUSUM ([\d.]+) PageHinkley ([\d.]+) EWMA ([\d.]+) slope11 ([\d.]+)', thr_line)
    mine = {c: [f'{H["thr"][c][t]:.4f}' for t in ('CUSUM', 'PageHinkley', 'EWMA', 'slope11')] for c in CAMPS}
    T.check('shared 1e-4: record Appendix A thresholds (4 decimals, all tests, both campaigns)',
            mine, {c: list(v) for c, *v in m})
    n = 0
    bad = []
    for l in lines[i0 + 3:]:
        if not l.strip() or l.startswith('#') or l.startswith('```'):
            break
        name = l.split()[0]
        role = l.split()[1]
        cells = re.findall(r'([\d.]+)/\s*(@\d+|-)', l)
        run = next((r for r in list(KNEE) + AUD.CTRL_21 + AUD.CTRL_4 + A2.HELD_21 + A2.HELD_4 + A2.MAE + A2.TINY_FLAT
                    if short(r) == name), None)
        if run is None:
            continue           # tiny_2, printed but not counted in the record
        for t, (pk, al) in zip(('CUSUM', 'PageHinkley', 'EWMA', 'slope11'), cells):
            rec = H['rec'][(run, t)]
            got = (f'{rec["peak"]:.3f}', '-' if rec['first_alarm_epoch'] is None else f'@{rec["first_alarm_epoch"]}')
            n += 1
            if got != (pk, al):
                bad.append((name, t, got, (pk, al)))
    T.check(f'shared 1e-4: record Appendix A per-run peak and first alarm ({n} run x test cells)', bad, [])
    T.check('shared 1e-4: Appendix A covers 57 runs x 4 tests', n, 57 * 4)


def g1_redrawn_record(H):
    """REPORT_SHARED, section 3: 'The existing column ... reproduces the record exactly:
    N=100: CUSUM thresholds 26.0734 (21x) and 48.6835 (4x), knee alarms 10/12 and 10/15, alarm minus
    onset median 0 over 20 alarms, held-out CUSUM 1/11 LeJEPA and 1/10 MAE'; table: mu04_nano_1 at 220,
    MAE s5 at 250."""
    T.check('redrawn 1e-4: CUSUM thresholds 26.0734 / 48.6835',
            (f'{H["thr"]["21x"]["CUSUM"]:.4f}', f'{H["thr"]["4x"]["CUSUM"]:.4f}'), ('26.0734', '48.6835'))
    k21, n21, d21 = H[('knee', '21x', 'CUSUM')]
    k4, n4, d4 = H[('knee', '4x', 'CUSUM')]
    T.check('redrawn 1e-4: knee CUSUM alarms 10/12, 10/15', (k21, n21, k4, n4), (10, 12, 10, 15))
    T.check('redrawn 1e-4: alarm - onset median 0 over 20 alarms, range -10 to +110',
            (statistics.median(d21 + d4), len(d21 + d4), min(d21 + d4), max(d21 + d4)), (0, 20, -10, 110))
    T.check('redrawn 1e-4: held-out CUSUM mu04_nano_1 at 220; MAE mae_5 at 250',
            ([(short(r), H['rec'][(r, 'CUSUM')]['first_alarm_epoch']) for r in H[('heldout', 'CUSUM')][0]],
             [(short(r), H['rec'][(r, 'CUSUM')]['first_alarm_epoch']) for r in H[('mae', 'CUSUM')][0]]),
            ([('mu04_nano_1', 220)], [('mae_5', 250)]))
    T.check('redrawn 1e-4: CUSUM timely 18 (10 + 8), the paper-level LiDAR count',
            (H[('timely', '21x', 'CUSUM')], H[('timely', '4x', 'CUSUM')]), (10, 8))


# ---- independent calculation for the training loss (no audit code imported in this block)
def indep_loss():
    """Loss read directly from the run logs, grid from the checkpoint files, standing detector written
    from the definitions with numpy. Returns (series, grid mismatches, thresholds, alarms, peaks)."""
    import numpy as np
    root = paths.RUNS
    runs = list(KNEE) + AUD.CTRL_21 + AUD.CTRL_4 + A2.HELD_21 + A2.HELD_4 + A2.TINY_FLAT + A2.MAE
    series, grids = {}, {}
    for run in runs:
        dirs = [d for d in os.listdir(root) if d.startswith(run + 'xxx')]
        assert len(dirs) == 1, (run, dirs)
        d = os.path.join(root, dirs[0])
        is_mae = run.startswith('rigb_mae_')
        sums, cnts = {}, {}
        with open(os.path.join(d, 'mae_r0.csv' if is_mae else 'lejepa_diag_r0.csv')) as fh:
            rd = csv.reader(fh)
            hdr = next(rd)
            ie, il = hdr.index('epoch'), hdr.index('loss')
            for row in rd:
                e = int(row[ie])
                sums[e] = sums.get(e, 0.0) + float(row[il])
                cnts[e] = cnts.get(e, 0) + 1
        tag = 'mae' if is_mae else 'jepa'
        g = sorted(int(mm.group(1)) for f in os.listdir(d) for mm in [re.fullmatch(tag + r'-ep(\d+)\.pth\.tar', f)] if mm)
        grids[run] = g
        series[run] = (g, [sums[N - 1] / cnts[N - 1] for N in g])

    def standardise(eps, xs, use_inc):
        eps, xs = np.asarray(eps), np.asarray(xs, dtype=np.float64)
        if use_inc:
            v, e2 = np.diff(xs), eps[1:]
        else:
            v, e2 = xs[1:], eps[1:]
        cal = v[(e2 >= 21) & (e2 <= 60)]
        mu = cal.mean()
        sd = float(np.sqrt(((cal - mu) ** 2).mean())) or 1e-12
        keep = e2 > 20
        return e2[keep], (v[keep] - mu) / sd

    def stat(test, e, z):
        n = len(z)
        if test == 'CUSUM':
            gp = gn = 0.0
            out = np.empty(n)
            for i in range(n):
                gp = max(0.0, gp + z[i] - 0.5)
                gn = max(0.0, gn - z[i] - 0.5)
                out[i] = max(gp, gn)
            return e, out
        if test == 'EWMA':
            s, out = 0.0, np.empty(n)
            for i in range(n):
                s = 0.2 * z[i] + 0.8 * s
                out[i] = abs(s)
            return e, out
        if test == 'PageHinkley':
            m = np.cumsum(z - 0.5)
            lo = np.minimum.accumulate(np.minimum(m, 0.0))
            hi = np.maximum.accumulate(np.maximum(m, 0.0))
            return e, np.maximum(m - lo, hi - m)
        if test == 'slope11':
            k = np.arange(11) - 5.0
            out = [abs(float(np.dot(k, z[i - 11:i] - z[i - 11:i].mean())) / 110.0) for i in range(11, n)]
            return e[11:], np.asarray(out)
    inc = {'CUSUM': True, 'EWMA': True, 'PageHinkley': False, 'slope11': False}
    traces = {}
    for run in runs:
        for t in TEST_ORDER:
            e, z = standardise(*series[run], inc[t])
            traces[(run, t)] = stat(t, e, z)
    thr = {}
    for camp, ctrl in (('21x', AUD.CTRL_21), ('4x', AUD.CTRL_4)):
        for t in TEST_ORDER:
            thr[(camp, t)] = 1.1 * max(float(traces[(c, t)][1].max()) for c in ctrl)
    alarms, peaks = {}, {}
    for run in runs:
        camp = '4x' if '_p384_' in run else '21x'
        for t in TEST_ORDER:
            e, s = traces[(run, t)]
            hit = np.nonzero(s > thr[(camp, t)])[0]
            alarms[(run, t)] = int(e[hit[0]]) if len(hit) else None
            peaks[(run, t)] = float(s.max())
    return series, grids, thr, alarms, peaks


def g1_loss(S_loss):
    meta = dict(name='loss', family='loss', role='', g1='independent')
    ev, _ = continuity_counts(meta, S_loss)
    H = harness_view(ev)
    iS, grids, ithr, ial, ipk = indep_loss()
    rg = loss_grid()
    T.check('loss (independent): checkpoint grid from the *-ep{N}.pth.tar files equals the readability grid, 57 runs',
            {r: grids[r] for r in SEL.ALL_RUNS} == {r: rg[r] for r in SEL.ALL_RUNS}, True)
    worst = max(abs(a - b) / abs(b) for r in SEL.ALL_RUNS for a, b in zip(iS[r][1], S_loss[r][1]))
    T.check(f'loss (independent): values at grid epoch N = mean of logged epoch N - 1 rows; max relative diff '
            f'{worst:.1e}', worst <= 1e-12, True)
    for camp in CAMPS:
        for t in TEST_ORDER:
            T.check(f'loss continuity threshold {camp} {t} (harness vs independent numpy)',
                    H['thr'][camp][t], ithr[(camp, t)], tol=1e-9, quiet=True)
    mism = [(short(r), t, H['rec'][(r, t)]['first_alarm_epoch'], ial[(r, t)]) for (r, t) in ial
            if H['rec'][(r, t)]['first_alarm_epoch'] != ial[(r, t)]]
    T.check('loss continuity: first alarm epoch, every run x test (57 x 4), harness vs independent', mism, [])
    pk = max(abs(H['rec'][(r, t)]['peak'] - v) / max(1.0, abs(v)) for (r, t), v in ipk.items())
    T.check(f'loss continuity: peak statistic, every run x test, max relative diff {pk:.1e}', pk <= 1e-9, True)
    for camp, knee in (('21x', AUD.KNEE_21), ('4x', AUD.KNEE_4)):
        for t in TEST_ORDER:
            d = [ial[(k, t)] - o for k, o in knee.items() if ial[(k, t)] is not None]
            hn, htot, hd = H[('knee', camp, t)]
            T.check(f'loss continuity knee {camp} {t}: alarms {len(d)}/{len(knee)}, timely '
                    f'{sum(abs(x) <= 10 for x in d)}', (hn, htot, sorted(hd), H[('timely', camp, t)]),
                    (len(d), len(knee), sorted(d), sum(abs(x) <= 10 for x in d)), quiet=True)
    for name, pop in (('heldout', A2.HELD_21 + A2.HELD_4), ('mae', A2.MAE), ('tiny', A2.TINY_FLAT)):
        for t in TEST_ORDER:
            T.check(f'loss continuity {name} {t}: fired runs', H[(name, t)][0],
                    [r for r in pop if ial[(r, t)] is not None], quiet=True)
    return ev, H


def gate():
    specs = series_specs()
    S_all = load_all_series()
    tl = md_tables(REPORT_LF)
    lf_rows = {p: md_table(tl, p) for p in ('### 3.1', '### 3.2', '### 3.3', '### 3.4', '### 3.5')}
    tp = md_tables(REPORT_PROJ)
    pj_rows = {p: md_table(tp, p) for p in ('### Headline', '### PageHinkley', '### EWMA', '### slope11',
                                            '### Any of the four', '### Which control')}
    pj_rows['### CUSUM'] = md_table(tp, '### CUSUM (', index=0)                      # the CUSUM table proper
    pj_rows['CUSUM alarm minus onset'] = md_table(tp, '### CUSUM (', index=1)        # the second table there
    assert all(len(v) == 7 for v in pj_rows['### CUSUM'].values())
    assert all(len(v) == 2 for v in pj_rows['CUSUM alarm minus onset'].values())
    print(f'parsed record tables: label-free report {sum(len(v) for v in lf_rows.values())} rows, projector report '
          f'{sum(len(v) for v in pj_rows.values())} labels')
    print('=== G1: continuity mode reproduces the standing record ===')
    n0 = T.n_pass + T.n_fail
    done = []
    for s in specs:
        if s['g1'] in ('labelfree', 'projector'):
            f0 = T.n_fail
            ev, H = g1_series(s, S_all[s['name']], lf_rows, pj_rows)
            if s['name'] == 'pooled lidar shared 1e-4':
                g1_shared_appendix(H)
            if s['name'] == 'pooled lidar redrawn 1e-4':
                g1_redrawn_record(H)
            done.append(s['name'])
            print(f"  {s['name']:34s} {'all checks PASS' if T.n_fail == f0 else 'FAILURES: ' + str(T.n_fail - f0)}",
                  flush=True)
    T.check('series covered by the counts_from_S + report comparison (21 label-free, 2 pooled LiDAR, 10 projector)',
            len(done), 33)
    print('=== G1: training loss against an independent calculation ===')
    g1_loss(S_all['loss'])
    print(f'G1 checks: {T.n_pass + T.n_fail - n0}')


# ================================================================ run
RUN_COLS = ['series', 'family', 'series_role', 'design', 'test', 'campaign', 'run', 'run_short', 'role', 'onset',
            'horizon', 'eval_status', 'status_reason', 'path_status', 'n_outputs', 'first_output_epoch',
            'last_output_epoch', 'threshold', 'threshold_status', 'threshold_reason', 'peak', 'peak_epoch',
            'first_alarm_epoch', 'alarm_minus_onset', 'alarm_status', 'result_reason', 'coverage', 'eligible',
            'populations', 'window_max', 'window_ratio', 'window_ratio_status']
FIT_COLS = ['series', 'design', 'campaign', 'fit', 'run', 'members', 'quantity', 'epoch', 'value', 'detail']
PATH_COLS = ['series', 'design', 'test', 'run', 'campaign', 'role', 'input_kind', 'input_epochs', 'inputs',
             'stat_epochs', 'stats']
MAN_COLS = ['kind', 'name', 'value', 'detail']


def fmtv(v):
    if v is None:
        return ''
    if isinstance(v, float):
        return repr(v)
    return str(v)


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def compute_all():
    specs = series_specs()
    S_all = load_all_series()
    runs, summ, fits, paths = [], [], [], []
    for s in specs:
        S = S_all[s['name']]
        by_design = {}
        for design in s['designs']:
            ev = evaluate(S, design)
            for r in ev['recs']:
                runs.append(dict(r, series=s['name'], family=s['family'], series_role=s['role']))
            by_design[design] = ev['recs']
            summ += summarize(s, design, ev)
            for f in ev['fits']:
                fits.append(dict(f, series=s['name']))
            for p in ev['paths']:
                paths.append(dict(series=s['name'], design=design, test=p['test'], run=p['run'], campaign=p['campaign'],
                                  role=p['role'], input_kind=p['input_kind'],
                                  input_epochs=' '.join(str(e) for e in p['epochs']),
                                  inputs=' '.join(repr(v) for v in p['inputs']),
                                  stat_epochs=' '.join(str(e) for e in p['stat_epochs']),
                                  stats=' '.join(repr(v) for v in p['stats'])))
        summ += paired_rows(s, by_design)
        print(f"  {s['name']:34s} {', '.join(s['designs'])}", flush=True)
    return specs, S_all, runs, summ, fits, paths


def manifest_rows(specs):
    rows = [dict(kind='protocol', name=os.path.basename(PROTOCOL), value=md5(PROTOCOL) if os.path.exists(PROTOCOL) else '',
                 detail=PROTOCOL_REVISION),
            dict(kind='run', name='date', value=time.strftime('%Y-%m-%d %H:%M:%S %Z'), detail=''),
            dict(kind='run', name='python', value=platform.python_version(), detail='')]
    inputs = [os.path.join(CSV_DIR, f) for f in ('knee_labelfree_metrics.csv', 'knee_labelfree_readability.csv',
                                                'knee_lidar.csv', 'knee_lidar_sharedshape.csv',
                                                'knee_lidar_delta_sweep.csv', 'knee_proj_lidar.csv',
                                                'knee_proj_rankme.csv')]
    for p in inputs:
        rows.append(dict(kind='input_csv', name=os.path.relpath(p, REPO), value=md5(p), detail=''))
    for run in SEL.ALL_RUNS:
        d = SEL.run_dir(run)
        p = os.path.join(d, 'mae_r0.csv' if run in SEL.MAE else 'lejepa_diag_r0.csv')
        rows.append(dict(kind='input_loss_log', name=os.path.relpath(p, paths.DATA), value=md5(p), detail=run))
    for mod in (S2, AUD, A2, RS, LFA, SEL, SEL.LC, SEL.LVR, PA, sys.modules['knee_labelfree_metrics'],
                sys.modules[__name__]):
        p = os.path.abspath(mod.__file__)
        rows.append(dict(kind='source', name=os.path.relpath(p, REPO), value=md5(p), detail=''))
    for p in (REPORT_LF, REPORT_PROJ, REPORT_SHARED):
        if not os.path.exists(p):
            continue
        rows.append(dict(kind='g1_record', name=os.path.relpath(p, REPO), value=md5(p), detail='read by --gate'))
    for camp, knee in (('21x', AUD.KNEE_21), ('4x', AUD.KNEE_4)):
        for r, o in knee.items():
            rows.append(dict(kind='onset', name=r, value=o, detail=f'knee {camp} (knee_lidar_audit.KNEE_{camp[:-1]})'))
    for lab, lst in (('controls 21x', AUD.CTRL_21), ('controls 4x', AUD.CTRL_4), ('held-out 21x', A2.HELD_21),
                     ('held-out 4x', A2.HELD_4), ('MAE', A2.MAE), ('ViT-Tiny flat', A2.TINY_FLAT)):
        rows.append(dict(kind='run_list', name=lab, value=' '.join(lst), detail=f'{len(lst)} runs'))
    rows.append(dict(kind='run_list', name='main comparison 21 events', value=' '.join(main21_set()), detail=''))
    for k, v in (('horizon 21x', 150), ('horizon 4x', 300), ('D2 first input', 70), ('D1 first increment', 30),
                 ('threshold multiplier', THR_MULT), ('timely', f'|alarm - onset| <= {TIMELY}'),
                 ('MAD factor', MAD_K), ('slope11 window', SLOPE_W), ('grid step above epoch 20', STEP)):
        rows.append(dict(kind='constant', name=k, value=v, detail=''))
    for s in specs:
        rows.append(dict(kind='series', name=s['name'], value=' '.join(s['designs']),
                         detail=f'{s["family"]}; {s["role"] or "-"}; {s["source"]}'))
    return rows


def write_csv(path, cols, rows, gz=False):
    assert not os.path.exists(path), f'refusing to overwrite {path}'
    part = path + '.partial'
    assert not os.path.exists(part), f'refusing to overwrite {part}'
    opener = (lambda p: gzip.open(p, 'wt', newline='')) if gz else (lambda p: open(p, 'w', newline=''))
    with opener(part) as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow({k: fmtv(r.get(k)) for k in cols})
    return part


def run():
    for p in OUT.values():
        assert not os.path.exists(p), f'refusing to overwrite {p}'
        assert not os.path.exists(p + '.partial'), f'refusing to overwrite {p}.partial'
    t0 = time.time()
    specs, S_all, runs, summ, fits, paths = compute_all()
    parts = {'runs': write_csv(OUT['runs'], RUN_COLS, runs),
             'summary': write_csv(OUT['summary'], SUM_COLS, summ),
             'fits': write_csv(OUT['fits'], FIT_COLS, fits),
             'paths': write_csv(OUT['paths'], PATH_COLS, paths, gz=True)}
    for k, p in parts.items():
        os.replace(p, OUT[k])
    man = manifest_rows(specs)
    for k in ('runs', 'summary', 'fits', 'paths'):
        man.append(dict(kind='output', name=os.path.relpath(OUT[k], REPO), value=md5(OUT[k]),
                        detail=f'{len({"runs": runs, "summary": summ, "fits": fits, "paths": paths}[k])} rows'))
    os.replace(write_csv(OUT['manifest'], MAN_COLS, man), OUT['manifest'])
    for k, p in OUT.items():
        print(f'wrote {p} ({os.path.getsize(p):,} bytes)')
    print(f'{len(runs)} per-run rows, {len(summ)} summary rows, {len(fits)} fit rows, {len(paths)} paths; '
          f'{time.time() - t0:.0f} s')


# ================================================================ report
def read_rows(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


DLAB = {'continuity': 'standing, continuity', 'standing_matched': 'standing, matched', 'D2_matched': 'D2, matched',
        'D1_matched': 'D1, matched'}
TLAB = {'CUSUM': 'CUSUM', 'EWMA': 'EWMA', 'PageHinkley': 'Page-Hinkley', 'slope11': 'slope11'}


def report(out=print):
    failures(out)
    out('')
    summ = read_rows(OUT['summary'])
    specs = [r for r in read_rows(OUT['manifest']) if r['kind'] == 'series']
    order = [r['name'] for r in specs]
    designs_of = {r['name']: r['value'].split() for r in specs}
    role_of = {r['name']: r['detail'].split('; ')[1] for r in specs}
    Q = {}
    for r in summ:
        Q[(r['series'], r['design'], r['test'], r['campaign'], r['population'], r['quantity'])] = r

    def g(s, d, t, c, p, q):
        return Q.get((s, d, t, c, p, q))

    def lab(s):
        rl = role_of[s]
        return s + ('' if rl == '-' else f' ({rl})')

    def rate(s, d, t, p, q='timely'):
        """'a + b = n / m (planned P)', with campaign split."""
        parts = []
        for c in CAMPS + ('total',):
            r = g(s, d, t, c, p, q)
            if r is None:
                return None
            parts.append(r)
        a, b, tot = parts
        txt = f'{a["count"]} + {b["count"]} = {tot["count"]} / {tot["scorable"]}'
        if tot['scorable'] != tot['planned']:
            txt += f' (planned {tot["planned"]})'
        return txt

    def fa(s, d, t, pop='heldout'):
        tot = g(s, d, t, 'total', pop, 'false_alarms')
        if tot is None:
            return 'not specified' if d not in designs_of[s] else 'n/a'
        if tot['value'] == 'n/a':
            return 'n/a'
        if pop == 'heldout':
            a, b = g(s, d, t, '21x', pop, 'false_alarms'), g(s, d, t, '4x', pop, 'false_alarms')
            txt = f'{a["count"]} + {b["count"]} = {tot["count"]} / {tot["scorable"]}'
        else:
            txt = f'{tot["count"]} / {tot["scorable"]}'
        if tot['scorable'] != tot['planned']:
            txt += f' (planned {tot["planned"]})'
        return txt

    out('## 1. Main comparison: 21 events, matched horizons\n')
    out('Timely first alarms (|alarm - onset| <= 10) among the 21 events, as 21x + 4x = total / scorable. '
        'The planned denominator is 21 (9 + 12) and is printed only where the scorable count differs. Held-out: '
        'event-free LeJEPA runs with any alarm within the matched horizon, 21x + 4x = total / scorable (planned 11 = '
        '7 + 4). "not specified": the protocol does not run that design on that series (delta 1e-6 and 1e-8 rows are '
        'standing and D1 only).\n')
    for t in MAIN_TESTS:
        out(f'\n### 1.{MAIN_TESTS.index(t) + 1} {TLAB[t]}\n')
        out('| series | standing: timely | standing: held-out | D2: timely | D2: held-out | D1: timely | D1: held-out |')
        out('|---|---|---|---|---|---|---|')
        for s in order:
            cells = []
            for d in MATCHED:
                if d not in designs_of[s]:
                    cells += ['not specified', 'not specified']
                    continue
                cells += [rate(s, d, t, 'main21') or '-', fa(s, d, t)]
            out(f'| {lab(s)} | ' + ' | '.join(cells) + ' |')
        # paired check
        diffs = []
        for s in order:
            for d in MATCHED:
                p = g(s, d, t, 'total', 'main21', 'timely_paired')
                o = g(s, d, t, 'total', 'main21', 'timely')
                if p and o and (p['count'], p['scorable']) != (o['count'], o['scorable']):
                    diffs.append(f'{s} {DLAB[d]}: paired {p["count"]} / {p["scorable"]}')
        out('\nPaired over the three designs (runs with a result in all three): ' +
            ('identical to the rows above for every series.' if not diffs else '; '.join(diffs)))

    out('\n## 2. The 25-event rows: standing and D1, matched horizons\n')
    out('The 21 events plus the four early onsets (p384_1 at 50, p384_5 at 60, p384_6 at 70, nano_6 at 70). '
        'Timely as 21x + 4x = total / scorable (planned 25 = 10 + 15).\n')
    out('| series | CUSUM standing | CUSUM D1 | EWMA standing | EWMA D1 | Page-Hinkley standing | Page-Hinkley D1 |')
    out('|---|---|---|---|---|---|---|')
    for s in order:
        cells = []
        for t in MAIN_TESTS:
            for d in ('standing_matched', 'D1_matched'):
                cells.append(rate(s, d, t, 'matched25') or '-')
        out(f'| {lab(s)} | ' + ' | '.join(cells) + ' |')
    out('\nThese rows differ from the section 1 rows only by the four early onsets; their per-run first alarms are in '
        'the per-run CSV, and the first-alarm status tallies over the 25 are in section 9.')

    out('\n## 3. Continuity rows: the standing detector at the original endpoints, 27 events\n')
    out('Thresholds 21x / 4x; knee runs with any alarm 21x, 4x; timely 21x + 4x (of 27); false alarms: held-out '
        '(of 11), MAE (of 10), ViT-Tiny flat (of 1). These are the standing numbers (gate G1).\n')
    for t in TEST_ORDER:
        out(f'\n### 3.{TEST_ORDER.index(t) + 1} {TLAB[t]}\n')
        extra = ' | slope11: at least one output in window: timely / events | full window: timely / events' \
            if t == 'slope11' else ''
        out('| series | threshold 21x / 4x | any alarm 21x, 4x | timely | held-out | MAE | ViT-Tiny flat' + extra + ' |')
        out('|---' * (7 + (2 if t == 'slope11' else 0)) + '|')
        for s in order:
            if 'continuity' not in designs_of[s]:
                continue
            th = [g(s, 'continuity', t, c, 'controls', 'threshold') for c in CAMPS]
            ths = ' / '.join(f'{float(x["value"]):.2f}' if x['value'] != 'undefined' else 'undefined' for x in th)
            an = ', '.join(f'{g(s, "continuity", t, c, "continuity27", "any_alarm")["count"]}/'
                           f'{g(s, "continuity", t, c, "continuity27", "any_alarm")["planned"]}' for c in CAMPS)
            ti = rate(s, 'continuity', t, 'continuity27')
            cells = [ths, an, ti, fa(s, 'continuity', t), fa(s, 'continuity', t, 'mae'), fa(s, 'continuity', t, 'tiny_flat')]
            if t == 'slope11':
                for p in ('slope11_any_output', 'slope11_full_window'):
                    r = g(s, 'continuity', t, 'total', p, 'timely')
                    cells.append(f'{r["count"]} / {r["scorable"]}' + (f' (planned {r["planned"]})'
                                                                     if r['scorable'] != r['planned'] else ''))
            out(f'| {lab(s)} | ' + ' | '.join(cells) + ' |')

    out('\n## 4. slope11, descriptive (matched designs)\n')
    out('No matched-horizon event has its full onset window after the first slope11 output (140 with inputs from 30; '
        '180 with inputs from 70), so its timely comparison is n/a. Over all 27 knee runs: onset-window coverage '
        'full / partial / none; first-alarm status early / timely / late / no alarm / no result; held-out false '
        'alarms; the threshold status.\n')
    out('| series | design | coverage full/partial/none | early/timely/late/no alarm/no result | held-out | '
        'threshold 21x / 4x |')
    out('|---|---|---|---|---|---|')
    for s in order:
        for d in MATCHED:
            if d not in designs_of[s]:
                continue
            cv = '/'.join(str(g(s, d, 'slope11', 'total', 'knee_all', f'coverage:{x}')['count'])
                          for x in ('full', 'partial', 'none'))
            st = '/'.join(str(g(s, d, 'slope11', 'total', 'knee_all', f'alarm_status:{x}')['count'])
                          for x in ('early', 'timely', 'late', 'no alarm', 'no result'))
            th = [g(s, d, 'slope11', c, 'controls', 'threshold') for c in CAMPS]
            ths = ' / '.join(f'{float(x["value"]):.3g}' if x['value'] != 'undefined' else 'undefined' for x in th)
            out(f'| {lab(s)} | {DLAB[d]} | {cv} | {st} | {fa(s, d, "slope11")} | {ths} |')

    out('\n## 5. MAE and ViT-Tiny flat: transfer checks (D2, 21x thresholds, horizon 150)\n')
    out('False alarms: runs with any alarm / scorable. MAE has no projector series (n/a). D1 is n/a for both '
        '(no same-family controls). Names with first-alarm epochs.\n')
    out('| series | test | MAE | MAE runs | ViT-Tiny flat |')
    out('|---|---|---|---|---|')
    for s in order:
        if 'D2_matched' not in designs_of[s]:
            continue
        for t in TEST_ORDER:
            m = g(s, 'D2_matched', t, 'total', 'mae', 'false_alarms')
            names = '' if m is None or m['value'] == 'n/a' else (m['detail'].replace('|', ';').lstrip('; ') or '-')
            out(f'| {lab(s)} | {TLAB[t]} | {fa(s, "D2_matched", t, "mae")} | {names} | {fa(s, "D2_matched", t, "tiny_flat")} |')

    out('\n## 6. Held-out false alarms by campaign, with names and first-alarm epochs\n')
    for d in DESIGNS:
        out(f'\n### 6.{DESIGNS.index(d) + 1} {DLAB[d]}\n')
        out('| series | test | 21x (of 7) | 4x (of 4) |')
        out('|---|---|---|---|')
        for s in order:
            if d not in designs_of[s]:
                continue
            for t in TEST_ORDER:
                cells = []
                for c in CAMPS:
                    r = g(s, d, t, c, 'heldout', 'false_alarms')
                    cells.append(f'{r["count"]} / {r["scorable"]}'
                                 + (f' (planned {r["planned"]})' if r['scorable'] != r['planned'] else '')
                                 + (f' {r["detail"].replace("|", ";").lstrip("; ")}' if r['detail'] else ''))
                out(f'| {lab(s)} | {TLAB[t]} | ' + ' | '.join(cells) + ' |')

    out('\n## 7. Thresholds and the control that sets each\n')
    out('Threshold = 1.1 x the largest statistic over the campaign\'s four mu = 0.97 controls (D1: each control '
        'scored against its nested fit). Cell: value (setting control).\n')
    for d in DESIGNS:
        out(f'\n### 7.{DESIGNS.index(d) + 1} {DLAB[d]}\n')
        out('| series | ' + ' | '.join(f'{TLAB[t]} {c}' for t in TEST_ORDER for c in CAMPS) + ' |')
        out('|---' * (1 + 2 * len(TEST_ORDER)) + '|')
        for s in order:
            if d not in designs_of[s]:
                continue
            cells = []
            for t in TEST_ORDER:
                for c in CAMPS:
                    r = g(s, d, t, c, 'controls', 'threshold')
                    if r['value'] == 'undefined':
                        cells.append('undefined: ' + r['detail'])
                    else:
                        cells.append(f'{float(r["value"]):.4g} ({r["detail"].split(" ")[2]})')
            out(f'| {lab(s)} | ' + ' | '.join(cells) + ' |')

    out('\n## 8. Onset-window ratio: max statistic in [onset - 10, onset + 10] / threshold\n')
    out('Median over the knee runs of the population (midpoint median; infinite values kept and sorted last), '
        'then min to max, then counts: finite / infinite / undefined (0/0) / empty window / no result.\n')
    for pop, designs, tests in (('main21', MATCHED, MAIN_TESTS), ('matched25', ['standing_matched', 'D1_matched'], MAIN_TESTS),
                                ('continuity27', ['continuity'], TEST_ORDER)):
        for t in tests:
            out(f'\n### {pop}, {TLAB[t]}\n')
            out('| series | ' + ' | '.join(DLAB[d] for d in designs) + ' |')
            out('|---' * (1 + len(designs)) + '|')
            for s in order:
                cells = []
                for d in designs:
                    r = g(s, d, t, 'total', pop, 'window_ratio')
                    if r is None:
                        cells.append('not specified' if d not in designs_of[s] else '-')
                        continue
                    det = dict(kv.split(' ', 1) for kv in r['detail'].replace('empty window', 'empty_window')
                               .replace('no result', 'no_result').split('; '))
                    med = '-' if r['value'] == '' else f'{float(r["value"]):.3g}'
                    mn = '-' if det['min'] == 'None' else f'{float(det["min"]):.3g}'
                    mx = '-' if det['max'] == 'None' else f'{float(det["max"]):.3g}'
                    cells.append(f'{med} ({mn} to {mx}); {det["finite"]}/{det["infinite"]}/{det["undefined"]}/'
                                 f'{det["empty_window"]}/{det["no_result"]}')
                if any(c not in ('not specified', '-') for c in cells):
                    out(f'| {lab(s)} | ' + ' | '.join(cells) + ' |')

    out('\n## 9. Per-run fields, tallied\n')
    unsc = sum(int(r['count']) for r in summ if r['population'] in ('main21', 'matched25', 'continuity27')
               and r['campaign'] == 'total' and r['quantity'] == 'eval_status:unscorable')
    full = sum(int(r['count']) for r in summ if r['population'] in ('main21', 'matched25', 'continuity27')
               and r['campaign'] == 'total' and r['quantity'] == 'coverage:full' and r['test'] != 'slope11')
    plan = sum(int(r['planned']) for r in summ if r['population'] in ('main21', 'matched25', 'continuity27')
               and r['campaign'] == 'total' and r['quantity'] == 'coverage:full' and r['test'] != 'slope11')
    out(f'Evaluation status: {unsc} unscorable run rows in the 21, 25 and 27-event populations, over every series, '
        f'design and test. Onset-window coverage for CUSUM, EWMA and Page-Hinkley in those populations: {full} of '
        f'{plan} run rows full. Cells below: first-alarm status early / timely / late / no alarm, then "/ r" with the '
        f'no-result count when it is not zero.\n')

    def st(s, d, t, pop):
        r = [g(s, d, t, 'total', pop, f'alarm_status:{x}') for x in ('early', 'timely', 'late', 'no alarm', 'no result')]
        if r[0] is None:
            return 'not specified' if d not in designs_of[s] else '-'
        txt = '/'.join(x['count'] for x in r[:4])
        return txt + (f' / r{r[4]["count"]}' if r[4]['count'] != '0' else '')
    for pop, designs, tests, title in (('main21', MATCHED, MAIN_TESTS, '9.1 The 21 events'),
                                       ('matched25', ['standing_matched', 'D1_matched'], MAIN_TESTS, '9.2 The 25 events'),
                                       ('continuity27', ['continuity'], TEST_ORDER, '9.3 Continuity, 27 events')):
        out(f'\n### {title}\n')
        cols = [(d, t) for t in tests for d in designs]
        out('| series | ' + ' | '.join(f'{TLAB[t]} {DLAB[d].split(",")[0]}' for d, t in cols) + ' |')
        out('|---' * (1 + len(cols)) + '|')
        for s in order:
            if pop == 'continuity27' and 'continuity' not in designs_of[s]:
                continue
            out(f'| {lab(s)} | ' + ' | '.join(st(s, d, t, pop) for d, t in cols) + ' |')


def failures(out=print):
    """Failure-rule outcomes (protocol section 3), read from the per-run CSV and the summary CSV."""
    runs = read_rows(OUT['runs'])
    summ = read_rows(OUT['summary'])
    out('## 0. Failure-rule outcomes and n/a rows (protocol section 3), from the CSVs\n')
    uns = defaultdict(int)
    for r in runs:
        if r['eval_status'] == 'unscorable':
            uns[(r['series'], r['design'], r['run_short'], r['status_reason'])] += 1
    out(f'- Unscorable run rows (missing or non-finite input, failed fit): {sum(uns.values())}.')
    for k, v in sorted(uns.items()):
        out(f'  - {k}: {v} test rows')
    und = defaultdict(list)
    for r in summ:
        if r['quantity'] == 'threshold' and r['value'] == 'undefined':
            und[(r['design'], r['campaign'], r['test'], r['detail'])].append(r['series'])
    out(f'- Undefined thresholds: {sum(len(v) for v in und.values())} (series x design x campaign x test).')
    for (d, c, t, why), ss in sorted(und.items()):
        out(f'  - {DLAB[d]}, {c}, {TLAB[t]}: {why}; {len(ss)} series (every series run in that design).')
    fits = [r for r in csv.DictReader(open(OUT['fits'])) if r['quantity'] == 'status']
    out(f'- D1 fits (5 per series and campaign): {len(fits)} computed; {sum(r["detail"] != "ok" for r in fits)} with a '
        f'zero, non-finite or uncomputable scale.')
    nores = defaultdict(int)
    for r in runs:
        if r['alarm_status'] == 'no result':
            nores[(r['design'], r['campaign'], r['test'], r['role'], r['result_reason'])] += 1
    out('- Run rows with no alarm result, by cause (these are never counted as "no alarm"):')
    for (d, c, t, rl, why), v in sorted(nores.items()):
        out(f'  - {DLAB[d]}, {c}, {TLAB[t]}, {rl}: {v} ({why})')
    na = defaultdict(int)
    for r in runs:
        if r['alarm_status'] == 'n/a':
            na[(r['design'], r['role'], r['result_reason'])] += 1
    out('- n/a rows (model and series or design combinations that do not exist):')
    for (d, rl, why), v in sorted(na.items()):
        out(f'  - {DLAB[d]}, {rl}: {v} run x test rows ({why})')
    zero = [r for r in summ if r['quantity'] == 'threshold' and r['value'] not in ('undefined', '') and float(r['value']) == 0]
    out(f'- Zero thresholds: {len(zero)}. Onset-window ratios over all knee run rows: ' +
        ', '.join(f'{st} {sum(1 for r in runs if r["role"] == "knee" and r["window_ratio_status"] == st)}'
                  for st in ('finite', 'infinite', 'undefined (0/0)', 'n/a (empty window)')) + '.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', action='store_true')
    ap.add_argument('--gate', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--report', action='store_true')
    a = ap.parse_args()
    if a.test:
        test_g2()
        print(f'\nG2: {T.n_pass + T.n_fail} checks: {T.n_pass} PASS, {T.n_fail} FAIL')
        if T.fails:
            print('FAILED: ' + '; '.join(T.fails))
    elif a.gate:
        gate()
        print(f'\nG1: {T.n_pass + T.n_fail} checks: {T.n_pass} PASS, {T.n_fail} FAIL')
        if T.fails:
            print('FAILED: ' + '; '.join(T.fails))
    elif a.run:
        run()
    elif a.report:
        report()
    else:
        ap.print_help()


if __name__ == '__main__':
    main()
