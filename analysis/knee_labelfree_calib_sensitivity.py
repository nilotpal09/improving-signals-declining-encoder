"""Calibration sensitivity of the CUSUM audit for the label-free metrics
(2026-09-24).
CSV-only, no GPU. New file, new CSVs; nothing existing is written.

Why: alpha-ReQ's onset step stands out (S2a rank fraction <= 0.10) in 7 of
15 4x knee runs, but CUSUM at the standing 4x threshold gives 0 alarms
within one grid step. Before any "silent" wording for the 4x campaign, the
threshold choice is varied exactly as knee_lidar_calib_sensitivity.py
(v2) varied it for LiDAR.

Calibrations per campaign (identical to knee_lidar_calib_sensitivity.py v2):
  A    standing: 1.1 x max CUSUM over the four mu=0.97 controls
  LOO  each control left out in turn
  B    the campaign's single flat mu=0 run (knee_lidar_m6.FLAT)
  H    21x only: the single 300-epoch control alone
Columns per threshold: knee runs with any CUSUM alarm; alarms within one
grid step of onset (|alarm - onset| <= 10); held-out false alarms (7 LeJEPA
at 21x, 4 at 4x, less the calibrator: v2 rule); alarms on the controls a
calibration left out; MAE (10) and ViT-Tiny flat (1) on 21x thresholds;
and the onset-window signal = max CUSUM statistic in [onset-10, onset+10]
divided by the threshold (median, max, runs above 1).

The core loop (calib_rows) is the loop of knee_lidar_calib_sensitivity.py v2,
transcribed unchanged except that the series dict and its label are
arguments. --check runs it on the two LiDAR series and must reproduce
results/knee_lidar_calib_sensitivity_v2.csv cell for cell before the
label-free rows are trusted; the standing-A row of each label-free variant
must also reproduce the CUSUM row of the label-free audit
(knee_lidar_replica_seed.counts_from_S).

Series: all 21 variants of results/knee_labelfree_metrics.csv (loaded with
knee_labelfree_audit.load_metrics); alpha_req centred and code_wls are the
two this audit is about.

Per-run table (alpha_req centred and code_wls, both campaigns): onset, S2a
local-deviation rank fraction at onset (knee_s2_monitors.local_dev /
rank_at), first CUSUM alarm under A, B and the lowest LOO threshold, and
the onset-window ratio under each.

Outputs (refuse to overwrite):
  results/knee_labelfree_calib_sensitivity.csv
  results/knee_labelfree_calib_sensitivity_perrun.csv

Usage (from analysis/):
  python -B knee_labelfree_calib_sensitivity.py --check    # LiDAR reproduction gate
  python -B knee_labelfree_calib_sensitivity.py            # writes the two CSVs
  python -B knee_labelfree_calib_sensitivity.py --report   # reads them, prints tables
"""
import sys

sys.dont_write_bytecode = True      # leave analysis/__pycache__ untouched

import argparse  # noqa: E402
import csv  # noqa: E402
import os  # noqa: E402
import statistics  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
sys.path.insert(0, _HERE)
import knee_lidar_audit as AUD  # noqa: E402
import knee_lidar_m6 as M6  # noqa: E402
import knee_s2_monitors as S2  # noqa: E402

OUT = os.path.join(paths.CSV, 'knee_labelfree_calib_sensitivity.csv')
OUT_RUN = os.path.join(paths.CSV, 'knee_labelfree_calib_sensitivity_perrun.csv')
REF_V2 = os.path.join(paths.CSV, 'knee_lidar_calib_sensitivity_v2.csv')
PROTO = {'existing': os.path.join(paths.CSV, 'knee_lidar.csv'),
         'shared': os.path.join(paths.CSV, 'knee_lidar_sharedshape.csv')}
# populations exactly as knee_lidar_calib_sensitivity.py
HELD_21 = ['rigb_lejp_p1s100_nano_5', 'rigd_lejp_p1s100_mu04_nano_1',
           'rigd_lejp_p1s100_mu04_nano_2', 'rigd_lejp_p1s100_mu07_nano_3',
           'rigd_lejp_p1s100_mu09_nano_1', 'rigd_lejp_p1s100_mu09_nano_2',
           'rigd_lejp_p1s100_mu09_nano_3']
HELD_21 += ['rigb_lejp_p1s100_nano_10', 'rigb_lejp_p1s100_nano_12']  # runs added 2026-10-13 (= A2.HELD_21)
HELD_4 = ['rigb_lejp_p1s100_p384_10', 'rigd_lejp_p1s100_mu07_p384_2',
          'rigd_lejp_p1s100_mu09_p384_1', 'rigd_lejp_p1s100_mu09_p384_2']
MAE = [f'rigb_mae_p1s100_long_nano_{s}' for s in range(1, 11)]
TINY = ['rigb_lejp_p1s100_tiny_1']
LONG_21 = 'rigd_lejp_p1s100_mu097_long_nano_1'
fn, on_inc = S2.TESTS['CUSUM']
NAMED = [('alpha_req', 'centred'), ('alpha_req', 'code_wls')]


def trace(S, run):
    return fn(S2.stdz(*S[run], on_inc))


def peak(S, run):
    return max(v for _, v in trace(S, run))


def first_alarm(S, run, thr):
    return next((e for e, v in trace(S, run) if v > thr), None)


def window_ratio(S, run, onset, thr):
    w = [v for e, v in trace(S, run) if onset - 10 <= e <= onset + 10]
    return max(w) / thr if w else None


def calibrations(S, camp, ctrl):
    ctrl = [c for c in ctrl if c in S]
    cals = {'A': ctrl}
    for c in ctrl:
        cals['LOO-' + c.split('p1s100_')[-1]] = [x for x in ctrl if x != c]
    cals['B-flat'] = [M6.FLAT[camp]]
    if camp == '21x':
        cals['H-300ep-only'] = [LONG_21]
    return ctrl, cals


def calib_rows(S, label):
    """The loop of knee_lidar_calib_sensitivity.py v2, series as argument."""
    rows = []
    for camp, knee, ctrl, held in (('21x', AUD.KNEE_21, AUD.CTRL_21, HELD_21),
                                   ('4x', AUD.KNEE_4, AUD.CTRL_4, HELD_4)):
        ctrl, cals = calibrations(S, camp, ctrl)
        for cname, cset in cals.items():
            thr = 1.1 * max(peak(S, c) for c in cset)
            al = {k: first_alarm(S, k, thr) for k in knee if k in S}
            n_al = sum(a is not None for a in al.values())
            n_near = sum(a is not None and abs(a - knee[k]) <= 10 for k, a in al.items())
            held_eval = [r for r in held if r in S and r not in cset]
            fa_held = sum(first_alarm(S, r, thr) is not None for r in held_eval)
            fa_ctrl = sum(first_alarm(S, c, thr) is not None for c in ctrl if c not in cset)
            extra = ''
            if camp == '21x':
                fa_mae = sum(first_alarm(S, r, thr) is not None for r in MAE if r in S)
                fa_tiny = sum(first_alarm(S, r, thr) is not None for r in TINY if r in S)
                extra = f'{fa_mae}/10 MAE, {fa_tiny}/1 tiny'
            ratios = []
            for k, o in knee.items():
                if k not in S:
                    continue
                r = window_ratio(S, k, o, thr)
                if r is not None:
                    ratios.append(r)
            rows.append(dict(protocol=label, campaign=camp, calibration=cname,
                             n_cal=len(cset), thr=round(thr, 3),
                             knee_alarms=f'{n_al}/{len(al)}', near_onset=f'{n_near}/{len(al)}',
                             heldout_fa=f'{fa_held}/{len(held_eval)}',
                             excluded_ctrl_fa=f'{fa_ctrl}/{len(ctrl) - len([c for c in cset if c in ctrl])}',
                             mae_tiny=extra,
                             onset_window_ratio_median=round(statistics.median(ratios), 3),
                             onset_window_ratio_max=round(max(ratios), 3),
                             frac_runs_window_above_thr=f'{sum(r > 1 for r in ratios)}/{len(ratios)}'))
    return rows


def perrun_rows(S, label):
    rows = []
    for camp, knee, ctrl in (('21x', AUD.KNEE_21, AUD.CTRL_21), ('4x', AUD.KNEE_4, AUD.CTRL_4)):
        ctrl, cals = calibrations(S, camp, ctrl)
        thr = {c: 1.1 * max(peak(S, x) for x in cset) for c, cset in cals.items()}
        loo = min((c for c in thr if c.startswith('LOO-')), key=thr.get)
        for k, o in sorted(knee.items(), key=lambda t: t[0]):
            r = S2.rank_at(S2.local_dev(*S[k]), o)
            row = dict(series=label, campaign=camp, run=k.split('p1s100_')[-1], onset=o,
                       s2a_rank=round(r, 3) if r else '')
            for tag, c in (('A', 'A'), ('B', 'B-flat'), ('LOOmin', loo)):
                a = first_alarm(S, k, thr[c])
                row[f'thr_{tag}'] = round(thr[c], 3)
                row[f'first_alarm_{tag}'] = '' if a is None else a
                row[f'alarm_minus_onset_{tag}'] = '' if a is None else a - o
                row[f'window_ratio_{tag}'] = round(window_ratio(S, k, o, thr[c]), 3)
            row['loo_left_out'] = loo[4:]
            row['peak_cusum'] = round(peak(S, k), 3)
            rows.append(row)
    return rows


def load_lidar(path):
    AUD.LID = path
    return AUD.load(100)


def check():
    """Gate 1: reproduce the v2 CSV from the LiDAR series, cell for cell.
    Gate 2: standing-A counts of every label-free variant = counts_from_S."""
    import knee_labelfree_audit as LA
    import knee_lidar_replica_seed as RS
    mine = []
    for proto, path in PROTO.items():
        mine += calib_rows(load_lidar(path), proto)
    ref = list(csv.DictReader(open(REF_V2)))
    assert len(ref) == len(mine), (len(ref), len(mine))
    bad = 0
    for a, b in zip(mine, ref):
        for col, v in b.items():
            if str(a[col]) != v:
                bad += 1
                print('  DIFF', a['protocol'], a['campaign'], a['calibration'], col, repr(a[col]), repr(v))
    print(f'gate 1 (LiDAR, v2 CSV): {len(ref)} rows x {len(ref[0])} columns, {bad} differing cells')
    M = LA.load_metrics()
    bad2 = 0
    for mv, S in sorted(M.items()):
        c = RS.counts_from_S(S)
        rows = {(r['campaign'], r['calibration']): r for r in calib_rows(S, ' '.join(mv))}
        for camp in ('21x', '4x'):
            k, n, d = c[('knee', camp, 'CUSUM')]
            near = sum(abs(x) <= 10 for x in d)
            r = rows[(camp, 'A')]
            ok = (r['knee_alarms'] == f'{k}/{n}' and r['near_onset'] == f'{near}/{n}'
                  and abs(r['thr'] - round(c['thr'][camp]['CUSUM'], 3)) < 1e-9)
            if not ok:
                bad2 += 1
                print('  DIFF', mv, camp, r['knee_alarms'], r['near_onset'], r['thr'], (k, n, near, c['thr'][camp]['CUSUM']))
        held = c[('heldout', 'CUSUM')]
        mine_held = sum(int(rows[(camp, 'A')]['heldout_fa'].split('/')[0]) for camp in ('21x', '4x'))
        if mine_held != len(held[0]):
            bad2 += 1
            print('  DIFF held-out', mv, mine_held, held)
    print(f'gate 2 (standing A vs counts_from_S): {len(M)} variants, {bad2} differences')
    return bad == 0 and bad2 == 0


def write_csv(path, rows):
    assert not os.path.exists(path), 'refusing to overwrite ' + path
    with open(path + '.partial', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    os.replace(path + '.partial', path)
    print('wrote', path, len(rows), 'rows')


def run():
    import knee_labelfree_audit as LA
    assert check(), 'gates failed; nothing written'
    M = LA.load_metrics()
    rows, prow = [], []
    for mv in sorted(M):
        rows += calib_rows(M[mv], ' '.join(mv))
    for mv in NAMED:
        prow += perrun_rows(M[mv], ' '.join(mv))
    write_csv(OUT, rows)
    write_csv(OUT_RUN, prow)
    report()


def report():
    rows = list(csv.DictReader(open(OUT)))
    print(f"\n{'series':26s} {'camp':4s} {'calibration':18s} {'thr':>8s} {'knee':>6s} {'near':>6s} "
          f"{'held':>5s} {'exctl':>5s} {'win med':>7s} {'win max':>7s} {'win>1':>6s}  mae/tiny")
    for r in rows:
        if r['calibration'].startswith('LOO-') and r['protocol'].split()[0] != 'alpha_req':
            continue
        print(f"{r['protocol']:26s} {r['campaign']:4s} {r['calibration']:18s} {float(r['thr']):8.2f} "
              f"{r['knee_alarms']:>6s} {r['near_onset']:>6s} {r['heldout_fa']:>5s} {r['excluded_ctrl_fa']:>5s} "
              f"{float(r['onset_window_ratio_median']):7.3f} {float(r['onset_window_ratio_max']):7.3f} "
              f"{r['frac_runs_window_above_thr']:>6s}  {r['mae_tiny']}")
    print('\nper run (alpha_req): S2a rank at onset; first alarm - onset and onset-window ratio under A, B, lowest LOO')
    for r in csv.DictReader(open(OUT_RUN)):
        print(f"{r['series']:20s} {r['campaign']:4s} {r['run']:14s} on {r['onset']:>4s} s2a {r['s2a_rank']:>6s} | "
              + ' | '.join(f"{t}: {r['alarm_minus_onset_' + t] or '-':>5s} {float(r['window_ratio_' + t]):6.3f}"
                           for t in ('A', 'B', 'LOOmin'))
              + f" | peak {float(r['peak_cusum']):.2f} (LOO out {r['loo_left_out']})")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--report', action='store_true')
    a = ap.parse_args()
    if a.check:
        sys.exit(0 if check() else 1)
    elif a.report:
        report()
    else:
        run()
