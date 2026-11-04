"""Kill test for the redrawn-grouping LiDAR statistic at the average-R2 onsets (2026-10-10).

Re-runs knee_lidar_killtest.py (2026-09-24) after the paper moved to the vetted average-R2 onset
rule (results/knee_onsets_avgr2.csv, 2026-10-08). The kill test depends on the 27 real onsets in
three places, all through knee_lidar_audit.KNEE_21 / KNEE_4:
  dose template      median over the 27 knee runs of R2(E)(PRE) - R2(E)(onset + k),
                     PRE = last grid epoch <= onset - 20 (asserted = pre_onset_epoch);
  synthetic onsets   the campaign medians of the knee onsets: 21x 115 -> 110, 4x 90 -> 70
                     (computed here from the new onsets, asserted on the grid). Timeliness
                     (|first CUSUM alarm - synthetic onset| <= 10) and the onset-window ratio use them;
  event-free PRE     last grid epoch <= synthetic onset - 20: 21x 90 (unchanged), 4x 70 -> 50.
Not affected: the standing CUSUM thresholds (unperturbed controls, standardised on epochs 21-60),
the noise directions, the dose-response table (fixed doses at the controls' final checkpoints),
the saved replica and probe features (read only).

knee_lidar_killtest is imported unchanged; patched in this process only:
  knee_lidar_audit.KNEE_21 / KNEE_4  -> the avg-R2 onsets of the same 27 runs (same key order)
  KT.ONSET_SYN                       -> the new campaign medians
  KT.out_names                       -> results/knee_lidar_killtest_avgr2.csv, ..._dose_avgr2.csv,
                                        ..._dirs_avgr2.csv; parts in
                                        <work folder>/replica_feats/killtest_parts_avgr2/
  KT.DECOMP_CSV                      -> results/knee_lidar_spectrum_decomp_avgr2.csv (real-event lambda1)
All outputs refuse to overwrite (KT.run asserts it).

Stages (from analysis/, always -B):
  CUDA_VISIBLE_DEVICES=1 python -B knee_lidar_killtest_avgr2.py --replay-old DIR
        the wrapper with the OLD anchors (KT defaults), outputs in DIR (outside the repository),
        then compared with the existing results/knee_lidar_killtest*.csv: the pipeline gate
  python -B knee_lidar_killtest_avgr2.py --compare-replay DIR   (the comparison alone)
  python -B knee_lidar_killtest_avgr2.py --report-replay DIR    (KT.report on the replay CSVs)
  CUDA_VISIBLE_DEVICES=1 python -B knee_lidar_killtest_avgr2.py --run       new anchors
  python -B knee_lidar_killtest_avgr2.py --gate      new CSVs against the old where anchors cannot matter
  python -B knee_lidar_killtest_avgr2.py --report    KT.report on the new CSVs, and the real-event
                                                     audit (27 knee runs) at the old and new onsets
"""
import sys

sys.dont_write_bytecode = True      # leave analysis/__pycache__ untouched

import argparse  # noqa: E402
import csv  # noqa: E402
import os  # noqa: E402
from statistics import median  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
sys.path.insert(0, _HERE)
import knee_lidar_audit as AUD  # noqa: E402
import knee_lidar_killtest as KT  # noqa: E402

ONS = os.path.join(paths.CSV, 'knee_onsets_avgr2.csv')
DECOMP_NEW = os.path.join(paths.CSV, 'knee_lidar_spectrum_decomp_avgr2.csv')
OUT = os.path.join(paths.CSV, 'knee_lidar_killtest_avgr2.csv')
OUT_DOSE = os.path.join(paths.CSV, 'knee_lidar_killtest_dose_avgr2.csv')
OUT_DIRS = os.path.join(paths.CSV, 'knee_lidar_killtest_dirs_avgr2.csv')
PARTS = os.path.join(KT.RFEAT_DIR, 'killtest_parts_avgr2')
OLD = {'series': KT.OUT, 'dose': KT.OUT_DOSE, 'dirs': KT.OUT.replace('.csv', '_dirs.csv')}
OLD_KNEE_21, OLD_KNEE_4 = dict(AUD.KNEE_21), dict(AUD.KNEE_4)
OLD_ONSET_SYN = dict(KT.ONSET_SYN)


def short(run):
    return run.split('xxx')[0]


def new_onsets():
    ons = {short(r['run']): r for r in csv.DictReader(open(ONS))}
    for k in list(OLD_KNEE_21) + list(OLD_KNEE_4):
        assert ons[k]['decline'] == '1', k
    return ons


def patch_new():
    """Anchors, synthetic onsets, outputs and the decomposition CSV -> the avg-R2 versions."""
    ons = new_onsets()
    AUD.KNEE_21 = {k: int(ons[k]['onset']) for k in OLD_KNEE_21}
    AUD.KNEE_4 = {k: int(ons[k]['onset']) for k in OLD_KNEE_4}
    syn = {}
    for camp, knee in (('21x', AUD.KNEE_21), ('4x', AUD.KNEE_4)):
        m = median(knee.values())
        assert m == int(m), (camp, m)
        syn[camp] = int(m)
    # check the old constants against the same rule on the old onsets
    orig = lambda k: [v for r, v in k.items() if r != 'rigb_lejp_p1s100_nano_11']  # noqa: E731  denom 2026-10-13: the
    assert {c: median(orig(k)) for c, k in (('21x', OLD_KNEE_21), ('4x', OLD_KNEE_4))} == OLD_ONSET_SYN  # original 27
    KT.ONSET_SYN = syn
    KT.out_names = lambda onset21: (OUT, OUT_DOSE, OUT_DIRS, PARTS)
    KT.DECOMP_CSV = DECOMP_NEW
    # template PRE = pre_onset_epoch of the onset CSV, on the readability grid the template uses
    rd = KT.load_readability()
    eps_of = {}
    for (r, e) in rd:
        eps_of.setdefault(r, []).append(e)
    for k, o in {**AUD.KNEE_21, **AUD.KNEE_4}.items():
        assert max(e for e in eps_of[k] if e <= o - 20) == int(ons[k]['pre_onset_epoch']), k
    # synthetic onsets on every event-free run's grid
    for s, (full, eps) in KT.plan().items():
        assert syn[KT.campaign(s)] in eps, s
    return syn


def patch_replay(d):
    """OLD anchors (KT defaults), outputs redirected to folder d (outside the repository)."""
    os.makedirs(d, exist_ok=True)
    KT.out_names = lambda onset21: (os.path.join(d, 'killtest.csv'), os.path.join(d, 'killtest_dose.csv'),
                                    os.path.join(d, 'killtest_dirs.csv'), os.path.join(d, 'parts'))


# ------------------------------------------------------------------ comparisons

def compare_csv(path_a, path_b, key_cols, skip=(), select=None, label=''):
    """Text identity and max numeric |diff| over rows with the same key."""
    A = {tuple(r[c] for c in key_cols): r for r in csv.DictReader(open(path_a)) if not select or select(r)}
    B = {tuple(r[c] for c in key_cols): r for r in csv.DictReader(open(path_b)) if not select or select(r)}
    common = set(A) & set(B)
    cols = [c for c in next(iter(A.values())) if c not in key_cols and c not in skip]
    n_txt, worst, worst_rel, worst_col = 0, 0.0, 0.0, ''
    for k in common:
        for c in cols:
            a, b = A[k][c], B[k][c]
            if a != b:
                n_txt += 1
                try:
                    x, y = float(a), float(b)
                    if abs(x - y) > worst:
                        worst, worst_col = abs(x - y), c
                    worst_rel = max(worst_rel, abs(x - y) / max(abs(x), 1e-30))
                except ValueError:
                    worst, worst_col = float('inf'), c
    print(f'  {label}: rows {len(A)} vs {len(B)}, matched {len(common)}; cells differing as text '
          f'{n_txt} of {len(common) * len(cols)}; max |diff| {worst:.3e} ({worst_col or "-"}), '
          f'max relative {worst_rel:.3e}')
    return len(A), len(B), len(common), n_txt


SERIES_KEY = ('run', 'epoch', 'arm', 'seed', 'grouping')


def compare_replay(d):
    print(f'== replay with the OLD anchors ({d}) against the existing kill-test CSVs ==')
    compare_csv(OLD['series'], os.path.join(d, 'killtest.csv'), SERIES_KEY, label='series')
    compare_csv(OLD['dose'], os.path.join(d, 'killtest_dose.csv'),
                ('run', 'epoch', 'arm', 'seed', 'dose_target'), label='dose')
    compare_csv(OLD['dirs'], os.path.join(d, 'killtest_dirs.csv'), ('run', 'epoch'), label='dirs')


def gate():
    """New CSVs against the old ones where the anchors cannot matter."""
    print('== new-anchor kill-test CSVs against the old ones, where the anchors cannot matter ==')
    compare_csv(OLD['series'], OUT, SERIES_KEY, skip=('onset_syn', 'k_since_onset'),
                select=lambda r: r['arm'] == 'null', label='null arm (unperturbed), every checkpoint')
    # before both synthetic onsets the arms are unperturbed copies of the null arm
    rows = list(csv.DictReader(open(OUT)))
    null = {(r['run'], r['epoch'], r['grouping']): r for r in rows if r['arm'] == 'null'}
    bad = sum(any(r[c] != null[(r['run'], r['epoch'], r['grouping'])][c] for c in ('lidar', 'lambda1', 'q'))
              for r in rows if r['arm'] != 'null' and r['k_since_onset'] == '')
    npre = sum(r['arm'] != 'null' and r['k_since_onset'] == '' for r in rows)
    print(f'  pre-onset arm rows equal to the null arm: {npre - bad} of {npre}')
    compare_csv(OLD['dose'], OUT_DOSE, ('run', 'epoch', 'arm', 'seed', 'dose_target'),
                label='dose-response (independent of onsets)')
    compare_csv(OLD['dirs'], OUT_DIRS, ('run', 'epoch'), skip=('k_since_onset',),
                label='direction diagnostics at checkpoints post-onset in both')
    post = sorted({(r['run'], int(r['epoch'])) for r in rows if r['k_since_onset'] != ''})
    print(f'  post-onset checkpoints: {len(post)} (old run: 340)')


def real_event(knee_21, knee_4):
    """The audit code (KT.audit_runs) on the real knee series, as KT.gate G3 runs it."""
    import knee_lidar_audit_sharedshape as A2
    out = {}
    for g in KT.GROUPINGS:
        S = KT.read_series(KT.LID_CSV[g])
        thr = A2.thresholds(S)
        cells = []
        tot = 0
        for camp, knee in (('21x', knee_21), ('4x', knee_4)):
            res = KT.audit_runs(S, thr, knee)
            ti = sum(v['timely'] for v in res.values())
            tot += ti
            cells.append(f'{camp}: timely {ti}/{len(res)}, alarms {sum(v["first"] is not None for v in res.values())}'
                         f'/{len(res)}, onset-window ratio median {median(v["ratio"] for v in res.values()):.3f}')
        out[g] = f'timely {tot}/27 ({"; ".join(cells)})'
    return out


def report_new():
    syn = patch_new()
    print(f'# kill test at the avg-R2 onsets: synthetic onsets {syn} (old {OLD_ONSET_SYN})\n')
    KT.report(syn['21x'])
    print('\n## Real event, 27 knee runs, CUSUM audit on the CSV series (KT.audit_runs, standing thresholds)')
    for lab, (k21, k4) in (('old onsets', (OLD_KNEE_21, OLD_KNEE_4)), ('avg-R2 onsets', (AUD.KNEE_21, AUD.KNEE_4))):
        for g, v in real_event(k21, k4).items():
            print(f'  {lab:14s} {g:8s} {v}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--replay-old', metavar='DIR')
    ap.add_argument('--compare-replay', metavar='DIR')
    ap.add_argument('--report-replay', metavar='DIR')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--gate', action='store_true')
    ap.add_argument('--report', action='store_true')
    a = ap.parse_args()
    if a.replay_old:
        KT.need_gpu()
        d = os.path.abspath(a.replay_old)
        assert not d.startswith(os.path.dirname(_HERE)), 'replay goes to a scratch folder, not the repository'
        patch_replay(d)
        KT.run(OLD_ONSET_SYN['21x'])
        compare_replay(d)
    elif a.compare_replay:
        compare_replay(os.path.abspath(a.compare_replay))
    elif a.report_replay:
        patch_replay(os.path.abspath(a.report_replay))
        KT.report(OLD_ONSET_SYN['21x'])
    elif a.run:
        KT.need_gpu()
        syn = patch_new()
        print(f'avg-R2 anchors: synthetic onsets {syn} (old {OLD_ONSET_SYN}); outputs {OUT}, {OUT_DOSE}, '
              f'{OUT_DIRS}; parts {PARTS}; decomposition CSV {KT.DECOMP_CSV}', flush=True)
        KT.run(syn['21x'])
    elif a.gate:
        gate()
    elif a.report:
        report_new()
    else:
        ap.print_help()
