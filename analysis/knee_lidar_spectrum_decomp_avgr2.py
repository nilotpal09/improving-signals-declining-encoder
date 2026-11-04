"""LiDAR spectrum decomposition at the average-R2 onsets (2026-10-10).

Re-anchors knee_lidar_spectrum_decomp.py (2026-09-23) at the vetted average-R2 onset rule
(results/knee_onsets_avgr2.csv), as knee_dir_sweep_avgr2.py did for the direction
sweep. Same 27 decline runs (knee_detector_chrono.KNEE = knee_lidar_audit.KNEE_21 + KNEE_4; all 27
have decline = 1 under the new rule), same 8 controls and 11 held-out event-free runs. Inference
only (re-encodes saved checkpoints); no training.

Machinery imported unchanged from knee_lidar_spectrum_decomp (build_plan, w_hats, main, report);
only these are patched, in this process:
  anchor source   knee_lidar_audit.KNEE_21 / KNEE_4 are replaced by the avg-R2 onsets of the same
                  runs (same key order). build_plan() then gives PRE = last grid epoch <= onset - 20
                  (asserted equal to pre_onset_epoch of knee_onsets_avgr2.csv), the onset checkpoint,
                  and POST = final checkpoint (asserted equal to last_epoch). Event-free runs keep
                  build_plan's rule (grid epoch at or below the campaign's knee-cohort median PRE),
                  recomputed from the new knee PREs.
  direction CSV   results/knee_dir_sweep_avgr2.csv (w_hat at the new PRE). It lists nano_1-3, so the
                  W_KNEE fallback is no longer used for them. Event-free runs keep w_bar built from
                  knee_t2_direction.W_KNEE (unchanged code path).
  output          results/knee_lidar_spectrum_decomp_avgr2.csv (refuses to overwrite).

Stages (from analysis/; always -B):
  CUDA_VISIBLE_DEVICES=1 python -B knee_lidar_spectrum_decomp_avgr2.py          # compute
  python -B knee_lidar_spectrum_decomp_avgr2.py --gate      # unchanged checkpoints vs the old CSV
  python -B knee_lidar_spectrum_decomp_avgr2.py --report    # the old report on the new CSV
  python -B knee_lidar_spectrum_decomp_avgr2.py --paper     # paper statements, old CSV and new CSV
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
import knee_lidar_audit as AUD  # noqa: E402
import knee_lidar_spectrum_decomp as SD  # noqa: E402

ONS = os.path.join(paths.CSV, 'knee_onsets_avgr2.csv')
DIRS_NEW = os.path.join(paths.CSV, 'knee_dir_sweep_avgr2.csv')
OLD_CSV = os.path.join(paths.CSV, 'knee_lidar_spectrum_decomp.csv')
NEW_CSV = os.path.join(paths.CSV, 'knee_lidar_spectrum_decomp_avgr2.csv')
OLD_ONSET = {**AUD.KNEE_21, **AUD.KNEE_4}           # the old anchors, kept for the gate
U_COLS = ('var_w_u', 'var_b_u', 'ratio_u', 'var_w_rest', 'var_b_rest', 'ratio_u_own')
NUM_SKIP = {'run', 'role', 'campaign', 'onset', 'phase', 'epoch', 'term', 'variant', 'B_set',
            'W_set', 'u_source'}


def short(run):
    return run.split('xxx')[0]


def new_onsets():
    """short run -> knee_onsets_avgr2.csv row, for the 27 runs (all must be declines)."""
    ons = {short(r['run']): r for r in csv.DictReader(open(ONS))}
    for k in OLD_ONSET:
        assert ons[k]['decline'] == '1', k
    return ons


def patch():
    """Point knee_lidar_spectrum_decomp at the new anchors, direction CSV and output."""
    ons = new_onsets()
    AUD.KNEE_21 = {k: int(ons[k]['onset']) for k in AUD.KNEE_21}
    AUD.KNEE_4 = {k: int(ons[k]['onset']) for k in AUD.KNEE_4}
    SD.DIRS = DIRS_NEW
    SD.OUT_CSV = NEW_CSV
    plan, ref = SD.build_plan()
    n = 0
    for full, s, role, camp, onset, cks in plan:
        if role != 'knee':
            continue
        ph = dict(cks)
        assert ph['pre'] == int(ons[s]['pre_onset_epoch']), (s, ph, ons[s]['pre_onset_epoch'])
        assert ph['onset'] == int(ons[s]['onset']), s
        assert ph['post'] == int(ons[s]['last_epoch']), s
        n += 1
    assert n == 28, n  # denominator runs (2026-10-13): 27 + nano_11
    return plan, ref


def compute():
    plan, ref = patch()
    print('anchors: knee_onsets_avgr2.csv (PRE = pre_onset_epoch, POST = last_epoch, asserted); '
          f'direction CSV {os.path.basename(SD.DIRS)}; output {SD.OUT_CSV}', flush=True)
    for full, s, role, camp, onset, cks in plan:
        if role == 'knee':
            ph = dict(cks)
            print(f'  {s:36s} onset {OLD_ONSET[s]:4d} -> {onset:4d}  PRE -> {ph["pre"]:4d}', flush=True)
    SD.main()


# ------------------------------------------------------------------ gate

def gate():
    """Rows at the same (run, phase, epoch, term, variant) in the old and the new CSV must agree.
    Non-direction columns: every matched row. Direction columns (U_COLS): only where the run's PRE
    and its w_hat source are unchanged (u is the PRE read-out of w_hat)."""
    rows_o = list(csv.DictReader(open(OLD_CSV)))
    rows_n = list(csv.DictReader(open(NEW_CSV)))
    key = lambda r: (short(r['run']), r['phase'], int(r['epoch']), r['term'], r['variant'])  # noqa: E731
    O = {key(r): r for r in rows_o}
    N = {key(r): r for r in rows_n}
    pre_o = {short(r['run']): int(r['epoch']) for r in rows_o if r['phase'] == 'pre'}
    pre_n = {short(r['run']): int(r['epoch']) for r in rows_n if r['phase'] == 'pre'}
    src_o = {short(r['run']): r['u_source'] for r in rows_o}
    src_n = {short(r['run']): r['u_source'] for r in rows_n}
    print(f'old {len(rows_o)} rows, new {len(rows_n)} rows; matched (same run, phase, epoch, term, '
          f'variant) {len(set(O) & set(N))}')
    role = {short(r['run']): r['role'] for r in rows_n}
    same_pre = sorted(s for s in pre_n if pre_n[s] == pre_o[s])
    moved = sorted(s for s in pre_n if pre_n[s] != pre_o[s])
    print(f'runs with PRE unchanged: {len(same_pre)} (knee {sum(role[s] == "knee" for s in same_pre)}, '
          f'event-free {sum(role[s] != "knee" for s in same_pre)}); PRE moved: {len(moved)} '
          f'(knee {sum(role[s] == "knee" for s in moved)}, event-free {sum(role[s] != "knee" for s in moved)})')
    for s in moved:
        print(f'  moved: {s:36s} PRE {pre_o[s]:4d} -> {pre_n[s]:4d}')
    cols = [c for c in rows_n[0] if c not in NUM_SKIP]
    worst_abs, worst_rel, n_diff_str, n_cells = {}, {}, {}, {}
    u_excluded = set()
    for k in sorted(set(O) & set(N)):
        s = k[0]
        u_ok = pre_n[s] == pre_o[s] and src_n[s] == src_o[s]
        if not u_ok:
            u_excluded.add(s)
        group_tag = ('unchanged-PRE runs' if pre_n[s] == pre_o[s] else 'moved-PRE runs')
        for c in cols:
            if c in U_COLS and not u_ok:
                continue
            a, b = O[k][c], N[k][c]
            if a == '' and b == '':
                continue
            assert (a == '') == (b == ''), (k, c, a, b)
            g = (group_tag, 'direction cols' if c in U_COLS else 'other cols')
            x, y = float(a), float(b)
            worst_abs[g] = max(worst_abs.get(g, 0.0), abs(x - y))
            worst_rel[g] = max(worst_rel.get(g, 0.0), abs(x - y) / max(abs(x), 1e-30))
            n_diff_str[g] = n_diff_str.get(g, 0) + (a != b)
            n_cells[g] = n_cells.get(g, 0) + 1
    print('\n== gate: matched cells, max |new - old| (absolute, relative) and cells whose text differs ==')
    for g in sorted(worst_abs):
        print(f'  {g[0]:20s} {g[1]:15s} cells {n_cells[g]:6d}: max abs {worst_abs[g]:.3e}, max rel '
              f'{worst_rel[g]:.3e}, text differs in {n_diff_str[g]}')
    print(f'  direction columns not compared (PRE moved or w_hat source changed): {len(u_excluded)} runs')
    # the two unchanged-PRE runs whose w_hat source changed (W_KNEE 3 decimals -> recomputed sweep)
    for s in sorted(u_excluded):
        if pre_n[s] == pre_o[s]:
            d = max(abs(float(O[k][c]) - float(N[k][c])) / abs(float(O[k][c]))
                    for k in set(O) & set(N) if k[0] == s for c in ('ratio_u', 'ratio_u_own'))
            print(f'  {s}: PRE unchanged, w_hat source {src_o[s]} -> {src_n[s]}; ratio_u / ratio_u_own '
                  f'max relative change {d:.3e}')


# ------------------------------------------------------------------ paper statements

def split(a, b):
    """The symmetric split of knee_lidar_spectrum_decomp.report (nested there, so restated here and
    checked by running it on the old CSV, which must give the 2026-09-23 report's numbers)."""
    f = lambda r, k: float(r[k])                                                       # noqa: E731
    q1, q2, H1, H2 = f(a, 'q'), f(b, 'q'), f(a, 'H_rest'), f(b, 'H_rest')
    g = lambda q, H: ((-q * math.log(q) - (1 - q) * math.log(1 - q)) + (1 - q) * H)    # noqa: E731
    dq = 0.5 * ((g(q2, H1) - g(q1, H1)) + (g(q2, H2) - g(q1, H2)))
    dH = 0.5 * ((1 - q1) + (1 - q2)) * (H2 - H1)
    return math.log(f(b, 'lidar_decomp')) - math.log(f(a, 'lidar_decomp')), dq, dH


def paper_numbers(path):
    rows = list(csv.DictReader(open(path)))
    f = lambda r, k: float(r[k])                                                       # noqa: E731
    R = {(short(r['run']), r['phase'], r['term'], r['variant']): r for r in rows}
    knee = sorted({short(r['run']) for r in rows if r['role'] == 'knee'})
    assert len(knee) == (27 if path == OLD_CSV else 28)  # denominator runs (2026-10-13): new CSV has + nano_11
    N = len(knee)
    out = {}
    t = 'redrawn|redrawn'
    pre = [f(R[(s, 'pre', t, 'raw')], 'lambda1') for s in knee]
    post = [f(R[(s, 'post', t, 'raw')], 'lambda1') for s in knee]
    out['redrawn lambda1 PRE median (range)'] = f'{median(pre):.2f} ({min(pre):.2f} to {max(pre):.2f})'
    out['redrawn lambda1 POST median (range)'] = f'{median(post):.2f} ({min(post):.2f} to {max(post):.2f})'
    out['redrawn lambda1 falls PRE->POST'] = f'{sum(b < a for a, b in zip(pre, post))}/{N}'
    sp = [split(R[(s, 'pre', t, 'raw')], R[(s, 'post', t, 'raw')]) for s in knee]
    rat = [dq / dl for dl, dq, dH in sp]
    bnd = [abs(dq) / (abs(dq) + abs(dH)) for dl, dq, dH in sp]
    out['redrawn d_q / dlogL median (range)'] = f'{median(rat):.3f} ({min(rat):.3f} to {max(rat):.3f})'
    out['redrawn |d_q|/(|d_q|+|d_H|) median (range)'] = f'{median(bnd):.3f} ({min(bnd):.3f} to {max(bnd):.3f})'
    out['redrawn LiDAR up PRE->POST'] = f'{sum(dl > 0 for dl, _, _ in sp)}/{N}'
    out['redrawn median dlogL, d_q, d_H'] = (f'{median(x[0] for x in sp):+.3f}, {median(x[1] for x in sp):+.3f}, '
                                             f'{median(x[2] for x in sp):+.3f}')
    out['redrawn |d_q| > |d_H|'] = f'{sum(abs(dq) > abs(dH) for _, dq, dH in sp)}/{N}'
    kr = [r for r in rows if r['term'] == t and r['variant'] == 'raw' and r['role'] == 'knee']
    c = [f(r, 'corr_top_energy') for r in kr]
    cs = [f(r, 'cos_top_energy') for r in kr]
    out[f'redrawn |corr(top proj, energy proj)| median (min), n={len(kr)} knee ckpts'] = f'{median(c):.3f} ({min(c):.3f})'
    out[f'redrawn |cos(top read-out, energy read-out)| median (min), n={len(kr)}'] = f'{median(cs):.3f} ({min(cs):.3f})'
    ks = [r for r in rows if r['term'] == 'shared|shared' and r['variant'] == 'raw' and r['role'] == 'knee']
    out['shared |corr| median'] = f'{median(f(r, "corr_top_energy") for r in ks):.3f}'
    t = 'shared|shared'
    A = [(R[(s, 'pre', t, 'raw')], R[(s, 'post', t, 'raw')]) for s in knee]
    sps = [split(a, b) for a, b in A]
    out['shared LiDAR up PRE->POST'] = f'{sum(dl > 0 for dl, _, _ in sps)}/{N}'
    out['shared lambda1 falls PRE->POST'] = f'{sum(f(b, "lambda1") < f(a, "lambda1") for a, b in A)}/{N}'
    out['shared lambda1 POST/PRE median'] = f'{median(f(b, "lambda1") / f(a, "lambda1") for a, b in A):.3f}'
    out['shared bulk grows'] = f'{sum(f(b, "bulk") > f(a, "bulk") for a, b in A)}/{N}'
    out['shared median dlogL, d_q, d_H'] = (f'{median(x[0] for x in sps):+.3f}, {median(x[1] for x in sps):+.3f}, '
                                            f'{median(x[2] for x in sps):+.3f}')
    lp = [f(a, 'lambda1') for a, b in A]
    lq = [f(b, 'lambda1') for a, b in A]
    out['shared lambda1 PRE / POST medians'] = f'{median(lp):.0f} / {median(lq):.0f}'
    return out


def paper():
    old, new = paper_numbers(OLD_CSV), paper_numbers(NEW_CSV)
    print(f'{"statistic (27 decline runs, raw terms)":66s} {"old CSV":>26s}   {"new CSV (avg-R2 anchors)":>26s}')
    for (k, v), (k2, v2) in zip(old.items(), new.items()):  # denominator runs (2026-10-13): keys carry n, matched by order
        print(f'{k2:66s} {v:>26s}   {v2:>26s}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--gate', action='store_true')
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--paper', action='store_true')
    a = ap.parse_args()
    if a.gate:
        gate()
    elif a.report:
        patch()
        SD.report()
    elif a.paper:
        paper()
    elif 'CUDA_VISIBLE_DEVICES' not in os.environ:
        raise SystemExit('set CUDA_VISIBLE_DEVICES explicitly (shared GPUs)')
    else:
        compute()
