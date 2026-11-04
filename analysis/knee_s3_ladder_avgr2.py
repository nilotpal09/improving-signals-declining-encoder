"""Capacity ladder (knee_s3_ladder.py) at the average-ridge-R^2 pre-onset
anchors (2026-10-10). New file; nothing existing is modified.

Why: knee_s3_ladder.py timed PRE from the old mean-CCA onset rule
(knee_s1_onsets.py R_A, fallback R_B). The paper now times declines with
the rule on average ridge R^2, results/knee_onsets_avgr2.csv: PRE =
pre_onset_epoch (last grid epoch <= onset - 20), POST = last_epoch (the
final checkpoint, unchanged). For the 13 anatomy runs (21x nano_1-3,
long_nano_1; 4x p384_1-9) PRE moves in 6 runs:
  long_nano_1 140->130, nano_1 70->60, p384_2 120->110, p384_3 70->50,
  p384_7 60->40, p384_8 70->30;  the other 7 PREs and all POSTs unchanged.

What runs: knee_s3_ladder.main() itself, imported unchanged, with two
module constants replaced: RUNS (the 13 runs at the new anchors) and
CSV_PATH (the new output). All 26 checkpoints are recomputed with the
original code (pooled ridge + CCA, token_max / token_median / token_flat
ridges, query85k and mlp1M at probe seeds 1 and 2), so the 20 checkpoints
whose epoch did not move are a reproduction gate against
results/knee_s3_ladder.csv.

Modes (from analysis/, run with -B):
  CUDA_VISIBLE_DEVICES=1 python -B knee_s3_ladder_avgr2.py           # compute
  CUDA_VISIBLE_DEVICES=1 python -B knee_s3_ladder_avgr2.py --resume  # append missing rows only
  python -B knee_s3_ladder_avgr2.py --report     # gate + paper statements (reads CSVs)
  python -B knee_s3_ladder_avgr2.py --write-s7   # S7(i) per-component CCA table
                                                 # from results/knee_traj_full.csv

Outputs (new; the script refuses to overwrite either):
  results/knee_s3_ladder_avgr2.csv  same columns as knee_s3_ladder.csv
  results/knee_s7i_cca_avgr2.csv    S7(i) cca_c1..c3 at old PRE, new PRE, POST
"""

import sys

sys.dont_write_bytecode = True      # leave analysis/__pycache__ untouched

import argparse  # noqa: E402
import csv  # noqa: E402
import os  # noqa: E402
from statistics import mean  # noqa: E402

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
sys.path.insert(0, _HERE)

ONSETS_CSV = os.path.join(paths.CSV, 'knee_onsets_avgr2.csv')
OLD_CSV = os.path.join(paths.CSV, 'knee_s3_ladder.csv')
TRAJ_CSV = os.path.join(paths.CSV, 'knee_traj_full.csv')
OUT_CSV = os.path.join(paths.CSV, 'knee_s3_ladder_avgr2.csv')
S7_CSV = os.path.join(paths.CSV, 'knee_s7i_cca_avgr2.csv')

# key -> (old PRE from knee_s3_ladder.csv, new PRE, POST), the expected values;
# every value is re-checked against the two CSVs before anything runs.
EXPECTED = {
    'rigb_lejp_p1s100_long_nano_1': (140, 130, 300),
    'rigb_lejp_p1s100_nano_1': (70, 60, 150),
    'rigb_lejp_p1s100_p384_2': (120, 110, 300),
    'rigb_lejp_p1s100_p384_3': (70, 50, 300),
    'rigb_lejp_p1s100_p384_7': (60, 40, 300),
    'rigb_lejp_p1s100_p384_8': (70, 30, 300),
    'rigb_lejp_p1s100_nano_2': (90, 90, 150),
    'rigb_lejp_p1s100_nano_3': (80, 80, 150),
    'rigb_lejp_p1s100_p384_1': (30, 30, 300),
    'rigb_lejp_p1s100_p384_4': (120, 120, 300),
    'rigb_lejp_p1s100_p384_5': (40, 40, 300),
    'rigb_lejp_p1s100_p384_6': (50, 50, 300),
    'rigb_lejp_p1s100_p384_9': (100, 100, 300),
}
PROBES = ('pooled_linear', 'query85k', 'mlp1M',
          'token_max', 'token_median', 'token_flat')
SEEDED = ('query85k', 'mlp1M')


def key_of(run):
    return run.split('xxx')[0]


def short(run):
    return key_of(run).split('p1s100_')[1]


def read(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def anchors():
    """[(glob pattern, new PRE, POST)] for the 13 runs, checked against
    knee_onsets_avgr2.csv (new PRE, POST) and knee_s3_ladder.csv (old PRE,
    POST)."""
    on = {key_of(r['run']): r for r in read(ONSETS_CSV)}
    old = {}
    for r in read(OLD_CSV):
        if key_of(r['run']) in EXPECTED:
            old.setdefault((key_of(r['run']), r['phase']), set()).add(int(r['epoch']))
    out = []
    for k, (old_pre, new_pre, post) in EXPECTED.items():
        r = on[k]
        assert r['decline'] == '1', (k, r['decline'])
        assert int(r['pre_onset_epoch']) == new_pre, (k, r['pre_onset_epoch'])
        assert int(r['last_epoch']) == post, (k, r['last_epoch'])
        assert new_pre <= int(r['onset']) - 20, (k, r['onset'])
        assert old[(k, 'pre')] == {old_pre}, (k, old[(k, 'pre')])
        assert old[(k, 'post')] == {post}, (k, old[(k, 'post')])
        out.append((k + 'xxx*', new_pre, post))
    return out


# ------------------------------------------------------------------ compute

def compute(resume):
    if os.path.exists(OUT_CSV) and not resume:
        raise SystemExit(f'refusing to overwrite {OUT_CSV} (use --resume to '
                         'append missing rows only)')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise SystemExit('set CUDA_VISIBLE_DEVICES=1 explicitly (shared GPUs)')
    runs = anchors()
    import torch
    import knee_s3_ladder as L
    L.RUNS = runs
    L.CSV_PATH = OUT_CSV
    print(f'GPU: {torch.cuda.get_device_name(0)}; output {L.CSV_PATH}', flush=True)
    for pat, pre, post in L.RUNS:
        print(f'  {pat:40s} PRE ep{pre:<4d} POST ep{post}', flush=True)
    L.main()


# ------------------------------------------------------------------- report

def index(rows):
    by = {}
    for r in rows:
        if key_of(r['run']) in EXPECTED:
            by.setdefault((key_of(r['run']), r['phase'], r['probe']), []).append(r)
    return by


def zm(r):
    return 0.5 * (float(r['r2_z1']) + float(r['r2_z2']))


def val(by, k, phase, probe, target='z'):
    rs = by[(k, phase, probe)]
    if probe in SEEDED:
        assert len(rs) == 2, (k, phase, probe, len(rs))
    else:
        assert len(rs) == 1, (k, phase, probe, len(rs))
    return mean(zm(r) if target == 'z' else float(r['r2_E']) for r in rs)


def epoch_of(by, k, phase):
    eps = {int(r['epoch']) for p in PROBES for r in by[(k, phase, p)]}
    assert len(eps) == 1, (k, phase, eps)
    return eps.pop()


def report():
    old_rows, new_rows = read(OLD_CSV), read(OUT_CSV)
    old, new = index(old_rows), index(new_rows)
    n_new = sum(len(v) for v in new.values())
    print(f'{OUT_CSV}: {len(new_rows)} rows ({n_new} for the 13 runs; '
          f'expected {13 * 2 * 8})')
    keys = list(EXPECTED)

    # ---- gate: checkpoints whose epoch did not move
    print('\n== GATE: recomputed rows vs knee_s3_ladder.csv at unchanged '
          'checkpoints ==')
    oldk = {(key_of(r['run']), r['phase'], r['probe'], r['probe_seed']): r
            for r in old_rows if key_of(r['run']) in EXPECTED}
    worst, n_rows, n_exact, n_step, n_seeded, moved = {}, 0, 0, 0, 0, set()
    for r in new_rows:
        k = key_of(r['run'])
        o = oldk[(k, r['phase'], r['probe'], r['probe_seed'])]
        if o['epoch'] != r['epoch']:
            moved.add((k, r['phase'], o['epoch'], r['epoch']))
            continue
        n_rows += 1
        cols = ['r2_z1', 'r2_z2', 'r2_E'] + (['mean_cca'] if r['mean_cca'] else [])
        d = max(abs(float(r[c]) - float(o[c])) for c in cols)
        worst[r['probe']] = max(worst.get(r['probe'], 0.0), d)
        n_exact += all(r[c] == o[c] for c in cols)
        if r['probe'] in SEEDED:
            n_seeded += 1
            n_step += (r['best_step'], r['cap']) == (o['best_step'], o['cap'])
    for p in PROBES:
        if p in worst:
            print(f'  {p:14s} max |d| (r2_z1, r2_z2, r2_E{", mean_cca" if p == "pooled_linear" else ""})'
                  f' = {worst[p]:.2e}')
    print(f'  rows compared {n_rows}; identical to all 6 printed decimals '
          f'{n_exact}/{n_rows}; learned-probe (best_step, cap) identical '
          f'{n_step}/{n_seeded}')
    print('  checkpoints not gated (epoch moved, old -> new):')
    for k, ph, eo, en in sorted(moved):
        print(f'    {short(k):12s} {ph:4s} ep{eo} -> ep{en}')

    # ---- statement 1: every readout drops PRE -> POST
    print('\n== S1: PRE -> POST per readout (learned probes = 2-seed mean); '
          'drop = PRE - POST ==')
    for tgt, lab in (('z', 'mean shape R^2 (z1, z2)'), ('E', 'log-energy R^2')):
        print(f'\n  -- {lab} --')
        print(f'  {"run":12s} {"PRE":>9s}  ' + '  '.join(f'{p:>21s}' for p in PROBES))
        for src, by in (('old', old), ('new', new)):
            for k in keys:
                pre_ep = epoch_of(by, k, 'pre')
                cells = []
                for p in PROBES:
                    a, b = val(by, k, 'pre', p, tgt), val(by, k, 'post', p, tgt)
                    cells.append(f'{a:6.3f}->{b:6.3f} ({a - b:+.3f})')
                print(f'  {src}:{short(k):8s} ep{pre_ep:<4d}  ' + '  '.join(f'{c:>21s}' for c in cells))
        for src, by in (('old', old), ('new', new)):
            cnt = {p: sum(val(by, k, 'pre', p, tgt) > val(by, k, 'post', p, tgt)
                          for k in keys) for p in PROBES}
            mins = {p: min(val(by, k, 'pre', p, tgt) - val(by, k, 'post', p, tgt)
                           for k in keys) for p in PROBES}
            print(f'  {src} runs with PRE > POST: ' + ', '.join(
                f'{p} {cnt[p]}/13 (min drop {mins[p]:+.3f})' for p in PROBES))
        for src, by in (('old', old), ('new', new)):
            fails = []
            for k in keys:
                for p in SEEDED:
                    pre = {r['probe_seed']: r for r in by[(k, 'pre', p)]}
                    post = {r['probe_seed']: r for r in by[(k, 'post', p)]}
                    for s in sorted(pre):
                        a = zm(pre[s]) if tgt == 'z' else float(pre[s]['r2_E'])
                        b = zm(post[s]) if tgt == 'z' else float(post[s]['r2_E'])
                        if not a > b:
                            fails.append(f'{short(k)} {p} s{s} {a:.4f}->{b:.4f}')
            print(f'  {src} per-seed learned fits with no drop: '
                  f'{len(fails)}/52' + (f' ({"; ".join(fails)})' if fails else ''))

    # ---- statement 2: POST learned <= pooled (final-only)
    print('\n== S2: POST, learned probe (2-seed mean) at or below pooled ridge ==')
    for tgt in ('z', 'E'):
        for p in SEEDED:
            for src, by in (('old', old), ('new', new)):
                above = [f'{short(k)} {val(by, k, "post", p, tgt):.4f}>'
                         f'{val(by, k, "post", "pooled_linear", tgt):.4f}'
                         for k in keys if val(by, k, 'post', p, tgt)
                         > val(by, k, 'post', 'pooled_linear', tgt)]
                print(f'  {tgt} {p:9s} {src}: {13 - len(above)}/13'
                      + (f'  above: {"; ".join(above)}' if above else ''))
    print('  PRE, same count (for reference, new anchors):')
    for tgt in ('z', 'E'):
        for p in SEEDED:
            for src, by in (('old', old), ('new', new)):
                n = sum(val(by, k, 'pre', p, tgt) <= val(by, k, 'pre', 'pooled_linear', tgt)
                        for k in keys)
                print(f'    {tgt} {p:9s} {src}: {n}/13')

    # ---- statement 5: token_flat POST values
    print('\n== S5: token_flat mean shape R^2 at POST ==')
    for nm in ('p384_8', 'p384_4', 'p384_3'):
        k = 'rigb_lejp_p1s100_' + nm
        o, n = val(old, k, 'post', 'token_flat'), val(new, k, 'post', 'token_flat')
        print(f'  {nm}: old {o:.7f}  new {n:.7f}  (POST ep{epoch_of(new, k, "post")})')
    print('  token_flat mean shape R^2 at PRE (old -> new anchor):')
    for k in keys:
        print(f'    {short(k):12s} ep{epoch_of(old, k, "pre")} {val(old, k, "pre", "token_flat"):.4f}'
              f' -> ep{epoch_of(new, k, "pre")} {val(new, k, "pre", "token_flat"):.4f}')

    s7_report(new)


# ----------------------------------------------------------------- S7(i)

def traj_index():
    t = {}
    for r in read(TRAJ_CSV):
        t[(key_of(r['run']), int(r['epoch']))] = r
    return t


def s7_rows(new):
    """Per run: S7(i) cca_c1..c3 at old PRE, new PRE and POST from
    knee_traj_full.csv; each new-anchor row is first checked against the
    recomputed pooled_linear row (mean_cca and the three ridge R^2)."""
    t = traj_index()
    out, worst = [], 0.0
    for k, (old_pre, new_pre, post) in EXPECTED.items():
        for phase, ep in (('pre', new_pre), ('post', post)):
            lr = new[(k, phase, 'pooled_linear')][0]
            assert int(lr['epoch']) == ep
            tr = t[(k, ep)]
            d = max(abs(float(lr['mean_cca']) - float(tr['mean_cca'])),
                    abs(float(lr['r2_z1']) - float(tr['ridge_r2_z1'])),
                    abs(float(lr['r2_z2']) - float(tr['ridge_r2_z2'])),
                    abs(float(lr['r2_E']) - float(tr['ridge_r2_E'])))
            worst = max(worst, d)
        row = dict(run=k, old_pre_epoch=old_pre, pre_epoch=new_pre, post_epoch=post)
        for tag, ep in (('oldpre', old_pre), ('pre', new_pre), ('post', post)):
            for c in (1, 2, 3):
                row[f'c{c}_{tag}'] = t[(k, ep)][f'cca_c{c}']
        out.append(row)
    return out, worst


S7_COLS = ['run', 'old_pre_epoch', 'pre_epoch', 'post_epoch',
           'c1_oldpre', 'c2_oldpre', 'c3_oldpre',
           'c1_pre', 'c2_pre', 'c3_pre', 'c1_post', 'c2_post', 'c3_post']


def s7_report(new):
    rows, worst = s7_rows(new)
    print('\n== S7(i): per-component pooled CCA (knee_traj_full.csv) ==')
    print(f'  check: knee_traj_full.csv vs recomputed pooled_linear rows at all '
          f'26 new-anchor checkpoints, max |d| over mean_cca and ridge R^2 = {worst:.2e}')
    print(f'  {"run":12s} {"old PRE":>8s} {"c1":>6s} {"c2":>6s} {"c3":>6s} | '
          f'{"new PRE":>8s} {"c1":>6s} {"c2":>6s} {"c3":>6s} | '
          f'{"POST":>5s} {"c1":>6s} {"c2":>6s} {"c3":>6s}')
    for r in rows:
        f = lambda x: f'{float(r[x]):6.3f}'
        print(f'  {short(r["run"]):12s} {"ep" + str(r["old_pre_epoch"]):>8s} '
              f'{f("c1_oldpre")} {f("c2_oldpre")} {f("c3_oldpre")} | '
              f'{"ep" + str(r["pre_epoch"]):>8s} {f("c1_pre")} {f("c2_pre")} {f("c3_pre")} | '
              f'{"ep" + str(r["post_epoch"]):>5s} {f("c1_post")} {f("c2_post")} {f("c3_post")}')
    for tag, cols in (('old anchors', ('oldpre', 'post')), ('new anchors', ('pre', 'post'))):
        c12 = [float(r[f'c{c}_{p}']) for r in rows for c in (1, 2) for p in cols]
        c3d = [float(r[f'c3_{cols[0]}']) - float(r['c3_post']) for r in rows]
        print(f'  {tag}: c1, c2 over PRE and POST, 13 runs: min {min(c12):.4f} '
              f'max {max(c12):.4f}; c3 PRE > POST in {sum(d > 0 for d in c3d)}/13 '
              f'(c3 drop min {min(c3d):.3f}, max {max(c3d):.3f})')


def write_s7():
    if os.path.exists(S7_CSV):
        raise SystemExit(f'refusing to overwrite {S7_CSV}')
    rows, worst = s7_rows(index(read(OUT_CSV)))
    assert worst < 5e-6, f'knee_traj_full.csv disagrees with the ladder rows ({worst})'
    with open(S7_CSV, 'x', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=S7_COLS, lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'wrote {S7_CSV} ({len(rows)} rows; traj/ladder max |d| {worst:.2e})')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--write-s7', action='store_true')
    a = ap.parse_args()
    if a.report:
        report()
    elif a.write_s7:
        write_s7()
    else:
        compute(a.resume)
