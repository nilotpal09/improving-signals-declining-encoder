"""Capacity ladder test-split rescoring (knee_s3_ladder_split.py) at the
average-ridge-R^2 pre-onset anchors (2026-10-10). New file; nothing
existing is modified.

knee_s3_ladder_split.main() is imported unchanged. It reads the 13 runs'
PRE/POST epochs from knee_s3_ladder.CSV_PATH; here that constant points to
results/knee_s3_ladder_avgr2.csv (written by knee_s3_ladder_avgr2.py, PRE =
pre_onset_epoch of results/knee_onsets_avgr2.csv, POST = final checkpoint), and
the split module's OUT_CSV points to a new file. Same test split
(rig_P1_Enar_S100_test.pt), same probes, seeds and selection rule. All 26
checkpoints are recomputed, so the 20 whose epoch did not move are a
reproduction gate against results/knee_s3_ladder_split.csv, and every
r2_*_eval value is checked against results/knee_s3_ladder_avgr2.csv.

Modes (from analysis/; run with -B):
  CUDA_VISIBLE_DEVICES=1 python -B knee_s3_ladder_split_avgr2.py           # compute
  CUDA_VISIBLE_DEVICES=1 python -B knee_s3_ladder_split_avgr2.py --resume  # append missing rows only
  python -B knee_s3_ladder_split_avgr2.py --report                          # reads CSVs only

Output (new; refuses to overwrite): results/knee_s3_ladder_split_avgr2.csv,
same columns as results/knee_s3_ladder_split.csv.
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

import knee_s3_ladder_avgr2 as A  # noqa: E402  (EXPECTED anchors, CSV readers)

NEW_LADDER = A.OUT_CSV
OLD_SPLIT = os.path.join(paths.CSV, 'knee_s3_ladder_split.csv')
OUT_CSV = os.path.join(paths.CSV, 'knee_s3_ladder_split_avgr2.csv')
SEEDED = ('query85k', 'mlp1M')
PROBES = ('pooled_linear',) + SEEDED
NUM_COLS = ['r2_z1_eval', 'r2_z2_eval', 'r2_E_eval',
            'r2_z1_test', 'r2_z2_test', 'r2_E_test',
            'r2_z1_final_test', 'r2_z2_final_test', 'r2_E_final_test',
            'r2_z1_final_eval', 'r2_z2_final_eval', 'r2_E_final_eval']


def check_ladder():
    """The new ladder CSV must hold all 13 x 2 x 8 rows at the expected
    anchors before the split reads its epochs."""
    rows = A.read(NEW_LADDER)
    seen = {}
    for r in rows:
        seen.setdefault((A.key_of(r['run']), r['phase']), []).append(int(r['epoch']))
    for k, (_, pre, post) in A.EXPECTED.items():
        for ph, ep in (('pre', pre), ('post', post)):
            eps = seen.get((k, ph), [])
            assert len(eps) == 8 and set(eps) == {ep}, (k, ph, eps)
    assert len(seen) == 26, sorted(seen)


def compute(resume):
    if os.path.exists(OUT_CSV) and not resume:
        raise SystemExit(f'refusing to overwrite {OUT_CSV} (use --resume to '
                         'append missing rows only)')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise SystemExit('set CUDA_VISIBLE_DEVICES=1 explicitly (shared GPUs)')
    check_ladder()
    import knee_s3_ladder as L
    import knee_s3_ladder_split as S
    L.CSV_PATH = NEW_LADDER
    S.OUT_CSV = OUT_CSV
    for run, phase, ep in S.ladder_runs(L.CSV_PATH):
        print(f'  {run:40s} {phase:4s} ep{ep}', flush=True)
    S.main()


def zm(r, split):
    return 0.5 * (float(r[f'r2_z1_{split}']) + float(r[f'r2_z2_{split}']))


def report():
    check_ladder()
    new = A.read(OUT_CSV)
    old = {(A.key_of(r['run']), r['phase'], r['probe'], r['probe_seed']): r
           for r in A.read(OLD_SPLIT)}
    lad = {(A.key_of(r['run']), r['phase'], r['probe'], r['probe_seed']): r
           for r in A.read(NEW_LADDER)}
    print(f'{OUT_CSV}: {len(new)} rows (expected 130)')

    print('\n== check 1: r2_*_eval vs knee_s3_ladder_avgr2.csv (all rows) ==')
    worst, n_step, n_seeded = {}, 0, 0
    for r in new:
        o = lad[(A.key_of(r['run']), r['phase'], r['probe'], r['probe_seed'])]
        assert o['epoch'] == r['epoch']
        d = max(abs(float(r[f'r2_{t}_eval']) - float(o[f'r2_{t}'])) for t in ('z1', 'z2', 'E'))
        worst[r['probe']] = max(worst.get(r['probe'], 0.0), d)
        if r['probe'] in SEEDED:
            n_seeded += 1
            n_step += r['best_step'] == o['best_step']
    for p in PROBES:
        print(f'  {p:14s} max |d| = {worst[p]:.2e}')
    print(f'  learned-probe best_step identical {n_step}/{n_seeded}')

    print('\n== GATE: vs knee_s3_ladder_split.csv at unchanged checkpoints '
          '(all 12 R^2 columns + best_step) ==')
    worst, n_rows, n_exact, n_step, n_seeded, moved = {}, 0, 0, 0, 0, set()
    for r in new:
        k = A.key_of(r['run'])
        o = old[(k, r['phase'], r['probe'], r['probe_seed'])]
        if o['epoch'] != r['epoch']:
            moved.add((k, r['phase'], o['epoch'], r['epoch']))
            continue
        n_rows += 1
        cols = [c for c in NUM_COLS if r[c] != '' or o[c] != '']
        d = max(abs(float(r[c]) - float(o[c])) for c in cols)
        worst[r['probe']] = max(worst.get(r['probe'], 0.0), d)
        n_exact += all(r[c] == o[c] for c in NUM_COLS)
        if r['probe'] in SEEDED:
            n_seeded += 1
            n_step += r['best_step'] == o['best_step']
    for p in PROBES:
        print(f'  {p:14s} max |d| = {worst[p]:.2e}')
    print(f'  rows compared {n_rows}; identical in every column {n_exact}/{n_rows}; '
          f'learned-probe best_step identical {n_step}/{n_seeded}')
    print('  not gated (epoch moved): ' + ', '.join(
        f'{A.short(k)} {ph} ep{eo}->ep{en}' for k, ph, eo, en in sorted(moved)))

    by = {}
    for r in new:
        by.setdefault((A.key_of(r['run']), r['phase'], r['probe']), []).append(r)
    obys = {}
    for k, r in old.items():
        if k[0] in A.EXPECTED:
            obys.setdefault(k[:3], []).append(r)

    def v(b, k, ph, p, split, tgt='z'):
        rs = b[(k, ph, p)]
        assert len(rs) == (2 if p in SEEDED else 1)
        return mean(zm(r, split) if tgt == 'z' else float(r[f'r2_E_{split}']) for r in rs)

    print('\n== test minus eval, mean shape R^2, per row (learned probes at the '
          'eval-selected step) ==')
    for src, rows in (('old', [r for r in old.values() if A.key_of(r['run']) in A.EXPECTED]),
                      ('new', new)):
        for p in PROBES:
            d = [zm(r, 'test') - zm(r, 'eval') for r in rows if r['probe'] == p]
            print(f'  {src} {p:14s} n={len(d):2d} mean {mean(d):+.4f} min {min(d):+.4f} '
                  f'max {max(d):+.4f} max|d| {max(abs(x) for x in d):.4f}')
    d = [abs(zm(r, 'test') - zm(r, 'eval')) for r in new
         if (A.key_of(r['run']), r['phase'], r['epoch']) in
         {(k, ph, en) for k, ph, eo, en in moved}]
    print(f'  new-anchor (moved) rows only: n={len(d)} max|test - eval| {max(d):.4f}')
    print('  (2-seed means, for reference)')
    for p in PROBES:
        d = [v(by, k, ph, p, 'test') - v(by, k, ph, p, 'eval')
             for k in A.EXPECTED for ph in ('pre', 'post')]
        print(f'    new {p:14s} n={len(d)} min {min(d):+.4f} max {max(d):+.4f}')

    print('\n== POST, learned probe (2-seed mean) at or below pooled ridge ==')
    for tgt in ('z', 'E'):
        for p in SEEDED:
            for split in ('eval', 'test', 'final_test'):
                ref = 'eval' if split == 'eval' else 'test'
                for src, b in (('old', obys), ('new', by)):
                    above = [A.short(k) for k in A.EXPECTED
                             if v(b, k, 'post', p, split, tgt) > v(b, k, 'post', 'pooled_linear', ref, tgt)]
                    print(f'  {tgt} {p:9s} {split:10s} {src}: {13 - len(above)}/13'
                          + (f' (above: {", ".join(above)})' if above else ''))

    print('\n== PRE -> POST on the test split (eval-selected step; 2-seed mean) ==')
    for tgt in ('z', 'E'):
        for p in PROBES:
            for src, b in (('old', obys), ('new', by)):
                drops = {A.short(k): v(b, k, 'pre', p, 'test', tgt) - v(b, k, 'post', p, 'test', tgt)
                         for k in A.EXPECTED}
                neg = [f'{n} {x:+.4f}' for n, x in drops.items() if not x > 0]
                print(f'  {tgt} {p:14s} {src}: drop > 0 in {13 - len(neg)}/13; '
                      f'min drop {min(drops.values()):+.4f}' + (f' ({"; ".join(neg)})' if neg else ''))

    print('\n== moved PRE rows, new values (z eval / z test / E eval / E test) ==')
    for k, ph, eo, en in sorted(moved):
        if ph != 'pre':
            continue
        for p in PROBES:
            print(f'  {A.short(k):12s} ep{eo}->ep{en} {p:14s} '
                  f'old {v(obys, k, ph, p, "eval"):.4f}/{v(obys, k, ph, p, "test"):.4f}/'
                  f'{v(obys, k, ph, p, "eval", "E"):.4f}/{v(obys, k, ph, p, "test", "E"):.4f}  '
                  f'new {v(by, k, ph, p, "eval"):.4f}/{v(by, k, ph, p, "test"):.4f}/'
                  f'{v(by, k, ph, p, "eval", "E"):.4f}/{v(by, k, ph, p, "test", "E"):.4f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--report', action='store_true')
    a = ap.parse_args()
    if a.report:
        report()
    else:
        compute(a.resume)
