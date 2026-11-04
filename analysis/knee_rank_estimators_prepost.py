"""Rank estimators at PRE/POST on ONE feature matrix (2026-09-23).

Why: the logged RankMe (published formula, live training batch) rises
PRE -> POST in 13 of 13 knee runs on rankme_event and 12 of 13 on
rankme_tokens (results/knee_rankme_logged_prepost.csv), while the record's
squared-spectrum RankMe on probe-cache features rises in 10 of 13, with 4x
seed 9 falling on it and on the participation ratio. Estimator, centring
and sample differ between the two, so this computes the candidates on the
same matrix.

Matrix: for each of the 13 knee runs with ladder rows (21x nano_1-3,
long_nano_1; 4x p384_1-9) at the PRE/POST epochs in results/knee_s3_ladder.csv,
the 4,000-event fitting split of rig_P1_Enar_S100.pt encoded exactly as
knee_s7_mech.py:80-90 does (wp1_probe_driver.encoder_from_ckpt on
jepa-ep{ep}.pth.tar, encode_meanpool: tokens -> LayerNorm -> mean,
then .double()).

Quantities (all on that matrix X, 4000 x 96):
  rankme_pub_centred    published RankMe, src/utils/logging.py:98-105
                        (imported, not transcribed): exp entropy of the
                        singular values normalised to sum 1; on X - mean
  rankme_sq_centred     knee_s7_mech.py:93-99 (transcribed; nested there):
                        exp entropy of the SQUARED singular values
                        normalised to sum 1; on X - mean
  pr_centred            participation ratio (sum s^2)^2 / sum s^4 on
                        X - mean (knee_s7_mech.py:97)
  rankme_pub_uncentred  published RankMe on X itself (the logged
                        implementation does not centre)
  rankme_pub_uncentred_b128  (addition) published RankMe on X uncentred,
                        computed on 31 consecutive 128-event batches of
                        the fitting split and averaged: the logged value's
                        batch size (config batch_size 128) on probe-cache
                        events
Percent change = 100 * (post - pre) / pre.

Output: results/knee_rank_estimators_prepost.csv (new).

Usage (from analysis/; set the GPU):
  CUDA_VISIBLE_DEVICES=0 python knee_rank_estimators_prepost.py
"""
import sys

sys.dont_write_bytecode = True      # leave analysis/__pycache__ untouched

import csv  # noqa: E402
import inspect  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
sys.path.insert(0, _HERE)

CACHE_PATH = os.path.join(paths.CACHE, 'rig_P1_Enar_S100.pt')
LADDER_CSV = os.path.join(paths.CSV, 'knee_s3_ladder.csv')
OUT = os.path.join(paths.CSV, 'knee_rank_estimators_prepost.csv')
RUN_RE = re.compile(r'^rigb_lejp_p1s100_(nano_[123]|long_nano_1|p384_[1-9])xxx')
QUANTS = ('rankme_pub_centred', 'rankme_sq_centred', 'pr_centred',
          'rankme_pub_uncentred', 'rankme_pub_uncentred_b128')
B_LOG = 128

# record values for the check (squared-spectrum RankMe and PR, PRE -> POST):
# the 21x runs; the 4x runs s2-s4 and s7-s9; and the 4x runs
# s1, s5, s6 (via the R_B supplement)
RECORD = {  # label: ((PR pre, PR post), (RankMe_sq pre, RankMe_sq post))
    'nano_1': ((1.90, 5.01), (2.80, 6.48)),
    'nano_2': ((1.38, 1.64), (1.83, 2.31)),
    'nano_3': ((1.76, 2.65), (2.53, 3.99)),
    'long_nano_1': ((3.33, 4.71), (4.74, 6.36)),
    'p384_2': ((2.41, 4.20), (3.34, 5.28)),
    'p384_3': ((3.18, 2.65), (4.10, 4.58)),
    'p384_4': ((3.08, 4.02), (3.68, 4.76)),
    'p384_7': ((2.33, 5.32), (3.19, 7.32)),
    'p384_8': ((2.68, 4.82), (3.76, 7.46)),
    'p384_9': ((3.15, 2.21), (4.36, 3.27)),
    'p384_1': ((1.61, 1.57), (2.04, 1.83)),
    'p384_5': ((1.81, 3.64), (2.32, 5.92)),
    'p384_6': ((1.52, 4.88), (1.78, 7.01)),
}


def label(run):
    return run.split('p1s100_')[1].split('xxx')[0]


def ladder_epochs():
    seen = {}
    with open(LADDER_CSV) as fh:
        for r in csv.DictReader(fh):
            if RUN_RE.match(r['run']):
                seen.setdefault(r['run'], {})[r['phase']] = int(r['epoch'])
    assert len(seen) == 13, sorted(seen)
    return {k: (v['pre'], v['post']) for k, v in sorted(seen.items())}


def spectrum_s7(X):
    """knee_s7_mech.py:93-99, transcribed (it is nested inside main())."""
    import torch
    Xc = X - X.mean(0)
    s = torch.linalg.svdvals(Xc)
    p = (s ** 2) / (s ** 2).sum()
    pr = float((s ** 2).sum() ** 2 / (s ** 4).sum())
    rankme = float(torch.exp(-(p * torch.log(p.clamp(min=1e-30))).sum()))
    return pr, rankme


def main():
    import torch
    import yaml
    import wp1_probe_driver as drv
    from src.utils.logging import effective_rank     # the published RankMe
    src = inspect.getsourcefile(effective_rank)
    assert src.endswith('train/src/utils/logging.py'), src
    print(f'published RankMe imported from {src}')
    print(f'GPU: {torch.cuda.get_device_name(0)} '
          f'(CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")})',
          flush=True)

    sets = torch.load(CACHE_PATH, map_location='cpu', weights_only=True)
    imgs = sets['imgs_tr']
    n_b = imgs.shape[0] // B_LOG                     # 31 batches of 128
    rows = []
    for run, (pre, post) in ladder_epochs().items():
        d = os.path.join(drv.ROOT, run)
        cfg = yaml.safe_load(open(os.path.join(d, 'config.yml')))
        assert cfg['logging']['write_tag'] == 'jepa'
        vals = {}
        for ph, ep in (('pre', pre), ('post', post)):
            enc = drv.encoder_from_ckpt(cfg, os.path.join(d, f'jepa-ep{ep}.pth.tar'))
            X = drv.encode_meanpool(enc, imgs).double()   # knee_s7_mech.feats
            del enc
            pr, rm_sq = spectrum_s7(X)
            vals[ph] = {
                'rankme_pub_centred': effective_rank(X - X.mean(0)),
                'rankme_sq_centred': rm_sq,
                'pr_centred': pr,
                'rankme_pub_uncentred': effective_rank(X),
                'rankme_pub_uncentred_b128': sum(
                    effective_rank(X[i * B_LOG:(i + 1) * B_LOG])
                    for i in range(n_b)) / n_b,
            }
        row = {'run': run, 'label': label(run), 'pre_ep': pre, 'post_ep': post}
        for q in QUANTS:
            a, b = vals['pre'][q], vals['post'][q]
            row[f'{q}_pre'] = f'{a:.6f}'
            row[f'{q}_post'] = f'{b:.6f}'
            row[f'{q}_pct'] = f'{100.0 * (b - a) / a:.2f}'
        rows.append(row)
        print(f"{row['label']:12s} pre {pre:3d} post {post:3d}  " + '  '.join(
            f"{q.replace('rankme_', 'rm_')}: {float(row[q + '_pre']):.3f}->"
            f"{float(row[q + '_post']):.3f} ({float(row[q + '_pct']):+.1f}%)"
            for q in QUANTS), flush=True)

    assert not os.path.exists(OUT), f'refusing to overwrite {OUT}'
    cols = ['run', 'label', 'pre_ep', 'post_ep'] + [
        f'{q}_{s}' for q in QUANTS for s in ('pre', 'post', 'pct')]
    with open(OUT, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {OUT}')

    print('\ncheck against the record (PR, squared-spectrum RankMe; 2 decimals):')
    worst = 0.0
    for row in rows:
        (pr0, pr1), (rm0, rm1) = RECORD[row['label']]
        got = (float(row['pr_centred_pre']), float(row['pr_centred_post']),
               float(row['rankme_sq_centred_pre']), float(row['rankme_sq_centred_post']))
        rec = (pr0, pr1, rm0, rm1)
        d = max(abs(g - r) for g, r in zip(got, rec))
        worst = max(worst, d)
        print(f"  {row['label']:12s} record PR {pr0:.2f}->{pr1:.2f} RM {rm0:.2f}->{rm1:.2f}"
              f" | now PR {got[0]:.2f}->{got[1]:.2f} RM {got[2]:.2f}->{got[3]:.2f}"
              f"  max|d| {d:.3f}")
    print(f'  worst |d| = {worst:.3f} (record prints 2 decimals)')

    print('\nrises PRE->POST out of 13 (pct > 0):')
    for q in QUANTS:
        n = sum(float(r[f'{q}_pct']) > 0 for r in rows)
        print(f'  {q:28s} {n:2d} / {len(rows)}')


if __name__ == '__main__':
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        raise SystemExit('set CUDA_VISIBLE_DEVICES explicitly (shared GPUs)')
    main()
