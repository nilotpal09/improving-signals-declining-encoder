"""Direction sweep at the average-R2 onsets (2026-10-10).

The paper uses one onset definition. knee_dir_sweep.py anchored each run at the
old mean-CCA rules (R_A, fallback R_B); this script re-anchors at the vetted average-R2 onset
(results/knee_onsets_avgr2.csv) and scores every decline under that rule (55 runs).
Machinery imported unchanged: knee_t2_direction (feats, ridge_resid, W_KNEE) and
knee_dir_sweep.cache_for (each run on its own dial-matched cache). Inference only (re-encodes saved
checkpoints); no training.

Anchors: PRE = pre_onset_epoch of knee_onsets_avgr2.csv (last grid epoch <= onset - 20); POST = last
checkpoint. Reference directions:
  wbar_old  normalized mean of W_KNEE (the four original width-2048 runs at their old anchors);
  wbar_new  normalized mean of the same four runs' directions recomputed here at the new anchors
            (knee seed 1/2/3 = rigb_lejp_p1s100_nano_1/2/3, knee long = rigb_lejp_p1s100_long_nano_1).
pre_contam = (peak - R(PRE)) / (peak - final) on average ridge R2 (knee_onsets_avgr2.load).
Gate: rows whose PRE and POST equal the old sweep's must reproduce its w, dR2, lam_share, cos_wbar.
Writes results/knee_dir_sweep_avgr2.csv (refuses to overwrite).
"""
import sys

sys.dont_write_bytecode = True

import csv  # noqa: E402
import glob  # noqa: E402
import os  # noqa: E402

import torch  # noqa: E402
import yaml  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
sys.path.insert(0, _HERE)
import wp1_probe_driver as drv  # noqa: E402
from knee_dir_sweep import CACHE, cache_for  # noqa: E402
from knee_onsets_avgr2 import load as load_readability  # noqa: E402
from knee_t2_direction import W_KNEE, feats, ridge_resid  # noqa: E402

ONS = os.path.join(paths.CSV, 'knee_onsets_avgr2.csv')
OLD = os.path.join(paths.CSV, 'knee_dir_sweep.csv')
OUT = os.path.join(paths.CSV, 'knee_dir_sweep_avgr2.csv')
REF = {'knee seed 1': 'rigb_lejp_p1s100_nano_1', 'knee seed 2': 'rigb_lejp_p1s100_nano_2',
       'knee seed 3': 'rigb_lejp_p1s100_nano_3', 'knee long': 'rigb_lejp_p1s100_long_nano_1'}
COLS = ['run', 'cache', 'onset', 'pre', 'post', 'old_pre', 'w_z1', 'w_z2', 'w_E', 'dR2', 'lam_share',
        'cos_wbar_old', 'cos_wbar_new', 'pre_contam', 'drop', 'peak_ep', 'in_old_sweep_included']


def unit(v):
    return v / v.norm()


def direction(run, pre, post, sets):
    d = glob.glob(os.path.join(drv.ROOT, run + 'xxx*'))[0]
    cfg = yaml.safe_load(open(os.path.join(d, 'config.yml')))
    e_pre, Yne = ridge_resid(*feats(d, cfg, sets, pre))
    e_post, _ = ridge_resid(*feats(d, cfg, sets, post))
    Sp = e_pre.T @ e_pre / e_pre.shape[0]
    Sq = e_post.T @ e_post / e_post.shape[0]
    Sy = Yne.T @ Yne / Yne.shape[0] + 1e-12 * torch.eye(3, dtype=Yne.dtype)
    Li = torch.linalg.inv(torch.linalg.cholesky(Sy))
    Mm = Li @ (Sq - Sp) @ Li.T
    lam, V = torch.linalg.eigh((Mm + Mm.T) / 2)
    wv = unit(Li.T @ V[:, -1])
    if wv[0] < 0:
        wv = -wv
    return wv, float(wv @ (Sq - Sp) @ wv / (wv @ Sy @ wv)), float(lam.abs().max() / lam.abs().sum())


def main():
    assert not os.path.exists(OUT), f'refusing to overwrite {OUT}'
    print('device', drv.DEV, flush=True)
    ons = {r['run'].split('xxx')[0]: r for r in csv.DictReader(open(ONS)) if r['decline'] == '1'}
    old = {r['run']: r for r in csv.DictReader(open(OLD))}
    old_incl = {k for k, r in old.items() if r['drop'] not in ('', 'nan') and float(r['drop']) > 0
                and r['pre_contam'] not in ('', 'nan') and float(r['pre_contam']) < 1}
    R3, _ = load_readability()
    avg = {k.split('xxx')[0]: {e: sum(v) / 3 for e, v in t.items()} for k, t in R3.items()}
    wbar_old = unit(torch.stack([unit(torch.tensor(v, dtype=torch.float64)) for v in W_KNEE.values()]).mean(0))
    sets_cache, res = {}, {}
    for run in sorted(ons):
        r = ons[run]
        pre, post = int(r['pre_onset_epoch']), int(r['last_epoch'])
        assert pre == max(e for e in avg[run] if e <= int(r['onset']) - 20), run
        cname = cache_for(run)
        if cname not in sets_cache:
            sets_cache[cname] = torch.load(os.path.join(CACHE, cname), map_location='cpu', weights_only=True)
        wv, dR2, lam = direction(run, pre, post, sets_cache[cname])
        res[run] = (cname, pre, post, wv, dR2, lam)
        print(f'{run:46s} pre {pre:4d} (old {old[run]["pre"] if run in old else "-":>4s}) '
              f'cos_old {abs(float(wv @ wbar_old)):.4f} dR2 {dR2:.3f} lam {lam:.3f}', flush=True)
    wbar_new = unit(torch.stack([res[REF[k]][3] for k in W_KNEE]).mean(0))
    print('\nreference directions per run (new anchors) vs W_KNEE:')
    for k in W_KNEE:
        print(f'  {k:12s} new {[round(float(x), 3) for x in res[REF[k]][3]]}  old {W_KNEE[k]}')
    print(f'wbar_old {[round(float(x), 4) for x in wbar_old]}  wbar_new {[round(float(x), 4) for x in wbar_new]}  '
          f'|cos| {abs(float(wbar_old @ wbar_new)):.5f}')
    # gate: unchanged anchors must reproduce the old rows
    worst = 0.0
    for run, (cname, pre, post, wv, dR2, lam) in res.items():
        o = old.get(run)
        if o and int(o['pre']) == pre and int(o['post']) == post and o['cache'] == cname:
            diff = max(abs(float(wv[0]) - float(o['w_z1'])), abs(float(wv[1]) - float(o['w_z2'])),
                       abs(float(wv[2]) - float(o['w_E'])), abs(dR2 - float(o['dR2'])),
                       abs(lam - float(o['lam_share'])), abs(abs(float(wv @ wbar_old)) - float(o['cos_wbar'])))
            worst = max(worst, diff)
    print(f'gate: unchanged-anchor rows reproduce the old sweep, max |diff| = {worst:.2e}')
    rows = []
    for run, (cname, pre, post, wv, dR2, lam) in sorted(res.items()):
        t = avg[run]; pk = int(ons[run]['peak_epoch']); drop = t[pk] - t[post]
        rows.append({'run': run, 'cache': cname, 'onset': ons[run]['onset'], 'pre': pre, 'post': post,
                     'old_pre': old[run]['pre'] if run in old else '',
                     'w_z1': f'{float(wv[0]):.6f}', 'w_z2': f'{float(wv[1]):.6f}', 'w_E': f'{float(wv[2]):.6f}',
                     'dR2': f'{dR2:.6f}', 'lam_share': f'{lam:.6f}',
                     'cos_wbar_old': f'{abs(float(wv @ wbar_old)):.6f}', 'cos_wbar_new': f'{abs(float(wv @ wbar_new)):.6f}',
                     'pre_contam': f'{(t[pk] - t[pre]) / drop:.6f}', 'drop': f'{drop:.6f}', 'peak_ep': pk,
                     'in_old_sweep_included': int(run in old_incl)})
    with open(OUT, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, lineterminator='\n'); w.writeheader(); w.writerows(rows)
    print(f'wrote {OUT} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
