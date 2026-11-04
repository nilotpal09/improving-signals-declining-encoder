"""Table-1 anatomy battery (S9 rank-one coherence + S5 log-energy readout change + PRE contamination)
re-anchored at the average-R2 pre-onset epochs (2026-10-10).

The paper's anomaly-anatomy table (13 runs: the four original width-2048 runs nano_1/2/3, long_nano_1 and
the nine Gaussian width-384 runs p384_1..9) was computed at the old mean-CCA anchors (S9 / 4x S9 / R_B
supplement in the record). The paper now times declines by the vetted average-R2 rule
(results/knee_onsets_avgr2.csv): PRE = pre_onset_epoch (last grid epoch <= onset - 20),
POST = final checkpoint (unchanged).

Machinery IMPORTED unchanged:
  knee_s9_consistency  resid, quantities, CACHE, SETS, N_BOOT, BOOT_SEED (formulas and bootstrap protocol)
  wp1_probe_driver     encoder_from_ckpt, encode_meanpool (features)
  wp1_metrics          ridge_r2 (house pooled-linear ridge R2, = S3 ladder 'pooled_linear' = S5(i))
  knee_s5s7_4x_report  T, at (S5(ii) ridge_r2_E trajectory lookup from results/knee_traj_full.csv)
  knee_onsets_avgr2    load (average-R2 trajectories)
  knee_t2_direction    W_KNEE (old reference direction)
The bootstrap loop below reproduces knee_s9_consistency.main()'s loop (generator re-seeded per run with
BOOT_SEED, N_BOOT resamples of the eval rows, 5%/95% quantiles of measured - predicted_exact).

Contamination (fraction of the run's fall already realised at PRE):
  mean CCA  (peak - CCA(PRE)) / (peak - CCA(final)), results/knee_trajectories.csv, peak = argmax over all
            saved epochs (the knee_dir_sweep.py / knee_dir_e1000.py definition).
  avg R2    same on average ridge R2 (knee_onsets_avgr2.load), peak = peak_epoch of knee_onsets_avgr2.csv
            (the knee_dir_sweep_avgr2.py definition).

Every run is computed at BOTH anchor sets (old and new); rows with pre_changed = 0 are the same window.
Gates (printed): old-anchor rows vs the record's S9 tables, knee_dir_sweep.csv, knee_s3_ladder.csv and
the 4x S5(ii) drops; new-anchor rows vs knee_dir_sweep_avgr2.csv. Inference only, no training.
Writes results/knee_s9_consistency_avgr2.csv (refuses to overwrite).
"""
import sys

sys.dont_write_bytecode = True

import contextlib  # noqa: E402
import csv  # noqa: E402
import glob  # noqa: E402
import io  # noqa: E402
import os  # noqa: E402

import torch  # noqa: E402
import yaml  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
sys.path.insert(0, _HERE)
import wp1_probe_driver as drv  # noqa: E402
import knee_s9_consistency as s9  # noqa: E402
from knee_onsets_avgr2 import load as load_readability  # noqa: E402
from knee_t2_direction import W_KNEE  # noqa: E402
from wp1_metrics import ridge_r2  # noqa: E402

with contextlib.redirect_stdout(io.StringIO()):          # module prints the old 4x report on import
    import knee_s5s7_4x_report as s5x  # noqa: E402

CSV = os.path.join(paths.CSV)
ONS = os.path.join(CSV, 'knee_onsets_avgr2.csv')
OLD_SWEEP = os.path.join(CSV, 'knee_dir_sweep.csv')
NEW_SWEEP = os.path.join(CSV, 'knee_dir_sweep_avgr2.csv')
LADDER = os.path.join(CSV, 'knee_s3_ladder.csv')
TRAJ = os.path.join(CSV, 'knee_trajectories.csv')
OUT = os.path.join(CSV, 'knee_s9_consistency_avgr2.csv')

P = 'rigb_lejp_p1s100_'
ORIG = [P + s for s in ('nano_1', 'nano_2', 'nano_3', 'long_nano_1')]
RA6 = [P + f'p384_{s}' for s in (2, 3, 4, 7, 8, 9)]
RBS = [P + f'p384_{s}' for s in (1, 5, 6)]
GROUP = {**{r: 'orig_w2048' for r in ORIG}, **{r: 'w384_RA' for r in RA6}, **{r: 'w384_RBsupp' for r in RBS}}
RUNS = ORIG + sorted(RA6 + RBS)
REF = {'knee seed 1': P + 'nano_1', 'knee seed 2': P + 'nano_2', 'knee seed 3': P + 'nano_3',
       'knee long': P + 'long_nano_1'}
# The expected old -> new PRE list, asserted against the sources (not used as a source).
EXPECT = {'long_nano_1': (140, 130), 'nano_1': (70, 60), 'nano_2': (90, 90), 'nano_3': (80, 80),
          'p384_1': (30, 30), 'p384_2': (120, 110), 'p384_3': (70, 50), 'p384_4': (120, 120),
          'p384_5': (40, 40), 'p384_6': (50, 50), 'p384_7': (60, 40), 'p384_8': (70, 30), 'p384_9': (100, 100)}

# Gate literals: the printed record.
#   S9 RESULT table: dR2(w), w_E, pred_exact, pred_reduced, measured, meas-pred, CI, share, eigs
REC21 = {P + 'nano_1': (0.682, 0.549, 0.2019, 0.2058, 0.1998, -0.0022, -0.0030, -0.0013, 0.993, (0.6824, 0.0001, -0.0045)),
         P + 'nano_2': (0.404, 0.573, 0.1307, 0.1329, 0.1274, -0.0033, -0.0048, -0.0015, 0.985, (0.4042, -0.0001, -0.0060)),
         P + 'nano_3': (0.486, 0.575, 0.1581, 0.1605, 0.1589, 0.0009, -0.0001, 0.0017, 0.989, (0.4858, 0.0045, -0.0009)),
         P + 'long_nano_1': (0.491, 0.569, 0.1563, 0.1590, 0.1567, 0.0003, -0.0007, 0.0013, 0.997, (0.4911, 0.0005, -0.0009))}
#   4x S9 table: dR2(w), w_E, predicted, measured, meas-pred, CI, share
REC4 = {P + 'p384_2': (0.680, 0.591, 0.2336, 0.2291, -0.0046, -0.0066, -0.0027, 0.983),
        P + 'p384_3': (0.245, 0.594, 0.0849, 0.0816, -0.0033, -0.0044, -0.0022, 0.961),
        P + 'p384_4': (0.526, 0.580, 0.1736, 0.1712, -0.0024, -0.0038, -0.0011, 0.991),
        P + 'p384_7': (0.540, 0.569, 0.1717, 0.1714, -0.0003, -0.0018, 0.0012, 0.992),
        P + 'p384_8': (0.058, 0.766, 0.0332, 0.0291, -0.0041, -0.0052, -0.0029, 0.787),
        P + 'p384_9': (0.628, 0.555, 0.1901, 0.1897, -0.0004, -0.0008, 0.0000, 0.979)}
#   R_B supplement S9: dR2(w), predicted, measured, meas-pred, share
RECS = {P + 'p384_1': (0.363, 0.1306, 0.1228, -0.0078, 0.961),
        P + 'p384_5': (0.545, 0.1808, 0.1685, -0.0123, 0.960),
        P + 'p384_6': (0.515, 0.1882, 0.1782, -0.0099, 0.966)}
#   4x S5(ii) drops, onset-20 -> end
RECS5 = {P + 'p384_1': 0.123, P + 'p384_2': 0.229, P + 'p384_3': 0.082, P + 'p384_4': 0.171, P + 'p384_5': 0.169,
         P + 'p384_6': 0.178, P + 'p384_7': 0.171, P + 'p384_8': 0.029, P + 'p384_9': 0.190}

COLS = ['run', 'group', 'anchor', 'pre_changed', 'onset_avgr2', 'pre', 'post', 'w_z1', 'w_z2', 'w_E', 'dR2_w',
        'pred_exact', 'pred_reduced', 'measured', 'meas_minus_pred', 'boot_lo', 'boot_hi', 'ci_excludes_0',
        'lam1', 'lam2', 'lam3', 'lam1_share', 'r2E_pre_pooled', 'r2E_post_pooled', 'r2E_drop_pooled',
        'r2E_pre_traj', 'r2E_end_traj', 'r2E_drop_traj', 'contam_meancca', 'contam_avgr2',
        'cos_wbar_old', 'cos_wbar_new']


def unit(v):
    return v / v.norm()


def battery(e_pre, e_post, Yne):
    """knee_s9_consistency.main()'s per-run computation, with the formulas imported."""
    meas, pex, pred, w, dr2_w, lam = s9.quantities(e_pre, e_post, Yne)
    share = float(lam[0].abs() / lam.abs().sum())
    g = torch.Generator().manual_seed(s9.BOOT_SEED)
    n = e_pre.shape[0]
    diffs = []
    for _ in range(s9.N_BOOT):
        idx = torch.randint(0, n, (n,), generator=g)
        m2, p2, _, _, _, _ = s9.quantities(e_pre[idx], e_post[idx], Yne[idx])
        diffs.append(m2 - p2)
    diffs = torch.tensor(diffs, dtype=torch.float64)
    lo, hi = float(diffs.quantile(0.05)), float(diffs.quantile(0.95))
    return dict(meas=meas, pex=pex, pred=pred, w=w, dr2_w=dr2_w, lam=lam, share=share, lo=lo, hi=hi)


def main():
    assert not os.path.exists(OUT), f'refusing to overwrite {OUT}'
    print('device', drv.DEV, flush=True)
    sets = torch.load(os.path.join(s9.CACHE, s9.SETS), map_location='cpu', weights_only=True)
    y_tr, y_ev = sets['y_tr'], sets['y_ev']

    # ---- anchors: new from knee_onsets_avgr2.csv, old from the S9 script (orig) and knee_dir_sweep.csv (w384)
    ons = {r['run'].split('xxx')[0]: r for r in csv.DictReader(open(ONS))}
    old_sweep = {r['run']: r for r in csv.DictReader(open(OLD_SWEEP))}
    new_sweep = {r['run']: r for r in csv.DictReader(open(NEW_SWEEP))}
    s9_old = {pat.split('xxx')[0]: pp for pat, pp in s9.RUNS.items()}
    anchors = {}
    for run in RUNS:
        o = ons[run]
        assert o['decline'] == '1', run
        new = (int(o['pre_onset_epoch']), int(o['last_epoch']))
        old = s9_old[run] if run in s9_old else (int(old_sweep[run]['pre']), int(old_sweep[run]['post']))
        if run in s9_old and run in old_sweep:                 # long_nano_1 is in both; they must agree
            assert old == (int(old_sweep[run]['pre']), int(old_sweep[run]['post'])), run
        assert new[1] == old[1], f'{run}: POST changed {old} -> {new}'
        assert EXPECT[run.replace(P, '')] == (old[0], new[0]), f'{run}: {old} -> {new} vs the expected list'
        anchors[run] = {'old': old, 'new': new}

    # ---- trajectories for contamination
    cca = {}
    for r in csv.DictReader(open(TRAJ)):
        cca.setdefault(r['run'].split('xxx')[0], {})[int(r['epoch'])] = float(r['mean_cca'])
    R3, _ = load_readability()
    avg = {k: {e: sum(v) / 3 for e, v in t.items()} for k, t in R3.items()}

    def contam_cca(run, pre, post):
        t = cca[run]
        pk = max(t, key=lambda e: t[e])
        total = t[pk] - t[post]
        return (t[pk] - t[pre]) / total if total > 0 else float('nan')

    def contam_avg(run, pre, post):
        t = avg[run]
        pk = int(ons[run]['peak_epoch'])
        return (t[pk] - t[pre]) / (t[pk] - t[post])

    # ---- encode each needed checkpoint once
    feats = {}
    for run in RUNS:
        d = glob.glob(os.path.join(drv.ROOT, run + 'xxx*'))
        assert len(d) == 1, (run, d)
        d = d[0]
        cfg = yaml.safe_load(open(os.path.join(d, 'config.yml')))
        eps = sorted({anchors[run]['old'][0], anchors[run]['new'][0], anchors[run]['new'][1]})
        for ep in eps:
            enc = drv.encoder_from_ckpt(cfg, os.path.join(d, f'jepa-ep{ep}.pth.tar'))
            feats[(run, ep)] = (drv.encode_meanpool(enc, sets['imgs_tr']), drv.encode_meanpool(enc, sets['imgs_ev']))
            del enc
        print(f'encoded {run} at {eps}', flush=True)

    # ---- the battery per (run, window)
    res = {}
    for run in RUNS:
        for which in ('old', 'new'):
            pre, post = anchors[run][which]
            if (run, pre, post) in res:
                continue
            e_pre, Yne = s9.resid(feats[(run, pre)][0], y_tr, feats[(run, pre)][1], y_ev)
            e_post, _ = s9.resid(feats[(run, post)][0], y_tr, feats[(run, post)][1], y_ev)
            b = battery(e_pre, e_post, Yne)
            b['r2E_pre'] = ridge_r2(feats[(run, pre)][0], y_tr, feats[(run, pre)][1], y_ev)[2]
            b['r2E_post'] = ridge_r2(feats[(run, post)][0], y_tr, feats[(run, post)][1], y_ev)[2]
            b['r2E_pre_traj'] = s5x.at(run, pre, 'ridge_r2_E')
            b['r2E_end_traj'] = s5x.at(run, max(s5x.T[run]), 'ridge_r2_E')
            assert pre in s5x.T[run] and max(s5x.T[run]) == post, run
            b['c_cca'] = contam_cca(run, pre, post)
            b['c_avg'] = contam_avg(run, pre, post)
            res[(run, pre, post)] = b

    wbar_old = unit(torch.stack([unit(torch.tensor(v, dtype=torch.float64)) for v in W_KNEE.values()]).mean(0))
    wbar_new = unit(torch.stack([res[(REF[k], *anchors[REF[k]]['new'])]['w'] for k in W_KNEE]).mean(0))
    print(f'wbar_old {[round(float(x), 4) for x in wbar_old]}  wbar_new {[round(float(x), 4) for x in wbar_new]}  '
          f'|cos| {abs(float(wbar_old @ wbar_new)):.5f}')

    # ---- gates
    def rec_diff(pairs):
        return max(abs(a - b) for a, b in pairs)

    print('\n== GATES ==')
    g_unch, g_chg = [], []                         # (|diff| vs printed record) split by whether PRE changed
    for run in RUNS:
        b = res[(run, *anchors[run]['old'])]
        if run in REC21:
            r = REC21[run]
            pairs = [(b['dr2_w'], r[0]), (float(b['w'][2]), r[1]), (b['pex'], r[2]), (b['pred'], r[3]),
                     (b['meas'], r[4]), (b['meas'] - b['pex'], r[5]), (b['lo'], r[6]), (b['hi'], r[7]),
                     (b['share'], r[8])] + [(float(b['lam'][i]), r[9][i]) for i in range(3)]
        elif run in REC4:
            r = REC4[run]
            pairs = [(b['dr2_w'], r[0]), (float(b['w'][2]), r[1]), (b['pex'], r[2]), (b['meas'], r[3]),
                     (b['meas'] - b['pex'], r[4]), (b['lo'], r[5]), (b['hi'], r[6]), (b['share'], r[7])]
        else:
            r = RECS[run]
            pairs = [(b['dr2_w'], r[0]), (b['pex'], r[1]), (b['meas'], r[2]), (b['meas'] - b['pex'], r[3]),
                     (b['share'], r[4])]
        dmax = rec_diff(pairs)
        (g_unch if anchors[run]['old'] == anchors[run]['new'] else g_chg).append(dmax)
        print(f'  S9 record (printed, 3-4 dp) {run.replace(P, ""):12s} old anchor {anchors[run]["old"]}: '
              f'max |diff| {dmax:.2e}')
    print(f'GATE S9 vs printed record: unchanged-PRE runs max |diff| {max(g_unch):.2e}; '
          f'changed-PRE runs at their OLD anchor max |diff| {max(g_chg):.2e} (printed rounding = 5e-4 at 3 dp)')

    worst_old, worst_new, worst_lad, worst_traj, worst_s5, worst_cc, worst_ca = 0, 0, 0, 0, 0, 0, 0
    for run in RUNS:
        bo = res[(run, *anchors[run]['old'])]
        bn = res[(run, *anchors[run]['new'])]
        if run in old_sweep and (int(old_sweep[run]['pre']), int(old_sweep[run]['post'])) == anchors[run]['old']:
            o = old_sweep[run]
            worst_old = max(worst_old, *(abs(float(bo['w'][i]) - float(o[c])) for i, c in enumerate(('w_z1', 'w_z2', 'w_E'))),
                            abs(bo['dr2_w'] - float(o['dR2'])), abs(bo['share'] - float(o['lam_share'])),
                            abs(abs(float(bo['w'] @ wbar_old)) - float(o['cos_wbar'])))
            worst_cc = max(worst_cc, abs(bo['c_cca'] - float(o['pre_contam'])))
        o = new_sweep[run]
        assert (int(o['pre']), int(o['post'])) == anchors[run]['new'], run
        worst_new = max(worst_new, *(abs(float(bn['w'][i]) - float(o[c])) for i, c in enumerate(('w_z1', 'w_z2', 'w_E'))),
                        abs(bn['dr2_w'] - float(o['dR2'])), abs(bn['share'] - float(o['lam_share'])),
                        abs(abs(float(bn['w'] @ wbar_old)) - float(o['cos_wbar_old'])),
                        abs(abs(float(bn['w'] @ wbar_new)) - float(o['cos_wbar_new'])))
        worst_ca = max(worst_ca, abs(bn['c_avg'] - float(o['pre_contam'])))
        for r in csv.DictReader(open(LADDER)):
            if r['run'].split('xxx')[0] == run and r['probe'] == 'pooled_linear':
                ep = int(r['epoch'])
                assert ep == anchors[run]['old'][{'pre': 0, 'post': 1}[r['phase']]], (run, r['phase'], ep)
                v = bo['r2E_pre'] if r['phase'] == 'pre' else bo['r2E_post']
                worst_lad = max(worst_lad, abs(v - float(r['r2_E'])))
        for b in (bo, bn):
            worst_traj = max(worst_traj, abs(b['r2E_pre'] - b['r2E_pre_traj']), abs(b['r2E_post'] - b['r2E_end_traj']))
        if run in RECS5:
            worst_s5 = max(worst_s5, abs((bo['r2E_pre_traj'] - bo['r2E_end_traj']) - RECS5[run]))
    print(f'GATE knee_dir_sweep.csv (old anchors, 6 dp: w, dR2, lam_share, cos_wbar): max |diff| {worst_old:.2e}')
    print(f'GATE knee_dir_sweep_avgr2.csv (new anchors, 6 dp: w, dR2, lam_share, cos_old, cos_new): max |diff| {worst_new:.2e}')
    print(f'GATE knee_s3_ladder.csv pooled_linear r2_E (old anchors, 6 dp): max |diff| {worst_lad:.2e}')
    print(f'GATE re-encoded pooled r2_E vs knee_traj_full ridge_r2_E (all anchors, 6 dp): max |diff| {worst_traj:.2e}')
    print(f'GATE 4x S5(ii) drops at old anchors vs printed (3 dp): max |diff| {worst_s5:.2e}')
    print(f'GATE mean-CCA contamination vs knee_dir_sweep.csv pre_contam (old anchors): max |diff| {worst_cc:.2e}')
    print(f'GATE avg-R2 contamination vs knee_dir_sweep_avgr2.csv pre_contam (new anchors): max |diff| {worst_ca:.2e}')

    # ---- per-run table
    print('\n== PER RUN (old anchor -> new anchor) ==')
    print(f'{"run":12s} {"anc":3s} {"pre":>4s} {"post":>4s} {"dR2(w)":>7s} {"w_E":>6s} {"pred":>7s} {"meas":>7s} '
          f'{"m-p":>8s} {"boot 90% CI":>19s} {"share":>6s} {"lam2":>8s} {"lam3":>8s} {"E pre":>6s} {"E post":>6s} '
          f'{"Edrop":>6s} {"Edrop_tr":>8s} {"c_cca":>6s} {"c_avg":>6s} {"cos_new":>7s}')
    rows = []
    for run in RUNS:
        for which in ('old', 'new'):
            pre, post = anchors[run][which]
            b = res[(run, pre, post)]
            exc = int(not (b['lo'] <= 0.0 <= b['hi']))
            ci = '[%+.4f,%+.4f]' % (b['lo'], b['hi'])
            print(f'{run.replace(P, ""):12s} {which:3s} {pre:4d} {post:4d} {b["dr2_w"]:7.3f} {float(b["w"][2]):6.3f} '
                  f'{b["pex"]:7.4f} {b["meas"]:7.4f} {b["meas"] - b["pex"]:+8.4f} '
                  f'{ci:>19s} {b["share"]:6.3f} {float(b["lam"][1]):+8.4f} '
                  f'{float(b["lam"][2]):+8.4f} {b["r2E_pre"]:6.3f} {b["r2E_post"]:6.3f} '
                  f'{b["r2E_pre"] - b["r2E_post"]:6.4f} {b["r2E_pre_traj"] - b["r2E_end_traj"]:8.4f} '
                  f'{b["c_cca"]:6.3f} {b["c_avg"]:6.3f} {abs(float(b["w"] @ wbar_new)):7.4f}')
            rows.append({'run': run, 'group': GROUP[run], 'anchor': which,
                         'pre_changed': int(anchors[run]['old'] != anchors[run]['new']),
                         'onset_avgr2': ons[run]['onset'], 'pre': pre, 'post': post,
                         'w_z1': f'{float(b["w"][0]):.6f}', 'w_z2': f'{float(b["w"][1]):.6f}',
                         'w_E': f'{float(b["w"][2]):.6f}', 'dR2_w': f'{b["dr2_w"]:.6f}',
                         'pred_exact': f'{b["pex"]:.6f}', 'pred_reduced': f'{b["pred"]:.6f}',
                         'measured': f'{b["meas"]:.6f}', 'meas_minus_pred': f'{b["meas"] - b["pex"]:.6f}',
                         'boot_lo': f'{b["lo"]:.6f}', 'boot_hi': f'{b["hi"]:.6f}', 'ci_excludes_0': exc,
                         'lam1': f'{float(b["lam"][0]):.6f}', 'lam2': f'{float(b["lam"][1]):.6f}',
                         'lam3': f'{float(b["lam"][2]):.6f}', 'lam1_share': f'{b["share"]:.6f}',
                         'r2E_pre_pooled': f'{b["r2E_pre"]:.6f}', 'r2E_post_pooled': f'{b["r2E_post"]:.6f}',
                         'r2E_drop_pooled': f'{b["r2E_pre"] - b["r2E_post"]:.6f}',
                         'r2E_pre_traj': f'{b["r2E_pre_traj"]:.6f}', 'r2E_end_traj': f'{b["r2E_end_traj"]:.6f}',
                         'r2E_drop_traj': f'{b["r2E_pre_traj"] - b["r2E_end_traj"]:.6f}',
                         'contam_meancca': f'{b["c_cca"]:.6f}', 'contam_avgr2': f'{b["c_avg"]:.6f}',
                         'cos_wbar_old': f'{abs(float(b["w"] @ wbar_old)):.6f}',
                         'cos_wbar_new': f'{abs(float(b["w"] @ wbar_new)):.6f}'})

    # ---- Table-1 quantities (raw ranges; no cut applied here)
    print('\n== TABLE-1 QUANTITIES (old anchor | new anchor) ==')
    for which in ('old', 'new'):
        B = {run: res[(run, *anchors[run][which])] for run in RUNS}
        sh = sorted((b['share'], run) for run, b in B.items())
        err = {run: b['meas'] - b['pex'] for run, b in B.items()}
        print(f'[{which}] rank-one share: min {sh[0][0]:.4f} ({sh[0][1].replace(P, "")}), '
              f'next {sh[1][0]:.4f} ({sh[1][1].replace(P, "")}), max {sh[-1][0]:.4f} ({sh[-1][1].replace(P, "")})')
        print(f'[{which}]   E drop (S9 measured) orig w2048: ' + ' '.join(f'{B[r]["meas"]:.4f}' for r in ORIG))
        print(f'[{which}]   E levels pooled orig w2048: ' + ', '.join(f'{B[r]["r2E_pre"]:.3f}->{B[r]["r2E_post"]:.3f}' for r in ORIG))
        w384 = sorted(RA6 + RBS)
        print(f'[{which}]   E drop traj (S5(ii)) w384: ' + ' '.join(
            f'{r.replace(P + "p384_", "s")}={B[r]["r2E_pre_traj"] - B[r]["r2E_end_traj"]:.3f}' for r in w384))
        print(f'[{which}]   |meas-pred|: max {max(abs(v) for v in err.values()):.4f}; orig max '
              f'{max(abs(err[r]) for r in ORIG):.4f}; RA6 max {max(abs(err[r]) for r in RA6):.4f}; supp '
              + ' '.join(f'{err[r]:+.4f}' for r in RBS))
        print(f'[{which}]   bootstrap CIs excluding zero: '
              f'{sum(not (b["lo"] <= 0 <= b["hi"]) for b in B.values())} of {len(B)}')

    with open(OUT, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    print(f'\nwrote {OUT} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
