"""Two-event split of rigd_lejp_p1s100_mu09_p384_e1000_1 (cosine s1) re-anchored at the average-R2 onset
(2026-10-10).

Record (post-hoc decomposition of this run): windows 70->1000 (the
anchored window), event A 70->300, event B 300->1000 and 300->500. Event A's start, 70, is the old anchor
(R_A onset 90 - 20); the split at 300 and the event-B windows were read off the trajectory (the plateau
between the two falls) and do not reference any onset rule. Under the vetted average-R2 rule
(results/knee_onsets_avgr2.csv) this run has onset 100, pre_onset_epoch 80, so the anchored window becomes
80->1000 and event A becomes 80->300; event B is unchanged.

Construction IMPORTED unchanged: knee_dir_e1000.direction (the S7(iv) construction on an explicit window:
ridge_resid + feats from knee_t2_direction), knee_dir_e1000.wbar_of and CACHE/CACHE_NAME, W_KNEE.
Reference directions: wbar_old from W_KNEE; wbar_new from the four reference runs' rows of
results/knee_dir_sweep_avgr2.csv (same definition as knee_dir_sweep_avgr2.py).
Gates: the four recorded windows vs the record (3-4 dp) and 70->1000 vs knee_dir_sweep.csv (6 dp);
80->1000 vs knee_dir_sweep_avgr2.csv (6 dp).
Also prints, per event-A start, how much of event A (start-of-window peak -> 300 plateau) is already
realised at the start, on mean CCA and on average ridge R2. Inference only, no training.
Writes results/knee_dir_e1000_split_avgr2.csv (refuses to overwrite).
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
import knee_dir_e1000 as e1k  # noqa: E402
from knee_dir_sweep_avgr2 import REF  # noqa: E402
from knee_onsets_avgr2 import load as load_readability  # noqa: E402
from knee_t2_direction import W_KNEE  # noqa: E402

CSV = os.path.join(paths.CSV)
RUN = 'rigd_lejp_p1s100_mu09_p384_e1000_1'
OUT = os.path.join(CSV, 'knee_dir_e1000_split_avgr2.csv')
# recorded windows (the record's post-hoc decomposition table): w, dR2, |cos wbar|, lam share
REC = {(70, 1000): ((0.854, -0.516, 0.073), 0.871, 0.8123, 0.524),
       (70, 300): ((0.457, -0.690, 0.561), 0.743, 0.9973, 0.989),
       (300, 1000): ((0.834, 0.006, -0.552), 0.827, 0.1117, 0.890),
       (300, 500): ((0.817, 0.036, -0.576), 0.963, 0.0701, 0.878)}
COLS = ['run', 'window', 'role', 'depends_on_onset', 'pre', 'post', 'w_z1', 'w_z2', 'w_E', 'dR2', 'lam_share',
        'cos_wbar_old', 'cos_wbar_new', 'eventA_realised_at_pre_meancca', 'eventA_realised_at_pre_avgr2']


def unit(v):
    return v / v.norm()


def main():
    assert not os.path.exists(OUT), f'refusing to overwrite {OUT}'
    print('device', drv.DEV, flush=True)
    ons = {r['run'].split('xxx')[0]: r for r in csv.DictReader(open(os.path.join(CSV, 'knee_onsets_avgr2.csv')))}[RUN]
    new_pre, last = int(ons['pre_onset_epoch']), int(ons['last_epoch'])
    old_row = {r['run']: r for r in csv.DictReader(open(os.path.join(CSV, 'knee_dir_sweep.csv')))}[RUN]
    sweep_new = {r['run']: r for r in csv.DictReader(open(os.path.join(CSV, 'knee_dir_sweep_avgr2.csv')))}
    old_pre = int(old_row['pre'])
    assert (old_pre, new_pre, last) == (70, 80, 1000), (old_pre, new_pre, last)
    print(f'onset rules: old R_A {old_row["onset"]} -> PRE {old_pre};  avg-R2 onset {ons["onset"]} -> PRE {new_pre}; '
          f'peak (avg R2) ep{ons["peak_epoch"]}')

    wk = {k: unit(torch.tensor(v, dtype=torch.float64)) for k, v in W_KNEE.items()}
    wbar_old = e1k.wbar_of(wk)
    wbar_new = unit(torch.stack([unit(torch.tensor([float(sweep_new[REF[k]][c]) for c in ('w_z1', 'w_z2', 'w_E')],
                                                   dtype=torch.float64)) for k in W_KNEE]).mean(0))
    print(f'wbar_old {[round(float(x), 4) for x in wbar_old]}  wbar_new {[round(float(x), 4) for x in wbar_new]}  '
          f'|cos| {abs(float(wbar_old @ wbar_new)):.5f}')

    d = glob.glob(os.path.join(drv.ROOT, RUN + 'xxx*'))
    assert len(d) == 1, d
    d = d[0]
    cfg = yaml.safe_load(open(os.path.join(d, 'config.yml')))
    sets = torch.load(os.path.join(e1k.CACHE, e1k.CACHE_NAME), map_location='cpu', weights_only=True)

    # trajectories for the event-A realised fraction
    cca = {}
    for r in csv.DictReader(open(os.path.join(CSV, 'knee_trajectories.csv'))):
        if r['run'].split('xxx')[0] == RUN:
            cca[int(r['epoch'])] = float(r['mean_cca'])
    R3, _ = load_readability()
    avg = {e: sum(v) / 3 for e, v in R3[RUN].items()}

    def realised(t, pre, split=300):
        pk = max((e for e in t if 20 <= e <= split), key=lambda e: t[e])     # event-A peak, before the plateau
        return (t[pk] - t[pre]) / (t[pk] - t[split]), pk

    windows = [((old_pre, last), 'anchored window (old)', 1), ((old_pre, 300), 'event A (old start)', 1),
               ((300, last), 'event B', 0), ((300, 500), 'event B (to 500)', 0),
               ((new_pre, last), 'anchored window (new)', 1), ((new_pre, 300), 'event A (new start)', 1)]
    rows, gate_rec, gate_csv = [], 0.0, 0.0
    print(f'\n{"window":10s} {"role":24s} {"w_z1":>7s} {"w_z2":>7s} {"w_E":>7s} {"dR2":>6s} {"lam":>6s} '
          f'{"cos_old":>7s} {"cos_new":>7s} {"A@pre cca":>9s} {"A@pre avg":>9s}')
    for (pre, post), role, dep in windows:
        w, dr2, cos_old, ls = e1k.direction(d, cfg, sets, pre, post, wbar_old)
        cos_new = abs(float(w @ wbar_new))
        ra_c = ra_a = float('nan')
        if post == 300:
            ra_c, pkc = realised(cca, pre)
            ra_a, pka = realised(avg, pre)
        if (pre, post) in REC:
            rw, rd, rc, rl = REC[(pre, post)]
            gate_rec = max(gate_rec, *(abs(float(w[i]) - rw[i]) for i in range(3)), abs(dr2 - rd), abs(cos_old - rc),
                           abs(ls - rl))
        if (pre, post) == (old_pre, last):
            o = old_row
            gate_csv = max(gate_csv, *(abs(float(w[i]) - float(o[c])) for i, c in enumerate(('w_z1', 'w_z2', 'w_E'))),
                           abs(dr2 - float(o['dR2'])), abs(cos_old - float(o['cos_wbar'])), abs(ls - float(o['lam_share'])))
        if (pre, post) == (new_pre, last):
            o = sweep_new[RUN]
            gate_csv = max(gate_csv, *(abs(float(w[i]) - float(o[c])) for i, c in enumerate(('w_z1', 'w_z2', 'w_E'))),
                           abs(dr2 - float(o['dR2'])), abs(cos_old - float(o['cos_wbar_old'])),
                           abs(cos_new - float(o['cos_wbar_new'])), abs(ls - float(o['lam_share'])))
        print(f'{f"{pre}->{post}":10s} {role:24s} {float(w[0]):7.3f} {float(w[1]):7.3f} {float(w[2]):7.3f} {dr2:6.3f} '
              f'{ls:6.3f} {cos_old:7.4f} {cos_new:7.4f} {ra_c:9.3f} {ra_a:9.3f}')
        rows.append({'run': RUN, 'window': f'{pre}->{post}', 'role': role, 'depends_on_onset': dep, 'pre': pre,
                     'post': post, 'w_z1': f'{float(w[0]):.6f}', 'w_z2': f'{float(w[1]):.6f}',
                     'w_E': f'{float(w[2]):.6f}', 'dR2': f'{dr2:.6f}', 'lam_share': f'{ls:.6f}',
                     'cos_wbar_old': f'{cos_old:.6f}', 'cos_wbar_new': f'{cos_new:.6f}',
                     'eventA_realised_at_pre_meancca': '' if post != 300 else f'{ra_c:.6f}',
                     'eventA_realised_at_pre_avgr2': '' if post != 300 else f'{ra_a:.6f}'})
    print(f'\nevent-A peak (20..300): mean CCA ep{pkc}, avg R2 ep{pka}')
    print(f'GATE recorded windows vs the record (3-4 dp): max |diff| {gate_rec:.2e} (rounding 5e-4 at 3 dp)')
    print(f'GATE 70->1000 vs knee_dir_sweep.csv and 80->1000 vs knee_dir_sweep_avgr2.csv (6 dp): max |diff| {gate_csv:.2e}')
    with open(OUT, 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=COLS, lineterminator='\n')
        wr.writeheader()
        wr.writerows(rows)
    print(f'wrote {OUT} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
