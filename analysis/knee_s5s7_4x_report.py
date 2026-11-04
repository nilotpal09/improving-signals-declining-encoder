"""S5(ii) energy-cost trajectory and S7(i) per-component CCA at 4x, from
csv/knee_traj_full.csv. Formulas/columns identical to the 21x report.
No GPU. Raw values only, no cut, no verdict."""
import csv
import os
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
T = defaultdict(dict)
for r in csv.DictReader(open(os.path.join(paths.CSV, 'knee_traj_full.csv'))):
    T[r['run'].split('xxx')[0]][int(r['epoch'])] = r

ONSET = {f'rigb_lejp_p1s100_p384_{s}': o for s, o in
         ((1, 50), (2, 140), (3, 90), (4, 140), (5, 60), (6, 70),
          (7, 80), (8, 90), (9, 120))}
CTRL = [f'rigd_lejp_p1s100_mu097_p384_{s}' for s in (1, 2, 3, 4)]
FLAT = ['rigb_lejp_p1s100_p384_10']


def at(run, ep, col):
    t = T[run]
    return float(t[min(t, key=lambda k: abs(k - ep))][col])


print('== S7(i) 4x: per-component CCA across the knee ==')
print(f"{'run':22s} {'ep':>4s} {'cca1':>6s} {'cca2':>6s} {'cca3':>6s} {'mean':>6s} "
      f"{'r2_z1':>6s} {'r2_z2':>6s} {'r2_E':>6s} {'orth':>6s}")
for run, on in ONSET.items():
    tag = run.replace('rigb_lejp_p1s100_', '')
    for lbl, ep in (('pre', on - 20), ('on', on), ('post', on + 20), ('end', max(T[run]))):
        print(f'{tag:22s} {ep:4d} ' + ' '.join(
            f'{at(run, ep, c):6.3f}' for c in
            ('cca_c1', 'cca_c2', 'cca_c3', 'mean_cca', 'ridge_r2_z1',
             'ridge_r2_z2', 'ridge_r2_E', 'orth_err')))

print('\ncontrols + flat run (same columns, at ep80/150/300):')
for run in CTRL + FLAT:
    tag = run.replace('rigd_lejp_p1s100_', '').replace('rigb_lejp_p1s100_', '')
    for ep in (80, 150, 300):
        print(f'{tag:22s} {ep:4d} ' + ' '.join(
            f'{at(run, ep, c):6.3f}' for c in
            ('cca_c1', 'cca_c2', 'cca_c3', 'mean_cca', 'ridge_r2_z1',
             'ridge_r2_z2', 'ridge_r2_E', 'orth_err')))

print('\n== S5(ii) 4x: ridge R2(E) trajectory ==')
print(f"{'run':22s} {'on-20':>7s} {'onset':>7s} {'on+20':>7s} {'end':>7s} {'drop':>7s}")
drops = []
for run, on in ONSET.items():
    tag = run.replace('rigb_lejp_p1s100_', '')
    v = [at(run, e, 'ridge_r2_E') for e in (on - 20, on, on + 20, max(T[run]))]
    drops.append(v[0] - v[3])
    print(f'{tag:22s} ' + ' '.join(f'{x:7.3f}' for x in v) + f' {v[0]-v[3]:7.3f}')
print('controls + flat run:')
for run in CTRL + FLAT:
    tag = run.replace('rigd_lejp_p1s100_', '').replace('rigb_lejp_p1s100_', '')
    v = [at(run, e, 'ridge_r2_E') for e in (80, 150, 250, max(T[run]))]
    print(f'{tag:22s} ' + ' '.join(f'{x:7.3f}' for x in v) + f' {v[0]-v[3]:7.3f}')
print(f'\nE drop (onset-20 -> end) over the 9 knee runs, sorted: '
      + ' '.join(f'{d:.3f}' for d in sorted(drops)))

print('\ncca3 pre->end per knee run (the 21x drop sat entirely in component 3):')
for run, on in ONSET.items():
    tag = run.replace('rigb_lejp_p1s100_', '')
    print(f'  {tag:22s} c1 {at(run,on-20,"cca_c1"):.3f}->{at(run,max(T[run]),"cca_c1"):.3f}   '
          f'c2 {at(run,on-20,"cca_c2"):.3f}->{at(run,max(T[run]),"cca_c2"):.3f}   '
          f'c3 {at(run,on-20,"cca_c3"):.3f}->{at(run,max(T[run]),"cca_c3"):.3f}')
