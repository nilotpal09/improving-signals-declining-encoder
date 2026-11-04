"""Raw monitor values around the onset: declines at their onset, healthy seeds at epoch 100 (2026-10-10).

Each run counted once; healthy runs have no onset, so their zero is one fixed epoch, 100,
the median onset of the decline runs. Replaces the
borrowed-onset reference of knee_response_raw.py (there two healthy seeds supplied 17 of 25 readings).

Runs: the 6 recipes (settings: run name minus the trailing _seed and the _long_ marker) that have both
outcomes: 25 decline runs of knee_detector_chrono.KNEE and 10 held-out seeds without a decline
(DC.A2.HELD_21 + HELD_4).
Onsets: results/knee_onsets_avgr2.csv. Series: knee_response_auc.all_series (checkpoint grid; training-log
columns at grid epoch N from logged epoch N - 1).

rule      group    zero of each run
onset     decline  its onset
e100      healthy  epoch 100 (the figure)
e50..e160 healthy  that epoch (sensitivity; e100 repeats the figure)
sharpest  healthy  checkpoint e in [50, 160] maximising R(e - 10) - R(e), R = mean readability
all_onsets healthy every one of the 25 onsets, each healthy seed read at all of them (equal weight per seed)
Per (rule, series, offset): n values and 25th / 50th / 75th percentiles (numpy linear interpolation),
over runs with a checkpoint at zero + offset. Offsets -40..100 by 10.
Prints, per series, whether sign(decline median - healthy median) is the same under every healthy rule,
after the onset (+40..+100) and before it (-40..-10).
Writes results/knee_response_e100.csv (refuses to overwrite).
"""
import sys

sys.dont_write_bytecode = True

import csv  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402

import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
sys.path.insert(0, _HERE)
import knee_detector_chrono as DC  # noqa: E402
import knee_response_auc as K  # noqa: E402

OUT = os.path.join(paths.CSV, 'knee_response_e100.csv')
OFFSETS = tuple(range(-40, 101, 10))
FIXED = tuple(range(50, 161, 10))
SERIES = ('readability R2 average', 'loss', 'log lejepa_sigreg', 'log feat_std_event', 'rankme pub_uncentred',
          'rankme pub_centred', 'pooled lidar shared 1e-4', 'alpha_req centred', 'kappa centred', 'id_twonn mle',
          'self_cluster sq_uncentred')


def recipe(r):
    # Settings (width, mu, N), not run name: the 300-epoch long_ runs join the runs with their settings (2026-10-26).
    return re.sub(r'_long_', '_', re.sub(r'_(\d+)$', '', r))


def quart(v):
    return (len(v), *np.percentile(np.array(v, float), [25, 50, 75]))


def main():
    assert not os.path.exists(OUT), f'refusing to overwrite {OUT}'
    ons = {r['run']: int(r['onset']) for r in csv.DictReader(open(K.ONS)) if r['decline'] == '1'}
    held = DC.A2.HELD_21 + DC.A2.HELD_4
    both = {recipe(r) for r in DC.KNEE} & {recipe(r) for r in held}
    dec = sorted(r for r in DC.KNEE if recipe(r) in both)
    hea = sorted(r for r in held if recipe(r) in both)
    print('decline runs', len(dec), '| healthy seeds', len(hea), '| median onset', np.median([ons[d] for d in dec]))
    S = K.all_series(dec + hea)
    R = S['readability R2 average']
    sharp = {}
    for h in hea:
        cand = [e for e in FIXED if e in R[h] and e - 10 in R[h]]
        sharp[h] = max(cand, key=lambda e: R[h][e - 10] - R[h][e])
    zeros = {'e100': {h: [100] for h in hea}}
    zeros.update({f'e{e}': {h: [e] for h in hea} for e in FIXED})
    zeros['sharpest'] = {h: [sharp[h]] for h in hea}
    zeros['all_onsets'] = {h: [ons[d] for d in dec] for h in hea}
    rows, med = [], {}
    for key in SERIES:
        x = S[key]
        for k in OFFSETS:
            v = [x[d][ons[d] + k] for d in dec if ons[d] + k in x[d]]
            q = quart(v); med[('onset', key, k)] = q[2]
            rows.append(('onset', key, k, 'decline', *q))
            for rule, z in zeros.items():
                v = [x[h][e + k] for h in hea for e in z[h] if e + k in x[h]]
                q = quart(v); med[(rule, key, k)] = q[2]
                rows.append((rule, key, k, 'healthy', *q))
    with open(OUT, 'w', newline='') as fh:
        w = csv.writer(fh, lineterminator='\n')
        w.writerow(['rule', 'series', 'offset', 'group', 'n', 'q25', 'median', 'q75'])
        w.writerows([r[:5] + tuple(f'{a:.6g}' for a in r[5:]) for r in rows])
    print(f'wrote {OUT} ({len(rows)} rows)')
    print('n decline by offset:', {k: sum(1 for d in dec if ons[d] + k in R[d]) for k in OFFSETS})
    rules = list(zeros)
    for label, ks in (('after (+40..+100)', (40, 50, 60, 70, 80, 90, 100)), ('before (-40..-10)', (-40, -30, -20, -10))):
        print(f'\nsign of decline median - healthy median, {label}; e100 pattern, then rules that differ from it')
        for key in SERIES:
            pat = {r: ''.join('+' if med[('onset', key, k)] > med[(r, key, k)] else '-' for k in ks) for r in rules}
            off = [r for r in rules if pat[r] != pat['e100']]
            print(f'  {key:28s} {pat["e100"]}  ' + ('same under all rules' if not off else 'differ: ' + ' '.join(f'{r}={pat[r]}' for r in off)))


if __name__ == '__main__':
    main()
