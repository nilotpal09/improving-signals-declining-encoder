"""Trajectory numbers the paper quotes, recomputed on the new readability
measure (2026-10-10): average held-out ridge R2 over z1, z2 and log E, with
the decline label, peak, drop and onset of the vetted average-R2 rule
(results/knee_onsets_avgr2.csv; trajectories from knee_onsets_avgr2.load()).

Items (old definitions found in the record, applied unchanged to the new
measure; each is first run on mean CCA and checked against the old numbers):
  1 Rebound (the record's rebound structure, 2026-08-20): trough =
    global minimum after the run's peak; rebound = max after the trough minus
    the trough (0 if the trough is the last checkpoint). Paper (Appendix C):
    "Ten of twelve width-2048 decline runs rebounded from their trough by at
    most 0.033, remaining below their earlier peak". Old peak on mean CCA =
    global max; new peak = the rule's peak (largest average R2, epoch >= 20).
  2 Final SIGReg vs drop from peak (the record's P1 result, 2026-09-02):
    final-epoch mean of sigreg_eval (and lejepa_sigreg) in lejepa_diag_r0.csv,
    read with knee_s2_monitors.series (imported); Spearman (scipy, average
    ranks) of level against drop within four cohorts: mu=0 width 384 (10),
    mu=0 width 2048 (10, incl. the three stopped at 150 and the long run),
    mu dial width 384 (13 = mu 0.4/0.7/0.9 seeds 1-3 + mu 0.97 seeds 1-4),
    mu dial width 2048 (12 = mu 0.4/0.7/0.9 seeds 1-3 + mu 0.97 seeds 1-3).
    The cohort membership is the one that reproduces the record's four rho.
    Also the dial cohorts with the six-seed extension (seeds 1-6).
  3 Occurrence table (Appendix D, macros \\occurrenceSummaryA..Q): runs and
    drop-from-peak range per condition; old drop = mean-CCA global max minus
    final; new drop = the rule's drop on average R2. Ranges printed at 4
    decimals and at 2 significant figures of the 4-decimal values, as the
    macros were made.
  4 Median drop at mu=0 with a seed bootstrap: numpy default_rng(0), 10,000
    resamples of the runs, median of each resample, 2.5 and 97.5 percentiles
    (the procedure recorded in the earlier analysis, 2026-08-19,
    as reproducing [0.0959, 0.1189] and [0.0890, 0.1128] from the printed
    per-run values; checked here).
  5 Decline counts per condition (new 'decline' column) against the counts
    the paper quotes, and against the mean-CCA large/small drop split.
  6 Onset range: paper "50--170" (\\eventOnsetRange, mu=0 runs of both
    widths; old onset = R_A, else R_B, columns old_RA/RB_meancca).

Writes NEW files (refuses to overwrite):
  results/knee_trajectory_numbers_avgr2_runs.csv     one row per run (131)
  results/knee_trajectory_numbers_avgr2_summary.csv  one row per quoted number
"""
import sys

sys.dont_write_bytecode = True

import csv  # noqa: E402
import glob  # noqa: E402
import os  # noqa: E402
from decimal import Decimal  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
sys.path.insert(0, _HERE)
import knee_onsets_avgr2 as KO  # noqa: E402
import knee_s2_monitors as S2  # noqa: E402

CSV = os.path.join(paths.CSV)
ONS = os.path.join(CSV, 'knee_onsets_avgr2.csv')
OUT_RUNS = os.path.join(CSV, 'knee_trajectory_numbers_avgr2_runs.csv')
OUT_SUM = os.path.join(CSV, 'knee_trajectory_numbers_avgr2_summary.csv')

B, L = 'rigb_lejp_p1s100_', 'rigd_lejp_p1s100_'


def seeds(prefix, ss):
    return [f'{prefix}{s}' for s in ss]


# (row, condition, width, epochs, runs); rows A-Q are the occurrence table's.
COND = [
    ('A', 'Gaussian mu=0', 384, 300, seeds(B + 'p384_', range(1, 11))),
    ('B', 'Gaussian mu=0', 2048, 300, seeds(B + 'nano_', range(4, 10)) + [B + 'long_nano_1']
     + seeds(B + 'nano_', (10, 11, 12))),  # + denominator runs (2026-10-12)
    ('B150', 'Gaussian mu=0, stopped at 150', 2048, 150, seeds(B + 'nano_', (1, 2, 3))),
    ('C', 'Mixture mu=0.4', 384, 300, seeds(L + 'mu04_p384_', range(1, 7))),
    ('D', 'Mixture mu=0.4', 2048, 300, seeds(L + 'mu04_nano_', range(1, 7))),
    ('E', 'Mixture mu=0.7', 384, 300, seeds(L + 'mu07_p384_', range(1, 7))),
    ('F', 'Mixture mu=0.7', 2048, 300, seeds(L + 'mu07_nano_', range(1, 7))),
    ('G', 'Mixture mu=0.9', 384, 300, seeds(L + 'mu09_p384_', range(1, 7))),
    ('H', 'Mixture mu=0.9', 2048, 300, seeds(L + 'mu09_nano_', range(1, 7))),
    ('I', 'Mixture mu=0.97', 384, 300, seeds(L + 'mu097_p384_', range(1, 7))),  # denominator runs: + 5-6
    ('J', 'Mixture mu=0.97', 2048, 300, [L + 'mu097_long_nano_1']
     + seeds(L + 'mu097_nano_', (4, 5, 6, 7, 8))),  # + denominator runs
    ('J150', 'Mixture mu=0.97, stopped at 150', 2048, 150, seeds(L + 'mu097_nano_', (1, 2, 3))),
    ('K', 'Mixture mu=0.9, cosine LR', 384, 1000, seeds(L + 'mu09_p384_e1000_', (1, 2, 3))),
    ('L', 'Mixture mu=0.9, constant LR', 384, 1000, seeds(L + 'mu09_p384_e1000_constlr_', (1, 2, 3))),
    ('M', 'Mixture mu=0.97, cosine LR', 384, 1000, seeds(L + 'mu097_p384_e1000_', (1, 2, 3))),
    ('N', 'Gaussian mu=0, rho=0.9', 384, 300, seeds(B + 'p384_ou09_', (1, 2, 3, 4, 5, 6))),  # denominator runs: + 4-6
    ('O', 'Gaussian mu=0, rho=0.5', 384, 300, seeds(B + 'p384_ou05_', (1, 2, 3, 4, 5, 6))),  # denominator runs: + 4-6
    ('P', 'Mixture mu=0.97, rho=0.9', 384, 300, seeds(L + 'mu097_p384_ou09_', (1, 2, 3))),
    ('Q', 'Gaussian mu=0, rho=0.9', 384, 1000, seeds(B + 'p384_ou09_e1000_', range(1, 7))),
    ('N30', 'Gaussian mu=0, N=30', 384, 300, seeds('rigb_lejp_p1s30_p384_', range(1, 7))),
    ('N300', 'Gaussian mu=0, N=300', 384, 300, seeds('rigb_lejp_p1s300_p384_', range(1, 7))),
    ('N1000', 'Gaussian mu=0, N=1000', 384, 300, seeds('rigb_lejp_p1s1000_p384_', (1, 2, 3))),
    ('CLR', 'Gaussian mu=0, constant LR', 384, 300, seeds(B + 'p384_constlr_', (1, 2, 3))),
    ('WD0', 'Gaussian mu=0, no weight decay', 384, 300, seeds(B + 'p384_wd0_', (1, 2, 3))),
    ('LAM', 'Gaussian mu=0, lambda=0.05', 384, 300, seeds(B + 'p384_lam005_', range(1, 11))),
    ('TINY', 'Gaussian mu=0, ViT-Tiny', 2048, 300, seeds(B + 'tiny_', (1, 2))),
    ('MAE', 'MAE, Gaussian mu=0', 0, 300, seeds('rigb_mae_p1s100_long_nano_', range(1, 11))),
]
# The denominator runs added above; the record gates (OLD_DROPS, OLD_MACRO) cover the original runs only.
DENOM = set(seeds(B + 'nano_', (10, 11, 12)) + seeds(L + 'mu097_nano_', (4, 5, 6, 7, 8))
             + seeds(L + 'mu097_p384_', (5, 6))
            + seeds(B + 'p384_ou09_', (4, 5, 6)) + seeds(B + 'p384_ou05_', (4, 5, 6)))

# Old printed per-run mean-CCA drops, sorted (tab:occurrence and tab:extended in
# the paper's appendix; seeds 4-6 from the record's six-seed extension;
# J150 from the tab:occurrence caption).
OLD_DROPS = {
    'A': [0.0000, 0.0773, 0.1058, 0.1145, 0.1150, 0.1151, 0.1170, 0.1189, 0.1196, 0.1402],
    'B': [0.0004, 0.0890, 0.0916, 0.0977, 0.1126, 0.1128, 0.1206],
    'C': [0.1074, 0.1137, 0.1192, 0.1200, 0.1216, 0.1272],
    'D': [0.0000, 0.0007, 0.0009, 0.0036, 0.1091, 0.1250],
    'E': [0.0000, 0.0969, 0.1123, 0.1164, 0.1193, 0.1674],
    'F': [0.0037, 0.0765, 0.1033, 0.1048, 0.1133, 0.1231],
    'G': [0.0000, 0.0001, 0.0001, 0.0001, 0.0998, 0.1105],
    'H': [0.0000, 0.0006, 0.0016, 0.0067, 0.0109, 0.1053],
    'I': [0.0000, 0.0051, 0.0156, 0.0226],
    'J': [0.0039],
    'J150': [0.0000, 0.0000, 0.0143],
    'K': [0.1193, 0.1363, 0.3766],
    'L': [0.1250, 0.1425, 0.2156],
    'M': [0.0027, 0.0034, 0.0191],
    'N': [0.0001, 0.0004, 0.0007],
    'O': [0.0000, 0.0000, 0.0004],
    'P': [0.0019, 0.0111, 0.0119],
    'Q': [0.0001, 0.0011, 0.0011, 0.0023, 0.0033, 0.0059],
    'N30': [0.0062, 0.0098, 0.0170, 0.1402, 0.1576, 0.1808],
    'N300': [0.0002, 0.0003, 0.0008, 0.0015, 0.0018, 0.0026],
}
OLD_MACRO = {'A': '0--0.14', 'B': '0.0004--0.12', 'C': '0.11--0.13', 'D': '0--0.12', 'E': '0--0.17',
             'F': '0.0037--0.12', 'G': '0--0.11', 'H': '0--0.11', 'I': '0--0.023', 'J': '0.0039',
             'K': '0.12--0.38', 'L': '0.12--0.22', 'M': '0.0027--0.019', 'N': '0.0001--0.0007',
             'O': '0--0.0004', 'P': '0.0019--0.012', 'Q': '0.0001--0.0059'}
# Old rebound table (the record): run -> (trough, epoch, rebound), mean CCA.
OLD_REBOUND = {
    B + 'nano_1': (0.864, 150, 0.000), B + 'nano_2': (0.895, 150, 0.000), B + 'nano_3': (0.889, 130, 0.005),
    B + 'long_nano_1': (0.880, 210, 0.007), B + 'nano_7': (0.883, 200, 0.007), B + 'nano_9': (0.872, 180, 0.011),
    L + 'mu04_nano_3': (0.868, 150, 0.014), B + 'nano_4': (0.867, 190, 0.017), B + 'nano_6': (0.863, 170, 0.017),
    B + 'nano_8': (0.840, 140, 0.033), L + 'mu07_nano_1': (0.876, 280, 0.001), L + 'mu07_nano_2': (0.871, 270, 0.002),
    B + 'p384_1': (0.889, 50, 0.040)}
OLD12 = [k for k in OLD_REBOUND if '_p384_' not in k]
# Old P1 cohorts and rho (the record's P1 result): (rho eval, rho train).
DIAL = ('mu04', 'mu07', 'mu09')
COHORTS = {
    'mu=0, width 384': (seeds(B + 'p384_', range(1, 11)), (-0.673, -0.527)),
    'mu=0, width 2048': (seeds(B + 'nano_', range(1, 10)) + [B + 'long_nano_1'], (-0.139, -0.273)),
    'mu dial, width 384': ([f'{L}{m}_p384_{s}' for m in DIAL for s in (1, 2, 3)]
                           + seeds(L + 'mu097_p384_', range(1, 5)), (-0.492, -0.376)),
    'mu dial, width 2048': ([f'{L}{m}_nano_{s}' for m in DIAL for s in (1, 2, 3)]
                            + seeds(L + 'mu097_nano_', (1, 2, 3)), (-0.560, -0.539)),
}
COHORTS6 = {
    'mu dial, width 384, six seeds': [f'{L}{m}_p384_{s}' for m in DIAL for s in range(1, 7)]
    + seeds(L + 'mu097_p384_', range(1, 5)),
    'mu dial, width 2048, six seeds': [f'{L}{m}_nano_{s}' for m in DIAL for s in range(1, 7)]
    + seeds(L + 'mu097_nano_', (1, 2, 3)),
}
OLD_BOOT = {'A': (0.1150, 0.0959, 0.1189), 'B': (0.0977, 0.0890, 0.1128)}
# Decline counts the paper quotes: row -> (old count text, source).
PAPER_COUNTS = {
    'A': ('9 of 10', 'Sec. 3: "In nine of ten default runs"'),
    'B': ('6 of 7', 'Sec. 3: "in six of the seven runs reaching 300 epochs"'),
    'B150': ('3 of 3', 'Sec. 3: "plus three that declined before stopping at epoch 150"'),
    'C': ('6 of 6', 'Sec. 3.2 comment: mu=0.4 6/6 at 384'),
    'D': ('2 of 6', 'Sec. 3.2 comment: 2/6 at 2048; "Only at mu=0.4 did the two widths differ"'),
    'E': ('5 of 6', 'Sec. 3.2 comment: mu=0.7 5/6 at both'),
    'F': ('5 of 6', 'same'),
    'G': ('2 of 6', 'Sec. 3.2: "declines were uncommon within 300 epochs" (comment 2/6 and 1/6)'),
    'H': ('1 of 6', 'same'),
    'K': ('3 of 3', 'Sec. 3.2: "all six longer runs declined"'),
    'L': ('3 of 3', 'same'),
    'I': ('0 of 4', 'Sec. 3.2: "At mu=0.97, runs stayed close to their peaks"'),
    'J': ('0 of 1', 'same'), 'J150': ('0 of 3', 'App. D: the three controls did not'),
    'M': ('0 of 3', 'Sec. 3.2: "even at the longer horizon"'),
    'N': ('0 of 3', 'Sec. 3.2: "all three seeds per refresh condition ... remained close to their peaks"'),
    'O': ('0 of 3', 'same'),
    'P': ('0 of 3', 'Sec. 3.2: "partial shape refresh also produced small drops there"'),
    'Q': ('0 of 6', 'Sec. 3.2: "Six refresh runs extended to 1000 epochs also stayed near their peaks"'),
    'MAE': ('0 of 10', 'Sec. 3.2: "all ten runs ended at their peak, although the onset rule fired during '
                       'warm-up in two"'),
    'LAM': ('3 of 10', 'Sec. 3.2: "retained the decline in three of ten runs" (seeds 2, 6, 8)'),
    'N30': ('3 of 6', 'Sec. 3.2 comment: three declined (0.14-0.18)'),
    'N300': ('0 of 6', 'Sec. 3.2: "At N=300 the decline did not appear"'),
    'CLR': ('2 of 3', 'App. H (commented out): two of three constant-rate runs decline'),
    'WD0': ('1 of 3', 'App. H (commented out): one of three runs declines'),
}


def short(k):
    return k.replace(B, '').replace(L, '').replace('rigb_', '')


def r4(x):
    return round(x + 0.0, 4) + 0.0


def sf2(x):
    v = r4(x)
    if v == 0:
        return '0'
    return format(Decimal(f'{v:.2g}'), 'f')


def rng_str(vals, f):
    lo, hi = min(vals), max(vals)
    return f(lo) if f(lo) == f(hi) and len(vals) == 1 else f'{f(lo)}--{f(hi)}'


def peak_drop_cca(t):
    pk = max(sorted(t), key=lambda e: t[e])
    return pk, t[pk], t[max(t)], t[pk] - t[max(t)]


def rebound(t, pk):
    after = [e for e in sorted(t) if e > pk]
    tr = min(after, key=lambda e: t[e])
    post = [e for e in sorted(t) if e > tr]
    mx = max(t[e] for e in post) if post else t[tr]
    return tr, t[tr], mx - t[tr], mx


def boot(x):
    x = np.asarray(x, float)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(x), size=(10000, len(x)))
    lo, hi = np.percentile(np.median(x[idx], axis=1), [2.5, 97.5])
    return float(np.median(x)), float(lo), float(hi)


def sigreg_final(run):
    hits = glob.glob(os.path.join(S2.ROOT, run + 'xxx*'))
    assert len(hits) == 1, (run, hits)
    if not os.path.exists(os.path.join(hits[0], 'lejepa_diag_r0.csv')):
        return None, None
    s = S2.series(run + 'xxx*')
    return s['sigreg_eval'][1][-1], s['lejepa_sigreg'][1][-1]


def main():
    for p in (OUT_RUNS, OUT_SUM):
        assert not os.path.exists(p), f'refusing to overwrite {p}'
    R, M = KO.load()
    ons = {r['run']: r for r in csv.DictReader(open(ONS))}
    cond_of = {k: c for c in COND for k in c[4]}
    assert sorted(cond_of) == sorted(R) == sorted(ons), 'conditions must cover all runs exactly'

    # ---------------------------------------------------------------- per run
    runs = {}
    worst_rule = 0.0
    for k in sorted(R):
        avg, pk, fin, drop, dip, tr_, dec, on, pre = KO.label(R[k])
        o = ons[k]
        assert (pk, fin, int(dec), on) == (int(o['peak_epoch']), int(o['last_epoch']), int(o['decline']),
                                          int(o['onset']) if o['onset'] else None), k
        worst_rule = max(worst_rule, abs(drop - float(o['drop'])))
        cpk, cpv, cfin, cdrop = peak_drop_cca(M[k])
        row = dict(run=k, row=cond_of[k][0], condition=cond_of[k][1], width=cond_of[k][2],
                   horizon=cond_of[k][3], last_epoch=fin, decline_new=int(dec), onset_new=on if on else '',
                   old_onset_RA_else_RB=o['old_RA_meancca'] or o['old_RB_meancca'],
                   peak_epoch_avg=pk, avg_peak=avg[pk], avg_final=avg[fin], drop_avg=drop,
                   peak_epoch_cca=cpk, cca_peak=cpv, cca_final=cfin, drop_cca=cdrop)
        for tag, t, p in (('avg', avg, pk), ('cca', M[k], cpk)):
            if p < max(t):
                trough, tv, rb, mx = rebound(t, p)
                row.update({f'trough_epoch_{tag}': trough, f'trough_{tag}': tv, f'rebound_{tag}': rb,
                            f'post_trough_max_{tag}': mx, f'post_trough_below_peak_{tag}': int(mx < t[p])})
            else:
                row.update({f'trough_epoch_{tag}': '', f'trough_{tag}': '', f'rebound_{tag}': '',
                            f'post_trough_max_{tag}': '', f'post_trough_below_peak_{tag}': ''})
        se, st = sigreg_final(k) if 'mae' not in k else (None, None)
        row.update(sigreg_eval_final=se if se is not None else '', sigreg_train_final=st if st is not None else '')
        runs[k] = row
    print(f'rule check: knee_onsets_avgr2.label() reproduces peak, last epoch, decline and onset of the CSV for all '
          f'{len(runs)} runs; max |drop diff| {worst_rule:.1e}')
    assert worst_rule < 1e-6

    summary = []

    def add(item, subject, n, old, new, note=''):
        summary.append(dict(item=item, subject=subject, n=n, old=old, new=new, note=note))

    # ---------------------------------------------------------------- 3 occurrence (gate first)
    print('\n=== GATE occurrence: mean-CCA drop (global max - final), rounded to 4 decimals, vs printed values ===')
    worst, nbad = 0.0, 0
    for row, *_ in COND:
        if row not in OLD_DROPS:
            continue
        got = sorted(r4(runs[k]['drop_cca']) for k in cond_of if cond_of[k][0] == row and k not in DENOM)
        exp = OLD_DROPS[row]
        d = max(abs(a - b) for a, b in zip(got, exp)) if len(got) == len(exp) else float('inf')
        worst = max(worst, d)
        nbad += d > 5e-5
        if d > 5e-5:
            print(f'   {row}: got {got} expected {exp}')
    b150 = [runs[k]['drop_cca'] for k in cond_of if cond_of[k][0] == 'B150']
    print(f'GATE occurrence: {len(OLD_DROPS)} conditions, {sum(len(v) for v in OLD_DROPS.values())} runs: max |diff| '
          f'{worst:.1e}; conditions differing {nbad}; B150 drops at stopping all > 0.089: {min(b150) > 0.089} '
          f'(min {min(b150):.4f})')
    macro_ok = all(rng_str([r4(runs[k]['drop_cca']) for k in cond_of if cond_of[k][0] == row and k not in DENOM], sf2)
                   == OLD_MACRO[row]
                   for row in OLD_MACRO)
    print(f'GATE occurrence: 2-significant-figure ranges from mean CCA equal all 17 \\occurrenceSummary macros: {macro_ok}')
    assert worst < 5e-5 and nbad == 0 and macro_ok

    print('\n=== occurrence table on average ridge R2 (rule drop), with decline counts ===')
    print(f'{"row":5s} {"condition":34s} {"width":>5s} {"ep":>5s} {"runs":>4s} | {"old range (mean CCA)":>20s} | '
          f'{"new range (avg R2), 4 dp":>24s} | {"2 s.f.":>14s} | declines old -> new')
    for row, cond, width, ep, rs in COND:
        da = [runs[k]['drop_avg'] for k in rs]
        dc = [runs[k]['drop_cca'] for k in rs]
        ndec = sum(runs[k]['decline_new'] for k in rs)
        old_c = PAPER_COUNTS.get(row, ('-', ''))[0]
        new4 = rng_str(da, lambda x: f'{r4(x):.4f}')
        new2 = rng_str(da, sf2)
        print(f'{row:5s} {cond:34s} {width:>5d} {ep:>5d} {len(rs):>4d} | {rng_str(dc, sf2):>20s} | {new4:>24s} | '
              f'{new2:>14s} | {old_c} -> {ndec} of {len(rs)}')
        if row in OLD_MACRO:
            add('occurrence drop range', f'{row}: {cond}, width {width}, {ep} epochs', len(rs),
                f'{OLD_MACRO[row]} (mean CCA)', f'{new2} (avg R2; 4 dp {new4})',
                'macros_tables.tex \\occurrenceSummary' + row)
        if row in PAPER_COUNTS:
            add('decline count', f'{row}: {cond}, width {width}, {ep} epochs', len(rs), PAPER_COUNTS[row][0],
                f'{ndec} of {len(rs)} ('
                + (', '.join(short(k) for k in rs if runs[k]['decline_new']) or 'none') + ')', PAPER_COUNTS[row][1])

    # mean-CCA large/small split vs new label, all runs
    dec = [runs[k]['drop_cca'] for k in runs if runs[k]['decline_new']]
    non = [runs[k]['drop_cca'] for k in runs if not runs[k]['decline_new']]
    print(f'\nmean-CCA drop of new decline runs: min {min(dec):.4f}; of runs without a decline: max {max(non):.4f} '
          f'-> the new label equals a mean-CCA large-drop split for every run')
    add('decline label vs mean CCA', f'all {len(runs)} runs', len(runs), 'mean-CCA drops: large >= 0.0548, small <= 0.0232',
        f'new decline runs have mean-CCA drop >= {min(dec):.4f}, the others <= {max(non):.4f}',
        'every new decline run sits above every non-decline run in mean-CCA drop')

    # ---------------------------------------------------------------- 4 bootstrap
    print('\n=== GATE bootstrap on the printed per-run values ===')
    for row in ('A', 'B'):
        m, lo, hi = boot(OLD_DROPS[row])
        print(f'{row}: median {m:.4f} [{lo:.4f}, {hi:.4f}] vs recorded {OLD_BOOT[row]}')
        assert (f'{m:.4f}', f'{lo:.4f}', f'{hi:.4f}') == tuple(f'{v:.4f}' for v in OLD_BOOT[row])
    for row in ('A', 'B'):
        rs = [k for k in cond_of if cond_of[k][0] == row]
        m, lo, hi = boot([runs[k]['drop_avg'] for k in rs])
        mc, loc, hic = boot([runs[k]['drop_cca'] for k in rs])
        print(f'{row} new (avg R2): median {m:.4f} [{lo:.4f}, {hi:.4f}], {len(rs)} runs; mean CCA from the trajectories '
              f'{mc:.4f} [{loc:.4f}, {hic:.4f}]')
        o = OLD_BOOT[row]
        add('median drop at mu=0, seed bootstrap', f'width {384 if row == "A" else 2048}, {len(rs)} runs', len(rs),
            f'{o[0]:.4f} [{o[1]:.4f}, {o[2]:.4f}] (mean CCA)', f'{m:.4f} [{lo:.4f}, {hi:.4f}] (avg R2)',
            'default_rng(0), 10,000 resamples, 2.5/97.5 percentiles')

    # ---------------------------------------------------------------- 1 rebound
    print('\n=== GATE rebound on mean CCA vs the recorded table ===')
    worst = 0.0
    for k, (tv, te, rb) in OLD_REBOUND.items():
        r = runs[k]
        worst = max(worst, abs(round(r['trough_cca'], 3) - tv), abs(round(r['rebound_cca'], 3) - rb))
        assert r['trough_epoch_cca'] == te, (k, r['trough_epoch_cca'], te)
    print(f'GATE rebound: 13 runs, trough epochs equal; max |diff| of trough and rebound (3 dp) {worst:.1e}')
    assert worst < 5e-4
    for tag, sel in (('the 12 runs of the old table', OLD12),
                     ('all width-2048 ViT-Nano decline runs (new rule)',
                      [k for k in runs if runs[k]['decline_new'] and runs[k]['width'] == 2048
                       and 'tiny' not in k])):
        rb = sorted((runs[k]['rebound_avg'], short(k)) for k in sel)
        rbc = sorted((runs[k]['rebound_cca'], short(k)) for k in sel)
        nz = sum(v > 0 for v, _ in rb)
        below = sum(runs[k]['post_trough_below_peak_avg'] for k in sel)
        print(f'{tag} ({len(sel)}): avg R2 rebound > 0 in {nz}, max {rb[-1][0]:.3f} ({rb[-1][1]}); post-trough max '
              f'below peak in {below} of {len(sel)}; mean-CCA: > 0 in {sum(v > 0 for v, _ in rbc)}, max '
              f'{rbc[-1][0]:.3f} ({rbc[-1][1]})')
        print('   avg R2 rebounds sorted: ' + ', '.join(f'{v:.3f} {n}' for v, n in rb))
        add('rebound after trough', f'width 2048, {tag}', len(sel),
            f'{sum(v > 0 for v, _ in rbc)} of {len(sel)} nonzero, max {rbc[-1][0]:.3f} ({rbc[-1][1]}) (mean CCA)'
            + (' [paper: ten of twelve, at most 0.033]' if sel is OLD12 else ''),
            f'{nz} of {len(sel)} nonzero, max {rb[-1][0]:.3f} ({rb[-1][1]}); below peak {below} of {len(sel)} (avg R2)',
            'trough = global min after peak; rebound = max after trough - trough')
    r1 = runs[B + 'p384_1']
    print(f'width-384 mu=0 seed 1 (old 0.040, the largest): avg R2 rebound {r1["rebound_avg"]:.3f} at trough epoch '
          f'{r1["trough_epoch_avg"]}')
    all_dec = sorted((runs[k]['rebound_avg'], short(k)) for k in runs if runs[k]['decline_new'])
    print(f'largest avg-R2 rebound over all {len(all_dec)} decline runs: ' + ', '.join(f'{v:.3f} {n}'
                                                                                   for v, n in all_dec[-5:]))
    add('rebound after trough', 'width 384, mu=0 seed 1', 1, '0.040 (mean CCA; the largest)',
        f'{r1["rebound_avg"]:.3f} (avg R2)', '')

    # ---------------------------------------------------------------- 2 SIGReg vs drop
    print('\n=== GATE final SIGReg vs mean-CCA drop (Spearman) vs the recorded P1 result ===')
    for c, (rs, (oe, ot)) in COHORTS.items():
        se = [runs[k]['sigreg_eval_final'] for k in rs]
        st = [runs[k]['sigreg_train_final'] for k in rs]
        ge = spearmanr(se, [runs[k]['drop_cca'] for k in rs])[0]
        gt = spearmanr(st, [runs[k]['drop_cca'] for k in rs])[0]
        print(f'{c:22s} n={len(rs):2d}: eval {ge:+.3f} (record {oe:+.3f}), train {gt:+.3f} (record {ot:+.3f}); '
              f'eval level {min(se):.3f}-{max(se):.3f}')
        assert (f'{ge:.3f}', f'{gt:.3f}') == (f'{oe:.3f}', f'{ot:.3f}')
    print('new: drop from peak on average R2')
    for c, (rs, (oe, ot)) in list(COHORTS.items()) + [(c, (rs, (None, None))) for c, rs in COHORTS6.items()]:
        se = [runs[k]['sigreg_eval_final'] for k in rs]
        st = [runs[k]['sigreg_train_final'] for k in rs]
        ne = spearmanr(se, [runs[k]['drop_avg'] for k in rs])[0]
        nt = spearmanr(st, [runs[k]['drop_avg'] for k in rs])[0]
        ce = spearmanr(se, [runs[k]['drop_cca'] for k in rs])[0]
        print(f'{c:32s} n={len(rs):2d}: eval {ne:+.3f}, train {nt:+.3f}' + (f' (old eval {oe:+.3f})' if oe else
                                                                         f' (mean CCA eval {ce:+.3f})'))
        add('final SIGReg vs drop, Spearman', c, len(rs),
            (f'eval {oe:+.3f}, train {ot:+.3f} (mean CCA)' if oe is not None else
             f'eval {ce:+.3f} (mean CCA, not in the record)'),
            f'eval {ne:+.3f}, train {nt:+.3f} (avg R2)', 'paper: -0.14 to -0.67 (eval) over the four cohorts'
            if oe is not None else 'six-seed extension; not in the paper')

    # ---------------------------------------------------------------- 6 onset range
    print('\n=== onset range ===')
    mu0 = [k for k in runs if runs[k]['row'] in ('A', 'B', 'B150') and runs[k]['decline_new']]
    old_on = [int(runs[k]['old_onset_RA_else_RB']) for k in mu0]
    new_on = [runs[k]['onset_new'] for k in mu0]
    print(f'GATE: old onsets (R_A else R_B, mean CCA) of the {len(mu0)} mu=0 decline runs: {min(old_on)}-{max(old_on)} '
          f'(paper 50--170)')
    assert (min(old_on), max(old_on)) == (50, 170)
    for tag, rr in (('mu=0, width 384', ('A',)), ('mu=0, width 2048 (incl. 3 stopped at 150)', ('B', 'B150'))):
        sel = [k for k in mu0 if runs[k]['row'] in rr]
        print(f'  {tag}: old {min(int(runs[k]["old_onset_RA_else_RB"]) for k in sel)}-'
              f'{max(int(runs[k]["old_onset_RA_else_RB"]) for k in sel)}, new {min(runs[k]["onset_new"] for k in sel)}-'
              f'{max(runs[k]["onset_new"] for k in sel)}')
    print(f'mu=0 both widths: new {min(new_on)}-{max(new_on)}')
    add('onset range', 'mu=0 decline runs, both widths (\\eventOnsetRange)', len(mu0), f'{min(old_on)}--{max(old_on)}',
        f'{min(new_on)}--{max(new_on)}', 'old: R_A else R_B on mean CCA; new: average-R2 rule')
    dec300 = [k for k in runs if runs[k]['decline_new'] and runs[k]['horizon'] == 300 and 'mae' not in k]
    v = [runs[k]['onset_new'] for k in dec300]
    vo = [int(runs[k]['old_onset_RA_else_RB']) for k in dec300 if runs[k]['old_onset_RA_else_RB']]
    print(f'all decline runs with a 300-epoch horizon ({len(dec300)}): new {min(v)}-{max(v)}; old {min(vo)}-{max(vo)}')
    add('onset range', 'all decline runs, 300-epoch horizon', len(dec300), f'{min(vo)}--{max(vo)}', f'{min(v)}--{max(v)}',
        'not quoted in the paper; for reference')
    dec_all = [k for k in runs if runs[k]['decline_new']]
    v = [runs[k]['onset_new'] for k in dec_all]
    print(f'all {len(dec_all)} decline runs: new {min(v)}-{max(v)}')

    # MAE extras
    mae = [k for k in runs if runs[k]['row'] == 'MAE']
    at_peak = sum(runs[k]['peak_epoch_avg'] == runs[k]['last_epoch'] for k in mae)
    rb_mae = [k for k in mae if ons[k]['old_RB_meancca']]
    print(f'\nMAE: peak at the final checkpoint in {at_peak} of {len(mae)} (avg R2); old R_B onsets in warm-up: '
          + ', '.join(f'{short(k)} at {ons[k]["old_RB_meancca"]}' for k in rb_mae)
          + '; the new rule defines onsets only for decline runs, so it cannot fire in MAE')
    add('MAE ended at peak', 'MAE', len(mae), '10 of 10 at peak; onset rule fired in warm-up in 2',
        f'{at_peak} of {len(mae)} at peak (avg R2); new rule gives no onset (no decline)',
        'the "fired during warm-up" clause has no counterpart under the new rule')

    # ---------------------------------------------------------------- write
    cols = list(next(iter(runs.values())))
    for path, rows, cc in ((OUT_RUNS, list(runs.values()), cols), (OUT_SUM, summary, list(summary[0]))):
        part = path + '.partial'
        assert not os.path.exists(part)
        with open(part, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=cc, lineterminator='\n')
            w.writeheader()
            for r in rows:
                w.writerow({c: (f'{v:.6f}' if isinstance(v, float) else v) for c, v in r.items()})
        os.replace(part, path)
        print(f'wrote {path}: {len(rows)} rows')


if __name__ == '__main__':
    main()
