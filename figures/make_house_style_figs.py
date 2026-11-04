"""Monitor paper Figures 2-4 (event, comparisons, audit) in the testbed house style.

Cosmetic port (2026-09-26) of the installed figures to the testbed house style.
The plotted numbers are unchanged: every value comes from the manifest that
the installed figures were built from (results_install_2026-09-26), or, for
the audit figure, from the same macros_tables rows. A self-check at the end
recomputes a sample of them from the source CSVs.

2026-10-09: readability switched to average ridge R2 of z1, z2,
log E and onsets to the average-R2 rule, via
avgr2_data.py: Fig 2(a) traces, 2(b) onset ticks, 2(d) anchors ("pre-onset"),
Fig 3 drops, peak/final and untrained floors, plus the 0.07 decline cutoff in
3(a, b). Loss, rank and per-factor traces are unchanged.

2026-10-20: data now come from CSVs in results/ only (loss:
knee_default_loss.csv; RankMe, per-factor R2, onsets and audit counts: the
CSVs that include the denominator-completion runs), not from the 09-26
manifest or the macros. Same values.

Installed 2026-09-26. Writes PNG and PDF to FIGURE_OUTPUT_DIR
(default: ../figs). Run from anywhere:
  FIGURE_OUTPUT_DIR=/some/dir python -B make_house_style_figs.py
"""
import csv, json, os, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
import tempfile
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'mplcache_monitor_figs'))
sys.path.insert(0, str(HERE))
import numpy as np
import _style as S
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import ScaledTranslation

ROOT = HERE.parent   # figures -> repository
# 2026-10-20: every figure here reads CSVs only. The 09-26 figure
# manifest (JSON; its 10-06 copy below) is no longer an input; when present, the
# self-check confirms the CSV values equal it. Figure 6 reads the detector
# summary CSVs instead of the macros_tables.tex rows, and the self-check
# confirms the paper's rows equal the CSV.
MAN_PATH = Path(os.environ.get('MONITOR_NOTES', ROOT / 'notes')) / 'figure_sources_seedext_2026-10-06.json'
MACROS = Path(os.environ.get('PAPER_DIR', ROOT / 'paper')) / 'macros_tables.tex'   # the paper source, if present
CHRONO = ROOT / 'results'   # knee_detector_chrono_avgr2(_logged).py
# Fig 2 highlighted seed (2026-10-26): seed 3, the median of the nine default declines in both
# drop (0.215) and onset (epoch 70), results/knee_onsets_avgr2.csv. Seed 9 (the earlier choice,
# the most extreme loss spike and RankMe dip at onset) stays the run checked against the 09-26
# manifest below.
REP = 'rigb_lejp_p1s100_p384_3'
MAN_REP = 'rigb_lejp_p1s100_p384_9'
REP_SEED = REP.rsplit('_', 1)[1]
sys.path.insert(0, str(ROOT / 'analysis'))
import avgr2_data as A  # readability (average R2), onsets, floors (2026-10-09)
OUT = Path(os.environ.get('FIGURE_OUTPUT_DIR', ROOT / 'figs'))
OUT.mkdir(parents=True, exist_ok=True)

# LeJEPA is this paper's method: the blue family. Variants are shades of it.
W384, W2048, REFRESH = S.BLUE, S.ORANGE, S.AQUA
PEAK = '#7b3fb5'   # Fig 3(c) peak crosses, purple
SHADES = {'ridge_r2_z1': S.ORANGE, 'ridge_r2_z2': S.AQUA, 'ridge_r2_E': '#7b3fb5'}   # Fig 2(d) factors (not all bluish)
FIG_W = 11.2
YTOP, CUT = 0.32, 0.07   # Fig 3(a, b) axis top; decline cutoff (average R2)


def save(fig, name):
    S.report(fig, name, FIG_W)
    fig.savefig(OUT / f'{name}.png', dpi=220, facecolor=S.SURFACE)
    fig.savefig(OUT / f'{name}.pdf', facecolor=S.SURFACE)
    plt.close(fig)


def rowgaps(fig, pairs):
    fig.canvas.draw(); r = fig.canvas.get_renderer(); fw = fig.get_size_inches()[0]
    g = [(q.get_tightbbox(r).x0 - p.get_tightbbox(r).x1) / fig.dpi * 6.5 / fw for p, q in pairs]
    print('  row gaps (in, on page):', ' '.join(f'{x:.3f}' for x in g))


def hollow(ax, x, y, color, marker='o', size=30, z=3):
    ax.scatter(x, y, s=S.fs(size), marker=marker, facecolors=S.SURFACE,
               edgecolors=color, linewidths=S.f(1.3), zorder=z)


# ----------------------------------------------------------------- Figure 2
def event():
    S.apply(FIG_W)
    # 2026-10-26: one row of four panels instead of 2x2, to save space; same data.
    fig, axs = plt.subplots(1, 4, figsize=(FIG_W, 3.9), sharex=True,
                            gridspec_kw=dict(wspace=0.70))
    a, b, c, d = axs.flat
    for k in A.TEN:
        hi = k == REP
        kw = dict(color=W384, lw=S.f(2.1 if hi else 0.9), alpha=1.0 if hi else 0.38,
                  zorder=4 if hi else 2)
        t = A.AVG[k]; ea = sorted(t)
        a.plot(ea, [t[e] for e in ea], **kw)
        b.plot(*A.loss_trace(k), **kw)
        c.plot(*A.rankme_trace(k), **kw)
    for k in A.NINE:
        o = A.onset(k)
        hi = k == REP
        b.plot([o, o], [0.93, 1.0], transform=b.get_xaxis_transform(),
               color=W384 if hi else S.INK2, lw=S.f(1.6 if hi else 1.0), zorder=5)
    e, targets = A.factor_traces(REP)
    for key, lab in (('ridge_r2_z1', '$z_1$'), ('ridge_r2_z2', '$z_2$'), ('ridge_r2_E', r'$\log E$')):
        y = targets[key]
        d.plot(e, y, color=SHADES[key], lw=S.f(1.8), zorder=3)
        nudge = {'ridge_r2_z1': 0.012, 'ridge_r2_E': -0.012}.get(key, 0.0)   # z1 and log E end 0.04 apart
        d.text(e[-1] + 4, y[-1] + nudge, lab, color=SHADES[key], va='center', ha='left',
               fontsize=S.f(9.5))
    pre, on = int(A.ONS[REP]['pre_onset_epoch']), A.onset(REP)
    for x, lab, ha, dx in ((pre, 'pre-onset', 'right', -3), (on, 'onset', 'left', 3)):
        d.axvline(x, color=S.MUTED, lw=S.f(0.9), zorder=1)
        d.text(x + dx, 0.47, lab, ha=ha, va='bottom', rotation=90, color=S.INK2,
               fontsize=S.f(8.8))   # vertical: the one-row panel is too narrow for flat labels
    a.set(ylabel='readability (average $R^2$)', ylim=(0.60, 1.01))
    b.set(ylabel='training loss', ylim=(0, 0.27))
    c.set(ylabel='RankMe', ylim=(0, 9))
    d.set(ylabel='held-out ridge $R^2$', ylim=(0.44, 1.01))
    for ax in axs.flat:
        ax.set_xlabel('training epoch')
        ax.set_xlim(0, 300)
        ax.set_xticks([0, 100, 200, 300])
        S.tidy(ax)
    S.panel_label(a, 'a', 'readability, ten seeds')
    S.panel_label(b, 'b', 'training loss')
    S.panel_label(c, 'c', 'RankMe')
    S.panel_label(d, 'd', f'each factor, seed {REP_SEED}')
    handles = [Line2D([], [], color=W384, lw=S.f(0.9), alpha=0.5, label='one seed'),
               Line2D([], [], color=W384, lw=S.f(2.1), label=f'seed {REP_SEED}'),
               Line2D([], [], color=S.INK2, lw=0, marker='|', markersize=S.f(9),
                      markeredgewidth=S.f(1.2), label='estimated onset')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, S.LEGEND_Y),
               ncol=3, frameon=False, columnspacing=2.0, handletextpad=0.6)
    fig.subplots_adjust(left=0.06, right=0.955, top=0.86, bottom=0.30)
    # The y labels differ in width, so place the panels by their drawn extents:
    # equal widths and S.PANEL_GAP_IN between neighbours on the page.
    for _ in range(3):   # repeat: tick labels can shift once the widths change
        fig.canvas.draw(); r = fig.canvas.get_renderer()
        boxes = [(ax.get_position(), ax.get_tightbbox(r)) for ax in axs.flat]
        lpad = [p.x0 * FIG_W - t.x0 / fig.dpi for p, t in boxes]
        rpad = [t.x1 / fig.dpi - p.x1 * FIG_W for p, t in boxes]
        gap, x = 0.15 * FIG_W / 6.5, 0.01 * FIG_W   # 0.15 in on the page, tighter than the house 0.24
        w = (0.98 * FIG_W - 3 * gap - sum(lpad) - sum(rpad)) / 4
        for ax, lp, rp in zip(axs.flat, lpad, rpad):
            p = ax.get_position()
            ax.set_position([(x + lp) / FIG_W, p.y0, w / FIG_W, p.height])
            x += lp + w + rp + gap
    rowgaps(fig, [(a, b), (b, c), (c, d)])
    save(fig, 'fig_results_event')


# ----------------------------------------------------------------- Figure 3
# 2026-10-12: redrawn as one bar per condition, the share of runs that
# declined (the vetted rule's decline flag, drop >= 0.07), count printed on each
# bar; default first, then one group per change. Per-run drops, the 1000-epoch
# refresh runs, N=1000 and the untrained floors moved to Appendix D.
# The canvas is CMP_FRAC of the house width and goes in at CMP_FRAC\linewidth,
# so S.apply(FIG_W) keeps text and marks the same size on the page.
CMP_FRAC = 0.66
CMP_COL = {384: '#1f4e8c', 2048: '#9cc0e6'}   # dark and light blue
CMP_BAND = '#f3f2ef'                          # backing for the default column


def declined(runs):
    return sum(A.ONS[k]['decline'] == '1' for k in runs), len(runs)


def comparisons():
    S.apply(FIG_W)
    occ = lambda m: [(384, A.OCCURRENCE[(m, 384)]), (2048, A.OCCURRENCE[(m, 2048)])]
    groups = [
        ('default settings\n$\\rho=1$, $\\mu=0$\n$N=100$', [('', occ(0.0))]),
        ("views' shape\ncorrelation, $\\rho$",
         [('0.9', [(384, A.REFRESH[(0.9, 300)])]), ('0.5', [(384, A.REFRESH[(0.5, 300)])])]),
        ('shape\ndistribution, $\\mu$', [(f'{m:g}', occ(m)) for m in (0.4, 0.7, 0.9, 0.97)]),
        ('elements per\nchannel, $N$\n(fewer = grainier)',
         [('30', [(384, A.RUNGS[30])]), ('300', [(384, A.RUNGS[300])])]),
    ]
    bw, step1, step2, gap, gap0, pad_r = 0.22, 0.36, 0.66, 0.5, 0.7, 0.45
    fig, ax = plt.subplots(figsize=(FIG_W * CMP_FRAC, 3.8))
    x, ticks, labels, spans = 0.0, [], [], []
    for gi, (head, cols) in enumerate(groups):
        x0 = None
        for lab, sets in cols:
            step = step2 if len(sets) == 2 else step1
            x += step / 2
            x0 = x if x0 is None else x0
            offs = [-bw / 2 - 0.012, bw / 2 + 0.012] if len(sets) == 2 else [0.0]
            for (w, runs), o in zip(sets, offs):
                d, n = declined(runs)
                # a zero keeps a thin stub, so "none" reads as a bar of height 0
                ax.bar(x + o, max(d / n, 0.012), width=bw, color=CMP_COL[w], zorder=2)
                nudge = 0.035 * (1 if o > 0 else -1 if o < 0 else 0)   # paired labels apart
                ax.text(x + o + nudge, d / n + 0.03, f'{d}/{n}', ha='center', va='bottom',
                        color=S.INK, fontsize=S.f(7.6))
            ticks.append(x); labels.append(lab)
            last, x = x, x + step / 2
        spans.append((x0, last, head, len(cols[0][1])))
        x += gap0 if gi == 0 else gap
    ax.axvspan(spans[0][0] - step2 / 2, spans[0][0] + step2 / 2, color=CMP_BAND, zorder=0, lw=0)
    # group descriptions below the tick labels, under a hairline spanning the group
    below = ax.get_xaxis_transform() + ScaledTranslation(0, -S.f(24) / 72, fig.dpi_scale_trans)
    for a0, a1, head, nb in spans:
        ext = bw * (1.0 if nb == 2 else 0.5) + 0.03
        ax.plot([a0 - ext, a1 + ext], [0, 0], color=S.INK2, lw=S.f(0.9), transform=below,
                clip_on=False)
        ax.annotate(head, xy=((a0 + a1) / 2, 0), xycoords=('data', 'axes fraction'),
                    xytext=(0, -S.f(29)), textcoords='offset points', ha='center', va='top',
                    color=S.INK, fontsize=S.f(9.5), linespacing=1.2, annotation_clip=False)
    ax.set_xticks(ticks, labels)
    ax.set_yticks([0, 0.5, 1], ['none', 'half', 'all'])
    ax.set(ylabel='runs that declined', ylim=(0, 1.1), xlim=(-0.15, x - gap + pad_r))
    S.tidy(ax)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.legend(handles=[Patch(color=CMP_COL[384], label='default projector'),
                       Patch(color=CMP_COL[2048], label='wide projector')],
              loc='upper right', bbox_to_anchor=(1.0, 1.02), frameon=False,
              handlelength=1.0, handleheight=1.0, labelspacing=0.5, borderaxespad=0)
    fig.subplots_adjust(left=0.11, right=0.995, top=0.95, bottom=0.31)
    for head, cols in groups:
        for lab, sets in cols:
            print(f'  Fig comparisons {head.split(chr(10))[0]:22s} {lab:4s}',
                  ' '.join(f'w{w} {"/".join(map(str, declined(r)))}' for w, r in sets))
    S.report(fig, 'fig_results_comparisons', FIG_W * CMP_FRAC)
    fig.savefig(OUT / 'fig_results_comparisons.png', dpi=220, facecolor=S.SURFACE)
    fig.savefig(OUT / 'fig_results_comparisons.pdf', facecolor=S.SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------- Figure 4
def chrono(name):
    Q = {}
    for r in csv.DictReader(open(CHRONO / name)):
        Q[(r['series'], r['design'], r['test'], r['campaign'], r['population'], r['quantity'])] = r
    return Q


def cell(Q, s, d, t, pop):
    """(timely count, scorable) on pop and (false alarms, scorable) on held-out runs,
    as the paper's audit rows report them."""
    a = Q[(s, d, t, 'total', pop, 'timely')]
    h = Q[(s, d, t, 'total', 'heldout', 'false_alarms')]
    return [(int(float(a['count'])), int(float(a['scorable']))),
            (int(float(h['count'])), int(float(h['scorable'])))]


def audit_row(Q, s, t):
    """standing matched25 | D2 main21 | D1 matched25: the integratedRow layout."""
    return (cell(Q, s, 'standing_matched', t, 'matched25') + cell(Q, s, 'D2_matched', t, 'main21')
            + cell(Q, s, 'D1_matched', t, 'matched25'))


AUDIT_ROWS = [('A', 'loss', 'CUSUM'), ('B', 'rankme pub_uncentred', 'CUSUM'),
              ('C', 'alpha_req centred', 'CUSUM'), ('D', 'kappa centred', 'CUSUM'),
              ('E', 'kappa centred', 'EWMA'), ('F', 'id_twonn mle', 'PageHinkley'),
              ('G', 'pooled lidar shared 1e-4', 'CUSUM')]


def audit():
    S.apply(FIG_W)
    Q = chrono('knee_detector_chrono_avgr2_summary.csv')
    rows = [audit_row(Q, s, t) for _, s, t in AUDIT_ROWS]
    # Last row: LiDAR on the shape-resampled groups of Section 5,
    # delta = 1e-4 (initial CUSUM; control-referenced CUSUM), as in Appendix G's
    # table. The two restarted-design slots are unused here.
    s = 'pooled lidar redrawn 1e-4'
    rows.append(cell(Q, s, 'standing_matched', 'CUSUM', 'matched25') + [None, None]
                + cell(Q, s, 'D1_matched', 'CUSUM', 'matched25'))
    # SIGReg row, second from the top: training-log SIGReg under EWMA,
    # one of the two best timely counts in Section 3.
    rows.insert(1, audit_row(chrono('knee_detector_chrono_avgr2_logged_summary.csv'),
                             'log lejepa_sigreg', 'EWMA'))
    labels = ['loss · CUSUM', 'SIGReg · EWMA', 'RankMe · CUSUM', r'$\alpha$-ReQ · CUSUM',
              r'$\kappa$, centred · CUSUM', r'$\kappa$, centred · EWMA', 'TwoNN · Page–Hinkley',
              'LiDAR · CUSUM', 'LiDAR, groups redraw shape · CUSUM']
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 5.0), sharey=True, gridspec_kw=dict(wspace=0.29))
    y = np.arange(len(rows))
    for ax, p in zip(axes, (0, 1)):
        for yi, r in zip(y, rows):
            v0 = 100 * r[0 + p][0] / r[0 + p][1]     # initial procedure
            v1 = 100 * r[4 + p][0] / r[4 + p][1]     # control-referenced check
            ax.plot([v0, v1], [yi, yi], color=S.MUTED, lw=S.f(0.9), zorder=2)
            hollow(ax, v0, yi, W384, size=40)
            ax.scatter(v1, yi, s=S.fs(40), marker='x', color=S.ORANGE, zorder=4, linewidths=S.f(1.5))
        ax.axhline(len(rows) - 1.5, color=S.MUTED, lw=S.f(0.6), zorder=1)   # sets the last row apart
        ax.set_xlim(-3, 103); ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel('runs (%)')
        S.tidy(ax, yaxis_grid=False); ax.grid(axis='x', zorder=0)
        ax.tick_params(axis='y', length=0)
    axes[0].set_yticks(y, labels); axes[0].invert_yaxis()
    S.panel_label(axes[0], 'a', 'timely first alarms, decline runs')
    S.panel_label(axes[1], 'b', 'alarms, runs without a decline')
    handles = [Line2D([], [], ls='none', marker='o', markerfacecolor=S.SURFACE, markeredgecolor=W384, markeredgewidth=S.f(1.3), markersize=S.f(7), label='initial procedure'),
               Line2D([], [], ls='none', marker='x', color=S.ORANGE, markersize=S.f(7), markeredgewidth=S.f(1.5), label='control-referenced check')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, S.LEGEND_Y),
               ncol=2, frameon=False, columnspacing=2.5, handletextpad=0.5)
    fig.subplots_adjust(left=0.325, right=0.955, top=0.895, bottom=0.215)
    save(fig, 'fig_audit_comparison')
    return rows


# --------------------------------------------------------------- self-check
def check(audit_rows):
    print('\n--- self-check against source files ---')
    # The paper's audit rows (macros_tables.tex) must equal the CSV rows plotted here.
    if not MACROS.exists():
        print(f'  paper source not present ({MACROS}); set PAPER_DIR to check Fig 6 against it')
    else:
        text = MACROS.read_text()
        def macro(name):
            line = next(l for l in text.splitlines() if l.startswith('\\newcommand{\\' + name + '}'))
            return [tuple(map(int, x)) for x in re.findall(r'(\d+)/(\d+)', line.split(' % ')[0])]
        names = ['integratedRowA', 'auditRowSIGReg'] + [f'integratedRow{k}' for k in 'BCDEFG']
        for n, r in zip(names, audit_rows):
            assert macro(n) == r, (n, macro(n), r)
        fb = macro('integratedRowFB')
        assert [fb[0], fb[1], None, None, fb[2], fb[3]] == audit_rows[-1], (fb, audit_rows[-1])
        print(f'  Fig 6 rows equal the paper\'s macros_tables.tex rows: {len(audit_rows)}/{len(audit_rows)}')
    if MAN_PATH.exists():   # migration check against the retired 09-26 manifest
        MAN = json.load(open(MAN_PATH))
        A.check(MAN)
        ev = MAN['fig_results_event']; T = ev['traces']
        assert list(T) == A.TEN
        for k, t in T.items():
            assert A.loss_trace(k) == (t['loss_epochs'], t['loss']), k
            assert A.rankme_trace(k) == (t['rank_epochs'], t['rankme_pub_uncentred']), k
        rep = ev['representative']
        assert rep['run'] == MAN_REP and A.factor_traces(MAN_REP) == (rep['epochs'], rep['targets'])
        print(f'  Fig 2 loss, RankMe and seed-9 traces equal the old manifest exactly ({len(T)} runs)')
    for lab, r in zip(['A', 'SIGReg', 'B', 'C', 'D', 'E', 'F', 'G'], audit_rows):
        print(f'  Fig 4 row {lab}: initial timely {r[0][0]}/{r[0][1]} held-out {r[1][0]}/{r[1][1]};'
              f' control-referenced timely {r[4][0]}/{r[4][1]} held-out {r[5][0]}/{r[5][1]}')


if __name__ == '__main__':
    event(); comparisons(); rows = audit(); check(rows)
