"""Draft figure (2026-10-10): the monitors around the known decline, raw values, each run once.
Readability (needs labels) first, then eleven label-free signals, one panel each, including all seven survey
families (RankMe as published, uncentred; centred RankMe is in the appendix version). Orange: runs that declined, x = epochs from
their onset. Grey: healthy seeds of the same recipes, x = epochs from epoch 100 (the median onset).
Line: median; band: interquartile range. Sources: results/knee_response_e100.csv
(knee_response_e100.py) and results/knee_response_e100_ext.csv (NE-Sum, DSE; knee_response_e100_ext.py), rules
onset (decline) and e100 (healthy); kappa uncentred from results/knee_response_e100_kappa.csv. Each family as the survey
computed it, except Self-Cluster (its printed form is constant on these runs; squared form). NE-Sum centred and DSE author-code normalization; the other recorded
variants of both give the same post-onset ordering. LiDAR on shape groups is left out
(it uses the simulator's groups). Writes fig_response_e100.{png,pdf}.
"""
import csv, os, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'mplcache_monitor_figs'))
sys.path.insert(0, str(HERE))
import _style as S
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

ROOT = HERE.parent
# 2026-10-19: the 147-run recompute (with the denominator-completion runs) replaces
# the earlier CSVs. Healthy runs stay at epoch 100. 2026-10-26: runs grouped by settings
# (knee_response_e100.recipe), 25 decline runs, median onset 100.
CSVD = ROOT / 'results'
SRC = [CSVD / 'knee_response_e100.csv', CSVD / 'knee_response_e100_ext.csv', CSVD / 'knee_response_e100_kappa.csv']
OUT = Path(os.environ.get('FIGURE_OUTPUT_DIR', ROOT / 'figs')); OUT.mkdir(parents=True, exist_ok=True)
FIG_W = 11.2
DEC, REF = S.ORANGE, S.INK2
TINT = '#eef4fb'  # house blue at about 8% on white
UP, DOWN, ONE = 'prefer higher', 'prefer lower', 'prefer closest to 1'
PANELS = [('readability R2 average', 'readability', 'needs labels', 1),
          ('loss', 'training loss', DOWN, 1), ('log lejepa_sigreg', 'SIGReg', DOWN, 1),
          ('log feat_std_event', 'feature std.', UP, 1), ('pooled lidar shared 1e-4', 'LiDAR', UP, 1),
          ('rankme pub_uncentred', 'RankMe', UP, 1),
          ('alpha_req centred', r'$\alpha$-ReQ', ONE, 1), ('ne_sum centred', 'NE-Sum', UP, 1),
          ('kappa uncentred', r'$\kappa$ ($\times 10^6$)', UP, 1e-6), ('self_cluster sq_uncentred', 'Self-Cluster', DOWN, 1),
          ('dse code', 'DSE', UP, 1), ('id_twonn mle', 'TwoNN', DOWN, 1)]


def load():
    D = {}
    for r in (r for f in SRC for r in csv.DictReader(open(f))):
        if (r['rule'], r['group']) in (('onset', 'decline'), ('e100', 'healthy')):
            D.setdefault((r['series'], r['group']), {})[int(r['offset'])] = (
                float(r['q25']), float(r['median']), float(r['q75']), int(r['n']))
    return {k: [(o, *v[o]) for o in sorted(v)] for k, v in D.items()}


def panel(ax, D, key, title, pref, scale):
    for grp, col, z in (('healthy', REF, 2), ('decline', DEC, 3)):
        t = D[(key, grp)]; k = [a[0] for a in t]
        ax.fill_between(k, [scale * a[1] for a in t], [scale * a[3] for a in t], color=col, alpha=0.18, lw=0, zorder=z - 1)
        ax.plot(k, [scale * a[2] for a in t], color=col, lw=S.f(1.8), zorder=z, solid_capstyle='round')
    ax.axvline(0, color=S.MUTED, lw=S.f(0.9), zorder=0)
    ax.set_title(title, fontsize=S.f(9.5), color=S.INK, pad=S.f(15))
    ax.text(0.5, 1.025, pref, transform=ax.transAxes, ha='center', va='bottom', fontsize=S.f(9.5), color=S.MUTED)
    ax.set_xlim(-42, 102); ax.set_xticks([0, 50, 100])
    ax.yaxis.set_major_locator(MaxNLocator(3))
    S.tidy(ax)


def main():
    D = load()
    S.apply(FIG_W)
    fig = plt.figure(figsize=(FIG_W, 5.6))
    gs = fig.add_gridspec(2, 6, wspace=0.55, hspace=0.70, left=0.055, right=0.985, top=0.79, bottom=0.115)
    first = None
    for i, (key, title, pref, scale) in enumerate(PANELS):
        ax = fig.add_subplot(gs[divmod(i, 6)], sharex=first); first = first or ax
        panel(ax, D, key, title, pref, scale)
        if key == 'readability R2 average':  # the reference: faint tint and a thin full frame
            ax.set_ylim(0.70, 0.99); ax.set_yticks([0.75, 0.85, 0.95])
            ax.set_facecolor(TINT)
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color(S.MUTED); sp.set_linewidth(S.f(0.8))
        if i < 6:
            plt.setp(ax.get_xticklabels(), visible=False)
    fig.text(0.5, 0.035, 'epochs from the onset (runs that did not decline: from epoch 100)', ha='center', va='center',
             fontsize=S.f(10), color=S.INK)
    nd = max(a[4] for a in D[('readability R2 average', 'decline')])
    nh = max(a[4] for a in D[('readability R2 average', 'healthy')])
    handles = [Line2D([], [], color=DEC, lw=S.f(1.8), label=f'declined ({nd} runs)'),
               Line2D([], [], color=REF, lw=S.f(1.8), label=f'did not decline ({nh} runs)'),
               Patch(facecolor=S.MUTED, alpha=0.3, edgecolor='none', label='line: median, band: IQR')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, S.LEGEND_Y), ncol=3,
               frameon=False, columnspacing=1.6, handletextpad=0.5)
    S.report(fig, 'fig_response_e100', FIG_W)
    fig.savefig(OUT / 'fig_response_e100.png', dpi=220, facecolor=S.SURFACE)
    fig.savefig(OUT / 'fig_response_e100.pdf', facecolor=S.SURFACE)
    print('wrote', OUT / 'fig_response_e100.png')


if __name__ == '__main__':
    main()
