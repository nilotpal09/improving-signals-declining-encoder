"""Figure 1 (fig_overview): the training pair, what we watch, and the aligned decline.

One left-to-right figure:
  (a) a real training pair (two renders of the same shape and energy with
      different sampled elements; rendered by the training collator, same
      event as the generator appendix figure) feeding one encoder, whose two
      read-outs point at the plots they produce: label-free signals (grey)
      and probes of the known factors (blue).
  (b) the nine declining width-384 runs at mu = 0, aligned at each run's
      onset: readability drops while the training loss keeps its downward
      trend and effective rank rises.

Values: loss traces from results/knee_default_loss.csv and rank traces
from knee_labelfree_metrics.csv, via avgr2_data.py (2026-10-20; until then read
from the 09-26 figure manifest, whose values they equal); readability (average ridge R2 of z1, z2, log E) and onsets (average-R2
rule) from avgr2_data.py (2026-10-09). Checkpoint-grid series
(readability, rank) use the relative epochs where all nine runs have a
checkpoint (multiples of 10); the per-epoch loss uses every epoch.
No new measurement. The training pair is read from training_pair_seed20260825.npz
(rendered on CPU by the training collator; render_pair() regenerates it if missing).
Writes PNG and PDF to FIGURE_OUTPUT_DIR (default ../figs). The canvas is trimmed
on the right to SAVE_W, so include it at SAVE_W / FIG_W of \\linewidth (0.93).
"""
import os, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'mplcache_monitor_figs'))
sys.path.insert(0, str(HERE))
import numpy as np
import _style as S
import matplotlib.pyplot as plt
import cmocean
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'analysis'))
import avgr2_data as A  # readability (average R2) and onsets (2026-10-09)
OUT = Path(os.environ.get('FIGURE_OUTPUT_DIR', ROOT / 'figs')); OUT.mkdir(parents=True, exist_ok=True)
PAIR_FILE = HERE / 'training_pair_seed20260825.npz'


def render_pair():
    """The appendix generator figure's event (seed 20260825) as its two training
    views at N = 100, through the training collator's own view construction."""
    import glob, torch, yaml
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path.insert(0, str(ROOT / 'train'))
    from src.lejepa_data import LeJEPALatentCollator
    from src.datasets.calo_flash import CaloNorm
    run = glob.glob(os.path.join(os.environ.get('MONITOR_DATA', str(ROOT / 'data')), 'runs', 'rigb_lejp_p1s100_p384_1xxx*'))[0]
    cfg = yaml.safe_load(open(os.path.join(run, 'config.yml')))
    coll = LeJEPALatentCollator(base_collator=lambda x: x, device='cpu', calo_cfg=dict(cfg['data']['calo_cfg']),
                                generator_cfg=dict(cfg['data']['generator_cfg']),
                                norm=CaloNorm(**(cfg['data'].get('norm') or {})))
    coll._build_backend()
    seed = 20260825
    p_E, p_x, p_y = coll._generate_truth(1, seed)
    long_seed = (seed * 7919 + 31337) & 0x7FFFFFFF; zseed = (long_seed * 2654435761 + 97) & 0x7FFFFFFF
    g0 = torch.randn((1, 2), generator=torch.Generator().manual_seed(zseed))
    lp = coll._lp_from_z(p_E, long_seed, g0)
    views = np.stack([coll._simulate_once(p_E, p_x, p_y, coll._sim_seed(seed, v), n_spots_override=100,
                                          long_params_override=lp)[0].cpu().numpy() for v in (0, 1)])
    np.savez(PAIR_FILE, views=views, E=float(p_E[0]), z=np.array(g0.tolist()))


if not PAIR_FILE.exists():
    render_pair()
PAIR = np.load(PAIR_FILE)
CMAP = 'inferno'   # views: inferno on black, the event-display look

FIG_W, FIG_H = 11.2, 4.7
SAVE_W = 10.42    # canvas trimmed on the right; include at SAVE_W / FIG_W = 0.93 \linewidth
FREE = S.INK2        # label-free signals
KNOWN = S.BLUE       # readability of the known factors
TINT = {FREE: '#f1f0ee', KNOWN: '#e8f1fc'}
XMIN, XMAX = -40, 160            # axis ends; all nine runs have data throughout
FULL = (-40, 160)                # relative epochs at which all nine runs have data
REL_LOSS = np.arange(XMIN, XMAX + 1)
REL_GRID = np.arange(XMIN, XMAX + 1, 10)


def aligned():
    runs = list(A.NINE)
    out = {}
    for name, trace, rel in (('cca', None, REL_GRID), ('loss', A.loss_trace, REL_LOSS),
                             ('rank', A.rankme_trace, REL_GRID)):
        rows = []
        for k in runs:
            d = A.AVG[k] if name == 'cca' else dict(zip(*trace(k))); o = A.onset(k)
            # every series starts at epoch 10, the first checkpoint on the 10-epoch grid
            rows.append([d.get(o + r, np.nan) if o + r >= 10 else np.nan for r in rel])
        M = np.array(rows, dtype=float)
        assert not np.isnan(M[:, (rel >= FULL[0]) & (rel <= FULL[1])]).any(), name
        out[name] = (rel, M)
    return runs, out


def tag(fig, x, y, w, h, title, body, color):
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.0,rounding_size=0.01',
                                      transform=fig.transFigure, facecolor=TINT[color],
                                      edgecolor='none'))
    fig.text(x + 0.012, y + h * 0.66, title, ha='left', va='center', color=color,
             fontsize=S.f(9.0), fontweight='semibold')
    fig.text(x + 0.012, y + h * 0.30, body, ha='left', va='center', color=S.INK2,
             fontsize=S.f(8.4))


def arrow(fig, p0, p1, color, rad=0.0, lw=1.2):
    fig.add_artist(FancyArrowPatch(p0, p1, transform=fig.transFigure, arrowstyle='-|>',
                                   mutation_scale=S.f(10), color=color, lw=S.f(lw),
                                   shrinkA=0, shrinkB=0, connectionstyle=f'arc3,rad={rad}'))


def main():
    S.apply(FIG_W)
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    aspect = FIG_W / FIG_H

    # ---------------- (a) the training pair -> one encoder -> two read-outs
    cmap = plt.get_cmap(CMAP).copy(); cmap.set_bad('black')
    imgs = [np.log1p(v.sum(0)) for v in PAIR['views']]
    vmax = max(i.max() for i in imgs)
    iw = 0.105; ih = iw * aspect
    ix = 0.025
    tops = (0.905, 0.905 - ih - 0.07)
    for j, top in enumerate(tops):
        ax = fig.add_axes([ix, top - ih, iw, ih])
        a_ = np.ma.masked_where(imgs[j] <= 0, imgs[j])
        ax.imshow(a_, cmap=cmap, vmin=0, vmax=vmax, interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(S.GRID); sp.set_linewidth(S.f(0.8))
        ax.text(0.5, 1.03, f'view {j + 1}', transform=ax.transAxes, ha='center', va='bottom',
                color=S.INK, fontsize=S.f(9.0))
    fig.text(ix + iw / 2, tops[1] - ih - 0.028,
             'same $z_1$, $z_2$, $E$,\ndifferent random\nelements',
             ha='center', va='top', color=S.INK2, fontsize=S.f(8.4), linespacing=1.3)

    # encoder: a trapezoid narrowing to the right, fed by both views
    ex0, ex1 = 0.158, 0.236
    ymid = (tops[0] + tops[1] - ih) / 2
    hl, hr = 0.30, 0.15
    fig.patches.append(Polygon([(ex0, ymid - hl / 2), (ex1, ymid - hr / 2), (ex1, ymid + hr / 2),
                                (ex0, ymid + hl / 2)], closed=True, transform=fig.transFigure,
                               facecolor='#ebebe8', edgecolor='none'))
    fig.text((ex0 + ex1) / 2 - 0.003, ymid, 'LeJEPA\nencoder', ha='center', va='center',
             color=S.INK, fontsize=S.f(9.0), linespacing=1.25)
    for j, top in enumerate(tops):
        yc = top - ih / 2
        arrow(fig, (ix + iw + 0.006, yc), (ex0 - 0.004, ymid + (0.07 if j == 0 else -0.07)),
              S.MUTED, lw=1.0)

    # read-outs: compact tags, each bracketing the strips it produces
    left, right, bot, top_, gap = 0.585, 0.905, 0.215, 0.905, 0.042
    unit = (top_ - bot - 2 * gap) / 3.35
    hts = [1.35 * unit, unit, unit]
    strip_y = [top_ - hts[0], top_ - hts[0] - gap - hts[1], bot]
    hs = hts[1]
    tx, tw, th = 0.275, 0.195, 0.13
    spans = {KNOWN: (strip_y[0], strip_y[0] + hts[0]), FREE: (strip_y[2], strip_y[1] + hs)}
    bx = 0.515
    for col, title, body in ((KNOWN, 'known-factor probes', 'readability of $z_1$, $z_2$, $\\log E$'),
                             (FREE, 'label-free signals', 'training loss, RankMe')):
        y0, y1 = spans[col]; yc = (y0 + y1) / 2
        tag(fig, tx, yc - th / 2, tw, th, title, body, col)
        fig.add_artist(Line2D([tx + tw, bx], [yc, yc], transform=fig.transFigure, color=col, lw=S.f(1.0)))
        fig.add_artist(Line2D([bx, bx], [y0 + 0.01, y1 - 0.01], transform=fig.transFigure, color=col, lw=S.f(1.0)))
        for yy in (y0 + 0.01, y1 - 0.01):
            fig.add_artist(Line2D([bx, bx + 0.008], [yy, yy], transform=fig.transFigure, color=col, lw=S.f(1.0)))
        arrow(fig, (ex1 + 0.004, ymid + (0.035 if col == KNOWN else -0.035)), (tx - 0.005, yc), col,
              rad=-0.2 if col == KNOWN else 0.2)

    # ---------------- (b) nine runs aligned at onset
    runs, A = aligned()
    specs = (('cca', 'average $R^2$', KNOWN, (0.66, 1.005), [0.7, 0.8, 0.9, 1.0]),
             ('loss', 'loss', FREE, (0.0, 0.10), [0, 0.05, 0.10]),
             ('rank', 'RankMe', FREE, (0, 9), [0, 4, 8]))
    axes = []
    for i, (key, ylab, col, ylim, yt) in enumerate(specs):
        ax = fig.add_axes([left, strip_y[i], right - left, hts[i]])
        rel, M = A[key]
        for row in M:
            ok = ~np.isnan(row)
            ax.plot(rel[ok], row[ok], color=col, lw=S.f(0.75), alpha=0.25, zorder=2)
        full = (rel >= FULL[0]) & (rel <= FULL[1])
        ax.plot(rel[full], np.median(M[:, full], axis=0), color=col, lw=S.f(2.3), zorder=4,
                solid_capstyle='round')
        if key == 'cca':
            pre = float(np.median(np.median(M[:, full & (rel < 0)], axis=0)))
            ax.axhline(pre, color=S.INK2, lw=S.f(1.1), ls=(0, (5, 3)), zorder=3)
            ax.text(XMAX - 2, pre + 0.008, 'level before onset', ha='right', va='bottom',
                    color=S.INK2, fontsize=S.f(8.6))
        ax.axvline(0, color=S.MUTED, lw=S.f(0.9), zorder=1)
        ax.set_xlim(XMIN, XMAX); ax.set_ylim(*ylim); ax.set_yticks(yt)
        ax.set_ylabel(ylab, color=col)
        S.tidy(ax)
        if i < 2:
            ax.tick_params(labelbottom=False)
        axes.append(ax)
    axes[0].text(2, 1.0, 'onset', ha='left', va='top', color=S.INK2, fontsize=S.f(8.6),
                 transform=axes[0].get_xaxis_transform())
    axes[-1].set_xlabel('epochs from the onset of the decline')
    axes[-1].set_xticks([-40, 0, 40, 80, 120, 160])

    fig.text((ix + tx + tw) / 2, 0.215 - S.f(S.LABEL_OFFSET_PT) / 72 / FIG_H, '(a) a training pair and what we watch', ha='center',
             va='top', color=S.INK, fontsize=S.f(S.LABEL_SIZE), fontname=S.LABEL_FONT)
    S.panel_label(axes[-1], 'b', 'nine runs, aligned at their onsets')
    handles = [Line2D([], [], color=S.INK2, lw=S.f(0.75), alpha=0.45, label='one run'),
               Line2D([], [], color=S.INK2, lw=S.f(2.3), label='median of the nine runs')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=((left + right) / 2, S.LEGEND_Y),
               ncol=2, frameon=False, columnspacing=2.0, handletextpad=0.6)
    S.report(fig, 'fig1', FIG_W)
    name = 'fig_overview'
    from matplotlib.transforms import Bbox
    box_in = Bbox([[0, 0], [SAVE_W, FIG_H]])   # fixed crop, not a content-dependent tight box
    fig.savefig(OUT / f'{name}.png', dpi=220, facecolor=S.SURFACE, bbox_inches=box_in)
    fig.savefig(OUT / f'{name}.pdf', facecolor=S.SURFACE, bbox_inches=box_in)
    print(f'saved {SAVE_W:.2f} x {FIG_H:.2f} in; include at {SAVE_W / FIG_W:.3f}\\linewidth')
    rel, M = A['cca']
    print('median readability at rel -10, 0, +30:', [round(float(np.median(M[:, rel == r])), 3) for r in (-10, 0, 30)])


if __name__ == '__main__':
    main()
