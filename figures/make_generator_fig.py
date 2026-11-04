"""Generator appendix figure (fig_rig) in Figure 1's style.

Same content as the installed figure: (a) the two
training views of one event at N = 100, (b) the same event at N = 30, 100, 300
and 1000. Renders come from the training collator on CPU
(generator_renders_seed20260825.npz; the views equal Figure 1's pair). Shown as
log(1 + intensity) summed over the depth layers, one shared scale, inferno on
black as in Figure 1. render_all() regenerates the
renders if the npz is missing. Writes to FIGURE_OUTPUT_DIR (default ../figs).
Panel (a) dropped (it is Figure 1a's pair); one row of four
element counts, no panel label.
"""
import os, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'mplcache_monitor_figs'))
sys.path.insert(0, str(HERE))
import numpy as np
import _style as S
import matplotlib.pyplot as plt

ROOT = HERE.parent
OUT = Path(os.environ.get('FIGURE_OUTPUT_DIR', ROOT / 'figs')); OUT.mkdir(parents=True, exist_ok=True)
RFILE = HERE / 'generator_renders_seed20260825.npz'


def render_all():
    """Seed-20260825 event through the training collator: two views at N = 100
    and view 0 at N = 30, 100, 300, 1000 (same truth and profile parameters)."""
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
    r = lambda v, n: coll._simulate_once(p_E, p_x, p_y, coll._sim_seed(seed, v), n_spots_override=n,
                                         long_params_override=lp)[0].cpu().numpy()
    np.savez(RFILE, views=np.stack([r(0, 100), r(1, 100)]),
             rungs=np.stack([r(0, n) for n in (30, 100, 300, 1000)]), N=np.array([30, 100, 300, 1000]))


if not RFILE.exists():
    render_all()
R = np.load(RFILE)
FIG_W, FIG_H = 11.2, 3.15
SIDE, GAP = 2.05, 0.35                    # square side and gap, inches on the canvas


def square(fig, x_in, y_in, img, vmax, label):
    ax = fig.add_axes([x_in / FIG_W, y_in / FIG_H, SIDE / FIG_W, SIDE / FIG_H])
    cmap = plt.get_cmap('inferno').copy(); cmap.set_bad('black')
    ax.imshow(np.ma.masked_where(img <= 0, img), cmap=cmap, vmin=0, vmax=vmax,
              interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(S.GRID); sp.set_linewidth(S.f(0.8))
    ax.text(0.5, 1.03, label, transform=ax.transAxes, ha='center', va='bottom',
            color=S.INK, fontsize=S.f(9.0))
    return ax


def main():
    S.apply(FIG_W)
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    views = [np.log1p(v.sum(0)) for v in R['views']]
    rungs = [np.log1p(v.sum(0)) for v in R['rungs']]
    vmax = max(i.max() for i in views + rungs)
    # (b) bottom row: four element counts
    x0 = (FIG_W - (4 * SIDE + 3 * GAP)) / 2
    yb = 0.35
    bottom = [square(fig, x0 + i * (SIDE + GAP), yb, img, vmax, f'$N = {n}$')
              for i, (n, img) in enumerate(zip(R['N'], rungs))]
    S.report(fig, 'fig5', FIG_W)
    fig.savefig(OUT / 'fig_rig.png', dpi=220, facecolor=S.SURFACE)
    fig.savefig(OUT / 'fig_rig.pdf', facecolor=S.SURFACE)
    print('N = 100 render equals view 1:', np.array_equal(R['rungs'][1], R['views'][0]))


if __name__ == '__main__':
    main()
