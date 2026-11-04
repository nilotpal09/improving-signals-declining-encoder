"""M5: two-view per-direction SNR, per the definition printed in the record
before computation (2026-08-23). The original computation left no script;
this reproduces it from the cached two-view set and writes
results/knee_m5_snr.csv.

Two renders (views) of each event: the cache m5_twoview.pt. House pooling
(tokens -> LayerNorm -> mean), house ridge (alpha=1) fitted on view A of the
train split, applied to both views of the eval split, giving predictions
yA, yB in standardised (z1, z2, E) space. For a unit direction w:
    SNR(w) = Var_i[w.(yA_i + yB_i)/2] / (1/2 mean_i[(w.yA_i - w.yB_i)^2])
Reported per encoder: SNR at the reference direction w_bar, its percentile
among 2000 pinned random unit directions (seed 20260950, the S8 set), the
analytic minimum-SNR direction and its |cos| to w_bar, and per-coordinate
SNR. Encoders: random-init seed 0, and width-384 mu=0 seed 2 at epoch 120.
"""
import csv
import os
import sys

import torch
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
sys.path.insert(0, _HERE)
import wp1_probe_driver as drv  # noqa: E402

CACHE = paths.CACHE
CFG_INIT = os.path.join(drv.ROOT, 'rigb_lejp_p1s100_nano_1xxx2i533m', 'config.yml')
RUN_PRE = os.path.join(drv.ROOT, 'rigb_lejp_p1s100_p384_2xxxa8r220')
EP_PRE = 120
N_REF, REF_SEED = 2000, 20260950
OUT = os.path.join(paths.CSV, 'knee_m5_snr.csv')

# w_bar as S8 built it: normalized mean of the four printed S7 dying directions.
W_KNEE = [(0.511, -0.662, 0.549), (0.511, -0.641, 0.573),
          (0.520, -0.632, 0.575), (0.513, -0.643, 0.569)]
# w_bar_new (knee_dir_sweep_avgr2, average ridge R2 anchors), for comparison.
W_BAR_NEW = (0.513, -0.645, 0.567)


def unit(v):
    v = torch.as_tensor(v, dtype=torch.float64)
    return v / v.norm()


def ridge_two_view(ha_tr, y_tr, ha_ev, hb_ev, alpha=1.0):
    """House ridge fitted on view A (train); standardized predictions for
    both views of the eval split."""
    ha_tr, y_tr, ha_ev, hb_ev = (t.double() for t in (ha_tr, y_tr, ha_ev, hb_ev))
    Xm, Xs = ha_tr.mean(0), ha_tr.std(0) + 1e-8
    Ym, Ys = y_tr.mean(0), y_tr.std(0) + 1e-8
    Xn, Yn = (ha_tr - Xm) / Xs, (y_tr - Ym) / Ys
    K = Xn.shape[1]
    W = torch.linalg.solve(Xn.T @ Xn + alpha * torch.eye(K, dtype=Xn.dtype),
                           Xn.T @ Yn)
    return ((ha_ev - Xm) / Xs) @ W, ((hb_ev - Xm) / Xs) @ W


def snr_mats(yA, yB):
    """(Between, Within) 3x3 matrices so SNR(w) = w'Bw / w'Ww."""
    m = (yA + yB) / 2
    mc = m - m.mean(0)
    B = mc.T @ mc / (m.shape[0] - 1)          # Var_i (unbiased, as torch.var)
    d = yA - yB
    Wm = 0.5 * d.T @ d / d.shape[0]
    return B, Wm


def snr(B, Wm, w):
    w = w if w.dim() == 2 else w[None]
    return ((w @ B) * w).sum(1) / ((w @ Wm) * w).sum(1)


def min_snr_dir(B, Wm):
    L = torch.linalg.cholesky(Wm)
    Li = torch.linalg.inv(L)
    M = Li @ B @ Li.T
    _, V = torch.linalg.eigh((M + M.T) / 2)
    w = Li.T @ V[:, 0]
    w = w / w.norm()
    return -w if w[0] < 0 else w


def main():
    d = torch.load(os.path.join(CACHE, 'm5_twoview.pt'), map_location='cpu',
                   weights_only=True)
    g = torch.Generator().manual_seed(REF_SEED)
    REF = torch.randn(N_REF, 3, generator=g, dtype=torch.float64)
    REF = REF / REF.norm(dim=1, keepdim=True)
    wbar = unit(torch.stack([unit(v) for v in W_KNEE]).mean(0))
    wbar_new = unit(W_BAR_NEW)

    encs = {}
    torch.manual_seed(0)
    encs['random_init_s0'] = drv.build_encoder_from_cfg(
        yaml.safe_load(open(CFG_INIT)))
    cfg_pre = yaml.safe_load(open(os.path.join(RUN_PRE, 'config.yml')))
    encs[f'p384_2_ep{EP_PRE}'] = drv.encoder_from_ckpt(
        cfg_pre, os.path.join(RUN_PRE, f'jepa-ep{EP_PRE}.pth.tar'))

    rows = []
    for name, enc in encs.items():
        ha_tr = drv.encode_meanpool(enc, d['a_tr'])
        ha_ev = drv.encode_meanpool(enc, d['a_ev'])
        hb_ev = drv.encode_meanpool(enc, d['b_ev'])
        yA, yB = ridge_two_view(ha_tr, d['y_tr'], ha_ev, hb_ev)
        B, Wm = snr_mats(yA, yB)
        s_ref = snr(B, Wm, REF)
        s_bar = float(snr(B, Wm, wbar)[0])
        w_min = min_snr_dir(B, Wm)
        coord = [float(snr(B, Wm, torch.eye(3, dtype=torch.float64)[i])[0])
                 for i in range(3)]
        rows.append(dict(
            encoder=name,
            snr_wbar=round(s_bar, 4),
            pct_wbar_among_ref=round(float((s_ref < s_bar).double().mean()) * 100, 2),
            min_dir_z1=round(float(w_min[0]), 4), min_dir_z2=round(float(w_min[1]), 4),
            min_dir_E=round(float(w_min[2]), 4),
            snr_min=round(float(snr(B, Wm, w_min)[0]), 4),
            abs_cos_min_wbar=round(abs(float(w_min @ wbar)), 4),
            abs_cos_min_wbar_new=round(abs(float(w_min @ wbar_new)), 4),
            snr_z1=round(coord[0], 3), snr_z2=round(coord[1], 3), snr_E=round(coord[2], 3),
            ref_min=round(float(s_ref.min()), 3), ref_median=round(float(s_ref.median()), 3),
            ref_max=round(float(s_ref.max()), 3)))
        print(rows[-1])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
