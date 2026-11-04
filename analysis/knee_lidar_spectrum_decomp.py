"""LiDAR spectrum decomposition and matrix substitution (2026-09-23).

Specified by question Q2 of the earlier LiDAR analysis, plus the
substitution terms with the within-group residual removed.

For every checkpoint below and both N = 100 replica sets
(knee_lidar_N100.pt = shape redrawn in a group, "redrawn";
knee_lidar_sharedshape_N100.pt = shape shared, "shared"), with features
exactly as knee_lidar.lidar_of (tokens -> LayerNorm -> mean, fp32):

  S_b, S_w      between- and within-group covariances exactly as lidar()
                (S_w used for whitening = raw S_w + delta I, delta = 1e-4)
  spectrum      eigenvalues of W S_b W, W = S_w^-1/2, clipped at 0
  lambda1, bulk (sum of the rest), q = lambda1 / total, H_rest = entropy of
                the rest renormalised; log LiDAR = h(q) + (1-q) H_rest exactly
  energy link   R^2(E) of the house ridge (alpha = 1) on the run's own probe
                cache at the same checkpoint (knee_dir_sweep.cache_for), and
                |cos| between the top eigenvector mapped to feature space
                (W v1) and the ridge's energy read-out (beta_E / sd_X)
  u             the dying direction's feature image: ridge read-out of
                w_hat . (z1, z2, E) at PRE (w_hat from results/knee_dir_sweep.csv;
                knee_t2_direction.W_KNEE for nano_1-3, not in that CSV; the
                reference w_bar for event-free runs), unit norm; u' S_w u,
                u' S_b u and their ratio (the whitened eigenvalue along u);
                ratio_u_own uses the read-out at the same checkpoint instead
  S_w vs delta  trace of raw S_w, count of raw S_w eigenvalues below delta

Terms per checkpoint (B set | W set): redrawn|redrawn, shared|shared,
redrawn|shared, shared|redrawn; each "raw" and "resid_removed"
(B - raw S_w(B set) / 8, eigenvalues clipped at 0).

Checkpoints: the 27 knee runs (onsets from knee_lidar_audit) at PRE (last
grid epoch <= onset - 20), onset and POST (final); the 8 mu = 0.97
calibration controls and the 11 held-out event-free runs at the nearest
grid epoch at or below their campaign's knee-cohort median PRE, and at
their final checkpoint.

Gates: lidar() on the same features must reproduce results/knee_lidar.csv and
results/knee_lidar_sharedshape.csv; exp(h(q) + (1-q) H_rest) is compared with
the same values; R^2(E) is compared with results/knee_traj_full.csv where that
file has the checkpoint.

Output: results/knee_lidar_spectrum_decomp.csv (new, refuses to overwrite).
Usage (from analysis/):
  CUDA_VISIBLE_DEVICES=0 python knee_lidar_spectrum_decomp.py
  python knee_lidar_spectrum_decomp.py --report
"""
import sys

sys.dont_write_bytecode = True      # leave analysis/__pycache__ untouched

import argparse  # noqa: E402
import csv  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from statistics import median  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
sys.path.insert(0, _HERE)

OUT_CSV = os.path.join(paths.CSV, 'knee_lidar_spectrum_decomp.csv')
LID_CSV = {'redrawn': os.path.join(paths.CSV, 'knee_lidar.csv'),
           'shared': os.path.join(paths.CSV, 'knee_lidar_sharedshape.csv')}
SET_FILE = {'redrawn': 'knee_lidar_N100.pt', 'shared': 'knee_lidar_sharedshape_N100.pt'}
TRAJ = os.path.join(paths.CSV, 'knee_traj_full.csv')
DIRS = os.path.join(paths.CSV, 'knee_dir_sweep.csv')
K_REP, DELTA, N_SPOTS = 8, 1e-4, 100
TERMS = (('redrawn', 'redrawn'), ('shared', 'shared'),
         ('redrawn', 'shared'), ('shared', 'redrawn'))       # (B set, W set)
COLS = ['run', 'role', 'campaign', 'onset', 'phase', 'epoch', 'term', 'variant',
        'B_set', 'W_set', 'lidar', 'lidar_decomp', 'lidar_repro', 'lidar_csv',
        'lambda1', 'bulk', 'q', 'H_rest', 'h_q', 'bulk_term', 'n_pos',
        'trace_B', 'trace_Wraw', 'n_Wraw_below_delta', 'resid_trace_over8',
        'cos_top_energy', 'corr_top_energy', 'r2E_probe', 's_probe', 'r2E_traj',
        'u_source', 'var_w_u', 'var_b_u', 'ratio_u', 'var_w_rest', 'var_b_rest',
        'ratio_u_own']


def short(run):
    return run.split('xxx')[0]


# ------------------------------------------------------------------ plan

def build_plan():
    """[(full_run, short_run, role, campaign, onset, [(phase, epoch), ...])]"""
    import knee_lidar_audit as AUD
    import knee_lidar_audit_sharedshape as A2
    grid, full = {}, {}
    with open(LID_CSV['redrawn']) as fh:
        for r in csv.DictReader(fh):
            if r['n_spots'] == str(N_SPOTS):
                s = short(r['run'])
                full[s] = r['run']
                grid.setdefault(s, set()).add(int(r['epoch']))
    plan, pre_by_camp = [], {'21x': [], '4x': []}
    for camp, knee in (('21x', AUD.KNEE_21), ('4x', AUD.KNEE_4)):
        for s, on in knee.items():
            g = sorted(grid[s])
            pre = max(e for e in g if e <= on - 20)
            assert on in g, (s, on)
            pre_by_camp[camp].append(pre)
            plan.append((full[s], s, 'knee', camp, on,
                         [('pre', pre), ('onset', on), ('post', max(g))]))
    ref = {c: median(v) for c, v in pre_by_camp.items()}
    for role, runs in (('control', AUD.CTRL_21 + AUD.CTRL_4),
                       ('heldout', A2.HELD_21 + A2.HELD_4)):
        for s in runs:
            camp = A2.campaign(s)
            g = sorted(grid[s])
            pre = max(e for e in g if e <= ref[camp])
            plan.append((full[s], s, role, camp, '', [('pre', pre), ('post', max(g))]))
    return plan, ref


def w_hats():
    """run(short) -> (w_hat as float64 tensor, source label)."""
    import torch
    from knee_t2_direction import W_KNEE
    out = {}
    with open(DIRS) as fh:
        for r in csv.DictReader(fh):
            out[r['run']] = (torch.tensor([float(r['w_z1']), float(r['w_z2']),
                                           float(r['w_E'])], dtype=torch.float64),
                             'knee_dir_sweep')
    for i, key in ((1, 'knee seed 1'), (2, 'knee seed 2'), (3, 'knee seed 3')):
        v = torch.tensor(W_KNEE[key], dtype=torch.float64)
        out.setdefault(f'rigb_lejp_p1s100_nano_{i}', (v / v.norm(), 'W_KNEE'))
    wk = [torch.tensor(v, dtype=torch.float64) for v in W_KNEE.values()]
    wbar = torch.stack([v / v.norm() for v in wk]).mean(0)
    return out, wbar / wbar.norm()


# ---------------------------------------------------------------- linear algebra

def mats(F):
    """lidar()'s S_b and raw S_w (no delta), same dtype/device as F."""
    Mc, Kc, D = F.shape
    mu_c = F.mean(1)
    mu = mu_c.mean(0)
    Sb = (mu_c - mu).T @ (mu_c - mu) / (Mc - 1)
    Xw = (F - mu_c[:, None, :]).reshape(-1, D)
    return Sb, Xw.T @ Xw / (Mc * (Kc - 1))


def whitener(Sw_raw):
    """lidar()'s W = (S_w + delta I)^-1/2 with its eigenvalue floor."""
    import torch
    Sw = Sw_raw + DELTA * torch.eye(Sw_raw.shape[0], device=Sw_raw.device)
    ew, U = torch.linalg.eigh(Sw)
    return U @ torch.diag(ew.clamp(min=1e-12).rsqrt()) @ U.T, Sw


def lidar_from_lam(lam):
    """lidar()'s final lines on clipped eigenvalues (same dtype)."""
    import torch
    p = lam / (lam.sum() + 1e-12) + 1e-12
    return float(torch.exp(-(p * torch.log(p)).sum()))


def decomp(lam):
    lam = lam.double().clamp(min=0).sort(descending=True).values
    p = lam / lam.sum()
    q = float(p[0])
    r = p[1:] / (1.0 - q)
    r = r[r > 0]
    H_rest = float(-(r * r.log()).sum())
    h_q = (-q * math.log(q) - (1 - q) * math.log(1 - q)) if 0 < q < 1 else 0.0
    return dict(lambda1=float(lam[0]), bulk=float(lam[1:].sum()), q=q, H_rest=H_rest,
                h_q=h_q, bulk_term=(1 - q) * H_rest,
                lidar_decomp=math.exp(h_q + (1 - q) * H_rest),
                n_pos=int((lam > 0).sum()))


def term_stats(Sb_used, Sw_raw_W, dirs, Fflat):
    """Spectrum, decomposition and direction quantities for one (B, W) pair.
    Fflat: the B set's 512 x D features (float64, cpu), for the projection
    correlation between the top eigenvector's read-out and the energy read-out
    (a plain cosine of two read-out vectors misleads when most S_w eigenvalues
    sit at the delta floor)."""
    import torch
    W, Sw = whitener(Sw_raw_W)
    Mw = W @ Sb_used @ W
    lam, V = torch.linalg.eigh((Mw + Mw.T) / 2)
    lam = lam.clamp(min=0)
    out = decomp(lam)
    out['lidar'] = lidar_from_lam(lam)
    a1 = (W @ V[:, -1]).double().cpu()
    Sb64, Sw64 = Sb_used.double().cpu(), Sw.double().cpu()
    D = Sb64.shape[0]
    bE = dirs['bE']
    out['cos_top_energy'] = abs(float(a1 @ bE / (a1.norm() * bE.norm())))
    pa, pb = Fflat @ a1, Fflat @ bE
    pa, pb = pa - pa.mean(), pb - pb.mean()
    out['corr_top_energy'] = abs(float(pa @ pb / (pa.norm() * pb.norm())))
    u = dirs['u']
    vw, vb = float(u @ Sw64 @ u), float(u @ Sb64 @ u)
    out.update(var_w_u=vw, var_b_u=vb, ratio_u=vb / vw,
               var_w_rest=(float(Sw64.trace()) - vw) / (D - 1),
               var_b_rest=(float(Sb64.trace()) - vb) / (D - 1))
    uo = dirs['u_own']
    out['ratio_u_own'] = float(uo @ Sb64 @ uo) / float(uo @ Sw64 @ uo)
    ew = torch.linalg.eigvalsh(Sw_raw_W.double().cpu())
    out.update(trace_B=float(Sb64.trace()), trace_Wraw=float(Sw_raw_W.double().trace()),
               n_Wraw_below_delta=int((ew < DELTA).sum()))
    return out


def ridge_dirs(h_tr, y_tr):
    """House ridge (wp1_metrics construction, alpha = 1): weights in
    standardised feature space (D x 3) and the feature sd."""
    import torch
    X, Y = h_tr.double(), y_tr.double()
    Xm, Xs = X.mean(0), X.std(0) + 1e-8
    Ym, Ys = Y.mean(0), Y.std(0) + 1e-8
    Xn, Yn = (X - Xm) / Xs, (Y - Ym) / Ys
    Wr = torch.linalg.solve(Xn.T @ Xn + torch.eye(X.shape[1], dtype=X.dtype), Xn.T @ Yn)
    return Wr, Xs


def unit(v):
    return v / v.norm()


# -------------------------------------------------------------------- main

def main():
    import torch
    import torch.nn.functional as F
    import yaml
    import knee_lidar as KL
    import knee_dir_sweep as DS
    from wp1_metrics import ridge_r2

    assert not os.path.exists(OUT_CSV), f'refusing to overwrite {OUT_CSV}'
    print(f'GPU: {torch.cuda.get_device_name(0)} '
          f'(CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")})', flush=True)
    plan, ref = build_plan()
    print(f'plan: {len(plan)} runs, {sum(len(p[5]) for p in plan)} checkpoints; '
          f'event-free PRE reference (knee-cohort median PRE): {ref}', flush=True)
    WH, wbar = w_hats()

    csv_lid = {}
    for tag, p in LID_CSV.items():
        with open(p) as fh:
            for r in csv.DictReader(fh):
                if r['n_spots'] == str(N_SPOTS):
                    csv_lid[(tag, short(r['run']), int(r['epoch']))] = float(r['lidar'])
    traj = {}
    with open(TRAJ) as fh:
        for r in csv.DictReader(fh):
            traj[(short(r['run']), int(r['epoch']))] = float(r['ridge_r2_E'])
    imgs = {k: torch.load(os.path.join(KL.CACHE, f), map_location='cpu', weights_only=True)
            for k, f in SET_FILE.items()}
    caches = {}

    @torch.no_grad()
    def feats_of(enc, x):
        """knee_lidar.lidar_of's feature path, returning (M, K, D)."""
        out = []
        enc.eval()
        for i in range(0, x.size(0), 64):
            h = enc(x[i:i + 64].to(KL.DEV, torch.float32)).float()
            h = F.layer_norm(h, (h.size(-1),))
            out.append(h.mean(dim=1))
        return torch.cat(out, dim=0).reshape(KL.M, KL.K, -1)

    fout = open(OUT_CSV, 'w', newline='')
    w = csv.DictWriter(fout, fieldnames=COLS, lineterminator='\n')
    w.writeheader()
    t_all = time.time()
    for full, s, role, camp, onset, cks in plan:
        t0 = time.time()
        cname = DS.cache_for(s)
        if cname not in caches:
            caches[cname] = torch.load(os.path.join(KL.CACHE, cname), map_location='cpu',
                                       weights_only=True)
        sets = caches[cname]
        cfg = yaml.safe_load(open(os.path.join(KL.drv.ROOT, full, 'config.yml')))
        tag = cfg['logging']['write_tag']
        if role == 'knee':
            what, u_src = WH[s]                  # every knee run has its own w_hat
        else:
            what, u_src = wbar, 'w_bar'          # event-free: the reference direction,
            #                                      even if knee_dir_sweep lists the run
        what = unit(what)
        u_pre = None
        for phase, ep in cks:
            enc = KL.drv.encoder_from_ckpt(cfg, os.path.join(KL.drv.ROOT, full,
                                                             f'{tag}-ep{ep}.pth.tar'))
            Fs = {k: feats_of(enc, imgs[k]) for k in SET_FILE}
            h_tr = KL.drv.encode_meanpool(enc, sets['imgs_tr'])
            h_ev = KL.drv.encode_meanpool(enc, sets['imgs_ev'])
            del enc
            r2E = ridge_r2(h_tr, sets['y_tr'], h_ev, sets['y_ev'])[2]
            Wr, Xs = ridge_dirs(h_tr, sets['y_tr'])
            u_own = unit((Wr @ what) / Xs)
            if phase == 'pre':
                u_pre = u_own
            dirs = {'bE': Wr[:, 2] / Xs, 'u': u_pre, 'u_own': u_own}
            repro = {k: KL.lidar(Fs[k]) for k in SET_FILE}
            M = {k: mats(Fs[k]) for k in SET_FILE}              # (Sb, Sw_raw)
            Ff = {k: Fs[k].reshape(-1, Fs[k].shape[-1]).double().cpu() for k in SET_FILE}
            common = dict(run=full, role=role, campaign=camp, onset=onset, phase=phase,
                          epoch=ep, r2E_probe=f'{r2E:.6f}',
                          s_probe=f'{r2E / (1 - r2E):.6f}',
                          r2E_traj=(f'{traj[(s, ep)]:.6f}' if (s, ep) in traj else ''),
                          u_source=u_src)
            for bset, wset in TERMS:
                Sb, Sw_raw_B = M[bset]
                for variant in ('raw', 'resid_removed'):
                    Sb_used = Sb if variant == 'raw' else Sb - Sw_raw_B / K_REP
                    st = term_stats(Sb_used, M[wset][1], dirs, Ff[bset])
                    row = dict(common, term=f'{bset}|{wset}', variant=variant,
                               B_set=bset, W_set=wset,
                               resid_trace_over8=f'{float(Sw_raw_B.double().trace()) / K_REP:.6e}')
                    if bset == wset and variant == 'raw':
                        row['lidar_repro'] = f'{repro[bset]:.6f}'
                        row['lidar_csv'] = f'{csv_lid[(bset, s, ep)]:.6f}'
                    for k, v in st.items():
                        row[k] = (f'{v:.6e}' if isinstance(v, float) else v)
                    w.writerow(row)
            fout.flush()
        print(f'{s:36s} {role:8s} {len(cks)} ckpts ({time.time() - t0:.0f}s; '
              f'total {time.time() - t_all:.0f}s)', flush=True)
    fout.close()
    print(f'DONE -> {OUT_CSV}')


# ------------------------------------------------------------------ report

def report():
    rows = list(csv.DictReader(open(OUT_CSV)))
    f = lambda r, k: float(r[k])                                    # noqa: E731
    key = lambda r: (short(r['run']), r['phase'], r['term'], r['variant'])  # noqa: E731
    R = {key(r): r for r in rows}
    print(f'{OUT_CSV}: {len(rows)} rows')

    print('\n== gates (unsubstituted raw terms) ==')
    for t in ('redrawn|redrawn', 'shared|shared'):
        sel = [r for r in rows if r['term'] == t and r['variant'] == 'raw']
        d1 = max(abs(f(r, 'lidar_repro') - f(r, 'lidar_csv')) for r in sel)
        d2 = max(abs(f(r, 'lidar_decomp') - f(r, 'lidar_csv')) for r in sel)
        d3 = max(abs(f(r, 'lidar') - f(r, 'lidar_csv')) for r in sel)
        print(f'  {t:16s} n={len(sel)}  max|lidar() - csv| {d1:.2e}  '
              f'max|exp(h(q)+(1-q)H_rest) - csv| {d2:.2e}  max|eigh-path - csv| {d3:.2e}')
    tr = [r for r in rows if r['term'] == 'redrawn|redrawn' and r['variant'] == 'raw'
          and r['r2E_traj']]
    print(f'  R2(E) vs knee_traj_full.csv: n={len(tr)} max|d| '
          f'{max(abs(f(r, "r2E_probe") - f(r, "r2E_traj")) for r in tr):.2e}')

    def split(a, b):
        """Symmetric split of log LiDAR(b) - log LiDAR(a) into the part from
        the top share q and the part from the spread of the rest H_rest."""
        q1, q2, H1, H2 = f(a, 'q'), f(b, 'q'), f(a, 'H_rest'), f(b, 'H_rest')
        g = lambda q, H: ((-q * math.log(q) - (1 - q) * math.log(1 - q)) + (1 - q) * H)  # noqa: E731
        dq = 0.5 * ((g(q2, H1) - g(q1, H1)) + (g(q2, H2) - g(q1, H2)))
        dH = 0.5 * ((1 - q1) + (1 - q2)) * (H2 - H1)
        return math.log(f(b, 'lidar_decomp')) - math.log(f(a, 'lidar_decomp')), dq, dH

    runs = sorted({(short(r['run']), r['role'], r['campaign']) for r in rows},
                  key=lambda x: ({'knee': 0, 'control': 1, 'heldout': 2}[x[1]], x[2], x[0]))
    for t in ('redrawn|redrawn', 'shared|shared'):
        print(f'\n== {t}, raw: PRE -> POST per run ==')
        print(f'{"run":22s} {"role":7s} {"LiDAR pre>on>post":>22s} {"lam1 pre>post":>17s} '
              f'{"bulk pre>post":>15s} {"q pre>post":>13s} {"Hrest pre>post":>15s} '
              f'{"dlogL":>7s} {"d_q":>7s} {"d_H":>7s} {"dlogl1":>7s} {"dlogB":>7s} '
              f'{"cosE pre/post":>13s} {"ratio_u pre>post":>17s}')
        for s, role, camp in runs:
            a, b = R[(s, 'pre', t, 'raw')], R[(s, 'post', t, 'raw')]
            o = R.get((s, 'onset', t, 'raw'))
            dl, dq, dH = split(a, b)
            print(f'{s.split("p1s100_")[1]:22s} {role:7s} '
                  f'{f(a, "lidar"):6.2f}>{(f(o, "lidar") if o else float("nan")):6.2f}>{f(b, "lidar"):6.2f} '
                  f'{f(a, "lambda1"):7.2f}>{f(b, "lambda1"):7.2f} {f(a, "bulk"):6.2f}>{f(b, "bulk"):6.2f} '
                  f'{f(a, "q"):5.3f}>{f(b, "q"):5.3f} {f(a, "H_rest"):6.3f}>{f(b, "H_rest"):6.3f} '
                  f'{dl:+7.3f} {dq:+7.3f} {dH:+7.3f} '
                  f'{math.log(f(b, "lambda1") / f(a, "lambda1")):+7.3f} '
                  f'{math.log(f(b, "bulk") / f(a, "bulk")):+7.3f} '
                  f'{f(a, "cos_top_energy"):5.3f}/{f(b, "cos_top_energy"):5.3f} '
                  f'{f(a, "ratio_u"):7.3f}>{f(b, "ratio_u"):7.3f}')

    print('\n== counts over the 27 knee runs, PRE -> POST (raw) ==')
    knee = [s for s, role, _ in runs if role == 'knee']
    for t in ('redrawn|redrawn', 'shared|shared'):
        A = [(R[(s, 'pre', t, 'raw')], R[(s, 'post', t, 'raw')]) for s in knee]
        sp = [split(a, b) for a, b in A]
        l1f = sum(f(b, 'lambda1') < f(a, 'lambda1') for a, b in A)
        bg = sum(f(b, 'bulk') > f(a, 'bulk') for a, b in A)
        hr = sum(f(b, 'H_rest') > f(a, 'H_rest') for a, b in A)
        big_q = sum(abs(dq) > abs(dH) for _, dq, dH in sp)
        up = sum(dl > 0 for dl, _, _ in sp)
        uu = sum(f(b, 'ratio_u') > f(a, 'ratio_u') for a, b in A)
        print(f'  {t:16s} LiDAR up {up}/27 | lambda1 falls {l1f}/27 | bulk grows {bg}/27 | '
              f'H_rest grows {hr}/27 | |d_q| > |d_H| {big_q}/27 | ratio_u rises {uu}/27')
        print(f'  {"":16s} medians: dlogL {median(x[0] for x in sp):+.3f}, d_q '
              f'{median(x[1] for x in sp):+.3f}, d_H {median(x[2] for x in sp):+.3f}; '
              f'lambda1 post/pre {median(f(b, "lambda1") / f(a, "lambda1") for a, b in A):.3f}; '
              f'bulk post/pre {median(f(b, "bulk") / f(a, "bulk") for a, b in A):.3f}; '
              f'ratio_u pre {median(f(a, "ratio_u") for a, b in A):.3f} post '
              f'{median(f(b, "ratio_u") for a, b in A):.3f}')

    print('\n== energy link (redrawn|redrawn, raw): lambda1 - 1/8 against R2(E)/(1-R2(E)) ==')
    kr = [r for r in rows if r['term'] == 'redrawn|redrawn' and r['variant'] == 'raw'
          and r['role'] == 'knee']
    x = [f(r, 'lambda1') - 1 / K_REP for r in kr]
    y = [f(r, 's_probe') for r in kr]
    print(f'  knee checkpoints n={len(kr)}: Spearman {spearman(x, y):+.3f}; ratio '
          f'(lambda1-1/8)/s_probe median {median(a / b for a, b in zip(x, y)):.3f} '
          f'(range {min(a / b for a, b in zip(x, y)):.3f}-{max(a / b for a, b in zip(x, y)):.3f})')
    co = sum((f(R[(s, 'post', 'redrawn|redrawn', 'raw')], 'lambda1')
              - f(R[(s, 'pre', 'redrawn|redrawn', 'raw')], 'lambda1'))
             * (f(R[(s, 'post', 'redrawn|redrawn', 'raw')], 's_probe')
                - f(R[(s, 'pre', 'redrawn|redrawn', 'raw')], 's_probe')) > 0 for s in knee)
    print(f'  PRE -> POST, lambda1 and s_probe move in the same direction: {co}/27')
    er = [r for r in rows if r['term'] == 'redrawn|redrawn' and r['variant'] == 'raw'
          and r['role'] != 'knee']
    print(f'  event-free checkpoints n={len(er)}: Spearman '
          f'{spearman([f(r, "lambda1") - 1 / K_REP for r in er], [f(r, "s_probe") for r in er]):+.3f}')
    print(f'  |corr(projection on top eigenvector read-out, projection on energy read-out)| over the 512 renders, redrawn: median '
          f'{median(f(r, "corr_top_energy") for r in kr):.3f} (min {min(f(r, "corr_top_energy") for r in kr):.3f}); shared: median '
          f'{median(f(r, "corr_top_energy") for r in rows if r["term"] == "shared|shared" and r["variant"] == "raw" and r["role"] == "knee"):.3f}')
    print(f'  |cos(top eigenvector, energy read-out)| redrawn: median '
          f'{median(f(r, "cos_top_energy") for r in kr):.3f} (min '
          f'{min(f(r, "cos_top_energy") for r in kr):.3f}); shared: median '
          f'{median(f(r, "cos_top_energy") for r in rows if r["term"] == "shared|shared" and r["variant"] == "raw" and r["role"] == "knee"):.3f}')

    print('\n== substitution terms: medians by role and phase (LiDAR value) ==')
    for role in ('knee', 'control', 'heldout'):
        for phase in ('pre', 'onset', 'post'):
            sel = [r for r in rows if r['role'] == role and r['phase'] == phase]
            if not sel:
                continue
            cells = []
            for t in ('redrawn|redrawn', 'shared|shared', 'redrawn|shared', 'shared|redrawn'):
                for v in ('raw', 'resid_removed'):
                    vals = [f(r, 'lidar') for r in sel if r['term'] == t and r['variant'] == v]
                    cells.append(f'{t}/{v[:5]} {median(vals):6.2f}')
            print(f'  {role:7s} {phase:5s} n={len(sel) // 8:2d}: ' + '; '.join(cells))
    print('\n== within-group scatter (median trace of raw S_w, /8, and count below delta) ==')
    for t, setname in (('redrawn|redrawn', 'redrawn'), ('shared|shared', 'shared')):
        for role in ('knee', 'control', 'heldout'):
            for phase in ('pre', 'onset', 'post'):
                sel = [r for r in rows if r['term'] == t and r['variant'] == 'raw'
                       and r['role'] == role and r['phase'] == phase]
                if sel:
                    print(f'  {setname:7s} {role:7s} {phase:5s}: trace(S_w)/8 median '
                          f'{median(f(r, "trace_Wraw") / K_REP for r in sel):.3e}; '
                          f'eigenvalues < 1e-4: median {median(int(r["n_Wraw_below_delta"]) for r in sel)} of 96')


def spearman(a, b):
    def ranks(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(o):
            r[i] = pos + 1.0
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    return num / (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', action='store_true')
    a = ap.parse_args()
    if a.report:
        report()
    elif 'CUDA_VISIBLE_DEVICES' not in os.environ:
        raise SystemExit('set CUDA_VISIBLE_DEVICES explicitly (shared GPUs)')
    else:
        main()
