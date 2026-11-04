"""LiDAR and RankMe on the LeJEPA projector output (2026-09-24). Inference
only, no training; new files only.

Why: every LiDAR and RankMe series so far reads the pooled backbone
representation (encoder tokens -> LayerNorm -> mean over tokens). The LeJEPA
loss reads a 16-dim projector output instead, and LiDAR's paper (arXiv
2312.04000, sec. 2) defines its metric on the embedding e = psi o f
(projector after encoder). This computes both metrics on that embedding.

Training path, as the training code builds it (read, not assumed):
  * train/src/train_lejepa.py:571-576 (masking 'none', all 48
    runs): h_all = encoder(cat(views)) -> pool_tokens(h_all) -> projector.
  * train/src/lejepa.py:420-425 pool_tokens = h.mean(dim=1), a
    plain token mean. No norm between the encoder output and the mean; the
    only norm is the encoder's own final affine LayerNorm, part of the
    encoder (src/models/vision_transformer.py:440-441).
  * train_lejepa.py:399-401 projector = LeJEPAProjector(encoder.embed_dim,
    hidden_dim=proj_hidden_dim, proj_dim=proj_dim), defaults 2048 / 16
    (train_lejepa.py:255-256); lejepa.py:398-417 = torchvision MLP
    Linear-BN-ReLU-Dropout(0) twice, then Linear. in_dim 96 (vit_nano) or
    192 (vit_tiny); hidden 2048 (21x campaign, ViT-Tiny) or 384 (4x).
  * train_lejepa.py:486-487 wraps both in DDP, so the saved projector keys
    carry a 'module.' prefix (stripped here; loaded strict=True).
  * train_lejepa.py:494-501 saves the same online-encoder state dict under
    'encoder' and 'target_encoder'; checked at every checkpoint here.
  * Training ran under bf16 autocast (use_bfloat16: True in all 48 configs;
    train_lejepa.py:564). Here fp32, as for every pooled series; the bf16
    effect is measured in --stage0.

BatchNorm modes (both reported):
  'eval'  (primary)  projector.eval(): BN uses the saved running statistics
                     (standard inference).
  'batch' (secondary) BN uses the batch statistics of the whole input set at
                     once (the 512 replica images of a grouping, or the
                     4,000 fitting events, or the 2,000 evaluation events),
                     via F.batch_norm(running stats None), so the saved
                     running statistics are not touched (checked per
                     checkpoint). The loss itself sees train-mode BN.

Inputs, unchanged from the pooled series:
  * LiDAR: knee_lidar.lidar (lidar() from analysis/lidar_metric.py, imported
    by knee_lidar.py), M = 64 truths x K = 8 replicas, delta = 1e-4, on the
    cached replica sets knee_lidar_N100.pt (shape redrawn; the set
    knee_lidar.replica_set(cfg, 100) loads) and knee_lidar_sharedshape_N100.pt
    (shape shared; knee_lidar_sharedshape.cache_path(100)). Encoder chunks of
    64 as knee_lidar.lidar_of.
  * RankMe: knee_labelfree_metrics.rankme_pub (centred False / True) and
    rankme_sq_centred, imported, on the projector output of the run's
    dial-matched cache fitting split (4,000 events,
    knee_dir_sweep.cache_for), encoder chunks of 256 as
    wp1_probe_driver.encode_meanpool.
  * Encoder: wp1_probe_driver.encoder_from_ckpt, unchanged.

Per-checkpoint checks recorded in the pass (the same encoder pass also
yields the pooled features): pooled LiDAR (LayerNorm path) against
results/knee_lidar.csv and results/knee_lidar_sharedshape.csv; pooled fitting and
evaluation features against labelfree_feats/<run>.pt; encoder ==
target_encoder; running statistics untouched.

Outputs (new; refuse to overwrite):
  proj_feats/<run>.pt under the work folder (paths.WORK), one per
    run (skip existing = resumable; '.partial' then os.replace):
    'emb': {'<set>_<bn>': (n_ckpt, n, 16) float32} for set in redrawn,
           shared (512 = 64 x 8, row m*8 + k), fit (4,000), ev (2,000);
    'lidar_proj', 'rankme_proj' (GPU pass values), 'check' (the
    per-checkpoint checks), 'epochs', config facts.
  results/knee_proj_lidar.csv   run, epoch, grouping, bn_mode, lidar
  results/knee_proj_rankme.csv  run, epoch, variant, bn_mode, value
    (--write-csv, after all 48 files exist; LiDAR = the GPU float32 values,
    delta = 1e-4, the primary series, same settings as the pooled series)
  results/knee_proj_lidar_delta.csv  (--check) projector LiDAR recomputed from the
    stored embeddings in float64 on the CPU with lidar()'s delta = 1e-4, 1e-3,
    1e-5, 1e-6, 1e-8 (the imported lidar(), its delta argument)
  results/knee_proj_lidar_delta_share.csv  (--delta-share) at knee PRE / POST and
    control final checkpoints: within-group variance along lidar()'s leading
    whitened direction in units of delta, and delta's share of that
    direction's denominator (delta = 1e-4)
  proj_feats/check/proj_lidar_within_spectrum.csv  (--check) per evaluation:
    smallest / largest eigenvalue and trace of the within-group covariance
    without +delta I, largest between-group eigenvalue

Usage (from analysis/):
  CUDA_VISIBLE_DEVICES=0 python -B knee_proj_lidar_rankme.py --stage0
  CUDA_VISIBLE_DEVICES=0 python -B knee_proj_lidar_rankme.py --gate
  CUDA_VISIBLE_DEVICES=0 python -B knee_proj_lidar_rankme.py          # the pass
  python -B knee_proj_lidar_rankme.py --write-csv
  CUDA_VISIBLE_DEVICES= python -B knee_proj_lidar_rankme.py --check   # CPU
  CUDA_VISIBLE_DEVICES= python -B knee_proj_lidar_rankme.py --delta-share
Then the audit: python -B knee_proj_audit.py --selftest; python -B knee_proj_audit.py
"""
import sys

sys.dont_write_bytecode = True      # leave analysis/__pycache__ untouched

import argparse  # noqa: E402
import csv  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from collections import OrderedDict  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
sys.path.insert(0, _HERE)

CACHE_DIR = paths.CACHE
OUT_DIR = os.path.join(paths.WORK, 'proj_feats')  # denom: 48 symlinks + 3 new
FEAT_DIR = os.path.join(paths.WORK, 'labelfree_feats')  # denom: 58 symlinks + 3 new
SRC = os.path.join(paths.CSV, 'knee_lidar.csv')
SHARED_CSV = os.path.join(paths.CSV, 'knee_lidar_sharedshape.csv')
MET_CSV = os.path.join(paths.CSV, 'knee_labelfree_metrics.csv')
OUT_LIDAR = os.path.join(paths.CSV, 'knee_proj_lidar.csv')
OUT_RANKME = os.path.join(paths.CSV, 'knee_proj_rankme.csv')
N_SPOTS = 100
GROUPINGS = ('shared', 'redrawn')
BN_MODES = ('eval', 'batch')                   # primary first
RANKME_VARIANTS = ('pub_uncentred', 'pub_centred', 'sq_centred')
SETS = ('redrawn', 'shared', 'fit', 'ev')
LIDAR_CHUNK, FEAT_CHUNK = 64, 256              # knee_lidar.lidar_of, encode_meanpool
N_RUNS, N_CKPT = 51, 1848  # denom 2026-10-13: + 3 runs x 38 checkpoints


# ------------------------------------------------------------------ plan

def plan():
    """LeJEPA run -> sorted epochs with an N = 100 row in results/knee_lidar.csv
    (all 58 runs minus the 10 MAE runs, which have no projector)."""
    runs = OrderedDict()
    with open(SRC) as fh:
        for r in csv.DictReader(fh):
            if r['n_spots'] == str(N_SPOTS) and '_mae_' not in r['run']:
                runs.setdefault(r['run'], set()).add(int(r['epoch']))
    P = OrderedDict((k, sorted(v)) for k, v in runs.items())
    assert len(P) == N_RUNS, len(P)
    assert sum(len(v) for v in P.values()) == N_CKPT
    return P


def out_path(run):
    return os.path.join(OUT_DIR, f'{run}.pt')


def ref_lidar():
    """(grouping, run, epoch) -> pooled LiDAR at N = 100 from the two CSVs."""
    out = {}
    for g, p in (('redrawn', SRC), ('shared', SHARED_CSV)):
        with open(p) as fh:
            for r in csv.DictReader(fh):
                if r['n_spots'] == str(N_SPOTS):
                    out[(g, r['run'], int(r['epoch']))] = float(r['lidar'])
    return out


def ref_rankme():
    """(run, epoch, variant) -> pooled RankMe from knee_labelfree_metrics.csv."""
    out = {}
    with open(MET_CSV) as fh:
        for r in csv.DictReader(fh):
            if r['metric'] == 'rankme':
                out[(r['run'], int(r['epoch']), r['variant'])] = float(r['value'])
    return out


# ------------------------------------------------------------------ model path

def load_cfg(run):
    import yaml
    import wp1_probe_driver as drv
    return yaml.safe_load(open(os.path.join(drv.ROOT, run, 'config.yml')))


def ckpt_path(run, cfg, ep):
    import wp1_probe_driver as drv
    return os.path.join(drv.ROOT, run, f"{cfg['logging']['write_tag']}-ep{ep}.pth.tar")


def encoder_is_target(d):
    """True if ckpt['encoder'] and ckpt['target_encoder'] hold the same
    tensors (train_lejepa.py:494-501 saves one dict under both keys)."""
    import torch
    a, b = d['encoder'], d['target_encoder']
    if a is b:
        return True
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def build_projector(cfg, enc, d):
    """train_lejepa.py:399-401 exactly: LeJEPAProjector(encoder.embed_dim,
    hidden_dim=proj_hidden_dim, proj_dim=proj_dim) with the config's values
    (defaults train_lejepa.py:255-256); the DDP 'module.' prefix stripped;
    strict load; eval mode."""
    import wp1_probe_driver as drv
    from src.lejepa import LeJEPAProjector
    l_cfg = cfg['meta'].get('lejepa') or {}
    prj = LeJEPAProjector(enc.embed_dim,
                          hidden_dim=int(l_cfg.get('proj_hidden_dim', 2048)),
                          proj_dim=int(l_cfg.get('proj_dim', 16)))
    sd = {}
    for k, v in d['projector'].items():
        assert k.startswith('module.'), k
        sd[k[len('module.'):]] = v
    assert len(sd) == 16, len(sd)
    prj.load_state_dict(sd, strict=True)
    return prj.to(drv.DEV).eval()


def tokens_pass(enc, imgs, chunk):
    """One encoder pass, two poolings: (LayerNorm -> token mean) = the pooled
    representation of every existing series (ops identical to
    knee_lidar.lidar_of and wp1_probe_driver.encode_meanpool), and the plain
    token mean = pool_tokens, the projector's input in training."""
    import torch
    import torch.nn.functional as F
    import wp1_probe_driver as drv
    ln, plain = [], []
    enc.eval()
    with torch.no_grad():
        for i in range(0, imgs.shape[0], chunk):
            h = enc(imgs[i:i + chunk].to(drv.DEV, torch.float32)).float()
            ln.append(F.layer_norm(h, (h.size(-1),)).mean(dim=1))
            plain.append(h.mean(dim=1))
    return torch.cat(ln), torch.cat(plain)


def proj_forward(prj, h, bn_mode):
    """Projector output in one BatchNorm mode. 'eval': the module in eval mode
    (saved running statistics). 'batch': the same layers, each BatchNorm1d
    replaced by F.batch_norm with batch statistics over all rows of h and no
    running statistics, so nothing stored is read or written."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    assert not prj.training
    with torch.no_grad():
        if bn_mode == 'eval':
            return prj(h)
        assert bn_mode == 'batch'
        x = h
        for m in prj.mlp:
            if isinstance(m, nn.BatchNorm1d):
                x = F.batch_norm(x, None, None, m.weight, m.bias, training=True,
                                 momentum=0.0, eps=m.eps)
            else:
                x = m(x)
        return x


def load_replica_sets():
    """The cached replica sets the pooled series used (read only)."""
    import torch
    import knee_lidar as KL
    import knee_lidar_sharedshape as KLS
    paths = {'redrawn': os.path.join(KL.CACHE, f'knee_lidar_N{N_SPOTS}.pt'),
             'shared': KLS.cache_path(N_SPOTS)}
    out = {}
    for g, p in paths.items():
        assert os.path.exists(p), p
        out[g] = torch.load(p, map_location='cpu', weights_only=True)
        assert tuple(out[g].shape) == (KL.M * KL.K, 4, 64, 64), out[g].shape
        print(f'replica set {g}: {p} sha256 {KLS.set_digest(out[g])}', flush=True)
    return out


# ------------------------------------------------------------------ stage 0

def stage0():
    """Training-path checks on a few checkpoints: encoder == target_encoder,
    projector geometry and strict load, 'batch' mode = a train-mode copy of
    the projector on the same rows with the original's statistics untouched,
    the training-loss scale reproduced from the loaded path (and not from the
    LayerNorm-pooled input), the bf16 effect, and timing."""
    import copy
    import torch
    import knee_lidar as KL
    import knee_dir_sweep as DS
    from src.lejepa import build_sigreg, lejepa_inv_loss
    drv = KL.drv
    P = plan()
    IM = load_replica_sets()
    picks = [('rigb_lejp_p1s100_nano_1xxx', (20, 90, 150)),
             ('rigb_lejp_p1s100_p384_2xxx', (60, 300)),
             ('rigb_lejp_p1s100_tiny_1xxx', (60, 300)),
             ('rigd_lejp_p1s100_mu04_nano_3xxx', (130,))]
    caches = {}
    for pref, eps in picks:
        run = next(r for r in P if r.startswith(pref))
        cfg = load_cfg(run)
        l_cfg = cfg['meta'].get('lejepa') or {}
        diag = {}
        with open(os.path.join(drv.ROOT, run, 'lejepa_diag_r0.csv')) as fh:
            for r in csv.DictReader(fh):
                diag.setdefault(int(r['epoch']), []).append(r)
        cname = DS.cache_for(run.split('xxx')[0])
        if cname not in caches:
            caches[cname] = torch.load(os.path.join(CACHE_DIR, cname), map_location='cpu',
                                       weights_only=True)
        sets = caches[cname]
        print(f'\n=== {run}: model {cfg["meta"]["model_name"]}, masking '
              f'{l_cfg.get("masking", "none")}, views '
              f'{cfg["data"]["calo_cfg"]["lejepa_latents"].get("views")}, proj hidden '
              f'{l_cfg.get("proj_hidden_dim")}, proj_dim {l_cfg.get("proj_dim")}, bf16 '
              f'{cfg["meta"]["use_bfloat16"]}, monitor seed {l_cfg.get("monitor_slice_seed")}, '
              f'fitting cache {cname}')
        for ep in eps:
            t0 = time.time()
            ck = ckpt_path(run, cfg, ep)
            d = torch.load(ck, map_location='cpu', weights_only=False)
            same = encoder_is_target(d)
            enc = drv.encoder_from_ckpt(cfg, ck)
            prj = build_projector(cfg, enc, d)
            shapes = {k: tuple(v.shape) for k, v in prj.state_dict().items()}
            print(f'-- ep{ep} (ckpt epoch field {d["epoch"]}): encoder == target_encoder: {same}; '
                  f'embed_dim {enc.embed_dim}; projector mlp.0 {shapes["mlp.0.weight"]}, '
                  f'mlp.4 {shapes["mlp.4.weight"]}, mlp.8 {shapes["mlp.8.weight"]}; '
                  f'BN num_batches_tracked {int(prj.state_dict()["mlp.1.num_batches_tracked"])}; '
                  f'strict load OK')
            # (a) 'batch' mode = train-mode copy; original running stats untouched
            ln_fit, plain_fit = tokens_pass(enc, sets['imgs_tr'], FEAT_CHUNK)
            sd0 = {k: v.clone() for k, v in prj.state_dict().items()}
            zb = proj_forward(prj, plain_fit, 'batch')
            cp = copy.deepcopy(prj).train()
            with torch.no_grad():
                zt = cp(plain_fit)
            untouched = all(torch.equal(v, prj.state_dict()[k]) for k, v in sd0.items())
            moved = not torch.equal(cp.state_dict()['mlp.1.running_mean'], sd0['mlp.1.running_mean'])
            ze = proj_forward(prj, plain_fit, 'eval')
            print(f'   batch mode vs deep-copied projector in train(): max|diff| '
                  f'{float((zb - zt).abs().max()):.2e}; original stats untouched {untouched}; '
                  f'the copy\'s running_mean moved {moved}; eval vs batch max|diff| '
                  f'{float((ze - zb).abs().max()):.3f}')
            # (b) training-loss scale from the loaded path
            sig = build_sigreg(num_slices=int(l_cfg.get('num_slices', 256)),
                               t_max=float(l_cfg.get('t_max', 3)),
                               n_points=int(l_cfg.get('n_points', 17))).to(drv.DEV)
            seed = int(l_cfg.get('monitor_slice_seed', 20260816))
            nb = 4000 // 128
            ln_sh, plain_sh = tokens_pass(enc, IM['shared'], LIDAR_CHUNK)

            def scale(inp_fit, inp_sh, mode):
                z = proj_forward(prj, inp_fit, mode)
                s = sum(float(sig.eval_statistic(z[i * 128:(i + 1) * 128][None], seed=seed))
                        for i in range(nb)) / nb
                zs = proj_forward(prj, inp_sh, mode).reshape(KL.M, KL.K, -1)
                pairs = torch.stack([zs[:, 0::2].reshape(-1, zs.shape[-1]),
                                     zs[:, 1::2].reshape(-1, zs.shape[-1])])   # (2, 256, 16)
                return s, float(lejepa_inv_loss(pairs))
            logged = diag.get(ep - 1, [])
            li = sum(float(r['lejepa_inv']) for r in logged) / max(1, len(logged))
            ls = sum(float(r['sigreg_eval']) for r in logged) / max(1, len(logged))
            print(f'   logged in training, epoch {ep - 1} (the last one before this checkpoint; '
                  f'{len(logged)} rows, mean): lejepa_inv {li:.5f}, sigreg_eval {ls:.4f} '
                  f'(null mean 1.056)')
            for lab, a, b in (('plain token mean (training path)', plain_fit, plain_sh),
                              ('LayerNorm then mean (pooled path; counterfactual)', ln_fit, ln_sh)):
                for mode in BN_MODES:
                    s, inv = scale(a, b, mode)
                    print(f'   input {lab:50s} BN {mode:5s}: sigreg_eval on 31 fitting-split batches '
                          f'of 128 {s:9.4f}; inv on 256 shared-shape replica pairs {inv:.5f}')
            # (c) bf16 autocast (the training precision) vs fp32, projector LiDAR eval mode
            for g in GROUPINGS:
                _, pl = tokens_pass(enc, IM[g], LIDAR_CHUNK)
                l32 = KL.lidar(proj_forward(prj, pl, 'eval').reshape(KL.M, KL.K, -1))
                with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    hs = []
                    for i in range(0, IM[g].shape[0], LIDAR_CHUNK):
                        hs.append(enc(IM[g][i:i + LIDAR_CHUNK].to(drv.DEV, torch.float32)).mean(dim=1))
                    zbf = prj(torch.cat(hs)).float()
                l16 = KL.lidar(zbf.reshape(KL.M, KL.K, -1))
                print(f'   projector LiDAR {g:7s} eval: fp32 {l32:.4f}, bf16 autocast {l16:.4f}')
            del enc, prj
            print(f'   ({time.time() - t0:.1f} s for this checkpoint incl. checks)', flush=True)


# ------------------------------------------------------------------ stage 1

def gate():
    """Stage 1: the pooled LiDAR (knee_lidar.lidar_of, unchanged) and pooled
    RankMe (the label-free module's functions on
    wp1_probe_driver.encode_meanpool features) reproduce the CSVs."""
    import numpy as np
    import torch
    import knee_lidar as KL
    import knee_labelfree_metrics as LF
    import knee_dir_sweep as DS
    drv = KL.drv
    P = plan()
    IM = load_replica_sets()
    RL, RR = ref_lidar(), ref_rankme()
    picks = [('rigb_lejp_p1s100_nano_1xxx', (2, 90)),
             ('rigb_lejp_p1s100_p384_2xxx', (140, 300)),
             ('rigb_lejp_p1s100_tiny_1xxx', (300,)),
             ('rigb_lejp_p1s100_tiny_2xxx', (150,)),
             ('rigd_lejp_p1s100_mu097_p384_1xxx', (200,)),
             ('rigd_lejp_p1s100_mu04_nano_3xxx', (130,))]
    worst_l, worst_r, worst_f = 0.0, {}, 0.0
    n_l = n_r = 0
    caches = {}
    for pref, eps in picks:
        run = next(r for r in P if r.startswith(pref))
        cfg = load_cfg(run)
        cname = DS.cache_for(run.split('xxx')[0])
        if cname not in caches:
            caches[cname] = torch.load(os.path.join(CACHE_DIR, cname), map_location='cpu',
                                       weights_only=True)
        Fst = torch.load(os.path.join(FEAT_DIR, f'{run}.pt'), map_location='cpu', weights_only=True)
        assert Fst['cache'] == cname, (Fst['cache'], cname)
        for ep in eps:
            enc = drv.encoder_from_ckpt(cfg, ckpt_path(run, cfg, ep))
            for g in GROUPINGS:
                v = KL.lidar_of(enc, IM[g])
                ref = RL[(g, run, ep)]
                worst_l = max(worst_l, abs(v - ref))
                n_l += 1
                print(f'LiDAR  {run.split("xxx")[0]:34s} ep{ep:<4d} {g:7s} csv {ref:.6f} now {v:.6f} '
                      f'|d| {abs(v - ref):.1e}', flush=True)
            X = drv.encode_meanpool(enc, caches[cname]['imgs_tr']).float()
            fd = float((X - Fst['feats_tr'][Fst['epochs'].index(ep)]).abs().max())
            worst_f = max(worst_f, fd)
            Xn = X.numpy()
            vals = {'pub_uncentred': LF.rankme_pub(Xn, False), 'pub_centred': LF.rankme_pub(Xn, True),
                    'sq_centred': LF.rankme_sq_centred(Xn)}
            n_r += 1
            cells = []
            for v in RANKME_VARIANTS:
                ref = RR[(run, ep, v)]
                d = abs(vals[v] - ref) / abs(ref)
                worst_r[v] = max(worst_r.get(v, 0.0), d)
                cells.append(f'{v} csv {ref:.10g} now {vals[v]:.10g} rel|d| {d:.1e}')
            print(f'RankMe {run.split("xxx")[0]:34s} ep{ep:<4d} pooled features vs stored max|d| '
                  f'{fd:.1e}; ' + '; '.join(cells), flush=True)
            del enc
    print(f'\nGATE LiDAR: {n_l} values ({len(picks)} runs), max |now - csv| = {worst_l:.2e} '
          f'(CSVs print 6 decimals)')
    print(f'GATE features: max |fresh - stored labelfree_feats| = {worst_f:.2e}')
    print('GATE RankMe: ' + ', '.join(f'{v} max rel |d| {w:.2e}' for v, w in worst_r.items())
          + f' over {n_r} checkpoints (CSV prints 10 significant digits)')
    assert np.isfinite(worst_l)


# ------------------------------------------------------------------ stage 2

def run_pass(only=None):
    import torch
    import knee_lidar as KL
    import knee_labelfree_metrics as LF
    import knee_dir_sweep as DS
    drv = KL.drv
    P = plan()
    runs = list(P) if only is None else [r for r in P if any(r.startswith(o) for o in only)]
    IM = load_replica_sets()
    RL = ref_lidar()
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f'GPU {torch.cuda.get_device_name(0)} (CUDA_VISIBLE_DEVICES='
          f'{os.environ.get("CUDA_VISIBLE_DEVICES")}); {len(runs)} runs, '
          f'{sum(len(P[r]) for r in runs)} checkpoints -> {OUT_DIR}', flush=True)
    caches = {}
    t_all = time.time()
    for run in runs:
        out = out_path(run)
        if os.path.exists(out):
            print(f'{run}: exists, skip', flush=True)
            continue
        t0 = time.time()
        eps = P[run]
        short = run.split('xxx')[0]
        cfg = load_cfg(run)
        Fst = torch.load(os.path.join(FEAT_DIR, f'{run}.pt'), map_location='cpu', weights_only=True)
        assert Fst['epochs'] == eps, run
        cname = DS.cache_for(short)
        assert Fst['cache'] == cname, (Fst['cache'], cname)
        if cname not in caches:
            caches[cname] = torch.load(os.path.join(CACHE_DIR, cname), map_location='cpu',
                                       weights_only=True)
        sets = caches[cname]
        imgs = {'redrawn': (IM['redrawn'], LIDAR_CHUNK), 'shared': (IM['shared'], LIDAR_CHUNK),
                'fit': (sets['imgs_tr'], FEAT_CHUNK), 'ev': (sets['imgs_ev'], FEAT_CHUNK)}
        emb = {f'{s}_{b}': [] for s in SETS for b in BN_MODES}
        lid = {f'{g}_{b}': [] for g in GROUPINGS for b in BN_MODES}
        rk = {f'{v}_{b}': [] for v in RANKME_VARIANTS for b in BN_MODES}
        chk = {'enc_is_target': [], 'bn_untouched': [], 'pooled_lidar_redrawn': [],
               'pooled_lidar_shared': [], 'pooled_lidar_absdiff_redrawn': [],
               'pooled_lidar_absdiff_shared': [], 'fit_pooled_maxdiff': [], 'ev_pooled_maxdiff': [],
               'sec': []}
        l_cfg = cfg['meta'].get('lejepa') or {}
        for i, ep in enumerate(eps):
            t1 = time.time()
            ck = ckpt_path(run, cfg, ep)
            d = torch.load(ck, map_location='cpu', weights_only=False)
            chk['enc_is_target'].append(bool(encoder_is_target(d)))
            enc = drv.encoder_from_ckpt(cfg, ck)
            prj = build_projector(cfg, enc, d)
            del d
            sd0 = {k: v.clone() for k, v in prj.state_dict().items()}
            for s in SETS:
                x, chunk = imgs[s]
                ln, plain = tokens_pass(enc, x, chunk)
                if s in GROUPINGS:
                    v = KL.lidar(ln.reshape(KL.M, KL.K, -1))       # = knee_lidar.lidar_of
                    chk[f'pooled_lidar_{s}'].append(v)
                    chk[f'pooled_lidar_absdiff_{s}'].append(abs(v - RL[(s, run, ep)]))
                else:
                    ref = Fst['feats_tr' if s == 'fit' else 'feats_ev'][i]
                    chk[f'{s}_pooled_maxdiff'].append(float((ln.cpu() - ref).abs().max()))
                for b in BN_MODES:
                    z = proj_forward(prj, plain, b).float()
                    emb[f'{s}_{b}'].append(z.cpu())
                    if s in GROUPINGS:
                        lid[f'{s}_{b}'].append(KL.lidar(z.reshape(KL.M, KL.K, -1)))
                    elif s == 'fit':
                        X = z.cpu().numpy()
                        rk[f'pub_uncentred_{b}'].append(LF.rankme_pub(X, False))
                        rk[f'pub_centred_{b}'].append(LF.rankme_pub(X, True))
                        rk[f'sq_centred_{b}'].append(LF.rankme_sq_centred(X))
            chk['bn_untouched'].append(all(torch.equal(v, prj.state_dict()[k]) for k, v in sd0.items()))
            chk['sec'].append(time.time() - t1)
            del enc, prj
        obj = {'run': run, 'epochs': list(eps), 'write_tag': cfg['logging']['write_tag'],
               'model_name': cfg['meta']['model_name'],
               'proj_hidden_dim': int(l_cfg.get('proj_hidden_dim', 2048)),
               'proj_dim': int(l_cfg.get('proj_dim', 16)), 'fit_cache': cname,
               'replica_sets': {'redrawn': f'knee_lidar_N{N_SPOTS}.pt',
                                'shared': f'knee_lidar_sharedshape_N{N_SPOTS}.pt'},
               'emb': {k: torch.stack(v) for k, v in emb.items()},
               'lidar_proj': lid, 'rankme_proj': rk, 'check': chk,
               'script': 'knee_proj_lidar_rankme.py', 'created': time.strftime('%Y-%m-%d %H:%M:%S')}
        assert not os.path.exists(out), out
        part = out + '.partial'
        if os.path.exists(part):                      # own leftover from a stopped run: keep it
            os.replace(part, part + f'.stale{int(time.time())}')
        torch.save(obj, part)
        os.replace(part, out)
        mx = lambda k: max(chk[k])                                  # noqa: E731
        print(f'{short:38s} {len(eps)} ckpts D={obj["emb"]["fit_eval"].shape[-1]} | pooled LiDAR '
              f'max|d| redrawn {mx("pooled_lidar_absdiff_redrawn"):.1e} shared '
              f'{mx("pooled_lidar_absdiff_shared"):.1e}; pooled feats max|d| fit '
              f'{mx("fit_pooled_maxdiff"):.1e} ev {mx("ev_pooled_maxdiff"):.1e}; enc==target '
              f'{sum(chk["enc_is_target"])}/{len(eps)}; BN untouched {sum(chk["bn_untouched"])}/'
              f'{len(eps)} ({time.time() - t0:.0f}s; total {time.time() - t_all:.0f}s)', flush=True)
    print('DONE knee_proj_lidar_rankme pass')


# ------------------------------------------------------------------ checks (CPU)

def sw_spectrum(Z, M, K):
    """Eigenvalues (ascending) of lidar()'s within-group covariance WITHOUT
    the +delta I, and of its between-group covariance, float64."""
    import torch
    Z = Z.double().reshape(M, K, -1)
    mu_c = Z.mean(1)
    mu = mu_c.mean(0)
    Sb = (mu_c - mu).T @ (mu_c - mu) / (M - 1)
    Xw = (Z - mu_c[:, None, :]).reshape(-1, Z.shape[-1])
    Sw = Xw.T @ Xw / (M * (K - 1))
    return torch.linalg.eigvalsh(Sw), torch.linalg.eigvalsh(Sb)


CHECK_DIR = os.path.join(OUT_DIR, 'check')
SENS_CSV = os.path.join(paths.CSV, 'knee_proj_lidar_delta.csv')
SHARE_CSV = os.path.join(paths.CSV, 'knee_proj_lidar_delta_share.csv')
SW_CSV = os.path.join(CHECK_DIR, 'proj_lidar_within_spectrum.csv')
DELTAS = (1e-4, 1e-3, 1e-5, 1e-6, 1e-8)        # lidar()'s default first


def leading_direction(Z, M, K, delta):
    """lidar()'s steps in float64 (analysis/lidar_metric.py: Sb, Sw +
    delta I, W = (Sw + delta I)^-1/2, eigenvalues of W Sb W), returning for the
    leading eigenvector u of W Sb W the feature-space direction v = W u (the
    top generalised eigenvector of (Sb, Sw + delta I)) and: the raw
    within-group variance along v/|v| (Sw without delta), delta's share of the
    denominator v'(Sw + delta I)v / |v|^2, the top eigenvalue, its share of
    the eigenvalue sum, and the LiDAR value (checked against lidar())."""
    import torch
    Z = Z.double().reshape(M, K, -1)
    D = Z.shape[-1]
    mu_c = Z.mean(1)
    mu = mu_c.mean(0)
    Sb = (mu_c - mu).T @ (mu_c - mu) / (M - 1)
    Xw = (Z - mu_c[:, None, :]).reshape(-1, D)
    Sw_raw = Xw.T @ Xw / (M * (K - 1))
    ew, U = torch.linalg.eigh(Sw_raw + delta * torch.eye(D, dtype=Z.dtype))
    W = U @ torch.diag(ew.clamp(min=1e-12).rsqrt()) @ U.T
    lam, V = torch.linalg.eigh(W @ Sb @ W)
    lam = lam.clamp(min=0)
    v = W @ V[:, -1]
    vh = v / v.norm()
    s_raw = float(vh @ Sw_raw @ vh)
    p = lam / (lam.sum() + 1e-12) + 1e-12
    lid = float(torch.exp(-(p * torch.log(p)).sum()))
    return {'sw_lead': s_raw, 'sw_lead_over_delta': s_raw / delta,
            'delta_share': delta / (s_raw + delta), 'lam_top': float(lam[-1]),
            'top_share': float(lam[-1] / lam.sum()), 'lidar': lid}


def delta_share():
    """Added 2026-09-24: at PRE (last grid epoch <= onset - 20)
    and POST (final checkpoint) of the 27 knee runs, and at the final
    checkpoint of the 8 mu = 0.97 controls, the within-group variance along
    lidar()'s leading whitened direction in units of delta, and delta's share
    of that direction's denominator; delta = 1e-4 (lidar()'s default, the
    primary series), per grouping and BN mode. Writes SHARE_CSV (new)."""
    import torch
    import knee_lidar as KL
    import knee_lidar_audit as AUD
    assert not os.path.exists(SHARE_CSV), f'refusing to overwrite {SHARE_CSV}'
    P = plan()
    full = {r.split('xxx')[0]: r for r in P}
    onset = dict(AUD.KNEE_21)
    onset.update(AUD.KNEE_4)
    todo = []
    for k, o in onset.items():
        eps = P[full[k]]
        todo.append((k, 'knee', 'pre', max(e for e in eps if e <= o - 20)))
        todo.append((k, 'knee', 'post', max(eps)))
    for k in AUD.CTRL_21 + AUD.CTRL_4:
        todo.append((k, 'control', 'final', max(P[full[k]])))
    assert len(todo) == 27 * 2 + 8
    rows, worst = [], 0.0
    cache = {}
    for k, role, phase, ep in todo:
        if k not in cache:
            cache = {k: torch.load(out_path(full[k]), map_location='cpu', weights_only=True)}
        O = cache[k]
        i = O['epochs'].index(ep)
        for g in GROUPINGS:
            for b in BN_MODES:
                Z = O['emb'][f'{g}_{b}'][i]
                r = leading_direction(Z, KL.M, KL.K, 1e-4)
                ref = KL.lidar(Z.double().reshape(KL.M, KL.K, -1))
                worst = max(worst, abs(r['lidar'] - ref))
                rows.append([full[k], role, phase, ep, g, b, f"{r['sw_lead']:.6e}",
                             f"{r['sw_lead_over_delta']:.6g}", f"{r['delta_share']:.6f}",
                             f"{r['lam_top']:.6g}", f"{r['top_share']:.6f}", f"{r['lidar']:.6f}"])
    print(f'leading-direction LiDAR vs lidar() float64: max |d| {worst:.2e}')
    cols = ['run', 'role', 'phase', 'epoch', 'grouping', 'bn_mode', 'sw_lead', 'sw_lead_over_delta',
            'delta_share', 'lam_top', 'top_share', 'lidar_f64']
    with open(SHARE_CSV + '.partial', 'w', newline='') as fh:
        w = csv.writer(fh, lineterminator='\n')
        w.writerow(cols)
        w.writerows(rows)
    assert not os.path.exists(SHARE_CSV)
    os.replace(SHARE_CSV + '.partial', SHARE_CSV)
    print(f'wrote {SHARE_CSV}: {len(rows)} rows')


def check():
    """From the stored files, no GPU: completeness, non-finite counts, the
    per-checkpoint checks, LiDAR recomputed on CPU (float32 as the pass, and
    float64) against the GPU values, RankMe recomputed, and the within-group
    covariance spectrum of every projector LiDAR evaluation. When all 48 files
    exist, also writes (new files, refuse to overwrite) proj_feats/check/
    proj_lidar_f64_delta.csv (float64 LiDAR with lidar()'s delta = 1e-4 and
    with 1e-5, 1e-3; the same imported lidar(), its delta argument) and
    proj_lidar_within_spectrum.csv (per evaluation: smallest and largest
    eigenvalue and trace of the within-group covariance without +delta I,
    largest between-group eigenvalue)."""
    import numpy as np
    import torch
    import knee_lidar as KL
    import knee_labelfree_metrics as LF
    P = plan()
    sens_rows, sw_rows = [], []
    n_gt = {1e-3: 0, 1e-4: 0}
    missing = [r for r in P if not os.path.exists(out_path(r))]
    print(f'{N_RUNS - len(missing)} / {N_RUNS} run files present' +
          (f'; missing: {missing}' if missing else ''))
    delta = 1e-4
    agg = {'nonfinite_emb': 0, 'nonfinite_lidar': 0, 'nonfinite_rankme': 0, 'enc_not_target': 0,
           'bn_touched': 0, 'n_ckpt': 0}
    worst = {'pooled_lidar_absdiff_redrawn': 0.0, 'pooled_lidar_absdiff_shared': 0.0,
             'fit_pooled_maxdiff': 0.0, 'ev_pooled_maxdiff': 0.0,
             'cpu32_vs_gpu': 0.0, 'cpu64_vs_gpu': 0.0, 'rankme_recompute': 0.0}
    sw = {f'{g}_{b}': [] for g in GROUPINGS for b in BN_MODES}
    sw_pool = {}
    rows_sw = []
    for run in P:
        if run in missing:
            continue
        O = torch.load(out_path(run), map_location='cpu', weights_only=True)
        assert O['epochs'] == P[run]
        n = len(O['epochs'])
        agg['n_ckpt'] += n
        for k, T in O['emb'].items():
            assert T.dtype == torch.float32 and T.shape[0] == n and T.shape[-1] == 16, (k, T.shape)
            agg['nonfinite_emb'] += int((~torch.isfinite(T)).sum())
        c = O['check']
        agg['enc_not_target'] += sum(not x for x in c['enc_is_target'])
        agg['bn_touched'] += sum(not x for x in c['bn_untouched'])
        for k in ('pooled_lidar_absdiff_redrawn', 'pooled_lidar_absdiff_shared',
                  'fit_pooled_maxdiff', 'ev_pooled_maxdiff'):
            worst[k] = max(worst[k], max(c[k]))
        for k, v in O['lidar_proj'].items():
            agg['nonfinite_lidar'] += sum(not math.isfinite(x) for x in v)
        for k, v in O['rankme_proj'].items():
            agg['nonfinite_rankme'] += sum(not math.isfinite(x) for x in v)
        for i, ep in enumerate(O['epochs']):
            for g in GROUPINGS:
                for b in BN_MODES:
                    Z = O['emb'][f'{g}_{b}'][i]
                    gpu = O['lidar_proj'][f'{g}_{b}'][i]
                    c32 = KL.lidar(Z.reshape(KL.M, KL.K, -1))
                    c64 = {dl: KL.lidar(Z.double().reshape(KL.M, KL.K, -1), delta=dl) for dl in DELTAS}
                    worst['cpu32_vs_gpu'] = max(worst['cpu32_vs_gpu'], abs(c32 - gpu))
                    worst['cpu64_vs_gpu'] = max(worst['cpu64_vs_gpu'], abs(c64[1e-4] - gpu))
                    for t in n_gt:
                        n_gt[t] += abs(c64[1e-4] - gpu) > t
                    ew, eb = sw_spectrum(Z, KL.M, KL.K)
                    sw[f'{g}_{b}'].append((float(ew[0]), float(ew[-1]), float(eb[-1]), float(ew.sum())))
                    rows_sw.append((run, ep, g, b, float(ew[0]), float(ew[-1]), float(ew.sum())))
                    sens_rows.append([run, ep, g, b] + [f'{c64[dl]:.6f}' for dl in DELTAS])
                    sw_rows.append([run, ep, g, b, f'{float(ew[0]):.6e}', f'{float(ew[-1]):.6e}',
                                    f'{float(ew.sum()):.6e}', f'{float(eb[-1]):.6e}'])
            if i in (0, n - 1) or i % 10 == 0:          # RankMe recomputed on a subset
                for b in BN_MODES:
                    X = O['emb'][f'fit_{b}'][i].numpy()
                    for v, val in (('pub_uncentred', LF.rankme_pub(X, False)),
                                   ('pub_centred', LF.rankme_pub(X, True)),
                                   ('sq_centred', LF.rankme_sq_centred(X))):
                        worst['rankme_recompute'] = max(worst['rankme_recompute'],
                                                        abs(val - O['rankme_proj'][f'{v}_{b}'][i]))
    print(f"checkpoints: {agg['n_ckpt']} (expected {N_CKPT})")
    print(f"non-finite: embeddings {agg['nonfinite_emb']}, projector LiDAR values "
          f"{agg['nonfinite_lidar']}, projector RankMe values {agg['nonfinite_rankme']}")
    print(f"encoder != target_encoder at {agg['enc_not_target']} checkpoints; BN running stats "
          f"changed at {agg['bn_touched']}")
    for k, w in worst.items():
        print(f'  max {k:32s} {w:.2e}')
    print(f'  projector LiDAR evaluations with |float64 CPU - GPU float32| > 1e-3: {n_gt[1e-3]}, '
          f'> 1e-4: {n_gt[1e-4]} (of {agg["n_ckpt"] * 4})')
    if not missing:
        os.makedirs(CHECK_DIR, exist_ok=True)
        for p, cols, rows in (
                (SENS_CSV, ['run', 'epoch', 'grouping', 'bn_mode'] + [f'lidar_f64_delta{dl:g}' for dl in DELTAS],
                 sens_rows),
                (SW_CSV, ['run', 'epoch', 'grouping', 'bn_mode', 'sw_min', 'sw_max', 'sw_trace', 'sb_max'],
                 sw_rows)):
            assert not os.path.exists(p), f'refusing to overwrite {p}'
            with open(p + '.partial', 'w', newline='') as fh:
                w = csv.writer(fh, lineterminator='\n')
                w.writerow(cols)
                w.writerows(rows)
            os.replace(p + '.partial', p)
            print(f'wrote {p}: {len(rows)} rows')
    print(f'\nwithin-group covariance of the projector output, WITHOUT lidar()\'s +{delta:g} I '
          f'(eigenvalues over {agg["n_ckpt"]} checkpoints):')
    for k, v in sw.items():
        mins = np.array([x[0] for x in v])
        maxs = np.array([x[1] for x in v])
        tr = np.array([x[3] for x in v])
        print(f'  {k:14s} smallest eig: min {mins.min():.2e}, median {np.median(mins):.2e}; '
              f'largest eig median {np.median(maxs):.2e}; condition number median '
              f'{np.median(maxs / mins):.2e}, max {np.max(maxs / mins):.2e}; trace median '
              f'{np.median(tr):.2e}; checkpoints with smallest eig < delta: {(mins < delta).sum()}, '
              f'< 10 delta: {(mins < 10 * delta).sum()}')
    return rows_sw


# ------------------------------------------------------------------ CSVs

def write_csv():
    import torch
    for p in (OUT_LIDAR, OUT_RANKME):
        assert not os.path.exists(p), f'refusing to overwrite {p}'
    P = plan()
    lrows, rrows = [], []
    for run in P:
        O = torch.load(out_path(run), map_location='cpu', weights_only=True)
        assert O['epochs'] == P[run]
        for i, ep in enumerate(O['epochs']):
            for g in GROUPINGS:
                for b in BN_MODES:
                    lrows.append([run, ep, g, b, f"{O['lidar_proj'][f'{g}_{b}'][i]:.6f}"])
            for v in RANKME_VARIANTS:
                for b in BN_MODES:
                    rrows.append([run, ep, v, b, f"{O['rankme_proj'][f'{v}_{b}'][i]:.10g}"])
    assert len(lrows) == N_CKPT * 4 and len(rrows) == N_CKPT * 6
    for p, cols, rows in ((OUT_LIDAR, ['run', 'epoch', 'grouping', 'bn_mode', 'lidar'], lrows),
                          (OUT_RANKME, ['run', 'epoch', 'variant', 'bn_mode', 'value'], rrows)):
        with open(p + '.partial', 'w', newline='') as fh:
            w = csv.writer(fh, lineterminator='\n')
            w.writerow(cols)
            w.writerows(rows)
        assert not os.path.exists(p), f'refusing to overwrite {p}'
        os.replace(p + '.partial', p)
        print(f'wrote {p}: {len(rows)} rows')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage0', action='store_true')
    ap.add_argument('--gate', action='store_true')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--write-csv', action='store_true')
    ap.add_argument('--delta-share', action='store_true')
    ap.add_argument('--only', default=None, help='comma list of run-name prefixes (pass only)')
    a = ap.parse_args()
    if a.write_csv:
        write_csv()
    elif a.delta_share:
        delta_share()
    elif a.check:
        check()
    else:
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            raise SystemExit('set CUDA_VISIBLE_DEVICES explicitly (shared GPUs)')
        sys.path.insert(0, paths.TRAIN)
        if a.stage0:
            stage0()
        elif a.gate:
            gate()
        else:
            run_pass(None if a.only is None else a.only.split(','))
