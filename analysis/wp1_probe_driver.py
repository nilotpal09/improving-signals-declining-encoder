"""WP1 probe driver — scores checkpoints with the M1-M4 identifiability
metrics against KNOWN latents.

Because the WP1 rig controls the latent draw (src/lejepa_data.py), the probe
sets are generated WITH their z recorded — no reconstruction from alpha/T is
needed, and the probe truth is exact by construction (the Eq.-11 injection
is inversion-tested in tests/test_lejepa_data.py). Frozen probe conventions
(fixed in advance; changes after the pilot freeze are logged protocol
changes):

  * Probe sets: single-view gun events, N_TRAIN=10_000 / N_EVAL=2_000,
    disjoint pinned seeds (20260817 / 20260818), N_spots=1000 (the canonical
    probe rung), generated in chunks of 256. The probe latent distribution
    MATCHES the run's training distribution (dist/mu from the run config;
    rho is irrelevant probe-side — one view per event). Cached per
    (dist, mu) under CACHE_DIR.
  * Targets: z1, z2 (the drawn latents) and standardized lnE (train-split
    mean/std), names ('z1', 'z2', 'E').
  * Features: encoder final tokens -> LayerNorm -> mean over tokens (the
    house event pooling, `encode_meanpool` below), fp32; encoder
    from ckpt['target_encoder'] (== the online encoder on the LeJEPA arm).
  * Random-init reference at matched probe: torch.manual_seed(0), fresh
    vit_nano from the run's own config geometry (the house seed-0
    convention), scored identically, epoch column = 0.
  * Output: results/wp1_metrics.csv, one row per (run, epoch, encoder_key),
    append-mode, resumable by that key.

Usage (from analysis/):
  python wp1_probe_driver.py --run lejepa_gun_paired_nano_1xxxhdwwx
  python wp1_probe_driver.py --run <dir> --epochs 2,20,300
  python wp1_probe_driver.py --run <dir> --root <experiments_root>
"""

import argparse
import csv
import os
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
_COND = paths.TRAIN
sys.path.insert(0, _HERE)
sys.path.insert(0, _COND)

torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

from src.masks.multiblock import MaskCollator as MBMaskCollator  # noqa: E402
from src.datasets.calo_flash import CaloNorm  # noqa: E402
import src.models.vision_transformer as vit  # noqa: E402
from src.lejepa_data import LeJEPALatentCollator, sample_z  # noqa: E402
from wp1_metrics import wp1_metric_row  # noqa: E402

ROOT = paths.RUNS
CACHE_DIR = paths.CACHE
CSV_PATH = os.path.join(paths.CSV, 'wp1_metrics.csv')

N_TRAIN, N_EVAL = 10_000, 2_000
TRAIN_SEED, EVAL_SEED = 20260817, 20260818
CHUNK = 256
PROBE_N_SPOTS = 1000
DEV = 'cuda:0' if torch.cuda.is_available() else 'cpu'

COLS = ['run', 'epoch', 'encoder_key', 'dist', 'mu', 'n_train', 'n_eval',
        'mean_cca', 'cca_c1', 'cca_c2', 'procrustes_mse', 'orth_err',
        'orth_scale', 'ridge_r2_z1', 'ridge_r2_z2', 'ridge_r2_E',
        'sec_gen', 'sec_feat', 'timestamp']


def _renderer(cfg):
    """A latent collator used purely as a renderer (base collator unused)."""
    base = MBMaskCollator(input_size=(64, 64), patch_size=4,
                          pred_mask_scale=(0.15, 0.2), enc_mask_scale=(0.85, 1.0),
                          aspect_ratio=(0.75, 1.5), nenc=1, npred=4,
                          allow_overlap=False, min_keep=10)
    ccfg = dict(cfg['data']['calo_cfg'])
    ccfg.pop('paired_fixed_long', None)
    ccfg['lejepa_latents'] = {'dist': 'gaussian', 'rho': 1.0}  # gate only
    return LeJEPALatentCollator(
        base_collator=base, sample_target=True, calo_cfg=ccfg,
        generator_cfg=dict(cfg['data']['generator_cfg']),
        norm=CaloNorm(**(cfg['data'].get('norm') or {})))


def probe_dist_of(cfg):
    """The run's training latent distribution (matched-probe convention)."""
    lat = (cfg['data']['calo_cfg'] or {}).get('lejepa_latents') or {}
    return lat.get('dist', 'gaussian'), lat.get('mu')


def gen_probe_set(cfg, dist, mu, n_events, seed_base):
    """(imgs fp16 cpu (N,4,64,64), z (N,2), lnE (N,)) with z KNOWN."""
    coll = _renderer(cfg)
    imgs, zs, lnEs = [], [], []
    n_chunks = (n_events + CHUNK - 1) // CHUNK
    for c in range(n_chunks):
        n = min(CHUNK, n_events - c * CHUNK)
        seed = (seed_base * 100003 + c * 7919 + 1) & 0x7FFFFFFF
        p_E, p_x, p_y = coll._generate_truth(n, seed)
        z = sample_z(n, dist=dist, mu=mu,
                     generator=torch.Generator().manual_seed(
                         (seed * 2654435761 + 97) & 0x7FFFFFFF))
        lp = coll._lp_from_z(p_E, (seed * 7919 + 31337) & 0x7FFFFFFF, z)
        img = coll.norm.forward(coll._simulate_once(
            p_E, p_x, p_y, coll._sim_seed(seed, 0),
            n_spots_override=PROBE_N_SPOTS, long_params_override=lp))
        imgs.append(img.detach().to('cpu', torch.float16))
        zs.append(z)
        lnEs.append(torch.log(p_E.reshape(-1).detach().cpu()))
    return torch.cat(imgs), torch.cat(zs), torch.cat(lnEs)


def load_probe_sets(cfg, dist, mu):
    os.makedirs(CACHE_DIR, exist_ok=True)
    tag = f'{dist}' + (f'_mu{mu}' if dist == 'bimodal' else '')
    path = os.path.join(CACHE_DIR, f'wp1_{tag}_{N_TRAIN}_{N_EVAL}.pt')
    if os.path.exists(path):
        d = torch.load(path, map_location='cpu', weights_only=True)
        return d
    t0 = time.time()
    itr, ztr, etr = gen_probe_set(cfg, dist, mu, N_TRAIN, TRAIN_SEED)
    iev, zev, eev = gen_probe_set(cfg, dist, mu, N_EVAL, EVAL_SEED)
    mu_e, sd_e = float(etr.mean()), float(etr.std()) + 1e-8
    d = {'imgs_tr': itr, 'imgs_ev': iev,
         'y_tr': torch.cat([ztr, ((etr - mu_e) / sd_e)[:, None]], dim=1),
         'y_ev': torch.cat([zev, ((eev - mu_e) / sd_e)[:, None]], dim=1),
         'sec_gen': time.time() - t0}
    torch.save(d, path)
    return d


def encode_meanpool(enc, imgs, chunk=256):
    """Encoder tokens -> LayerNorm -> mean over tokens (house pooling)."""
    outs = []
    enc.eval()
    with torch.no_grad():
        for i in range(0, imgs.shape[0], chunk):
            x = imgs[i:i + chunk].to(DEV, torch.float32)
            h = enc(x)
            h = torch.nn.functional.layer_norm(h, (h.size(-1),))
            outs.append(h.mean(dim=1).float().cpu())
    return torch.cat(outs)


def build_encoder_from_cfg(cfg):
    """Fresh vit_nano matching the run geometry (mirrors the house
    build_encoder semantics; seed the caller controls)."""
    model_name = cfg['meta']['model_name']
    patch = cfg['mask']['patch_size']
    ccfg = cfg['data']['calo_cfg']
    return getattr(vit, model_name)(
        patch_size=patch, img_size=[ccfg['N_cells_x']],
        in_chans=ccfg['N_cells_z']).to(DEV)


def projector_from_ckpt(cfg, ckpt_path, enc):
    """(encoder, projector) probe path for the m~n arm: replicate the
    TRAINING feature path exactly — tokens -> plain mean over tokens (no
    LayerNorm; pool_tokens) -> projector (BN in eval mode, running stats)
    -> proj_dim features. Returns a callable imgs -> features."""
    import sys as _s
    _s.path.insert(0, _COND)
    from src.lejepa import LeJEPAProjector
    d = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    l_cfg = (cfg['meta'].get('lejepa') or {})
    prj = LeJEPAProjector(96, int(l_cfg.get('proj_hidden_dim', 2048)),
                          int(l_cfg.get('proj_dim', 16)))
    sd = {k.replace('module.', '', 1): v for k, v in d['projector'].items()}
    prj.load_state_dict(sd)
    prj = prj.to(DEV).eval()

    def encode(imgs, chunk=256):
        outs = []
        with torch.no_grad():
            for i in range(0, imgs.shape[0], chunk):
                x = imgs[i:i + chunk].to(DEV, torch.float32)
                h = enc(x).mean(dim=1)          # pool_tokens: plain mean
                outs.append(prj(h).float().cpu())
        return torch.cat(outs)
    return encode


def encoder_from_ckpt(cfg, ckpt_path):
    d = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    # MAE checkpoints keep the online weights under 'model' with an
    # '(module.)encoder.' prefix; every other arm under (target_)encoder
    # with a '(module.)' prefix. Strip 'module.' FIRST, then 'encoder.'.
    raw = d.get('target_encoder') or d.get('encoder') or d['model']
    sd = {}
    for k, v in raw.items():
        if k.startswith('module.'):
            k = k[len('module.'):]
        if 'model' in d and not (d.get('target_encoder') or d.get('encoder')):
            if not k.startswith('encoder.'):
                continue                      # decoder etc. — not the encoder
            k = k[len('encoder.'):]
        sd[k] = v
    torch.manual_seed(0)
    enc = build_encoder_from_cfg(cfg)
    enc.load_state_dict(sd, strict=True)
    return enc


def existing_keys():
    if not os.path.exists(CSV_PATH):
        return set()
    with open(CSV_PATH) as f:
        return {(r['run'], r['epoch'], r['encoder_key'])
                for r in csv.DictReader(f)}


def append_row(row):
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new:
            w.writeheader()
        w.writerow(row)


def score(run, cfg, sets, dist, mu, epoch, encoder_key, enc):
    t0 = time.time()
    h_tr = encode_meanpool(enc, sets['imgs_tr'])
    h_ev = encode_meanpool(enc, sets['imgs_ev'])
    sec_feat = time.time() - t0
    m = wp1_metric_row(h_tr, sets['y_tr'], h_ev, sets['y_ev'])
    row = {
        'run': run, 'epoch': epoch, 'encoder_key': encoder_key,
        'dist': dist, 'mu': '' if mu is None else mu,
        'n_train': N_TRAIN, 'n_eval': N_EVAL,
        'mean_cca': f"{m['mean_cca']:.6f}",
        'cca_c1': f"{m['cca_per_component'][0]:.6f}",
        'cca_c2': f"{m['cca_per_component'][1]:.6f}",
        'procrustes_mse': f"{m['procrustes_mse']:.6f}",
        'orth_err': f"{m['orth_err']:.6f}",
        'orth_scale': f"{m['orth_scale']:.6e}",
        'ridge_r2_z1': f"{m['ridge_r2_z1']:.6f}",
        'ridge_r2_z2': f"{m['ridge_r2_z2']:.6f}",
        'ridge_r2_E': f"{m['ridge_r2_E']:.6f}",
        'sec_gen': f"{sets.get('sec_gen', 0.0):.1f}",
        'sec_feat': f"{sec_feat:.1f}",
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    append_row(row)
    print(f"[{run} ep{epoch} {encoder_key}] mean_cca={m['mean_cca']:.4f} "
          f"orth_err={m['orth_err']:.4f} r2=({m['ridge_r2_z1']:.3f},"
          f"{m['ridge_r2_z2']:.3f},{m['ridge_r2_E']:.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='run dir name under the root')
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--epochs', default='all',
                    help='"all" or comma list of checkpoint epoch tags')
    args = ap.parse_args()

    run_dir = os.path.join(args.root, args.run)
    cfg = yaml.safe_load(open(os.path.join(run_dir, 'config.yml')))
    tag = cfg['logging']['write_tag']
    dist, mu = probe_dist_of(cfg)
    sets = load_probe_sets(cfg, dist, mu)
    done = existing_keys()

    import glob
    ckpts = sorted(glob.glob(os.path.join(run_dir, f'{tag}-ep*.pth.tar')),
                   key=lambda p: int(p.split('-ep')[-1].split('.')[0]))
    if args.epochs != 'all':
        want = {int(e) for e in args.epochs.split(',')}
        ckpts = [c for c in ckpts
                 if int(c.split('-ep')[-1].split('.')[0]) in want]

    # random-init reference at matched probe (house seed-0 convention)
    if (args.run, '0', 'random_init') not in done:
        torch.manual_seed(0)
        enc0 = build_encoder_from_cfg(cfg)
        score(args.run, cfg, sets, dist, mu, 0, 'random_init', enc0)

    for ck in ckpts:
        ep = int(ck.split('-ep')[-1].split('.')[0])
        if (args.run, str(ep), 'target_encoder') not in done:
            enc = encoder_from_ckpt(cfg, ck)
            score(args.run, cfg, sets, dist, mu, ep, 'target_encoder', enc)
        # m~n arm: additionally probe the projector output (training path)
        if (cfg['meta'].get('lejepa') or {}).get('proj_dim') == 2 and \
                (args.run, str(ep), 'projector_out') not in done:
            enc = encoder_from_ckpt(cfg, ck)
            encode = projector_from_ckpt(cfg, ck, enc)
            t0 = time.time()
            h_tr = encode(sets['imgs_tr'])
            h_ev = encode(sets['imgs_ev'])
            m = wp1_metric_row(h_tr, sets['y_tr'], h_ev, sets['y_ev'])
            row = {'run': args.run, 'epoch': ep, 'encoder_key': 'projector_out',
                   'dist': dist, 'mu': '' if mu is None else mu,
                   'n_train': N_TRAIN, 'n_eval': N_EVAL,
                   'mean_cca': f"{m['mean_cca']:.6f}",
                   'cca_c1': f"{m['cca_per_component'][0]:.6f}",
                   'cca_c2': f"{m['cca_per_component'][1]:.6f}",
                   'procrustes_mse': f"{m['procrustes_mse']:.6f}",
                   'orth_err': f"{m['orth_err']:.6f}",
                   'orth_scale': f"{m['orth_scale']:.6e}",
                   'ridge_r2_z1': f"{m['ridge_r2_z1']:.6f}",
                   'ridge_r2_z2': f"{m['ridge_r2_z2']:.6f}",
                   'ridge_r2_E': f"{m['ridge_r2_E']:.6f}",
                   'sec_gen': '0.0', 'sec_feat': f"{time.time() - t0:.1f}",
                   'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}
            append_row(row)
            print(f"[{args.run} ep{ep} projector_out] "
                  f"mean_cca={m['mean_cca']:.4f} orth={m['orth_err']:.4f}")


if __name__ == '__main__':
    main()
