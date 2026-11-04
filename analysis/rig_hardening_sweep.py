"""Training-free calibration sweep for the WP1 rig-hardening study.

Evaluates 16 MULTI-PARTICLE rig variants (4 particle-configs x 2 E-bands x
2 spot-counts) where every particle in an event shares the event's latent
z = (z1, z2) (the LeJEPALatentCollator._lp_from_z construction, multi-
particle-enabled 2026-08-17). Per variant, on freshly generated gun probe
sets (4000 train / 2000 eval, GAUSSIAN latents, seed bases 20260910 /
20260911 — disjoint from the study's pinned 20260817/20260818 and the
N_spots sweep's 20260901/20260902):

  FLOOR         : pooled features of RANDOM-INIT vit_nano
                  (torch.manual_seed(0) and (1)) -> wp1_metric_row.
  CEILING PROXY : physics-summary features (per z-layer: total energy,
                  E-weighted mean x/y, var_x, var_y, cov_xy -> 24 feats),
                  scored with the SAME wp1_metric_row.

Targets: z1, z2 (the drawn per-EVENT latents) and standardized
lnE_total = log(sum of real-particle energies per event), train-split
mean/std. Variant configs are built directly (no run config.yml); the
renderer machinery is wp1_probe_driver's, with a local gen loop because
the driver's lnE column is per-particle (N=1 only) while this sweep needs
the per-event total.

Sanity gate: variant P1/Enar/S1000 random_init_s0 must reproduce the known
floor mean_cca ~0.91 (nspots_sweep.csv N=1000: 0.9104) before the full
sweep runs.

Probe sets are cached to CACHE_DIR/rig_{pcfg}_{eband}_{spots}.pt so reruns
are free. Rows append to results/rig_hardening_sweep.csv (resumable by
(variant, readout)).

Usage (from analysis/):
  CUDA_VISIBLE_DEVICES=1 python rig_hardening_sweep.py --variants P1_Enar_S1000
  CUDA_VISIBLE_DEVICES=1 python rig_hardening_sweep.py                 # full
"""

import argparse
import csv
import math
import os
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
sys.path.insert(0, _HERE)

import wp1_probe_driver as drv  # noqa: E402  (inserts analysis/ and train/ on sys.path)
from src.lejepa_data import sample_z  # noqa: E402
from wp1_metrics import wp1_metric_row  # noqa: E402

CACHE_DIR = paths.CACHE
CSV_PATH = os.path.join(paths.CSV, 'rig_hardening_sweep.csv')

N_TRAIN, N_EVAL = 4000, 2000
TRAIN_SEED, EVAL_SEED = 20260910, 20260911
CHUNK = 256
READOUTS = ('random_init_s0', 'random_init_s1', 'physics_summary')

# sanity gate: known floor at P1/Enar/S1000 (nspots_sweep.csv N=1000 s0)
GATE_VARIANT = 'P1_Enar_S1000'
GATE_REF, GATE_TOL = 0.91, 0.03

COLS = ['variant', 'pcfg', 'eband', 'spots', 'readout', 'mean_cca',
        'cca_c1', 'cca_c2', 'procrustes_mse', 'orth_err', 'ridge_r2_z1',
        'ridge_r2_z2', 'ridge_r2_E', 'sec']

PCFGS = {
    'P1':  {'N_range': [1, 1],   'x_range': [0.0, 0.0],   'y_range': [0.0, 0.0]},
    'P5':  {'N_range': [5, 5],   'x_range': [-4.0, 4.0],  'y_range': [-4.0, 4.0]},
    'P10': {'N_range': [10, 10], 'x_range': [-4.0, 4.0],  'y_range': [-4.0, 4.0]},
    'P20': {'N_range': [20, 20], 'x_range': [-4.0, 4.0],  'y_range': [-4.0, 4.0]},
}
EBANDS = {'Enar': [20000.0, 40000.0], 'Ewide': [10000.0, 100000.0]}
SPOTS = {'S250': 250, 'S1000': 1000, 'S100': 100, 'S50': 50}

NORM_SCALE = 5500.0
LOG1P_SCALE = math.log1p(NORM_SCALE)


def make_cfg(pcfg, eband, spots):
    """Variant config dict, shaped like a run's config.yml (only the keys the
    driver machinery reads). Calo geometry = the existing WP1 rig's."""
    return {
        'data': {
            'calo_cfg': {
                'N_cells_x': 64, 'N_cells_y': 64, 'N_cells_z': 4,
                'N_spots_per_layer': SPOTS[spots],
                'Z': 82, 'cell_e_threshold': 0.0,
                'depth': 20.0, 'height': 10.0, 'width': 10.0,
            },
            'generator_cfg': {
                'kind': 'gun', 'pad_value': float('nan'),
                'E_range': list(EBANDS[eband]), **{k: list(v) for k, v
                                                   in PCFGS[pcfg].items()},
            },
            'norm': {'mode': 'log1p_scale', 'scale': NORM_SCALE},
        },
        'meta': {'model_name': 'vit_nano'},
        'mask': {'patch_size': 4},
    }


# ----------------------------------------------------------------- probe sets

def gen_probe_set(cfg, n_events, seed_base, n_spots):
    """(imgs fp16 cpu (N,4,64,64), z (N,2), lnE_total (N,)) with the per-EVENT
    z KNOWN. Mirrors drv.gen_probe_set (same seed derivations, gaussian dial)
    except lnE is the log of the SUMMED real-particle energy per event —
    the driver's per-particle column only exists for N_range [1,1]."""
    coll = drv._renderer(cfg)
    imgs, zs, lnEs = [], [], []
    n_chunks = (n_events + CHUNK - 1) // CHUNK
    for c in range(n_chunks):
        n = min(CHUNK, n_events - c * CHUNK)
        seed = (seed_base * 100003 + c * 7919 + 1) & 0x7FFFFFFF
        p_E, p_x, p_y = coll._generate_truth(n, seed)
        z = sample_z(n, dist='gaussian', mu=None,
                     generator=torch.Generator().manual_seed(
                         (seed * 2654435761 + 97) & 0x7FFFFFFF))
        lp = coll._lp_from_z(p_E, (seed * 7919 + 31337) & 0x7FFFFFFF, z)
        img = coll.norm.forward(coll._simulate_once(
            p_E, p_x, p_y, coll._sim_seed(seed, 0),
            n_spots_override=n_spots, long_params_override=lp))
        imgs.append(img.detach().to('cpu', torch.float16))
        zs.append(z)
        lnEs.append(torch.log(
            p_E.detach().cpu().double().nansum(dim=1)).float())
    return torch.cat(imgs), torch.cat(zs), torch.cat(lnEs)


def load_variant_set(cfg, variant, n_spots):
    """Cached (imgs_tr fp16, imgs_ev fp16, y_tr, y_ev, sec_gen)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f'rig_{variant}.pt')
    if os.path.exists(path):
        return torch.load(path, map_location='cpu', weights_only=True)
    t0 = time.time()
    itr, ztr, etr = gen_probe_set(cfg, N_TRAIN, TRAIN_SEED, n_spots)
    iev, zev, eev = gen_probe_set(cfg, N_EVAL, EVAL_SEED, n_spots)
    mu_e, sd_e = float(etr.mean()), float(etr.std()) + 1e-8
    d = {'imgs_tr': itr, 'imgs_ev': iev,
         'y_tr': torch.cat([ztr, ((etr - mu_e) / sd_e)[:, None]], dim=1),
         'y_ev': torch.cat([zev, ((eev - mu_e) / sd_e)[:, None]], dim=1),
         'sec_gen': time.time() - t0}
    torch.save(d, path)
    return d


# ------------------------------------------------------------ ceiling proxy

def _raw_energy(imgs_f, log1p_scale=LOG1P_SCALE):
    """Invert the log1p_scale norm: normalized -> raw MeV, >= 0."""
    return torch.expm1(imgs_f * log1p_scale).clamp(min=0)


def physics_summary(imgs, chunk=CHUNK):
    """(N,4,64,64) fp16 normed -> (N,24) float64: per layer total raw energy,
    E-weighted mean x, mean y, var_x, var_y, cov_xy (cell-index coords).
    Same construction as nspots_floor_sweep.physics_summary."""
    X = torch.arange(64, dtype=torch.float64).view(1, 1, 1, 64)
    Y = torch.arange(64, dtype=torch.float64).view(1, 1, 64, 1)
    outs = []
    for i in range(0, imgs.shape[0], chunk):
        E = _raw_energy(imgs[i:i + chunk].double())
        tot = E.sum((2, 3))                                   # (B, 4)
        w = E / (tot[..., None, None] + 1e-12)
        mx = (w * X).sum((2, 3))
        my = (w * Y).sum((2, 3))
        vx = ((w * X * X).sum((2, 3)) - mx * mx).clamp(min=0)
        vy = ((w * Y * Y).sum((2, 3)) - my * my).clamp(min=0)
        cxy = (w * X * Y).sum((2, 3)) - mx * my
        outs.append(torch.stack([tot, mx, my, vx, vy, cxy],
                                dim=-1).reshape(E.shape[0], -1))
    return torch.cat(outs)


# ---------------------------------------------------------------- bookkeeping

def existing_keys():
    if not os.path.exists(CSV_PATH):
        return set()
    with open(CSV_PATH) as f:
        return {(r['variant'], r['readout']) for r in csv.DictReader(f)}


def append_row(variant, pcfg, eband, spots, readout, m, sec):
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new:
            w.writeheader()
        w.writerow({
            'variant': variant, 'pcfg': pcfg, 'eband': eband,
            'spots': spots, 'readout': readout,
            'mean_cca': f"{m['mean_cca']:.6f}",
            'cca_c1': f"{m['cca_per_component'][0]:.6f}",
            'cca_c2': f"{m['cca_per_component'][1]:.6f}",
            'procrustes_mse': f"{m['procrustes_mse']:.6f}",
            'orth_err': f"{m['orth_err']:.6f}",
            'ridge_r2_z1': f"{m['ridge_r2_z1']:.6f}",
            'ridge_r2_z2': f"{m['ridge_r2_z2']:.6f}",
            'ridge_r2_E': f"{m['ridge_r2_E']:.6f}",
            'sec': f"{sec:.1f}",
        })
    print(f"[{variant} {readout}] mean_cca={m['mean_cca']:.4f} "
          f"orth_err={m['orth_err']:.4f} r2=({m['ridge_r2_z1']:.3f},"
          f"{m['ridge_r2_z2']:.3f},{m['ridge_r2_E']:.3f}) sec={sec:.1f}",
          flush=True)


def score_variant(variant, pcfg, eband, spots, sets, done):
    """All three readouts for one variant; returns {readout: mean_cca}."""
    out = {}
    for seed, readout in ((0, 'random_init_s0'), (1, 'random_init_s1')):
        if (variant, readout) in done:
            continue
        t0 = time.time()
        torch.manual_seed(seed)
        cfg = make_cfg(pcfg, eband, spots)
        enc = drv.build_encoder_from_cfg(cfg)
        h_tr = drv.encode_meanpool(enc, sets['imgs_tr'], chunk=CHUNK)
        h_ev = drv.encode_meanpool(enc, sets['imgs_ev'], chunk=CHUNK)
        del enc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        m = wp1_metric_row(h_tr, sets['y_tr'], h_ev, sets['y_ev'])
        append_row(variant, pcfg, eband, spots, readout, m, time.time() - t0)
        out[readout] = m['mean_cca']
    if (variant, 'physics_summary') not in done:
        t0 = time.time()
        h_tr = physics_summary(sets['imgs_tr'])
        h_ev = physics_summary(sets['imgs_ev'])
        m = wp1_metric_row(h_tr, sets['y_tr'], h_ev, sets['y_ev'])
        append_row(variant, pcfg, eband, spots, 'physics_summary', m,
                   time.time() - t0)
        out['physics_summary'] = m['mean_cca']
    return out


# ----------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', default='all',
                    help='"all" or comma list like P1_Enar_S1000')
    args = ap.parse_args()

    order = [f'{p}_{e}_{s}' for p in PCFGS for e in EBANDS for s in SPOTS]
    # gate variant first
    order.sort(key=lambda v: v != GATE_VARIANT)
    if args.variants != 'all':
        want = set(args.variants.split(','))
        unknown = want - set(order)
        assert not unknown, f'unknown variants: {sorted(unknown)}'
        order = [v for v in order if v in want]

    done = existing_keys()
    gate_checked = (GATE_VARIANT, 'random_init_s0') in done

    for variant in order:
        pcfg, eband, spots = variant.split('_')
        cfg = make_cfg(pcfg, eband, spots)
        sets = load_variant_set(cfg, variant, SPOTS[spots])
        etot = _raw_energy(sets['imgs_tr'].double()).sum((1, 2, 3))
        print(f"[{variant}] gen={sets.get('sec_gen', 0.0):.1f}s  train "
              f"E_tot/img: mean={float(etot.mean()):.1f} MeV "
              f"std={float(etot.std()):.1f} MeV", flush=True)

        res = score_variant(variant, pcfg, eband, spots, sets, done)

        if not gate_checked and variant == GATE_VARIANT:
            v = res.get('random_init_s0')
            assert v is not None
            if abs(v - GATE_REF) > GATE_TOL:
                print(f"SANITY GATE FAILED: {GATE_VARIANT} random_init_s0 "
                      f"mean_cca={v:.4f}, expected ~{GATE_REF} "
                      f"(+-{GATE_TOL}). Stopping before the full sweep.",
                      flush=True)
                sys.exit(1)
            print(f"sanity gate OK: {GATE_VARIANT} random_init_s0 "
                  f"mean_cca={v:.4f} ~ {GATE_REF}", flush=True)
            gate_checked = True
        del sets

    # final floor/ceiling/gap table (re-read CSV so resumed runs are complete)
    with open(CSV_PATH) as f:
        rows = {(r['variant'], r['readout']): float(r['mean_cca'])
                for r in csv.DictReader(f)}
    print('\nfloor/ceiling table (mean_cca)')
    print(f"{'variant':<16}{'floor':>10}{'ceiling':>10}{'gap':>10}")
    for v in order:
        s0 = rows.get((v, 'random_init_s0'))
        s1 = rows.get((v, 'random_init_s1'))
        ce = rows.get((v, 'physics_summary'))
        fl = 0.5 * (s0 + s1) if (s0 is not None and s1 is not None) else None
        gap = (ce - fl) if (fl is not None and ce is not None) else None
        fmt = lambda x: f'{x:>10.4f}' if x is not None else f'{"-":>10}'
        print(f'{v:<16}{fmt(fl)}{fmt(ce)}{fmt(gap)}')


if __name__ == '__main__':
    main()
