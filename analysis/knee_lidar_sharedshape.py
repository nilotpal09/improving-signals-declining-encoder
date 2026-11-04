"""LiDAR with training-style groups: profile shape fixed within a group,
element noise varying (2026-09-23).

Why: knee_lidar.py builds its replica sets with generate_replicas(...) and
no fixed_long_seed, so each of the K re-renders of a truth re-draws the
profile shape (alpha/T) as well as the spot Monte Carlo. Training views at
rho = 1 share the shape (src/lejepa_data.py:213-246). This script measures
LiDAR on groups that DO share the shape: fixed_long_seed = 778 fixes the
alpha/T draw per truth across the K re-renders while the spot seed still
changes per re-render (calo_flash.py:953-1010, 725-746, 634-652).

Imported unchanged from knee_lidar.py: lidar() (from
analysis/lidar_metric.py), lidar_of() (encoder tokens ->
LayerNorm -> mean over tokens -> lidar on (M, K, D)), M = 64, K = 8,
make_collator (analysis/calo_collator.py), the driver (encoder_from_ckpt).
The only change from knee_lidar.replica_set: fixed_long_seed=778 in the
generate_replicas call. Truth seed = 10007*0 + n as before; normalize=True;
config = the first 21x run's (rigb_lejp_p1s100_nano_1), as before.

Run list = results/knee_lidar.csv (read only): all 58 runs in it, every
checkpoint with a row there, and at each checkpoint the rungs it has there
(N = 100 for all 58; N = 1000 for the 51 that have it).

Outputs (new files; nothing existing is written):
  results/knee_lidar_sharedshape.csv   run, epoch, n_spots, lidar
                                   (append-mode, resumable by the key)
  probe_caches/knee_lidar_sharedshape_N{100,1000}.pt
                                   (512 fp16 images, 64 truths x 8; refuse
                                   to overwrite)

Usage (from analysis/; set the GPU):
  CUDA_VISIBLE_DEVICES=0 python knee_lidar_sharedshape.py --verify
  CUDA_VISIBLE_DEVICES=0 python knee_lidar_sharedshape.py
"""
import sys

sys.dont_write_bytecode = True      # leave analysis/__pycache__ untouched

import argparse  # noqa: E402
import csv  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from collections import OrderedDict  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
sys.path.insert(0, _HERE)

FIXED_LONG_SEED = 778
SRC_CSV = os.path.join(paths.CSV, 'knee_lidar.csv')
OUT = os.path.join(paths.CSV, 'knee_lidar_sharedshape.csv')
CFG0_RUN = 'rigb_lejp_p1s100_nano_1xxx2i533m'   # knee_lidar.RUNS[0]
RUNGS = (100, 1000)


def cache_path(n_spots):
    import knee_lidar as KL
    return os.path.join(KL.CACHE, f'knee_lidar_sharedshape_N{n_spots}.pt')


def load_cfg(run):
    import yaml
    import knee_lidar as KL
    return yaml.safe_load(open(os.path.join(KL.drv.ROOT, run, 'config.yml')))


def collator_for(cfg, n_spots):
    """knee_lidar.replica_set's collator construction, verbatim."""
    import knee_lidar as KL
    ccfg = {k: v for k, v in cfg['data']['calo_cfg'].items()
            if k not in ('lejepa_latents', 'paired_fixed_long')}
    return KL.make_collator(ccfg, dict(cfg['data']['generator_cfg']), n_spots,
                            cfg['data'].get('norm') or {})


def replica_set_shared(cfg, n_spots):
    """knee_lidar.replica_set with fixed_long_seed=778 and a new cache name.

    If the disk refuses the new file (for example when a file-count
    quota is exceeded), the set
    is kept in memory and the run continues: the build is deterministic
    (verify() regenerates the existing sets with max |diff| = 0), and the
    sha256 printed by set_digest identifies the exact set used."""
    import torch
    import knee_lidar as KL
    path = cache_path(n_spots)
    if os.path.exists(path):
        return torch.load(path, map_location='cpu', weights_only=True)
    coll = collator_for(cfg, n_spots)
    _, reps = coll.generate_replicas(batch_size=KL.M, n_replicas=KL.K,
                                     seed=10007 * 0 + n_spots, normalize=True,
                                     fixed_long_seed=FIXED_LONG_SEED)
    imgs = reps.reshape(KL.M * KL.K, *reps.shape[2:]).to('cpu', torch.float16)
    del coll
    torch.cuda.empty_cache()
    try:
        torch.save(imgs, path)
    except (OSError, RuntimeError) as e:
        if os.path.exists(path):          # partial file from THIS call only
            os.remove(path)
        print(f'WARNING: could not write {path} ({e}); continuing with the '
              f'in-memory set, sha256 {set_digest(imgs)}', flush=True)
    return imgs


def set_digest(imgs):
    import hashlib
    return hashlib.sha256(imgs.contiguous().numpy().tobytes()).hexdigest()[:16]


def source_plan():
    """run -> {epoch: sorted rungs} from results/knee_lidar.csv, in file order."""
    plan = OrderedDict()
    with open(SRC_CSV) as fh:
        for r in csv.DictReader(fh):
            plan.setdefault(r['run'], {}).setdefault(int(r['epoch']), set()).add(
                int(r['n_spots']))
    return plan


# ----------------------------------------------------------------- verify

def layer_fraction_spread(coll, imgs, M, K):
    """Mean over truths and layers of the within-group sd of each layer's
    share of the event energy (raw MeV via the collator's norm.inverse)."""
    import torch
    x = coll.norm.inverse(imgs.float().to(coll.device)).double()
    per_layer = x.sum(dim=(2, 3))                         # (M*K, n_layers)
    frac = per_layer / per_layer.sum(1, keepdim=True)
    frac = frac.reshape(M, K, -1)
    tot = per_layer.sum(1).reshape(M, K)
    return (float(frac.std(dim=1).mean()),
            float((tot.std(dim=1) / tot.mean(dim=1)).mean()))


def verify():
    import inspect
    import torch
    import knee_lidar as KL

    cfg0 = load_cfg(CFG0_RUN)
    print(f'config for the replica sets: {CFG0_RUN} (knee_lidar.RUNS[0])')
    for n in RUNGS:
        coll = collator_for(cfg0, n)
        src = inspect.getsourcefile(type(coll))
        sig = inspect.signature(type(coll).generate_replicas)
        print(f'\n--- N = {n}: collator class {type(coll).__name__} from {src}')
        print(f'    generate_replicas{sig}')
        assert src.endswith('train/src/datasets/calo_flash.py'), src
        assert 'fixed_long_seed' in sig.parameters

        # (a) the existing full-stochasticity set regenerates exactly
        truth_a, reps_a = coll.generate_replicas(batch_size=KL.M, n_replicas=KL.K,
                                                 seed=10007 * 0 + n, normalize=True)
        imgs_a = reps_a.reshape(KL.M * KL.K, *reps_a.shape[2:]).to('cpu', torch.float16)
        old = torch.load(os.path.join(KL.CACHE, f'knee_lidar_N{n}.pt'),
                         map_location='cpu', weights_only=True)
        print(f'    existing knee_lidar_N{n}.pt regenerated in memory: '
              f'max|diff| = {float((imgs_a.float() - old.float()).abs().max()):.3e}'
              f' (shape {tuple(old.shape)})')

        # (b) the shape draw under fixed_long_seed is identical for every
        #     replica (the same call _simulate_once makes per replica)
        p_E = truth_a[0]
        lps = [coll._compute_long_params(p_E, FIXED_LONG_SEED) for _ in range(KL.K)]
        same = all(torch.equal(lp[k], lps[0][k]) for lp in lps
                   for k in ('alpha', 'T', 'beta'))
        other = coll._compute_long_params(p_E, FIXED_LONG_SEED + 1)
        print(f'    _compute_long_params(p_E, {FIXED_LONG_SEED}) x{KL.K}: alpha/T/beta '
              f'identical = {same}; differs under seed {FIXED_LONG_SEED + 1} = '
              f'{not torch.equal(other["alpha"], lps[0]["alpha"])}; '
              f'alpha range {float(lps[0]["alpha"].min()):.3f}-'
              f'{float(lps[0]["alpha"].max()):.3f}, T range '
              f'{float(lps[0]["T"].min()):.3f}-{float(lps[0]["T"].max()):.3f} '
              f'over the {p_E.numel()} truths')
        assert same
        spot_seeds = [coll._sim_seed(10007 * 0 + n, k) for k in range(KL.K)]
        print(f'    spot seeds per replica (_sim_seed(seed, k)): {spot_seeds}')
        assert len(set(spot_seeds)) == KL.K

        # (c) the shared-shape set (built and cached here if absent)
        truth_b, reps_b = coll.generate_replicas(batch_size=KL.M, n_replicas=KL.K,
                                                 seed=10007 * 0 + n, normalize=True,
                                                 fixed_long_seed=FIXED_LONG_SEED)
        imgs_b = reps_b.reshape(KL.M * KL.K, *reps_b.shape[2:]).to('cpu', torch.float16)
        same_truth = all(torch.equal(a, b) for a, b in zip(truth_a, truth_b))
        print(f'    truths (E, x, y) identical to the existing set: {same_truth}')
        assert same_truth
        imgs_c = replica_set_shared(cfg0, n)
        if os.path.exists(cache_path(n)):
            disk = torch.load(cache_path(n), map_location='cpu', weights_only=True)
            print(f'    cache {os.path.basename(cache_path(n))} (read from disk): '
                  f'max|diff| to a fresh in-memory build = '
                  f'{float((disk.float() - imgs_b.float()).abs().max()):.3e}; '
                  f'sha256 disk {set_digest(disk)} fresh {set_digest(imgs_b)}')
        else:
            print(f'    cache {os.path.basename(cache_path(n))} NOT on disk; '
                  f'in-memory set sha256 {set_digest(imgs_c)}, fresh build '
                  f'{set_digest(imgs_b)}')
        d01_a = float((reps_a[:, 0] - reps_a[:, 1]).abs().mean())
        d01_b = float((reps_b[:, 0] - reps_b[:, 1]).abs().mean())
        print(f'    mean|replica0 - replica1| (normalized units): full {d01_a:.4f}, '
              f'shared-shape {d01_b:.4f}  (nonzero = spots vary)')
        assert d01_b > 0
        fa, ta = layer_fraction_spread(coll, imgs_a, KL.M, KL.K)
        fb, tb = layer_fraction_spread(coll, imgs_b, KL.M, KL.K)
        print(f'    within-group sd of per-layer energy share (mean over truths, '
              f'layers): full {fa:.5f}, shared-shape {fb:.5f}')
        print(f'    within-group CV of total deposited energy: full {ta:.5f}, '
              f'shared-shape {tb:.5f}')
        del coll
        torch.cuda.empty_cache()

    # (d) the existing LiDAR rows reproduce with the existing sets
    print('\n--- existing knee_lidar.csv rows recomputed with the existing sets')
    old_rows = {}
    with open(SRC_CSV) as fh:
        for r in csv.DictReader(fh):
            old_rows[(r['run'], int(r['epoch']), int(r['n_spots']))] = float(r['lidar'])
    imgs_old = {n: torch.load(os.path.join(KL.CACHE, f'knee_lidar_N{n}.pt'),
                              map_location='cpu', weights_only=True) for n in RUNGS}
    picks = [('rigb_lejp_p1s100_nano_1xxx2i533m', (2, 70, 150)),
             ('rigb_lejp_p1s100_p384_2xxxa8r220', (120, 300)),
             ('rigd_lejp_p1s100_mu097_long_nano_1xxx', (300,)),
             ('rigb_mae_p1s100_long_nano_1xxxyh', (20, 300)),
             ('rigb_mae_p1s100_long_nano_7xxx6j', (300,)),
             ('rigb_lejp_p1s100_tiny_1xxx0syrpc', (300,))]
    worst = 0.0
    for run, eps in picks:
        cfg = load_cfg(run)
        tag = cfg['logging']['write_tag']
        for ep in eps:
            enc = KL.drv.encoder_from_ckpt(
                cfg, os.path.join(KL.drv.ROOT, run, f'{tag}-ep{ep}.pth.tar'))
            for n in RUNGS:
                if (run, ep, n) not in old_rows:
                    continue
                v = KL.lidar_of(enc, imgs_old[n])
                d = abs(v - old_rows[(run, ep, n)])
                worst = max(worst, d)
                print(f'    {run.split("xxx")[0]:36s} ep{ep:<4d} N={n:<5d} '
                      f'csv {old_rows[(run, ep, n)]:.6f} now {v:.6f} |d| {d:.2e}')
            del enc
    print(f'    worst |d| = {worst:.2e}')


# ------------------------------------------------------------------- main

def main():
    import torch
    import knee_lidar as KL

    print(f'GPU: {torch.cuda.get_device_name(0)} '
          f'(CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")})',
          flush=True)
    plan = source_plan()
    n_ck = sum(len(v) for v in plan.values())
    n_rows = sum(len(r) for v in plan.values() for r in v.values())
    print(f'source {SRC_CSV}: {len(plan)} runs, {n_ck} checkpoints, {n_rows} rows',
          flush=True)
    assert len(plan) == 61, len(plan)  # denominator runs (2026-10-13): 58 + nano_10-12

    done = set()
    if os.path.exists(OUT):
        with open(OUT) as fh:
            done = {(r['run'], int(r['epoch']), int(r['n_spots']))
                    for r in csv.DictReader(fh)}
    new = not os.path.exists(OUT)
    fout = open(OUT, 'a', newline='')
    w = csv.writer(fout, lineterminator='\n')
    if new:
        w.writerow(['run', 'epoch', 'n_spots', 'lidar'])

    cfg0 = load_cfg(CFG0_RUN)
    IMGS = {n: replica_set_shared(cfg0, n) for n in RUNGS}
    for n in RUNGS:
        print(f'replica set N={n}: {tuple(IMGS[n].shape)}, sha256 '
              f'{set_digest(IMGS[n])}, on disk: {os.path.exists(cache_path(n))}',
              flush=True)

    t_all = time.time()
    for run, eps in plan.items():
        t0 = time.time()
        cfg = load_cfg(run)
        tag = cfg['logging']['write_tag']
        n_new = 0
        for ep in sorted(eps):
            todo = [n for n in sorted(eps[ep]) if (run, ep, n) not in done]
            if not todo:
                continue
            ck = os.path.join(KL.drv.ROOT, run, f'{tag}-ep{ep}.pth.tar')
            assert os.path.exists(ck), ck
            enc = KL.drv.encoder_from_ckpt(cfg, ck)
            for n in todo:
                w.writerow([run, ep, n, f'{KL.lidar_of(enc, IMGS[n]):.6f}'])
                n_new += 1
            fout.flush()
            del enc
        print(f'{run.split("xxx")[0]:38s} {n_new:3d} new rows '
              f'({len(eps)} ckpts, {time.time() - t0:.0f}s; '
              f'total {time.time() - t_all:.0f}s)', flush=True)
    fout.close()
    print('DONE knee_lidar_sharedshape')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--verify', action='store_true')
    a = ap.parse_args()
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        raise SystemExit('set CUDA_VISIBLE_DEVICES explicitly (shared GPUs)')
    if a.verify:
        verify()
    else:
        main()
