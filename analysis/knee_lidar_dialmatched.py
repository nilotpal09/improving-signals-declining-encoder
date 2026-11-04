"""Study 3 (2026-09-23): LiDAR on replica populations matched to each run's
own latent distribution.

The existing replica sets draw the shape latents from the Gaussian (mu = 0)
distribution for every run. For the runs trained at mu = 0.4, 0.7, 0.9 and
0.97 (two-spike latents), this builds replica sets whose truths draw z from
the run's own dial distribution, with the wp1_probe_driver machinery:
sample_z(dist='bimodal', mu), LeJEPALatentCollator._lp_from_z,
_simulate_once(n_spots_override=100), norm.forward. The renderer is
wp1_probe_driver._renderer on the config of rigb_lejp_p1s100_nano_1 (the
mu runs' calo, generator and norm configs are identical to it; only
lejepa_latents differs, and the renderer overrides that).

Per mu, 64 truths x 8 replicas, energy and position fixed within a truth
(one _generate_truth draw), two variants sharing truths and spot seeds:
  shared   one z per truth, the same long_params for all 8 replicas
  redrawn  8 independent z draws per truth from the same dial distribution
Seeds (recorded; checked against the probe-cache seed bases in use,
the test split 20260992, and the LiDAR replica seeds):
  truth seed = base: 20261001 (mu 0.4), 20261002 (0.7), 20261003 (0.9),
               20261004 (0.97)
  z generator seed j: (base * 2654435761 + 97 + 7919 * j) & 0x7FFFFFFF;
               j = 0 for the shared z, j = k + 1 for replica k of redrawn
  long-stats seed: (base * 7919 + 31337) & 0x7FFFFFFF (statistics only;
               _lp_from_z discards its draws)
  spot seed of replica k: _sim_seed(base, k)
New caches (refuse to overwrite):
  paths.CACHE/knee_lidar_mu{04,07,09,097}_{shared,redrawn}_N100.pt
Series: results/knee_lidar_dialmatched.csv (run, epoch, n_spots, mu, variant,
lidar) for every N = 100 checkpoint of the runs at those mu, each on its
own mu's sets.

--report: audit per protocol with thresholds recalibrated on the four
mu = 0.97 controls per campaign scored on the mu = 0.97 sets, mu = 0.4 / 0.7
/ 0.9 runs on their own sets, and every mu = 0 run (knee, held-out, MAE,
ViT-Tiny) on the existing Gaussian sets; the four-way table (protocol x
population) of knee alarms and held-out false alarms; per control, the
PRE -> POST drift under matched versus Gaussian populations.

Usage (from analysis/):
  CUDA_VISIBLE_DEVICES=1 python knee_lidar_dialmatched.py --build
  CUDA_VISIBLE_DEVICES=1 python knee_lidar_dialmatched.py
  python knee_lidar_dialmatched.py --report
"""
import sys

sys.dont_write_bytecode = True      # leave analysis/__pycache__ untouched

import argparse  # noqa: E402
import csv  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from statistics import median  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
sys.path.insert(0, _HERE)

N = 100
MUS = {0.4: ('mu04', 20261001), 0.7: ('mu07', 20261002),
       0.9: ('mu09', 20261003), 0.97: ('mu097', 20261004)}
VARIANTS = ('shared', 'redrawn')
OUT = os.path.join(paths.CSV, 'knee_lidar_dialmatched.csv')
GAUSS = {'redrawn': os.path.join(paths.CSV, 'knee_lidar.csv'),
         'shared': os.path.join(paths.CSV, 'knee_lidar_sharedshape.csv')}
MASK = 0x7FFFFFFF


def zseed(base, j):
    return (base * 2654435761 + 97 + 7919 * j) & MASK


def long_seed(base):
    return (base * 7919 + 31337) & MASK


def cache_name(tag, variant):
    return f'knee_lidar_{tag}_{variant}_N{N}.pt'


def run_mu(run):
    m = re.search(r'_mu0(\d+)_', run)
    return {'4': 0.4, '7': 0.7, '9': 0.9, '97': 0.97}[m.group(1)] if m else 0.0


# ------------------------------------------------------------------- build

def seed_check():
    """Every seed this study derives must differ from the probe-cache seed
    families and from the LiDAR replica sets' seeds."""
    import knee_s3_ladder_split as LS
    ours = set()
    for _, base in MUS.values():
        ours |= {base, long_seed(base)} | {zseed(base, j) for j in range(9)}
        ours |= {(base * 31337 + k + 1) & MASK for k in range(8)}
    used = set()
    for b in LS.USED_SEED_BASES + [LS.TEST_SEED]:
        used |= LS.chunk_seed_family(b)
    for s in (30, 100, 300, 1000, 10107):                     # LiDAR truth seeds
        used |= {s} | {(s * 31337 + k + 1) & MASK for k in range(8)}
    used |= {778, 779}
    hit = ours & used
    assert not hit, f'seed collision: {sorted(hit)[:5]}'
    print(f'seed check: {len(ours)} derived seeds, none among {len(used)} in use', flush=True)


def build():
    import torch
    import yaml
    import knee_lidar as KL
    import knee_lidar_sharedshape as KLS
    from src.lejepa_data import sample_z
    seed_check()
    cfg0 = yaml.safe_load(open(os.path.join(KL.drv.ROOT, KLS.CFG0_RUN, 'config.yml')))
    coll = KL.drv._renderer(cfg0)
    for mu, (tag, base) in MUS.items():
        for v in VARIANTS:
            assert not os.path.exists(os.path.join(KL.CACHE, cache_name(tag, v))), \
                f'refusing to overwrite {cache_name(tag, v)}'
        p_E, p_x, p_y = coll._generate_truth(KL.M, base)
        ls = long_seed(base)

        def z_of(j):
            return sample_z(KL.M, dist='bimodal', mu=mu,
                            generator=torch.Generator().manual_seed(zseed(base, j)))

        def render(z, k):
            lp = coll._lp_from_z(p_E, ls, z)
            return coll.norm.forward(coll._simulate_once(
                p_E, p_x, p_y, coll._sim_seed(base, k), n_spots_override=N,
                long_params_override=lp))

        z_sh = z_of(0)
        z_rd = [z_of(k + 1) for k in range(KL.K)]
        lp0 = coll._lp_from_z(p_E, ls, z_sh)
        lp7 = coll._lp_from_z(p_E, ls, z_sh)
        same = torch.equal(lp0['alpha'], lp7['alpha']) and torch.equal(lp0['T'], lp7['T'])
        zall = torch.cat([z_sh] + z_rd)
        print(f'mu={mu}: truth seed {base}; shared alpha/T repeatable across replicas: '
              f'{same}; redrawn z differ across replicas: '
              f'{not any(torch.equal(z_rd[0], z) for z in z_rd[1:])}; z mean '
              f'{[round(float(x), 3) for x in zall.mean(0)]} sd '
              f'{[round(float(x), 3) for x in zall.std(0)]}; share with |z| > mu/2: '
              f'{float((zall.abs() > mu / 2).float().mean()):.3f}', flush=True)
        for v, zs in (('shared', [z_sh] * KL.K), ('redrawn', z_rd)):
            reps = torch.stack([render(zs[k], k) for k in range(KL.K)], dim=1)
            imgs = reps.reshape(KL.M * KL.K, *reps.shape[2:]).to('cpu', torch.float16)
            p = os.path.join(KL.CACHE, cache_name(tag, v))
            assert not os.path.exists(p)
            torch.save(imgs, p)
            back = torch.load(p, map_location='cpu', weights_only=True)
            print(f'   saved {cache_name(tag, v)} {tuple(back.shape)} read-back max|diff| '
                  f'{float((back.float() - imgs.float()).abs().max()):.1e} sha256 '
                  f'{KLS.set_digest(back)}', flush=True)
        torch.cuda.empty_cache()


# -------------------------------------------------------------------- main

def main():
    import torch
    import knee_lidar as KL
    import knee_lidar_sharedshape as KLS
    print(f'GPU: {torch.cuda.get_device_name(0)} '
          f'(CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")})', flush=True)
    IMGS = {}
    for mu, (tag, _) in MUS.items():
        for v in VARIANTS:
            p = os.path.join(KL.CACHE, cache_name(tag, v))
            assert os.path.exists(p), 'run --build first'
            IMGS[(mu, v)] = torch.load(p, map_location='cpu', weights_only=True)
    plan = KLS.source_plan()
    runs = {run: sorted(e for e, rungs in eps.items() if N in rungs)
            for run, eps in plan.items() if run_mu(run) > 0}
    print(f'{len(runs)} runs at mu > 0, {sum(len(v) for v in runs.values())} checkpoints', flush=True)
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as fh:
            done = {(r['run'], int(r['epoch']), r['variant']) for r in csv.DictReader(fh)}
    new = not os.path.exists(OUT)
    fout = open(OUT, 'a', newline='')
    w = csv.writer(fout, lineterminator='\n')
    if new:
        w.writerow(['run', 'epoch', 'n_spots', 'mu', 'variant', 'lidar'])
    t_all = time.time()
    for run, eps in runs.items():
        t0 = time.time()
        mu = run_mu(run)
        cfg = KLS.load_cfg(run)
        wtag = cfg['logging']['write_tag']
        for ep in eps:
            need = [v for v in VARIANTS if (run, ep, v) not in done]
            if not need:
                continue
            enc = KL.drv.encoder_from_ckpt(cfg, os.path.join(KL.drv.ROOT, run,
                                                             f'{wtag}-ep{ep}.pth.tar'))
            for v in need:
                w.writerow([run, ep, N, mu, v, f'{KL.lidar_of(enc, IMGS[(mu, v)]):.6f}'])
            fout.flush()
            del enc
        print(f'{run.split("xxx")[0]:38s} mu={mu} {len(eps)} ckpts ({time.time() - t0:.0f}s; '
              f'total {time.time() - t_all:.0f}s)', flush=True)
    fout.close()
    print('DONE knee_lidar_dialmatched')


# ------------------------------------------------------------------ report

def load_gauss(proto):
    import knee_lidar_audit as AUD
    AUD.LID = GAUSS[proto]
    return AUD.load(N)


def load_matched(proto):
    """mu > 0 runs from the dial-matched series, every other run from the
    existing Gaussian series of the same protocol."""
    S = dict(load_gauss(proto))
    by = {}
    with open(OUT) as fh:
        for r in csv.DictReader(fh):
            if r['variant'] == proto:
                by.setdefault(r['run'].split('xxx')[0], {})[int(r['epoch'])] = float(r['lidar'])
    for run, d in by.items():
        S[run] = (sorted(d), [d[e] for e in sorted(d)])
    return S


def report():
    import knee_lidar_audit as AUD
    import knee_lidar_audit_sharedshape as A2
    import knee_lidar_replica_seed as RS
    S = {(p, pop): (load_gauss(p) if pop == 'Gaussian' else load_matched(p))
         for p in VARIANTS for pop in ('Gaussian', 'matched')}
    print('== four-way table, N = 100 (protocol x replica population) ==')
    for k, s in S.items():
        c = RS.counts_from_S(s)
        thr = c['thr']
        print(f'\n-- {k[0]} shape, {k[1]} population: CUSUM thr 21x '
              f'{thr["21x"]["CUSUM"]:.3f}, 4x {thr["4x"]["CUSUM"]:.3f}')
        for t in A2.TESTS:
            parts = []
            for camp in ('21x', '4x'):
                n, tot, d = c[('knee', camp, t)]
                parts.append(f'knee {camp} {n}/{tot} (within one grid step '
                             f'{sum(abs(x) <= 10 for x in d)})')
            ho, hn = c[('heldout', t)]
            mo, mn = c[('mae', t)]
            to, tn = c[('tiny', t)]
            parts.append(f'held-out LeJEPA {len(ho)}/{hn}'
                         + (f' ({", ".join(A2.short(r) for r in ho)})' if ho else ''))
            parts.append(f'MAE {len(mo)}/{mn}')
            parts.append(f'ViT-Tiny flat {len(to)}/{tn}')
            print(f'  {t:12s} ' + ' | '.join(parts))
        print(f'  any of four: held-out LeJEPA {len(c[("heldout", "any")][0])}/'
              f'{c[("heldout", "any")][1]}, MAE {len(c[("mae", "any")][0])}/{c[("mae", "any")][1]}, '
              f'ViT-Tiny flat {len(c[("tiny", "any")][0])}/{c[("tiny", "any")][1]}')

    def raw(s, run):
        eps, xs = s[run]
        d = dict(zip(eps, xs))
        fin = max(eps)
        return (median([d[e] for e in eps if e > fin - 30])
                - median([d[e] for e in eps if 21 <= e <= 60]))

    pre_ref = {}
    for camp, knee in (('21x', AUD.KNEE_21), ('4x', AUD.KNEE_4)):
        pre_ref[camp] = median([o - 20 for o in knee.values()])
    print('\n== controls: drift under matched vs Gaussian populations ==')
    print('   raw change = 2026-09-05 section 4b statistic; PRE->POST = nearest grid epoch '
          'at or below the campaign knee-cohort median PRE -> final')
    for p in VARIANTS:
        for c_run in AUD.CTRL_21 + AUD.CTRL_4:
            cells = []
            for pop in ('Gaussian', 'matched'):
                s = S[(p, pop)]
                eps, xs = s[c_run]
                d = dict(zip(eps, xs))
                pe = max(e for e in eps if e <= pre_ref[A2.campaign(c_run)])
                cells.append(f'{pop}: raw {raw(s, c_run):+.2f}, PRE(ep{pe}) {d[pe]:.2f} -> '
                             f'POST {d[max(eps)]:.2f} ({d[max(eps)] - d[pe]:+.2f})')
            print(f'  {p:7s} {A2.short(c_run):18s} ' + ' | '.join(cells))
    print('\n== knee runs at mu > 0 and held-out at mu > 0: level change matched vs Gaussian ==')
    for p in VARIANTS:
        for k, o in list(AUD.KNEE_21.items()) + list(AUD.KNEE_4.items()):
            if run_mu(k) == 0:
                continue
            cells = []
            for pop in ('Gaussian', 'matched'):
                eps, xs = S[(p, pop)][k]
                d = dict(zip(eps, xs))
                pe = max(e for e in eps if e <= o - 20)
                cells.append(f'{pop}: {d[pe]:.2f} -> {d[max(eps)]:.2f} ({d[max(eps)] - d[pe]:+.2f})')
            print(f'  {p:7s} knee {A2.short(k):16s} ' + ' | '.join(cells))
        for h in A2.HELD_21 + A2.HELD_4:
            if run_mu(h) == 0:
                continue
            print(f'  {p:7s} held {A2.short(h):16s} Gaussian raw {raw(S[(p, "Gaussian")], h):+.2f} | '
                  f'matched raw {raw(S[(p, "matched")], h):+.2f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', action='store_true')
    ap.add_argument('--report', action='store_true')
    a = ap.parse_args()
    if a.report:
        report()
    elif 'CUDA_VISIBLE_DEVICES' not in os.environ:
        raise SystemExit('set CUDA_VISIBLE_DEVICES explicitly (shared GPUs)')
    elif a.build:
        build()
    else:
        main()
