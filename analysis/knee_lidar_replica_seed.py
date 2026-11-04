"""Study 2 (2026-09-23): replica-seed replication of both LiDAR protocols
at N = 100.

Both replica sets are rebuilt with an independent truth seed, ps = 1 in the
imported formula: seed = 10007*1 + n = 10107 for n = 100. The shared-shape
set also uses a different shape seed, fixed_long_seed = 779 (the ps = 0 set
used 778). Everything else is knee_lidar_sharedshape's construction
unchanged: config of rigb_lejp_p1s100_nano_1, make_collator,
generate_replicas(batch_size=64, n_replicas=8, normalize=True), features
and lidar() via knee_lidar.lidar_of.

New caches (refuse to overwrite):
  probe_caches/knee_lidar_N100_ps1.pt              shape redrawn
  probe_caches/knee_lidar_sharedshape_N100_ps1.pt  shape shared
New series (append-mode, resumable by (run, epoch, n_spots)):
  results/knee_lidar_ps1.csv, results/knee_lidar_sharedshape_ps1.csv
for every run and checkpoint that has an N = 100 row in results/knee_lidar.csv.

--report prints, for the four series (protocol x seed): the recalibrated
audit counts (1.1 x max over each campaign's four mu = 0.97 controls,
knee_lidar_audit_sharedshape's scoring), knee alarm alignment, held-out
false alarms (11 LeJEPA, 10 MAE, ViT-Tiny flat), the PRE -> POST level change
of the knee runs, the raw change of the event-free populations (the
statistic of the earlier audit), and per-run correlations of the
ps = 1 series with the ps = 0 series.

Usage (from analysis/):
  CUDA_VISIBLE_DEVICES=1 python knee_lidar_replica_seed.py --build
  CUDA_VISIBLE_DEVICES=1 python knee_lidar_replica_seed.py
  python knee_lidar_replica_seed.py --report
"""
import sys

sys.dont_write_bytecode = True      # leave analysis/__pycache__ untouched

import argparse  # noqa: E402
import csv  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from statistics import median  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
sys.path.insert(0, _HERE)

PS, N = 1, 100
SEED = 10007 * PS + N                  # 10107
FIXED_LONG_SEED = 779
CACHE_NAME = {'redrawn': 'knee_lidar_N100_ps1.pt',
              'shared': 'knee_lidar_sharedshape_N100_ps1.pt'}
OUT = {'redrawn': os.path.join(paths.CSV, 'knee_lidar_ps1.csv'),
       'shared': os.path.join(paths.CSV, 'knee_lidar_sharedshape_ps1.csv')}
PS0 = {'redrawn': os.path.join(paths.CSV, 'knee_lidar.csv'),
       'shared': os.path.join(paths.CSV, 'knee_lidar_sharedshape.csv')}


def cache_path(tag):
    import knee_lidar as KL
    return os.path.join(KL.CACHE, CACHE_NAME[tag])


def build():
    import torch
    import knee_lidar as KL
    import knee_lidar_sharedshape as KLS
    cfg0 = KLS.load_cfg(KLS.CFG0_RUN)
    coll = KLS.collator_for(cfg0, N)
    out = {}
    truth_r, reps_r = coll.generate_replicas(batch_size=KL.M, n_replicas=KL.K,
                                             seed=SEED, normalize=True)
    truth_s, reps_s = coll.generate_replicas(batch_size=KL.M, n_replicas=KL.K,
                                             seed=SEED, normalize=True,
                                             fixed_long_seed=FIXED_LONG_SEED)
    truth_0, _ = coll.generate_replicas(batch_size=KL.M, n_replicas=1, seed=N,
                                        normalize=True)
    print(f'truths ps1 identical across the two ps1 sets: '
          f'{all(torch.equal(a, b) for a, b in zip(truth_r, truth_s))}; '
          f'energies differ from ps0: {not torch.equal(truth_r[0], truth_0[0])} '
          f'(ps1 E range {float(truth_r[0].min()):.0f}-{float(truth_r[0].max()):.0f} MeV)')
    lps = [coll._compute_long_params(truth_s[0], FIXED_LONG_SEED) for _ in range(KL.K)]
    print(f'shared set: alpha/T identical across the {KL.K} replicas: '
          f'{all(torch.equal(lp["alpha"], lps[0]["alpha"]) and torch.equal(lp["T"], lps[0]["T"]) for lp in lps)}')
    for tag, reps in (('redrawn', reps_r), ('shared', reps_s)):
        imgs = reps.reshape(KL.M * KL.K, *reps.shape[2:]).to('cpu', torch.float16)
        p = cache_path(tag)
        assert not os.path.exists(p), f'refusing to overwrite {p}'
        torch.save(imgs, p)
        back = torch.load(p, map_location='cpu', weights_only=True)
        print(f'saved {p}: {tuple(imgs.shape)}; read-back max|diff| '
              f'{float((back.float() - imgs.float()).abs().max()):.1e}; sha256 '
              f'{KLS.set_digest(back)}', flush=True)
        out[tag] = back
    return out


def main():
    import torch
    import knee_lidar as KL
    import knee_lidar_sharedshape as KLS
    print(f'GPU: {torch.cuda.get_device_name(0)} '
          f'(CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")})', flush=True)
    IMGS = {}
    for tag in CACHE_NAME:
        p = cache_path(tag)
        assert os.path.exists(p), 'run --build first'
        IMGS[tag] = torch.load(p, map_location='cpu', weights_only=True)
        print(f'{tag}: {CACHE_NAME[tag]} sha256 {KLS.set_digest(IMGS[tag])}', flush=True)
    plan = KLS.source_plan()                      # run -> {epoch: rungs}
    todo_eps = {run: sorted(e for e, rungs in eps.items() if N in rungs)
                for run, eps in plan.items()}
    print(f'{len(todo_eps)} runs, {sum(len(v) for v in todo_eps.values())} '
          f'checkpoints at N = {N}', flush=True)
    done, fouts, writers = {}, {}, {}
    for tag, p in OUT.items():
        done[tag] = set()
        if os.path.exists(p):
            with open(p) as fh:
                done[tag] = {(r['run'], int(r['epoch'])) for r in csv.DictReader(fh)}
        new = not os.path.exists(p)
        fouts[tag] = open(p, 'a', newline='')
        writers[tag] = csv.writer(fouts[tag], lineterminator='\n')
        if new:
            writers[tag].writerow(['run', 'epoch', 'n_spots', 'lidar'])
    t_all = time.time()
    for run, eps in todo_eps.items():
        t0 = time.time()
        cfg = KLS.load_cfg(run)
        wtag = cfg['logging']['write_tag']
        for ep in eps:
            need = [t for t in OUT if (run, ep) not in done[t]]
            if not need:
                continue
            enc = KL.drv.encoder_from_ckpt(cfg, os.path.join(KL.drv.ROOT, run,
                                                             f'{wtag}-ep{ep}.pth.tar'))
            for t in need:
                writers[t].writerow([run, ep, N, f'{KL.lidar_of(enc, IMGS[t]):.6f}'])
            for t in need:
                fouts[t].flush()
            del enc
        print(f'{run.split("xxx")[0]:38s} {len(eps)} ckpts ({time.time() - t0:.0f}s; '
              f'total {time.time() - t_all:.0f}s)', flush=True)
    for f in fouts.values():
        f.close()
    print('DONE knee_lidar_replica_seed')


# ------------------------------------------------------------------ report

def series_counts(path):
    """Audit counts for one series at N = 100 (recalibrated thresholds)."""
    import knee_lidar_audit as AUD
    AUD.LID = path
    return counts_from_S(AUD.load(N))


def counts_from_S(S):
    """Audit counts for a {run: (epochs, values)} series dict: thresholds
    1.1 x max over each campaign's four mu = 0.97 controls in S."""
    import knee_lidar_audit as AUD
    import knee_lidar_audit_sharedshape as A2
    thr = A2.thresholds(S)
    sc = A2.score(S, thr)
    out = {'thr': thr, 'S': S, 'sc': sc}
    for camp, knee in (('21x', AUD.KNEE_21), ('4x', AUD.KNEE_4)):
        for t in A2.TESTS:
            d = [sc[k][t][2] - o for k, o in knee.items() if k in S and sc[k][t][2] is not None]
            out[('knee', camp, t)] = (len(d), len([k for k in knee if k in S]), d)
    pops = {'heldout': A2.HELD_21 + A2.HELD_4, 'mae': A2.MAE, 'tiny': A2.TINY_FLAT}
    for name, pop in pops.items():
        have = [r for r in pop if r in S]
        for t in A2.TESTS:
            out[(name, t)] = ([r for r in have if sc[r][t][2] is not None], len(have))
        out[(name, 'any')] = ([r for r in have if any(sc[r][t][2] is not None for t in A2.TESTS)],
                              len(have))
    return out


def level_changes(S):
    import knee_lidar_audit as AUD
    import knee_lidar_audit_sharedshape as A2
    res = {}
    for camp, knee in (('21x', AUD.KNEE_21), ('4x', AUD.KNEE_4)):
        dl = []
        for k, o in knee.items():
            eps, xs = S[k]
            d = dict(zip(eps, xs))
            pe = max(e for e in eps if e <= o - 20)
            dl.append(d[max(eps)] - d[pe])
        res[f'knee {camp}'] = dl

    def raw(run):
        eps, xs = S[run]
        d = dict(zip(eps, xs))
        fin = max(eps)
        return (median([d[e] for e in eps if e > fin - 30])
                - median([d[e] for e in eps if 21 <= e <= 60]))
    for name, pop in (('21x controls', AUD.CTRL_21), ('4x controls', AUD.CTRL_4),
                      ('21x held-out', A2.HELD_21), ('4x held-out', A2.HELD_4),
                      ('MAE', A2.MAE), ('ViT-Tiny flat', A2.TINY_FLAT)):
        res[name] = [raw(r) for r in pop if r in S]
    return res


def pearson(a, b):
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / den if den > 0 else float('nan')


def report():
    import knee_lidar_audit as AUD
    import knee_lidar_audit_sharedshape as A2
    series = {('redrawn', 0): PS0['redrawn'], ('shared', 0): PS0['shared'],
              ('redrawn', 1): OUT['redrawn'], ('shared', 1): OUT['shared']}
    C = {k: series_counts(p) for k, p in series.items()}
    print('== four-way table, N = 100 (protocol x replica seed) ==')
    for k in series:
        c = C[k]
        thr = c['thr']
        print(f'\n-- {k[0]} shape, ps{k[1]}: CUSUM thr 21x {thr["21x"]["CUSUM"]:.3f}, '
              f'4x {thr["4x"]["CUSUM"]:.3f}')
        for t in A2.TESTS:
            parts = []
            for camp in ('21x', '4x'):
                n, tot, d = c[('knee', camp, t)]
                near = sum(abs(x) <= 10 for x in d)
                parts.append(f'knee {camp} {n}/{tot} (within one grid step {near}; '
                             f'alarm-onset {" ".join(f"{x:+d}" for x in sorted(d)) or "-"})')
            ho, hn = c[('heldout', t)]
            mo, mn = c[('mae', t)]
            to, tn = c[('tiny', t)]
            parts.append(f'held-out LeJEPA {len(ho)}/{hn}'
                         + (f' ({", ".join(A2.short(r) for r in ho)})' if ho else ''))
            parts.append(f'MAE {len(mo)}/{mn}' + (f' ({", ".join(A2.short(r) for r in mo)})' if mo else ''))
            parts.append(f'ViT-Tiny flat {len(to)}/{tn}')
            print(f'  {t:12s} ' + ' | '.join(parts))
        print(f'  any of four: held-out LeJEPA {len(c[("heldout", "any")][0])}/'
              f'{c[("heldout", "any")][1]}, MAE {len(c[("mae", "any")][0])}/{c[("mae", "any")][1]}, '
              f'ViT-Tiny flat {len(c[("tiny", "any")][0])}/{c[("tiny", "any")][1]}')

    print('\n== level change (knee: PRE -> POST; event-free: 2026-09-05 section 4b raw change) ==')
    for k in series:
        L = level_changes(C[k]['S'])
        cells = []
        for name, v in L.items():
            extra = f', rises {sum(x > 0 for x in v)}/{len(v)}' if name.startswith('knee') else ''
            cells.append(f'{name}: median {median(v):+.2f} ({min(v):+.2f} to {max(v):+.2f}){extra}')
        print(f'  {k[0]:7s} ps{k[1]}: ' + '; '.join(cells))

    print('\n== per-run Pearson correlation of the ps1 series with the ps0 series (all checkpoints) ==')
    for proto in ('redrawn', 'shared'):
        S0, S1 = C[(proto, 0)]['S'], C[(proto, 1)]['S']
        by = {}
        for run in S0:
            e0, x0 = S0[run]
            if run not in S1:
                continue
            e1, x1 = S1[run]
            d1 = dict(zip(e1, x1))
            common = [(a, d1[e]) for e, a in zip(e0, x0) if e in d1]
            by.setdefault(A2.role(run), []).append(
                (A2.short(run), pearson([a for a, _ in common], [b for _, b in common])))
        for role, v in sorted(by.items()):
            vals = [x for _, x in v]
            lo = min(v, key=lambda x: x[1])
            print(f'  {proto:7s} {role:9s} n={len(v):2d}: median {median(vals):+.3f}, '
                  f'min {lo[1]:+.3f} ({lo[0]})')


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
