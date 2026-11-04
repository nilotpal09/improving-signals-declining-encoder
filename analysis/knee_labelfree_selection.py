"""Within-run checkpoint selection by the label-free metrics, the two LiDAR
series and the training loss (2026-09-24). Replaces four earlier one-off
scripts (select.py, clock.py, clock_all.py, window.py; not included) with one
tested calculation. Reads CSVs and the run logs only;
writes one new CSV (refuses to overwrite it). No GPU, no training.

DEFINITIONS (one place)
  candidates  the grid checkpoints of a run with epoch > 20 (WARM, as in the
              S2 audit): 30, 40, ... every 10 epochs to 150 or 300.
  own-direction quality q(x): x if the metric's source says higher is better,
              -x if lower, -|x - 1| for alpha-ReQ ("closer to 1"). Sources
              and quotes in the table below.
  pick        the candidate with the largest q. Ties: the earliest epoch; the
              number of other candidates sharing the best q is recorded
              (n_ties; float equality).
  R2(E) cost  max over the candidates of ridge R^2(E) minus R^2(E) at the
              pick. mean-CCA cost: the same with mean CCA. Readability from
              results/knee_labelfree_readability.csv via knee_labelfree_audit.
  reference epoch  knee runs: their onset (knee_lidar_audit.KNEE_21/_4).
              Event-free runs and MAE: the campaign's median knee onset
              (pseudo-onset; 115 at 21x, 90 at 4x; ViT-Tiny and MAE count as
              21x, by knee_lidar_audit_sharedshape.campaign).
  drift       within-run Spearman (knee_lidar_vs_readability.spearman) of q
              against epoch over the candidates in a window, if the window
              holds >= 4 checkpoints (MIN_RHO). Windows: before = epoch <
              reference, after = epoch >= reference, whole = all candidates.
              Reported as the median over runs and the count of runs with
              rho > 0. A run whose q is constant in the window gets NaN and is
              left out of the median (none occurs on this data).
  baselines   fixed rules on the same candidates: first checkpoint > 20,
              final checkpoint, epoch 60; the oracle is the max-R2(E)
              checkpoint itself (series 'readability R2(E)', direction higher;
              'readability mean CCA' likewise).
  populations knee (27: knee_lidar_audit.KNEE_21 + KNEE_4); event-free LeJEPA
              (20: CTRL_21 + CTRL_4 + HELD_21 + HELD_4 + TINY_FLAT; the matched
              window table leaves out ViT-Tiny, 19, as the earlier window.py
              did); MAE (10). tiny_2 (a knee run outside the record's
              populations) is left out, as in the record.

DIRECTIONS AND SOURCES (paper texts: survey arXiv 2608.23182v1 and each
metric's paper; quotes below are exact up to ligatures and symbol spelling)
  alpha-ReQ (centred, code_wls, uncentred): CLOSER TO 1, q = -|alpha - 1|.
    Agrawal et al., NeurIPS 2022, sec. 1, contribution 2: "Generally, the best
    representations are those where alpha is in a range that is close to 1";
    sec. 2.2: "when alpha is close to 1, the generalization properties are at
    their best". The evidence is across models: sec. 3.2 "Label-agnostic
    metric for Model selection" compares Barlow Twins models over
    hyperparameters. No within-run statement. Survey sec. 3.1: a
    "'Goldilocks' regime around alpha ~ 1". On this data no alpha value is
    below 1 (min 1.44; counted per run in n_below_1), so the pick equals the
    "lower alpha" pick the earlier scripts used (checked; column pick_if_lower).
  RankMe (pub_uncentred, pub_centred, sq_centred): HIGHER.
    Garrido et al., ICML 2023, sec. 4.1: "having a higher rank is necessary for
    better performance, and using RankMe to find the best value of an
    hyperparameter is as simple as choosing the value that leads to the
    highest rank"; Fig. 2 caption: "having a high rank is a necessary
    condition for good downstream performance". Across hyperparameter
    settings; no within-run statement found (text fetched from arXiv
    2210.02885).
  NE-Sum (centred, uncentred): HIGHER, FLAGGED (source non-monotone).
    He & Ozay, ICML 2022, sec. 4: "we view larger values of NESum to denote
    whiter representations", but "too high a value of NESum ... results in
    lower test accuracy too" and "The relationship between extent of feature
    whitening & generalisation performance in SSL is not monotonic." The
    direction used is the survey's (sec. 3.1: "higher NE Sum values are
    associated with better downstream performance") and Tsitsulin et al.'s
    (sec. 4.1.1: "High stable rank, NESum, and coherence seem to indicate good
    test test performance"). Across models.
  kappa, condition number (uncentred, centred): LOWER, FLAGGED (signs differ).
    Tsitsulin, Munkhoeva & Perozzi 2023, sec. 3.2: "kappa_2 can be used to
    bound the sensitivity of the system to the change in the input" and "We
    use kappa_2 to measure stability of learned representations"; survey sec.
    3.1: "a small kappa indicates that the representation is
    well-conditioned". But the reported correlations point the other way:
    survey sec. 4.2, "kappa shows a statistically significant,
    mild-to-moderate positive correlation" with OLS accuracy (attributed there
    to the OLS solver); Tsitsulin Fig. 1 reports rho = -0.48 for a
    'Cond. Num.' axis spanning 1e-6 to 1e-2, which cannot be sigma_1/sigma_n
    >= 1 (possibly its reciprocal; the paper does not say); Tsitsulin Table 4
    (graph embeddings): +0.53 and -0.43. Across models.
  Self-Cluster (sq_uncentred, lit_uncentred, sq_centred, lit_centred): LOWER,
    FLAGGED (signs differ). Survey sec. 3.2: "one can expect that a lower
    Self-Cluster value corresponds to higher downstream accuracy"; Tsitsulin
    Fig. 2 (ImageNet validation): rho = -0.31; but Tsitsulin Table 4 (graph
    embeddings): +0.55 and +0.60, "Only SelfCluster and Coherence are aligned
    with the downstream evaluation". Across models.
  DSE (code, paper, paper_codebw, code_l2, code_l2_centred): HIGHER, FLAGGED.
    Liao et al., CISS 2024, sec. 5.3 (the one source with a within-training
    statement): "DSE in proper learning (i.e. supervised or contrastive
    learning on correct labels) increases as the model performs better on the
    classification task. When the model is forced to memorize random nonsense
    labels, the entropy increases similarly even though the classification
    performance is stagnant." Sec. 5.5, across 962 ImageNet models: "S_D(Z; X)
    and I_D(Z; X) do not show strong statistical significance" (its Fig. 10
    plots DSE as S_D(Z); only DSMI I_D(Z; Y) is reported as strongly
    correlated). Survey sec. 3.2: "High DSE
    indicates ... representations with a rich, distributed structure".
  ID, TwoNN (fit, mle): LOWER, FLAGGED. Ansuini et al., NeurIPS 2019, sec. 1:
    "the lower the ID in this layer, the better the network capability of
    correctly classifying the image categories in a test set" (across
    networks, sec. 3.2); sec. 4: "the ID, after a large, initial drop, can
    slightly increase during training without producing overfitting". Survey
    sec. 3.3: lower ID "is associated with better generalisation performance".
  LiDAR (redrawn, shared): HIGHER. Thilak et al., arXiv 2312.04000, sec. 5:
    "LiDAR consistently selects more performant hyperparameters than both
    variants of RankMe"; sec. 4, "Top ranking model": "the top ranking model
    given a metric of interest"; sec. 2.2 motivates early stopping ("finding
    an appropriate early stopping criterion is critical"). Its checkpoint
    tables correlate across hyperparameter settings at each of several epochs
    (VICReg Table 7), not within one run. The redrawn series lies outside
    LiDAR's stated domain; its direction is the paper's.
  loss: LOWER (the quantity training minimises; not a quality metric from a
    paper; LiDAR sec. 2.2: "the loss used is often uninformative of the
    actual metric of interest").
  Every label-free variant is also scored in the REVERSED direction (series
  '<name> [reversed]'; 'farther from 1' for alpha-ReQ) as a sensitivity row.

TRAINING LOSS AND ITS ALIGNMENT
  Read exactly as knee_loss_census.series reads it (imported): per logged
  epoch, the mean of the 'loss' rows of lejepa_diag_r0.csv (LeJEPA; one row
  every 10 iterations, 50 of 500) or mae_r0.csv (MAE; every iteration), run
  directory glob <run>xxx* under knee_loss_census.ROOT (exactly one match
  asserted). Alignment: checkpoint file <tag>-ep{N}.pth.tar holds the model
  after N - 1 completed epochs, and the logs number epochs from 1
  (train/src/train_lejepa.py, NOTE at line 686, and
  src/mae/train.py, save_checkpoint(epoch + 1)). So the loss at grid epoch N
  is the logged epoch N - 1: the epoch whose updates end in checkpoint N.
  --test checks this on real checkpoints: the stored 'epoch' is N - 1 and the
  stored 'loss' (the all-iteration mean) equals the mean of jepa_r0.csv /
  mae_r0.csv at logged epoch N - 1. Two sensitivity series: 'loss [census at
  epoch N]' (the naive alignment) and 'loss [all iterations, N-1]' (the
  every-iteration log, jepa_r0.csv / mae_r0.csv, i.e. the checkpoint's stored
  loss up to 5-decimal rounding of the rows).

Loaders imported, not copied: knee_labelfree_audit.load_metrics,
load_readability, load_lidar with LIDAR[...]; populations from
knee_lidar_audit and knee_lidar_audit_sharedshape; Spearman from
knee_lidar_vs_readability; the loss reader from knee_loss_census. The S2a
lines of the earlier select.py are reproduced in --test with
knee_s2_monitors.local_dev / rank_at.

Usage (from analysis/):
  python -B knee_labelfree_selection.py --test     # synthetic + real-data tests; writes nothing
  python -B knee_labelfree_selection.py            # writes results/knee_labelfree_selection.csv, prints tables
  python -B knee_labelfree_selection.py --report   # tables from the CSV
"""
import sys

sys.dont_write_bytecode = True      # leave analysis/__pycache__ untouched

import argparse  # noqa: E402
import csv  # noqa: E402
import glob  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
from collections import defaultdict  # noqa: E402
from statistics import median  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
sys.path.insert(0, _HERE)
import knee_labelfree_audit as LA  # noqa: E402
import knee_lidar_audit as AUD  # noqa: E402
import knee_lidar_audit_sharedshape as A2  # noqa: E402
import knee_lidar_vs_readability as LVR  # noqa: E402
import knee_loss_census as LC  # noqa: E402

OUT = os.path.join(paths.CSV, 'knee_labelfree_selection.csv')
WARM = 20
MIN_RHO = 4

KNEE = dict(AUD.KNEE_21)
KNEE.update(AUD.KNEE_4)
FREE = AUD.CTRL_21 + AUD.CTRL_4 + A2.HELD_21 + A2.HELD_4 + A2.TINY_FLAT
MAE = list(A2.MAE)
GROUPS = [('knee', list(KNEE)), ('event-free LeJEPA', FREE), ('MAE', MAE)]
ALL_RUNS = list(KNEE) + FREE + MAE
PSEUDO = {'21x': median(AUD.KNEE_21.values()), '4x': median(AUD.KNEE_4.values())}

DIRECTION = {'alpha_req': 'near1', 'rankme': 'higher', 'ne_sum': 'higher', 'kappa': 'lower',
             'self_cluster': 'lower', 'dse': 'higher', 'id_twonn': 'lower'}
FLAG = {'alpha_req': 'across models',
        'rankme': 'across models',
        'ne_sum': 'across models; source non-monotone',
        'kappa': 'across models; reported signs differ',
        'self_cluster': 'across models; reported signs differ',
        'dse': 'within training; also rises under memorisation',
        'id_twonn': 'across models; source notes a within-training rise',
        'lidar': 'across hyperparameter settings',
        'loss': 'training objective',
        'readability': 'labels (oracle)',
        'rule': 'fixed rule'}
REVERSE = {'higher': 'lower', 'lower': 'higher', 'near1': 'far1', 'far1': 'near1'}
SHOW_DIR = {'higher': 'higher', 'lower': 'lower', 'near1': 'closer to 1', 'far1': 'farther from 1',
            'rule': 'rule'}
RULES = [('rule: first checkpoint > 20', 'first'), ('rule: final checkpoint', 'final'),
         ('rule: epoch 60', 'epoch60')]
COLS = ['series', 'kind', 'direction', 'source_flag', 'run', 'group', 'campaign', 'onset', 'ref_epoch',
        'n_cand', 'first_epoch', 'final_epoch', 'pick_epoch', 'pick_value', 'n_ties', 'pick_if_lower',
        'n_below_1', 'r2e_pick', 'r2e_max', 'r2e_max_epoch', 'r2e_cost', 'cca_pick', 'cca_max',
        'cca_max_epoch', 'cca_cost', 'pick_minus_ref', 'rho_whole', 'n_whole', 'rho_before', 'n_before',
        'rho_after', 'n_after']


# ================================================================ core

def own(direction, x):
    """Own-direction quality: larger = better by the metric's source."""
    if direction == 'higher':
        return x
    if direction == 'lower':
        return -x
    if direction == 'near1':
        return -abs(x - 1.0)
    if direction == 'far1':
        return abs(x - 1.0)
    raise ValueError(direction)


def pick(eps, xs, direction, warm=WARM):
    """(epoch of the best own-direction value among epochs > warm, number of
    other candidates tied with it). Ties go to the earliest epoch."""
    cand = [(e, own(direction, x)) for e, x in zip(eps, xs) if e > warm]
    best = max(q for _, q in cand)
    tied = sorted(e for e, q in cand if q == best)
    return tied[0], len(tied) - 1


def rule_pick(eps, rule, warm=WARM):
    cand = [e for e in eps if e > warm]
    if rule == 'first':
        return min(cand)
    if rule == 'final':
        return max(cand)
    if rule == 'epoch60':
        assert 60 in cand, 'epoch 60 not on the grid'
        return 60
    raise ValueError(rule)


def readability_costs(run, eps, pick_e, rd, warm=WARM):
    cand = [e for e in eps if e > warm]
    out = {}
    for qi, tag in ((0, 'r2e'), (1, 'cca')):
        vals = {e: rd[(run, e)][qi] for e in cand}
        mx = max(vals.values())
        out[f'{tag}_pick'] = vals[pick_e]
        out[f'{tag}_max'] = mx
        out[f'{tag}_max_epoch'] = min(e for e in cand if vals[e] == mx)
        out[f'{tag}_cost'] = mx - vals[pick_e]
    return out


def drift(eps, xs, direction, lo=None, hi=None, min_n=MIN_RHO, warm=WARM):
    """(Spearman of own-direction quality vs epoch over candidates with
    lo <= epoch < hi, number of checkpoints). rho is None when fewer than
    min_n checkpoints, NaN when the quality is constant."""
    pairs = [(e, own(direction, x)) for e, x in zip(eps, xs)
             if e > warm and (lo is None or e >= lo) and (hi is None or e < hi)]
    if len(pairs) < min_n:
        return None, len(pairs)
    q = [p[1] for p in pairs]
    if max(q) == min(q):
        return float('nan'), len(pairs)
    return LVR.spearman([p[0] for p in pairs], q), len(pairs)


def group_of(run):
    for g, runs in GROUPS:
        if run in runs:
            return g
    raise KeyError(run)


def ref_epoch(run):
    return KNEE[run] if run in KNEE else PSEUDO[A2.campaign(run)]


# ================================================================ series

_LOSS_ALL = {}


def loss_all_iterations(run_dir, kind):
    """Per logged epoch, the mean over every iteration (jepa_r0.csv for
    LeJEPA, mae_r0.csv for MAE): the checkpoint's stored loss up to the
    5-decimal rounding of the rows. Read from results/loss_all_iterations.csv
    (make_loss_all_iterations.py): the LeJEPA every-iteration logs are not in
    the released record."""
    if not _LOSS_ALL:
        for r in csv.DictReader(open(os.path.join(paths.CSV, 'loss_all_iterations.csv'))):
            _LOSS_ALL.setdefault(r['run'], []).append((int(r['epoch']), float(r['loss_mean'])))
    s = sorted(_LOSS_ALL[os.path.basename(run_dir).split('xxx')[0]])
    return [e for e, _ in s], [v for _, v in s]


def run_dir(run):
    hits = glob.glob(os.path.join(LC.ROOT, run + 'xxx*'))
    assert len(hits) == 1, (run, hits)
    return hits[0]


def loss_series(grid, source='census', shift=1):
    """run -> (grid epochs, loss at logged epoch N - shift for grid epoch N)."""
    out = {}
    for run in ALL_RUNS:
        kind = 'mae' if run in MAE else 'lejepa'
        d = run_dir(run)
        if source == 'census':
            eps, loss = LC.series(os.path.basename(d), kind)
        else:
            eps, loss = loss_all_iterations(d, kind)
        by = dict(zip(eps, loss))
        g = grid[run]
        out[run] = (g, [by[e - shift] for e in g])
    return out


def load_all():
    """[(name, kind, metric, direction, S)], rd; S: run -> (epochs, values)."""
    from knee_labelfree_metrics import VARIANTS
    M = LA.load_metrics()
    rd = LA.load_readability()
    grid = {r: sorted(e for (rr, e) in rd if rr == r) for r in ALL_RUNS}
    series = []
    for m, v, *_ in VARIANTS:
        series.append((f'{m} {v}', 'metric', m, DIRECTION[m], M[(m, v)]))
    for name, path in LA.LIDAR.items():
        series.append((name, 'lidar', 'lidar', 'higher', LA.load_lidar(path)))
    series.append(('loss', 'loss', 'loss', 'lower', loss_series(grid)))
    series.append(('loss [census at epoch N]', 'loss-sensitivity', 'loss', 'lower',
                   loss_series(grid, shift=0)))
    series.append(('loss [all iterations, N-1]', 'loss-sensitivity', 'loss', 'lower',
                   loss_series(grid, source='all')))
    for qi, name in ((0, 'readability R2(E)'), (1, 'readability mean CCA')):
        series.append((name, 'oracle' if qi == 0 else 'reference', 'readability', 'higher',
                       {r: (grid[r], [rd[(r, e)][qi] for e in grid[r]]) for r in ALL_RUNS}))
    for name, kind, m, d, S in series:
        for r in ALL_RUNS:
            assert S[r][0] == grid[r], (name, r)
            assert all(math.isfinite(x) for x in S[r][1]), (name, r)
    return series, rd, grid


# ================================================================ rows

def make_row(name, kind, direction, flag, run, eps, xs, rd, rule=None):
    g = group_of(run)
    ref = ref_epoch(run)
    cand = [e for e in eps if e > WARM]
    if rule is None:
        pe, nt = pick(eps, xs, direction)
        pval = xs[eps.index(pe)]
    else:
        pe, nt, pval = rule_pick(eps, rule), None, None
    row = dict(series=name, kind=kind, direction=direction, source_flag=flag, run=run, group=g,
               campaign=A2.campaign(run), onset=KNEE.get(run), ref_epoch=ref, n_cand=len(cand),
               first_epoch=min(cand), final_epoch=max(cand), pick_epoch=pe, pick_value=pval, n_ties=nt,
               pick_if_lower=None, n_below_1=None, pick_minus_ref=pe - ref)
    row.update(readability_costs(run, eps, pe, rd))
    if direction in ('near1', 'far1'):
        row['n_below_1'] = sum(x < 1.0 for x in xs)
        row['pick_if_lower'] = pick(eps, xs, 'lower')[0] if direction == 'near1' else None
    for key, lo, hi in (('whole', None, None), ('before', None, ref), ('after', ref, None)):
        if rule is None:
            rho, n = drift(eps, xs, direction, lo, hi)
        else:
            rho, n = None, None
        row[f'rho_{key}'], row[f'n_{key}'] = rho, n
    return row


def compute():
    series, rd, grid = load_all()
    rows = []
    for name, kind, m, d, S in series:
        for run in ALL_RUNS:
            rows.append(make_row(name, kind, d, FLAG[m], run, *S[run], rd))
        if kind == 'metric':
            for run in ALL_RUNS:
                rows.append(make_row(f'{name} [reversed]', 'metric-reversed', REVERSE[d], FLAG[m], run,
                                     *S[run], rd))
    for name, rule in RULES:
        for run in ALL_RUNS:
            rows.append(make_row(name, 'rule', 'rule', FLAG['rule'], run, grid[run], None, rd, rule=rule))
    return rows


def fmt(v, key):
    if v is None:
        return ''
    if isinstance(v, float):
        if math.isnan(v):
            return 'nan'
        if key == 'pick_value':
            return f'{v:.10g}'
        if key in ('ref_epoch', 'pick_minus_ref'):
            return f'{v:g}'
        return f'{v:.6f}'
    return str(v)


def to_strings(rows):
    return [{k: fmt(r[k], k) for k in COLS} for r in rows]


def write_csv(rows):
    assert not os.path.exists(OUT), f'refusing to overwrite {OUT}'
    part = OUT + '.partial'
    assert not os.path.exists(part), f'refusing to overwrite {part}'
    with open(part, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, lineterminator='\n')
        w.writeheader()
        w.writerows(to_strings(rows))
    os.replace(part, OUT)
    print(f'wrote {OUT}: {len(rows)} rows')


INTS = {'onset', 'n_cand', 'first_epoch', 'final_epoch', 'pick_epoch', 'n_ties', 'pick_if_lower', 'n_below_1',
        'r2e_max_epoch', 'cca_max_epoch', 'n_whole', 'n_before', 'n_after'}
FLOATS = {'ref_epoch', 'pick_value', 'r2e_pick', 'r2e_max', 'r2e_cost', 'cca_pick', 'cca_max', 'cca_cost',
          'pick_minus_ref', 'rho_whole', 'rho_before', 'rho_after'}


def read_csv(path=OUT):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            for k in INTS:
                r[k] = int(r[k]) if r[k] != '' else None
            for k in FLOATS:
                r[k] = float(r[k]) if r[k] != '' else None
            rows.append(r)
    return rows


# ================================================================ report

def _md(v, spec):
    return format(v, spec)


def cell_rho(rs):
    """median (runs > 0 / runs) over defined, non-NaN rho values."""
    vals = [r for r in rs if r is not None and not math.isnan(r)]
    nan = sum(1 for r in rs if r is not None and math.isnan(r))
    if not vals:
        return '-'
    return f'{median(vals):+.2f} ({sum(v > 0 for v in vals)}/{len(vals)})' + (f' [{nan} const]' if nan else '')


def report(rows, out=print):
    by = defaultdict(list)
    for r in rows:
        by[(r['series'], r['group'])].append(r)
    order = []
    for r in rows:
        if r['series'] not in order:
            order.append(r['series'])
    kinds = {r['series']: r['kind'] for r in rows}
    dirs = {r['series']: r['direction'] for r in rows}
    flags = {r['series']: r['source_flag'] for r in rows}
    main = [s for s in order if kinds[s] in ('metric', 'lidar', 'loss')] + \
           [s for s in order if kinds[s] in ('oracle', 'reference', 'rule')]

    out(f'# Within-run checkpoint selection (csv/knee_labelfree_selection.csv, {len(rows)} rows)\n')
    out(f'Candidates: grid checkpoints with epoch > {WARM}. Reference epoch: onset for knee runs; '
        f'campaign median knee onset for the others ({PSEUDO["21x"]:g} at 21x, {PSEUDO["4x"]:g} at 4x). '
        f'Drift: within-run Spearman of own-direction quality vs epoch, >= {MIN_RHO} checkpoints.\n')

    def sel_table(groupname, series_list, title):
        ref = 'onset' if groupname == 'knee' else 'pseudo-onset'
        out(f'\n## {title}\n')
        out(f'| series | direction | pick at/after {ref} | pick = first ckpt > 20 | pick = final ckpt | '
            f'R2(E) cost median (max) | mean-CCA cost median (max) | pick - {ref}, median | '
            f'runs with ties (tied ckpts) |')
        out('|---|---|---|---|---|---|---|---|---|')
        for s in series_list:
            rs = by[(s, groupname)]
            n = len(rs)
            post = sum(r['pick_epoch'] >= r['ref_epoch'] for r in rs)
            first = sum(r['pick_epoch'] == r['first_epoch'] for r in rs)
            final = sum(r['pick_epoch'] == r['final_epoch'] for r in rs)
            c = [r['r2e_cost'] for r in rs]
            cc = [r['cca_cost'] for r in rs]
            off = median(r['pick_minus_ref'] for r in rs)
            if kinds[s] == 'rule':
                ties = '-'
            else:
                ties = f'{sum(r["n_ties"] > 0 for r in rs)} ({sum(r["n_ties"] for r in rs)})'
            out(f'| {s} | {SHOW_DIR[dirs[s]]} | {post} / {n} | {first} / {n} | {final} / {n} | '
                f'{median(c):.3f} ({max(c):.3f}) | {median(cc):.3f} ({max(cc):.3f}) | {off:+g} | {ties} |')

    for g, runs in GROUPS:
        sel_table(g, main, f'Selection, {g} runs ({len(runs)})')

    loss_rows = [s for s in order if kinds[s] in ('loss', 'loss-sensitivity')]
    sel_table('knee', loss_rows, 'Loss alignment sensitivity, knee runs (27)')
    sel_table('event-free LeJEPA', loss_rows, 'Loss alignment sensitivity, event-free LeJEPA runs (20)')

    out('\n## Reversed direction (sensitivity), label-free variants\n')
    out('| series | reversed direction | knee: pick at/after onset | knee: pick = final ckpt | '
        'knee: R2(E) cost median (max) | event-free: R2(E) cost median (max) | '
        'own direction, knee: at/after onset; R2(E) cost median |')
    out('|---|---|---|---|---|---|---|')
    for s in order:
        if kinds[s] != 'metric-reversed':
            continue
        base = s.replace(' [reversed]', '')
        k, f, kb = by[(s, 'knee')], by[(s, 'event-free LeJEPA')], by[(base, 'knee')]
        out(f'| {base} | {SHOW_DIR[dirs[s]]} | {sum(r["pick_epoch"] >= r["ref_epoch"] for r in k)} / {len(k)} | '
            f'{sum(r["pick_epoch"] == r["final_epoch"] for r in k)} / {len(k)} | '
            f'{median(r["r2e_cost"] for r in k):.3f} ({max(r["r2e_cost"] for r in k):.3f}) | '
            f'{median(r["r2e_cost"] for r in f):.3f} ({max(r["r2e_cost"] for r in f):.3f}) | '
            f'{sum(r["pick_epoch"] >= r["ref_epoch"] for r in kb)} / {len(kb)}; '
            f'{median(r["r2e_cost"] for r in kb):.3f} |')

    out('\n## Drift: own-direction quality vs epoch, within-run Spearman, median (runs > 0 / runs)\n')
    out('Event-free matched window: the 19 event-free LeJEPA runs without ViT-Tiny, split at the '
        'pseudo-onset. Whole run: all candidates (event-free: 20 runs with ViT-Tiny).\n')
    out('| series | direction | knee, before onset | knee, after onset | event-free, before pseudo-onset | '
        'event-free, after pseudo-onset | knee, whole run | event-free, whole run | MAE, whole run |')
    out('|---|---|---|---|---|---|---|---|---|')
    drift_series = [s for s in order if kinds[s] in ('metric', 'lidar', 'loss', 'oracle', 'reference')]
    agree = []
    for s in drift_series:
        k = by[(s, 'knee')]
        f19 = [r for r in by[(s, 'event-free LeJEPA')] if r['run'] not in A2.TINY_FLAT]
        f20 = by[(s, 'event-free LeJEPA')]
        m = by[(s, 'MAE')]
        cells = [cell_rho([r['rho_before'] for r in k]), cell_rho([r['rho_after'] for r in k]),
                 cell_rho([r['rho_before'] for r in f19]), cell_rho([r['rho_after'] for r in f19]),
                 cell_rho([r['rho_whole'] for r in k]), cell_rho([r['rho_whole'] for r in f20]),
                 cell_rho([r['rho_whole'] for r in m])]
        out(f'| {s} | {SHOW_DIR[dirs[s]]} | ' + ' | '.join(cells) + ' |')

        def med(rs, key):
            v = [r[key] for r in rs if r[key] is not None and not math.isnan(r[key])]
            return median(v)
        if kinds[s] == 'metric':
            agree.append((s, [math.copysign(1, med(k, 'rho_before')), math.copysign(1, med(k, 'rho_after'))],
                          [math.copysign(1, med(f19, 'rho_before')), math.copysign(1, med(f19, 'rho_after'))]))
    both = [s for s, a, b in agree if a == b]
    out(f'\nLabel-free variants whose median drift has the same sign in the knee runs and in the matched '
        f'event-free window, both before and after: {len(both)} of {len(agree)} '
        f'({", ".join(both)}). Differing: '
        + ', '.join(f'{s} (knee {"".join("+" if x > 0 else "-" for x in a)}, event-free '
                    f'{"".join("+" if x > 0 else "-" for x in b)})' for s, a, b in agree if a != b) + '.')

    out('\n## alpha-ReQ: values below 1 and the "lower" pick\n')
    out('| series | checkpoints with alpha < 1 (all 57 runs, all grid epochs) | runs where the |alpha - 1| pick '
        'differs from the lower-alpha pick |')
    out('|---|---|---|')
    for s in order:
        if dirs[s] == 'near1':
            rs = [r for g, _ in GROUPS for r in by[(s, g)]]
            out(f'| {s} | {sum(r["n_below_1"] for r in rs)} | '
                f'{sum(r["pick_if_lower"] != r["pick_epoch"] for r in rs)} of {len(rs)} |')

    out('\n## Picked epochs, knee runs (sorted), for the series story v2 sec. 1.4 names\n')
    for s in ('alpha_req centred', 'alpha_req code_wls', 'alpha_req uncentred', 'kappa uncentred',
              'kappa centred', 'loss', 'lidar redrawn', 'lidar shared', 'readability R2(E)'):
        rs = by[(s, 'knee')]
        out(f'- {s}: {sorted(r["pick_epoch"] for r in rs)}; pick - onset '
            f'{sorted(r["pick_epoch"] - r["onset"] for r in rs)}')

    out('\n## Source flags\n')
    for s in main:
        if kinds[s] in ('metric', 'lidar', 'loss'):
            out(f'- {s}: {SHOW_DIR[dirs[s]]}; {flags[s]}')


# ================================================================ tests

class T:
    n_pass = 0
    n_fail = 0

    @classmethod
    def check(cls, name, got, expect, tol=None):
        if tol is None:
            ok = got == expect
        else:
            ok = abs(got - expect) <= tol
        if ok:
            cls.n_pass += 1
        else:
            cls.n_fail += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got!r} expect {expect!r}"
              + (f' (tol {tol})' if tol is not None else ''), flush=True)
        return ok


def test_synthetic():
    print('=== synthetic: pick (argmax / argmin over epochs > 20, earliest on ties) ===')
    eps = [2, 10, 20, 30, 40, 50, 60]
    xs = [9.0, 9.0, 9.0, 1.0, 5.0, 3.0, 2.0]
    T.check('higher: epochs <= 20 excluded, argmax at 40', pick(eps, xs, 'higher'), (40, 0))
    T.check('lower: argmin at 30', pick(eps, xs, 'lower'), (30, 0))
    xs2 = [-9.0, -9.0, -9.0, 5.0, 4.0, 3.0, 6.0]
    T.check('lower: argmin at 50 (smaller values at epochs <= 20 ignored)', pick(eps, xs2, 'lower'), (50, 0))
    T.check('two-way tie (40, 50): earliest, 1 tie', pick(eps, [0, 0, 0, 2.0, 5.0, 5.0, 1.0], 'higher'), (40, 1))
    T.check('three-way tie (30, 40, 50) under lower: earliest, 2 ties',
            pick(eps, [0, 0, 0, 1.0, 1.0, 1.0, 2.0], 'lower'), (30, 2))
    print('=== synthetic: closer to 1 ===')
    e5 = [30, 40, 50, 60, 70]
    a = [1.5, 0.875, 1.25, 2.0, 0.75]          # |a - 1| = .5 .125 .25 1 .25 (exact in binary)
    T.check('near1 picks alpha = 0.875 (epoch 40)', pick(e5, a, 'near1'), (40, 0))
    T.check('lower picks alpha = 0.75 (epoch 70): differs from near1 when alpha < 1', pick(e5, a, 'lower'), (70, 0))
    T.check('far1 picks alpha = 2.0 (epoch 60)', pick(e5, a, 'far1'), (60, 0))
    T.check('near1 tie 1.25 / 0.75 (epochs 30, 50): earliest, 1 tie', pick([30, 40, 50], [1.25, 3.0, 0.75], 'near1'),
            (30, 1))
    a_gt1 = [3.0, 2.5, 1.5, 2.0, 4.0]
    T.check('near1 = lower when every alpha > 1', pick(e5, a_gt1, 'near1'), pick(e5, a_gt1, 'lower'))
    print('=== synthetic: rules ===')
    g = [2, 20, 30, 40, 50, 60, 70]
    T.check('rule first > 20', rule_pick(g, 'first'), 30)
    T.check('rule final', rule_pick(g, 'final'), 70)
    T.check('rule epoch 60', rule_pick(g, 'epoch60'), 60)
    print('=== synthetic: Spearman and drift sign ===')
    T.check('LVR.spearman hand example [1,2,3,4] vs [1,3,2,4] = 1 - 6*2/(4*15) = 0.8',
            LVR.spearman([1, 2, 3, 4], [1, 3, 2, 4]), 0.8, tol=1e-12)
    m_eps = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    up = [0.1 * i for i in range(10)]
    T.check('increasing series, higher: rho = +1', drift(m_eps, up, 'higher')[0], 1.0, tol=1e-12)
    T.check('increasing series, lower: rho = -1', drift(m_eps, up, 'lower')[0], -1.0, tol=1e-12)
    T.check('alpha falling toward 1 from above, near1: rho = +1',
            drift(m_eps, [3.0 - 0.2 * i for i in range(10)], 'near1')[0], 1.0, tol=1e-12)
    T.check('alpha rising toward 1 from below, near1: rho = +1',
            drift(m_eps, [0.1 + 0.08 * i for i in range(10)], 'near1')[0], 1.0, tol=1e-12)
    T.check('number of points = epochs > 20', drift(m_eps, up, 'higher')[1], 8)
    T.check('window before onset 70: epochs 30-60 (n = 4)', drift(m_eps, up, 'higher', None, 70)[1], 4)
    T.check('window after onset 70: epochs 70-100 (n = 4)', drift(m_eps, up, 'higher', 70, None)[1], 4)
    T.check('window before onset 60: 3 points -> rho None', drift(m_eps, up, 'higher', None, 60), (None, 3))
    rho_c = drift(m_eps, [1.0] * 10, 'higher')[0]
    T.check('constant series -> NaN', math.isnan(rho_c), True)
    print('=== synthetic: cost arithmetic on a hand-built readability table ===')
    rd = {('r', 10): (0.99, 0.95), ('r', 30): (0.90, 0.88), ('r', 40): (0.95, 0.85), ('r', 50): (0.80, 0.70),
          ('r', 60): (0.85, 0.60)}
    ge = [10, 30, 40, 50, 60]
    pe, _ = pick(ge, [100.0, 1.0, 2.0, 5.0, 3.0], 'higher')
    c = readability_costs('r', ge, pe, rd)
    T.check('pick = 50', pe, 50)
    T.check('R2(E) run max over epochs > 20 = 0.95 (epoch 10 = 0.99 excluded)', c['r2e_max'], 0.95)
    T.check('R2(E) max epoch = 40', c['r2e_max_epoch'], 40)
    T.check('R2(E) cost = 0.95 - 0.80', c['r2e_cost'], 0.15, tol=1e-12)
    T.check('mean-CCA cost = 0.88 - 0.70 (CCA max at 30, not at the R2(E) max)', c['cca_cost'], 0.18, tol=1e-12)
    T.check('rule first (30): R2(E) cost 0.05', readability_costs('r', ge, rule_pick(ge, 'first'), rd)['r2e_cost'],
            0.05, tol=1e-12)
    T.check('rule final (60): R2(E) cost 0.10', readability_costs('r', ge, rule_pick(ge, 'final'), rd)['r2e_cost'],
            0.10, tol=1e-12)
    po, _ = pick(ge, [rd[('r', e)][0] for e in ge], 'higher')
    T.check('oracle series (R2(E) itself) picks 40 with cost 0',
            (po, readability_costs('r', ge, po, rd)['r2e_cost']), (40, 0.0))
    onset = 40
    T.check('pick - onset = +10, at/after onset', (pe - onset, pe >= onset), (10, True))
    print('=== synthetic: loss alignment (grid epoch N <- logged epoch N - 1) ===')
    logged = {e: 1.0 / e for e in range(1, 61)}
    gridx = [2, 30, 60]
    T.check('shift 1', [logged[e - 1] for e in gridx], [1.0, 1.0 / 29, 1.0 / 59])


# ---- expected numbers, transcribed from the earlier scripts' own output
# (run 2026-09-24 03:36, python -B; they write nothing)
SEL = {  # select.py: name -> (dir, at/after onset, R2E cost median, max, CCA cost median, offset median)
    'alpha_req centred': (-1, 1, '0.028', '0.118', '0.022', '-60'),
    'alpha_req code_wls': (-1, 1, '0.023', '0.118', '0.015', '-60'),
    'rankme pub_uncentred': (+1, 27, '0.201', '0.260', '0.110', '+170'),
    'rankme sq_centred': (+1, 23, '0.197', '0.262', '0.110', '+150'),
    'ne_sum centred': (+1, 24, '0.205', '0.265', '0.108', '+90'),
    'kappa uncentred': (-1, 0, '0.021', '0.046', '0.012', '-60'),
    'self_cluster sq_uncentred': (-1, 26, '0.205', '0.260', '0.113', '+60'),
    'self_cluster lit_uncentred': (-1, 26, '0.205', '0.260', '0.113', '+60'),
    'dse code': (+1, 21, '0.196', '0.272', '0.101', '+30'),
    'dse paper': (+1, 23, '0.201', '0.272', '0.106', '+30'),
    'id_twonn fit': (-1, 26, '0.199', '0.276', '0.108', '+30'),
    'id_twonn mle': (-1, 26, '0.195', '0.276', '0.106', '+40'),
    'lidar redrawn': (+1, 27, '0.201', '0.261', '0.107', '+160'),
    'lidar shared': (+1, 26, '0.200', '0.261', '0.107', '+130'),
}
SEL_ORACLE = (0, -30)
SEL_PREPOST = 'R2E PRE median 0.967 (range 0.803-0.994); POST median 0.790 (range 0.733-0.849); drop median 0.169'
S2A = {
    ('alpha_req centred', '21x'): (12, 1, 4, '0.04 0.11 0.11 0.17 0.22 0.22 0.33 0.41 0.50 0.56 0.59 0.89'),
    ('alpha_req centred', '4x'): (15, 7, 8, '0.04 0.04 0.04 0.07 0.07 0.07 0.07 0.11 0.41 0.52 0.52 0.59 0.63 0.74 0.89'),
    ('alpha_req code_wls', '21x'): (12, 1, 3, '0.04 0.11 0.19 0.33 0.37 0.41 0.50 0.50 0.52 0.89 0.93 0.96'),
    ('alpha_req code_wls', '4x'): (15, 4, 8, '0.04 0.04 0.04 0.07 0.11 0.11 0.11 0.19 0.22 0.30 0.37 0.44 0.56 0.59 1.00'),
    ('id_twonn fit', '21x'): (12, 1, 5, '0.04 0.11 0.15 0.17 0.17 0.25 0.30 0.48 0.52 0.52 0.59 0.96'),
    ('id_twonn fit', '4x'): (15, 3, 7, '0.04 0.04 0.04 0.11 0.15 0.15 0.19 0.26 0.30 0.30 0.33 0.37 0.41 0.59 0.78'),
    ('lidar redrawn', '21x'): (12, 4, 8, '0.04 0.07 0.07 0.08 0.11 0.15 0.17 0.17 0.22 0.37 0.41 0.41'),
    ('lidar redrawn', '4x'): (15, 7, 9, '0.04 0.07 0.07 0.07 0.07 0.07 0.07 0.15 0.19 0.44 0.59 0.63 0.67 0.89 0.93'),
}
CLOCK_BASE = 'first checkpoint > 20: 0.026 (0.046); final: 0.197 (0.262); epoch 60: 0.005 (0.187)'
CLOCK = {  # clock.py: name -> (first count, picked epochs, knee raw rho, pos, n, event-free raw rho, pos, n,
    #                            event-free cost median, max)
    'alpha_req centred': (24, [30] * 24 + [60, 110, 110], '+0.84', 27, 27, '+0.81', 20, 20, '0.024', '0.054'),
    'kappa uncentred': (22, [30] * 22 + [40, 50, 50, 50, 70], '+0.68', 26, 27, '+0.62', 16, 20, '0.018', '0.054'),
    'alpha_req code_wls': (20, [30] * 20 + [40, 60, 70, 80, 100, 110, 150], '+0.76', 25, 27, '+0.60', 17, 20,
                           '0.022', '0.054'),
}
CLOCK_ALL = {  # clock_all.py: name -> (dir, before rho, pos, n, after rho, pos, n, event-free rho, pos, n, last)
    'alpha_req centred': (-1, '-0.94', 3, 25, '-0.70', 4, 27, '-0.81', 0, 20, 0),
    'kappa uncentred': (-1, '-0.87', 0, 25, '-0.21', 11, 27, '-0.62', 4, 20, 0),
    'rankme pub_uncentred': (+1, '+0.85', 25, 25, '+0.88', 27, 27, '+0.92', 20, 20, 14),
    'rankme sq_centred': (+1, '+0.41', 15, 25, '+0.94', 24, 27, '-0.43', 8, 20, 10),
    'ne_sum centred': (+1, '+0.30', 15, 25, '+0.77', 23, 27, '-0.19', 10, 20, 4),
    'self_cluster sq_uncentred': (-1, '+0.83', 25, 25, '+0.50', 19, 27, '+0.86', 20, 20, 10),
    'dse code': (+1, '+0.83', 25, 25, '-0.78', 4, 27, '+0.01', 10, 20, 0),
    'dse paper': (+1, '+0.86', 25, 25, '-0.57', 5, 27, '+0.13', 12, 20, 0),
    'id_twonn fit': (-1, '+0.98', 25, 25, '-0.75', 1, 27, '+0.09', 12, 20, 0),
    'lidar redrawn': (+1, '+0.14', 16, 25, '+0.83', 25, 27, '+0.02', 11, 20, 8),
    'lidar shared': (+1, '+0.70', 18, 25, '+0.91', 24, 27, '+0.58', 12, 20, 4),
}
WIN_PO = {'21x': 115.0, '4x': 90}
WIN_FREE = {  # window.py: raw metric vs epoch, 19 event-free runs, (before rho, rising, n, after rho, rising, n)
    'id_twonn fit': ('-0.77', 3, 19, '+0.70', 18, 19),
    'dse code': ('+0.93', 18, 19, '-0.76', 6, 19),
    'ne_sum centred': ('-0.58', 4, 19, '+0.36', 11, 19),
    'rankme sq_centred': ('-0.75', 4, 19, '+0.17', 10, 19),
    'lidar redrawn': ('-0.26', 7, 19, '+0.10', 10, 19),
    'R2E': ('+0.71', 17, 19, '+0.71', 17, 19),
}
WIN_KNEE = {
    'id_twonn fit': ('-0.98', 0, 25, '+0.75', 26, 27),
    'dse code': ('+0.83', 25, 25, '-0.78', 4, 27),
    'ne_sum centred': ('+0.30', 15, 25, '+0.77', 23, 27),
    'rankme sq_centred': ('+0.41', 15, 25, '+0.94', 24, 27),
    'R2E': ('+0.14', 14, 25, '+0.17', 18, 27),
}


def _rhos(S, runs, direction, lo, hi, min_n):
    out = []
    for r in runs:
        rho, _ = drift(*S[r], direction, lo(r), hi(r), min_n=min_n)
        if rho is not None:
            out.append(rho)
    return out


def test_real():
    import knee_s2_monitors as S2
    series, rd, grid = load_all()
    S = {name: s for name, _, _, _, s in series}
    S['R2E'] = S['readability R2(E)']
    d = {-1: 'lower', +1: 'higher'}
    none = lambda r: None  # noqa: E731

    print('=== real: series coverage (57 runs, every series on the readability grid, finite) ===')
    T.check('series loaded (21 variants + 2 LiDAR + 3 loss + 2 readability)', len(series), 28)
    T.check('runs covered by every series', all(set(s) >= set(ALL_RUNS) for s in S.values()), True)
    T.check('first candidate epoch is 30 in every run', {min(e for e in grid[r] if e > WARM) for r in ALL_RUNS},
            {30})
    T.check('pseudo-onsets (campaign median knee onset)', PSEUDO, WIN_PO)

    print('=== real: reproduce scratch select.py (knee runs, its directions: alpha "lower") ===')
    for name, (dr, post_x, cm_x, cx_x, ccm_x, off_x) in SEL.items():
        post, cost, ccost, off = 0, [], [], []
        for r, o in KNEE.items():
            pe, _ = pick(*S[name][r], d[dr])
            c = readability_costs(r, S[name][r][0], pe, rd)
            post += pe >= o
            cost.append(c['r2e_cost'])
            ccost.append(c['cca_cost'])
            off.append(pe - o)
        T.check(f'select {name}', (post, f'{median(cost):.3f}', f'{max(cost):.3f}', f'{median(ccost):.3f}',
                                   f'{median(off):+.0f}'), (post_x, cm_x, cx_x, ccm_x, off_x))
    off = [pick(*S['R2E'][r], 'higher')[0] - o for r, o in KNEE.items()]
    T.check('select max-R2E checkpoint: at/after onset, epoch - onset median',
            (sum(x >= 0 for x in off), median(off)), SEL_ORACLE)
    pre, pst = [], []
    for r, o in KNEE.items():
        eps = S['R2E'][r][0]
        pre.append(rd[(r, max(e for e in eps if e <= o - 20))][0])
        pst.append(rd[(r, max(eps))][0])
    T.check('select PRE/POST line',
            f'R2E PRE median {median(pre):.3f} (range {min(pre):.3f}-{max(pre):.3f}); POST median '
            f'{median(pst):.3f} (range {min(pst):.3f}-{max(pst):.3f}); drop median '
            f'{median([a - b for a, b in zip(pre, pst)]):.3f}', SEL_PREPOST)
    for (name, camp), (n_x, a_x, b_x, vals_x) in S2A.items():
        knee = AUD.KNEE_21 if camp == '21x' else AUD.KNEE_4
        at = []
        for k, o in knee.items():
            rr = S2.rank_at(S2.local_dev(*S[name][k]), o)
            at.append(rr if rr else float('nan'))
        at_s = sorted(a for a in at if a == a)
        T.check(f'select S2a {name} {camp}', (len(at_s), sum(a <= 0.10 for a in at_s), sum(a <= 0.20 for a in at_s),
                                               ' '.join(f'{a:.2f}' for a in at_s)), (n_x, a_x, b_x, vals_x))

    print('=== real: reproduce scratch clock.py ===')

    def base_cost(rule):
        c = [readability_costs(r, grid[r], rule_pick(grid[r], rule), rd)['r2e_cost'] for r in KNEE]
        return median(c), max(c)
    T.check('clock baselines', 'first checkpoint > 20: %.3f (%.3f); final: %.3f (%.3f); epoch 60: %.3f (%.3f)' % (
        *base_cost('first'), *base_cost('final'), *base_cost('epoch60')), CLOCK_BASE)
    for name, (first_x, picks_x, kr_x, kp_x, kn_x, fr_x, fp_x, fn_x, fc_x, fm_x) in CLOCK.items():
        picks = [pick(*S[name][r], 'lower')[0] for r in KNEE]
        first = sum(p == 30 for p in picks)
        kr = _rhos(S[name], list(KNEE), 'higher', none, none, 5)
        fr = _rhos(S[name], FREE, 'higher', none, none, 5)
        fc = [readability_costs(r, S[name][r][0], pick(*S[name][r], 'lower')[0], rd)['r2e_cost'] for r in FREE]
        T.check(f'clock {name}', (first, sorted(picks), f'{median(kr):+.2f}', sum(x > 0 for x in kr), len(kr),
                                  f'{median(fr):+.2f}', sum(x > 0 for x in fr), len(fr), f'{median(fc):.3f}',
                                  f'{max(fc):.3f}'),
                (first_x, picks_x, kr_x, kp_x, kn_x, fr_x, fp_x, fn_x, fc_x, fm_x))

    print('=== real: reproduce scratch clock_all.py (own direction, >= 4 checkpoints) ===')
    for name, (dr, b_x, bp_x, bn_x, a_x, ap_x, an_x, f_x, fp_x, fn_x, last_x) in CLOCK_ALL.items():
        b = _rhos(S[name], list(KNEE), d[dr], none, lambda r: KNEE[r], 4)
        a = _rhos(S[name], list(KNEE), d[dr], lambda r: KNEE[r], none, 4)
        f = _rhos(S[name], FREE, d[dr], none, none, 4)
        last = sum(pick(*S[name][r], d[dr])[0] == max(S[name][r][0]) for r in KNEE)
        T.check(f'clock_all {name}', (f'{median(b):+.2f}', sum(x > 0 for x in b), len(b), f'{median(a):+.2f}',
                                      sum(x > 0 for x in a), len(a), f'{median(f):+.2f}', sum(x > 0 for x in f),
                                      len(f), last),
                (b_x, bp_x, bn_x, a_x, ap_x, an_x, f_x, fp_x, fn_x, last_x))

    print('=== real: reproduce scratch window.py (raw metric vs epoch, >= 4 checkpoints) ===')
    free19 = [r for r in FREE if r not in A2.TINY_FLAT]
    po = lambda r: PSEUDO[A2.campaign(r)]  # noqa: E731
    for name, (b_x, bp_x, bn_x, a_x, ap_x, an_x) in WIN_FREE.items():
        b = _rhos(S[name], free19, 'higher', none, po, 4)
        a = _rhos(S[name], free19, 'higher', po, none, 4)
        T.check(f'window event-free {name}', (f'{median(b):+.2f}', sum(x > 0 for x in b), len(b), f'{median(a):+.2f}',
                                              sum(x > 0 for x in a), len(a)), (b_x, bp_x, bn_x, a_x, ap_x, an_x))
    for name, (b_x, bp_x, bn_x, a_x, ap_x, an_x) in WIN_KNEE.items():
        b = _rhos(S[name], list(KNEE), 'higher', none, lambda r: KNEE[r], 4)
        a = _rhos(S[name], list(KNEE), 'higher', lambda r: KNEE[r], none, 4)
        T.check(f'window knee {name}', (f'{median(b):+.2f}', sum(x > 0 for x in b), len(b), f'{median(a):+.2f}',
                                        sum(x > 0 for x in a), len(a)), (b_x, bp_x, bn_x, a_x, ap_x, an_x))

    print('=== real: alpha-ReQ values below 1; |alpha - 1| pick = lower-alpha pick ===')
    for v in ('centred', 'code_wls', 'uncentred'):
        s = S[f'alpha_req {v}']
        nb = sum(x < 1 for r in ALL_RUNS for x in s[r][1])
        same = sum(pick(*s[r], 'near1') == pick(*s[r], 'lower') for r in ALL_RUNS)
        T.check(f'alpha_req {v}: checkpoints with alpha < 1 (reported; the scratch assumed 0)', nb, 0)
        T.check(f'alpha_req {v}: runs where the near-1 pick equals the lower pick', same, len(ALL_RUNS))

    print('=== real: loss alignment on stored checkpoints (CPU) ===')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    for run, N in (('rigb_lejp_p1s100_nano_1', 100), ('rigb_lejp_p1s100_nano_7', 30),
                   ('rigb_lejp_p1s100_p384_1', 300), ('rigd_lejp_p1s100_mu097_nano_1', 150),
                   ('rigd_lejp_p1s100_mu04_p384_2', 60), ('rigb_mae_p1s100_long_nano_1', 150)):
        kind = 'mae' if run in MAE else 'lejepa'
        dd = run_dir(run)
        ck = torch.load(os.path.join(dd, f'{"mae" if kind == "mae" else "jepa"}-ep{N}.pth.tar'),
                        map_location='cpu', weights_only=True)
        eps, full = loss_all_iterations(dd, kind)
        full = dict(zip(eps, full))
        ce, cl = LC.series(os.path.basename(dd), kind)
        cen = dict(zip(ce, cl))
        T.check(f'{run} ep{N}: stored epoch = N - 1', int(ck['epoch']), N - 1)
        T.check(f'{run} ep{N}: stored loss = all-iteration mean at logged epoch N - 1 '
                f'(epoch N: {full.get(N, float("nan")):.6f}; census N - 1: {cen[N - 1]:.6f})',
                float(ck['loss']), full[N - 1], tol=1e-5)
        dN1, dN = abs(float(ck['loss']) - full[N - 1]), abs(float(ck['loss']) - full[N])
        T.check(f'{run} ep{N}: stored loss closer to logged epoch N - 1 than to N '
                f'(|diff| {dN1:.1e} vs {dN:.1e})', dN1 < dN, True)
        T.check(f'{run} ep{N}: series "loss" at grid epoch N = census at logged epoch N - 1',
                S['loss'][run][1][S['loss'][run][0].index(N)], cen[N - 1])

    if os.path.exists(OUT):
        print('=== real: the CSV on disk equals a fresh recomputation ===')
        fresh = to_strings(compute())
        with open(OUT) as fh:
            disk = list(csv.DictReader(fh))
        T.check('CSV row count', len(disk), len(fresh))
        T.check('CSV rows identical to recomputation (all fields)', disk == fresh, True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', action='store_true')
    ap.add_argument('--report', action='store_true')
    a = ap.parse_args()
    if a.test:
        test_synthetic()
        test_real()
        print(f'\n{T.n_pass + T.n_fail} checks: {T.n_pass} PASS, {T.n_fail} FAIL')
    elif a.report:
        report(read_csv())
    else:
        assert not os.path.exists(OUT), f'refusing to overwrite {OUT}'
        write_csv(compute())
        report(read_csv())


if __name__ == '__main__':
    main()
