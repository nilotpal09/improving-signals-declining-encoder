"""Label-free representation-quality metrics, their tests, and the pass over
the stored pooled features (2026-09-23). New files only; no training.

Metric list: the three families of Arputharaj, Jonsson & Eilertsen,
arXiv 2608.23182v1 (2026-08-24), "A Comparative Study of Label-free
Representation Quality Metrics in Deep Learning" (below: "survey"). Every
definition was taken from the original paper and, where one exists, from
the authors' code; differences are listed per metric. X is the N x d matrix
of pooled features (rows = events). All arithmetic is float64 unless noted.

SPECTRAL
alpha-ReQ  Agrawal, Mondal, Ghosh & Richards, NeurIPS 2022, sec. 2.1-2.2 and
   3.1: features "assume centered"; Sigma_n = (1/n) sum_i f(x_i) f(x_i)^T;
   eigenvalues lambda_1 >= lambda_2 >= ...; power law lambda_j ~ j^-alpha
   "for j in [min, max]"; alpha estimated "by fitting a linear model on the
   eigenspectrum in log-log scale". The original fixes no range; the survey
   (sec. 3.1) uses j in [10, floor(0.9 d)], used here (1-based j; the
   survey's App. B.1 ranges read as 0-based, so its window may be shifted
   by one index). Authors' code (kumarkrishna/fastssl): every alpha in their
   scripts comes from utils/powerlaw.py stringer_get_powerlaw, a WEIGHTED
   least-squares fit of log(lambda_j) on [-log j, 1] with weights 1/j, on
   get_eigenspectrum's PCA explained-variance ratios, over 0-based index
   ranges such as arange(3, 100), (5, 50), (11, 50). Their plain-OLS
   fit_powerlaw is reached only through robust_fit_powerlaw, which nothing
   calls (found by the independent review, 2026-09-23; our first label
   "as coded" for the OLS fit was wrong). Variants: 'centred' (OLS on the
   centred covariance eigenvalues over the survey's window: as published,
   survey), 'code_wls' (the authors' weighted estimator, same window, on
   eigenvalues from the SVD of X - mean: as the authors' code does it),
   'uncentred' (OLS on the uncentred second moment; ours). The OLS variants
   take eigenvalues from the covariance, which on the worst-conditioned
   checkpoints (lambda_1 / lambda_86 ~ 1e12) moves alpha by up to ~4e-7
   against the SVD route (review); negligible here.
RankMe  Garrido, Balestriero, Najman & LeCun, ICML 2023. Reused UNCHANGED from
   knee_rank_estimators_prepost.py: 'pub_uncentred' = train/src/utils/logging.py
   effective_rank on X (the published formula, singular values normalised to
   sum 1, float32, as the training log computes it); 'pub_centred' = the same
   on X - mean; 'sq_centred' = knee_s7_mech.py:93-99 (squared singular values
   of X - mean), via knee_rank_estimators_prepost.spectrum_s7. The survey
   uses the published formula on the backbone output, no centring stated.
NE-Sum  He & Ozay, ICML 2022, Definition 4.1: NESum = sum_i lambda_i /
   lambda_1 for the eigenvalues of the feature covariance, 0/0 := 0.
   Implemented as trace(C) / lambda_max(C). Variants: 'centred' (as
   published; survey sec. 3.1, "eigenvalues of the covariance matrix");
   'uncentred' (second moment; equals the stable rank ||X||_F^2 / ||X||_2^2
   of Tsitsulin et al. Def. 3.3 on raw X).
kappa (condition number)  Tsitsulin, Munkhoeva & Perozzi, 2023, Definition
   3.2: kappa_2(M) = ||M||_2 ||M^+||_2 = sigma_1 / sigma_n, M the embedding
   matrix. The survey adds "(excluding any zero singular values)".
   Numerically zero here means below tol = 10 * eps_float32 * rms(M) *
   (sqrt(N) + sqrt(d)), where M is the stored matrix; on these features that
   is 11-18 times the spectral norm of simulated float32 storage rounding
   (review, 2026-09-23). Where the smallest singular values decay
   continuously into the rounding level (no gap), kappa depends on this
   choice; the audit reports kappa at 1x, 10x and 100x. The LayerNorm
   pooling puts every feature vector in the subspace
   orthogonal to the ones-vector, so one singular value is zero up to
   rounding. numpy's default rank tolerance (sigma_1 * max(N, d) * eps)
   would also discard genuine singular values on these features, so it is
   not used. Recorded per checkpoint: number excluded, and the gap (smallest
   kept / largest excluded). Variants: 'uncentred' (the matrix as given: as
   published; survey); 'centred' (ours).

RELATIONAL
Self-Cluster  Tsitsulin et al. 2023, Definition 3.5 (reprinted in survey sec.
   3.2): with L2-normalised rows W, SelfCluster = (d Q - n (d + n - 1)) /
   ((d - 1)(n - 1) n), Q printed as ||W W^T||_F, with E[Q] = n + n(n - 1)/d
   (uniform on the sphere) and max Q = n^2 (collapse). Those two statements
   hold only for the SQUARED norm: E[sum_ij <w_i, w_j>^2] = n + n(n - 1)/d,
   and sum_ij <w_i, w_j>^2 = n^2 at collapse. With the unsquared norm the
   collapse value is -1/(d - 1), not 1. Implemented, computed through the
   d x d matrix W^T W as the survey does: 'sq_uncentred' (Q = ||W^T W||_F^2,
   the reading under which the paper's normalisation holds; primary),
   'lit_uncentred' (Q = ||W^T W||_F, exactly as printed), and 'sq_centred',
   'lit_centred' (rows centred before normalising; ours). No code from
   either paper was found.
DSE  Liao, Liu, Christensen, Tong, Huguet, Wolf, Nickel, Adelstein &
   Krishnaswamy, CISS 2024 (arXiv 2312.04823). Paper, Eqns 1, 2, 5, 6 and
   Definition 4.1: G(z1, z2) = exp(-||z1 - z2||^2 / sigma); anisotropic
   K = G(z1, z2) / (||G(z1, .)||_1^a ||G(z2, .)||_1^a) with a = 1/2; P = K
   row-normalised; DSE = -sum_i a_i log a_i, a_i = |lambda_i^t| / sum_j
   |lambda_j^t| over the eigenvalues of P. Official code
   (ChenLiu-1996/DiffusionSpectralEntropy, dse_dsmi/diffusion.py and dse.py,
   defaults sigma = 10, t = 1, max_N = 10000): G = exp(-D^2 / (2 sigma^2)) /
   (sigma sqrt(2 pi)); K = diag(1/sqrt(rowsum G)) G diag(1/sqrt(rowsum G));
   eigenvalues of K (numpy eigvalsh); |lambda|^t normalised; entropy in
   bits with float64 eps added to every p. The code's K has the spectrum of
   the a = 0 random-walk matrix D^-1 G, not of the paper's a = 1/2 P; its
   comment claiming the same eigenvalues holds only for a = 0. The survey
   (sec. 3.2 and App. A) reprints the code's construction, sigma = 10, t = 1,
   on the backbone output. The paper fixes no sigma for DSE. Implemented:
   'code' (the official construction; primary; survey), 'paper' (Eqns 1, 2,
   5, 6 with sigma = 10 placed in exp(-d^2 / sigma), t = 1, eigenvalues of P
   via its symmetric conjugate, entropy in bits; the paper's log base is not
   stated), 'paper_codebw' (the same construction with sigma = 2 * 10^2 =
   200, i.e. the code's kernel width, so that it differs from 'code' only by
   the a = 1/2 normalisation and row-normalisation), 'code_l2' and
   'code_l2_centred' (the code on L2-normalised rows, uncentred and centred
   first: the case the survey's Appendix A analyses). 'paper' is 20 times
   narrower in d^2 than 'code'; the review found 97-99% of the code-paper
   difference is that bandwidth, which 'paper_codebw' isolates. DSE on raw
   rows depends on pairwise distances only, so centring does not change it.
   N = 4000 < max_N, so no subsampling.

MANIFOLD
ID (TwoNN)  Facco, d'Errico, Rodriguez & Laio, Sci. Rep. 2017: mu_i =
   r2(i)/r1(i); F_emp by sorting mu; fit {(log mu_i, -log(1 - F_emp(mu_i)))}
   with a straight line through the origin, discarding the 10% of points with
   the highest mu; slope = d. Ansuini, Laio, Macke & Zoccolan, NeurIPS 2019
   (the survey's source), code IntrinsicDimDeep/IDNN/intrinsic_dimension.py
   estimate(): drops points with r1 = 0 or r1 = r2, F_emp = i/N, uses
   mu[:-2], fits the first floor(0.9 N) points through the origin. The
   survey (sec. 3.3) writes the maximum-likelihood form (P(mu | d) = d^N
   prod mu_i^-(d+1), so d = N / sum log mu_i) and averages 20 random 90%
   subsamples per batch. Implemented: 'fit' (Facco/Ansuini, primary) and
   'mle' (the survey's written form, on the same retained points), each
   averaged over 20 subsamples of 90% of the 4,000 events with fixed seed 0
   (the same subsample indices at every checkpoint). Distances are float64
   and computed without the dot-product shortcut. Second reference in the
   tests: scikit-dimension's TwoNN (source fetched, not installed), which
   keeps points with r1 = 0 or r1 = r2; on data without ties it is the same
   fit.

Usage (from analysis/):
  CUDA_VISIBLE_DEVICES=0 python knee_labelfree_metrics.py --test
  CUDA_VISIBLE_DEVICES=0 python knee_labelfree_metrics.py --test-real
  CUDA_VISIBLE_DEVICES=0 python knee_labelfree_metrics.py --part 0 --nparts 2
  python knee_labelfree_metrics.py --merge --nparts 2
The pass appends to <FEAT_DIR>/metrics_parts/knee_labelfree_metrics.part<k>.csv
(resumable by checkpoint); --merge writes results/knee_labelfree_metrics.csv and refuses to
overwrite it.
"""
import sys

sys.dont_write_bytecode = True      # leave analysis/__pycache__ untouched

import argparse  # noqa: E402
import csv  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
sys.path.insert(0, _HERE)
sys.path.insert(0, paths.TRAIN)

EPS32 = float(np.finfo(np.float32).eps)
KAPPA_TOL_MULT = 10.0
AREQ_LO, AREQ_HI_FRAC = 10, 0.9
DSE_SIGMA, DSE_T = 10.0, 1
ID_NSUB, ID_FRAC, ID_FIT_FRACTION, ID_SEED = 20, 0.9, 0.9, 0
FEAT_DIR = os.path.join(paths.WORK, 'labelfree_feats')  # denom: 58 symlinks + 3 new
OUT_CSV = os.path.join(paths.CSV, 'knee_labelfree_metrics.csv')
PART_DIR = os.path.join(FEAT_DIR, 'metrics_parts')     # intermediate, outside the repo
REF_DIR = os.environ.get('MONITOR_REF_DIR', os.path.join(paths.WORK, 'reference_code'))   # authors' reference code, for --test
COLS = ['run', 'epoch', 'metric', 'variant', 'value', 'diag1_name', 'diag1',
        'diag2_name', 'diag2']

# (metric, variant, as published, as the authors' code does it, survey)
VARIANTS = [
    ('alpha_req', 'centred', 'yes (OLS; range not given)', 'no (theirs is weighted)', 'yes, j in [10, 0.9d]'),
    ('alpha_req', 'code_wls', 'no', 'yes (stringer_get_powerlaw; survey window)', 'no'),
    ('alpha_req', 'uncentred', 'no', 'no', 'no'),
    ('rankme', 'pub_uncentred', 'yes', 'not checked', 'yes'),
    ('rankme', 'pub_centred', 'no', 'no', 'no'),
    ('rankme', 'sq_centred', 'no', 'no (knee_s7_mech)', 'no'),
    ('ne_sum', 'centred', 'yes', 'not checked', 'yes'),
    ('ne_sum', 'uncentred', 'no', 'not checked', 'no'),
    ('kappa', 'uncentred', 'yes', 'none found', 'yes'),
    ('kappa', 'centred', 'no', 'none found', 'no'),
    ('self_cluster', 'sq_uncentred', 'the reading its constants need', 'none found', 'no'),
    ('self_cluster', 'lit_uncentred', 'as printed', 'none found', 'as printed'),
    ('self_cluster', 'sq_centred', 'no', 'none found', 'no'),
    ('self_cluster', 'lit_centred', 'no', 'none found', 'no'),
    ('dse', 'code', 'no', 'yes', 'yes'),
    ('dse', 'paper', 'the form (sigma not given)', 'no', 'no'),
    ('dse', 'paper_codebw', 'the form, at the code kernel width', 'no', 'no'),
    ('dse', 'code_l2', 'no', 'code on L2 rows', 'App. A analysis case'),
    ('dse', 'code_l2_centred', 'no', 'no', 'no'),
    ('id_twonn', 'fit', 'yes (Facco)', 'yes (Ansuini)', 'cites Ansuini'),
    ('id_twonn', 'mle', 'no', 'no', 'as written'),
]


# =================================================================== spectral

def _f64(X):
    return np.asarray(X, dtype=np.float64)


def cov_eigs(X, centred):
    """Eigenvalues of (1/N) X^T X (X centred or not), descending."""
    X = _f64(X)
    if centred:
        X = X - X.mean(0)
    C = X.T @ X / X.shape[0]
    return np.linalg.eigvalsh(C)[::-1], C


def alpha_req(X, centred=True, lo=AREQ_LO, hi_frac=AREQ_HI_FRAC):
    """-slope of the OLS fit of log(lambda_j) on log(j), j in [lo, floor(hi_frac d)].
    Returns (alpha, fit R^2). NaN if an eigenvalue in the range is <= 0."""
    ev, _ = cov_eigs(X, centred)
    d = X.shape[1]
    hi = int(math.floor(hi_frac * d))
    j = np.arange(lo, hi + 1)
    lam = ev[j - 1]
    if np.any(lam <= 0):
        return float('nan'), float('nan')
    x, y = np.log(j), np.log(lam)
    A = np.stack([x, np.ones_like(x)], 1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    r2 = 1.0 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum())
    return float(-coef[0]), r2


def alpha_req_wls(X, lo=AREQ_LO, hi_frac=AREQ_HI_FRAC):
    """The alpha-ReQ authors' estimator (fastssl utils/powerlaw.py
    stringer_get_powerlaw, float64 here): weighted least squares of
    log(lambda_j) on [-log j, 1] with weights 1/j, j in [lo, floor(hi_frac d)]
    (1-based), eigenvalues of the centred covariance from the SVD of X - mean
    (their get_eigenspectrum: PCA explained-variance ratios; the scale does
    not change the slope). Returns alpha."""
    Xc = _f64(X)
    Xc = Xc - Xc.mean(0)
    s = np.linalg.svd(Xc, compute_uv=False)
    lam = s ** 2 / Xc.shape[0]
    hi = int(math.floor(hi_frac * X.shape[1]))
    j = np.arange(lo, hi + 1, dtype=np.float64)
    y = np.log(lam[lo - 1:hi])
    A = np.stack([-np.log(j), np.ones_like(j)], 1)
    w = (1.0 / j)[:, None]
    b = np.linalg.solve(A.T @ (A * w), (A * w).T @ y)
    return float(b[0])


def ne_sum(X, centred=True):
    """trace(C) / lambda_max(C); 0/0 := 0 (He & Ozay Def. 4.1)."""
    ev, C = cov_eigs(X, centred)
    lmax = float(ev[0])
    if lmax <= 0:
        return 0.0
    return float(np.trace(C)) / lmax


def kappa_tol(X, tol_mult=KAPPA_TOL_MULT):
    N, d = X.shape
    rms = math.sqrt(float((X ** 2).mean()))
    return tol_mult * EPS32 * rms * (math.sqrt(N) + math.sqrt(d))


def condition_number(X, centred=False, tol_mult=KAPPA_TOL_MULT):
    """sigma_1 / smallest non-zero sigma. Returns (kappa, n_excluded, gap, tol).
    The zero tolerance comes from the matrix AS STORED, for both variants:
    float32 rounding sits on the stored values, and centring does not remove
    it. (First pass, 2026-09-23: the centred variant took it from the centred
    matrix, whose rms is ~20x smaller on these features, and kept the
    rounding-level LayerNorm zero at 35 of 213 checkpoints. Stopped, fixed,
    retested before the pass reran.)"""
    X = _f64(X)
    tol = kappa_tol(X, tol_mult)
    if centred:
        X = X - X.mean(0)
    s = np.linalg.svd(X, compute_uv=False)
    keep = s > tol
    if not keep.any():
        return float('nan'), int(s.size), float('nan'), tol
    n_excl = int((~keep).sum())
    gap = float(s[keep].min() / s[~keep].max()) if n_excl else float('inf')
    return float(s[0] / s[keep].min()), n_excl, gap, tol


def rankme_pub(X, centred):
    """The published RankMe exactly as logged (src/utils/logging.py, imported)."""
    import torch
    from src.utils.logging import effective_rank
    X = torch.from_numpy(_f64(X))            # knee_rank_estimators_prepost's path: float64 torch,
    if centred:                              # centred in torch, effective_rank casts
        X = X - X.mean(0)                    # to float32 itself
    return float(effective_rank(X))


def rankme_sq_centred(X):
    """knee_s7_mech.py:93-99 via knee_rank_estimators_prepost.spectrum_s7."""
    import torch
    import knee_rank_estimators_prepost as KR
    _, rm = KR.spectrum_s7(torch.from_numpy(_f64(X)))
    return float(rm)


# ================================================================= relational

def _l2rows(X, centred):
    X = _f64(X)
    if centred:
        X = X - X.mean(0)
    nrm = np.linalg.norm(X, axis=1)
    if np.any(nrm == 0):
        return None
    return X / nrm[:, None]


def self_cluster(X, centred=False, squared=True):
    W = _l2rows(X, centred)
    if W is None:
        return float('nan')
    N, d = W.shape
    M = W.T @ W
    Q = float((M * M).sum())                 # ||W^T W||_F^2 = ||W W^T||_F^2
    if not squared:
        Q = math.sqrt(Q)                     # ||W W^T||_F as printed
    return (d * Q - N * (d + N - 1)) / ((d - 1) * (N - 1) * N)


def _sqdist(X, device):
    """Exact float64 squared Euclidean distances (no dot-product shortcut)."""
    import torch
    T = torch.as_tensor(_f64(X), device=device)
    D = torch.cdist(T, T, compute_mode='donot_use_mm_for_euclid_dist')
    return D * D


def _entropy_bits(ev, t):
    import torch
    ev = ev.abs() ** t
    p = ev / ev.sum()
    p = p + float(np.finfo(float).eps)       # the official code's eps
    return float(-(p * torch.log2(p)).sum())


def dse_code(X, sigma=DSE_SIGMA, t=DSE_T, device='cpu', D2=None):
    import torch
    if D2 is None:
        D2 = _sqdist(X, device)
    G = (1.0 / (sigma * math.sqrt(2 * math.pi))) * torch.exp(-D2 / (2 * sigma ** 2))
    r = G.sum(1)
    K = G / torch.sqrt(r[:, None] * r[None, :])
    ev = torch.linalg.eigvalsh(K)
    return _entropy_bits(ev, t)


def dse_paper(X, sigma=DSE_SIGMA, t=DSE_T, a=0.5, device='cpu', D2=None):
    import torch
    if D2 is None:
        D2 = _sqdist(X, device)
    G = torch.exp(-D2 / sigma)                           # Eqn 1
    q = G.sum(1)
    K = G / (q[:, None] ** a * q[None, :] ** a)          # anisotropic kernel, Eqn 5
    r = K.sum(1)                                         # P = diag(r)^-1 K (Eqn 2)
    S = K / torch.sqrt(r[:, None] * r[None, :])          # same eigenvalues as P
    ev = torch.linalg.eigvalsh(S)
    return _entropy_bits(ev, t)


# =================================================================== manifold

def _two_nn_from_D2(D2):
    """r1, r2 for each row of a squared-distance matrix, self excluded.
    Sets the diagonal of D2 to inf in place."""
    import torch
    D2.fill_diagonal_(float('inf'))
    v, _ = torch.topk(D2, 2, dim=1, largest=False)
    r = torch.sqrt(v).cpu().numpy()
    return r[:, 0], r[:, 1]


def _two_nn_dists(Xs, device):
    """r1, r2 for each point of Xs (float64), self excluded."""
    return _two_nn_from_D2(_sqdist(Xs, device))


def twonn_from_r(k1, k2, fit_fraction=ID_FIT_FRACTION):
    """Ansuini's estimate() steps on (r1, r2); returns (fit, mle, n_good)."""
    zeros = np.where(k1 == 0)[0]
    degen = np.where(k1 == k2)[0]
    good = np.setdiff1d(np.arange(k1.shape[0]), zeros)
    good = np.setdiff1d(good, degen)
    k1, k2 = k1[good], k2[good]
    N = good.shape[0]
    npoints = int(np.floor(N * fit_fraction))
    mu = np.sort(k2 / k1, kind='quicksort')
    Femp = np.arange(1, N + 1, dtype=np.float64) / N
    x = np.log(mu[:-2])
    y = -np.log(1 - Femp[:-2])
    xf, yf = x[:npoints], y[:npoints]
    fit = float((xf @ yf) / (xf @ xf))                 # OLS through the origin
    mle = float(N / np.log(mu).sum())                  # survey's written MLE
    return fit, mle, N


def twonn(X, n_sub=ID_NSUB, frac=ID_FRAC, seed=ID_SEED, device='cpu', D2=None):
    """Each subsample's distances are sliced from one full matrix. The slices
    are bit-identical to recomputing them per subsample (checked on 180
    subsamples of 9 real checkpoints, 2026-09-23), so this is only faster."""
    import torch
    X = _f64(X)
    N = X.shape[0]
    m = int(np.floor(frac * N))
    rng = np.random.default_rng(seed)
    if D2 is None:
        D2 = _sqdist(X, device)
    fits, mles, dropped = [], [], []
    for _ in range(n_sub):
        idx = rng.choice(N, size=m, replace=False)
        ti = torch.as_tensor(idx, device=D2.device)
        k1, k2 = _two_nn_from_D2(D2[ti][:, ti])          # advanced indexing copies
        f, mle, n_good = twonn_from_r(k1, k2)
        fits.append(f)
        mles.append(mle)
        dropped.append(m - n_good)
    return (float(np.mean(fits)), float(np.std(fits)), float(np.mean(mles)),
            float(np.std(mles)), float(np.mean(dropped)))


# ============================================================ all metrics

def all_metrics(X, device='cpu'):
    """[(metric, variant, value, d1name, d1, d2name, d2)] for one checkpoint."""
    out = []
    for v, c in (('centred', True), ('uncentred', False)):
        a, r2 = alpha_req(X, centred=c)
        out.append(('alpha_req', v, a, 'fit_r2', r2, '', ''))
    out.append(('alpha_req', 'code_wls', alpha_req_wls(X), '', '', '', ''))
    out.append(('rankme', 'pub_uncentred', rankme_pub(X, False), '', '', '', ''))
    out.append(('rankme', 'pub_centred', rankme_pub(X, True), '', '', '', ''))
    out.append(('rankme', 'sq_centred', rankme_sq_centred(X), '', '', '', ''))
    for v, c in (('centred', True), ('uncentred', False)):
        out.append(('ne_sum', v, ne_sum(X, centred=c), '', '', '', ''))
    for v, c in (('uncentred', False), ('centred', True)):
        k, n_ex, gap, _ = condition_number(X, centred=c)
        out.append(('kappa', v, k, 'n_excluded', n_ex, 'gap', gap))
    for v, c, s in (('sq_uncentred', False, True), ('lit_uncentred', False, False),
                    ('sq_centred', True, True), ('lit_centred', True, False)):
        out.append(('self_cluster', v, self_cluster(X, c, s), '', '', '', ''))
    D2 = _sqdist(X, device)                  # shared by DSE code / paper and TwoNN
    out.append(('dse', 'code', dse_code(X, device=device, D2=D2), '', '', '', ''))
    out.append(('dse', 'paper', dse_paper(X, device=device, D2=D2), '', '', '', ''))
    out.append(('dse', 'paper_codebw', dse_paper(X, sigma=2 * DSE_SIGMA ** 2, device=device, D2=D2),
                '', '', '', ''))
    out.append(('dse', 'code_l2', dse_code(_l2rows(X, False), device=device), '', '', '', ''))
    out.append(('dse', 'code_l2_centred', dse_code(_l2rows(X, True), device=device), '', '', '', ''))
    f, fs, mle, ms, nd = twonn(X, device=device, D2=D2)
    out.append(('id_twonn', 'fit', f, 'sd_over_subsamples', fs, 'mean_dropped', nd))
    out.append(('id_twonn', 'mle', mle, 'sd_over_subsamples', ms, 'mean_dropped', nd))
    return out


# ================================================================= tests

class T:
    rows = []

    @classmethod
    def check(cls, tier, name, value, expect, tol, rel=False, note=''):
        """tier 'A' = implementation (closed form or reference code);
        'B' = estimator behaviour on data with a known answer (stated tolerance);
        'R' = reported only, no pass/fail."""
        if isinstance(expect, bool):
            err, ok = '', bool(value) == expect
        else:
            err = abs(value - expect) / (abs(expect) if rel and expect != 0 else 1.0)
            ok = bool(err <= tol)
        status = 'REPORT' if tier == 'R' else ('PASS' if ok else 'FAIL')
        cls.rows.append(dict(tier=tier, name=name, value=value, expect=expect,
                             tol=tol, kind='rel' if rel else 'abs', err=err, status=status,
                             note=note))
        e = f'{err:.2e}' if err != '' else ''
        print(f"  {tier} [{status}] {name}: got {value!r} expect {expect!r} |err| {e} "
              f"tol {tol} ({'rel' if rel else 'abs'}) {note}", flush=True)
        return ok

    @classmethod
    def summary(cls, path=None):
        for tier in ('A', 'B', 'R'):
            rs = [r for r in cls.rows if r['tier'] == tier]
            nf = sum(r['status'] == 'FAIL' for r in rs)
            print(f'tier {tier}: {len(rs)} checks, {nf} failed')
        if path:
            with open(path, 'w', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=list(cls.rows[0]), lineterminator='\n')
                w.writeheader()
                w.writerows(cls.rows)
            print('wrote', path)


def exact_matrix(N, s, seed=0, mean=None):
    """X = U diag(s) V^T with U orthonormal AND column-centred, V orthogonal:
    the centred covariance has eigenvalues exactly s^2 / N (sorted)."""
    rng = np.random.default_rng(seed)
    d = len(s)
    Gm = rng.standard_normal((N, d))
    Gm -= Gm.mean(0)
    U, _ = np.linalg.qr(Gm)
    V, _ = np.linalg.qr(rng.standard_normal((d, d)))
    X = (U * np.asarray(s, dtype=np.float64)[None, :]) @ V.T
    if mean is not None:
        X = X + np.asarray(mean)[None, :]
    return X


def _load_refs():
    sys.path.insert(0, os.path.join(REF_DIR, 'dse_ref'))
    sys.path.insert(0, os.path.join(REF_DIR, 'ansuini_ref'))
    from dse_dsmi.dse import diffusion_spectral_entropy as dse_official
    import intrinsic_dimension as ansuini
    import importlib.util
    spec = importlib.util.spec_from_file_location('fastssl_powerlaw',
                                                  os.path.join(REF_DIR, 'fastssl_powerlaw.py'))
    fpl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fpl)
    return dse_official, ansuini, fpl


def _load_skdim():
    """scikit-dimension's TwoNN (j-bac/scikit-dimension, skdim/id/_TwoNN.py and
    skdim/_commonfuncs.py, source files fetched, not installed). Its rule keeps
    points with r1 = 0 or r1 = r2 (Ansuini's drops them); on data without
    ties the two fits are the same computation."""
    sys.path.insert(0, os.path.join(REF_DIR, 'skdim_ref'))
    from skdim.id._TwoNN import TwoNN
    return TwoNN


def fastssl_alpha_wls(X, fpl):
    """The authors' estimator exactly as their scripts call it:
    stringer_get_powerlaw(get_eigenspectrum(X), trange), trange 0-based,
    here arange(lo - 1, floor(0.9 d)) so that j = trange + 1 spans the
    survey's window. (It adds 1 to trange in place, hence a fresh array.)"""
    ev = fpl.get_eigenspectrum(np.asarray(X, dtype=np.float64))
    trange = np.arange(AREQ_LO - 1, int(math.floor(AREQ_HI_FRAC * X.shape[1])))
    return float(fpl.stringer_get_powerlaw(ev, trange)[0])


def fastssl_alpha(X, fpl):
    ev = fpl.get_eigenspectrum(np.asarray(X, dtype=np.float64))
    d = X.shape[1]
    return float(fpl.fit_powerlaw(ev, AREQ_LO, int(math.floor(AREQ_HI_FRAC * d)))[0])


def dse_paper_numpy(X, sigma=DSE_SIGMA, t=DSE_T, a=0.5):
    """Independent path: Eqns 1, 2, 5, 6 with numpy, eigenvalues of the
    non-symmetric P itself."""
    from scipy.spatial.distance import pdist, squareform
    D2 = squareform(pdist(_f64(X), 'sqeuclidean'))
    G = np.exp(-D2 / sigma)
    q = G.sum(1)
    K = G / np.outer(q ** a, q ** a)
    P = K / K.sum(1)[:, None]
    ev = np.abs(np.linalg.eigvals(P)) ** t
    p = ev / ev.sum() + np.finfo(float).eps
    return float(-(p * np.log2(p)).sum())


def run_tests(device='cpu'):
    rng = np.random.default_rng(7)
    N = 4000
    print('=== tier A, spectral: exact-spectrum constructions (N = 4000) ===')
    for d, k in ((96, 10), (96, 40), (192, 25)):
        s = np.zeros(d)
        s[:k] = 3.0
        X = exact_matrix(N, s, seed=k)
        T.check('A', f'rankme pub_centred, rank-{k} equal spectrum, d={d}', rankme_pub(X, True), k, 1e-4,
                rel=True, note='(effective_rank runs in float32; zero singular values come out near eps32*s1)')
        T.check('A', f'rankme sq_centred, rank-{k} equal spectrum, d={d}', rankme_sq_centred(X), k, 1e-9,
                rel=True)
        T.check('A', f'ne_sum centred, rank-{k} equal spectrum, d={d}', ne_sum(X, True), k, 1e-9, rel=True)
    s = np.sort(rng.uniform(0.05, 5.0, 96))[::-1]
    X = exact_matrix(N, s, seed=3)
    p1 = s / s.sum()
    p2 = s ** 2 / (s ** 2).sum()
    T.check('A', 'rankme pub_centred, random spectrum = exp H(s/sum s)', rankme_pub(X, True),
            float(np.exp(-(p1 * np.log(p1)).sum())), 1e-5, rel=True, note='(float32)')
    T.check('A', 'rankme sq_centred, random spectrum = exp H(s^2/sum s^2)', rankme_sq_centred(X),
            float(np.exp(-(p2 * np.log(p2)).sum())), 1e-9, rel=True)
    T.check('A', 'ne_sum centred, random spectrum = sum s^2 / max s^2', ne_sum(X, True),
            float((s ** 2).sum() / (s ** 2).max()), 1e-9, rel=True)
    T.check('A', 'ne_sum, zero matrix (0/0 := 0)', ne_sum(np.zeros((100, 96)), True), 0.0, 0.0)
    s = np.linspace(10.0, 0.1, 96)
    X = exact_matrix(N, s, seed=4)
    k, nex, gap, _ = condition_number(X, centred=True)
    T.check('A', 'kappa centred, singular values linspace(10, 0.1): = 100', k, 100.0, 1e-9, rel=True)
    T.check('A', 'kappa centred, same: nothing excluded', nex, 0, 0)
    s0 = s.copy()
    s0[-1] = 0.0
    X0 = exact_matrix(N, s0, seed=5)
    k, nex, gap, _ = condition_number(X0, centred=True)
    T.check('A', 'kappa, one exact zero singular value: 1 excluded', nex, 1, 0)
    T.check('A', 'kappa, one exact zero: = s_1 / s_{d-1}', k, 10.0 / s0[-2], 1e-9, rel=True)
    X32 = X0.astype(np.float32).astype(np.float64)
    k, nex, gap, _ = condition_number(X32, centred=True)
    T.check('A', 'kappa, same matrix rounded to float32: 1 excluded', nex, 1, 0, note=f'(gap {gap:.1e})')
    T.check('A', 'kappa, same matrix rounded to float32: value', k, 10.0 / s0[-2], 1e-5, rel=True)
    # LayerNorm-like structural zero, as in the pooled features
    import torch
    g = torch.Generator().manual_seed(11)
    Wmix = torch.randn(96, 96, generator=g, dtype=torch.float64) / 10
    tok = torch.randn(N, 16, 96, generator=g, dtype=torch.float64) @ Wmix
    tok = tok + 0.5 * torch.randn(N, 1, 96, generator=g, dtype=torch.float64)
    pooled = torch.nn.functional.layer_norm(tok.float(), (96,)).mean(1).double().numpy()
    sv = np.linalg.svd(pooled, compute_uv=False)
    k, nex, gap, tol = condition_number(pooled, centred=False)
    T.check('A', 'kappa, LayerNorm-then-mean-pool features (float32): exactly 1 excluded', nex, 1, 0,
            note=f'(smallest two singular values {sv[-1]:.1e}, {sv[-2]:.1e}; tol {tol:.1e}; gap {gap:.1e})')
    T.check('A', 'kappa, LayerNorm features: = s_1 / s_{d-1}', k, float(sv[0] / sv[-2]), 1e-12, rel=True)
    npdef = sv[0] * max(pooled.shape) * EPS32
    T.check('R', 'kappa, LayerNorm features: count numpy default tol (s1*max(N,d)*eps32) would exclude',
            int((sv <= npdef).sum()), 1, 0, note=f'(numpy default tol {npdef:.1e})')
    # The same features plus a large common offset orthogonal to the ones-vector (the
    # LayerNorm zero survives), stored in float32: the stored rms is ~100x the centred rms
    # (real LeJEPA features: ~20x). Added after the first pass exposed the centred-variant
    # tolerance bug; the old rule (tolerance from the centred matrix) is reported alongside.
    Pc = pooled - pooled.mean(0)
    off = np.random.default_rng(12).standard_normal(96)
    off -= off.mean()
    off *= 100.0 * math.sqrt(float((Pc ** 2).mean())) / math.sqrt(float((off ** 2).mean()))
    big = (pooled + off[None, :]).astype(np.float32).astype(np.float64)
    bc = big - big.mean(0)
    svc = np.linalg.svd(bc, compute_uv=False)
    k, nex, gap, tol = condition_number(big, centred=True)
    T.check('A', 'kappa centred, LayerNorm features + large offset (float32): exactly 1 excluded', nex, 1, 0,
            note=f'(smallest two singular values {svc[-1]:.1e}, {svc[-2]:.1e}; tol {tol:.1e}; gap {gap:.1e})')
    T.check('A', 'kappa centred, LayerNorm features + large offset: = s_1 / s_{d-1} of X - mean', k,
            float(svc[0] / svc[-2]), 1e-12, rel=True)
    T.check('A', 'kappa uncentred, LayerNorm features + large offset: exactly 1 excluded',
            condition_number(big, centred=False)[1], 1, 0)
    T.check('R', 'kappa centred, same matrix, OLD rule (tolerance from the centred matrix): count excluded',
            int((svc <= kappa_tol(bc)).sum()), 1, 0, note=f'(old tolerance {kappa_tol(bc):.1e})')
    for kk in (5, 20):
        Z = rng.standard_normal((N, kk))
        A, _ = np.linalg.qr(rng.standard_normal((96, kk)))
        X = Z @ A.T
        k, nex, gap, _ = condition_number(X, centred=False)
        sv = np.linalg.svd(X, compute_uv=False)
        T.check('A', f'kappa, rank-{kk} Gaussian: {96 - kk} zero singular values excluded', nex, 96 - kk, 0)
        T.check('A', f'kappa, rank-{kk} Gaussian: = s_1 / s_{kk}', k, float(sv[0] / sv[kk - 1]), 1e-9,
                rel=True)
        T.check('B', f'rankme pub_uncentred, rank-{kk} Gaussian: close to {kk}', rankme_pub(X, False), kk,
                0.01, rel=True, note='(sampling spread of the k singular values)')
    s = np.sort(rng.uniform(0.1, 3.0, 96))[::-1]
    m = rng.standard_normal(96) * 0.5
    X = exact_matrix(N, s, seed=6, mean=m)
    sv = np.linalg.svd(X, compute_uv=False)
    T.check('A', 'ne_sum uncentred = stable rank ||X||_F^2 / ||X||_2^2 (via SVD)', ne_sum(X, False),
            float((sv ** 2).sum() / sv[0] ** 2), 1e-9, rel=True)
    T.check('A', 'ne_sum centred = stable rank of X - mean (known s)', ne_sum(X, True),
            float((s ** 2).sum() / s[0] ** 2), 1e-9, rel=True)
    k, _, _, _ = condition_number(X, centred=False)
    T.check('A', 'kappa uncentred = s_1/s_d of raw X', k, float(sv[0] / sv[-1]), 1e-9, rel=True)
    k, _, _, _ = condition_number(X, centred=True)
    T.check('A', 'kappa centred = s_1/s_d of X - mean (known s)', k, float(s[0] / s[-1]), 1e-9, rel=True)
    p = sv / sv.sum()
    T.check('A', 'rankme pub_uncentred = exp H(sv/sum sv) of raw X', rankme_pub(X, False),
            float(np.exp(-(p * np.log(p)).sum())), 1e-5, rel=True, note='(float32)')

    print('=== tier A, alpha-ReQ: exact power-law spectra ===')
    for d in (64, 96, 192):
        for a in (0.5, 1.0, 1.5, 2.0, 3.0):
            j = np.arange(1, d + 1)
            X = exact_matrix(N, np.sqrt(N * j ** (-a)), seed=int(10 * a) + d)
            got, r2 = alpha_req(X, centred=True)
            T.check('A', f'alpha_req centred, exact lambda_j = j^-{a}, d={d}', got, a, 1e-8,
                    note=f'(fit R2 {r2:.12f})')
    j = np.arange(1, 97)
    lam = np.where(j < 50, j ** -1.0, 49 ** -1.0 * (j / 49.0) ** -3.0)
    X = exact_matrix(N, np.sqrt(N * lam), seed=77)
    xx, yy = np.log(np.arange(10, 87)), np.log(lam[9:86])
    ref = -np.polyfit(xx, yy, 1)[0]
    T.check('A', 'alpha_req, broken power law (1 then 3 at j=49): = OLS slope over j in [10, 86]',
            alpha_req(X)[0], float(ref), 1e-8, note='(checks the fit range endpoints)')

    print('=== tier B, alpha-ReQ: sampled Gaussian data with power-law covariance, N=4000 ===')
    for d in (96, 192):
        for a in (0.5, 1.0, 2.0):
            lam = np.arange(1, d + 1) ** (-a)
            Vq, _ = np.linalg.qr(rng.standard_normal((d, d)))
            X = (rng.standard_normal((N, d)) * np.sqrt(lam)[None, :]) @ Vq.T
            got, r2 = alpha_req(X, centred=True)
            T.check('B', f'alpha_req sampled, lambda_j = j^-{a}, d={d}', got, a, 0.05,
                    note=f'(bias {got - a:+.4f}; fit R2 {r2:.4f})')

    print('=== tier A, Self-Cluster ===')
    d, n = 96, 1000
    X = np.tile(rng.standard_normal((1, d)), (n, 1))
    T.check('A', 'self_cluster sq, collapse (all rows identical) = 1', self_cluster(X), 1.0, 1e-12)
    T.check('A', 'self_cluster literal-as-printed, collapse = -1/(d-1), not 1', self_cluster(X, squared=False),
            -1.0 / (d - 1), 1e-12)
    T.check('A', 'self_cluster sq, orthonormal rows (n = d) = -1/(d-1)', self_cluster(np.eye(d)),
            -1.0 / (d - 1), 1e-12)
    for kc in (2, 4, 8):
        X = np.repeat(np.eye(d)[:kc] * 5.0, n // kc, axis=0)
        NN = X.shape[0]
        Q = NN ** 2 / kc
        T.check('A', f'self_cluster sq, {kc} orthogonal tight clusters (exact)', self_cluster(X),
                (d * Q - NN * (d + NN - 1)) / ((d - 1) * (NN - 1) * NN), 1e-12)
    X = rng.standard_normal((2000, 96)) + 0.3
    for c in (False, True):
        W = _l2rows(X, c)
        G = W @ W.T                                   # the N x N route
        Qn = float((G * G).sum())
        ref = (96 * Qn - 2000 * (96 + 2000 - 1)) / (95 * 1999 * 2000)
        T.check('A', f'self_cluster sq {"centred" if c else "uncentred"}: d x d route = N x N route',
                self_cluster(X, c), ref, 1e-12)
        ref_l = (96 * math.sqrt(Qn) - 2000 * (96 + 2000 - 1)) / (95 * 1999 * 2000)
        T.check('A', f'self_cluster lit {"centred" if c else "uncentred"}: d x d route = N x N route',
                self_cluster(X, c, squared=False), ref_l, 1e-12)
    print('=== tier B, Self-Cluster ===')
    vals = [self_cluster(np.random.default_rng(100 + sd).standard_normal((4000, 96))) for sd in range(5)]
    T.check('B', 'self_cluster sq, uniform on the sphere (5 seeds, mean) ~ 0', float(np.mean(vals)), 0.0,
            1e-3, note=f'(values {", ".join("%.1e" % v for v in vals)})')
    Z = np.random.default_rng(200).standard_normal((4000, 96)) + 3.0
    T.check('B', 'self_cluster sq_centred, shifted Gaussian ~ 0', self_cluster(Z, centred=True), 0.0, 1e-3)
    T.check('B', 'self_cluster sq_uncentred, same shifted set > 0.5', self_cluster(Z) > 0.5, True, 0)

    print('=== tier A, DSE: limiting cases (code and paper constructions) ===')
    X = np.tile(rng.standard_normal((1, 96)), (400, 1))
    T.check('A', 'dse code, 400 identical points = 0', dse_code(X, device=device), 0.0, 1e-9)
    T.check('A', 'dse paper, 400 identical points = 0', dse_paper(X, device=device), 0.0, 1e-9)
    Xf = np.eye(96)[:60] * 1000.0
    T.check('A', 'dse code, 60 far-apart points = log2 60', dse_code(Xf, device=device), math.log2(60), 1e-9)
    T.check('A', 'dse paper, 60 far-apart points = log2 60', dse_paper(Xf, device=device), math.log2(60), 1e-9)
    for kc in (1, 2, 4, 8):
        X = np.repeat(np.eye(96)[:kc] * 1000.0, 400 // kc, axis=0)
        X = X + 1e-5 * rng.standard_normal(X.shape)
        T.check('A', f'dse code, {kc} far tight clusters = log2 {kc}', dse_code(X, device=device),
                math.log2(kc), 1e-6)
        T.check('A', f'dse paper, {kc} far tight clusters = log2 {kc}', dse_paper(X, device=device),
                math.log2(kc), 1e-6)
    for name, Xs in (('gaussian*0.3 300x96', rng.standard_normal((300, 96)) * 0.3),
                     ('gaussian 300x96', rng.standard_normal((300, 96))),
                     ('3 clusters 300x96', np.repeat(np.eye(96)[:3] * 3, 100, 0) + rng.standard_normal((300, 96)))):
        T.check('A', f'dse paper vs independent numpy path (eigenvalues of P itself), {name}',
                dse_paper(Xs, device=device), dse_paper_numpy(Xs), 1e-9)

    print('=== tier B, ID: estimator steps on exact Pareto ratios ===')
    for dd in (2, 5, 10, 20):
        mu = np.random.default_rng(dd).pareto(dd, size=3600) + 1.0      # f(mu) = d mu^-(d+1)
        f, mle, ng = twonn_from_r(np.ones(3600), mu)
        T.check('B', f'id fit, exact Pareto(d={dd}) ratios, N=3600', f, dd, 0.05, rel=True,
                note=f'(bias {100 * (f - dd) / dd:+.2f}%)')
        T.check('B', f'id mle, exact Pareto(d={dd}) ratios, N=3600', mle, dd, 0.05, rel=True,
                note=f'(bias {100 * (mle - dd) / dd:+.2f}%; sampling sd ~{100 / math.sqrt(3600):.1f}%)')
    Xs = rng.standard_normal((3000, 5))
    Xd = np.concatenate([Xs, Xs[:50]])
    k1, k2 = _two_nn_dists(Xd, device)
    fd, _, ng = twonn_from_r(k1, k2)
    # First run (2026-09-23) expected 3000 - 50 and failed with 2892: that count
    # missed the reference's second rule. A point whose nearest neighbour is
    # duplicated has r1 = r2 exactly and is dropped too. Expected count now
    # from an independent path (the reference's own: sort each full distance
    # row, self included).
    from scipy.spatial.distance import pdist, squareform
    Y = np.sort(squareform(pdist(Xd)), axis=1)
    n_zero = int((Y[:, 1] == 0).sum())
    n_deg = int(((Y[:, 1] == Y[:, 2]) & (Y[:, 1] != 0)).sum())
    T.check('A', f'id: points kept with 50 duplicated rows (independent count: {n_zero} with r1 = 0, '
                 f'{n_deg} more with r1 = r2)', ng, len(Xd) - n_zero - n_deg, 0)
    ansuini = _load_refs()[1]
    _, _, ref_slope, _, _ = ansuini.estimate(squareform(pdist(Xd)), fraction=0.9)
    T.check('A', 'id fit vs Ansuini estimate() on the data with duplicates', fd, float(ref_slope), 1e-9,
            rel=True)

    print('=== tier B / R, ID (TwoNN, 20 subsamples of 90%): N=4000 on d-manifolds in R^96 ===')
    tol_by_d = {2: 0.05, 5: 0.05, 10: 0.10}
    for dd in (2, 5, 10, 20):
        R, _ = np.linalg.qr(rng.standard_normal((96, 96)))
        Z = rng.standard_normal((4000, dd + 1))
        Z /= np.linalg.norm(Z, axis=1)[:, None]
        Xa = Z @ R[:, :dd + 1].T                                  # uniform on S^d
        Xb = rng.uniform(size=(4000, dd)) @ R[:, :dd].T           # uniform in the d-cube
        Wf = rng.standard_normal((dd, 48)) * 2.0
        U = rng.uniform(size=(4000, dd))
        Xc = np.concatenate([np.sin(U @ Wf), np.cos(U @ Wf)], 1)  # curved image of the cube
        for name, Xm, tier in (('sphere S^d', Xa, 'B' if dd in tol_by_d else 'R'),
                               ('cube', Xb, 'R'), ('Fourier-curved cube', Xc, 'R')):
            f, fs, mle, ms, _ = twonn(Xm, device=device)
            T.check(tier, f'id fit, {name}, d={dd}', f, dd, tol_by_d.get(dd, 0.0), rel=True,
                    note=f'(bias {100 * (f - dd) / dd:+.1f}%, sd {fs:.3f}; mle {mle:.3f}, '
                         f'bias {100 * (mle - dd) / dd:+.1f}%, sd {ms:.3f})')

    print('=== tier A, reference implementations on synthetic data ===')
    dse_official, ansuini, fpl = _load_refs()
    from scipy.spatial.distance import pdist, squareform
    for name, Xr in (('gaussian 500x96', rng.standard_normal((500, 96))),
                     ('gaussian*5 1500x96', 5 * rng.standard_normal((1500, 96))),
                     ('4 clusters 800x96', np.repeat(np.eye(96)[:4] * 20, 200, 0) + rng.standard_normal((800, 96)))):
        T.check('A', f'dse code vs official diffusion_spectral_entropy, {name}', dse_code(Xr, device=device),
                float(dse_official(Xr)), 1e-9)
        Wl = Xr / np.linalg.norm(Xr, axis=1)[:, None]
        T.check('A', f'dse code_l2 vs official on L2-normalised rows, {name}', dse_code(Wl, device=device),
                float(dse_official(Wl)), 1e-9)
    Z = rng.standard_normal((3000, 6))
    Z /= np.linalg.norm(Z, axis=1)[:, None]
    R, _ = np.linalg.qr(rng.standard_normal((96, 96)))
    TwoNN = _load_skdim()
    for name, Xr in (('sphere S^5 in R^96', Z @ R[:, :6].T), ('gaussian 3000x96', rng.standard_normal((3000, 96))),
                     ('gaussian 3000x5', rng.standard_normal((3000, 5)))):
        _, _, ref_slope, _, _ = ansuini.estimate(squareform(pdist(Xr)), fraction=0.9)
        k1, k2 = _two_nn_dists(Xr, device)
        mine, _, _ = twonn_from_r(k1, k2)
        T.check('A', f'id fit vs Ansuini estimate(), {name}', mine, float(ref_slope), 1e-9, rel=True)
        T.check('A', f'id fit vs scikit-dimension TwoNN, {name}', mine, float(TwoNN().fit(Xr).dimension_), 1e-9,
                rel=True, note='(skdim: pairwise_distances_chunked for d > 25, NearestNeighbors below)')
    for d, a in ((96, 1.0), (96, 2.0), (192, 1.5)):
        j = np.arange(1, d + 1)
        X = exact_matrix(N, np.sqrt(N * j ** (-a)) * (1 + 0.05 * rng.standard_normal(d)) ** 2, seed=d,
                         mean=rng.standard_normal(d))
        T.check('A', f'alpha_req vs fastssl get_eigenspectrum + fit_powerlaw, d={d}, a~{a}', alpha_req(X)[0],
                fastssl_alpha(X, fpl), 1e-8)
        T.check('A', f'alpha_req code_wls vs the authors\' stringer_get_powerlaw, d={d}, a~{a}', alpha_req_wls(X),
                fastssl_alpha_wls(X, fpl), 1e-6, note='(their weights are float32)')

    print('=== tier A, variants added after the independent review ===')
    for d in (96, 192):
        for a in (0.5, 1.0, 2.0):
            j = np.arange(1, d + 1)
            X = exact_matrix(N, np.sqrt(N * j ** (-a)), seed=int(10 * a) + d)
            T.check('A', f'alpha_req code_wls, exact lambda_j = j^-{a}, d={d}', alpha_req_wls(X), a, 1e-9)
    ang = 2 * np.pi * np.arange(360) / 360
    Xcirc = np.zeros((360, 96))
    Xcirc[:, 0], Xcirc[:, 1] = 10 * np.cos(ang), 10 * np.sin(ang)
    Xcirc = Xcirc @ np.linalg.qr(rng.standard_normal((96, 96)))[0].T
    T.check('A', 'dse paper_codebw = code on equally spaced points on a circle (uniform density: the '
                 'normalisations coincide)', dse_paper(Xcirc, sigma=2 * DSE_SIGMA ** 2, device=device),
            dse_code(Xcirc, device=device), 1e-9)
    for name, Xs in (('gaussian*0.3 300x96', rng.standard_normal((300, 96)) * 0.3),
                     ('gaussian*3 300x96', rng.standard_normal((300, 96)) * 3)):
        T.check('A', f'dse paper_codebw vs independent numpy path (sigma = 200), {name}',
                dse_paper(Xs, sigma=2 * DSE_SIGMA ** 2, device=device),
                dse_paper_numpy(Xs, sigma=2 * DSE_SIGMA ** 2), 1e-9)
    Xe = exact_matrix(4000, np.sort(rng.uniform(0.1, 3.0, 96))[::-1], seed=8, mean=rng.standard_normal(96))
    full = {(r[0], r[1]): r for r in all_metrics(Xe, device=device)}
    same = all(full[(r[0], r[1])] == r for r in extra_metrics(Xe, device=device))
    T.check('A', 'extra pass rows = the same variants from all_metrics (bit-identical)', same, True, 0)
    T.summary(os.path.join(REF_DIR, '..', 'lf_tests_synthetic.csv'))


# ============================================================ real-data tests

def load_feats(run):
    import torch
    return torch.load(os.path.join(FEAT_DIR, f'{run}.pt'), map_location='cpu', weights_only=True)


def _pick_runs(files):
    want = ['rigb_lejp_p1s100_nano_1xxx', 'mae', 'tiny']
    out = []
    for w in want:
        hit = [f for f in files if w in f and f not in out]
        if hit:
            out.append(hit[0])
    for f in files:
        if len(out) >= 3:
            break
        if f not in out:
            out.append(f)
    return out


def run_real_tests(device):
    """Extraction check, determinism, references, fp16 effect, RankMe gate."""
    import torch
    dse_official, ansuini, fpl = _load_refs()
    TwoNN = _load_skdim()
    from scipy.spatial.distance import pdist, squareform
    files = sorted(f[:-3] for f in os.listdir(FEAT_DIR) if f.endswith('.pt'))
    runs = _pick_runs(files)
    print(f'{len(files)} feature files present; test runs: {runs}')
    picks = []
    for run in runs:
        F = load_feats(run)
        for i in (len(F['epochs']) // 2, len(F['epochs']) - 1):
            picks.append((run, F['epochs'][i], F['feats_tr'][i].numpy()))
    lab = lambda run, ep: f'{run.split("xxx")[0]} ep{ep}'   # noqa: E731

    print('=== stored features = fresh re-extraction (bit-exact) ===')
    import yaml
    import wp1_probe_driver as drv
    import knee_dir_sweep as DS
    for run, ep, X in (picks[0], picks[-1]):
        cfg = yaml.safe_load(open(os.path.join(drv.ROOT, run, 'config.yml')))
        tag = cfg['logging']['write_tag']
        sets = torch.load(os.path.join(paths.CACHE,
                                       DS.cache_for(run.split('xxx')[0])), map_location='cpu', weights_only=True)
        enc = drv.encoder_from_ckpt(cfg, os.path.join(drv.ROOT, run, f'{tag}-ep{ep}.pth.tar'))
        fresh = drv.encode_meanpool(enc, sets['imgs_tr']).float().numpy()
        T.check('A', f'stored = fresh encode_meanpool, {lab(run, ep)}: max |diff|',
                float(np.abs(fresh - X).max()), 0.0, 0.0)
    print('=== determinism: all metrics twice on two checkpoints of different runs ===')
    for run, ep, X in (picks[1], picks[3]):
        t0 = time.time()
        a = all_metrics(X, device=device)
        dt = time.time() - t0
        b = all_metrics(X, device=device)
        same = all((x[2] == y[2]) or (math.isnan(x[2]) and math.isnan(y[2])) for x, y in zip(a, b))
        T.check('A', f'determinism, {lab(run, ep)}: all {len(a)} values identical', same, True, 0,
                note=f'({dt:.1f} s per checkpoint)')
        for m, v, val, n1, d1, n2, d2 in a:
            print(f'      {m:13s} {v:16s} {val:.8g}  {n1} {d1} {n2} {d2}')
        full = {(r[0], r[1]): r for r in a}
        T.check('A', f'extra pass rows = the same variants from all_metrics, {lab(run, ep)}',
                all(full[(r[0], r[1])] == r for r in extra_metrics(X, device=device)), True, 0)
    print('=== references on real checkpoints ===')
    for run, ep, X in picks:
        X64 = X.astype(np.float64)
        T.check('A', f'dse code vs official, {lab(run, ep)}', dse_code(X, device=device),
                float(dse_official(X64)), 1e-9)
        T.check('A', f'dse paper vs numpy path, {lab(run, ep)}', dse_paper(X, device=device),
                dse_paper_numpy(X64), 1e-8)
        T.check('A', f'alpha_req vs fastssl, {lab(run, ep)}', alpha_req(X)[0], fastssl_alpha(X64, fpl), 1e-8)
        T.check('A', f'alpha_req code_wls vs the authors\' stringer_get_powerlaw, {lab(run, ep)}',
                alpha_req_wls(X), fastssl_alpha_wls(X64, fpl), 1e-6, note='(their weights are float32)')
        T.check('A', f'dse paper_codebw vs numpy path (sigma = 200), {lab(run, ep)}',
                dse_paper(X, sigma=2 * DSE_SIGMA ** 2, device=device),
                dse_paper_numpy(X64, sigma=2 * DSE_SIGMA ** 2), 1e-8)
        rng = np.random.default_rng(ID_SEED)
        idx = rng.choice(X.shape[0], size=int(np.floor(ID_FRAC * X.shape[0])), replace=False)
        _, _, ref_slope, _, _ = ansuini.estimate(squareform(pdist(X64[idx])), fraction=0.9)
        k1, k2 = _two_nn_dists(X64[idx], device)
        mine, _, _ = twonn_from_r(k1, k2)
        T.check('A', f'id fit (first subsample) vs Ansuini, {lab(run, ep)}', mine, float(ref_slope), 1e-9,
                rel=True)
        T.check('A', f'id fit (first subsample) vs scikit-dimension TwoNN, {lab(run, ep)}', mine,
                float(TwoNN().fit(X64[idx]).dimension_), 1e-9, rel=True)
        W = _l2rows(X64, False)
        Gm = W @ W.T
        Qn = float((Gm * Gm).sum())
        Nn, dn = W.shape
        T.check('A', f'self_cluster sq: d x d route = N x N route, {lab(run, ep)}', self_cluster(X),
                (dn * Qn - Nn * (dn + Nn - 1)) / ((dn - 1) * (Nn - 1) * Nn), 1e-12)
        sv = np.linalg.svd(X64 - X64.mean(0), compute_uv=False)
        T.check('A', f'ne_sum centred = stable rank of X - mean, {lab(run, ep)}', ne_sum(X, True),
                float((sv ** 2).sum() / sv[0] ** 2), 1e-9, rel=True)
    print('=== kappa exclusion on real checkpoints ===')
    for run, ep, X in picks:
        X64 = X.astype(np.float64)
        for c in (False, True):
            Xc = X64 - X64.mean(0) if c else X64
            sv = np.linalg.svd(Xc, compute_uv=False)
            k, nex, gap, tol = condition_number(X64, centred=c)
            npdef = sv[0] * max(Xc.shape) * EPS32
            print(f'  {lab(run, ep)} {"centred  " if c else "uncentred"}: smallest 3 sv '
                  f'{", ".join("%.3g" % v for v in sv[-3:])}; tol {tol:.2g} -> {nex} excluded, gap {gap:.3g}; '
                  f'tol x0.1 -> {condition_number(X64, c, tol_mult=1.0)[1]}, x10 -> '
                  f'{condition_number(X64, c, tol_mult=100.0)[1]}; numpy default ({npdef:.2g}) -> '
                  f'{int((sv <= npdef).sum())}')
    print('=== fp16 storage effect (why float32 is stored) ===')
    for run, ep, X in picks:
        Xh = X.astype(np.float16).astype(np.float64)
        for nm, fn in (('kappa uncentred', lambda Z: condition_number(Z, False)[0]),
                       ('alpha_req centred', lambda Z: alpha_req(Z, True)[0]),
                       ('ne_sum centred', lambda Z: ne_sum(Z, True)),
                       ('dse code', lambda Z: dse_code(Z, device=device)),
                       ('id fit (2 subsamples)', lambda Z: twonn(Z, n_sub=2, device=device)[0])):
            v32, v16 = fn(X), fn(Xh)
            T.check('R', f'fp16 rounding, {lab(run, ep)}, {nm}: relative change', abs(v16 - v32) / abs(v32),
                    0.0, 0.0, note=f'(float32 {v32:.6g}, fp16 {v16:.6g})')
    print('=== float32-level sensitivity: every metric on X and on X * (1 + u), u ~ U(-eps32/2, eps32/2) ===')
    for run, ep, X in picks:
        Xp = X.astype(np.float64) * (1.0 + 0.5 * EPS32 * np.random.default_rng(1).uniform(-1, 1, X.shape))
        a = all_metrics(X, device=device)
        b = all_metrics(Xp, device=device)
        for x, y in zip(a, b):
            rel = abs(y[2] - x[2]) / abs(x[2]) if x[2] != 0 else abs(y[2])
            T.check('R', f'float32-level perturbation, {lab(run, ep)}, {x[0]} {x[1]}: relative change', rel,
                    0.0, 0.0, note=f'({x[2]:.8g} -> {y[2]:.8g})')
    print('=== RankMe gate: stored features vs csv/knee_rank_estimators_prepost.csv (item 3b) ===')
    worst = {}
    n_ok = 0
    for r in csv.DictReader(open(os.path.join(paths.CSV, 'knee_rank_estimators_prepost.csv'))):
        if r['run'] not in files:
            continue
        F = load_feats(r['run'])
        for ph in ('pre', 'post'):
            ep = int(r[f'{ph}_ep'])
            if ep not in F['epochs']:
                print(f'  {r["run"]} {ph} epoch {ep} not on the stored grid')
                continue
            X = F['feats_tr'][F['epochs'].index(ep)].numpy()
            n_ok += 1
            for q, v in (('rankme_pub_centred', rankme_pub(X, True)),
                         ('rankme_pub_uncentred', rankme_pub(X, False)),
                         ('rankme_sq_centred', rankme_sq_centred(X))):
                worst[q] = max(worst.get(q, 0.0), abs(v - float(r[f'{q}_{ph}'])))
    for q, w in worst.items():
        T.check('A', f'RankMe gate, {q}, {n_ok} checkpoints: max |stored-feature value - 3b CSV|', w, 0.0,
                5e-6, note='(3b CSV printed to 6 decimals)')
    T.summary(os.path.join(REF_DIR, '..', 'lf_tests_real.csv'))


# =================================================================== pass

def extra_metrics(X, device='cpu'):
    """The two variants added after the independent review (2026-09-23), for
    the checkpoints the main pass had already scored; same functions and
    arguments as in all_metrics."""
    D2 = _sqdist(X, device)
    return [('alpha_req', 'code_wls', alpha_req_wls(X), '', '', '', ''),
            ('dse', 'paper_codebw', dse_paper(X, sigma=2 * DSE_SIGMA ** 2, device=device, D2=D2),
             '', '', '', '')]


def run_pass(part, nparts, device, extra=False):
    runs = sorted(f[:-3] for f in os.listdir(FEAT_DIR) if f.endswith('.pt'))
    runs = [r for i, r in enumerate(runs) if i % nparts == part]
    os.makedirs(PART_DIR, exist_ok=True)
    out = os.path.join(PART_DIR, f'knee_labelfree_metrics.{"extra" if extra else "part"}{part}.csv')
    fn = extra_metrics if extra else all_metrics
    done = set()
    if os.path.exists(out):
        with open(out) as fh:
            done = {(r['run'], int(r['epoch'])) for r in csv.DictReader(fh)}
    new = not os.path.exists(out)
    fout = open(out, 'a', newline='')
    w = csv.writer(fout, lineterminator='\n')
    if new:
        w.writerow(COLS)
    t_all = time.time()
    for run in runs:
        t0 = time.time()
        F = load_feats(run)
        for i, ep in enumerate(F['epochs']):
            if (run, ep) in done:
                continue
            X = F['feats_tr'][i].numpy()
            for m, v, val, n1, d1, n2, d2 in fn(X, device=device):
                w.writerow([run, ep, m, v, f'{val:.10g}', n1, d1 if d1 == '' else f'{d1:.6g}',
                            n2, d2 if d2 == '' else f'{d2:.6g}'])
            fout.flush()
        print(f'{run.split("xxx")[0]:38s} {len(F["epochs"])} ckpts ({time.time() - t0:.0f}s; '
              f'total {time.time() - t_all:.0f}s)', flush=True)
    fout.close()
    print('DONE metrics', 'extra' if extra else 'part', part)


def merge(nparts):
    assert not os.path.exists(OUT_CSV), f'refusing to overwrite {OUT_CSV}'
    rows = []
    for kind in ('part', 'extra'):
        for k in range(nparts):
            with open(os.path.join(PART_DIR, f'knee_labelfree_metrics.{kind}{k}.csv')) as fh:
                rows += list(csv.DictReader(fh))
    keys = [(r['run'], int(r['epoch']), r['metric'], r['variant']) for r in rows]
    assert len(set(keys)) == len(keys), 'duplicate rows'
    want = {(m, v) for m, v, *_ in VARIANTS}
    per = {}
    for k in keys:
        per.setdefault(k[:2], set()).add(k[2:])
    assert all(v == want for v in per.values()), 'a checkpoint lacks a variant'
    rows.sort(key=lambda r: (r['run'], int(r['epoch']), r['metric'], r['variant']))
    with open(OUT_CSV, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {OUT_CSV}: {len(rows)} rows, {len({k[:2] for k in keys})} checkpoints')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', action='store_true')
    ap.add_argument('--test-real', action='store_true')
    ap.add_argument('--part', type=int, default=None)
    ap.add_argument('--nparts', type=int, default=1)
    ap.add_argument('--merge', action='store_true')
    ap.add_argument('--extra', action='store_true', help='only the two variants added after the review')
    a = ap.parse_args()
    dev = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
    if a.test:
        run_tests(device=dev)
    elif a.test_real:
        run_real_tests(device=dev)
    elif a.merge:
        merge(a.nparts)
    elif a.part is not None:
        if not os.environ.get('CUDA_VISIBLE_DEVICES'):
            raise SystemExit('set CUDA_VISIBLE_DEVICES explicitly (shared GPUs)')
        run_pass(a.part, a.nparts, dev, extra=a.extra)
