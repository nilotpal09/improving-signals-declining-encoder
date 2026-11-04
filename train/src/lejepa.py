"""LeJEPA objective — SIGReg (sliced Epps-Pulley) + mean-over-views prediction.

This module is a PURE ADDITION, built to the same rules as `src/byol.py` and
`src/contrastive.py`. Nothing in the JEPA/MAE/BYOL/contrastive loss paths
imports it; the encoder is the standard `vit_nano` built by
`src.helper.init_model`; the data path is the standard calo loader. The only
new trainable parameters are the projector head defined here.

Why this arm exists
-------------------
LeJEPA (Balestriero & LeCun, arXiv 2511.08544) is the heuristics-free JEPA:
no EMA target, no stop-gradient, no predictor. Collapse is prevented by an
explicit distributional regularizer, SIGReg, which pushes every view's batch
of embeddings toward N(0, I) by averaging a univariate goodness-of-fit
statistic (Epps-Pulley) over random 1D projections ("slices"). Klindt, LeCun
& Balestriero (arXiv 2605.26379) prove, in an idealized setting, that this
objective is linearly identifiable iff the latents are Gaussian. The study
this module serves tests that theorem's exported
prediction on the calo rig, where the generating latent z is exactly N(0, I)
by construction and continuously deformable away from it.

Provenance (ground rule 1: implemented from the paper + official repository,
never from a summary)
---------------------------------------------------------------------------
Official repo `github.com/galilai-group/lejepa`, pinned commit
`c293d291ca87cd4fddee9d3fffe4e914c7272052` (a read-only clone at that commit
is what tests/test_lejepa.py compares against). Classes here
are transcriptions:

  EppsPulley             <- lejepa/univariate/epps_pulley.py, class EppsPulley
                            (lines 62-99) + the `world_size` property of
                            lejepa/univariate/base.py (lines 31-35).
  SlicingUnivariateTest  <- lejepa/multivariate/slicing.py (lines 91-153).
  LeJEPAProjector        <- MINIMAL.md ViTEncoder.proj:
                            MLP(in, [2048, 2048, proj_dim],
                                norm_layer=BatchNorm1d)  (torchvision.ops.MLP,
                            exactly as MINIMAL imports it).
  lejepa_inv_loss /      <- MINIMAL.md main loop:
  lejepa_loss                 inv_loss    = (proj.mean(0) - proj).square().mean()
                              lejepa_loss = sigreg*lamb + inv*(1-lamb)

`tests/test_lejepa.py` enforces bitwise CPU parity (values and gradients) with
the vendored pinned classes, so any edit here that changes the statistic fails
the battery. Where the paper text and the pinned code disagree (14 catalogued
discrepancies, numbered in LEJEPA_README.md), THE PINNED CODE
WINS (tiebreak rule, 2026-08-16): t_max=3 (not the text's 5), weight
w(t)=exp(-t^2/2) (not exp(-t^2)), 17 trapezoid knots on the half-grid with
doubled interior weights, 256 slices (MINIMAL + paper pseudocode; the text
says 1024), lambda=0.02 (MINIMAL's small-scale recommendation; the text says
0.05 — the study's single sensitivity check fixed in advance).

Named deviations from the pinned sources (everything else is verbatim):
  D1  The two `all_reduce` helpers (epps_pulley.py uses the differentiable
      torch.distributed.nn AVG collective; slicing.py uses
      torch.distributed._functional_collectives) keep the pinned dist-
      initialized guard but import lazily inside the guarded branch. This
      harness is single-process by construction (job submission asserts ngpus=1),
      so the branch never executes and behavior is identical; the lazy import
      just avoids touching private torch modules at import time.
  D2  EppsPulley subclasses nn.Module directly and carries only the base
      class's `world_size` property. The pinned `UnivariateTest` base also
      provides sorting machinery (`prepare_data`) and a Normal(0,1) member
      that the Epps-Pulley forward path never touches (sorting is for the
      CDF-based tests); they are omitted, not reimplemented.
  D3  `SlicingUnivariateTest.eval_statistic` is an ADDITION with no pinned
      counterpart: a frozen-slice, no_grad evaluation of the same statistic
      for the monitor audit. It never participates in the loss and
      does not advance the training slice counter.
  D4  `num_slices` has no default in the pinned class; `build_sigreg` fixes
      the study's frozen configuration (256) in one place.
  D5  MINIMAL's backbone emits a single 512-d vector (a timm classifier head
      on the CLS token); ViT-Nano emits the (B, 256, 96) token grid with no
      head. The embedding fed to the projector is the plain mean over tokens
      (`pool_tokens`, the same pooling the BYOL and contrastive twin arms
      use); the projector's first Linear plays the role MINIMAL's head plays.
      Projector widths are kept LITERAL from MINIMAL (2048-2048-16) rather
      than rescaled to the nano encoder — fixed-from-source beats
      proportionality, and the projector is discarded at probe time.

A gradient identity worth knowing (found red-first by the unit battery)
-----------------------------------------------------------------------
With the all-views-global center (MINIMAL's form; both study configs), the
prediction term's center-gradient contribution cancels IDENTICALLY:
sum_v (mean_v p - p_v) = 0, so d inv/d p is the same whether or not the
center is detached. LeJEPA's "no stop-gradient" is structurally true here
but gradient-EQUIVALENT to a detached center; the distinction would only
bind for V_g < V (paper multi-crop with local views), which this study never
runs. Consequence for interpretation: on this rig, "no stop-gradient" cannot
by itself explain any behavioral difference from a stop-gradient variant of
the same term — the collapse-relevant difference from I-JEPA/BYOL is the
absence of the EMA target and the presence of SIGReg, not the center's
gradient path. Pinned by test_inv_loss_center_gradient_cancels_exactly.

Reading the numbers
-------------------
The SIGReg statistic is >= 0 with an exact finite-N null mean under
embeddings ~ N(0, I): E[stat] = sum_i W_i (1 - exp(-t_i^2)) = 1.0561 at the
frozen quadrature (any batch size, any slice count — the N * batch-mean
construction cancels N in expectation; tests pin this law). A healthy run
should show the training SIGReg term fluctuating near that line, NOT at 0:
driving it far below the null mean would mean overfitting the finite batch to
the test. The prediction term is a plain squared distance in projector space;
0 is total view-collapse of the projector output, which SIGReg exists to make
unattractive. The loss actually trained is lamb-weighted, so compare runs at
equal lamb only.

Checkpoint keys: the training path (src/train_lejepa.py) saves the SIGReg
module's `global_step` buffer inside its own state dict, so a resumed run
continues the slice stream instead of restarting it.
"""

import torch
import torch.nn as nn
from torch import distributed as dist
from torchvision.ops import MLP


def _dist_ready():
    return dist.is_available() and dist.is_initialized()


def _all_reduce_avg(x):
    """epps_pulley.py's `all_reduce(x, op="AVG")` (differentiable collective);
    identity in this single-process harness. See deviation D1."""
    if _dist_ready():
        from torch.distributed.nn import all_reduce as functional_all_reduce
        from torch.distributed.nn import ReduceOp
        return functional_all_reduce(x, ReduceOp.AVG)
    return x


def _all_reduce_max(x):
    """slicing.py's `all_reduce(x, op="MAX")` (functional collective, used
    under no_grad for counter sync); identity in this single-process harness.
    See deviation D1."""
    if _dist_ready():
        from torch.distributed._functional_collectives import (
            all_reduce as functional_all_reduce)
        return functional_all_reduce(x, 'max', dist.group.WORLD)
    return x


class EppsPulley(nn.Module):
    """Epps-Pulley goodness-of-fit statistic against N(0, 1), vectorized over
    a trailing slice dimension. Transcribed from the pinned
    `lejepa/univariate/epps_pulley.py` class `EppsPulley` (the production
    statistic; NOT the `DeprecatedEppsPulley` that shadowed it before commit
    fd27093 — discrepancy 13).

    Input (*, N, K): N samples of K slices. Output (*, K): one statistic per
    slice,  stat = N_global * sum_i W_i [ (cos_mean_i - phi_i)^2 +
    sin_mean_i^2 ]  with W = trapezoid weights on the half-grid [0, t_max]
    (doubled interior weights = the symmetric integral over [-t_max, t_max])
    times the Gaussian window phi(t) = exp(-t^2/2), which is also the target
    characteristic function (the CF of N(0,1) is real, so only the cosine
    channel is centered).
    """

    def __init__(
        self, t_max: float = 3, n_points: int = 17, integration: str = "trapezoid"
    ):
        super().__init__()
        assert n_points % 2 == 1
        self.integration = integration
        self.n_points = n_points
        # Precompute phi

        # Linearly spaced positive points (including 0)
        t = torch.linspace(0, t_max, n_points, dtype=torch.float32)
        self.register_buffer("t", t)
        dt = t_max / (n_points - 1)
        weights = torch.full((n_points,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt  # Half-weight at t=0
        self.register_buffer("phi", self.t.square().mul_(0.5).neg_().exp_())
        self.register_buffer("weights", weights * self.phi)

    @property
    def world_size(self):
        # lejepa/univariate/base.py::UnivariateTest.world_size, verbatim guard
        if _dist_ready():
            return dist.get_world_size()
        return 1

    def forward(self, x):
        N = x.size(-2)
        # Compute cos/sin only for t >= 0
        x_t = x.unsqueeze(-1) * self.t  # (*, N, K, n_points)
        cos_vals = torch.cos(x_t)
        sin_vals = torch.sin(x_t)

        # Mean across batch
        cos_mean = cos_vals.mean(-3)  # (*, n_points)
        sin_mean = sin_vals.mean(-3)  # (*, n_points)

        # DDP reduction
        cos_mean = _all_reduce_avg(cos_mean)
        sin_mean = _all_reduce_avg(sin_mean)

        # Compute error (symmetry already in weights)
        err = (cos_mean - self.phi).square() + sin_mean.square()

        # Weighted integration
        return (err @ self.weights) * N * self.world_size


class SlicingUnivariateTest(nn.Module):
    """Sliced multivariate wrapper: project (*, N, D) samples onto
    `num_slices` random unit directions and aggregate the univariate
    statistic across slices. Transcribed from the pinned
    `lejepa/multivariate/slicing.py`.

    The slice matrix is drawn UNDER no_grad from a generator seeded with the
    internal step counter, so (a) the directions are constants of the graph —
    gradients flow only through the projected samples — and (b) the slice
    stream is reproducible: same counter state => same directions on any
    process (the library's DDP-sync mechanism, equal on one GPU to the paper
    pseudocode's `manual_seed(global_step)`). MINIMAL.md instead draws
    unseeded fresh directions each call — same distribution, no
    reproducibility; documented as discrepancy 4. The counter is a
    registered buffer: it checkpoints, so resume continues the stream.
    """

    def __init__(
        self,
        univariate_test,
        num_slices: int,
        reduction: str = "mean",
        sampler: str = "gaussian",
        clip_value: float = None,
    ):
        super().__init__()
        self.reduction = reduction
        self.num_slices = num_slices
        self.sampler = sampler
        self.univariate_test = univariate_test
        self.clip_value = clip_value
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))

        # Generator reuse
        self._generator = None
        self._generator_device = None

    def _get_generator(self, device, seed):
        """Get or create generator for given device and seed."""
        if self._generator is None or self._generator_device != device:
            self._generator = torch.Generator(device=device)
            self._generator_device = device
        self._generator.manual_seed(seed)
        return self._generator

    def forward(self, x):
        with torch.no_grad():
            # Synchronize global_step across all ranks
            global_step_sync = _all_reduce_max(self.global_step.clone())
            seed = global_step_sync.item()
            dev = dict(device=x.device)

            # Get reusable generator
            g = self._get_generator(x.device, seed)

            proj_shape = (x.size(-1), self.num_slices)
            A = torch.randn(proj_shape, **dev, generator=g)
            A /= A.norm(p=2, dim=0)
            self.global_step.add_(1)

        stats = self.univariate_test(x @ A)
        if self.clip_value is not None:
            stats[stats < self.clip_value] = 0
        if self.reduction == "mean":
            return stats.mean()
        elif self.reduction == "sum":
            return stats.sum()
        elif self.reduction is None:
            return stats

    @torch.no_grad()
    def eval_statistic(self, x, seed: int):
        """Monitor-only variant (deviation D3): the same statistic on
        a FIXED slice set selected by `seed`, without touching the training
        counter or the autograd graph. Deterministic per (x, seed, device) —
        CPU and CUDA generators draw different (equally valid) streams.
        Mirrors `forward`'s clip/reduction semantics so monitor numbers sit
        on the training statistic's scale."""
        g = torch.Generator(device=x.device)
        g.manual_seed(seed)
        A = torch.randn((x.size(-1), self.num_slices), device=x.device,
                        generator=g)
        A /= A.norm(p=2, dim=0)
        stats = self.univariate_test(x @ A)
        if self.clip_value is not None:
            stats[stats < self.clip_value] = 0
        if self.reduction == "mean":
            return stats.mean()
        elif self.reduction == "sum":
            return stats.sum()
        return stats


class MomentsPerView(nn.Module):
    """The whitening-arm statistic: transcription of the pinned
    `lejepa/univariate/moments.py` class `Moments`, plus a per-view adapter
    (deviation D7). With the study's k_max = 2 it constrains the MEAN and
    VARIANCE of each slice only —

        stat = |mean| + e^{-2} |E[x^2] - 1|   per slice —

    i.e. the moment/VICReg-family constraint the LeJEPA paper argues
    against via its Theorem 3 (matching finitely many moments does not
    imply Gaussianity). That insufficiency is exactly why the arm exists:
    this statistic is BLIND to the shape dial by construction (verified by
    test), so LeJEPA-with-moment-T isolates whether SIGReg's full
    distributional constraint, versus mere whitening, carries the
    Gaussianity-dependence. NOT N-scaled (the pinned class isn't); its
    magnitude differs from EppsPulley's — arm comparisons are within-arm
    across the dial, never raw-loss across statistics.

    Adapter note: the pinned forward supports (N, K) input only (its
    reductions are dim-0/dim-1 specific); for (V, B, K) view stacks we
    apply the 2-D forward per view and stack, so SlicingUnivariateTest's
    mean reduction averages over views and slices exactly as with
    EppsPulley. 2-D calls are bitwise-parity-tested against the pinned
    class, values and gradients. The pinned base class's `sorted=True`
    means prepare_data passes x through unsorted (moments are
    permutation-invariant); transcribed faithfully."""

    def __init__(self, k_max: int = 2):
        super().__init__()
        from scipy.stats import norm   # lazy: keep training-node import light
        self.k_max = k_max
        self.sorted = True
        moments = []
        for i in range(2, k_max + 1, 2):
            moment_val = norm(loc=0, scale=1).moment(i)
            moments.append(moment_val)
        self.register_buffer("moments", torch.Tensor(moments).unsqueeze(1))
        self.register_buffer("weights",
                             torch.arange(2, self.k_max + 1).neg().exp())

    @property
    def world_size(self):
        if _dist_ready():
            return dist.get_world_size()
        return 1

    def _dist_mean(self, x):
        # base.py::UnivariateTest.dist_mean, single-process identity
        if _dist_ready():
            from torch.distributed.nn.functional import all_reduce
            all_reduce(x, dist.ReduceOp.AVG)
        return x

    def prepare_data(self, x):
        if self.sorted:
            return x
        return x.sort(descending=False, dim=-2)[0]

    def _forward_2d(self, x):
        # pinned Moments.forward, verbatim semantics for (N, K) input
        x = self.prepare_data(x)
        k = torch.arange(2, self.k_max + 1, device=x.device, dtype=x.dtype).view(
            -1, 1, 1
        )
        m1 = self._dist_mean(x.mean(0)).abs_()
        if self.k_max >= 2:
            xpow = self._dist_mean((x**k).mean(1))
            xpow[::2].sub_(self.moments)
            m2 = xpow.abs_().T.matmul(self.weights)
            return m1.add_(m2) / self.world_size
        return m1 / self.world_size

    def forward(self, x):
        if x.dim() == 2:
            return self._forward_2d(x)
        lead = x.shape[:-2]
        flat = x.reshape(-1, x.shape[-2], x.shape[-1])
        out = torch.stack([self._forward_2d(flat[i])
                           for i in range(flat.shape[0])])
        return out.reshape(*lead, x.shape[-1])


def build_sigreg(num_slices: int = 256, t_max: float = 3, n_points: int = 17,
                 reduction: str = "mean", clip_value: float = None,
                 statistic: str = "epps_pulley"):
    """The study's frozen SIGReg configuration in one place
    (deviation D4 for the slice default).
    `statistic` selects the univariate T: 'epps_pulley' (the LeJEPA arm) or
    'moments' (the whitening-control arm, k_max=2 — see MomentsPerView)."""
    assert statistic in ("epps_pulley", "moments"), (
        f"unknown statistic {statistic!r}")
    if statistic == "moments":
        t = MomentsPerView(k_max=2)
    else:
        t = EppsPulley(t_max=t_max, n_points=n_points)
    return SlicingUnivariateTest(
        t, num_slices=num_slices, reduction=reduction, clip_value=clip_value)


class LeJEPAProjector(nn.Module):
    """MINIMAL.md's projector head, exactly as it builds it:
    `MLP(in_dim, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)` from
    torchvision.ops (Linear-BN-ReLU-Dropout(0) twice, bare final Linear).
    Applied to the (V*B, D) flattened view stack, so BatchNorm pools views
    and events jointly — MINIMAL's behavior, kept deliberately. The LeJEPA
    loss reads the projector output; every probe reads the backbone, so this
    head is discarded at probe time (saved under its own checkpoint key)."""

    def __init__(self, in_dim: int, hidden_dim: int = 2048, proj_dim: int = 16):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.proj_dim = int(proj_dim)
        self.mlp = MLP(self.in_dim, [self.hidden_dim, self.hidden_dim,
                                     self.proj_dim],
                       norm_layer=nn.BatchNorm1d)

    def forward(self, x):
        return self.mlp(x)


def pool_tokens(h):
    """(B, N, D) standard token grid -> (B, D), plain mean over tokens — the
    same pooling the BYOL and contrastive twin arms feed their heads
    (src/byol.py::pool_tokens). See deviation D5 for why this stands in for
    MINIMAL's timm classifier head."""
    return h.mean(dim=1)


def lejepa_inv_loss(proj):
    """MINIMAL.md's prediction ("invariance") term, verbatim semantics:
    every view regresses onto the UN-DETACHED mean over views,

        inv_loss = (proj.mean(0) - proj).square().mean()

    `proj` is (V, B, D) — views leading, exactly MINIMAL's layout. With two
    views this is the paper's Eq. (pred_as) with V_g = V = 2 up to its
    additive constant; MINIMAL treats all views as global, and both study
    configs (harness-matched V=2, faithful V=4) do the same. NO stop-gradient
    anywhere: the center carries gradient by design — that is LeJEPA's claim,
    and tests/test_lejepa.py fails if one is introduced."""
    return (proj.mean(0) - proj).square().mean()


def lejepa_loss(proj, sigreg, lamb):
    """MINIMAL.md's total loss on a (V, B, D) projector-output stack:

        lejepa_loss = sigreg(proj) * lamb + inv_loss * (1 - lamb)

    `sigreg` is a SlicingUnivariateTest (build_sigreg()); called on the full
    (V, B, D) stack it computes the per-view statistic over the batch
    dimension and means over views and slices — MINIMAL's exact call shape.
    Returns (total, parts) with float parts for the csv line."""
    assert proj.dim() == 3 and proj.size(0) >= 2, (
        f'expected (V, B, D) with V >= 2 views, got {tuple(proj.shape)}')
    inv = lejepa_inv_loss(proj)
    sig = sigreg(proj)
    total = sig * lamb + inv * (1 - lamb)
    parts = {
        'lejepa_inv': float(inv),
        'lejepa_sigreg': float(sig),
        'lejepa_total': float(total),
    }
    return total, parts
