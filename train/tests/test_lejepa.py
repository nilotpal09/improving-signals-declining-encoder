"""Unit battery for the LeJEPA objective module (SIGReg fidelity pass).

The study this arm serves lives or dies on one question: is OUR SIGReg the
SAME statistic as the official one at the pinned commit? Ground rule 1
forbids implementing from summaries, and an independent adversarial diff
against the pinned repo runs after the pilot.
These tests are therefore built around direct numerical parity with the pinned
sources: a read-only clone of

    github.com/galilai-group/lejepa @ c293d291ca87cd4fddee9d3fffe4e914c7272052

whose path is given by the environment variable LEJEPA_REF (the tests are
skipped without it).

What these tests protect, in order of how much the study depends on them:

  1. PARITY WITH THE PINNED CODE. Our EppsPulley must equal the pinned
     `lejepa.univariate.epps_pulley.EppsPulley` bitwise on CPU — values AND
     gradients, all reductions, with and without clip_value — and our slicing
     wrapper must equal the pinned `lejepa.multivariate.slicing
     .SlicingUnivariateTest` across successive calls (same seeded slice
     stream). A vendor-shadowing guard pins the import to the vendored clone,
     never a pip wheel (the pip 0.0.1 wheel may predate the class-shadowing
     fix, discrepancy 13).
  2. THE STATISTIC IS THE FORMULA. Independently of any implementation, the
     module must equal a from-scratch complex-arithmetic Epps-Pulley
     N * integral(|ecf - cf|^2 w(t)) computed on the symmetric grid — this
     validates the [0, t_max] half-grid + doubled-weights trick against the
     paper's Eq. (Epps-Pulley) — and must obey the exact finite-N null mean
     E[stat] = sum_i W_i (1 - phi(t_i)^2), which pins the N-scaling and the
     batch-mean semantics distribution-side.
  3. THE SLICES ARE UNIT VECTORS, SEEDED BY THE STEP COUNTER, AND CARRY NO
     GRADIENT. Column-normalized Gaussian directions, drawn under no_grad from
     a generator seeded with the internal counter (the library mechanism =
     paper pseudocode on 1 GPU; MINIMAL's unseeded variant is the documented
     discrepancy 4), advancing every call, reproducible across instances.
  4. THE LOSS IS MINIMAL.md's, WITH NO STOP-GRADIENT. inv_loss regresses every
     view onto the un-detached mean over views; total = lamb * sigreg +
     (1 - lamb) * inv. Gradient must flow through the center term (a detached
     center is a different objective — LeJEPA's whole point is no
     stop-gradient), and lamb endpoints must recover the pure terms.
  5. THE PROJECTOR IS MINIMAL's. torchvision MLP(in, [2048, 2048, proj_dim],
     norm_layer=BatchNorm1d): Linear-BN-ReLU twice, bare final Linear, loss on
     projector output — pinned layer-by-layer so a torchvision upgrade cannot
     silently change the head.
  6. HARNESS SAFETY. Finite under bf16 autocast (the house precision), fp32
     buffers, inputs never mutated, world_size = 1 without distributed init,
     SIGReg state (step counter) survives a state_dict round-trip, and the
     frozen-slice monitor variant is deterministic and does not
     advance the training slice stream.

Run: python -m pytest tests/test_lejepa.py -q
"""

import math
import os
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import pytest
import torch
import torch.nn as nn

torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

# The pinned official repo: a clone at the commit above, path in LEJEPA_REF.
_VENDOR = os.environ.get('LEJEPA_REF', '')
if not os.path.isdir(os.path.join(_VENDOR, 'lejepa')):
    pytest.skip('set LEJEPA_REF to a clone of github.com/galilai-group/lejepa at c293d29',
                allow_module_level=True)
sys.path.insert(1, _VENDOR)

from src.lejepa import (  # noqa: E402
    EppsPulley, SlicingUnivariateTest, LeJEPAProjector, build_sigreg,
    lejepa_inv_loss, lejepa_loss, pool_tokens)

import lejepa as _pinned_pkg  # noqa: E402
from lejepa.univariate.epps_pulley import EppsPulley as PinnedEppsPulley  # noqa: E402
from lejepa.multivariate.slicing import (  # noqa: E402
    SlicingUnivariateTest as PinnedSlicing)

DEV = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# the real calo geometry this arm trains on
IMG, PATCH, CHANS = 64, 4, 4
NTOK, EMBED = (IMG // PATCH) ** 2, 96          # 256 tokens, 96 dims


def _x(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


# --------------------------------------------- 0. the reference IS the reference

def test_pinned_import_resolves_to_the_vendored_clone_not_a_pip_wheel():
    """Discrepancy 13: pip's lejepa 0.0.1 may predate the EppsPulley
    class-shadowing fix. Every parity test below is meaningless unless the
    `lejepa` package we compare against is the vendored pinned clone."""
    got = os.path.realpath(os.path.dirname(_pinned_pkg.__file__))
    want = os.path.realpath(os.path.join(_VENDOR, 'lejepa'))
    assert got == want, f'pinned import shadowed: {got} != {want}'
    # and the clone is at the pinned commit
    head = open(os.path.join(_VENDOR, '.git', 'HEAD')).read().strip()
    assert head.startswith('c293d291'), f'vendor clone not at pinned commit: {head}'


# ------------------------------------- 1. parity with the pinned implementation

def _pairs(seed0=0):
    """(ours, pinned) EppsPulley instances with default (t_max=3, n_points=17)."""
    return EppsPulley(), PinnedEppsPulley()


@pytest.mark.parametrize('shape', [(64, 1), (64, 5), (128, 32), (2, 64, 7),
                                   (3, 2, 33, 4)])
def test_epps_pulley_values_match_pinned_bitwise_on_cpu(shape):
    ours, pinned = _pairs()
    x = _x(shape, seed=hash(shape) % 2**31)
    a, b = ours(x), pinned(x)
    assert a.shape == b.shape
    assert torch.equal(a, b), f'max abs diff {(a - b).abs().max().item():.3e}'


def test_epps_pulley_gradients_match_pinned_bitwise_on_cpu():
    ours, pinned = _pairs()
    x1 = _x((96, 8), seed=11).requires_grad_(True)
    x2 = x1.detach().clone().requires_grad_(True)
    ours(x1).sum().backward()
    pinned(x2).sum().backward()
    assert torch.equal(x1.grad, x2.grad)


def test_epps_pulley_nondefault_args_match_pinned():
    ours = EppsPulley(t_max=5.0, n_points=21)
    pinned = PinnedEppsPulley(t_max=5.0, n_points=21)
    x = _x((200, 3), seed=5)
    assert torch.equal(ours(x), pinned(x))


def test_epps_pulley_rejects_even_n_points_like_pinned():
    with pytest.raises(AssertionError):
        EppsPulley(n_points=16)
    with pytest.raises(AssertionError):
        PinnedEppsPulley(n_points=16)


@pytest.mark.parametrize('reduction', ['mean', 'sum', None])
def test_slicing_matches_pinned_across_three_successive_calls(reduction):
    """Fresh instances share counter state 0; the seeded generator must give
    the identical slice stream, so outputs match call-for-call as the counter
    advances. This is the mechanism that makes the SIGReg loss reproducible."""
    ours = SlicingUnivariateTest(EppsPulley(), num_slices=32, reduction=reduction)
    pinned = PinnedSlicing(PinnedEppsPulley(), num_slices=32, reduction=reduction)
    for call in range(3):
        x = _x((64, 12), seed=100 + call)
        a, b = ours(x), pinned(x)
        assert torch.equal(a, b), f'call {call}: sliced stats diverge'
    assert int(ours.global_step) == int(pinned.global_step) == 3


def test_slicing_clip_value_matches_pinned():
    ours = SlicingUnivariateTest(EppsPulley(), num_slices=64, reduction=None,
                                 clip_value=0.5)
    pinned = PinnedSlicing(PinnedEppsPulley(), num_slices=64, reduction=None,
                           clip_value=0.5)
    x = _x((48, 6), seed=7)
    a, b = ours(x), pinned(x)
    assert torch.equal(a, b)
    assert (a == 0).any(), 'clip test vacuous: no statistic fell below clip_value'


def test_slicing_gradients_match_pinned():
    ours = SlicingUnivariateTest(EppsPulley(), num_slices=16)
    pinned = PinnedSlicing(PinnedEppsPulley(), num_slices=16)
    x1 = _x((32, 9), seed=3).requires_grad_(True)
    x2 = x1.detach().clone().requires_grad_(True)
    ours(x1).backward()
    pinned(x2).backward()
    assert torch.equal(x1.grad, x2.grad)


# ------------------------------------------------- 2. the statistic IS the formula

def _brute_force_ep(x_slice, t_max=3.0, n_points=17):
    """Paper Eq. (Epps-Pulley), no tricks: complex ECF on the SYMMETRIC grid
    [-t_max, t_max] (2*n_points - 1 nodes, same spacing as the module's
    half-grid), Gaussian window exp(-t^2/2) (the pinned code's window,
    discrepancy 11), trapezoidal rule, scaled by N."""
    n = x_slice.shape[0]
    t = torch.linspace(-t_max, t_max, 2 * n_points - 1, dtype=torch.float64)
    xt = x_slice.double().unsqueeze(-1) * t          # (N, T)
    ecf = torch.complex(torch.cos(xt), torch.sin(xt)).mean(0)
    cf = torch.exp(-0.5 * t ** 2)
    w = torch.exp(-0.5 * t ** 2)
    integrand = (ecf - cf).abs() ** 2 * w
    return float(torch.trapz(integrand, t)) * n


def test_epps_pulley_equals_brute_force_symmetric_integral():
    ep = EppsPulley()
    x = _x((256, 4), seed=21)
    got = ep(x)
    for k in range(x.shape[1]):
        want = _brute_force_ep(x[:, k])
        assert got[k].item() == pytest.approx(want, rel=1e-5), f'slice {k}'


def test_epps_pulley_buffers_are_the_documented_quadrature():
    """Pins the half-grid construction, incl. that `t` survives the pinned
    code's in-place chain (phi is built from a fresh tensor, t is unchanged)."""
    ep = EppsPulley()
    t_want = torch.linspace(0, 3, 17, dtype=torch.float32)
    assert torch.equal(ep.t, t_want), 't buffer was corrupted at init'
    assert torch.allclose(ep.phi, torch.exp(-0.5 * t_want ** 2))
    dt = 3 / 16
    trap = torch.full((17,), 2 * dt)
    trap[0] = trap[-1] = dt
    assert torch.allclose(ep.weights, trap * ep.phi)
    for buf in (ep.t, ep.phi, ep.weights):
        assert buf.dtype == torch.float32


def test_epps_pulley_exact_null_mean_law():
    """Under H0 (true N(0,1) samples) the discrete statistic obeys, exactly at
    every N,   E[stat] = sum_i W_i (1 - phi(t_i)^2),
    because E[(cos_mean - phi)^2 + sin_mean^2] = Var(cos(tX))/N + Var(sin(tX))/N
    = (1 - phi(t)^2)/N and the statistic multiplies by N. This pins the
    N * batch-mean semantics against the distribution itself."""
    ep = EppsPulley()
    want = float((ep.weights * (1.0 - ep.phi ** 2)).sum())
    reps, n = 400, 256
    g = torch.Generator().manual_seed(1234)
    vals = []
    for _ in range(reps):
        x = torch.randn(n, 1, generator=g)
        vals.append(float(ep(x)))
    got = sum(vals) / reps
    se = torch.tensor(vals).std().item() / math.sqrt(reps)
    assert abs(got - want) < 5 * se + 0.02, (
        f'null mean {got:.4f} vs exact {want:.4f} (5*SE={5*se:.4f})')


def test_epps_pulley_orders_alternatives_above_the_null():
    """Monotone sanity on 1D samples: the statistic must rank the standard
    normal below shifted, inflated, and bimodal alternatives at large N."""
    ep = EppsPulley()
    g = torch.Generator().manual_seed(77)
    z = torch.randn(4096, 1, generator=g)
    null = float(ep(z))
    shifted = float(ep(z + 0.5))
    inflated = float(ep(z * 2.0))
    sign = torch.where(torch.rand(4096, 1, generator=g) < 0.5, -1.0, 1.0)
    bimodal = float(ep(sign * 0.97 + z * math.sqrt(1 - 0.97 ** 2)))
    assert null < 0.5 * min(shifted, inflated, bimodal), (
        f'null {null:.3f} not well below alternatives '
        f'{shifted:.3f}/{inflated:.3f}/{bimodal:.3f}')
    assert null < 3.0, f'null statistic implausibly large: {null:.3f}'


def test_statistic_scales_linearly_with_batch_size_under_an_alternative():
    """The N * |ecf - cf|^2 scaling: under a FIXED alternative the population
    discrepancy L is constant, so stat ~ N * L doubles when N doubles."""
    ep = EppsPulley()
    g = torch.Generator().manual_seed(9)
    big = torch.randn(8192, 1, generator=g) + 1.0
    s_full = float(ep(big))
    s_half = float(ep(big[:4096]))
    assert s_full / s_half == pytest.approx(2.0, rel=0.05)


# ----------------------------------------------------- 3. the slices themselves

def test_slices_are_unit_columns_drawn_from_the_seeded_counter():
    sl = SlicingUnivariateTest(EppsPulley(), num_slices=64)
    # reproduce call 0's draw with the library mechanism: seed = counter = 0
    g = torch.Generator(device='cpu').manual_seed(0)
    A = torch.randn(12, 64, generator=g)
    A = A / A.norm(p=2, dim=0)
    assert torch.allclose(A.norm(p=2, dim=0), torch.ones(64), atol=1e-6)
    x = _x((32, 12), seed=1)
    got = sl(x)
    want = EppsPulley()(x @ A).mean()
    assert torch.equal(got, want), 'call 0 did not use the seed-0 slice draw'


def test_slice_stream_advances_and_is_reproducible_across_instances():
    a = SlicingUnivariateTest(EppsPulley(), num_slices=16)
    b = SlicingUnivariateTest(EppsPulley(), num_slices=16)
    x = _x((64, 8), seed=2)
    first_a, first_b = a(x), b(x)
    assert torch.equal(first_a, first_b), 'same counter state, different slices'
    second_a = a(x)
    assert not torch.equal(first_a, second_a), 'slice stream did not advance'
    assert int(a.global_step) == 2 and int(b.global_step) == 1


def test_slices_carry_no_gradient_and_input_gets_one():
    sl = SlicingUnivariateTest(EppsPulley(), num_slices=8)
    x = _x((32, 6), seed=4).requires_grad_(True)
    out = sl(x)
    out.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert not sl.global_step.requires_grad


def test_sigreg_state_survives_a_state_dict_round_trip():
    """The counter is a registered buffer: checkpoint resume must continue the
    slice stream, not restart it."""
    a = SlicingUnivariateTest(EppsPulley(), num_slices=8)
    x = _x((16, 5), seed=6)
    a(x), a(x), a(x)
    b = SlicingUnivariateTest(EppsPulley(), num_slices=8)
    b.load_state_dict(a.state_dict())
    assert int(b.global_step) == 3
    assert torch.equal(a(x), b(x))


def test_build_sigreg_defaults_are_the_frozen_config():
    """The frozen configuration: 256 slices (MINIMAL + paper pseudocode),
    t_max=3, 17 points (pinned code defaults), mean reduction."""
    sig = build_sigreg()
    assert sig.num_slices == 256
    assert sig.reduction == 'mean'
    assert sig.clip_value is None
    ep = sig.univariate_test
    assert isinstance(ep, EppsPulley)
    assert ep.n_points == 17
    assert float(ep.t[-1]) == pytest.approx(3.0)


# ------------------------------------------- 4. the loss is MINIMAL.md's, no SG

def _minimal_sigreg_reference(proj, A):
    """MINIMAL.md's SIGReg.forward, transcribed, with the slice matrix
    injected so it can be compared against the library path on equal terms."""
    t = torch.linspace(0, 3, 17, dtype=torch.float32)
    dt = 3 / 16
    weights = torch.full((17,), 2 * dt)
    weights[[0, -1]] = dt
    window = torch.exp(-t.square() / 2.0)
    phi, weights = window, weights * window
    x_t = (proj @ A).unsqueeze(-1) * t
    err = (x_t.cos().mean(-3) - phi).square() + x_t.sin().mean(-3).square()
    return ((err @ weights) * proj.size(-2)).mean()


def test_library_path_equals_minimal_reference_given_the_same_slices():
    """MINIMAL's hardcoded statistic and the library EppsPulley+slicing path
    are the same mathematics; with the slice draw held fixed they must agree
    on a (V, B, D) view stack — including the per-view vectorization."""
    proj = _x((4, 32, 16), seed=13)
    sl = SlicingUnivariateTest(EppsPulley(), num_slices=64)
    got = sl(proj)
    g = torch.Generator(device='cpu').manual_seed(0)   # counter state 0
    A = torch.randn(16, 64, generator=g)
    A = A / A.norm(p=2, dim=0)
    want = _minimal_sigreg_reference(proj, A)
    assert torch.allclose(got, want, atol=1e-6)


def test_inv_loss_is_the_undetached_mean_over_views_regression():
    """Hand check on V=2, B=2, D=1: inv = mean over all views/events of
    (view_mean - view)^2 = mean of (p1-p2)^2/4 elementwise."""
    p = torch.tensor([[[1.0], [2.0]],      # view 1
                      [[3.0], [8.0]]])     # view 2
    want = (((1 - 3) ** 2 / 4) + ((2 - 8) ** 2 / 4) +
            ((3 - 1) ** 2 / 4) + ((8 - 2) ** 2 / 4)) / 4
    assert lejepa_inv_loss(p).item() == pytest.approx(want)


def test_inv_loss_center_gradient_cancels_exactly():
    """Pin a real property of the objective, discovered red-first: with the
    ALL-views-global center (MINIMAL's form, used by both study configs), the
    center's gradient contribution cancels IDENTICALLY — for the quadratic
    loss, sum_v (mean_v p - p_v) = 0, so d inv/d p_u = -2 (m - p_u)/(VBD)
    whether or not the center is detached. LeJEPA's "no stop-gradient" is
    structurally true (test below) but gradient-EQUIVALENT to a detached
    center whenever every view enters the mean; it would only bite for
    V_g < V (paper multi-crop), which this study never runs. The equality is
    load-bearing knowledge for the fidelity README, so it is pinned."""
    p1 = _x((3, 8, 4), seed=17).requires_grad_(True)
    lejepa_inv_loss(p1).backward()
    p2 = p1.detach().clone().requires_grad_(True)
    (p2.mean(0).detach() - p2).square().mean().backward()
    assert torch.allclose(p1.grad, p2.grad, atol=1e-7), (
        'center-gradient cancellation identity broken — the pred term is no '
        'longer the all-global quadratic mean-over-views form')
    assert torch.isfinite(p1.grad).all()
    # and every view receives gradient
    assert (p1.grad.abs().sum(dim=(1, 2)) > 0).all()


def test_inv_loss_source_contains_no_detach_or_no_grad():
    """The structural half of the no-stop-gradient claim: the implementation
    must not introduce detach()/no_grad anywhere in the loss path (the house
    ast-check convention, as in test_byol)."""
    import inspect
    src_txt = inspect.getsource(lejepa_inv_loss) + inspect.getsource(lejepa_loss)
    assert 'detach(' not in src_txt and 'no_grad' not in src_txt


def test_lejepa_loss_lambda_endpoints_recover_the_pure_terms():
    proj = _x((2, 64, 16), seed=19)
    sig0 = SlicingUnivariateTest(EppsPulley(), num_slices=32)
    total0, parts0 = lejepa_loss(proj, sig0, lamb=0.0)
    assert total0.item() == pytest.approx(parts0['lejepa_inv'], rel=1e-6)
    sig1 = SlicingUnivariateTest(EppsPulley(), num_slices=32)
    total1, parts1 = lejepa_loss(proj, sig1, lamb=1.0)
    assert total1.item() == pytest.approx(parts1['lejepa_sigreg'], rel=1e-6)
    sig = SlicingUnivariateTest(EppsPulley(), num_slices=32)
    lam = 0.02
    total, parts = lejepa_loss(proj, sig, lamb=lam)
    assert total.item() == pytest.approx(
        lam * parts['lejepa_sigreg'] + (1 - lam) * parts['lejepa_inv'], rel=1e-6)
    assert parts['lejepa_total'] == pytest.approx(total.item(), rel=1e-6)


def test_lejepa_loss_rejects_a_single_view():
    sig = SlicingUnivariateTest(EppsPulley(), num_slices=8)
    with pytest.raises(AssertionError):
        lejepa_loss(_x((1, 16, 8), seed=23), sig, lamb=0.02)
    with pytest.raises(AssertionError):
        lejepa_loss(_x((16, 8), seed=23), sig, lamb=0.02)


# ------------------------------------------------------- 5. the projector head

def test_projector_is_minimals_torchvision_mlp_layer_for_layer():
    """MLP(96, [2048, 2048, 16], norm_layer=BatchNorm1d): Linear-BN-ReLU-Drop
    twice, then a bare Linear(2048, 16)(-Drop). Pinned layer-by-layer so a
    torchvision upgrade cannot silently change the head."""
    prj = LeJEPAProjector(in_dim=EMBED)
    seq = [m for m in prj.mlp.modules() if not isinstance(
        m, (type(prj.mlp), nn.Sequential))]
    names = [type(m).__name__ for m in seq]
    assert names == ['Linear', 'BatchNorm1d', 'ReLU', 'Dropout',
                     'Linear', 'BatchNorm1d', 'ReLU', 'Dropout',
                     'Linear', 'Dropout'], names
    lin = [m for m in seq if isinstance(m, nn.Linear)]
    assert (lin[0].in_features, lin[0].out_features) == (96, 2048)
    assert (lin[1].in_features, lin[1].out_features) == (2048, 2048)
    assert (lin[2].in_features, lin[2].out_features) == (2048, 16)
    drops = [m for m in seq if isinstance(m, nn.Dropout)]
    assert all(d.p == 0.0 for d in drops)


def test_projector_forward_shapes_and_batchnorm_uses_the_flat_batch():
    """MINIMAL projects the (N*V, D) flattened stack, so BatchNorm statistics
    pool views and events jointly; eval mode must use running stats."""
    prj = LeJEPAProjector(in_dim=EMBED)
    x = _x((6 * 4, EMBED), seed=29)
    out = prj(x)
    assert out.shape == (24, 16)
    prj.eval()
    one = prj(x[:1])
    assert one.shape == (1, 16) and torch.isfinite(one).all()


def test_pool_tokens_is_the_house_mean_over_tokens():
    h = _x((5, NTOK, EMBED), seed=31)
    assert torch.equal(pool_tokens(h), h.mean(dim=1))


# ------------------------------------------------- 6. harness safety + monitor

def test_forward_is_finite_under_bf16_autocast_and_buffers_stay_fp32():
    dev_type = 'cuda' if DEV.startswith('cuda') else 'cpu'
    sig = build_sigreg().to(DEV)
    proj = _x((2, 64, 16), seed=37).to(DEV)
    with torch.autocast(dev_type, dtype=torch.bfloat16):
        total, parts = lejepa_loss(proj, sig, lamb=0.02)
    assert torch.isfinite(total).all()
    assert all(math.isfinite(v) for v in parts.values())
    ep = sig.univariate_test
    assert ep.t.dtype == ep.phi.dtype == ep.weights.dtype == torch.float32


def test_forward_does_not_mutate_its_input():
    sig = build_sigreg()
    proj = _x((2, 32, 16), seed=41)
    keep = proj.clone()
    lejepa_loss(proj, sig, lamb=0.02)
    assert torch.equal(proj, keep)


def test_world_size_is_one_without_distributed_init():
    assert EppsPulley().world_size == 1


def test_eval_statistic_is_frozen_deterministic_and_off_the_training_stream():
    """Monitor variant: a FIXED slice set (own seed), no_grad, and the
    training counter must not move — logging the monitor between steps must
    not perturb the training-time slice stream."""
    sl = SlicingUnivariateTest(EppsPulley(), num_slices=32)
    x = _x((64, 12), seed=43)
    step_before = int(sl.global_step)
    m1 = sl.eval_statistic(x, seed=999)
    m2 = sl.eval_statistic(x, seed=999)
    assert torch.equal(m1, m2), 'frozen-slice monitor is not deterministic'
    assert int(sl.global_step) == step_before, 'monitor advanced the counter'
    assert not m1.requires_grad
    m3 = sl.eval_statistic(x, seed=1000)
    assert not torch.equal(m1, m3), 'seed does not select the slice set'
    # semantics: it is the same statistic on a seed-selected slice draw
    g = torch.Generator(device='cpu').manual_seed(999)
    A = torch.randn(12, 32, generator=g)
    A = A / A.norm(p=2, dim=0)
    want = EppsPulley()(x @ A).mean()
    assert torch.allclose(m1, want, atol=1e-6)
    # and the training stream is exactly where it would have been
    y1 = sl(x)
    fresh = SlicingUnivariateTest(EppsPulley(), num_slices=32)
    assert torch.equal(y1, fresh(x))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no GPU on this node')
def test_gpu_smoke_null_statistic_and_finite_loss():
    """CUDA generators draw a different (equally valid) slice stream than CPU
    ones, so cross-device bitwise comparison is meaningless; the device-
    independent check is the exact null-mean law plus loss finiteness."""
    sig = build_sigreg().to(DEV)
    ep = sig.univariate_test
    want = float((ep.weights * (1.0 - ep.phi ** 2)).sum().cpu())
    torch.manual_seed(4747)
    vals = [float(sig(torch.randn(512, 16, device=DEV))) for _ in range(50)]
    got = sum(vals) / len(vals)
    assert got == pytest.approx(want, rel=0.25), (
        f'GPU null statistic {got:.4f} far from exact mean {want:.4f}')
    proj = torch.randn(2, 128, 16, device=DEV)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        total, _ = lejepa_loss(proj, sig, lamb=0.02)
    assert torch.isfinite(total).all()


# ---------------------------- 7. the whitening-arm statistic (Moments, k_max=2)

from src.lejepa import MomentsPerView  # noqa: E402
from lejepa.univariate.moments import Moments as PinnedMoments  # noqa: E402


def test_moments_matches_pinned_bitwise_on_2d():
    ours, pinned = MomentsPerView(k_max=2), PinnedMoments(k_max=2)
    x = _x((256, 9), seed=51)
    assert torch.equal(ours(x), pinned(x))
    x2 = _x((64, 3), seed=52).requires_grad_(True)
    x3 = x2.detach().clone().requires_grad_(True)
    ours(x2).sum().backward()
    pinned(x3).sum().backward()
    assert torch.equal(x2.grad, x3.grad)


def test_moments_3d_is_the_per_view_stack_of_2d_calls():
    ours, pinned = MomentsPerView(k_max=2), PinnedMoments(k_max=2)
    x = _x((3, 128, 5), seed=53)
    got = ours(x)
    want = torch.stack([pinned(x[v]) for v in range(3)])
    assert got.shape == (3, 5)
    assert torch.equal(got, want)


def test_moments_is_blind_to_the_shape_dial_but_not_to_whitening():
    """The whitening arm's defining property: k_max=2 constrains mean and
    variance only, so the bimodal deformation at fixed mean/variance is
    invisible while shift/scale violations fire."""
    m = MomentsPerView(k_max=2)
    g = torch.Generator().manual_seed(54)
    z = torch.randn(8192, 1, generator=g)
    sign = torch.where(torch.rand(8192, 1, generator=g) < 0.5, -1.0, 1.0)
    bimod = sign * 0.97 + math.sqrt(1 - 0.97 ** 2) * torch.randn(
        8192, 1, generator=g)
    null, bim = float(m(z)), float(m(bimod))
    shifted, scaled = float(m(z + 1.0)), float(m(z * 2.0))
    assert abs(bim - null) < 0.05, f'moment test SAW the dial: {null} vs {bim}'
    assert shifted > 10 * null and scaled > 3 * null


def test_build_sigreg_statistic_knob():
    sig_ep = build_sigreg()
    sig_mt = build_sigreg(statistic='moments')
    assert isinstance(sig_ep.univariate_test, EppsPulley)
    assert isinstance(sig_mt.univariate_test, MomentsPerView)
    assert sig_mt.univariate_test.k_max == 2
    proj = _x((2, 64, 16), seed=55)
    total, parts = lejepa_loss(proj, sig_mt, lamb=0.02)
    assert torch.isfinite(total).all()
    with pytest.raises(AssertionError):
        build_sigreg(statistic='cdf')
