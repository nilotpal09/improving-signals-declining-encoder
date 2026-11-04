"""Unit battery for the LeJEPA explicit-latent machinery (src/lejepa_data.py).

The study trains on gun events whose per-event latent z is drawn from a
CONTROLLED distribution (the non-Gaussianity dial) and, for the OU pilot arm,
renders view pairs from an explicitly rho-correlated latent pair
z' = rho*z + sqrt(1-rho^2)*eta — the theorem paper's Eq. (1). The machinery
must (a) produce exactly zero-mean unit-variance marginals at every dial
point (the guard fixed in advance: only SHAPE moves), (b) inject the drawn z
through the same Eq.-11 reconstruction the z-conditioning study transcribed
(guarded bitwise by tests/test_shared_z_reconstruction.py), and (c) stay
deterministic per batch without perturbing the global RNG stream. What these
tests protect:

  1. THE DIAL MOVES SHAPE ONLY. Every distribution (gaussian, bimodal(mu),
     uniform, laplace) has mean 0 and variance 1 to statistical tolerance,
     and the SHAPE actually changes: fourth moments match the closed forms
     (gaussian 3; bimodal mu^4 + 6 mu^2 s^2 + 3 s^4 with s^2 = 1 - mu^2;
     uniform 1.8; laplace 6). A dial that silently rescales variance would
     confound every dial curve.
  2. THE OU CHANNEL IS THE THEOREM'S. corr(z, z') = rho, stationary unit
     marginal for z'.
  3. THE INJECTION IS EQ. 11. _lp_from_z must be exactly invertible back to
     the injected z through the library's own stats (the same identity the
     shared-z study pinned).
  4. DETERMINISM AND ISOLATION. Same batch -> bitwise-same images; the draw
     comes from a private seeded Generator.
  5. THE GUARDS. The explicit-latent collator refuses configurations it was
     not built for (multi-particle guns, shift dial, dose, blur, blend,
     single-view mode, OU with V > 2) instead of silently mis-rendering.

Run: python -m pytest tests/test_lejepa_data.py -q
"""

import math
import os
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import pytest
import torch

torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from src.lejepa_data import (  # noqa: E402
    sample_z, ou_partner, LeJEPALatentCollator)
from src.masks.multiblock import MaskCollator as MBMaskCollator  # noqa: E402
from src.datasets.calo_flash import CaloNorm  # noqa: E402

DEV = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# The study's gun geometry (single particle, fixed position, narrow E band)
CALO_CFG = dict(N_cells_x=64, N_cells_y=64, N_cells_z=4, N_spots_per_layer=1000,
                Z=82, cell_e_threshold=0.0, depth=20.0, height=10.0, width=10.0)
GEN_CFG = dict(kind='gun', E_range=[20000.0, 40000.0], N_range=[1, 1],
               x_range=[0.0, 0.0], y_range=[0.0, 0.0], pad_value=float('nan'))


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


def _collator(latents, gen_cfg=None, calo_cfg=None, sample_target=True):
    base = MBMaskCollator(input_size=(64, 64), patch_size=4,
                          pred_mask_scale=(0.15, 0.2), enc_mask_scale=(0.85, 1.0),
                          aspect_ratio=(0.75, 1.5), nenc=1, npred=4,
                          allow_overlap=False, min_keep=10)
    ccfg = dict(calo_cfg or CALO_CFG)
    ccfg['lejepa_latents'] = latents
    return LeJEPALatentCollator(
        base_collator=base, sample_target=sample_target, device=DEV,
        calo_cfg=ccfg, generator_cfg=dict(gen_cfg or GEN_CFG), norm=CaloNorm())


# --------------------------------------------- 1. the dial moves shape only

@pytest.mark.parametrize('dist,mu,kurt', [
    ('gaussian', None, 3.0),
    ('bimodal', 0.0, 3.0),
    ('bimodal', 0.4, 0.4**4 + 6 * 0.4**2 * (1 - 0.4**2) + 3 * (1 - 0.4**2)**2),
    ('bimodal', 0.97, 0.97**4 + 6 * 0.97**2 * (1 - 0.97**2) + 3 * (1 - 0.97**2)**2),
    ('uniform', None, 1.8),
    ('laplace', None, 6.0),
])
def test_sample_z_zero_mean_unit_variance_and_the_right_shape(dist, mu, kurt):
    z = sample_z(200_000, dist=dist, mu=mu, generator=_gen(1))
    assert z.shape == (200_000, 2)
    assert float(z.mean()) == pytest.approx(0.0, abs=0.02)
    assert float(z.var(dim=0).mean()) == pytest.approx(1.0, rel=0.02)
    m4 = float((z ** 4).mean())
    tol = 0.05 if dist != 'laplace' else 0.12   # heavy tail, slower convergence
    assert m4 == pytest.approx(kurt, rel=tol), f'{dist}(mu={mu}) m4={m4:.3f}'


def test_sample_z_bimodal_is_actually_bimodal_at_high_mu():
    z = sample_z(100_000, dist='bimodal', mu=0.97, generator=_gen(2))
    # at mu=0.97, sigma=0.243: essentially no mass near zero
    frac_near_zero = float((z.abs() < 0.3).float().mean())
    assert frac_near_zero < 0.02


def test_sample_z_deterministic_per_seed_and_stream_isolated():
    torch.manual_seed(999)
    before = torch.get_rng_state()
    a = sample_z(64, dist='bimodal', mu=0.7, generator=_gen(5))
    assert torch.equal(torch.get_rng_state(), before), (
        'sample_z consumed the global RNG stream')
    b = sample_z(64, dist='bimodal', mu=0.7, generator=_gen(5))
    assert torch.equal(a, b)


def test_sample_z_rejects_unknown_dist_and_bimodal_without_mu():
    with pytest.raises(AssertionError):
        sample_z(8, dist='cauchy', generator=_gen(0))
    with pytest.raises(AssertionError):
        sample_z(8, dist='bimodal', mu=None, generator=_gen(0))


# ------------------------------------------------- 2. the OU channel

def test_ou_partner_correlation_and_stationary_marginal():
    z = sample_z(200_000, dist='gaussian', generator=_gen(7))
    zp = ou_partner(z, rho=0.9, generator=_gen(8))
    assert zp.shape == z.shape
    for d in range(2):
        c = float(torch.corrcoef(torch.stack([z[:, d], zp[:, d]]))[0, 1])
        assert c == pytest.approx(0.9, abs=0.005)
    assert float(zp.mean()) == pytest.approx(0.0, abs=0.02)
    assert float(zp.var(dim=0).mean()) == pytest.approx(1.0, rel=0.02)


def test_ou_partner_rho_one_is_identity_and_rho_range_guarded():
    z = sample_z(32, dist='gaussian', generator=_gen(9))
    assert torch.equal(ou_partner(z, rho=1.0, generator=_gen(10)), z)
    with pytest.raises(AssertionError):
        ou_partner(z, rho=1.2, generator=_gen(10))
    with pytest.raises(AssertionError):
        ou_partner(z, rho=-0.1, generator=_gen(10))


# --------------------------------------- 3. the injection is Eq. 11, invertible

def test_lp_from_z_inverts_back_to_the_injected_z():
    coll = _collator({'dist': 'gaussian', 'rho': 1.0})
    B = 64
    p_E, p_x, p_y = coll._generate_truth(B, seed=1234)
    z_ev = sample_z(B, dist='bimodal', mu=0.9, generator=_gen(11))
    lp = coll._lp_from_z(p_E, long_seed=555, z_ev=z_ev)
    # invert with the library's own stats (the probes' z-target convention)
    A = (torch.log(lp['alpha']) - lp['mean_ln_alpha']) / lp['sigma_ln_alpha']
    Tt = (torch.log(lp['T']) - lp['mean_ln_T']) / lp['sigma_ln_T']
    rho = lp['rho_ln_T_ln_alpha']
    a = torch.sqrt((1 + rho).clamp(min=0))
    b = torch.sqrt((1 - rho).clamp(min=0))
    z1 = (Tt + A) / (2 ** 0.5 * a)
    z2 = (Tt - A) / (2 ** 0.5 * b)
    got = torch.stack([z1, z2], dim=1).cpu()
    assert torch.allclose(got, z_ev.to(got.dtype), atol=1e-4), (
        f'max dev {(got - z_ev).abs().max():.2e}')


# ------------------------------------------- 4. determinism and full pipeline

def test_collator_call_is_deterministic_and_pair_shaped():
    coll = _collator({'dist': 'gaussian', 'rho': 0.9})
    batch = [0, 1, 2, 3]
    udata1, me1, mp1 = coll(batch)
    z_first = coll._last_z.clone()
    udata2, me2, mp2 = coll(batch)
    assert len(udata1) >= 2
    assert udata1[0].shape == udata1[1].shape == (4, 4, 64, 64)
    assert torch.equal(udata1[0], udata2[0]) and torch.equal(udata1[1], udata2[1])
    assert torch.equal(z_first, coll._last_z)
    assert torch.isfinite(udata1[0]).all() and torch.isfinite(udata1[1]).all()
    # the two views are different renderings (different sim seeds)
    assert not torch.equal(udata1[0], udata1[1])


def test_collator_last_z_shares_z_at_rho_one_and_not_below():
    c1 = _collator({'dist': 'gaussian', 'rho': 1.0})
    c1([0, 1, 2, 3])
    assert torch.equal(c1._last_z[0], c1._last_z[1])
    c2 = _collator({'dist': 'gaussian', 'rho': 0.9})
    c2([0, 1, 2, 3])
    assert not torch.equal(c2._last_z[0], c2._last_z[1])


def test_collator_v4_renders_four_views_sharing_z():
    coll = _collator({'dist': 'gaussian', 'rho': 1.0, 'views': 4})
    udata, _, _ = coll([0, 1])
    views = [u for u in udata if torch.is_tensor(u) and u.ndim == 4]
    assert len(views) == 4
    for v in views[1:]:
        assert v.shape == views[0].shape
        assert not torch.equal(v, views[0])
    assert coll._last_z.shape == (4, 2, 2)
    for v in range(1, 4):
        assert torch.equal(coll._last_z[v], coll._last_z[0])


def test_collator_dial_reaches_the_images():
    """mu=0.97 vs gaussian must change the rendered images for the same batch
    indices — the dial must actually reach the simulator."""
    ca = _collator({'dist': 'gaussian', 'rho': 1.0})
    cb = _collator({'dist': 'bimodal', 'mu': 0.97, 'rho': 1.0})
    ua, _, _ = ca([0, 1, 2, 3])
    ub, _, _ = cb([0, 1, 2, 3])
    assert not torch.equal(ua[0], ub[0])


# ----------------------------------------------------------- 5. the guards

def test_collator_refuses_configs_it_was_not_built_for():
    # (multi-particle guns are ALLOWED since 2026-08-17 — per-event z;
    #  see section 8 below)
    with pytest.raises(AssertionError):   # single-view mode
        _collator({'dist': 'gaussian', 'rho': 1.0}, sample_target=False)([0, 1])
    with pytest.raises(AssertionError):   # OU with more than two views
        _collator({'dist': 'gaussian', 'rho': 0.9, 'views': 4})([0, 1])
    with pytest.raises(AssertionError):   # shift dial not supported here
        _collator({'dist': 'gaussian', 'rho': 1.0},
                  calo_cfg=dict(CALO_CFG, pair_shift_max=1.0))([0, 1])
    with pytest.raises(AssertionError):   # dose mode not supported here
        c = _collator({'dist': 'gaussian', 'rho': 1.0})
        c.sample_target_prob = 0.5
        c([0, 1])
    with pytest.raises(AssertionError):   # frozen alpha/T contradicts the dial
        _collator({'dist': 'gaussian', 'rho': 1.0},
                  calo_cfg=dict(CALO_CFG, frozen_alpha_t=[2.0, 0.5]))([0, 1])


# ------------------------- 6. quantile maps and the copula-OU channel

from src.lejepa_data import quantile_fn, gaussian_scores_to_z  # noqa: E402


@pytest.mark.parametrize('dist,mu', [('gaussian', None), ('bimodal', 0.4),
                                     ('bimodal', 0.97), ('uniform', None),
                                     ('laplace', None)])
def test_quantile_map_preserves_the_dial_guarantees(dist, mu):
    """z = Q(Phi(g)) for g ~ N(0,1) must have the SAME distribution as
    sample_z's direct draw: zero mean, unit variance, matching 4th moment."""
    g = torch.randn(300_000, 2, generator=_gen(61))
    z = gaussian_scores_to_z(g, dist=dist, mu=mu)
    assert float(z.mean()) == pytest.approx(0.0, abs=0.02)
    assert float(z.var(dim=0).mean()) == pytest.approx(1.0, rel=0.02)
    ref = sample_z(300_000, dist=dist, mu=mu, generator=_gen(62))
    assert float((z ** 4).mean()) == pytest.approx(
        float((ref ** 4).mean()), rel=0.08)


def test_gaussian_scores_pass_through_unchanged_for_gaussian_dist():
    """Q_gaussian = identity BITWISE (no Phi/Phi^-1 roundtrip), so the
    rho < 1 gaussian channel is exactly the pilot's ou_partner channel."""
    g = torch.randn(64, 2, generator=_gen(63))
    assert gaussian_scores_to_z(g, dist='gaussian') is g


def test_quantile_fn_is_monotone_and_inverts_the_bimodal_cdf():
    q = quantile_fn('bimodal', mu=0.9)
    p = torch.linspace(0.001, 0.999, 999, dtype=torch.float64)
    x = q(p)
    assert (x[1:] > x[:-1]).all()
    sigma = (1 - 0.9 ** 2) ** 0.5
    cdf = 0.5 * torch.special.ndtr((x - 0.9) / sigma) + \
        0.5 * torch.special.ndtr((x + 0.9) / sigma)
    assert torch.allclose(cdf, p, atol=1e-9)


def test_copula_channel_is_stationary_and_rho_correlated_off_gaussian():
    """The grid's OU arm at non-Gaussian dial points: both marginals keep
    the exact dial distribution (stationarity) and the gaussian scores
    carry correlation rho."""
    g = torch.randn(200_000, 2, generator=_gen(64))
    gp = ou_partner(g, rho=0.9, generator=_gen(65))
    z = gaussian_scores_to_z(g, dist='bimodal', mu=0.9)
    zp = gaussian_scores_to_z(gp, dist='bimodal', mu=0.9)
    for t in (z, zp):
        assert float(t.mean()) == pytest.approx(0.0, abs=0.02)
        assert float(t.var(dim=0).mean()) == pytest.approx(1.0, rel=0.02)
    m4 = 0.9**4 + 6 * 0.9**2 * (1 - 0.9**2) + 3 * (1 - 0.9**2)**2
    assert float((zp ** 4).mean()) == pytest.approx(m4, rel=0.05)
    c = float(torch.corrcoef(torch.stack([g[:, 0], gp[:, 0]]))[0, 1])
    assert c == pytest.approx(0.9, abs=0.005)
    # the z-space correlation exists too (weaker than rho, monotone map)
    cz = float(torch.corrcoef(torch.stack([z[:, 0], zp[:, 0]]))[0, 1])
    assert 0.5 < cz < 0.95


def test_collator_supports_rho_below_one_at_nongaussian_dial_points():
    coll = _collator({'dist': 'bimodal', 'mu': 0.9, 'rho': 0.9})
    udata, _, _ = coll([0, 1, 2, 3])
    assert coll._last_z.shape == (2, 4, 2)
    assert not torch.equal(coll._last_z[0], coll._last_z[1])
    udata2, _, _ = coll([0, 1, 2, 3])
    assert torch.equal(udata[0], udata2[0])


# ------------------------------------------- 7. single-view mode (MAE dial)

def test_collator_single_view_mode_for_mae():
    coll = _collator({'dist': 'bimodal', 'mu': 0.97, 'rho': 1.0, 'views': 1},
                     sample_target=False)
    out = coll([0, 1, 2, 3])
    imgs = out[0] if torch.is_tensor(out[0]) else out[0][0]
    views = [u for u in (out[0] if isinstance(out[0], (list, tuple)) else out)
             if torch.is_tensor(u) and u.ndim == 4]
    assert len(views) == 1
    assert views[0].shape == (4, 4, 64, 64)
    assert coll._last_z.shape == (1, 4, 2)
    out2 = coll([0, 1, 2, 3])
    v2 = [u for u in (out2[0] if isinstance(out2[0], (list, tuple)) else out2)
          if torch.is_tensor(u) and u.ndim == 4]
    assert torch.equal(views[0], v2[0])


def test_collator_single_view_requires_views_one_and_vice_versa():
    with pytest.raises(AssertionError):   # paired mode with views: 1
        _collator({'dist': 'gaussian', 'views': 1}, sample_target=True)([0, 1])
    with pytest.raises(AssertionError):   # single-view mode with views: 2
        _collator({'dist': 'gaussian', 'views': 2}, sample_target=False)([0, 1])


# ---------------------- 8. multi-particle rigs: per-event z for all particles

def test_lp_from_z_shares_the_event_z_across_all_particles():
    """Rig-hardening study: with N particles per event, every
    real particle must carry ITS EVENT'S z through Eq. 11 — invert per
    particle and compare to the event draw."""
    gen_cfg = dict(GEN_CFG, N_range=[5, 5], x_range=[-4.0, 4.0],
                   y_range=[-4.0, 4.0])
    coll = _collator({'dist': 'gaussian', 'rho': 1.0}, gen_cfg=gen_cfg)
    B, P = 16, 5
    p_E, p_x, p_y = coll._generate_truth(B, seed=777)
    assert p_E.shape == (B, P)
    z_ev = sample_z(B, dist='bimodal', mu=0.7, generator=_gen(71))
    lp = coll._lp_from_z(p_E, long_seed=888, z_ev=z_ev)
    A = (torch.log(lp['alpha']) - lp['mean_ln_alpha']) / lp['sigma_ln_alpha']
    Tt = (torch.log(lp['T']) - lp['mean_ln_T']) / lp['sigma_ln_T']
    rho = lp['rho_ln_T_ln_alpha']
    a = torch.sqrt((1 + rho).clamp(min=0))
    b = torch.sqrt((1 - rho).clamp(min=0))
    z1 = ((Tt + A) / (2 ** 0.5 * a)).reshape(B, P).cpu()
    z2 = ((Tt - A) / (2 ** 0.5 * b)).reshape(B, P).cpu()
    for p in range(P):
        assert torch.allclose(z1[:, p], z_ev[:, 0].to(z1.dtype), atol=1e-4)
        assert torch.allclose(z2[:, p], z_ev[:, 1].to(z2.dtype), atol=1e-4)


def test_collator_renders_multiparticle_shared_z_pairs():
    gen_cfg = dict(GEN_CFG, N_range=[5, 5], x_range=[-4.0, 4.0],
                   y_range=[-4.0, 4.0])
    coll = _collator({'dist': 'bimodal', 'mu': 0.9, 'rho': 1.0},
                     gen_cfg=gen_cfg)
    udata, _, _ = coll([0, 1, 2, 3])
    assert udata[0].shape == udata[1].shape == (4, 4, 64, 64)
    assert torch.isfinite(udata[0]).all()
    assert coll._last_z.shape == (2, 4, 2)
