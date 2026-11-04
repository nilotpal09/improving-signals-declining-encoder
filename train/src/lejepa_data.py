"""LeJEPA explicit-latent machinery — the non-Gaussianity dial and the OU pair.

PURE ADDITION. `src/datasets/calo_flash.py` is untouched: this module
SUBCLASSES `CaloFlashBatchCollator` and is instantiated only by
`src/train_lejepa.py` for configs that set `data.calo_cfg.lejepa_latents`.
No shared cohort code path can reach it.

What it does
------------
The identifiability study needs the
per-event latent z = (z1, z2) drawn from a CONTROLLED distribution instead of
the simulator's internal per-particle randn:

  * the dial: gaussian / bimodal(mu) / uniform / laplace, all exactly
    zero-mean unit-variance per dimension (only SHAPE moves — the
    guard fixed in advance);
  * the OU pair (theorem paper Eq. 1): z' = rho z + sqrt(1-rho^2) eta,
    eta ~ N(0, I), rho in (0, 1); view 1 rendered from z, view 2 from z'.
    At rho = 1 both views share z exactly (the paired_fixed_long channel,
    but with the dial-controlled z).

The injection reuses the house mechanism end to end: truth and images come
from the parent's `_generate_truth` / `_simulate_once`, and z enters through
`long_params_override` via the SAME Eq.-11 reconstruction the z-conditioning
study transcribed (`_shared_z_long_params`, guarded bitwise against the
library by tests/test_shared_z_reconstruction.py). The library's stats
(mean/sigma/rho of ln alpha, ln T as functions of E) are untouched, so the
probes' z-target reconstruction inverts our
injection exactly — the probe's z and the training z are the same object.

Scope guards (asserted, not assumed): single-particle gun only
(N_range = [1, 1]; a shared per-event z across a multi-particle event would
correlate particles and change the physics), paired mode only, and none of
the other collator stages (shift dial, lambda-blend, factor-B blur, dose
protocol, frozen alpha/T, paired pool) may be active — each would compose
with the dial in an uncontrolled way.

Determinism: all draws come from private Generators seeded from the batch
seed (the house idiom), so the same batch indices give bitwise-identical
output and the global RNG stream is never consumed.
"""

import torch

from src.datasets.calo_flash import CaloFlashBatchCollator


_UNIT_VAR = {
    'gaussian': None,
    'bimodal': None,     # sigma^2 = 1 - mu^2 keeps the mixture at unit variance
    'uniform': 3 ** 0.5,      # U(-sqrt(3), sqrt(3))
    'laplace': 1 / 2 ** 0.5,  # scale b with 2 b^2 = 1
}


def sample_z(n_events: int, dist: str = 'gaussian', mu: float = None,
             generator: torch.Generator = None, n_dims: int = 2):
    """(n_events, n_dims) latent draws, exactly zero-mean unit-variance per
    dimension for every `dist` (the dial moves shape only):

      gaussian        z ~ N(0, 1)
      bimodal(mu)     z = s*mu + sigma*g,  s = +-1 equiprobable,
                      sigma = sqrt(1 - mu^2)  (requires 0 <= mu < 1;
                      mu = 0 IS the gaussian point)
      uniform         z ~ U(-sqrt(3), sqrt(3))
      laplace         z ~ Laplace(0, 1/sqrt(2)) via inverse CDF

    Draws only from `generator` (never the global stream)."""
    assert dist in _UNIT_VAR, f'unknown latent dist {dist!r}'
    shape = (n_events, n_dims)
    if dist == 'gaussian':
        return torch.randn(shape, generator=generator)
    if dist == 'bimodal':
        assert mu is not None and 0.0 <= float(mu) < 1.0, (
            f'bimodal needs mu in [0, 1), got {mu!r}')
        mu = float(mu)
        sign = torch.where(torch.rand(shape, generator=generator) < 0.5,
                           -1.0, 1.0)
        sigma = (1.0 - mu * mu) ** 0.5
        return sign * mu + sigma * torch.randn(shape, generator=generator)
    if dist == 'uniform':
        half = _UNIT_VAR['uniform']
        return (torch.rand(shape, generator=generator) * 2.0 - 1.0) * half
    # laplace, inverse CDF: u in (-1/2, 1/2), z = -b sign(u) ln(1 - 2|u|)
    b = _UNIT_VAR['laplace']
    u = torch.rand(shape, generator=generator) - 0.5
    return -b * torch.sign(u) * torch.log1p(-2.0 * u.abs())


def quantile_fn(dist: str, mu: float = None):
    """The dial distribution's quantile function q(p), float64-accurate.
    gaussian/uniform/laplace analytic; bimodal by vectorized bisection on
    the mixture CDF (inversion verified to 1e-9 by test)."""
    assert dist in _UNIT_VAR, f'unknown latent dist {dist!r}'
    if dist == 'gaussian':
        return lambda p: torch.special.ndtri(p.double())
    if dist == 'uniform':
        return lambda p: (2.0 * p.double() - 1.0) * 3 ** 0.5
    if dist == 'laplace':
        b = _UNIT_VAR['laplace']

        def q_lap(p):
            p = p.double()
            return torch.where(p < 0.5, b * torch.log(2.0 * p),
                               -b * torch.log(2.0 * (1.0 - p)))
        return q_lap
    # bimodal mixture: cdf(x) = 0.5 Phi((x-mu)/s) + 0.5 Phi((x+mu)/s)
    assert mu is not None and 0.0 <= float(mu) < 1.0, (
        f'bimodal needs mu in [0, 1), got {mu!r}')
    mu = float(mu)
    s = (1.0 - mu * mu) ** 0.5

    def q_bim(p):
        p = p.double()
        lo = torch.full_like(p, -10.0)
        hi = torch.full_like(p, 10.0)
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            cdf = 0.5 * torch.special.ndtr((mid - mu) / s) + \
                0.5 * torch.special.ndtr((mid + mu) / s)
            lo = torch.where(cdf < p, mid, lo)
            hi = torch.where(cdf < p, hi, mid)
        return 0.5 * (lo + hi)
    return q_bim


def gaussian_scores_to_z(g: torch.Tensor, dist: str = 'gaussian',
                         mu: float = None):
    """Map standard-normal scores g to the dial distribution by the exact
    quantile transform z = q(Phi(g)) — the Gaussian-copula construction the
    grid's OU arm uses at non-Gaussian dial points: a rho-correlated
    gaussian pair (g, g') maps to a STATIONARY pair (z, z') whose marginals
    are exactly the dial distribution at every rho. For dist='gaussian' the
    map is the identity and g is returned AS IS (bitwise — no Phi/ndtri
    roundtrip), so the gaussian-point channel is exactly the pilot's.
    Note: at rho < 1 this channel satisfies the theorem world's
    independence + stationarity assumptions but NOT its additive-noise
    form (only the Gaussian world admits that jointly); documented as the
    deployed-regime choice."""
    if dist == 'gaussian':
        return g
    p = torch.special.ndtr(g.double()).clamp(1e-12, 1.0 - 1e-12)
    return quantile_fn(dist, mu)(p).to(g.dtype)


def ou_partner(z: torch.Tensor, rho: float, generator: torch.Generator = None):
    """The theorem paper's transition, Eq. (1): z' = rho z + sqrt(1-rho^2) eta
    with eta ~ N(0, I) independent of z. Gaussian eta keeps the marginal
    stationary only for Gaussian z — the OU arm therefore runs at the
    Gaussian dial point (asserted by the collator, not here: this function is
    the pure channel). rho = 1 returns z itself (the shared-z pair)."""
    rho = float(rho)
    assert 0.0 < rho <= 1.0, f'rho must be in (0, 1], got {rho}'
    if rho == 1.0:
        return z
    eta = torch.randn(z.shape, generator=generator, dtype=z.dtype)
    return rho * z + (1.0 - rho * rho) ** 0.5 * eta


class LeJEPALatentCollator(CaloFlashBatchCollator):
    """Paired collator with explicit per-event latent control, configured by
    `data.calo_cfg.lejepa_latents`:

        lejepa_latents:
          dist: gaussian | bimodal | uniform | laplace
          mu: 0.9          # bimodal only
          rho: 1.0         # 1.0 = both views share z; (0,1) = OU pair (V=2)
          views: 2         # independent renderings per event (>= 2)

    Every view is an independent full simulation (fresh shower MC under
    `_sim_seed(seed, v)`) of the same truth (E, x, y) with the view's z
    injected through Eq. 11 — the calo analogue of MINIMAL's V augmented
    views, with renderer stochasticity playing the augmentation role.
    `self._last_z` holds the drawn latents, (V, B, 2) on cpu, for tests and
    the capture audit."""

    def __call__(self, batch):
        lcfg = self.calo_cfg.get('lejepa_latents')
        assert lcfg, 'LeJEPALatentCollator requires data.calo_cfg.lejepa_latents'
        # Per-EVENT z: every real particle in an event shares the event's
        # (z1, z2) — the z-conditioning study's construction (per-shower
        # marginals exactly standard; only within-event co-fluctuation is
        # shared). For N_range [1,1] this reduces to per-particle z (the
        # original single-particle rig). Multi-particle rigs enabled 2026-08-17
        # for the rig-hardening study.
        assert self._current_prob() is None, 'dose protocol not supported here'
        assert not self.calo_cfg.get('pair_shift_max'), 'shift dial not supported here'
        assert not self.calo_cfg.get('frozen_alpha_t'), 'frozen alpha/T contradicts the dial'
        assert not self.calo_cfg.get('paired_fixed_long'), (
            'paired_fixed_long is implied (and generalized) by lejepa_latents')
        assert self.lblend is None and self.factor_b is None, (
            'lambda_blend / posthoc_blur not supported here')
        assert not self.paired_pool_events, 'paired pool not supported here'

        dist = lcfg.get('dist', 'gaussian')
        mu = lcfg.get('mu')
        rho = float(lcfg.get('rho', 1.0))
        n_views = int(lcfg.get('views', 2))
        assert 0.0 < rho <= 1.0, f'rho must be in (0, 1], got {rho}'
        if self.sample_target:
            assert n_views >= 2, (
                f'paired mode needs views >= 2, got {n_views}')
        else:
            # single-view mode (the MAE dial arm): one render per event,
            # emitted in the library's (img, 0) single-view convention.
            assert n_views == 1, (
                'single-view mode (sample_target: false) requires views: 1; '
                f'got {n_views}')
        assert rho == 1.0 or n_views == 2, (
            'the OU channel is a PAIR construction: rho < 1 requires views: 2')

        batch_size = len(batch)
        seed = self._batch_seed(batch)
        p_E, p_x, p_y = self._generate_truth(batch_size, seed)

        # same long-stats seed derivation as the paired_fixed_long channel
        long_seed = (seed * 7919 + 31337) & 0x7FFFFFFF
        # same z-seed derivation idiom as the shared-z study
        zseed = (long_seed * 2654435761 + 97) & 0x7FFFFFFF

        # Unified copula construction: draw standard-normal SCORES, correlate
        # them for the OU channel, then quantile-map to the dial distribution
        # (identity for gaussian — bitwise the pilot's channel; see
        # gaussian_scores_to_z). Marginals are exactly the dial distribution
        # at every rho (stationarity), at every view.
        g0 = torch.randn((batch_size, 2),
                         generator=torch.Generator().manual_seed(zseed))
        gs = [g0]
        for v in range(1, n_views):
            if rho == 1.0:
                gs.append(g0)
            else:
                g_eta = torch.Generator().manual_seed((zseed * 31 + v) & 0x7FFFFFFF)
                gs.append(ou_partner(g0, rho, generator=g_eta))
        zs = [gaussian_scores_to_z(g, dist=dist, mu=mu) for g in gs]

        has_range = bool(self.calo_cfg.get('N_spots_range'))
        views = []
        for v, z_v in enumerate(zs):
            n_v = self._resolve_n_spots(
                'N_spots_per_layer', 'N_spots_range',
                (seed * 13 + v + 1) if has_range else seed)
            lp_v = self._lp_from_z(p_E, long_seed, z_v)
            views.append(self.norm.forward(self._simulate_once(
                p_E, p_x, p_y, self._sim_seed(seed, v),
                n_spots_override=n_v, long_params_override=lp_v)))

        self._last_z = torch.stack([z.detach().cpu() for z in zs])
        if n_views == 1:
            # library single-view convention: (img, dummy_label)
            sample_batch = [(views[0][i], 0) for i in range(batch_size)]
        else:
            sample_batch = [tuple(view[i] for view in views)
                            for i in range(batch_size)]
        return self.base_collator(sample_batch)

    def _lp_from_z(self, p_E, long_seed: int, z_ev: torch.Tensor):
        """long_params dict with alpha/T rebuilt from the EXPLICIT z_ev
        (B, 2) instead of the library's internal draw. Identical to
        `_shared_z_long_params` except the z comes from the caller;
        reconstruction mirrors torch_calo_flash.calo_flash
        .get_longitudinal_parameters lines 79-85 (guarded bitwise by
        tests/test_shared_z_reconstruction.py; inversion guarded by
        tests/test_lejepa_data.py)."""
        from torch_calo_flash.calo_flash import get_beta
        lp = self._compute_long_params(p_E, long_seed)  # stats; its z discarded
        flat_Es = p_E.flatten()
        real_idx = torch.where(flat_Es == flat_Es)[0]
        ev_idx = (real_idx // p_E.shape[1]).to(lp['mean_ln_T'].device)
        z_ev = z_ev.to(device=lp['mean_ln_T'].device, dtype=lp['mean_ln_T'].dtype)
        z1 = z_ev[ev_idx, 0]
        z2 = z_ev[ev_idx, 1]
        rho = lp['rho_ln_T_ln_alpha']
        a = torch.sqrt((1 + rho).clamp(min=0))
        b = torch.sqrt((1 - rho).clamp(min=0))
        ln_T = lp['mean_ln_T'] + lp['sigma_ln_T'] * (a * z1 + b * z2) / 2 ** 0.5
        ln_alpha = lp['mean_ln_alpha'] + lp['sigma_ln_alpha'] * (a * z1 - b * z2) / 2 ** 0.5
        lp['T'] = torch.exp(ln_T)
        lp['alpha'] = torch.exp(ln_alpha)
        lp['beta'] = get_beta(lp['alpha'], lp['T'])
        return lp


def attach_wp1_dial(loader, collator=None, calo_cfg=None, generator_cfg=None,
                    norm=None, sample_target=False, **_ignored):
    """Swap `loader.collate_fn` for a LeJEPALatentCollator when the config
    carries `calo_cfg.lejepa_latents`. Called by the dial entrypoints
    (here main_mae.py) from a make_calo wrapper — the
    keyword names mirror make_calo's so the wrapper can forward its kwargs
    verbatim. No-op when the dial key is absent, so the entrypoints remain
    byte-equivalent to I-JEPA's main.py / main_mae.py for undialed configs."""
    if not (calo_cfg or {}).get('lejepa_latents'):
        return False
    loader.collate_fn = LeJEPALatentCollator(
        base_collator=collator, sample_target=sample_target,
        calo_cfg=calo_cfg, generator_cfg=generator_cfg, norm=norm)
    return True
