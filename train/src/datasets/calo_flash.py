"""
PyCaloflash-backed calorimeter loader for i-JEPA / MAE training.

Batches are generated on the fly with `pycaloflash`.  The loader yields:
    ((imgs, labels), masks_enc, masks_pred)      -- standard mode
    ((ctx_imgs, tgt_imgs), masks_enc, masks_pred) -- sample_target=True mode

Images have shape (B, N_cells_z, N_cells_x, N_cells_y): all z-layers are used
as input channels.  The number of model input channels equals calo_cfg['N_cells_z'].

In sample_target mode the truth particles are simulated twice (different RNG
seeds), giving statistically independent detector responses for the same event.
The training loop detects this via the shape of udata[1].

For inference, call CaloFlashBatchCollator.generate_replicas() directly.
"""

from logging import getLogger
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

_logger = getLogger()


from torch_calo_flash.calorimeter import CaloBlock
from torch_calo_flash.calo_flash import get_longitudinal_parameters
from torch_calo_flash.generator import EventGenerator, JetEventGenerator

_DEFAULT_N_CELLS_Z = 8
_DEFAULT_N_CELLS_XY = 64


class CaloNorm:
    """Log-compress calorimeter cell energies for ViT training.

    forward : raw MeV  -> normalized (model input)
    inverse : normalized -> raw MeV  (for plotting / physics analysis)

    Modes
    -----
    'none'
        Identity — no normalization.
    'log1p_scale'  (default)
        y = log1p(x) / log1p(scale)
        Maps [0, scale] roughly to [0, 1] with log compression.
        scale (MeV) should be set near the observed max single-cell energy.
        Default 5500 MeV covers the ~5426 MeV max seen in profiling runs.
        Empty cells (below threshold) map to 0; hit cells land in ~[0.37, 1.0].
    'log1p_standardize'
        y = (log1p(x) - mean) / std
        Requires precomputed mean / std over log1p(cell_e).
        Set mean and std after running estimate_log1p_stats().
    """

    MODES = ('none', 'log1p_scale', 'log1p_standardize')

    def __init__(
        self,
        mode: str = 'log1p_scale',
        scale: float = 5500.0,
        mean: float = 0.0,
        std: float = 1.0,
    ):
        if mode not in self.MODES:
            raise ValueError(f'CaloNorm mode must be one of {self.MODES}, got {mode!r}')
        self.mode = mode
        self.scale = float(scale)
        self._log1p_scale = float(torch.log1p(torch.tensor(scale)))
        self.mean = float(mean)
        self.std = float(std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == 'none':
            return x
        x = torch.log1p(x)
        if self.mode == 'log1p_scale':
            return x / self._log1p_scale
        return (x - self.mean) / self.std  # log1p_standardize

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Return raw MeV from a normalized tensor."""
        if self.mode == 'none':
            return y
        if self.mode == 'log1p_scale':
            return torch.expm1(y * self._log1p_scale).clamp_(min=0.0)
        return torch.expm1(y * self.std + self.mean).clamp_(min=0.0)  # log1p_standardize


def _as_device(device):
    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device) if isinstance(device, str) else device
    if device.type == 'cuda' and not torch.cuda.is_available():
        device = torch.device('cpu')
    return device


def _default_calo_cfg(calo_cfg=None):
    default_cfg = {
        'Z': 82,
        'width': 10.0,
        'height': 10.0,
        'depth': 16.0,
        'N_cells_x': _DEFAULT_N_CELLS_XY,
        'N_cells_y': _DEFAULT_N_CELLS_XY,
        'N_cells_z': _DEFAULT_N_CELLS_Z,
        'N_spots_per_layer': 1000,
        'cell_e_threshold': 25.0,
    }
    if calo_cfg:
        default_cfg.update(calo_cfg)
    if default_cfg.get('height') is None:
        default_cfg['height'] = default_cfg['width']
    return default_cfg


def _default_generator_cfg(calo_cfg, generator_cfg=None):
    default_cfg = {
        'x_range': (-calo_cfg['width'] * 0.8 / 2, calo_cfg['width'] * 0.8 / 2),
        'y_range': (-calo_cfg['height'] * 0.8 / 2, calo_cfg['height'] * 0.8 / 2),
        'E_range': (1000.0, 100_000.0),
        'N_range': (4, 32),
        'pad_value': float('nan'),
    }
    if generator_cfg:
        default_cfg.update(generator_cfg)
    return default_cfg


def n_channels_from_cfg(calo_cfg: Optional[Dict[str, Any]] = None) -> int:
    """Return the number of image channels (= N_cells_z) for a given calo_cfg."""
    if calo_cfg and 'N_cells_z' in calo_cfg:
        return int(calo_cfg['N_cells_z'])
    return _DEFAULT_N_CELLS_Z


def img_size_from_cfg(calo_cfg: Optional[Dict[str, Any]] = None) -> Tuple[int, int]:
    """Return (N_cells_x, N_cells_y) — the spatial image dimensions.

    Raises ValueError if N_cells_x != N_cells_y, since the ViT patch embedding
    only supports square inputs.
    """
    nx = int((calo_cfg or {}).get('N_cells_x', _DEFAULT_N_CELLS_XY))
    ny = int((calo_cfg or {}).get('N_cells_y', _DEFAULT_N_CELLS_XY))
    if nx != ny:
        raise ValueError(
            f'N_cells_x ({nx}) != N_cells_y ({ny}): the ViT patch embedding '
            'requires square images. Use equal cell counts or add a non-square ViT.')
    return nx, ny


class PostHocFactorB:
    """Gated post-hoc 'factor B' stage — the conditions-study coupling dial.

    Three stages, in this order:

      1. a parametric transform of the FINISHED DETECTOR IMAGE (raw MeV, before
         the model-input normalizer) — the DECOUPLED dataset-level factor. Its
         parameter plays the role alpha/T plays for factor A, except that it is
         a transform of an already-drawn image rather than a parameter of the
         draw.
      2. the normalizer, exactly where it sits in every other calo path.
      3. an INDEPENDENT per-view pixel-noise stage, added AFTER the transform,
         its amplitude calibrated (gate G1) against the within-pair nuisance.

    Stage 3 is what the brief's construction rests on. Without it the within-pair
    spot difference passes through the same transform, so the residual
    within-pair noise statistics depend on the factor and factor B is secretly
    COUPLED — the construction trap named in the brief. Whether stage 3 actually
    cures that is MEASURED, not assumed: gate G2, `src/g2_coupling.py`. (The
    measured answer is that it cures the amplitude and spectral channels but not
    the variance-map one — see FLAGSHIP_READOUT.md §G2.)

    Stages 1 and 3 sit on opposite sides of the normalizer on purpose. The
    transform is physical, so it belongs on the energy image; the noise must be
    signal-INDEPENDENT, so it belongs after the log compression. Adding it in MeV
    would make its statistics scale with the local energy, and the local energy
    depends on the factor — which would smuggle the coupling back in. For the
    same reason the noise is not clipped at zero.

    `transform` picks the family (all are measured in `csv/g1_survey.csv`):

      'blur'  Gaussian lateral spread, parameter = width in cells (the brief's
              worked example; cannot reach factor A's imprint — see G1)
      'zoom'  isotropic magnification about the image centre
      'shift' rigid translation along the image diagonal, in cells
      'warp'  a fixed smooth displacement-field template, parameter = amplitude
      'tilt'  longitudinal reweighting exp(p*u_c), event energy preserved
      'gamma' pointwise contrast E -> E^p, event energy preserved

    `mode` sets the channel placement, i.e. which of the three arms this is:

      'shared'  -> INDIRECT: one parameter per event, the SAME for both views
                   (the factor varies across samples, never within a pair)
      'perview' -> DIRECT:   parameter drawn independently for each view
      'frozen'  -> ABSENT:   parameter == frozen_sigma everywhere. Defaults to
                   the MIDPOINT of sigma_range and is never zero — the absent
                   arm sees one setting of the transform, not no transform.

    Config (calo_cfg['posthoc_blur']); absent key == stage does not exist:

        posthoc_blur:
          mode: shared | perview | frozen
          transform: blur | zoom | shift | warp | tilt | gamma   # default blur
          sigma_range: [lo, hi]     # the factor's range; uniform draw
          frozen_sigma: <float>     # 'frozen' only; default = midpoint
          noise_sigma: <float>      # required, >= 0; normalized-image units
          kernel_radius: <int>      # blur only; default ceil(4*hi)
          warp_seed: <int>          # warp only; default 20260805
    """

    MODES = ('shared', 'perview', 'frozen')
    TRANSFORMS = ('blur', 'zoom', 'shift', 'warp', 'tilt', 'gamma')

    # Distinct odd multipliers so the sigma stream and the two noise streams
    # cannot collide with each other or with the collator's own derived seeds.
    _SIGMA_SALT = 2246822519
    _NOISE_SALT = 3266489917

    def __init__(self, cfg: Dict[str, Any]):
        mode = cfg.get('mode')
        if mode not in self.MODES:
            raise ValueError(
                f'posthoc_blur.mode must be one of {self.MODES}, got {mode!r}')
        self.mode = mode

        tf = cfg.get('transform', 'blur')
        if tf not in self.TRANSFORMS:
            raise ValueError(
                f'posthoc_blur.transform must be one of {self.TRANSFORMS}, '
                f'got {tf!r}')
        self.transform = tf
        self.warp_seed = int(cfg.get('warp_seed', 20260805))
        self._warp_cache = {}

        rng = cfg.get('sigma_range')
        if not rng or len(rng) != 2:
            raise ValueError('posthoc_blur.sigma_range must be [lo, hi]')
        self.sigma_lo, self.sigma_hi = float(rng[0]), float(rng[1])
        if not self.sigma_lo <= self.sigma_hi:
            raise ValueError(
                f'posthoc_blur.sigma_range must satisfy lo <= hi, got {rng}')
        if self.transform in ('blur', 'zoom', 'gamma') and self.sigma_lo <= 0.0:
            raise ValueError(
                f'posthoc_blur.sigma_range lower end must be > 0 for '
                f'transform={self.transform!r}, got {rng}')

        fs = cfg.get('frozen_sigma')
        self.frozen_sigma = (0.5 * (self.sigma_lo + self.sigma_hi)
                             if fs is None else float(fs))
        if self.mode == 'frozen' and self.frozen_sigma <= 0.0:
            raise ValueError(
                'posthoc_blur.frozen_sigma must be > 0 — the absent arm is '
                'frozen at a blur width, not at no blur')

        ns = cfg.get('noise_sigma')
        if ns is None or float(ns) < 0.0:
            raise ValueError(
                'posthoc_blur.noise_sigma is required and must be >= 0 — the '
                'independent noise stage is what decouples factor B')
        self.noise_sigma = float(ns)

        import math
        kr = cfg.get('kernel_radius')
        self.radius = int(kr) if kr is not None else int(
            math.ceil(4.0 * max(self.sigma_hi, self.frozen_sigma)))
        if self.radius < 1:
            raise ValueError('posthoc_blur.kernel_radius must be >= 1')

    # ------------------------------------------------------------- seeding

    def _sigma_seed(self, seed: int, view: int) -> int:
        # 'shared'/'frozen' ignore the view: one draw per event for both views.
        v = view if self.mode == 'perview' else 0
        return (int(seed) * self._SIGMA_SALT + 1315423911 * v + 7) & 0x7FFFFFFF

    def _noise_seed(self, seed: int, view: int) -> int:
        return (int(seed) * self._NOISE_SALT + 2654435761 * int(view) + 11) & 0x7FFFFFFF

    # ------------------------------------------------------------- the draws

    def draw_sigma(self, batch_size: int, seed: int, view: int,
                   device=None, dtype=torch.float32) -> torch.Tensor:
        """(B,) blur widths for this view. Own Generator: the global RNG stream
        is untouched, so B runs stay RNG-comparable to their own controls."""
        if self.mode == 'frozen':
            return torch.full((batch_size,), self.frozen_sigma,
                              device=device, dtype=dtype)
        g = torch.Generator().manual_seed(self._sigma_seed(seed, view))
        u = torch.rand(batch_size, generator=g, dtype=torch.float32)
        s = self.sigma_lo + (self.sigma_hi - self.sigma_lo) * u
        return s.to(device=device, dtype=dtype)

    def _kernels(self, sigma: torch.Tensor) -> torch.Tensor:
        x = torch.arange(-self.radius, self.radius + 1,
                         device=sigma.device, dtype=sigma.dtype)
        k = torch.exp(-0.5 * (x[None, :] / sigma[:, None]) ** 2)
        return k / k.sum(dim=1, keepdim=True)

    def blur(self, imgs: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Separable Gaussian blur with a PER-SAMPLE width, zero-padded.

        Written as an explicit shift-and-accumulate rather than a grouped conv:
        the summation order is then fixed by the loop, so the result does not
        depend on which cuDNN algorithm happens to be autotuned in. Zero padding
        is the physical choice — there is no calorimeter outside the block.
        """
        k = self._kernels(sigma.to(imgs.dtype))           # (B, 2R+1)
        R, K = self.radius, 2 * self.radius + 1
        w = k.view(k.shape[0], K, 1, 1, 1)
        out = imgs
        for dim, pad in ((3, (R, R, 0, 0)), (2, (0, 0, R, R))):
            xp = F.pad(out, pad)
            acc = torch.zeros_like(out)
            for j in range(K):
                acc = acc + w[:, j] * xp.narrow(dim, j, out.shape[dim])
            out = acc
        return out

    # ---------------------------------- the other transform families

    @staticmethod
    def _base_grid(B, H, W, device, dtype):
        ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
        xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        return torch.stack([gx, gy], dim=-1)[None].expand(B, H, W, 2)

    @staticmethod
    def _sample(x, flow):
        return F.grid_sample(x, flow, mode='bilinear', padding_mode='zeros',
                             align_corners=False)

    def zoom(self, x, p):
        B, C, H, W = x.shape
        g = self._base_grid(B, H, W, x.device, x.dtype)
        return self._sample(x, g / p.to(x.dtype)[:, None, None, None])

    def shift(self, x, p):
        B, C, H, W = x.shape
        g = self._base_grid(B, H, W, x.device, x.dtype)
        d = (p.to(x.dtype)[:, None, None, None] * (2.0 / W)) / (2 ** 0.5)
        return self._sample(x, g + d)

    def warp(self, x, p):
        B, C, H, W = x.shape
        key = (H, W, str(x.device), str(x.dtype))
        if key not in self._warp_cache:
            g = torch.Generator().manual_seed(self.warp_seed)
            f = torch.randn(1, 2, H, W, generator=g)
            f = self.blur(f, torch.tensor([6.0]))
            f = f / f.reshape(2, -1).norm(dim=1).mean() * (H * W) ** 0.5
            self._warp_cache[key] = f.to(device=x.device, dtype=x.dtype)
        f = self._warp_cache[key].expand(B, 2, H, W).permute(0, 2, 3, 1)
        g = self._base_grid(B, H, W, x.device, x.dtype)
        return self._sample(x, g + f * (p.to(x.dtype)[:, None, None, None]
                                        * (2.0 / W)))

    def tilt(self, x, p):
        C = x.shape[1]
        u = torch.linspace(-1, 1, C, device=x.device, dtype=x.dtype)
        w = torch.exp(p[:, None].to(x.dtype) * u[None, :])
        w = w / w.mean(dim=1, keepdim=True)
        y = x * w[:, :, None, None]
        t0 = x.sum(dim=(1, 2, 3), keepdim=True)
        t1 = y.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
        return y * (t0 / t1)

    def gamma(self, x, p):
        y = x.clamp_min(0).pow(p[:, None, None, None].to(x.dtype))
        t0 = x.sum(dim=(1, 2, 3), keepdim=True)
        t1 = y.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
        return y * (t0 / t1)

    def apply_transform(self, raw: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """The factor's action on the finished detector image (raw MeV)."""
        return getattr(self, 'blur' if self.transform == 'blur'
                       else self.transform)(raw, p)

    def add_noise(self, imgs: torch.Tensor, seed: int, view: int) -> torch.Tensor:
        """Per-view independent pixel noise, i.i.d. over every cell. Unclipped.
        Applied AFTER the normalizer, so its amplitude does not track the local
        energy (which would depend on the factor)."""
        if self.noise_sigma == 0.0:
            return imgs
        g = torch.Generator(device=imgs.device)
        g.manual_seed(self._noise_seed(seed, view))
        n = torch.randn(imgs.shape, generator=g, device=imgs.device,
                        dtype=imgs.dtype)
        return imgs + self.noise_sigma * n

    # ---------------------------------------------------------------- apply

    def apply_view(self, raw: torch.Tensor, norm, seed: int, view: int,
                   sigma: Optional[torch.Tensor] = None):
        """transform(raw) -> normalize -> add noise. Returns (imgs, param)."""
        if sigma is None:
            sigma = self.draw_sigma(raw.shape[0], seed, view,
                                    device=raw.device, dtype=raw.dtype)
        out = self.apply_transform(raw, sigma)
        out = out if norm is None else norm.forward(out)
        return self.add_noise(out, seed, view), sigma

    def apply_pair(self, raw_ctx: torch.Tensor, raw_tgt: torch.Tensor, norm,
                   seed: int):
        """Both views. Returns (ctx, tgt, param_ctx, param_tgt)."""
        ctx, s_c = self.apply_view(raw_ctx, norm, seed, 0)
        s_t = None if self.mode == 'perview' else s_c
        tgt, s_t = self.apply_view(raw_tgt, norm, seed, 1, sigma=s_t)
        return ctx, tgt, s_c, s_t


class LambdaBlend:
    """Gated λ-blend of the spot-generating profile — the conditions coupling dial.

    A fraction λ of the Monte-Carlo spots is thrown from the shower's true α/T
    longitudinal profile; the remaining 1−λ are thrown from a FIXED reference
    profile (the absent arm's frozen (4.456, 6.606)). `shoot()` gives every dart
    dE/N, so a k-dart simulation is an unbiased estimate of the same mean image
    for any k, and the blend is EXACT as a weighted sum of two ordinary
    simulations:

        view = λ·S(round(λN) darts, true α/T) + (1−λ)·R(N−round(λN) darts, ref)

      mean      λ·μ_αT + (1−λ)·μ_ref — the α/T-dependent part is scaled by λ, so
                the factor's imprint falls with λ and gate G1 re-matches it.
      noise     λ²·σ²/λ + (1−λ)²·σ²/(1−λ) = σ² — the total within-pair noise is
                preserved at every λ, so the pairing roof does not drift.
      coupling  only the λ-component's noise is α/T-shaped, so its share of the
                within-pair variance is λ. That is the dial.

    `width` re-matches the imprint by widening the α/T draw in log space about
    the prior median: ln a → ln a_med + w·(ln a − ln a_med), same for T. The
    per-λ value comes from gate G1 (conditions/csv/lblend_gates_d*.csv); w = 1
    is the natural prior, and λ = 1 with w = 1 reduces to the ordinary path.

    Config (calo_cfg['lambda_blend']); absent key == the stage does not exist:

        lambda_blend:
          lam: <float in (0, 1]>     # fraction of spots from the α/T profile
          width: <float >= 1>        # the α/T widening that re-matches G1
          ref_alpha_t: [4.456, 6.606]  # the fixed reference profile
    """

    ALPHA_MED, T_MED = 4.444, 6.637      # step-0 item-1, csv/item1_prior_span.csv
    _REF_SALT = 1900987

    def __init__(self, cfg: Dict[str, Any]):
        lam = cfg.get('lam')
        if lam is None or not 0.0 < float(lam) <= 1.0:
            raise ValueError(f'lambda_blend.lam must be in (0, 1], got {lam!r}')
        self.lam = float(lam)
        w = float(cfg.get('width', 1.0))
        if w < 1.0:
            raise ValueError(f'lambda_blend.width must be >= 1, got {w}')
        self.width = w
        ref = cfg.get('ref_alpha_t', [4.456, 6.606])
        if not ref or len(ref) != 2:
            raise ValueError('lambda_blend.ref_alpha_t must be [alpha, T]')
        self.ref_alpha_t = (float(ref[0]), float(ref[1]))

    def widen(self, lp: dict) -> dict:
        """Widen the α/T draw about the prior median in log space."""
        if self.width == 1.0:
            return lp
        from torch_calo_flash.calo_flash import get_beta
        out = dict(lp)
        am = torch.as_tensor(self.ALPHA_MED, device=lp['alpha'].device,
                             dtype=lp['alpha'].dtype)
        tm = torch.as_tensor(self.T_MED, device=lp['T'].device, dtype=lp['T'].dtype)
        a = torch.exp(torch.log(am) + self.width * (torch.log(lp['alpha']) - torch.log(am)))
        t = torch.exp(torch.log(tm) + self.width * (torch.log(lp['T']) - torch.log(tm)))
        out['alpha'], out['T'], out['beta'] = a, t, get_beta(a, t)
        out['mean_ln_alpha'] = torch.log(a)
        return out

    def split(self, n_darts: int):
        """(shaped darts, reference darts) at this λ."""
        ns = int(round(self.lam * int(n_darts)))
        return ns, int(n_darts) - ns

    def ref_seed(self, seed: int) -> int:
        """A spot seed for the reference component, well separated from the
        shaped component's, so the two halves are drawn independently."""
        return (int(seed) * self._REF_SALT + 41) & 0x7FFFFFFF


class VirtualCaloDataset(Dataset):
    """Index source so batch generation can happen in the collate function."""

    def __init__(self, n_events: int):
        super().__init__()
        self.n_events = int(n_events)

    def __len__(self):
        return self.n_events

    def __getitem__(self, idx):
        return int(idx)


class CaloFlashBatchCollator:
    """Generate a full pycaloflash batch and delegate masking to the base collator."""

    def __init__(
        self,
        base_collator,
        sample_target: bool = False,
        sample_target_prob: Optional[float] = None,
        sample_target_schedule: Optional[list] = None,
        paired_pool_events: Optional[int] = None,
        paired_pool_seed: int = 777,
        device=None,
        calo_cfg: Optional[Dict[str, Any]] = None,
        generator_cfg: Optional[Dict[str, Any]] = None,
        norm: Optional['CaloNorm'] = None,
    ):
        self.base_collator = base_collator
        self.sample_target = sample_target
        self.sample_target_prob = sample_target_prob
        # Protocol knobs (dose/protocol study). `sample_target_schedule` is a list
        # of [until_epoch, p] phases (same format as meta.checkpoint_schedule);
        # requires the train loop to set `self.epoch` each epoch (num_workers=0).
        # `paired_pool_events`: draw the paired samples from a FIXED pool of this
        # many pre-generated pairs instead of fresh resims (realism arm).
        self.sample_target_schedule = sample_target_schedule
        self.paired_pool_events = paired_pool_events
        self.paired_pool_seed = paired_pool_seed
        self.epoch = 0
        self._pool = None
        self._last_keep = None   # per-batch dose mask; read by the weighted-loss path
        self.device = _as_device(device)
        self.calo_cfg = _default_calo_cfg(calo_cfg=calo_cfg)
        self.generator_cfg = _default_generator_cfg(self.calo_cfg, generator_cfg=generator_cfg)
        self.n_channels = self.calo_cfg['N_cells_z']
        self.norm = norm if norm is not None else CaloNorm()

        # Gated factor-B stage (conditions flagship). No `posthoc_blur` key ->
        # None -> the stage does not exist and every pre-existing path is
        # byte-identical (asserted by tests/capture_batches.py).
        _pb = self.calo_cfg.get('posthoc_blur')
        self.factor_b = PostHocFactorB(_pb) if _pb else None
        self._last_sigma = None   # (sigma_ctx, sigma_tgt); read by tests/probes
        self._last_shift = None   # (B, 2) unit direction of the pair shift;
                                  # read by train.py when predictor_z_source
                                  # is 'pair_shift' (Study A shift dial)

        # Gated λ-blend of the spot-generating profile. No `lambda_blend` key ->
        # None -> every pre-existing path is byte-identical.
        _lb = self.calo_cfg.get('lambda_blend')
        self.lblend = LambdaBlend(_lb) if _lb else None
        if self.lblend is not None and self.calo_cfg.get('frozen_alpha_t'):
            raise ValueError(
                'lambda_blend and frozen_alpha_t are mutually exclusive: the '
                'blend needs a live alpha/T draw to blend away from')

        self._generator = None
        self._calorimeter = None
        # Diagnostic counters: track how many cells the defensive nan_to_num had to
        # neutralize. Stays at 0 if the data pipeline is clean. Reset_nan_counts()
        # to clear; nan_counts() to read. Logged occasionally by the training loop.
        self._n_nan_cells_total = 0       # cumulative NaN/Inf cells observed
        self._n_nan_batches_total = 0     # batches where >=1 cell was NaN/Inf
        self._n_batches_total = 0         # batches simulated (incl. clean ones)

    def _fork_devices(self):
        if self.device.type != 'cuda':
            return []
        if self.device.index is not None:
            return [self.device.index]
        return [torch.cuda.current_device()]

    def _build_backend(self):
        if self._generator is None:
            kind = self.generator_cfg.get('kind', 'gun')
            if kind == 'gun':
                self._generator = EventGenerator(self.generator_cfg, device=self.device)
            elif kind == 'jet':
                self._generator = JetEventGenerator(self.generator_cfg, device=self.device)
            else:
                raise ValueError(
                    f'unknown generator kind: {kind!r}. expected "gun" or "jet".')
        if self._calorimeter is None:
            self._calorimeter = CaloBlock(self.calo_cfg, device=self.device)

    @staticmethod
    def _batch_seed(batch):
        values = torch.as_tensor(batch, dtype=torch.long)
        weights = torch.arange(1, values.numel() + 1, dtype=torch.long)
        seed = int((values * weights).sum().item())
        return seed & 0x7FFFFFFF

    def _generate_truth(self, batch_size: int, seed: int):
        """Generate truth particles (p_E, p_x, p_y) reproducibly from seed."""
        self._build_backend()
        with torch.random.fork_rng(devices=self._fork_devices()):
            torch.manual_seed(seed)
            if self.device.type == 'cuda':
                torch.cuda.manual_seed_all(seed)
            p_E, p_x, p_y = self._generator.generate(batch_size)
        return p_E, p_x, p_y

    @staticmethod
    def _sample_n_spots_log_uniform(rng_pair, seed):
        """Sample N_spots log-uniformly in [low, high] using a deterministic
        per-batch RNG so the same batch always gets the same N_spots."""
        import numpy as np
        lo, hi = float(rng_pair[0]), float(rng_pair[1])
        if lo == hi:
            return int(round(lo))
        rs = np.random.RandomState(seed & 0xFFFFFFFF)
        n = int(round(np.exp(rs.uniform(np.log(lo), np.log(hi)))))
        return max(n, 1)

    def _resolve_n_spots(self, scalar_key, range_key, seed):
        """Decide the N_spots value for this call. Range overrides scalar.
        Returns None to fall back to the simulator's default (constructor value)."""
        rng_val = self.calo_cfg.get(range_key)
        if rng_val:
            return self._sample_n_spots_log_uniform(rng_val, seed)
        scalar_val = self.calo_cfg.get(scalar_key)
        if scalar_val is not None:
            return int(scalar_val)
        return None

    def _compute_long_params(self, p_E, long_seed: int):
        """Pre-compute α/T longitudinal shower parameters under a dedicated seed.

        Returns the long_params dict matching what shoot() would compute internally.
        Used by `_simulate_once(..., long_seed=...)` to hold shower SHAPE constant
        across replicas while re-rolling spot positions only — for the MC-only
        ablation that isolates Monte Carlo sampling variance from intrinsic shape variance.

        Must replicate `simulate`'s flattening + masking exactly so the returned
        per-particle α/T values align with the real particles `shoot` will see.
        """
        flat_Es = p_E.reshape(-1)
        real_idx = torch.where(flat_Es == flat_Es)[0]   # NaN-aware: keeps non-pad
        real_Es = flat_Es[real_idx].to(device=self.device, dtype=torch.float32)
        with torch.random.fork_rng(devices=self._fork_devices()):
            torch.manual_seed(long_seed)
            if self.device.type == 'cuda':
                torch.cuda.manual_seed_all(long_seed)
            return get_longitudinal_parameters(real_Es, self.calo_cfg['Z'])

    def _shared_z_long_params(self, p_E, long_seed: int):
        """Event-level shared shower-shape fluctuation (z-conditioning study).

        Draws ONE (z1, z2) pair per EVENT and applies it to every particle in
        that event; per-particle energy-dependent means/sigmas/rho come from
        the library so marginal per-shower statistics are exactly standard —
        only the within-event co-fluctuation changes. Reconstruction mirrors
        torch_calo_flash.calo_flash.get_longitudinal_parameters lines 79-85.
        Stores the drawn z per event on self._last_long_z ([B, 2], cpu).
        """
        from torch_calo_flash.calo_flash import get_beta
        lp = self._compute_long_params(p_E, long_seed)   # library stats; its z draws discarded
        flat_Es = p_E.flatten()
        real_idx = torch.where(flat_Es == flat_Es)[0]
        ev_idx = (real_idx // p_E.shape[1]).to(lp['mean_ln_T'].device)
        B = p_E.shape[0]
        with torch.random.fork_rng(devices=self._fork_devices()):
            zseed = (long_seed * 2654435761 + 97) & 0x7FFFFFFF
            torch.manual_seed(zseed)
            if self.device.type == 'cuda':
                torch.cuda.manual_seed_all(zseed)
            z_ev = torch.randn(B, 2, device=lp['mean_ln_T'].device,
                               dtype=lp['mean_ln_T'].dtype)
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
        self._last_long_z = z_ev.detach().cpu()
        return lp

    def _frozen_long_params(self, p_E, alpha0, T0):
        """Frozen-shape control: constant (alpha, T) for EVERY particle in EVERY
        event, regardless of energy. Removes ALL shower-shape variation (within-pair
        AND across-sample), so the encoder never sees alpha/T vary. Used to test
        whether across-sample (indirect-channel) alpha/T variation is what lets the
        low-within-pair conditions acquire alpha/T-invariance late in training."""
        from torch_calo_flash.calo_flash import get_beta
        lp = self._compute_long_params(p_E, 0)   # correct per-real-particle structure
        a = torch.full_like(lp['alpha'], float(alpha0))
        t = torch.full_like(lp['T'], float(T0))
        lp['alpha'], lp['T'], lp['beta'] = a, t, get_beta(a, t)
        lp['mean_ln_alpha'] = torch.log(a)
        return lp

    def _blend_once(self, p_E, p_x, p_y, seed: int, n_spots: int,
                    lp_shaped: dict) -> torch.Tensor:
        """One λ-blended view in RAW MeV: λ of the darts thrown from `lp_shaped`,
        the rest from the fixed reference profile, weighted so the result is an
        unbiased estimate of λ·μ_shaped + (1−λ)·μ_ref and the total noise is the
        same as an ordinary N-dart simulation's."""
        lb = self.lblend
        ns, nr = lb.split(n_spots)
        out = None
        if ns > 0:
            out = lb.lam * self._simulate_once(
                p_E, p_x, p_y, seed, n_spots_override=ns,
                long_params_override=lb.widen(lp_shaped))
        if nr > 0:
            lp_ref = self._frozen_long_params(p_E, *lb.ref_alpha_t)
            r = (1.0 - lb.lam) * self._simulate_once(
                p_E, p_x, p_y, lb.ref_seed(seed), n_spots_override=nr,
                long_params_override=lp_ref)
            out = r if out is None else out + r
        return out

    def _simulate_once(self, p_E, p_x, p_y, seed: int,
                       n_spots_override: Optional[int] = None,
                       long_seed: Optional[int] = None,
                       long_params_override: Optional[dict] = None) -> torch.Tensor:
        """Run one stochastic detector simulation.

        Returns (B, N_cells_z, N_cells_x, N_cells_y) — all z-layers as channels.
        If n_spots_override is given, uses that value instead of the simulator's
        construction-time N_spots_per_layer.
        If long_seed is given, the α/T longitudinal parameters are drawn under
        long_seed (so they're deterministic across calls that pass the same
        long_seed), while spot positions are still drawn from the main `seed`.
        Use this to keep shower shapes fixed across replicas while only varying
        Monte Carlo spot sampling.
        If long_params_override is given (a dict from _compute_long_params,
        optionally edited per-particle), it is passed straight to the simulator
        and takes precedence over long_seed. Used by the per-particle α/T
        sensitivity probe to inject a controlled single-particle perturbation.
        """
        long_params = long_params_override
        if long_params is None and long_seed is not None:
            long_params = self._compute_long_params(p_E, long_seed)

        with torch.random.fork_rng(devices=self._fork_devices()):
            torch.manual_seed(seed)
            if self.device.type == 'cuda':
                torch.cuda.manual_seed_all(seed)
            sim_kwargs = dict(return_grid=True, return_hits=False, return_truth=False)
            if n_spots_override is not None:
                sim_kwargs['N_spots_per_layer'] = n_spots_override
            if long_params is not None:
                sim_kwargs['long_params'] = long_params
            calo_dict = self._calorimeter.simulate(
                p_E, p_x, p_y,
                **sim_kwargs,
            )

        grid = calo_dict['cell_e']  # (B, N_cells_x, N_cells_y, N_cells_z)

        # Defensive safety net: replace any NaN/Inf cells with 0. With pad_value=NaN
        # this should be a no-op, but keep the cleanup so any rare numerical edge
        # case in pycaloflash itself can't crash a 500-epoch training run. The
        # counters let us audit how often this triggers — should be 0 in healthy runs.
        bad = ~torch.isfinite(grid)
        if bad.any():
            n_bad = int(bad.sum().item())
            self._n_nan_cells_total += n_bad
            self._n_nan_batches_total += 1
            grid = torch.where(bad, torch.zeros_like(grid), grid)
        self._n_batches_total += 1

        if grid.dim() == 3:
            grid = grid.unsqueeze(0)
        # move z-axis (last) to channel position -> (B, N_cells_z, N_cells_x, N_cells_y)
        return grid.permute(0, 3, 1, 2).contiguous()

    def _sim_seed(self, truth_seed: int, replica_idx: int) -> int:
        """Derive a simulation seed well-separated from the truth seed."""
        return (truth_seed * 31337 + replica_idx + 1) & 0x7FFFFFFF

    def _current_prob(self):
        """Effective paired prevalence for this call (dose/protocol knobs)."""
        if self.sample_target_schedule:
            for until, p in self.sample_target_schedule:
                if self.epoch < int(until):
                    return float(p)
            return float(self.sample_target_schedule[-1][1])
        return self.sample_target_prob

    def _paired_pool(self, ref):
        """Lazily build the fixed paired pool (paired_pool_events pairs).

        Built in 8 groups so the pool spans the N_spots range (one log-uniform
        N draw per group, mirroring the per-batch draw of normal training)."""
        if self._pool is None:
            P = int(self.paired_pool_events)
            n_groups = min(8, P)
            per = max(1, P // n_groups)
            cs, ts = [], []
            for gi in range(n_groups):
                gseed = (self.paired_pool_seed * 7919 + gi) & 0x7FFFFFFF
                p_E, p_x, p_y = self._generate_truth(per, gseed)
                n = self._resolve_n_spots('N_spots_per_layer', 'N_spots_range', gseed)
                cs.append(self.norm.forward(self._simulate_once(
                    p_E, p_x, p_y, self._sim_seed(gseed, 0), n_spots_override=n)))
                ts.append(self.norm.forward(self._simulate_once(
                    p_E, p_x, p_y, self._sim_seed(gseed, 1), n_spots_override=n)))
            self._pool = (torch.cat(cs).to(ref.device), torch.cat(ts).to(ref.device))
        return self._pool

    def __call__(self, batch):
        batch_size = len(batch)
        seed = self._batch_seed(batch)
        p_E, p_x, p_y = self._generate_truth(batch_size, seed)

        # Study A shift dial: `pair_shift_max` (cm) shifts the TARGET view's
        # particle positions by a fixed-radius, random-angle global translation
        # per event — the within-pair transform whose magnitude is the dial.
        # Fixed radius (not uniform in a disc) keeps the dial sharp. Own
        # Generator keyed on the batch seed, so runs stay RNG-comparable to
        # their unshifted anchors (the dose-mode convention). The unit
        # direction is stored for the predictor-token arm; pads (NaN) stay NaN.
        shift_m = self.calo_cfg.get('pair_shift_max')
        if shift_m and self.sample_target:
            g = torch.Generator().manual_seed((seed * 92821 + 5) & 0x7FFFFFFF)
            theta = torch.rand(batch_size, generator=g) * (2.0 * torch.pi)
            ux, uy = torch.cos(theta), torch.sin(theta)
            p_x_t = p_x + (float(shift_m) * ux)[:, None].to(p_x.device, p_x.dtype)
            p_y_t = p_y + (float(shift_m) * uy)[:, None].to(p_y.device, p_y.dtype)
            self._last_shift = torch.stack([ux, uy], dim=1).cpu()
        else:
            p_x_t, p_y_t = p_x, p_y
            self._last_shift = None

        # N_spots resolution per call. Three knobs in calo_cfg, in priority order:
        #   - N_spots_context_range / N_spots_target_range: paired-mode-only;
        #     allow asymmetric stochasticity between context and target.
        #   - N_spots_range: log-uniform sampling for a single sim, any mode.
        #   - N_spots_per_layer: scalar default.
        # Different per-batch seeds so context and target sample independently.
        #
        # `paired_fixed_long: true` (sample_target only): hold α/T constant across
        # the two paired views per particle so paired-sim sees only Monte Carlo
        # spot variance during training. The long seed still varies per batch, so
        # different events sample different shower shapes — only the within-pair
        # shape variance is removed. Used to test whether paired-sim's
        # shape-invariance is a trained property of the objective or generic.
        if self.sample_target:
            n_ctx = self._resolve_n_spots(
                'N_spots_per_layer', 'N_spots_context_range', seed * 13 + 1) \
                if self.calo_cfg.get('N_spots_context_range') \
                else self._resolve_n_spots('N_spots_per_layer', 'N_spots_range', seed)
            n_tgt = self._resolve_n_spots(
                'N_spots_per_layer', 'N_spots_target_range', seed * 13 + 2) \
                if self.calo_cfg.get('N_spots_target_range') \
                else self._resolve_n_spots('N_spots_per_layer', 'N_spots_range', seed)
            # With the factor-B stage on, the two sims are kept as RAW MeV so the
            # transform can act on the finished detector image; the stage then
            # applies the normalizer itself. With it off, `_nm` IS
            # self.norm.forward and this is the pre-existing path unchanged.
            _fb = self.factor_b
            _nm = (lambda _t: _t) if _fb is not None else self.norm.forward
            frozen = self.calo_cfg.get('frozen_alpha_t')   # [alpha0, T0]: constant shape everywhere
            if frozen is not None:
                lp_frozen = self._frozen_long_params(p_E, frozen[0], frozen[1])
                imgs_ctx = _nm(self._simulate_once(
                    p_E, p_x, p_y, self._sim_seed(seed, 0),
                    n_spots_override=n_ctx, long_params_override=lp_frozen))
                imgs_tgt = _nm(self._simulate_once(
                    p_E, p_x_t, p_y_t, self._sim_seed(seed, 1),
                    n_spots_override=n_tgt, long_params_override=lp_frozen))
            elif self.lblend is not None:
                # λ-blend. `paired_fixed_long` decides the CHANNEL exactly as it
                # does on the ordinary path: true -> one α/T draw shared by both
                # views (INDIRECT, what the λ fleet trains); false -> a fresh
                # draw per view (DIRECT). Each view then throws its own darts,
                # λ of them from its draw and the rest from the reference.
                if self.calo_cfg.get('paired_fixed_long'):
                    lp_c = lp_t = self._compute_long_params(
                        p_E, (seed * 7919 + 31337) & 0x7FFFFFFF)
                else:
                    lp_c = self._compute_long_params(p_E, self._sim_seed(seed, 0))
                    lp_t = self._compute_long_params(p_E, self._sim_seed(seed, 1))
                imgs_ctx = _nm(self._blend_once(
                    p_E, p_x, p_y, self._sim_seed(seed, 0), n_ctx, lp_c))
                imgs_tgt = _nm(self._blend_once(
                    p_E, p_x_t, p_y_t, self._sim_seed(seed, 1), n_tgt, lp_t))
            else:
                paired_long = (seed * 7919 + 31337) & 0x7FFFFFFF \
                    if self.calo_cfg.get('paired_fixed_long') else None
                imgs_ctx = _nm(self._simulate_once(
                    p_E, p_x, p_y, self._sim_seed(seed, 0),
                    n_spots_override=n_ctx, long_seed=paired_long))
                imgs_tgt = _nm(self._simulate_once(
                    p_E, p_x_t, p_y_t, self._sim_seed(seed, 1),
                    n_spots_override=n_tgt, long_seed=paired_long))
            if _fb is not None:
                imgs_ctx, imgs_tgt, s_c, s_t = _fb.apply_pair(
                    imgs_ctx, imgs_tgt, self.norm, seed)
                self._last_sigma = (s_c.detach().cpu(), s_t.detach().cpu())
            # Dose mode (sample_target only): per sample, keep the independent
            # resim as target with probability p; otherwise target = context
            # image, i.e. vanilla semantics for that sample. p comes from the
            # epoch schedule when present, else sample_target_prob. Own Generator
            # keyed on the batch seed: deterministic per batch and leaves the
            # global RNG stream untouched (dose runs stay RNG-comparable to the
            # p=0 vanilla and p=1 paired anchors).
            p_eff = self._current_prob()
            if p_eff is not None and p_eff < 1.0:
                g = torch.Generator().manual_seed((seed * 104729 + 11) & 0x7FFFFFFF)
                keep = torch.rand(batch_size, generator=g) < p_eff
                keep_dev = keep.to(imgs_tgt.device)
                imgs_tgt = torch.where(keep_dev[:, None, None, None], imgs_tgt, imgs_ctx)
                if self.paired_pool_events and bool(keep.any()):
                    # Realism arm: paired samples come from a fixed reused pool
                    # (both views replaced), not from fresh resims.
                    pool_c, pool_t = self._paired_pool(imgs_ctx)
                    idx = torch.randint(len(pool_c), (int(keep.sum()),), generator=g).to(pool_c.device)
                    imgs_ctx = imgs_ctx.clone(); imgs_tgt = imgs_tgt.clone()
                    imgs_ctx[keep_dev] = pool_c[idx].to(imgs_ctx.dtype)
                    imgs_tgt[keep_dev] = pool_t[idx].to(imgs_tgt.dtype)
                self._last_keep = keep
            else:
                self._last_keep = None
            sample_batch = [(imgs_ctx[i], imgs_tgt[i]) for i in range(batch_size)]
        else:
            n = self._resolve_n_spots('N_spots_per_layer', 'N_spots_range', seed)
            lp_override = None
            if self.calo_cfg.get('share_long_z_per_event'):
                long_seed = (seed * 6364136223846793005 + 1442695040888963407) & 0x7FFFFFFF
                lp_override = self._shared_z_long_params(p_E, long_seed)
            if self.lblend is not None:
                raw = self._blend_once(
                    p_E, p_x, p_y, self._sim_seed(seed, 0), n,
                    self._compute_long_params(p_E, seed))
            else:
                raw = self._simulate_once(
                    p_E, p_x, p_y, self._sim_seed(seed, 0), n_spots_override=n,
                    long_params_override=lp_override)
            if self.factor_b is not None:
                imgs, s_v = self.factor_b.apply_view(raw, self.norm, seed, 0)
                self._last_sigma = (s_v.detach().cpu(), None)
            else:
                imgs = self.norm.forward(raw)
            sample_batch = [(imgs[i], 0) for i in range(batch_size)]

        return self.base_collator(sample_batch)

    def generate_replicas(
        self,
        batch_size: int,
        n_replicas: int = 25,
        seed: Optional[int] = None,
        normalize: bool = False,
        fixed_long_seed: Optional[int] = None,
        fixed_spot_seed: Optional[int] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """Inference helper: n_replicas independent detector responses for the same truth.

        Args:
            normalize: apply self.norm.forward() to replicas (False by default so
                       callers get raw MeV for physics analysis / plotting).
                       Use self.norm.inverse() to go back to MeV from a normalized tensor.
            fixed_long_seed: if set, all replicas use this single seed for the α/T
                       longitudinal draw, so the shower SHAPE is fixed per particle
                       across replicas and only the spot-level Monte Carlo varies.
                       Used for the MC-only ablation (isolates Stage-2). Default None
                       reproduces full stochasticity (both shape and spots vary).
            fixed_spot_seed: the MC-only ablation's mirror image (isolates Stage-1).
                       If set, all replicas share this single spot-sampling seed
                       (common random numbers) while the α/T draw VARIES per replica.
                       Within-truth variation is then driven only by α/T. Mutually
                       exclusive with fixed_long_seed. Used for the α/T-only probe.

        Returns:
            truth    -- (p_E, p_x, p_y) each shape (B, N_particles)
            replicas -- shape (B, n_replicas, N_cells_z, N_cells_x, N_cells_y)
        """
        if seed is None:
            seed = int(torch.randint(0, 0x7FFFFFFF, (1,)).item())
        self._build_backend()

        p_E, p_x, p_y = self._generate_truth(batch_size, seed)
        if fixed_spot_seed is not None:
            assert fixed_long_seed is None, \
                'fixed_spot_seed (α/T-only) and fixed_long_seed (MC-only) are mutually exclusive'
            # α/T-only: spot RNG shared across replicas (CRN); α/T draw varies per replica.
            replicas = [
                self._simulate_once(
                    p_E, p_x, p_y, fixed_spot_seed,
                    long_seed=self._sim_seed(seed, k),
                )
                for k in range(n_replicas)
            ]
        else:
            replicas = [
                self._simulate_once(
                    p_E, p_x, p_y, self._sim_seed(seed, k),
                    long_seed=fixed_long_seed,
                )
                for k in range(n_replicas)
            ]
        stacked = torch.stack(replicas, dim=1)
        if normalize:
            stacked = self.norm.forward(stacked)
        return (p_E, p_x, p_y), stacked


def make_calo(
    batch_size=64,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    training=True,
    drop_last=True,
    sample_target: bool = False,
    sample_target_prob: Optional[float] = None,
    sample_target_schedule: Optional[list] = None,
    paired_pool_events: Optional[int] = None,
    persistent_workers: bool = True,
    num_events: int = 100_000,
    device=None,
    calo_cfg: Optional[Dict[str, Any]] = None,
    generator_cfg: Optional[Dict[str, Any]] = None,
    norm: Optional[CaloNorm] = None,
    sampler_seed: int = 0,
):
    """
    Build a synthetic calorimeter dataset and DataLoader for i-JEPA / MAE training.

    Image channels correspond to all N_cells_z z-layers of the calorimeter.
    Use n_channels_from_cfg(calo_cfg) to get the channel count before dataset creation.

    When sample_target=True the collator produces paired (context, target) images
    from independent detector simulations of the same truth particles.

    For inference use CaloFlashBatchCollator.generate_replicas() directly.
    """
    if collator is None:
        raise ValueError('make_calo requires a mask collator')

    device = _as_device(device)
    if device.type == 'cuda':
        if num_workers > 0:
            _logger.warning(
                'pycaloflash GPU backend: forcing num_workers=0 '
                'to avoid multiple CUDA contexts.')
            num_workers = 0
            persistent_workers = False
        if pin_mem:
            _logger.warning(
                'pycaloflash GPU backend: forcing pin_memory=False — '
                'batches are already on CUDA, pinning would error.')
            pin_mem = False

    dataset = VirtualCaloDataset(n_events=num_events)
    batch_collator = CaloFlashBatchCollator(
        base_collator=collator,
        sample_target=sample_target,
        sample_target_prob=sample_target_prob,
        sample_target_schedule=sample_target_schedule,
        paired_pool_events=paired_pool_events,
        device=device,
        calo_cfg=calo_cfg,
        generator_cfg=generator_cfg,
        norm=norm,
    )

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=training,
        seed=sampler_seed,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=batch_collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=persistent_workers and num_workers > 0,
    )

    img_size = img_size_from_cfg(calo_cfg)
    _logger.info(
        'CaloFlash dataset created (N=%d, img_size=%dx%d, N_cells_z=%d, batch_size=%d, device=%s, sample_target=%s)',
        len(dataset),
        img_size[0],
        img_size[1],
        batch_collator.n_channels,
        batch_size,
        device,
        sample_target,
    )
    _logger.info(
        'CaloFlash data loader created (num_workers=%d, world_size=%d)',
        num_workers,
        world_size,
    )

    return dataset, data_loader, dist_sampler
