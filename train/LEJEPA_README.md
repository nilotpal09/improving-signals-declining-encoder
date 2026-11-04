# LeJEPA arm — implementation provenance and fidelity record

Status 2026-08-16: SIGReg/loss module + unit battery DONE; training path,
latent machinery, and pilot configs DONE (sections below). All additions,
zero shared-file modifications (`git status`: untracked files only).

Written for the independent adversarial diff that runs after the pilot:
everything you need to compare is pinned below.

## What is followed, exactly

| source | version |
|---|---|
| LeJEPA paper | arXiv **2511.08544 v3** (2025-11-14) |
| Official repository | `github.com/galilai-group/lejepa`, commit **`c293d291ca87cd4fddee9d3fffe4e914c7272052`** (HEAD of `main`, 2026-01-25, "MINIMAL num_classes"; `rbalestr-lab/lejepa` redirects here) |
| Read-only clone used by the tests | the official repository at the commit above |
| Theorem paper (study framing only) | arXiv 2605.26379 v1 |

**Tiebreak rule (2026-08-16): where the paper text and the pinned code
disagree, the pinned code wins, and the choice is documented here.**

## The module

`src/lejepa.py` — pure addition; no shared cohort code path was modified.

- `EppsPulley` — transcription of `lejepa/univariate/epps_pulley.py` class
  `EppsPulley` (lines 62–99; the production statistic, NOT the
  `DeprecatedEppsPulley` that shadowed it before commit `fd27093`), plus the
  `world_size` property of `lejepa/univariate/base.py` (lines 31–35).
- `SlicingUnivariateTest` — transcription of
  `lejepa/multivariate/slicing.py` (lines 91–153).
- `LeJEPAProjector` — MINIMAL.md's head, built with the same
  `torchvision.ops.MLP(in, [2048, 2048, proj_dim], norm_layer=BatchNorm1d)`
  call MINIMAL uses (torchvision 0.21.0 in the house env; the realized layer
  sequence is pinned by test).
- `lejepa_inv_loss`, `lejepa_loss` — MINIMAL.md's loss lines:
  `inv = (proj.mean(0) - proj).square().mean()`;
  `total = sigreg * lamb + inv * (1 - lamb)` on the (V, B, D) view stack.
- `build_sigreg()` — the study's frozen SIGReg configuration in one place.
- `SlicingUnivariateTest.eval_statistic` — monitor-only addition,
  no pinned counterpart, never in the loss path.

## Resolution of the 14 paper-vs-code discrepancies

"Code" = the pinned commit.

| # | issue | our resolution |
|---|---|---|
| 1 | EP window: text/pseudocode [−5,5]; code t_max=3 | **t_max=3** (code default; also the theorem paper's experiments) |
| 2 | slices: text 1024; pseudocode+MINIMAL 256; scripts 1000/1024; class no default | **256** (`build_sigreg`; MINIMAL + pseudocode; our scale is MINIMAL-scale) |
| 3 | λ: text 0.05; MINIMAL 0.02 (small scale) | **0.02 primary**; 0.05 is the single sensitivity check fixed in advance |
| 4 | slice seeding: pseudocode global_step; library internal MAX-synced counter; MINIMAL unseeded | **library counter mechanism** (= pseudocode semantics on 1 GPU; reproducible; checkpoints via the `global_step` buffer). MINIMAL's unseeded draw NOT used |
| 5 | views count (10 vs 8 vs 4) | harness-matched **V=2**; faithful **V=4** all-global (MINIMAL's recommended run). No multi-crop exists in this codebase |
| 6 | resolutions 224/96 vs 238/98 | n/a — calo images are 64×64×4 (house geometry); "views" are the harness's paired renderings, not crops |
| 7 | teacher-student flag ON in ViT-L scripts | **OFF** — no EMA, no SWA anywhere in this arm (the heuristics-free configuration is the study's subject; the flag lives in `stable_pretraining`, which is not shipped in the pinned repo) |
| 8 | patch_mask_ratio=0.3 in ViT-L scripts | harness-matched config keeps the HOUSE masking convention for apples-to-apples (exact wiring documented when the training path lands); faithful config is unmasked, per MINIMAL |
| 9 | lr/wd sets differ three ways | harness-matched: house optimizer conventions (apples-to-apples with I-JEPA/MAE cohorts); faithful: MINIMAL's recipe (AdamW wd 5e-2, lr 2e-3, 1-epoch linear warmup start_factor 0.01 → cosine) |
| 10 | LR floor: README lr/1000 vs MINIMAL eta_min=1e-3 | faithful config: **MINIMAL's absolute eta_min=1e-3** (code wins) |
| 11 | EP weight: text e^{−t²}; code e^{−t²/2} | **e^{−t²/2}** (code); the text-faithful variant is mutation M1 below and fails 18 tests |
| 12 | README quick start uses a wrong kwarg | n/a (we never call that API form; our kwarg is `n_points`, as in code) |
| 13 | pip 0.0.1 may predate the class-shadowing fix | pip never used; the vendored **git clone at the pinned commit** is the comparison target, and the test battery asserts the imported package resolves to it |
| 14 | projector absent from paper listing, present in MINIMAL/scripts | **projector used** (BN-MLP 2048-2048-16), loss on projector output, probes on backbone — MINIMAL's semantics |

## Named code deviations (D1–D5 in the module docstring)

1. **D1** `all_reduce` helpers keep the pinned dist-initialized guard but
   import the collectives lazily inside the branch; the harness is
   single-process (submit asserts ngpus=1), so behavior is identical.
2. **D2** `EppsPulley` subclasses `nn.Module` and carries only `world_size`
   from the pinned base class; the base's sorting machinery and Normal
   member are unused by the EP forward path (they serve the CDF tests).
3. **D3** `eval_statistic` added (frozen-slice monitor).
4. **D4** `build_sigreg` fixes `num_slices=256` (the pinned class has no
   default).
5. **D5** Embedding = plain mean over the (B, 256, 96) token grid (the BYOL/
   contrastive twin pooling); MINIMAL's timm 512-d classifier head has no
   counterpart on ViT-Nano, and the projector's first Linear plays its role.
   Projector widths kept literal (2048-2048-16), not rescaled to nano.

## A gradient identity, found red-first

With the all-views-global center, the prediction term's center-gradient
cancels identically (`sum_v (mean - p_v) = 0`): detaching the center changes
NO gradient. LeJEPA's "no stop-gradient" is structurally true and enforced by
a source-level test, but it is gradient-equivalent to a detached center in
every configuration this study runs (it would bind only for V_g < V
multi-crop). Interpretation guard: differences from I-JEPA/BYOL on this rig
attribute to the missing EMA target and to SIGReg — never to the center's
gradient path. Pinned by `test_inv_loss_center_gradient_cancels_exactly`.

## Verification (all in `tests/test_lejepa.py`; 38 tests, green 2026-08-16)

1. **Bitwise CPU parity with the pinned classes** — values AND autograd
   gradients, shapes up to 4-D, nondefault args, all three reductions,
   clip_value path, three successive sliced calls (same seeded slice
   stream), plus a guard that the compared package IS the vendored clone.
2. **The statistic is the formula** — equality with an independent
   complex-arithmetic Epps-Pulley on the symmetric grid (validates the
   half-grid + doubled-weights trick against paper Eq. (Epps-Pulley));
   exact finite-N null-mean law `E[stat] = Σ W_i (1 − e^{−t_i²})` = 1.0561
   at the frozen quadrature (pins N-scaling and batch-mean semantics
   distribution-side); alternatives (shift/scale/bimodal) ranked above the
   null; linear N-scaling under a fixed alternative.
3. **Slice mechanics** — unit columns, counter-seeded, reproducible across
   instances, advancing, gradient-free, state_dict round-trip continues the
   stream.
4. **Loss + projector + harness safety** — MINIMAL-reference equality given
   the same slices on a (V, B, D) stack; hand-computed inv_loss; λ
   endpoints; layer-pinned projector; bf16-autocast finiteness; fp32
   buffers; no input mutation; monitor variant off the training stream.
5. **Mutation check** (battery has teeth): wrong window e^{−t²} → 18 tests
   fail; dropped N-scaling → 17 fail; detached center → 1 fails (the
   structural test — by the identity above, gradients cannot see it, which
   is why that test exists); wrong normalization axis → 9 fail. Restored →
   38 pass, and the full house suite (125 tests) passes.

Run: `python -m pytest tests/test_lejepa.py -q`.

## The training path (pure additions, BYOL-arm precedent)

| file | role |
|---|---|
| `main_lejepa.py` | `main_byol.py` with the one import line changed (house precedent) |
| `src/train_lejepa.py` | the loop: gate `meta.objective: lejepa`; `init_model` UNCHANGED (JEPA predictor built then dropped, RNG parity); no EMA, no target network; house collapse diagnostics on a FULL view (same scale as every arm) |
| `src/lejepa_data.py` | explicit-latent machinery: `sample_z` (gaussian / bimodal(mu) / uniform / laplace, exactly zero-mean unit-variance), `ou_partner` (theorem Eq. 1), `LeJEPALatentCollator` (subclass; explicit z through Eq.-11 `long_params_override`; single-particle gun only, guards asserted) |
| `tests/test_lejepa_data.py` | 17 tests: moment laws per dial point (4th moments vs closed forms), OU correlation, Eq.-11 inversion back to the injected z, bitwise batch determinism, guards. Mutation-checked (2/2 caught) |
| `configs/conditions_lejepa/` | `gen_lejepa_configs.py` + 6 pilot configs (see its docstring) |

Key contracts:
- **View/masking modes** (`meta.lejepa.masking`): `none` = every view
  encoded in full (paired-view / faithful V=4); `context` = view 1 through
  the harness's encoder mask exactly as I-JEPA's context branch, view 2
  full (the harness-matched masked drop-in; deployed LeJEPA itself trains
  with patch masking, discrepancy 8).
- **Optimizer recipes** (`meta.lejepa.optimizer`): `house` (WarmupCosine +
  CosineWD, the harness-matched apples-to-apples) or `minimal` (MINIMAL.md
  verbatim: AdamW wd 5e-2, 1-epoch LinearLR warmup 0.01, CosineAnnealingLR
  eta_min=1e-3 absolute — discrepancy 10 code-side).
- **Checkpoints**: NO EMA exists, so `encoder` and `target_encoder` hold
  the SAME online-encoder state dict (deliberate: the v2 probe's
  `prefer='target_encoder'` then reads LeJEPA's one true readout encoder);
  `projector` and `sigreg` (slice-stream `global_step`) under their own
  keys. Verified: probe-side `encoder_state_dict_from_ckpt` loads with 0
  missing / 0 unexpected and forwards to the standard (B, 256, 96) grid.
- **Monitors** (`lejepa_diag_r{rank}.csv`): loss parts, `sigreg_eval`
  (frozen-slice statistic, seed `meta.lejepa.monitor_slice_seed` — a
  candidate monitor, off the training slice stream), `sigreg_step`,
  feat_std/RankMe.
- **Latent channel per config**: Gaussian-point masked/paired configs use
  the EXISTING `paired_fixed_long` path (shared per-particle z within the
  pair, library-drawn N(0, I) — exact by construction, no new code in the
  loop); only the OU arm (rho=0.9) and the future dial points use
  `LeJEPALatentCollator` (activated by `data.calo_cfg.lejepa_latents`,
  swapped in as the loader's collate_fn — shared files untouched).

Smoke record (2026-08-16, 1 epoch, 512 events, local TITAN): all 6 pilot
variants train with finite losses; initial SIGReg ~38-76 vs null mean 1.056
(expected at init); faithful V=4 bs 256 peaks ~13.8 GB GPU (submit to
A6000); I-JEPA gun loss 0.455, MAE 0.693 first-iter.
