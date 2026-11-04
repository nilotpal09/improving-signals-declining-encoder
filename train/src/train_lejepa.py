"""LeJEPA training loop — the provable-JEPA arm.

PURE ADDITION. `src/train.py`, `src/train_byol.py`, `src/train_contrastive.py`,
`src/mae/train.py`, `src/helper.py`, `src/models/`, `src/datasets/` and
`src/masks/` are untouched; this file is reached only through
`main_lejepa.py`, and only for configs that set `meta.objective: lejepa`
(asserted below). Loss provenance and every paper-vs-code choice:
`LEJEPA_README.md` + `src/lejepa.py`'s docstring.

What is held identical to the JEPA arms
---------------------------------------
* data: the same `make_calo` loader and the same `MBMaskCollator` doing the
  collation. When `data.calo_cfg.lejepa_latents` is present (the explicit-latent dial /
  OU arm), the loader's collate_fn is swapped for `LeJEPALatentCollator`, a
  subclass that reuses the same truth/sim machinery with an explicit latent
  draw — shared files stay untouched.
* encoder: `src.helper.init_model` is called UNCHANGED, so the encoder init
  consumes exactly the RNG a JEPA run at the same seed consumes (including
  the predictor construction); the JEPA predictor object is then dropped
  unused — the BYOL arm's precedent, repeated here.
* the collapse diagnostics: rankme_tokens / rankme_event / feat_std_event
  computed the way train.py's forward_target computes them (LayerNorm, mean
  pool, effective_rank), on a FULL (unmasked) view, so the numbers sit on
  the same scale as every other arm.

What is different, and only this
--------------------------------
No EMA target, no stop-gradient, no predictor. Both/all views go through the
SAME encoder; pooled tokens go through MINIMAL.md's BatchNorm-MLP projector;
the loss is `src.lejepa.lejepa_loss` = lamb * SIGReg + (1 - lamb) * mean-
over-views prediction, on projector outputs.

Two view/masking modes (`meta.lejepa.masking`):
  'none'    — every view encoded in full (the paired-view configuration; V
              from the data path: 2 for the paired loader, 2..V for the
              latent collator's `views`).
  'context' — view 1 is encoded through the harness's encoder mask exactly
              as I-JEPA's context branch does (`encoder(view1, masks_enc)`),
              view 2 in full; V = 2 enforced. This is the harness-matched
              "masked LeJEPA" drop-in: same mask stream, same pairing, only
              the objective differs. (The deployed LeJEPA ViT recipe itself
              trains with patch masking — discrepancy 8.)

Two optimizer recipes (`meta.lejepa.optimizer`):
  'house'   — AdamW + WarmupCosineSchedule + CosineWDSchedule on the config's
              optimization block, the same construction every house arm uses
              (apples-to-apples for the harness-matched configs).
  'minimal' — MINIMAL.md's recipe verbatim: AdamW(lr=optimization.lr,
              weight_decay=optimization.weight_decay), 1-epoch LinearLR
              warmup (start_factor 0.01) then CosineAnnealingLR with
              ABSOLUTE eta_min=1e-3 (discrepancy 10: code wins), constant
              weight decay. For the faithful configs.

Checkpoints
-----------
There is NO EMA in this objective, so `ckpt['target_encoder']` and
`ckpt['encoder']` hold the SAME online-encoder state dict. This is
deliberate: the v2 probe reads `prefer='target_encoder'` on every arm, and
MINIMAL probes its (only) backbone — the online encoder IS LeJEPA's readout
encoder. `projector` and `sigreg` (whose `global_step` buffer is the slice-
stream position — resume continues the stream) are stored under their own
keys and never enter either encoder state dict.

Monitors
--------
`lejepa_diag_r{rank}.csv` logs, besides the loss parts: `sigreg_eval` — the
SIGReg statistic on the SAME projector outputs but with a FROZEN slice set
(`eval_statistic`, seed `meta.lejepa.monitor_slice_seed`), so the monitor
number is not confounded by the per-step slice redraw; and `sigreg_step`,
the slice-stream position.
"""

import os

try:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import time
import logging
import sys
import yaml

import numpy as np

import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from src.masks.multiblock import MaskCollator as MBMaskCollator
from src.utils.distributed import init_distributed, AllReduce
from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    effective_rank,
    AverageMeter)
from src.utils.schedulers import (
    WarmupCosineSchedule,
    CosineWDSchedule)
from src.datasets.calo_flash import (
    make_calo, n_channels_from_cfg, img_size_from_cfg, CaloNorm)

from src.helper import init_model
from src.lejepa import (
    LeJEPAProjector, build_sigreg, lejepa_loss, pool_tokens)
from src.lejepa_data import LeJEPALatentCollator

# --
log_timings = True
log_freq = 10
# --

_GLOBAL_SEED = 1
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def init_opt_lejepa_house(
    encoder,
    projector,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    use_bfloat16=False,
    ipe_scale=1.25,
):
    """`src.helper.init_opt` with the JEPA predictor's param groups replaced by
    the projector's. Same optimizer, same bias/1-d no-WD split, same
    schedules — the harness-matched recipe."""
    param_groups = [
        {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in projector.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }, {
            'params': (p for n, p in projector.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }
    ]
    logger.info('Using AdamW (lejepa house recipe: encoder + projector)')
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch))
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch))
    scaler = torch.amp.GradScaler('cuda') if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler


def init_opt_lejepa_minimal(
    encoder,
    projector,
    iterations_per_epoch,
    ref_lr,
    num_epochs,
    wd,
    use_bfloat16=False,
):
    """MINIMAL.md's optimizer/schedule, verbatim semantics: AdamW on the net
    (one group, lr=cfg.lr, weight_decay from the config — the faithful
    configs set MINIMAL's 5e-2), LinearLR warmup over ONE epoch with
    start_factor 0.01, then CosineAnnealingLR to the ABSOLUTE floor
    eta_min=1e-3 (discrepancy 10: the code's floor, not the README's
    lr/1000). Weight decay constant (MINIMAL has no wd schedule)."""
    params = list(encoder.parameters()) + list(projector.parameters())
    logger.info('Using AdamW (lejepa MINIMAL recipe: lr=%.2e wd=%.2e)', ref_lr, wd)
    optimizer = torch.optim.AdamW(params, lr=ref_lr, weight_decay=wd)
    warmup_steps = iterations_per_epoch
    total_steps = iterations_per_epoch * num_epochs
    s1 = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    # max(1, ...) guards degenerate smoke runs whose warmup spans the whole
    # budget; any real config has total >> warmup and is unaffected.
    s2 = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps),
                           eta_min=1e-3)
    scheduler = SequentialLR(optimizer, schedulers=[s1, s2],
                             milestones=[warmup_steps])
    scaler = torch.amp.GradScaler('cuda') if use_bfloat16 else None
    return optimizer, scaler, scheduler, None


def main(args, resume_preempt=False, run_folder=None, debug=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- OBJECTIVE GATE. This entrypoint refuses anything not explicitly marked
    #    lejepa, so no other arm's config can be silently retrained under this
    #    loss (and vice versa).
    objective = args.get('meta', {}).get('objective')
    assert objective == 'lejepa', (
        f"main_lejepa.py requires meta.objective: lejepa, got {objective!r}. "
        "JEPA configs run through main.py, MAE through main_mae.py, "
        "BYOL through main_byol.py, contrastive through main_contrastive.py.")

    # -- META
    local_seed = args.get('meta', {}).get('seed', _GLOBAL_SEED)
    if local_seed != _GLOBAL_SEED:
        np.random.seed(local_seed)
        torch.manual_seed(local_seed)

    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    checkpoint_freq = args['meta'].get('checkpoint_freq', 1)
    checkpoint_schedule = args['meta'].get('checkpoint_schedule', None)

    def _ckpt_due(ep1):  # ep1 is 1-indexed epoch just completed
        if checkpoint_schedule:
            prev = 0
            for until, every in checkpoint_schedule:
                if ep1 <= until:
                    return (ep1 - prev) % every == 0
                prev = until
            return ep1 % checkpoint_schedule[-1][1] == 0
        return ep1 % checkpoint_freq == 0

    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']

    # -- LeJEPA knobs (defaults = the frozen study configuration)
    l_cfg = args['meta'].get('lejepa') or {}
    lamb = float(l_cfg.get('lamb', 0.02))
    num_slices = int(l_cfg.get('num_slices', 256))
    t_max = float(l_cfg.get('t_max', 3))
    n_points = int(l_cfg.get('n_points', 17))
    proj_hidden_dim = int(l_cfg.get('proj_hidden_dim', 2048))
    proj_dim = int(l_cfg.get('proj_dim', 16))
    masking = l_cfg.get('masking', 'none')
    opt_recipe = l_cfg.get('optimizer', 'house')
    statistic = l_cfg.get('statistic', 'epps_pulley')
    monitor_slice_seed = int(l_cfg.get('monitor_slice_seed', 20260816))
    assert masking in ('none', 'context'), f'unknown masking mode {masking!r}'
    assert opt_recipe in ('house', 'minimal'), f'unknown optimizer {opt_recipe!r}'

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    if args.get('meta', {}).get('deterministic', False):
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.cuda.manual_seed_all(local_seed)
        logger.info(f'[determinism] forced deterministic kernels (seed={local_seed})')

    # -- DATA (calo only: this arm is calo-family)
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    dataset_type = args['data'].get('dataset_type', 'calo')
    assert dataset_type == 'calo', (
        f'the lejepa path is calo-only for now, got dataset_type={dataset_type!r}')
    data_cfg = args['data'].get('calo_cfg')
    assert args['data'].get('sample_target', False), (
        'lejepa training needs paired views: set data.sample_target: true')
    img_size = img_size_from_cfg(data_cfg)

    # -- MASK (produced by the collator exactly as on the JEPA path; consumed
    #    only in masking: 'context' mode, and then only masks_enc)
    allow_overlap = args['mask']['allow_overlap']
    patch_size = args['mask']['patch_size']
    num_enc_masks = args['mask']['num_enc_masks']
    min_keep = args['mask']['min_keep']
    enc_mask_scale = args['mask']['enc_mask_scale']
    num_pred_masks = args['mask']['num_pred_masks']
    pred_mask_scale = args['mask']['pred_mask_scale']
    aspect_ratio = args['mask']['aspect_ratio']
    if masking == 'context':
        assert num_enc_masks == 1, (
            'masking: context pools one kept-token set per view; '
            f'set mask.num_enc_masks: 1 (got {num_enc_masks})')

    # -- OPTIMIZATION
    ipe_scale = args['optimization']['ipe_scale']
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']

    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']
    dense_diag = args['logging'].get('dense_diag', False)
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # -- resolve run folder
    if debug:
        run_folder = None
        log_file = None
        save_path = None
        latest_path = None
    else:
        _folder_pre_created = run_folder is not None
        if not _folder_pre_created:
            run_name = args['logging'].get('run_name')
            if run_name:
                run_folder = os.path.join(folder, run_name)
                if os.path.exists(run_folder):
                    run_folder = os.path.join(folder, f'{run_name}_{time.strftime("%Y%m%d-%H%M%S")}')
            else:
                run_folder = os.path.join(folder, f'{tag}_{time.strftime("%Y%m%d-%H%M%S")}')
            os.makedirs(run_folder, exist_ok=True)
            with open(os.path.join(run_folder, 'config.yml'), 'w') as f:
                yaml.dump(args, f)

        log_file = os.path.join(run_folder, f'{tag}_r{rank}.csv')
        save_path = os.path.join(run_folder, f'{tag}' + '-ep{epoch}.pth.tar')
        latest_path = os.path.join(run_folder, f'{tag}-latest.pth.tar')

    # -- csv loggers. `{tag}_r{rank}.csv` keeps the JEPA schema so every existing
    #    watcher / loss-floor reader works on these runs unchanged.
    csv_logger = None
    lejepa_csv = None
    if log_file is not None:
        csv_logger = CSVLogger(log_file,
                               ('%d', 'epoch'),
                               ('%d', 'itr'),
                               ('%.5f', 'loss'),
                               ('%.5f', 'mask-A'),
                               ('%.5f', 'mask-B'),
                               ('%d', 'time (ms)'))
        lejepa_csv = CSVLogger(os.path.join(run_folder, f'lejepa_diag_r{rank}.csv'),
                               ('%d', 'epoch'),
                               ('%d', 'itr'),
                               ('%.6f', 'loss'),
                               ('%.6f', 'lejepa_inv'),
                               ('%.6f', 'lejepa_sigreg'),
                               ('%.6f', 'sigreg_eval'),
                               ('%d', 'sigreg_step'),
                               ('%.6f', 'feat_std_event'),
                               ('%.4f', 'rankme_tokens'),
                               ('%.4f', 'rankme_event'))

    from src.utils.logging import UnifiedLogger
    unified_logger = UnifiedLogger(csv_logger)

    in_chans = n_channels_from_cfg(data_cfg)
    # init_model is called UNCHANGED so the encoder init matches a JEPA run at
    # the same seed bit for bit; the JEPA predictor it also builds is dropped
    # here and never used (the BYOL arm's precedent).
    encoder, _unused_predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=img_size[0],
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name,
        in_chans=in_chans,
        predictor_z_dim=(2 if args['meta'].get('predictor_z') else 0))
    del _unused_predictor
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    projector = LeJEPAProjector(encoder.embed_dim,
                                hidden_dim=proj_hidden_dim,
                                proj_dim=proj_dim).to(device)
    sigreg = build_sigreg(num_slices=num_slices, t_max=t_max,
                          n_points=n_points, statistic=statistic).to(device)
    logger.info('[lejepa] projector %d -> %d -> %d | statistic=%s slices=%d '
                't_max=%g n_points=%d | lamb=%g | masking=%s | optimizer=%s',
                projector.in_dim, projector.hidden_dim, projector.proj_dim,
                statistic, num_slices, t_max, n_points, lamb, masking,
                opt_recipe)

    mask_collator = MBMaskCollator(
        input_size=img_size,
        patch_size=patch_size,
        pred_mask_scale=pred_mask_scale,
        enc_mask_scale=enc_mask_scale,
        aspect_ratio=aspect_ratio,
        nenc=num_enc_masks,
        npred=num_pred_masks,
        allow_overlap=allow_overlap,
        min_keep=min_keep)

    norm_cfg = args['data'].get('norm') or {}
    norm = CaloNorm(**norm_cfg) if norm_cfg else None

    dataset, unsupervised_loader, unsupervised_sampler = make_calo(
        batch_size=batch_size,
        collator=mask_collator,
        pin_mem=pin_mem,
        training=True,
        num_workers=num_workers,
        world_size=world_size,
        rank=rank,
        drop_last=True,
        sample_target=args['data'].get('sample_target', False),
        persistent_workers=args['data'].get('persistent_workers', True),
        num_events=args['data'].get('num_events', 100_000),
        sampler_seed=local_seed,
        norm=norm,
        calo_cfg=data_cfg,
        generator_cfg=args['data'].get('generator_cfg'),
        sample_target_prob=args['data'].get('sample_target_prob'),
        sample_target_schedule=args['data'].get('sample_target_schedule'),
        paired_pool_events=args['data'].get('paired_pool_events'),
    )
    # -- Explicit-latent path: swap the collate_fn for the latent collator
    #    (same truth/sim machinery, explicit z; shared files untouched).
    #    num_workers is 0 on the calo GPU backend, so the swap takes effect at
    #    the next iterator creation (i.e. from epoch 0).
    if (data_cfg or {}).get('lejepa_latents'):
        latent_collator = LeJEPALatentCollator(
            base_collator=mask_collator,
            sample_target=True,
            calo_cfg=data_cfg,
            generator_cfg=args['data'].get('generator_cfg'),
            norm=norm)
        unsupervised_loader.collate_fn = latent_collator
        logger.info('[lejepa] explicit-latent collator active: %s',
                    data_cfg['lejepa_latents'])
        if masking == 'context':
            assert int(data_cfg['lejepa_latents'].get('views', 2)) == 2, (
                'masking: context is a pair construction; views must be 2')
    ipe = len(unsupervised_loader)

    if opt_recipe == 'house':
        optimizer, scaler, scheduler, wd_scheduler = init_opt_lejepa_house(
            encoder=encoder,
            projector=projector,
            wd=wd,
            final_wd=final_wd,
            start_lr=start_lr,
            ref_lr=lr,
            final_lr=final_lr,
            iterations_per_epoch=ipe,
            warmup=warmup,
            num_epochs=num_epochs,
            ipe_scale=ipe_scale,
            use_bfloat16=use_bfloat16)
    else:
        optimizer, scaler, scheduler, wd_scheduler = init_opt_lejepa_minimal(
            encoder=encoder,
            projector=projector,
            iterations_per_epoch=ipe,
            ref_lr=lr,
            num_epochs=num_epochs,
            wd=wd,
            use_bfloat16=use_bfloat16)
    encoder = DistributedDataParallel(encoder, static_graph=True)
    projector = DistributedDataParallel(projector, static_graph=True)

    start_epoch = 0

    def save_checkpoint(epoch):
        if debug or save_path is None:
            return
        enc_sd = encoder.state_dict()
        save_dict = {
            # NO EMA in this objective: the online encoder IS the readout
            # encoder. Saved under BOTH keys so the v2 probe's
            # prefer='target_encoder' resolves on this arm exactly as on
            # every other arm (see module docstring).
            'encoder': enc_sd,
            'target_encoder': enc_sd,
            'projector': projector.state_dict(),
            'sigreg': sigreg.state_dict(),        # global_step = slice stream
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr,
            'objective': 'lejepa',
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if _ckpt_due(epoch + 1):
                torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        unsupervised_sampler.set_epoch(epoch)
        try:
            unsupervised_loader.collate_fn.epoch = epoch
        except AttributeError:
            pass

        loss_meter = AverageMeter()
        inv_meter = AverageMeter()
        sig_meter = AverageMeter()
        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        time_meter = AverageMeter()

        itr_start = time.time()
        for itr, (udata, masks_enc, masks_pred) in enumerate(unsupervised_loader):
            data_time_ms = (time.time() - itr_start) * 1e3  # collate (sim) time

            views = [u.to(device, non_blocking=True) for u in udata
                     if torch.is_tensor(u) and u.ndim == 4]
            assert len(views) >= 2, (
                'lejepa training needs >= 2 views per event; the loader emitted '
                f'{len(views)} image tensors. Check data.sample_target / '
                'lejepa_latents.views.')
            if masking == 'context':
                assert len(views) == 2, 'masking: context is a pair construction'
                masks_enc_dev = [m.to(device, non_blocking=True) for m in masks_enc]
            maskA_meter.update(len(masks_enc[0][0]))
            maskB_meter.update(len(masks_pred[0][0]))

            _diag = {}

            def train_step():
                _new_lr_wd = [None, None]
                if opt_recipe == 'house':
                    _new_lr_wd[0] = scheduler.step()
                    _new_lr_wd[1] = wd_scheduler.step()
                # (MINIMAL recipe steps its torch scheduler AFTER the optimizer
                #  step, as MINIMAL.md's loop does.)

                B = views[0].shape[0]

                # -- one encoder, no target network, no stop-gradient anywhere
                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bfloat16):
                    if masking == 'context':
                        h_ctx = encoder(views[0], masks_enc_dev)   # (B, keep, D)
                        h_full = encoder(views[1])                 # (B, N, D)
                        pooled = torch.stack([pool_tokens(h_ctx),
                                              pool_tokens(h_full)])   # (2, B, D)
                    else:
                        h_all = encoder(torch.cat(views, dim=0))   # (V*B, N, D)
                        h_full = h_all[-B:]                        # last view, full
                        pooled = torch.stack(
                            list(pool_tokens(h_all).split(B, dim=0)))  # (V, B, D)
                    V = pooled.shape[0]
                    proj = projector(pooled.flatten(0, 1)).reshape(V, B, -1)
                    loss, parts = lejepa_loss(proj, sigreg, lamb)
                loss = AllReduce.apply(loss)

                # Collapse signals on a FULL view's encoder features, computed
                # the way train.py's forward_target computes them so the numbers
                # sit on the same scale as the JEPA/MAE/BYOL arms'.
                with torch.no_grad():
                    hv = h_full.detach()
                    hv = torch.nn.functional.layer_norm(hv, (hv.size(-1),))
                    feats_event = hv.mean(dim=1).float()
                    feats_tokens = hv.reshape(-1, hv.size(-1)).float()
                    _diag['rankme_tokens'] = effective_rank(feats_tokens)
                    _diag['rankme_event'] = effective_rank(feats_event)
                    _diag['feat_std_event'] = float(feats_event.std(dim=0).mean())
                    coll = unsupervised_loader.collate_fn
                    _diag['data_nan_cells_total'] = float(getattr(coll, '_n_nan_cells_total', 0))
                    _diag['data_nan_batches_total'] = float(getattr(coll, '_n_nan_batches_total', 0))
                    _diag.update(parts)
                    _diag['sigreg_step'] = float(sigreg.global_step)
                    if (itr % log_freq == 0) or dense_diag:
                        # Monitor: same statistic, FROZEN slice set —
                        # off the training slice stream by construction.
                        _diag['sigreg_eval'] = float(sigreg.eval_statistic(
                            proj.detach().float(), seed=monitor_slice_seed))

                if use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                grad_stats = grad_logger(encoder.named_parameters())
                optimizer.zero_grad()

                if opt_recipe == 'minimal':
                    scheduler.step()
                    _new_lr_wd[0] = optimizer.param_groups[0]['lr']
                    _new_lr_wd[1] = optimizer.param_groups[0]['weight_decay']

                return (float(loss), _new_lr_wd[0], _new_lr_wd[1], grad_stats)

            (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            inv_meter.update(_diag.get('lejepa_inv', 0.0))
            sig_meter.update(_diag.get('lejepa_sigreg', 0.0))
            time_meter.update(etime)

            def log_stats():
                unified_logger.log(epoch + 1, itr, loss, maskA_meter.val, maskB_meter.val, etime)

                if lejepa_csv is not None and ((itr % log_freq == 0) or dense_diag):
                    lejepa_csv.log(epoch + 1, itr, loss,
                                   _diag.get('lejepa_inv', 0.0),
                                   _diag.get('lejepa_sigreg', 0.0),
                                   _diag.get('sigreg_eval', 0.0),
                                   int(_diag.get('sigreg_step', 0)),
                                   _diag.get('feat_std_event', 0.0),
                                   _diag.get('rankme_tokens', 0.0),
                                   _diag.get('rankme_event', 0.0))

                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info('[%d, %5d] loss: %.4f (inv: %.4f sigreg: %.4f '
                                'eval: %.4f | null-mean 1.056) '
                                'feat-std: %.5f rankme-ev: %.2f '
                                '[wd: %.2e] [lr: %.2e] '
                                '[mem cur/peak: %.2e / %.2e MB] '
                                '(sim: %.1f ms | train: %.1f ms)'
                                % (epoch + 1, itr,
                                   loss_meter.avg,
                                   inv_meter.avg,
                                   sig_meter.avg,
                                   _diag.get('sigreg_eval', float('nan')),
                                   _diag.get('feat_std_event', float('nan')),
                                   _diag.get('rankme_event', float('nan')),
                                   _new_wd if _new_wd is not None else float('nan'),
                                   _new_lr if _new_lr is not None else float('nan'),
                                   torch.cuda.memory_allocated() / 1024.**2,
                                   torch.cuda.max_memory_allocated() / 1024.**2,
                                   data_time_ms,
                                   time_meter.avg))

                    if grad_stats is not None:
                        logger.info('[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                                    % (epoch + 1, itr,
                                       grad_stats.first_layer,
                                       grad_stats.last_layer,
                                       grad_stats.min,
                                       grad_stats.max))

            log_stats()

            assert not np.isnan(loss), 'loss is nan'
            itr_start = time.time()

        logger.info('avg. loss %.4f | avg. inv %.4f | avg. sigreg %.4f'
                    % (loss_meter.avg, inv_meter.avg, sig_meter.avg))
        # NOTE the off-by-one is the house convention, copied verbatim from
        # train.py: called with epoch+1, and the file tag is that +1 again, so
        # `jepa-ep{N}.pth.tar` holds the model after N-1 completed epochs and
        # the epoch-tagged checkpoints the probes read line up with every
        # existing arm.
        save_checkpoint(epoch + 1)


if __name__ == "__main__":
    main()
