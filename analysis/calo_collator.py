"""A data generator pinned at one element count, for probe images and LiDAR groups.

make_collator is copied verbatim from the I-JEPA conditions code
(diagnostics/probe_n_decodability.py), which the analysis previously imported."""
import sys

import torch

import paths

sys.path.insert(0, paths.TRAIN)

from src.datasets.calo_flash import CaloFlashBatchCollator, CaloNorm  # noqa: E402

DEVICE = torch.device('cuda:0')


def make_collator(calo_cfg, gen_cfg, n_spots, norm_cfg=None):
    cfg = {k: v for k, v in calo_cfg.items()
           if k not in ('N_spots_range', 'N_spots_context_range', 'N_spots_target_range')}
    cfg['N_spots_per_layer'] = int(n_spots)
    bc = CaloFlashBatchCollator(base_collator=lambda x: x, device=DEVICE,
                                 calo_cfg=cfg, generator_cfg=gen_cfg,
                                 norm=CaloNorm(**(norm_cfg or {})))
    bc._build_backend()
    return bc
