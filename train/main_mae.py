# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# MAE training entry point: I-JEPA's main_mae.py plus the data patch below,
# which gives MAE the same explicit-latent data path as LeJEPA. Run from this
# folder:
#     python main_mae.py --fname ../configs/<run>.yaml --devices cuda:0

import argparse

import multiprocessing as mp
import pprint
import yaml

from src.utils.distributed import init_distributed
from src.mae.train import main as app_main

# --- explicit-latent data patch ---------------------------------------------
# Wrap make_calo in the training module's namespace so configs carrying
# data.calo_cfg.lejepa_latents get the explicit-latent collator
# (src/lejepa_data.py). Configs without that key pass through unchanged
# (attach_wp1_dial does nothing without it). The patch is applied at import
# time, so spawned worker processes re-apply it when they re-import this file.
import src.mae.train as _wp1_train_mod
from src.lejepa_data import attach_wp1_dial as _wp1_attach
_wp1_orig_make_calo = _wp1_train_mod.make_calo


def _wp1_make_calo(*a, **kw):
    ds, loader, sampler = _wp1_orig_make_calo(*a, **kw)
    if _wp1_attach(loader, **kw):
        import logging as _lg
        _lg.getLogger().info('[wp1-dial] explicit-latent collator active: %s',
                             (kw.get('calo_cfg') or {}).get('lejepa_latents'))
    return ds, loader, sampler


_wp1_train_mod.make_calo = _wp1_make_calo
# ------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs.yaml')
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='which devices to use on local machine')
parser.add_argument(
    '--run_folder', type=str, default=None,
    help='output folder (default: logging.folder/logging.run_name)')


def process_main(rank, fname, world_size, devices, run_folder=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])

    import logging
    logging.basicConfig()
    logger = logging.getLogger()
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f'called-params {fname}')

    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f'Running... (rank: {rank}/{world_size})')
    app_main(args=params, run_folder=run_folder)


if __name__ == '__main__':
    args = parser.parse_args()

    mp.set_start_method('spawn', force=True)

    procs = []
    num_gpus = len(args.devices)
    for rank in range(num_gpus):
        p = mp.Process(
            target=process_main,
            args=(rank, args.fname, num_gpus, args.devices),
            kwargs=dict(run_folder=args.run_folder))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
