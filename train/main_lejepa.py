# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# LeJEPA training entry point: I-JEPA's main.py with the training loop taken
# from src/train_lejepa.py (loss provenance: LEJEPA_README.md). Run from this
# folder:
#     python main_lejepa.py --fname ../configs/<run>.yaml --devices cuda:0
# src/train_lejepa.py refuses any config without meta.objective: lejepa.

import argparse

import multiprocessing as mp
import pprint
import yaml

from src.utils.distributed import init_distributed
from src.train_lejepa import main as app_main

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
parser.add_argument(
    '--debug', action='store_true',
    help='skip folder creation and checkpoints; quick print-and-go')


def process_main(rank, fname, world_size, devices, run_folder=None, debug=False):
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
    app_main(args=params, run_folder=run_folder, debug=debug)


if __name__ == '__main__':
    args = parser.parse_args()

    mp.set_start_method('spawn', force=True)

    procs = []
    num_gpus = len(args.devices)
    for rank in range(num_gpus):
        p = mp.Process(
            target=process_main,
            args=(rank, args.fname, num_gpus, args.devices),
            kwargs=dict(run_folder=args.run_folder, debug=args.debug))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
