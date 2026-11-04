"""Where the analysis reads and writes.

DATA is the unzipped Zenodo record: runs/<run_folder>/ (checkpoints, training logs)
and probe_caches/ (probe images, LiDAR groups). Set MONITOR_DATA to point elsewhere.
Feature caches the scoring scripts write go to WORK (MONITOR_WORK). Result tables
are read from and written to results/ in this repository.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.environ.get('MONITOR_DATA', os.path.join(REPO, 'data'))
RUNS = os.path.join(DATA, 'runs')
CACHE = os.path.join(DATA, 'probe_caches')
WORK = os.environ.get('MONITOR_WORK', os.path.join(REPO, 'work'))
CSV = os.path.join(REPO, 'results')
TRAIN = os.path.join(REPO, 'train')
