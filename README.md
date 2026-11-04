# Code for "Improving Signals, Declining Encoder"

Code, run configurations and result tables for the paper **"Improving Signals,
Declining Encoder: A Known-Answer Test of Label-Free Monitoring in
Self-Supervised Learning"** ([arXiv:2610.35766](https://arxiv.org/abs/2610.35766)).

Author: Nilotpal Kakati, Weizmann Institute of Science, Rehovot, Israel.

The trained checkpoints, training logs and probe images are on Zenodo (DOI:
[10.5281/zenodo.23666713](https://doi.org/10.5281/zenodo.23666713)). With them, every figure and printed number can be recomputed without
retraining. Training is not bitwise reproducible across machines, which is why
the checkpoints are released.

## What is here

| Folder | Contents |
|---|---|
| `train/` | Training code: LeJEPA (`main_lejepa.py`) and MAE (`main_mae.py`), built on Meta's I-JEPA code. `tests/` checks the LeJEPA loss against the official implementation. |
| `configs/` | The configuration of each of the 147 training runs, named by run. |
| `analysis/` | Scoring of the checkpoints (readability, label-free signals, LiDAR, detectors, checkpoint selection) and the derived tables. |
| `figures/` | Scripts for the paper's six figures. |
| `results/` | Every result table the analysis reads or writes (CSV), plus `runs.csv`, one row per run. |
| `figs/` | The six figures as they appear in the paper. |

## Setup

Python 3.11 with PyTorch 2.6 (CUDA). `requirements.txt` lists the versions used.
The image generator, PyCaloFlash, is a separate package: install it from its
v1.0 tag and cite its archived release (Zenodo DOI 10.5281/zenodo.20588830).

Download the Zenodo record and unzip all archives into `data/` (or set
`MONITOR_DATA` to where you unzipped them). This gives `data/runs/<run_folder>/`
for each run and `data/probe_caches/`. The scoring scripts write feature
caches to `work/` (or `MONITOR_WORK`).

## Training a run

From `train/`:

```
python main_lejepa.py --fname ../configs/<run>.yaml --devices cuda:0
python main_mae.py    --fname ../configs/<run>.yaml --devices cuda:0   # the MAE runs
```

The run folder is `logging.folder/<run_name>` (by default `train/runs/<run>`).
`results/runs.csv` gives each run's setting.

## Recomputing results

Each script in `analysis/` reads and writes `results/`. Most derived tables
refuse to overwrite an existing file, so move the recorded file aside first and
compare the new one with it. The per-checkpoint scoring scripts read the
checkpoints from `data/runs/` and append only missing rows.

The probe caches can be rebuilt from code instead of downloaded: the builders
(`knee_lidar.py`, `knee_lidar_sharedshape.py`, `knee_lidar_replica_seed.py`,
`knee_lidar_dialmatched.py`, `rig_hardening_sweep.py`, `knee_s3_ladder_split.py
--build-test`, `knee_t2_caches.py`, `build_nsweep_caches.py`, and `rig_sets` in
`score_rigd.py`) write byte-identical files. One cache, `m5_twoview.pt`, has no
builder and ships as data only.

The figure scripts read `results/` and write `figs/`:

```
python figures/make_overview_fig.py          # Figure 1
python figures/make_house_style_figs.py      # Figures 2, 4 and 6
python figures/make_response_e100_fig.py     # Figure 3
python figures/make_generator_fig.py         # Figure 5 (the generator, Appendix A)
```

## How this code was checked

We checked this code against the code the paper's results ran on:

- training: the first two batches of all 147 configs and 8 training steps of 9
  configs give bit-identical tensors, checkpoints and loss logs;
- analysis: every script behind a printed number or figure, rerun, writes
  byte-identical tables (figures identical apart from the PDF creation date);
- probe caches: 21 of the 22 released caches rebuild byte-identical;
- released checkpoints, scored with this code, give the stored readability and
  LiDAR values.

## Licence

CC BY-NC 4.0 (see `LICENSE` and `NOTICE.md`): the training code derives from
Meta's I-JEPA, released under that licence.

## Citation

Please cite the paper:

```bibtex
@misc{kakati2026improvingsignalsdecliningencoder,
      title={Improving Signals, Declining Encoder: A Known-Answer Test of Label-Free Monitoring in Self-Supervised Learning}, 
      author={Nilotpal Kakati},
      year={2026},
      eprint={2610.35766},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2610.35766}, 
}
```
