# Provenance and licensing

- **`train/`** derives from Meta AI's I-JEPA
  (https://github.com/facebookresearch/ijepa), released under Attribution-NonCommercial
  4.0 International (CC BY-NC 4.0); see `LICENSE`. Its MAE training path is adapted
  from Meta AI's MAE. Our additions: the calorimeter data path, the explicit-latent
  data path (`src/lejepa_data.py`), and the LeJEPA objective and training loop
  (`src/lejepa.py`, `src/train_lejepa.py`, `main_lejepa.py`).
- **The LeJEPA objective** (`train/src/lejepa.py`) is transcribed from the official
  LeJEPA repository (https://github.com/galilai-group/lejepa, commit c293d29), whose
  LICENSE file is CC BY-NC 4.0. `train/tests/test_lejepa.py` checks the transcription
  against that commit.
- **`analysis/`, `figures/`, `configs/` and `results/`** are original to this work and
  released under the same CC BY-NC 4.0 terms. `analysis/lidar_metric.py` implements
  LiDAR (Thilak et al., arXiv:2312.04000); `analysis/probe_model.py` is the query probe
  of the companion paper's code.
- **PyCaloFlash** (the calorimeter shower generator) is a separate dependency, not
  bundled here. Install it from its upstream repository (tag v1.0) and cite the
  archived release (Zenodo DOI: 10.5281/zenodo.20588830).
