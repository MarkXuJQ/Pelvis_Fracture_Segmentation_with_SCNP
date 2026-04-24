## Pelvis Fracture Segmentation with SCNP

This repository is the code workspace for our pelvis fracture ROI segmentation experiments based on nnU-Net and SCNP-style priors.
The GitHub repository is intended to track code, experiment overlays, and documentation.
Large local assets such as `source/`, `dataset/nnUNet_raw_data/`, `dataset/nnUNet_preprocessed/`, and `dataset/nnUNet_results/` are intentionally ignored by Git.

Run all commands from the repository root inside an environment that already provides `nnUNetv2`.
All entrypoints resolve paths against the local repository root through `PELVIS_SCNP_DATA_ROOT`, so local commands consistently target this workspace instead of an external checkout.

## Repository Layout

- `dataset/`
  Thin local entry scripts for dataset build, preprocessing, and audit.
- `training/`
  Main training package, including runtime utilities, experiment registry, evaluation flows, and nnUNetv2 overlays.
- `inference/`
  Compatibility prediction entrypoints that call the same local training package.
- `paper/`
  Thesis and manuscript material.

Within `training/`:

- `training/data/`
  Dataset build, preprocessing, and audit logic.
- `training/run/`
  Unified launchers plus experiment registry code.
- `training/runtime/`
  Shared path resolution, split generation, disMap loading, and trainer-base utilities.
- `training/evaluation/`
  Prediction, validation, postprocessing, and full-CT comparison flows.
- `training/experiments/`
  Experiment-specific trainer overlays and supporting example code.

## Data Assumptions

The local workspace expects a repository-root data layout like this when you actually run preprocessing or training:

```text
<repo-root>/
  source/
    images/
    labels/
  dataset/
    nnUNet_raw_data/
    nnUNet_preprocessed/
    nnUNet_results/
```

Pipeline semantics follow the official FracSegNet fracture-stage setting:

- Source ROI labels use `1` for the main fracture segment and `2..N` for additional fragments.
- Training targets are collapsed into the official four-class semantic label space:
  `0 background / 1 main fracture segment / 2 segment 2 / 3 segment 3`.
- Source fragment labels `>=4` are merged into class `3`.
- The network remains CT-only at train and test time.
- The `disMap` prior is used only as a training-side prior.

## Common Commands

Build the raw dataset from the local repository-root `source/` directory:

```bash
python dataset/build_dataset.py
```

Preprocess once and reuse cached outputs:

```bash
python dataset/preprocess_dataset.py
```

Audit local dataset state with explicit local paths:

```bash
python dataset/audit_dataset.py --dataset_dir D:\Code\Pelvis_SCNP\dataset\nnUNet_raw_data\Dataset503_SCNP --preprocessed_dataset_dir D:\Code\Pelvis_SCNP\dataset\nnUNet_preprocessed\Dataset503_SCNP
```

List registered experiments:

```bash
python training/run_experiment.py list
```

Train a standard single-RF experiment:

```bash
python training/run_experiment.py train single_rf3_thr03 --preprocess --split_mode patient --fold 0
```

Train other non-soft variants through compatibility shims:

```bash
python training/train_single_rf3_thr05.py --split_mode patient --fold 0
python training/train_single_rf5_thr03.py --split_mode patient --fold 0
python training/train_single_no_threshold.py --split_mode patient --fold 0
python training/train_multi_rf3_thr03_rf5_thr05.py --split_mode patient --fold 0
python training/train_multi_rf0307.py --split_mode patient --fold 0
```

Predict with the unified entrypoint:

```bash
python training/run_experiment.py predict single_rf3_thr03 --fold all
```

Or with compatibility wrappers:

```bash
python inference/predict_single_rf3_thr03.py --fold all
python inference/predict_single_rf5_thr03.py --fold all
python inference/predict_single_no_threshold.py --fold all
python inference/predict_multi_rf3_thr03_rf5_thr05.py --fold all
python inference/predict_multi_rf0307.py --fold all
```

## Notes

- `training/auto_start_rf3_thr03.py` was removed because training is now driven by the unified launcher and explicit CLI entrypoints.
- Soft-variant training and inference entrypoints remain in the repository, but this cleanup round focuses on the non-soft single-RF and multi-RF workflows you asked to publish first.
