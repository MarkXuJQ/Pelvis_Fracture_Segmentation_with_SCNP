## Pelvis Fracture Segmentation with SCNP

This repository is the code workspace for our pelvis fracture ROI segmentation experiments based on nnU-Net and SCNP-style priors.
The GitHub repository is intended to track code, experiment overlays, and documentation.
Large local assets such as `source/`, `dataset/nnUNet_raw_data/`, `dataset/nnUNet_preprocessed/`, and `dataset/nnUNet_results/` are intentionally ignored by Git.

Run all commands from the repository root inside an environment that already provides `nnUNetv2`.
All entrypoints resolve paths against the local repository root through `PELVIS_SCNP_DATA_ROOT`, so local commands consistently target this workspace instead of an external checkout.

## Repository Layout

- `dataset/`
  Thin local entry scripts for the stage-1 anatomy step and stage-2 training dataset preparation.
- `training/`
  Main training package, including runtime utilities, experiment registry, data preparation, and nnUNetv2 trainer overlays.
- `inference/`
  Prediction entrypoints. These are kept out of the first training-code-only commit.
- `paper/`
  Thesis and manuscript material.

Within `training/`:

- `training/data/`
  Stage-1 anatomy prediction, stage-2 masked-CT dataset build, preprocessing, and audit logic.
- `training/run/`
  Unified launchers plus experiment registry code.
- `training/runtime/`
  Shared path resolution, split generation, disMap loading, and trainer-base utilities.
- `training/evaluation/`
  Prediction and validation flows. These are not part of the first training-code-only commit.
- `training/experiments/`
  Experiment-specific trainer overlays and supporting example code.

## Data Assumptions

The local workspace writes a nnU-Net stage-2 training dataset under the repository root. The second-stage build reads the
original full CTs and the stage-1 anatomy predictions instead of the legacy cropped ROI files:

```text
<repo-root>/
  source/
    source_manifest.json
  dataset/
    nnUNet_raw_data/
    nnUNet_preprocessed/
    nnUNet_results/

<repo-root>/external/Dataset501_FracAnatomy/
  imagesTr/
  labelsTr/

<repo-root>/dataset/stage1_anatomy_predictions/anatomy_cascade/
  FracAnatomy_0001.nii.gz
  ...
```

The external roots can be overridden with `PELVIS_SCNP_FULL_CT_DATASET_DIR` and
`PELVIS_SCNP_STAGE1_ANATOMY_LABEL_DIR`. The first-stage FracSegNet weight root can be overridden with
`PELVIS_SCNP_FRACSEGNET_ANATOMICAL_MODEL_DIR`. By default the code looks under `<repo-root>/external/` so a fresh
clone does not depend on any personal absolute path.
If the first-stage FracSegNet anatomy model must run in a separate nnU-Net v1 environment, set `NNUNET_V1_PY` to that
environment's Python executable before running `dataset/generate_stage1_anatomy_predictions.py`.

Pipeline semantics follow the official FracSegNet fracture-stage setting and the thesis description:

- Stage 1 runs on the complete raw CT and produces an anatomy label image.
- Stage 2 images are full-grid FracSegNet-style masked CT:
  CT values are preserved only where the stage-1 anatomy prediction equals the target bone label
  (`1 sacrum / 2 left hip / 3 right hip`), and all other voxels are set to `0`.
- Fracture GT is never used to generate the CT input mask. It is used only as the supervision label and for `disMap`/SCNP loss inputs.
- Training targets are collapsed into the thesis three-class semantic label space:
  `0 background / 1 main fracture segment / 2 secondary fragments`.
- Source fragment labels `>=2` are merged into class `2` after the per-bone full-grid fracture label is built.
- The network remains CT-only at train and test time; anatomy labels are an upstream preprocessing/inference product, not a network input channel.
- The `disMap` prior is used only as a training-side prior.

## nnU-Net Dataset Name

The default nnU-Net dataset is `Dataset503_SCNP` because `503` was the historical local task id left after several
experimental rebuilds. It is not part of the method or the thesis claim. The code keeps it only so existing local
preprocessed caches and training folders remain readable.

For a clean new project, set these before running data preparation or training. In PowerShell:

```powershell
$env:PELVIS_SCNP_NNUNET_DATASET_ID = "510"
$env:PELVIS_SCNP_NNUNET_DATASET_NAME = "Dataset510_PelvisStage2SCNP"
```

On Linux/macOS shells, use `export NAME=value` instead. If you change either value after preprocessing, rebuild the raw
and preprocessed nnU-Net dataset because nnU-Net stores the id/name in folder names and manifests.

## Training Data Scripts

The preferred entrypoint names describe the stage and purpose directly:

- `dataset/generate_stage1_anatomy_predictions.py`
  Runs the FracSegNet anatomy model on complete CT volumes and writes stage-1 anatomy labels.
- `dataset/build_stage2_masked_ct_dataset.py`
  Builds the stage-2 nnU-Net raw dataset: one CT-only masked image per target bone, plus the three-class fracture label.
- `dataset/preprocess_stage2_training_dataset.py`
  Runs nnU-Net preprocessing and generates training-side `disMap` sidecars for the SCNP loss.
- `dataset/audit_stage2_training_dataset.py`
  Checks that raw and preprocessed data match the thesis data semantics before training.

## Common Commands

Generate stage-1 anatomy predictions, then build the raw stage-2 masked-CT training dataset:

```bash
python dataset/generate_stage1_anatomy_predictions.py
python dataset/build_stage2_masked_ct_dataset.py --reset_existing
```

Re-run preprocessing after any stage-2 dataset rebuild:

```bash
python dataset/preprocess_stage2_training_dataset.py --reset_preprocess --configs 3d_fullres --dismap_workers 4
```

Audit local dataset state with explicit local paths:

```bash
python dataset/audit_stage2_training_dataset.py --dataset_dir D:\Code\Pelvis_SCNP\dataset\nnUNet_raw_data\Dataset503_SCNP --preprocessed_dataset_dir D:\Code\Pelvis_SCNP\dataset\nnUNet_preprocessed\Dataset503_SCNP
```

If you change `PELVIS_SCNP_NNUNET_DATASET_NAME`, replace `Dataset503_SCNP` in the audit paths with your chosen name.

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

Train the ablation family, including the full-image hard-SCNP run without FDM:

```bash
python training/train_hard_no_fdm.py --split_mode patient --fold 0
python training/train_soft_no_fdm.py --split_mode patient --fold 0
python training/train_scnp_soft_fdm.py --split_mode patient --fold 0
python training/train_soft_soft_fdm.py --split_mode patient --fold 0
```

## Notes

- `training/auto_start_rf3_thr03.py` was removed because training is now driven by the unified launcher and explicit CLI entrypoints.
- The soft-variant family lives under `training/experiments/scnp_soft_variants/`, where `hard_no_fdm` is the hard-SCNP full-image variant without a `disMap` loss prior.
