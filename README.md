# Pelvis Fracture Segmentation with SCNP

This repository provides a compact nnU-Net v2 example for pelvis fracture segmentation with FracSegNet-style FDM-gated SCNP training. The released code focuses on the method path used in the paper: CT-only stage-2 fracture segmentation with three labels, `0 background`, `1 main fracture segment`, and `2 secondary fragments`.

## Installation

Create a Python environment with nnU-Net v2 and the usual medical-imaging dependencies used by nnU-Net, PyTorch, NumPy, SciPy, SimpleITK, and nibabel.

The code expects the nnU-Net command line tools to be available:

```bash
nnUNetv2_train --help
nnUNetv2_predict --help
```

Optional environment variables can be used to adapt the example to another local dataset layout:

```bash
export PELVIS_SCNP_DATA_ROOT=/path/to/project_or_data_root
export PELVIS_SCNP_NNUNET_DATASET_ID=520
export PELVIS_SCNP_NNUNET_DATASET_NAME=Dataset520_PelvisSCNP
export PELVIS_SCNP_FULL_CT_DATASET_DIR=/path/to/full_ct_dataset
export PELVIS_SCNP_STAGE1_ANATOMY_LABEL_DIR=/path/to/stage1_anatomy_predictions
```

If these variables are not set, paths are resolved relative to the repository root. The dataset id and name are examples only; they can be changed as long as the raw, preprocessed, and result folders are rebuilt consistently.

## Usage

The method follows a two-stage FracSegNet-style flow:

1. Generate or provide stage-1 anatomy predictions on complete CT volumes.
2. Build the stage-2 masked-CT nnU-Net dataset. CT intensity is preserved only inside the predicted target anatomy region. Fracture labels are not used to create the input mask.
3. Preprocess the stage-2 dataset and generate the training-side FDM/disMap sidecars used by the SCNP loss.
4. Train or run inference with the core `single_rf3_thr03` experiment.

Useful preparation commands:

```bash
python training/data/generate_stage1_anatomy_predictions.py
python training/data/build_stage2_masked_ct_dataset.py --reset_existing
python training/data/preprocess_stage2_training_dataset.py --reset_preprocess --configs 3d_fullres --dismap_workers 4
python training/data/audit_stage2_training_dataset.py
```

The stage-2 network input remains CT-only. Stage-1 anatomy labels are used to build masked CT volumes before training or inference; FDM/disMap sidecars are used only by the training loss.

## Train

The public training path keeps the core SCNP+FDM setting:

```bash
python training/run_experiment.py list
python training/run_experiment.py train single_rf3_thr03 --preprocess --split_mode patient --fold 0
```

Equivalent compatibility entrypoint:

```bash
python training/train_single_rf3_thr03.py --preprocess --split_mode patient --fold 0
```

The core trainer uses SCNP with receptive field `3` and an FDM threshold of `0.3`. Training targets are collapsed to the paper's three-class segmentation schema: background, main fracture segment, and secondary fragments.

## Inference

Run prediction on stage-2 masked CT images named `*_0000.nii.gz`:

```bash
python training/run_experiment.py predict single_rf3_thr03 --folds all --checkpoint checkpoint_final.pth
```

Equivalent inference-only wrapper:

```bash
python inference/predict_single_rf3_thr03.py --folds all --checkpoint checkpoint_final.pth
```

By default, inference reads `imagesTs` from the configured nnU-Net raw dataset and writes predictions under `dataset/predictions/`. Use `--input_dir` and `--output_dir` to choose other folders. The inference entrypoints do not read ground-truth labels and do not compute evaluation metrics.
