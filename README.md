# Pelvis Fracture Segmentation with SCNP

This repository provides a compact nnU-Net v2 implementation of pelvis fracture segmentation with FDM-gated Same Class Neighbor Penalization (SCNP). The released code focuses on the core method: CT-only fracture segmentation with three labels, `0 background`, `1 main fracture segment`, and `2 secondary fragments`.

## Installation

Create a Python environment with nnU-Net v2, PyTorch, NumPy, SciPy, SimpleITK, nibabel, blosc2, and the usual nnU-Net medical-imaging dependencies.

The nnU-Net v2 command line tools should be available:

```bash
nnUNetv2_train --help
nnUNetv2_predict --help
```

Optional environment variables can point the scripts to a local nnU-Net workspace:

```bash
export PELVIS_SCNP_DATA_ROOT=/path/to/workspace
export PELVIS_SCNP_NNUNET_DATASET_ID=520
export PELVIS_SCNP_NNUNET_DATASET_NAME=Dataset520_PelvisSCNP
```

The expected workspace layout follows nnU-Net v2:

```text
dataset/
  nnUNet_raw_data/Dataset520_PelvisSCNP/
    imagesTr/
    labelsTr/
    imagesTs/
    dataset.json
  nnUNet_preprocessed/
  nnUNet_results/
```

`dataset.json` must define exactly the three labels used by the method:

```json
{
  "channel_names": {"0": "CT"},
  "labels": {
    "background": 0,
    "main fracture segment": 1,
    "secondary fragments": 2
  },
  "file_ending": ".nii.gz"
}
```

## Usage

The public workflow assumes that the user has already prepared a standard nnU-Net v2 dataset in the three-class label space above. The training-side FDM/disMap sidecars are generated from `labelsTr` after nnU-Net preprocessing; they are used only by the loss during training and are not model inputs during inference.

Prepare preprocessing and FDM sidecars:

```bash
python training/prepare_fdm_sidecars.py --run_preprocessing --configs 3d_fullres
```

If nnU-Net preprocessing has already been run, generate or refresh only the FDM sidecars:

```bash
python training/prepare_fdm_sidecars.py --configs 3d_fullres --overwrite_dismap
```

The network forward path remains CT-only. SCNP receives FDM/disMap only inside the training loss.

## Train

Run the core SCNP+FDM setting:

```bash
python training/train_single_rf3_thr03.py --fold 0
```

Optional unified entrypoint:

```bash
python training/run_experiment.py list
python training/run_experiment.py train single_rf3_thr03 --fold 0
```

The default experiment uses SCNP receptive field `3` and FDM threshold `0.3`. Runtime hyperparameters can be overridden with command-line options such as `--scnp_rf`, `--scnp_fdm_threshold`, `--num_epochs`, and `--initial_lr`.

## Inference

Run prediction on masked CT images named `*_0000.nii.gz`:

```bash
python inference/predict_single_rf3_thr03.py --folds 0 --checkpoint checkpoint_final.pth
```

Optional unified entrypoint:

```bash
python training/run_experiment.py predict single_rf3_thr03 --folds 0 --checkpoint checkpoint_final.pth
```

By default, inference reads `imagesTs` from the configured nnU-Net raw dataset and writes predictions under `dataset/predictions/`. Use `--input_dir` and `--output_dir` to choose other folders. The inference entrypoints read only CT images and do not read ground-truth labels or compute evaluation metrics.
