## Pelvis SCNP Workspace

This repository is a standalone workspace for Pelvis SCNP experiments.
It keeps the source data snapshot, reusable nnU-Net caches, experiment entrypoints, and local nnUNetv2 overlays in one place.

Directory layout:

- `source/`: original ROI `CT + GT`
- `dataset/`: dataset build and preprocess entry scripts
- `dataset/nnUNet_raw_data/`: built CT-only raw dataset
- `dataset/nnUNet_preprocessed/`: reusable nnUNet preprocessing cache and training-only `disMap` sidecars
- `dataset/nnUNet_results/`: model outputs
- `training/`: training package, now split by stage like the official FracSegNet layout
- `inference/`: prediction and validation entry scripts

Run all commands inside an environment that provides `nnUNetv2`.

Pipeline semantics follow the official FracSegNet fracture stage:

- `source/images` and `source/labels` store ROI fracture cases.
- `source/labels` store ROI fragment instance ids with the main fracture segment encoded as `1` and additional fragments encoded as `2..N`.
- Our ROI model corresponds to the official FracSegNet fracture-stage second network and can be run after an anatomy-stage checkpoint for full-CT segmentation.
- The raw nnU-Net dataset keeps CT-only inputs and official fracture-stage semantic targets `0 background / 1 main fracture segment / 2 segment 2 / 3 segment 3`.
- When a source ROI contains more than three foreground fragments, source labels `>=4` are collapsed into `segment 3` so the training target stays inside the official FracSegNet 4-class label space.
- Preprocessing generates a single-channel FracSegNet-style `disMap` for training cases only.
- The `disMap` prior still follows the official FracSegNet rule of contrasting `label 1` against all non-main fragments `>=2`.
- Network forward during both training and inference is CT-only; `disMap` is used only as a training-time loss prior.

Code organization follows the same principle:

- `training/data/` holds dataset build, preprocessing, and audit logic.
- `training/run/` holds unified launchers and experiment registry code.
- `training/runtime/` holds path, split, disMap, and trainer-base utilities.
- `training/evaluation/` holds prediction, validation, postprocessing, and full-CT comparison logic.
- `training/experiments/` holds experiment-specific overlays and supporting example code only.
- Prediction entrypoints under `inference/` call the same local repository code and do not depend on any external project checkout.

### Build Dataset

```bash
python dataset/build_dataset.py
```

Use `python dataset/build_dataset.py --reset_existing` only when you want to rebuild the raw dataset from `source/`.

### Preprocess Once And Reuse

```bash
python dataset/preprocess_dataset.py
```

Optional dataset audit:

```bash
python dataset/audit_dataset.py --dataset_dir dataset/nnUNet_raw_data/Dataset503_SCNP --preprocessed_dataset_dir dataset/nnUNet_preprocessed/Dataset503_SCNP
```

If the preprocessed cache and `disMap` sidecars already exist, the script reuses them automatically.
The raw dataset layout is also reused when the expected ROI cases are already present.
Offline preprocessing covers `imagesTr` only; `imagesTs` remains the held-out test split for prediction and evaluation.

### Train

Unified launcher:

```bash
python training/run_experiment.py list
python training/run_experiment.py train single_rf3_thr03 --preprocess --split_mode patient --fold 0
```

Compatibility launchers:

Single-RF `rf=3`, threshold `0.5`:

```bash
python training/train_single_rf3_thr05.py --split_mode patient --fold 0
```

Single-RF `rf=3`, threshold `0.3`:

```bash
python training/train_single_rf3_thr03.py --preprocess --split_mode patient --fold 0
```

Single-RF `rf=5`, threshold `0.3`:

```bash
python training/train_single_rf5_thr03.py --split_mode patient --fold 0
```

Single-RF without threshold:

```bash
python training/train_single_no_threshold.py --split_mode patient --fold 0
```

Multi-RF:

```bash
python training/train_multi_rf3_thr03_rf5_thr05.py --split_mode patient --fold 0
```

Soft-SCNP without FDM prior:

```bash
python training/train_soft_no_fdm.py --split_mode patient --fold 0
```

Soft-SCNP with soft FDM prior:

```bash
python training/train_soft_soft_fdm.py --split_mode patient --fold 0
```

### Predict

```bash
python training/run_experiment.py predict single_rf3_thr03 --fold all
```

Compatibility launcher:

```bash
python inference/predict_single_rf3_thr03.py --fold all
```

All inference wrapper entrypoints now default to the same global-LCC cleanup used by the official FracSegNet export flow.
