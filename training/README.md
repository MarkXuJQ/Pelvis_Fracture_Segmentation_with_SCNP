# Training

Training is now organized more like the official FracSegNet codebase:
phase-oriented folders first, experiment overlays second.
The goal is to keep experiment-specific code only where the behavior is actually different.
The stage-2 semantic models here correspond to the official FracSegNet fracture-stage second network.
The training dataset follows the official FracSegNet two-stage data flow: stage 1 predicts an anatomy label on the complete raw CT, then stage 2 receives a CT-only masked image where voxels are preserved only inside the target stage-1 anatomy label (`1 sacrum / 2 left hip / 3 right hip`).
Fracture GT is not used to build the input mask; it is used only as the stage-2 supervision label and to generate the `disMap` prior consumed by the SCNP loss.
The stage-2 training target follows the thesis semantic label space directly: `0 background / 1 main fracture segment / 2 secondary fragments`, with any source fragment id `>=2` collapsed into class `2` during dataset build.

## Recommended Entry Point

Use the unified launcher when possible:

```bash
python training/run_experiment.py list
python training/run_experiment.py train single_rf3_thr03 --fold 0
python training/run_experiment.py predict single_rf3_thr03 --folds all
```

The old top-level launchers such as `train_single_rf3_thr03.py` still exist, but they are now compatibility shims around the same shared launcher.
The same applies to the soft-variant family, including the full-image `hard_no_fdm` run.

## Layout

- `run_experiment.py`
  - Unified CLI for train and inference-only prediction.
- `data/`
  - Dataset build, preprocessing, and audit code.
- `run/`
  - Unified launchers plus experiment registry and train flows.
- `runtime/`
  - Shared runtime utilities such as path resolution, split generation, disMap loading, and trainer base classes.
- `experiments/`
  - Experiment overlays only.
  - Each subfolder now mainly contains `nnunetv2_overlay/` and supporting example implementations, instead of repeating train/predict/audit wrappers.

## Structure Notes

- Default trainers and SCNP/FDM thresholds now live in one place: `training/run/experiment_registry.py`.
- Shared execution code is grouped by stage, closer to the official FracSegNet split between `run` and preprocessing code.
- Training launchers verify the stage-2 raw/preprocessed manifests so stale cropped-ROI or fracture-GT-masked preprocessing caches are rejected before training.
- Experiment directories are now much thinner: only the code that is truly experiment-specific stays there, mainly trainer classes, loss wiring, and overlay exports.
- `experiments/scnp_soft_variants/` now contains the full soft-variant family:
  `hard_no_fdm`, `soft_no_fdm`, `scnp_soft_fdm`, and `soft_soft_fdm`.
