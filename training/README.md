# Training

Training is now organized more like the official FracSegNet codebase:
phase-oriented folders first, experiment overlays second.
The goal is to keep experiment-specific code only where the behavior is actually different.
The ROI semantic models here correspond to the official FracSegNet fracture-stage second network and can be chained after anatomy-stage checkpoints for full-CT inference.
The ROI training target now follows the official fracture-stage label space directly: `0 background / 1 main fracture segment / 2 segment 2 / 3 segment 3`, with any source fragment id `>=4` collapsed into class `3` during dataset build.

## Recommended Entry Point

Use the unified launcher when possible:

```bash
python training/run_experiment.py list
python training/run_experiment.py train single_rf3_thr03 --fold 0
python training/run_experiment.py predict multi_rf3_thr03_rf5_thr05 --fold all
python training/run_experiment.py compare single_rf5_thr03 --case_min 101 --case_max 150
```

The old top-level launchers such as `train_single_rf3_thr03.py` still exist, but they are now compatibility shims around the same shared launcher.
The same applies to the soft-variant family, including the full-image `hard_no_fdm` run.

## Layout

- `run_experiment.py`
  - Unified CLI for `train`, `predict`, `validate`, and `compare`.
- `data/`
  - Dataset build, preprocessing, and audit code.
- `run/`
  - Unified launchers plus experiment registry and train flows.
- `runtime/`
  - Shared runtime utilities such as path resolution, split generation, disMap loading, and trainer base classes.
- `evaluation/`
  - Prediction, validation, postprocessing, and full-CT comparison flows.
- `experiments/`
  - Experiment overlays only.
  - Each subfolder now mainly contains `nnunetv2_overlay/` and supporting example implementations, instead of repeating train/predict/audit wrappers.

## Structure Notes

- Default trainers, thresholds, compare settings, stage-2 ROI method names, and FracSegNet-style postprocessing defaults now live in one place: `training/run/experiment_registry.py`.
- Shared execution code is grouped by stage, closer to the official FracSegNet split between `run`, `preprocessing`, and `inference`.
- Experiment directories are now much thinner: only the code that is truly experiment-specific stays there, mainly trainer classes, loss wiring, and overlay exports.
- `experiments/scnp_soft_variants/` now contains the full soft-variant family:
  `hard_no_fdm`, `soft_no_fdm`, `scnp_soft_fdm`, and `soft_soft_fdm`.
