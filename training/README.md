# Training

This package contains the training-side code for the released SCNP+FDM method example.

The public path is intentionally narrow:

- `data/` builds and audits the stage-2 masked-CT nnU-Net dataset.
- `run/` contains the unified launcher and the single registered experiment.
- `runtime/` contains shared path, FDM/disMap, split, and trainer-base utilities.
- `experiments/scnp_single_rf3_thr03/` contains the core nnU-Net v2 overlay for SCNP with FDM gating.

The stage-2 label schema is `0 background / 1 main fracture segment / 2 secondary fragments`. Fracture ground truth is used as supervision and to prepare training-side FDM/disMap sidecars; it is not used as a model input or as an inference-time dependency.

Recommended commands:

```bash
python training/run_experiment.py list
python training/run_experiment.py train single_rf3_thr03 --preprocess --split_mode patient --fold 0
python training/run_experiment.py predict single_rf3_thr03 --folds all --checkpoint checkpoint_final.pth
```
