# Training

This package contains the training-side code for the released SCNP+FDM method example.

The public path is intentionally narrow:

- `method/` contains the core FDM calculation, disMap dataloader, SCNP loss, and SCNP nnU-Net trainer.
- `prepare_fdm_sidecars.py` prepares training-side FDM/disMap sidecars for a standard nnU-Net v2 dataset.
- `run/` contains the unified launcher and the single registered experiment.
- `experiments/scnp_single_rf3_thr03/` contains the thin nnU-Net v2 overlay needed by `nnUNetv2_train`.

The label schema is `0 background / 1 main fracture segment / 2 secondary fragments`. Ground-truth labels are used as training supervision and to prepare FDM/disMap sidecars; they are not used as network input or inference-time dependencies.

Recommended commands:

```bash
python training/prepare_fdm_sidecars.py --run_preprocessing --configs 3d_fullres
python training/train_single_rf3_thr03.py --fold 0
python inference/predict_single_rf3_thr03.py --folds 0 --checkpoint checkpoint_final.pth
```

The unified launcher remains available as an optional wrapper:

```bash
python training/run_experiment.py list
python training/run_experiment.py train single_rf3_thr03 --fold 0
python training/run_experiment.py predict single_rf3_thr03 --folds 0 --checkpoint checkpoint_final.pth
```
