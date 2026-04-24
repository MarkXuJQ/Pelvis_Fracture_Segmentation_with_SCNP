# Experiments

This folder holds experiment-specific assets only.

Each experiment directory should primarily contain:

- `nnunetv2_overlay/`: the local nnUNetv2 trainer/loss overlay used by that variant.
- `example/`: supporting trainer or loss implementations referenced by the overlay.
- `__init__.py`: package marker.

User-facing training, prediction, validation, and compare entrypoints are intentionally kept outside this folder:

- training launchers live in `training/` and `training/run/`
- prediction launchers live in `inference/`
- dataset build and preprocessing live in `dataset/` and `training/data/`

This keeps experiment folders focused on what is actually different between variants.
