# Soft Variant Family

This experiment family keeps the same CT-only network input and dataset setup as the other
SCNP runs, but changes how SCNP is propagated and whether a `disMap` prior is used inside the loss.

Available variants:

- `hard_no_fdm`
  Hard SCNP applied over the full image, with no `disMap` loss prior.
- `soft_no_fdm`
  Soft SCNP applied over the full image, with no `disMap` loss prior.
- `scnp_soft_fdm`
  Hard SCNP combined with a soft sigmoid `disMap` gate.
- `soft_soft_fdm`
  Soft SCNP combined with a soft sigmoid `disMap` gate.

Implementation layout:

- `example/compound_losses_scnp.py`
  Soft-variant CE+Dice losses.
- `example/nnUNetTrainerSoftSCNP.py`
  Trainer definitions for the four variants.
- `nnunetv2_overlay/`
  Minimal nnUNetv2 overlay exports used by local training and prediction entrypoints.
