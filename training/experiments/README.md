# Experiments

Only the core paper method example is kept here:

- `scnp_single_rf3_thr03/`

This experiment contains the nnU-Net v2 overlay and example implementation for FDM-gated SCNP training with receptive field `3` and FDM threshold `0.3`.

The overlay exports:

- `nnUNetTrainerSCNPLoss`
- `nnUNetTrainerSCNPLossRF3TH03`
- `SCNPCEDice`

Other ablation and comparison variants are intentionally left out of this release so the repository stays focused on the main method.
