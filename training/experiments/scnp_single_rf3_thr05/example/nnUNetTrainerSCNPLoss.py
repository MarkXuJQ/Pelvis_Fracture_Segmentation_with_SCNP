from nnunetv2.training.loss.compound_losses_scnp import SCNPCEDice

from training.runtime.scnp_ct_only_trainer import nnUNetTrainerSCNPSingleRFBase


class nnUNetTrainerSCNPLoss(nnUNetTrainerSCNPSingleRFBase):
    """
    Standard single-RF SCNP trainer with CT-only network input.
    """

    scnp_loss_class = SCNPCEDice
    scnp_gate_mode = "hard_threshold"
