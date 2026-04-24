from nnunetv2.training.loss.compound_losses_scnp import SCNPCEDice

from training.runtime.scnp_ct_only_trainer import nnUNetTrainerSCNPSingleRFBase


class nnUNetTrainerSCNPLoss(nnUNetTrainerSCNPSingleRFBase):
    """
    Standard single-RF SCNP trainer with CT-only network input.
    """

    scnp_loss_class = SCNPCEDice
    scnp_gate_mode = "hard_threshold"


class nnUNetTrainerSCNPLossRF5TH03(nnUNetTrainerSCNPLoss):
    """
    Named preset for rf=5 and threshold=0.3.
    """

    scnp_gate_mode = "hard_threshold_rf5_thr03"
