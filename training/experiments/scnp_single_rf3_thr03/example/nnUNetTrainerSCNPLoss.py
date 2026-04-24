from nnunetv2.training.loss.compound_losses_scnp import SCNPCEDice

from training.runtime.scnp_ct_only_trainer import nnUNetTrainerSCNPSingleRFBase


class nnUNetTrainerSCNPLoss(nnUNetTrainerSCNPSingleRFBase):
    """
    Standard single-RF SCNP trainer: CT-only network forward with a training-only
    disMap prior loaded from the preprocessed cache.
    """

    scnp_loss_class = SCNPCEDice
    scnp_gate_mode = "hard_threshold"


class nnUNetTrainerSCNPLossRF3TH03(nnUNetTrainerSCNPLoss):
    """
    Named preset for rf=3 and threshold=0.3.
    """

    pass


class nnUNetTrainerSCNPLossRF3TH03TrueDisMap(nnUNetTrainerSCNPLossRF3TH03):
    pass
