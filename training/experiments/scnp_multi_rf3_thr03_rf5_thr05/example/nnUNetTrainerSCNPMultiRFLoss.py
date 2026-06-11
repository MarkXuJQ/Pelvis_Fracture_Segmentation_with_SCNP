from nnunetv2.training.loss.compound_losses_scnp import SCNPCEDiceMultiRF

from training.runtime.scnp_ct_only_trainer import nnUNetTrainerSCNPMultiRFBase


class nnUNetTrainerSCNPMultiRFLoss(nnUNetTrainerSCNPMultiRFBase):
    """
    Multi-RF SCNP trainer: CT-only forward with FracSegNet-style disMap prior
    used only inside the training loss.
    """

    scnp_loss_class = SCNPCEDiceMultiRF
    scnp_gate_mode = "multi_rf3_thr03_rf5_thr05"
