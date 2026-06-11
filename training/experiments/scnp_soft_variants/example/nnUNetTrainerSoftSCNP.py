from nnunetv2.training.loss.compound_losses_scnp import (
    SCNPCEDiceHardSCNP,
    SCNPCEDiceSoftFDM,
    SCNPCEDiceSoftSCNP,
    SCNPCEDiceSoftSCNPSoftFDM,
)

from training.runtime.scnp_ct_only_trainer import nnUNetTrainerSoftSCNPBase as _nnUNetTrainerSoftSCNPBase


class nnUNetTrainerSoftSCNPBase(_nnUNetTrainerSoftSCNPBase):
    """
    Shared soft-SCNP trainer family with CT-only network input.
    """

    scnp_loss_class = SCNPCEDiceSoftSCNPSoftFDM
    scnp_gate_mode = "soft_scnp_soft_fdm"


class nnUNetTrainerHardSCNPNoFDM(nnUNetTrainerSoftSCNPBase):
    """
    Hard-SCNP over the full image without a disMap loss prior.
    """

    scnp_loss_class = SCNPCEDiceHardSCNP
    scnp_gate_mode = "hard_scnp_global_no_fdm"
    use_fdm_as_loss_prior = False


class nnUNetTrainerSoftSCNPNoFDM(nnUNetTrainerSoftSCNPBase):
    """
    Soft-SCNP over the full image without a disMap loss prior.
    """

    scnp_loss_class = SCNPCEDiceSoftSCNP
    scnp_gate_mode = "soft_scnp_global_no_fdm"
    use_fdm_as_loss_prior = False


class nnUNetTrainerSCNPSoftFDM(nnUNetTrainerSoftSCNPBase):
    """
    Hard SCNP combined with a soft disMap gate.
    """

    scnp_loss_class = SCNPCEDiceSoftFDM
    scnp_gate_mode = "hard_scnp_soft_fdm"


class nnUNetTrainerSoftSCNPSoftFDM(nnUNetTrainerSoftSCNPBase):
    """
    Soft-SCNP combined with a soft disMap gate.
    """

    scnp_loss_class = SCNPCEDiceSoftSCNPSoftFDM
    scnp_gate_mode = "soft_scnp_soft_fdm"
