import os

from nnunetv2.training.loss.compound_losses_scnp import SCNPCEDice, SCNPCEDiceNoThreshold

from training.runtime.scnp_ct_only_trainer import nnUNetTrainerSCNPSingleRFBase


class nnUNetTrainerSCNPLoss(nnUNetTrainerSCNPSingleRFBase):
    """
    Standard thresholded single-RF SCNP trainer.
    """

    scnp_loss_class = SCNPCEDice
    scnp_gate_mode = "hard_threshold"


class nnUNetTrainerSCNPLossNoThreshold(nnUNetTrainerSCNPLoss):
    """
    Single-RF SCNP variant without a hard FDM threshold.
    """

    scnp_loss_class = SCNPCEDiceNoThreshold
    scnp_gate_mode = "continuous_no_threshold"

    def _scnp_gate_log_message(self) -> tuple[str, ...]:
        return (
            "SCNP single-RF gating:",
            f"mode={self.scnp_gate_mode},",
            "fdm_threshold=disabled,",
            f"fdm_power={os.environ.get('NNUNET_SCNP_FDM_POWER', '1.0')},",
            f"rf={os.environ.get('NNUNET_SCNP_RF', '3')},",
            "network_input=ct_only,",
            "fdm_loss_prior=on",
        )
