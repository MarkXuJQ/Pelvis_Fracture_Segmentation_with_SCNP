from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


TaskFamily = Literal["single", "multi", "soft"]

TRAINING_ROOT = Path(__file__).resolve().parents[1]
COMMON_OVERLAY_FILES = (
    "nnunetv2/training/nnUNetTrainer/{trainer}.py",
    "nnunetv2/training/loss/compound_losses_scnp.py",
)
SOFT_TRAINER_CHOICES = (
    "nnUNetTrainerHardSCNPNoFDM",
    "nnUNetTrainerSoftSCNPNoFDM",
    "nnUNetTrainerSCNPSoftFDM",
    "nnUNetTrainerSoftSCNPSoftFDM",
)


@dataclass(frozen=True)
class TrainDefaults:
    description: str
    default_trainer: str
    trainer_choices: tuple[str, ...]
    default_rf: int | None = None
    default_fdm_threshold: float | None = None
    default_rf_low: int | None = None
    default_rf_high: int | None = None
    default_fdm_low: float | None = None
    default_fdm_high: float | None = None


@dataclass(frozen=True)
class ExperimentSpec:
    key: str
    label: str
    family: TaskFamily
    task_dir_name: str
    train: TrainDefaults
    aliases: tuple[str, ...] = ()

    @property
    def task_dir(self) -> Path:
        return TRAINING_ROOT / "experiments" / self.task_dir_name


EXPERIMENT_SPECS = (
    ExperimentSpec(
        key="multi_rf3_thr03_rf5_thr05",
        label="Multi-RF rf3@0.3 rf5@0.5",
        family="multi",
        task_dir_name="scnp_multi_rf3_thr03_rf5_thr05",
        train=TrainDefaults(
            description="Train the multi-RF (rf=3 @ thr=0.3, rf=5 @ thr=0.5) SCNP experiment on the stage-2 masked-CT training dataset.",
            default_trainer="nnUNetTrainerSCNPMultiRFLoss",
            trainer_choices=("nnUNetTrainerSCNPMultiRFLoss",),
            default_rf_low=3,
            default_rf_high=5,
            default_fdm_low=0.3,
            default_fdm_high=0.5,
        ),
        aliases=("multi_rf0307", "scnp_multi_rf0307"),
    ),
    ExperimentSpec(
        key="single_rf3_thr05",
        label="Single RF=3 thr=0.5",
        family="single",
        task_dir_name="scnp_single_rf3_thr05",
        train=TrainDefaults(
            description="Train the single-RF rf=3, threshold=0.5 SCNP experiment on the stage-2 masked-CT training dataset.",
            default_trainer="nnUNetTrainerSCNPLoss",
            trainer_choices=("nnUNetTrainerSCNPLoss",),
            default_rf=3,
            default_fdm_threshold=0.5,
        ),
        aliases=("single_rf05", "scnp_single_rf05", "rf05"),
    ),
    ExperimentSpec(
        key="single_rf3_thr03",
        label="Single RF=3 thr=0.3",
        family="single",
        task_dir_name="scnp_single_rf3_thr03",
        train=TrainDefaults(
            description="Train the rf=3, threshold=0.3 single-RF SCNP experiment on the stage-2 masked-CT training dataset.",
            default_trainer="nnUNetTrainerSCNPLossRF3TH03",
            trainer_choices=(
                "nnUNetTrainerSCNPLoss",
                "nnUNetTrainerSCNPLossRF3TH03",
                "nnUNetTrainerSCNPLossRF3TH03TrueDisMap",
            ),
            default_rf=3,
            default_fdm_threshold=0.3,
        ),
        aliases=("scnp_single_rf3_thr03", "rf3_thr03"),
    ),
    ExperimentSpec(
        key="single_rf5_thr03",
        label="Single RF=5 thr=0.3",
        family="single",
        task_dir_name="scnp_single_rf5_thr03",
        train=TrainDefaults(
            description="Train the rf=5, threshold=0.3 single-RF SCNP experiment on the stage-2 masked-CT training dataset.",
            default_trainer="nnUNetTrainerSCNPLossRF5TH03",
            trainer_choices=(
                "nnUNetTrainerSCNPLoss",
                "nnUNetTrainerSCNPLossRF5TH03",
            ),
            default_rf=5,
            default_fdm_threshold=0.3,
        ),
        aliases=("scnp_single_rf5_thr03", "rf5_thr03"),
    ),
    ExperimentSpec(
        key="single_no_threshold",
        label="Single no-threshold",
        family="single",
        task_dir_name="scnp_single_rf_no_threshold",
        train=TrainDefaults(
            description="Train the no-threshold single-RF SCNP experiment on the stage-2 masked-CT training dataset.",
            default_trainer="nnUNetTrainerSCNPLossNoThreshold",
            trainer_choices=(
                "nnUNetTrainerSCNPLoss",
                "nnUNetTrainerSCNPLossNoThreshold",
            ),
            default_rf=3,
            default_fdm_threshold=0.3,
        ),
        aliases=("scnp_single_rf_no_threshold", "single_rf_no_threshold", "no_threshold"),
    ),
    ExperimentSpec(
        key="soft_no_fdm",
        label="Soft-SCNP global no FDM",
        family="soft",
        task_dir_name="scnp_soft_variants",
        train=TrainDefaults(
            description="Train the soft-SCNP full-image variant without any disMap loss prior on the stage-2 masked-CT training dataset.",
            default_trainer="nnUNetTrainerSoftSCNPNoFDM",
            trainer_choices=SOFT_TRAINER_CHOICES,
            default_rf=3,
            default_fdm_threshold=0.5,
        ),
        aliases=("soft_scnp_no_fdm", "soft_global_no_fdm"),
    ),
    ExperimentSpec(
        key="soft_soft_fdm",
        label="Soft-SCNP soft FDM",
        family="soft",
        task_dir_name="scnp_soft_variants",
        train=TrainDefaults(
            description="Train the soft-SCNP variant with a sigmoid soft-FDM gate on the stage-2 masked-CT training dataset.",
            default_trainer="nnUNetTrainerSoftSCNPSoftFDM",
            trainer_choices=SOFT_TRAINER_CHOICES,
            default_rf=3,
            default_fdm_threshold=0.5,
        ),
        aliases=("soft_scnp_soft_fdm",),
    ),
    ExperimentSpec(
        key="scnp_soft_fdm",
        label="SCNP soft FDM",
        family="soft",
        task_dir_name="scnp_soft_variants",
        train=TrainDefaults(
            description="Train the hard-SCNP variant with a sigmoid soft-FDM gate on the stage-2 masked-CT training dataset.",
            default_trainer="nnUNetTrainerSCNPSoftFDM",
            trainer_choices=SOFT_TRAINER_CHOICES,
            default_rf=3,
            default_fdm_threshold=0.5,
        ),
        aliases=("train_scnp_soft_fdm", "hard_soft_fdm"),
    ),
    ExperimentSpec(
        key="hard_no_fdm",
        label="Hard-SCNP global no FDM",
        family="soft",
        task_dir_name="scnp_soft_variants",
        train=TrainDefaults(
            description="Train the hard-SCNP full-image variant without any disMap loss prior on the stage-2 masked-CT training dataset.",
            default_trainer="nnUNetTrainerHardSCNPNoFDM",
            trainer_choices=SOFT_TRAINER_CHOICES,
            default_rf=3,
            default_fdm_threshold=0.5,
        ),
        aliases=("hard_scnp_no_fdm", "hard_scnp_global_no_fdm", "hard_no_FDM"),
    ),
)


_SPEC_BY_KEY = {spec.key: spec for spec in EXPERIMENT_SPECS}
_SPEC_BY_ALIAS = {
    alias: spec
    for spec in EXPERIMENT_SPECS
    for alias in (spec.key, *spec.aliases)
}


def get_experiment_spec(name: str) -> ExperimentSpec:
    try:
        return _SPEC_BY_ALIAS[name]
    except KeyError as exc:
        available = ", ".join(spec.key for spec in EXPERIMENT_SPECS)
        raise KeyError(f"Unknown experiment '{name}'. Available: {available}") from exc


def supports_action(spec: ExperimentSpec, action: str) -> bool:
    if action == "train":
        return True
    raise ValueError(f"Unsupported action: {action}")


def list_experiment_specs(action: str | None = None) -> list[ExperimentSpec]:
    if action is None:
        return list(EXPERIMENT_SPECS)
    return [spec for spec in EXPERIMENT_SPECS if supports_action(spec, action)]
