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
class PredictDefaults:
    description: str
    default_trainer: str
    method_name: str
    required_overlay_files: tuple[str, ...]
    apply_global_lcc_by_default: bool = True


@dataclass(frozen=True)
class ValidateDefaults:
    default_method_name: str
    default_trainer: str


@dataclass(frozen=True)
class CompareDefaults:
    description: str
    default_semantic_trainer: str
    validation_method_name: str
    required_overlay_files: tuple[str, ...]


@dataclass(frozen=True)
class ExperimentSpec:
    key: str
    label: str
    family: TaskFamily
    task_dir_name: str
    train: TrainDefaults
    predict: PredictDefaults | None = None
    validate: ValidateDefaults | None = None
    compare: CompareDefaults | None = None
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
            description="Train the multi-RF (rf=3 @ thr=0.3, rf=5 @ thr=0.5) SCNP experiment on Dataset503.",
            default_trainer="nnUNetTrainerSCNPMultiRFLoss",
            trainer_choices=("nnUNetTrainerSCNPMultiRFLoss",),
            default_rf_low=3,
            default_rf_high=5,
            default_fdm_low=0.3,
            default_fdm_high=0.5,
        ),
        predict=PredictDefaults(
            description="Predict and validate the multi-RF (rf=3 @ thr=0.3, rf=5 @ thr=0.5) SCNP experiment on Dataset503.",
            default_trainer="nnUNetTrainerSCNPMultiRFLoss",
            method_name="nnUNetTrainerSCNPMultiRFLoss",
            required_overlay_files=COMMON_OVERLAY_FILES,
            apply_global_lcc_by_default=True,
        ),
        validate=ValidateDefaults(
            default_method_name="nnUNetTrainerSCNPMultiRFLoss",
            default_trainer="nnUNetTrainerSCNPMultiRFLoss",
        ),
        compare=CompareDefaults(
            description=(
                "Compare Dataset503 semantic model performance on cases 101-150 between direct ROI inference "
                "and FracSegNet-style full-CT -> anatomy -> ROI inference."
            ),
            default_semantic_trainer="nnUNetTrainerSCNPMultiRFLoss",
            validation_method_name="nnUNetTrainerSCNPMultiRFLoss",
            required_overlay_files=COMMON_OVERLAY_FILES,
        ),
        aliases=("multi_rf0307", "scnp_multi_rf0307"),
    ),
    ExperimentSpec(
        key="single_rf3_thr05",
        label="Single RF=3 thr=0.5",
        family="single",
        task_dir_name="scnp_single_rf3_thr05",
        train=TrainDefaults(
            description="Train the single-RF rf=3, threshold=0.5 SCNP experiment on Dataset503.",
            default_trainer="nnUNetTrainerSCNPLoss",
            trainer_choices=("nnUNetTrainerSCNPLoss",),
            default_rf=3,
            default_fdm_threshold=0.5,
        ),
        predict=PredictDefaults(
            description="Predict and validate the single-RF rf=3, threshold=0.5 SCNP experiment on Dataset503.",
            default_trainer="nnUNetTrainerSCNPLoss",
            method_name="nnUNetTrainerSCNPLoss",
            required_overlay_files=COMMON_OVERLAY_FILES,
            apply_global_lcc_by_default=True,
        ),
        validate=ValidateDefaults(
            default_method_name="nnUNetTrainerSCNPLoss",
            default_trainer="nnUNetTrainerSCNPLoss",
        ),
        compare=CompareDefaults(
            description=(
                "Compare Dataset503 semantic model performance on cases 101-150 between direct ROI inference "
                "and FracSegNet-style full-CT -> anatomy -> ROI inference."
            ),
            default_semantic_trainer="nnUNetTrainerSCNPLoss",
            validation_method_name="nnUNetTrainerSCNPLoss",
            required_overlay_files=COMMON_OVERLAY_FILES,
        ),
        aliases=("single_rf05", "scnp_single_rf05", "rf05"),
    ),
    ExperimentSpec(
        key="single_rf3_thr03",
        label="Single RF=3 thr=0.3",
        family="single",
        task_dir_name="scnp_single_rf3_thr03",
        train=TrainDefaults(
            description="Train the rf=3, threshold=0.3 single-RF SCNP experiment on Dataset503.",
            default_trainer="nnUNetTrainerSCNPLossRF3TH03",
            trainer_choices=(
                "nnUNetTrainerSCNPLoss",
                "nnUNetTrainerSCNPLossRF3TH03",
                "nnUNetTrainerSCNPLossRF3TH03TrueDisMap",
            ),
            default_rf=3,
            default_fdm_threshold=0.3,
        ),
        predict=PredictDefaults(
            description="Predict and validate the rf=3, threshold=0.3 single-RF SCNP experiment on Dataset503.",
            default_trainer="nnUNetTrainerSCNPLossRF3TH03",
            method_name="nnUNetTrainerSCNPLossRF3TH03",
            required_overlay_files=COMMON_OVERLAY_FILES,
            apply_global_lcc_by_default=True,
        ),
        validate=ValidateDefaults(
            default_method_name="nnUNetTrainerSCNPLossRF3TH03",
            default_trainer="nnUNetTrainerSCNPLossRF3TH03",
        ),
        compare=CompareDefaults(
            description=(
                "Compare Dataset503 semantic model performance on cases 101-150 between direct ROI inference "
                "and FracSegNet-style full-CT -> anatomy -> ROI inference."
            ),
            default_semantic_trainer="nnUNetTrainerSCNPLoss",
            validation_method_name="nnUNetTrainerSCNPLossRF3TH03",
            required_overlay_files=COMMON_OVERLAY_FILES,
        ),
        aliases=("scnp_single_rf3_thr03", "rf3_thr03"),
    ),
    ExperimentSpec(
        key="single_rf5_thr03",
        label="Single RF=5 thr=0.3",
        family="single",
        task_dir_name="scnp_single_rf5_thr03",
        train=TrainDefaults(
            description="Train the rf=5, threshold=0.3 single-RF SCNP experiment on Dataset503.",
            default_trainer="nnUNetTrainerSCNPLossRF5TH03",
            trainer_choices=(
                "nnUNetTrainerSCNPLoss",
                "nnUNetTrainerSCNPLossRF5TH03",
            ),
            default_rf=5,
            default_fdm_threshold=0.3,
        ),
        predict=PredictDefaults(
            description="Predict and validate the rf=5, threshold=0.3 single-RF SCNP experiment on Dataset503.",
            default_trainer="nnUNetTrainerSCNPLossRF5TH03",
            method_name="nnUNetTrainerSCNPLossRF5TH03",
            required_overlay_files=COMMON_OVERLAY_FILES,
            apply_global_lcc_by_default=True,
        ),
        validate=ValidateDefaults(
            default_method_name="nnUNetTrainerSCNPLossRF5TH03",
            default_trainer="nnUNetTrainerSCNPLossRF5TH03",
        ),
        compare=CompareDefaults(
            description=(
                "Compare Dataset503 semantic model performance on cases 101-150 between direct ROI inference "
                "and FracSegNet-style full-CT -> anatomy -> ROI inference."
            ),
            default_semantic_trainer="nnUNetTrainerSCNPLossRF5TH03",
            validation_method_name="nnUNetTrainerSCNPLossRF5TH03",
            required_overlay_files=COMMON_OVERLAY_FILES,
        ),
        aliases=("scnp_single_rf5_thr03", "rf5_thr03"),
    ),
    ExperimentSpec(
        key="single_no_threshold",
        label="Single no-threshold",
        family="single",
        task_dir_name="scnp_single_rf_no_threshold",
        train=TrainDefaults(
            description="Train the no-threshold single-RF SCNP experiment on Dataset503.",
            default_trainer="nnUNetTrainerSCNPLossNoThreshold",
            trainer_choices=(
                "nnUNetTrainerSCNPLoss",
                "nnUNetTrainerSCNPLossNoThreshold",
            ),
            default_rf=3,
            default_fdm_threshold=0.3,
        ),
        predict=PredictDefaults(
            description="Predict and validate the no-threshold single-RF SCNP experiment on Dataset503.",
            default_trainer="nnUNetTrainerSCNPLossNoThreshold",
            method_name="nnUNetTrainerSCNPLossNoThreshold",
            required_overlay_files=COMMON_OVERLAY_FILES,
            apply_global_lcc_by_default=True,
        ),
        validate=ValidateDefaults(
            default_method_name="nnUNetTrainerSCNPLossNoThreshold",
            default_trainer="nnUNetTrainerSCNPLossNoThreshold",
        ),
        compare=CompareDefaults(
            description=(
                "Compare Dataset503 semantic model performance on cases 101-150 between direct ROI inference "
                "and FracSegNet-style full-CT -> anatomy -> ROI inference."
            ),
            default_semantic_trainer="nnUNetTrainerSCNPLossNoThreshold",
            validation_method_name="nnUNetTrainerSCNPLossNoThreshold",
            required_overlay_files=COMMON_OVERLAY_FILES,
        ),
        aliases=("scnp_single_rf_no_threshold", "single_rf_no_threshold", "no_threshold"),
    ),
    ExperimentSpec(
        key="soft_no_fdm",
        label="Soft-SCNP no FDM",
        family="soft",
        task_dir_name="scnp_soft_variants",
        train=TrainDefaults(
            description="Train one of the soft-variant SCNP ablation experiments on Dataset503.",
            default_trainer="nnUNetTrainerSoftSCNPNoFDM",
            trainer_choices=SOFT_TRAINER_CHOICES,
            default_rf=3,
            default_fdm_threshold=0.5,
        ),
        predict=PredictDefaults(
            description="Predict and validate a soft-variant SCNP experiment on Dataset503.",
            default_trainer="nnUNetTrainerSoftSCNPNoFDM",
            method_name="nnUNetTrainerSoftSCNPNoFDM",
            required_overlay_files=COMMON_OVERLAY_FILES,
            apply_global_lcc_by_default=True,
        ),
        validate=ValidateDefaults(
            default_method_name="nnUNetTrainerSoftSCNPNoFDM",
            default_trainer="nnUNetTrainerSoftSCNPNoFDM",
        ),
        aliases=("soft_scnp_no_fdm",),
    ),
    ExperimentSpec(
        key="soft_soft_fdm",
        label="Soft-SCNP soft FDM",
        family="soft",
        task_dir_name="scnp_soft_variants",
        train=TrainDefaults(
            description="Train one of the soft-variant SCNP ablation experiments on Dataset503.",
            default_trainer="nnUNetTrainerSoftSCNPSoftFDM",
            trainer_choices=SOFT_TRAINER_CHOICES,
            default_rf=3,
            default_fdm_threshold=0.5,
        ),
        predict=PredictDefaults(
            description="Predict and validate a soft-variant SCNP experiment on Dataset503.",
            default_trainer="nnUNetTrainerSoftSCNPSoftFDM",
            method_name="nnUNetTrainerSoftSCNPSoftFDM",
            required_overlay_files=COMMON_OVERLAY_FILES,
            apply_global_lcc_by_default=True,
        ),
        validate=ValidateDefaults(
            default_method_name="nnUNetTrainerSoftSCNPSoftFDM",
            default_trainer="nnUNetTrainerSoftSCNPSoftFDM",
        ),
        aliases=("soft_scnp_soft_fdm",),
    ),
    ExperimentSpec(
        key="scnp_soft_fdm",
        label="SCNP soft FDM",
        family="soft",
        task_dir_name="scnp_soft_variants",
        train=TrainDefaults(
            description="Train one of the soft-variant SCNP ablation experiments on Dataset503.",
            default_trainer="nnUNetTrainerSCNPSoftFDM",
            trainer_choices=SOFT_TRAINER_CHOICES,
            default_rf=3,
            default_fdm_threshold=0.5,
        ),
        predict=PredictDefaults(
            description="Predict and validate a soft-variant SCNP experiment on Dataset503.",
            default_trainer="nnUNetTrainerSCNPSoftFDM",
            method_name="nnUNetTrainerSCNPSoftFDM",
            required_overlay_files=COMMON_OVERLAY_FILES,
            apply_global_lcc_by_default=True,
        ),
        validate=ValidateDefaults(
            default_method_name="nnUNetTrainerSCNPSoftFDM",
            default_trainer="nnUNetTrainerSCNPSoftFDM",
        ),
        aliases=("train_scnp_soft_fdm",),
    ),
    ExperimentSpec(
        key="hard_no_fdm",
        label="Hard-SCNP no FDM",
        family="soft",
        task_dir_name="scnp_soft_variants",
        train=TrainDefaults(
            description="Train one of the soft-variant SCNP ablation experiments on Dataset503.",
            default_trainer="nnUNetTrainerHardSCNPNoFDM",
            trainer_choices=SOFT_TRAINER_CHOICES,
            default_rf=3,
            default_fdm_threshold=0.5,
        ),
        predict=PredictDefaults(
            description="Predict and validate a soft-variant SCNP experiment on Dataset503.",
            default_trainer="nnUNetTrainerHardSCNPNoFDM",
            method_name="nnUNetTrainerHardSCNPNoFDM",
            required_overlay_files=COMMON_OVERLAY_FILES,
            apply_global_lcc_by_default=True,
        ),
        validate=ValidateDefaults(
            default_method_name="nnUNetTrainerHardSCNPNoFDM",
            default_trainer="nnUNetTrainerHardSCNPNoFDM",
        ),
        aliases=("hard_scnp_no_fdm",),
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
    if action == "predict":
        return spec.predict is not None
    if action == "validate":
        return spec.validate is not None
    if action == "compare":
        return spec.compare is not None
    raise ValueError(f"Unsupported action: {action}")


def list_experiment_specs(action: str | None = None) -> list[ExperimentSpec]:
    if action is None:
        return list(EXPERIMENT_SPECS)
    return [spec for spec in EXPERIMENT_SPECS if supports_action(spec, action)]
