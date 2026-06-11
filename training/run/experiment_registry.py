from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class TrainDefaults:
    description: str
    default_trainer: str
    trainer_choices: tuple[str, ...]
    default_rf: int | None = None
    default_fdm_threshold: float | None = None


@dataclass(frozen=True)
class ExperimentSpec:
    key: str
    label: str
    task_dir_name: str
    train: TrainDefaults
    aliases: tuple[str, ...] = ()

    @property
    def task_dir(self) -> Path:
        return TRAINING_ROOT / "experiments" / self.task_dir_name


EXPERIMENT_SPECS = (
    ExperimentSpec(
        key="single_rf3_thr03",
        label="Single RF=3 thr=0.3",
        task_dir_name="scnp_single_rf3_thr03",
        train=TrainDefaults(
            description="Train the rf=3, threshold=0.3 single-RF SCNP experiment on the stage-2 masked-CT training dataset.",
            default_trainer="nnUNetTrainerSCNPLossRF3TH03",
            trainer_choices=(
                "nnUNetTrainerSCNPLoss",
                "nnUNetTrainerSCNPLossRF3TH03",
            ),
            default_rf=3,
            default_fdm_threshold=0.3,
        ),
        aliases=("scnp_single_rf3_thr03", "rf3_thr03"),
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
    if action in {"train", "predict"}:
        return True
    raise ValueError(f"Unsupported action: {action}")


def list_experiment_specs(action: str | None = None) -> list[ExperimentSpec]:
    if action is None:
        return list(EXPERIMENT_SPECS)
    return [spec for spec in EXPERIMENT_SPECS if supports_action(spec, action)]
