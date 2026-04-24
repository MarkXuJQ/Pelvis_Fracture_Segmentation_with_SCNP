from __future__ import annotations

import os
import sys
from pathlib import Path

from training.run.experiment_registry import get_experiment_spec


def configure_top_level_environment(script_path: str | Path) -> Path:
    repo_root = Path(script_path).resolve().parents[1]
    os.environ.setdefault("PELVIS_SCNP_DATA_ROOT", str(repo_root))
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def run_named_train(experiment_name: str) -> None:
    spec = get_experiment_spec(experiment_name)
    if spec.family == "single":
        from training.run.train_single_rf import main as train_single_rf_main

        train_defaults = spec.train
        train_single_rf_main(
            task_dir=spec.task_dir,
            default_trainer=train_defaults.default_trainer,
            trainer_choices=list(train_defaults.trainer_choices),
            default_rf=int(train_defaults.default_rf),
            default_fdm_threshold=float(train_defaults.default_fdm_threshold),
            description=train_defaults.description,
        )
        return

    if spec.family == "multi":
        from training.run.train_multi_rf import main as train_multi_rf_main

        train_defaults = spec.train
        train_multi_rf_main(
            task_dir=spec.task_dir,
            description=train_defaults.description,
        )
        return

    if spec.family == "soft":
        from training.run.train_soft_variants import main as train_soft_variants_main

        train_defaults = spec.train
        train_soft_variants_main(
            task_dir=spec.task_dir,
            default_trainer=train_defaults.default_trainer,
            description=train_defaults.description,
        )
        return

    raise RuntimeError(f"Unsupported experiment family: {spec.family}")


def run_named_predict(experiment_name: str) -> None:
    spec = get_experiment_spec(experiment_name)
    if spec.predict is None:
        raise RuntimeError(f"Experiment '{spec.key}' does not define predict defaults.")

    from training.evaluation.predict_semantic import main as predict_main

    predict_defaults = spec.predict
    predict_main(
        task_dir=spec.task_dir,
        default_trainer=predict_defaults.default_trainer,
        method_name=predict_defaults.method_name,
        required_overlay_files=list(predict_defaults.required_overlay_files),
        apply_global_lcc_by_default=predict_defaults.apply_global_lcc_by_default,
        description=predict_defaults.description,
    )


def run_named_validate(experiment_name: str) -> None:
    spec = get_experiment_spec(experiment_name)
    if spec.validate is None:
        raise RuntimeError(f"Experiment '{spec.key}' does not define validation defaults.")

    from training.evaluation.roi_validation import main as validation_main

    validate_defaults = spec.validate
    validation_main(
        default_method_name=validate_defaults.default_method_name,
        default_trainer=validate_defaults.default_trainer,
    )


def run_named_compare(experiment_name: str) -> None:
    spec = get_experiment_spec(experiment_name)
    if spec.compare is None:
        raise RuntimeError(f"Experiment '{spec.key}' does not define compare defaults.")

    from training.evaluation.compare_full_ct_semantic_pipeline import main as compare_main

    compare_defaults = spec.compare
    compare_main(
        task_dir=spec.task_dir,
        default_semantic_trainer=compare_defaults.default_semantic_trainer,
        validation_method_name=compare_defaults.validation_method_name,
        required_overlay_files=list(compare_defaults.required_overlay_files),
        description=compare_defaults.description,
    )
