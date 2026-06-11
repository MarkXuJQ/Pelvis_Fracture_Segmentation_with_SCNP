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


def run_named_predict(experiment_name: str) -> None:
    from inference.predict_stage2_fragments import main_for_experiment

    main_for_experiment(experiment_name)
