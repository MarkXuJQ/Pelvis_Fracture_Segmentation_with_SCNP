from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.run.experiment_registry import get_experiment_spec, list_experiment_specs
from training.run.launcher import configure_top_level_environment
from training.runtime.experiment_runtime import (
    apply_windows_runtime_limits,
    build_overlay_pythonpath,
    resolve_dataset_name,
    resolve_model_root,
    resolve_predict_exe,
    run_logged_command,
)
from training.runtime.project_paths import DATASET_ID, get_project_paths


def _experiment_choices() -> list[str]:
    choices: list[str] = []
    for spec in list_experiment_specs("train"):
        choices.append(spec.key)
        choices.extend(spec.aliases)
    return choices


def _default_output_dir(paths, experiment_name: str, network: str, checkpoint: str) -> Path:
    checkpoint_name = checkpoint
    if checkpoint_name.endswith(".pth"):
        checkpoint_name = checkpoint_name[: -len(".pth")]
    return paths.dataset_root / "predictions" / f"{experiment_name}_{network}_{checkpoint_name}"


def _required_overlay_files(trainer_name: str) -> list[str]:
    return [
        f"nnunetv2/training/nnUNetTrainer/{trainer_name}.py",
        "nnunetv2/training/loss/compound_losses_scnp.py",
    ]


def _check_model_checkpoint(
    *,
    results_root: Path,
    dataset_name: str,
    trainer_name: str,
    plans_name: str,
    network: str,
    folds: list[str],
    checkpoint_name: str,
) -> None:
    model_root = resolve_model_root(results_root, dataset_name, trainer_name, plans_name, network)
    if not model_root.is_dir():
        raise RuntimeError(
            "Could not find trained model directory. "
            f"Expected one of the standard nnU-Net layouts under: {model_root}"
        )

    missing: list[str] = []
    for fold in folds:
        fold_dir = model_root / f"fold_{fold}"
        checkpoint_path = fold_dir / checkpoint_name
        if not checkpoint_path.is_file():
            missing.append(str(checkpoint_path))

    if missing:
        raise RuntimeError(
            "Missing trained model checkpoint(s). Train the model first or choose another --folds/--checkpoint.\n"
            + "\n".join(f"- {path}" for path in missing)
        )


def main(default_experiment: str | None = None) -> None:
    configure_top_level_environment(__file__)

    default_paths = get_project_paths()
    parser = argparse.ArgumentParser(
        description=(
            "Run nnU-Net v2 prediction with a trained SCNP model on stage-2 masked-CT images. "
            "This entrypoint performs inference only; it does not read ground-truth labels or compute metrics."
        )
    )
    if default_experiment is None:
        parser.add_argument("experiment", choices=_experiment_choices(), help="Registered experiment key or alias.")

    parser.add_argument("--raw_base", type=Path, default=default_paths.data_root)
    parser.add_argument("--preprocessed", type=Path, default=default_paths.nnunet_preprocessed_root)
    parser.add_argument("--results", type=Path, default=default_paths.nnunet_results_root)
    parser.add_argument("--task_id", type=int, default=DATASET_ID)
    parser.add_argument("--network", type=str, default="3d_fullres")
    parser.add_argument("--plans", type=str, default="nnUNetPlans")
    parser.add_argument("--trainer_name", type=str, default=None)
    parser.add_argument("--folds", nargs="+", default=["all"], help="Fold(s) passed to nnUNetv2_predict, for example: all or 0 1 2.")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_final.pth")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=None,
        help="Directory containing stage-2 masked CT files named *_0000.nii.gz. Defaults to the raw dataset imagesTs.",
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument("--num_processes_preprocessing", type=int, default=None)
    parser.add_argument("--num_processes_segmentation_export", type=int, default=None)
    parser.add_argument("--disable_tta", action="store_true")
    parser.add_argument("--save_probabilities", action="store_true")
    parser.add_argument("--continue_prediction", action="store_true")
    parser.add_argument("--skip_model_check", action="store_true")
    parser.add_argument("--log_file", type=Path, default=None)
    args = parser.parse_args()

    experiment_name = default_experiment or args.experiment
    spec = get_experiment_spec(experiment_name)
    trainer_name = args.trainer_name or spec.train.default_trainer
    if trainer_name not in spec.train.trainer_choices:
        choices = ", ".join(spec.train.trainer_choices)
        raise RuntimeError(f"Trainer {trainer_name!r} is not registered for {spec.key}. Choices: {choices}")

    raw_base = args.raw_base.resolve()
    preprocessed = args.preprocessed.resolve()
    results = args.results.resolve()
    paths = get_project_paths(raw_base)
    dataset_name = resolve_dataset_name(paths.nnunet_raw_root, int(args.task_id))

    input_dir = args.input_dir.resolve() if args.input_dir is not None else paths.nnunet_raw_root / dataset_name / "imagesTs"
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else _default_output_dir(paths, spec.key, args.network, args.checkpoint).resolve()
    )

    if not input_dir.is_dir():
        raise RuntimeError(f"Missing prediction input directory: {input_dir}")
    if not any(input_dir.glob("*_0000.nii.gz")):
        raise RuntimeError(f"No stage-2 CT channel files found in {input_dir}; expected files named *_0000.nii.gz")

    os.environ["nnUNet_raw"] = str(paths.nnunet_raw_root)
    os.environ["nnUNet_preprocessed"] = str(preprocessed)
    os.environ["nnUNet_results"] = str(results)
    apply_windows_runtime_limits()

    folds = [str(fold) for fold in args.folds]
    if not args.skip_model_check:
        _check_model_checkpoint(
            results_root=results,
            dataset_name=dataset_name,
            trainer_name=trainer_name,
            plans_name=args.plans,
            network=args.network,
            folds=folds,
            checkpoint_name=args.checkpoint,
        )

    predict_env = os.environ.copy()
    predict_env["PYTHONPATH"] = build_overlay_pythonpath(
        predict_env.get("PYTHONPATH"),
        overlay_root=spec.task_dir / "nnunetv2_overlay",
        required_relative_paths=_required_overlay_files(trainer_name),
        repo_root=REPO_ROOT,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    predict_cmd = [
        resolve_predict_exe(),
        "-i",
        str(input_dir),
        "-o",
        str(output_dir),
        "-d",
        str(int(args.task_id)),
        "-c",
        str(args.network),
        "-f",
        *folds,
        "-tr",
        str(trainer_name),
        "-p",
        str(args.plans),
        "-chk",
        str(args.checkpoint),
    ]
    if args.device:
        predict_cmd += ["-device", str(args.device)]
    if args.num_gpus is not None:
        predict_cmd += ["-num_gpus", str(int(args.num_gpus))]
    if args.num_processes_preprocessing is not None:
        predict_cmd += ["-npp", str(int(args.num_processes_preprocessing))]
    if args.num_processes_segmentation_export is not None:
        predict_cmd += ["-nps", str(int(args.num_processes_segmentation_export))]
    if args.disable_tta:
        predict_cmd.append("--disable_tta")
    if args.save_probabilities:
        predict_cmd.append("--save_probabilities")
    if args.continue_prediction:
        predict_cmd.append("--continue_prediction")

    run_config = {
        "experiment": spec.key,
        "trainer_name": trainer_name,
        "task_id": int(args.task_id),
        "dataset_name": dataset_name,
        "network": args.network,
        "plans": args.plans,
        "folds": folds,
        "checkpoint": args.checkpoint,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "label_schema": {
            "0": "background",
            "1": "main fracture segment",
            "2": "secondary fragments",
        },
        "ground_truth_used": False,
    }
    (output_dir / "prediction_run_config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")

    run_logged_command(predict_cmd, env=predict_env, log_file=args.log_file)


def main_for_experiment(experiment_name: str) -> None:
    main(default_experiment=experiment_name)


if __name__ == "__main__":
    main()
