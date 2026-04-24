from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from training.runtime.experiment_runtime import (
    apply_windows_runtime_limits,
    build_overlay_pythonpath,
    ensure_preprocessed_dataset_matches_raw,
    prepare_patient_level_splits,
    resolve_dataset_name,
    resolve_n_proc_da,
    resolve_train_exe,
    run_logged_command,
)
from training.runtime.project_paths import DATASET_ID, get_project_paths


def main(
    *,
    task_dir: Path,
    default_trainer: str,
    trainer_choices: list[str],
    default_rf: int,
    default_fdm_threshold: float,
    description: str = "Train a CT-only single-RF SCNP model on Dataset503.",
) -> None:
    task_dir = task_dir.resolve()
    os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")
    os.environ.setdefault("OCL_ICD_VENDORS", str(task_dir))

    default_paths = get_project_paths()
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--raw_base", type=str, default=str(default_paths.data_root))
    parser.add_argument("--preprocessed", type=str, default=str(default_paths.nnunet_preprocessed_root))
    parser.add_argument("--results", type=str, default=str(default_paths.nnunet_results_root))
    parser.add_argument("--task_id", type=int, default=DATASET_ID)

    parser.add_argument("--network", type=str, default="3d_fullres")
    parser.add_argument("--fold", type=str, default="all")
    parser.add_argument("--trainer_name", type=str, default=default_trainer, choices=trainer_choices)

    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--verify_dataset_integrity", action="store_true", default=True)
    parser.add_argument("--reset_preprocess", action="store_true")
    parser.add_argument("--continue_training", action="store_true")
    parser.add_argument("--pretrained_weights", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument(
        "--split_mode",
        type=str,
        default="keep_existing",
        choices=["patient", "keep_existing"],
        help="patient: regenerate splits_final.json by patient id before training; keep_existing: keep the existing split file.",
    )
    parser.add_argument("--patient_val_count", type=int, default=30)
    parser.add_argument("--patient_split_seed", type=int, default=20260328)

    parser.add_argument("--scnp_rf", type=int, default=default_rf)
    parser.add_argument("--scnp_kappa", type=float, default=9999.0)
    parser.add_argument("--scnp_weight_ce", type=float, default=1.0)
    parser.add_argument("--scnp_weight_dice", type=float, default=1.0)
    parser.add_argument("--scnp_joint_std_weight", type=float, default=0.0)
    parser.add_argument("--scnp_fdm_threshold", type=float, default=default_fdm_threshold)
    parser.add_argument("--scnp_fdm_power", type=float, default=1.0)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--initial_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=3e-5)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--n_proc_da", type=int, default=None)
    args = parser.parse_args()

    raw_base = os.path.abspath(args.raw_base)
    preprocessed = os.path.abspath(args.preprocessed)
    results = os.path.abspath(args.results)
    task_id = int(args.task_id)

    os.makedirs(raw_base, exist_ok=True)
    os.makedirs(preprocessed, exist_ok=True)
    os.makedirs(results, exist_ok=True)

    dataset_paths = get_project_paths(raw_base)
    v2_raw_root = str(dataset_paths.nnunet_raw_root)
    _ = resolve_dataset_name(v2_raw_root, task_id)

    os.environ["nnUNet_raw"] = v2_raw_root
    os.environ["nnUNet_preprocessed"] = preprocessed
    os.environ["nnUNet_results"] = results
    apply_windows_runtime_limits()

    os.environ["NNUNET_SCNP_RF"] = str(int(args.scnp_rf))
    os.environ["NNUNET_SCNP_KAPPA"] = str(float(args.scnp_kappa))
    os.environ["NNUNET_SCNP_WEIGHT_CE"] = str(float(args.scnp_weight_ce))
    os.environ["NNUNET_SCNP_WEIGHT_DICE"] = str(float(args.scnp_weight_dice))
    os.environ["NNUNET_SCNP_JOINT_STD_WEIGHT"] = str(float(args.scnp_joint_std_weight))
    os.environ["NNUNET_SCNP_FDM_THRESHOLD"] = str(float(args.scnp_fdm_threshold))
    os.environ["NNUNET_SCNP_FDM_POWER"] = str(float(args.scnp_fdm_power))
    os.environ["NNUNET_SCNP_NUM_EPOCHS"] = str(int(args.num_epochs))
    os.environ["NNUNET_SCNP_INITIAL_LR"] = str(float(args.initial_lr))
    os.environ["NNUNET_SCNP_WEIGHT_DECAY"] = str(float(args.weight_decay))
    os.environ["NNUNET_SCNP_SAVE_EVERY"] = str(int(args.save_every))

    resolved_n_proc_da = resolve_n_proc_da(args.n_proc_da)
    if resolved_n_proc_da is not None:
        os.environ["nnUNet_n_proc_DA"] = str(int(resolved_n_proc_da))
        print(f"[runtime] nnUNet_n_proc_DA={resolved_n_proc_da}")

    if args.preprocess:
        preprocess_py = (task_dir.parents[1] / "data" / "preprocess_dataset.py").resolve()
        preprocess_cmd = [
            sys.executable,
            str(preprocess_py),
            "--raw_base",
            raw_base,
            "--preprocessed",
            preprocessed,
            "--results",
            results,
            "--task_id",
            str(task_id),
            "--configs",
            str(args.network),
        ]
        if bool(args.verify_dataset_integrity):
            preprocess_cmd.append("--verify_dataset_integrity")
        if bool(args.reset_preprocess):
            preprocess_cmd.append("--reset_preprocess")
        subprocess.check_call(preprocess_cmd)
        ensure_preprocessed_dataset_matches_raw(raw_base=raw_base, preprocessed=preprocessed, task_id=task_id)
    else:
        ensure_preprocessed_dataset_matches_raw(raw_base=raw_base, preprocessed=preprocessed, task_id=task_id)

    if args.split_mode == "patient" and str(args.fold).lower() != "all":
        prepare_patient_level_splits(
            raw_base=raw_base,
            preprocessed=preprocessed,
            task_id=task_id,
            selected_fold=int(args.fold),
            patient_val_count=int(args.patient_val_count),
            seed=int(args.patient_split_seed),
            generated_by=Path(__file__),
        )

    train_env = os.environ.copy()
    train_env["PYTHONPATH"] = build_overlay_pythonpath(
        train_env.get("PYTHONPATH"),
        overlay_root=task_dir / "nnunetv2_overlay",
        required_relative_paths=[
            f"nnunetv2/training/nnUNetTrainer/{args.trainer_name}.py",
            "nnunetv2/training/loss/compound_losses_scnp.py",
        ],
        repo_root=task_dir.parents[2],
    )

    train_cmd = [
        resolve_train_exe(),
        str(task_id),
        str(args.network),
        str(args.fold),
        "-tr",
        str(args.trainer_name),
    ]
    if args.continue_training:
        train_cmd.append("--c")
    if args.pretrained_weights:
        train_cmd += ["-pretrained_weights", os.path.abspath(args.pretrained_weights)]
    if args.num_gpus is not None:
        train_cmd += ["-num_gpus", str(int(args.num_gpus))]
    if args.device:
        train_cmd += ["-device", str(args.device)]

    run_logged_command(train_cmd, env=train_env, log_file=args.log_file)
