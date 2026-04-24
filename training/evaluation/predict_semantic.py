from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from training.runtime.experiment_runtime import (
    build_ct_only_input_dir,
    build_overlay_pythonpath,
    resolve_dataset_name,
    resolve_model_root,
    resolve_predict_exe,
    safe_remove_directory,
)
from training.evaluation.postprocess import keep_only_global_largest_component
from training.runtime.project_paths import DATASET_ID, get_project_paths
from training.evaluation.roi_validation import run_validation


def main(
    *,
    task_dir: Path,
    default_trainer: str,
    method_name: str,
    required_overlay_files: list[str],
    apply_global_lcc_by_default: bool = True,
    description: str = "Run nnUNetv2 prediction and ROI validation for Dataset503.",
) -> None:
    default_paths = get_project_paths()
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--raw_base", type=Path, default=default_paths.data_root)
    parser.add_argument("--preprocessed", type=Path, default=default_paths.nnunet_preprocessed_root)
    parser.add_argument("--results", type=Path, default=default_paths.nnunet_results_root)
    parser.add_argument("--task_id", type=int, default=DATASET_ID)
    parser.add_argument("--network", type=str, default="3d_fullres")
    parser.add_argument("--trainer", type=str, default=default_trainer)
    parser.add_argument("--plans", type=str, default="nnUNetPlans")
    parser.add_argument("--fold", type=str, default="all")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_final.pth")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--disable_tta", action="store_true")
    parser.add_argument("--pred_dir", type=Path, default=None)
    parser.add_argument("--validation_dir", type=Path, default=None)
    parser.add_argument("--method_name", type=str, default=None)
    parser.add_argument("--apply_global_lcc", action=argparse.BooleanOptionalAction, default=apply_global_lcc_by_default)
    args = parser.parse_args()
    resolved_method_name = (
        str(args.method_name)
        if args.method_name
        else (str(method_name) if str(args.trainer) == str(default_trainer) else str(args.trainer))
    )

    task_dir = task_dir.resolve()
    raw_base = args.raw_base.resolve()
    preprocessed = args.preprocessed.resolve()
    results = args.results.resolve()
    v2_raw_root = get_project_paths(raw_base).nnunet_raw_root
    dataset_name = resolve_dataset_name(v2_raw_root, int(args.task_id))
    dataset_dir = v2_raw_root / dataset_name
    input_dir = dataset_dir / "imagesTs"
    gt_dir = dataset_dir / "labelsTs"
    if not input_dir.is_dir():
        raise RuntimeError(f"Missing imagesTs: {input_dir}")
    if not gt_dir.is_dir():
        raise RuntimeError(f"Missing labelsTs: {gt_dir}")

    model_root = resolve_model_root(results, dataset_name, str(args.trainer), str(args.plans), str(args.network))
    fold_name = "fold_all" if str(args.fold).lower() == "all" else f"fold_{args.fold}"
    checkpoint_tag = Path(str(args.checkpoint)).stem
    pred_dir = args.pred_dir.resolve() if args.pred_dir else model_root / fold_name / f"predictions_labelsTs_{checkpoint_tag}"
    validation_dir = args.validation_dir.resolve() if args.validation_dir else model_root / fold_name / f"custom_validation_labelsTs_{checkpoint_tag}"
    pred_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)

    overlay_root = task_dir / "nnunetv2_overlay"
    ct_only_input_dir = build_ct_only_input_dir(input_dir, pred_dir / "_ct_only_input")

    os.environ["nnUNet_raw"] = str(v2_raw_root)
    os.environ["nnUNet_preprocessed"] = str(preprocessed)
    os.environ["nnUNet_results"] = str(results)

    env = os.environ.copy()
    env["PYTHONPATH"] = build_overlay_pythonpath(
        env.get("PYTHONPATH"),
        overlay_root=overlay_root,
        required_relative_paths=[relative_path.format(trainer=args.trainer) for relative_path in required_overlay_files],
        repo_root=task_dir.parents[2],
    )

    predict_exe = resolve_predict_exe()
    predict_cmd = [
        predict_exe,
        "-i",
        str(ct_only_input_dir),
        "-o",
        str(pred_dir),
        "-d",
        str(int(args.task_id)),
        "-c",
        str(args.network),
        "-f",
        str(args.fold),
        "-tr",
        str(args.trainer),
        "-p",
        str(args.plans),
        "-chk",
        str(args.checkpoint),
        "-npp",
        "1",
        "-nps",
        "1",
        "-step_size",
        "0.5",
        "-device",
        str(args.device),
    ]
    if args.disable_tta:
        predict_cmd.append("--disable_tta")

    print(f"[predict] {' '.join(predict_cmd)}")
    subprocess.check_call(predict_cmd, env=env)

    eval_pred_dir = pred_dir
    if bool(args.apply_global_lcc):
        eval_pred_dir = validation_dir / "_global_lcc_predictions"
        safe_remove_directory(eval_pred_dir)
        keep_only_global_largest_component(pred_dir, eval_pred_dir)

    run_validation(
        pred_dir=eval_pred_dir,
        gt_dir=gt_dir,
        output_dir=validation_dir,
        method_name=resolved_method_name,
    )

    run_config = {
        "dataset_name": dataset_name,
        "input_dir": str(ct_only_input_dir),
        "original_input_dir": str(input_dir),
        "gt_dir": str(gt_dir),
        "pred_dir": str(pred_dir),
        "eval_pred_dir": str(eval_pred_dir),
        "validation_dir": str(validation_dir),
        "task_id": int(args.task_id),
        "network": str(args.network),
        "trainer": str(args.trainer),
        "method_name": resolved_method_name,
        "plans": str(args.plans),
        "fold": str(args.fold),
        "checkpoint": str(args.checkpoint),
        "device": str(args.device),
        "disable_tta": bool(args.disable_tta),
        "apply_global_lcc": bool(args.apply_global_lcc),
    }
    (validation_dir / "prediction_run_config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    print(f"[done] Predictions: {pred_dir}")
    print(f"[done] Validation: {validation_dir}")
