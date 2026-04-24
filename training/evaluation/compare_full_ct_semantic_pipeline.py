from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import SimpleITK as sitk

from training.runtime.experiment_runtime import build_overlay_pythonpath, resolve_predict_exe
from training.runtime.project_paths import get_project_paths
from training.evaluation.roi_validation import run_validation
from training.evaluation.postprocess import keep_only_global_largest_component
from training.evaluation.full_ct_pipeline_helpers import (
    _copy_model,
    _ensure_dir,
    _patch_generic_unet_lambda,
    _patch_nnunet_nd_softmax_lambda,
    _patch_trainer_lambda_identity,
    _python_cuda_available,
    _run,
    _write_plans_pkl_from_model_pkl,
)


ROI_TO_LABEL = {"SA": 1, "LI": 2, "RI": 3}


def _extract_case_id_from_name(name: str) -> int:
    match = re.search(r"(\d{4})", name)
    if match is None:
        raise RuntimeError(f"Could not parse case id from: {name}")
    return int(match.group(1))


def _copy_selected_case_range(src_dir: Path, dst_dir: Path, case_min: int, case_max: int) -> list[Path]:
    _ensure_dir(dst_dir)
    selected: list[Path] = []
    for file_path in sorted(src_dir.glob("*_0000.nii.gz")):
        case_id = _extract_case_id_from_name(file_path.name)
        if case_min <= case_id <= case_max:
            output_path = dst_dir / file_path.name
            shutil.copy2(file_path, output_path)
            selected.append(output_path)
    if not selected:
        raise RuntimeError(f"No *_0000.nii.gz files selected from {src_dir} in range [{case_min}, {case_max}]")
    return selected


def _generate_fracsegnet_style_roi_inputs(full_ct_input: Path, anat_pred_dir: Path, roi_input_dir: Path) -> list[str]:
    _ensure_dir(roi_input_dir)
    case_stems: list[str] = []
    for ct_file in sorted(full_ct_input.glob("*_0000.nii.gz")):
        case_id = _extract_case_id_from_name(ct_file.name)
        anat_name = ct_file.name.replace("_0000.nii.gz", ".nii.gz")
        anat_file = anat_pred_dir / anat_name
        if not anat_file.is_file():
            raise RuntimeError(f"Missing anatomy prediction for {ct_file.name}: {anat_file}")

        ct_img = sitk.ReadImage(str(ct_file))
        anat_img = sitk.ReadImage(str(anat_file))
        ct_arr = sitk.GetArrayFromImage(ct_img)
        anat_arr = sitk.GetArrayFromImage(anat_img)
        if ct_arr.shape != anat_arr.shape:
            raise RuntimeError(
                f"Shape mismatch between CT and anatomy prediction for {ct_file.name}: {ct_arr.shape} vs {anat_arr.shape}"
            )

        for roi_tag, target_label in ROI_TO_LABEL.items():
            out_arr = ct_arr.copy()
            out_arr[anat_arr != int(target_label)] = 0
            out_img = sitk.GetImageFromArray(out_arr.astype(np.float32, copy=False))
            out_img.SetDirection(ct_img.GetDirection())
            out_img.SetSpacing(ct_img.GetSpacing())
            out_img.SetOrigin(ct_img.GetOrigin())
            out_stem = f"Frac_{case_id:04d}_{roi_tag}"
            sitk.WriteImage(out_img, str(roi_input_dir / f"{out_stem}_0000.nii.gz"))
            case_stems.append(out_stem)
    return sorted(case_stems)


def _run_anatomy_inference(
    args: argparse.Namespace,
    dataset_root: Path,
    full_ct_input: Path,
    anatomy_lowres_out: Path,
    anatomy_cascade_out: Path,
    logs_dir: Path,
) -> None:
    results_folder = dataset_root / "nnUNet_results"
    low_model_dir = (
        results_folder
        / "nnUNet"
        / "3d_lowres"
        / args.ana_task_name
        / f"nnUNetTrainerV2__{args.ana_plans}"
        / "all"
    )
    low_model_root = low_model_dir.parent
    cas_model_dir = (
        results_folder
        / "nnUNet"
        / "3d_cascade_fullres"
        / args.ana_task_name
        / f"nnUNetTrainerV2CascadeFullRes__{args.ana_plans}"
        / "all"
    )
    cas_model_root = cas_model_dir.parent

    _copy_model(Path(args.ana_lowres_model).resolve(), Path(args.ana_lowres_pkl).resolve(), low_model_dir)
    _copy_model(Path(args.ana_cascade_model).resolve(), Path(args.ana_cascade_pkl).resolve(), cas_model_dir)
    _write_plans_pkl_from_model_pkl(Path(args.ana_lowres_pkl).resolve(), low_model_root)
    _write_plans_pkl_from_model_pkl(Path(args.ana_cascade_pkl).resolve(), cas_model_root)

    env_v1 = os.environ.copy()
    env_v1["nnUNet_raw_data_base"] = str(dataset_root / "nnUNet_raw_data")
    env_v1["nnUNet_preprocessed"] = str(dataset_root / "nnUNet_preprocessed")
    env_v1["RESULTS_FOLDER"] = str(results_folder)
    env_v1["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    _patch_nnunet_nd_softmax_lambda(args.nnunet_py)
    _patch_trainer_lambda_identity(args.nnunet_py)
    _patch_generic_unet_lambda(args.nnunet_py)

    cmd_low = [
        args.nnunet_py,
        "-m",
        "nnunet.inference.predict_simple",
        "-i",
        str(full_ct_input),
        "-o",
        str(anatomy_lowres_out),
        "-t",
        str(int(args.ana_task_id)),
        "-m",
        "3d_lowres",
        "-f",
        "all",
        "-tr",
        "nnUNetTrainerV2",
        "-p",
        args.ana_plans,
        "-chk",
        "model_final_checkpoint",
        "--num_threads_preprocessing",
        "1",
        "--num_threads_nifti_save",
        "1",
    ]
    if args.ana_disable_tta:
        cmd_low.append("--disable_tta")
    _run(cmd_low, env=env_v1, quiet=bool(args.quiet_subprocess), log_file=logs_dir / "anatomy_lowres.log")

    lowres_segs = dataset_root / "nnUNet_preprocessed" / args.ana_task_name / "pred_next_stage"
    _ensure_dir(lowres_segs)
    for file_path in sorted(anatomy_lowres_out.glob("*.nii.gz")):
        shutil.copy2(file_path, lowres_segs / file_path.name)

    cmd_cas = [
        args.nnunet_py,
        "-m",
        "nnunet.inference.predict_simple",
        "-i",
        str(full_ct_input),
        "-o",
        str(anatomy_cascade_out),
        "-t",
        str(int(args.ana_task_id)),
        "-m",
        "3d_cascade_fullres",
        "-f",
        "all",
        "-tr",
        "nnUNetTrainerV2",
        "-ctr",
        "nnUNetTrainerV2CascadeFullRes",
        "-p",
        args.ana_plans,
        "-chk",
        "model_final_checkpoint",
        "-l",
        str(anatomy_lowres_out),
        "--num_threads_preprocessing",
        "1",
        "--num_threads_nifti_save",
        "1",
    ]
    if args.ana_disable_tta:
        cmd_cas.append("--disable_tta")
    _run(cmd_cas, env=env_v1, quiet=bool(args.quiet_subprocess), log_file=logs_dir / "anatomy_cascade.log")


def _run_semantic_predict(
    nnunetv2_py: str,
    dataset_root: Path,
    task_dir: Path,
    input_dir: Path,
    output_dir: Path,
    trainer: str,
    config: str,
    plans: str,
    fold: str,
    checkpoint: str,
    quiet: bool,
    log_file: Path,
    disable_tta: bool,
    required_overlay_files: Sequence[str],
) -> None:
    env_v2 = os.environ.copy()
    env_v2["nnUNet_raw"] = str(dataset_root / "nnUNet_raw_data")
    env_v2["nnUNet_preprocessed"] = str(dataset_root / "nnUNet_preprocessed")
    env_v2["nnUNet_results"] = str(dataset_root / "nnUNet_results")
    env_v2["PYTHONPATH"] = build_overlay_pythonpath(
        env_v2.get("PYTHONPATH"),
        overlay_root=task_dir / "nnunetv2_overlay",
        required_relative_paths=[relative_path.format(trainer=trainer) for relative_path in required_overlay_files],
        repo_root=task_dir.parents[2],
    )

    cmd = [
        resolve_predict_exe(nnunetv2_py),
        "-i",
        str(input_dir),
        "-o",
        str(output_dir),
        "-d",
        "503",
        "-c",
        config,
        "-f",
        str(fold),
        "-tr",
        trainer,
        "-p",
        plans,
        "-chk",
        checkpoint,
        "-npp",
        "1",
        "-nps",
        "1",
        "-step_size",
        "0.5",
        "-device",
        "cuda",
    ]
    if disable_tta:
        cmd.append("--disable_tta")
    _run(cmd, env=env_v2, quiet=quiet, log_file=log_file)


def _resample_like(img: sitk.Image, ref: sitk.Image, is_label: bool) -> sitk.Image:
    if (
        img.GetSize() == ref.GetSize()
        and tuple(float(x) for x in img.GetSpacing()) == tuple(float(x) for x in ref.GetSpacing())
        and tuple(float(x) for x in img.GetOrigin()) == tuple(float(x) for x in ref.GetOrigin())
        and tuple(float(x) for x in img.GetDirection()) == tuple(float(x) for x in ref.GetDirection())
    ):
        return img

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetOutputPixelType(img.GetPixelID())
    return resampler.Execute(img)


def _apply_anatomy_constraint(pred_dir: Path, anatomy_cascade_out: Path, output_dir: Path) -> None:
    _ensure_dir(output_dir)
    for pred_path in sorted(pred_dir.glob("Frac_*.nii.gz")):
        case_id = _extract_case_id_from_name(pred_path.name)
        roi_tag = pred_path.stem.split("_")[-1]
        target_label = ROI_TO_LABEL[roi_tag]
        anatomy_path = anatomy_cascade_out / f"FracAnatomy_{case_id:04d}.nii.gz"
        if not anatomy_path.is_file():
            raise RuntimeError(f"Missing anatomy prediction for case {case_id}: {anatomy_path}")

        pred_img = sitk.ReadImage(str(pred_path))
        anatomy_img = sitk.ReadImage(str(anatomy_path))
        anatomy_img = _resample_like(anatomy_img, pred_img, is_label=True)

        pred_arr = sitk.GetArrayFromImage(pred_img)
        anatomy_arr = sitk.GetArrayFromImage(anatomy_img)
        pred_arr[anatomy_arr != int(target_label)] = 0

        out_img = sitk.GetImageFromArray(pred_arr.astype(np.uint8, copy=False))
        out_img.CopyInformation(pred_img)
        sitk.WriteImage(out_img, str(output_dir / pred_path.name))


def _run_custom_validation(pred_dir: Path, gt_dir: Path, output_dir: Path, method_name: str) -> None:
    run_validation(pred_dir=pred_dir, gt_dir=gt_dir, output_dir=output_dir, method_name=method_name)


def _read_aggregate_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["Group"]: row for row in rows}


def _read_per_case_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["CaseName"]: row for row in rows}


def _safe_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _write_comparison_report(baseline_eval_dir: Path, full_eval_dir: Path, output_path: Path) -> None:
    baseline_agg = _read_aggregate_csv(baseline_eval_dir / "aggregate_by_group.csv")
    full_agg = _read_aggregate_csv(full_eval_dir / "aggregate_by_group.csv")

    baseline_case = _read_per_case_csv(baseline_eval_dir / "per_case_metrics.csv")
    full_case = _read_per_case_csv(full_eval_dir / "per_case_metrics.csv")
    common_case_names = sorted(set(baseline_case) & set(full_case))

    case_deltas: list[tuple[str, float]] = []
    improved = 0
    worsened = 0
    unchanged = 0
    for name in common_case_names:
        baseline_value = _safe_float(baseline_case[name]["All_DSC"])
        full_value = _safe_float(full_case[name]["All_DSC"])
        delta = full_value - baseline_value
        case_deltas.append((name, delta))
        if delta > 1e-6:
            improved += 1
        elif delta < -1e-6:
            worsened += 1
        else:
            unchanged += 1

    deltas = np.asarray([delta for _, delta in case_deltas], dtype=np.float64) if case_deltas else np.asarray([], dtype=np.float64)
    best_cases = sorted(case_deltas, key=lambda item: item[1], reverse=True)[:10]
    worst_cases = sorted(case_deltas, key=lambda item: item[1])[:10]

    rows = []
    for group_name in ["Overall", "Ilium", "Sacrum", "LeftHip", "RightHip"]:
        if group_name not in baseline_agg or group_name not in full_agg:
            continue
        baseline_row = baseline_agg[group_name]
        full_row = full_agg[group_name]
        for metric_name in ["All_DSC_mean", "All_LDSC10_mean", "All_HD95_mean", "All_ASSD_mean"]:
            baseline_value = _safe_float(baseline_row.get(metric_name, ""))
            full_value = _safe_float(full_row.get(metric_name, ""))
            rows.append((group_name, metric_name, baseline_value, full_value, full_value - baseline_value))

    lines = [
        "# Baseline vs Full-CT Semantic Comparison",
        "",
        f"Baseline eval dir: `{baseline_eval_dir}`",
        f"Full-pipeline eval dir: `{full_eval_dir}`",
        "",
        "## Case-Level All_DSC Delta",
        "",
        f"- Common ROI cases: {len(common_case_names)}",
    ]
    if deltas.size > 0:
        lines.append(f"- Mean delta (Full - Baseline): {float(deltas.mean()):.4f}")
        lines.append(f"- Median delta (Full - Baseline): {float(np.median(deltas)):.4f}")
    lines.extend(
        [
            f"- Improved cases: {improved}",
            f"- Worsened cases: {worsened}",
            f"- Unchanged cases: {unchanged}",
            "",
            "## Aggregate Comparison",
            "",
            "| Group | Metric | Baseline | FullPipeline | Delta |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for group_name, metric_name, baseline_value, full_value, delta in rows:
        lines.append(f"| {group_name} | {metric_name} | {baseline_value:.4f} | {full_value:.4f} | {delta:.4f} |")

    lines.extend(
        [
            "",
            "## Top Improved Cases",
            "",
            "| Rank | CaseName | Delta_All_DSC |",
            "| --- | --- | ---: |",
        ]
    )
    for index, (name, delta) in enumerate(best_cases, start=1):
        lines.append(f"| {index} | {name} | {delta:.4f} |")

    lines.extend(
        [
            "",
            "## Top Worsened Cases",
            "",
            "| Rank | CaseName | Delta_All_DSC |",
            "| --- | --- | ---: |",
        ]
    )
    for index, (name, delta) in enumerate(worst_cases, start=1):
        lines.append(f"| {index} | {name} | {delta:.4f} |")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_case_delta_csv(baseline_eval_dir: Path, full_eval_dir: Path, output_path: Path) -> None:
    baseline_case = _read_per_case_csv(baseline_eval_dir / "per_case_metrics.csv")
    full_case = _read_per_case_csv(full_eval_dir / "per_case_metrics.csv")
    common_case_names = sorted(set(baseline_case) & set(full_case))

    fieldnames = [
        "CaseName",
        "Bone",
        "Baseline_All_DSC",
        "FullPipeline_All_DSC",
        "Delta_All_DSC",
        "Baseline_All_LDSC10",
        "FullPipeline_All_LDSC10",
        "Delta_All_LDSC10",
        "Baseline_All_HD95",
        "FullPipeline_All_HD95",
        "Delta_All_HD95",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name in common_case_names:
            baseline_row = baseline_case[name]
            full_row = full_case[name]
            writer.writerow(
                {
                    "CaseName": name,
                    "Bone": baseline_row.get("Bone", ""),
                    "Baseline_All_DSC": baseline_row.get("All_DSC", ""),
                    "FullPipeline_All_DSC": full_row.get("All_DSC", ""),
                    "Delta_All_DSC": f"{_safe_float(full_row.get('All_DSC', '')) - _safe_float(baseline_row.get('All_DSC', '')):.4f}",
                    "Baseline_All_LDSC10": baseline_row.get("All_LDSC10", ""),
                    "FullPipeline_All_LDSC10": full_row.get("All_LDSC10", ""),
                    "Delta_All_LDSC10": f"{_safe_float(full_row.get('All_LDSC10', '')) - _safe_float(baseline_row.get('All_LDSC10', '')):.4f}",
                    "Baseline_All_HD95": baseline_row.get("All_HD95", ""),
                    "FullPipeline_All_HD95": full_row.get("All_HD95", ""),
                    "Delta_All_HD95": f"{_safe_float(full_row.get('All_HD95', '')) - _safe_float(baseline_row.get('All_HD95', '')):.4f}",
                }
            )


def main(
    *,
    task_dir: Path,
    default_semantic_trainer: str,
    validation_method_name: str,
    required_overlay_files: list[str],
    description: str,
) -> None:
    default_paths = get_project_paths()

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset_root", type=Path, default=default_paths.dataset_root)
    parser.add_argument(
        "--full_ct_input",
        type=Path,
        default=default_paths.nnunet_raw_root / "Dataset501_FracAnatomy" / "imagesTr",
    )
    parser.add_argument(
        "--roi_input",
        type=Path,
        default=default_paths.nnunet_raw_root / "Dataset503_SCNP" / "imagesTr",
    )
    parser.add_argument(
        "--gt_dir",
        type=Path,
        default=default_paths.nnunet_raw_root / "Dataset503_SCNP" / "labelsTr",
    )
    parser.add_argument("--case_min", type=int, default=101)
    parser.add_argument("--case_max", type=int, default=150)
    parser.add_argument("--output_root", type=Path, default=None)

    parser.add_argument("--nnunet_py", type=str, default=os.environ.get("NNUNET_V1_PY", "python"))
    parser.add_argument("--nnunetv2_py", type=str, default=os.environ.get("NNUNET_V2_PY", "python"))

    parser.add_argument(
        "--ana_lowres_model",
        type=Path,
        default=Path("FracSegNet") / "code" / "inference" / "AnatomicalSegModel" / "lowres_model" / "Ana_lowres.model",
    )
    parser.add_argument(
        "--ana_lowres_pkl",
        type=Path,
        default=Path("FracSegNet") / "code" / "inference" / "AnatomicalSegModel" / "lowres_model" / "Ana_lowres.model.pkl",
    )
    parser.add_argument(
        "--ana_cascade_model",
        type=Path,
        default=Path("FracSegNet")
        / "code"
        / "inference"
        / "AnatomicalSegModel"
        / "cascadeFullres_model"
        / "Ana_cascade_fullres.model",
    )
    parser.add_argument(
        "--ana_cascade_pkl",
        type=Path,
        default=Path("FracSegNet")
        / "code"
        / "inference"
        / "AnatomicalSegModel"
        / "cascadeFullres_model"
        / "Ana_cascadeFullres.model.pkl",
    )
    parser.add_argument("--ana_task_id", type=int, default=600)
    parser.add_argument("--ana_task_name", type=str, default="Task600_ContinueTrainCtPelvicAnatomical120")
    parser.add_argument("--ana_plans", type=str, default="nnUNetPlans_pretrained_ContinueTrainPelvicSeg")
    parser.add_argument("--ana_disable_tta", action="store_true")

    parser.add_argument("--semantic_trainer", type=str, default=default_semantic_trainer)
    parser.add_argument("--semantic_config", type=str, default="3d_fullres")
    parser.add_argument("--semantic_plans", type=str, default="nnUNetPlans")
    parser.add_argument("--semantic_fold", type=str, default="0")
    parser.add_argument("--semantic_checkpoint", type=str, default="checkpoint_best.pth")
    parser.add_argument("--semantic_disable_tta", action="store_true")
    parser.add_argument("--apply_global_lcc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quiet_subprocess", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    task_dir = task_dir.resolve()
    dataset_root = args.dataset_root.resolve()
    full_ct_input = args.full_ct_input.resolve()
    roi_input = args.roi_input.resolve()
    gt_dir = args.gt_dir.resolve()
    if not full_ct_input.is_dir():
        raise RuntimeError(f"full_ct_input not found: {full_ct_input}")
    if not roi_input.is_dir():
        raise RuntimeError(f"roi_input not found: {roi_input}")
    if not gt_dir.is_dir():
        raise RuntimeError(f"gt_dir not found: {gt_dir}")

    if not _python_cuda_available(args.nnunet_py):
        raise RuntimeError(f"nnUNet v1 env has no CUDA: {args.nnunet_py}")
    if not _python_cuda_available(args.nnunetv2_py):
        raise RuntimeError(f"nnUNet v2 env has no CUDA: {args.nnunetv2_py}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_root:
        out_root = args.output_root.resolve()
    else:
        out_root = dataset_root / "inference_output" / f"semantic_compare_{args.case_min}_{args.case_max}_{timestamp}"
    _ensure_dir(out_root)
    logs_dir = _ensure_dir(out_root / "logs")

    selected_full_ct_dir = _ensure_dir(out_root / "selected_full_ct")
    selected_roi_dir = _ensure_dir(out_root / "baseline_roi_inputs")
    anatomy_lowres_out = _ensure_dir(out_root / "anatomy_lowres")
    anatomy_cascade_out = _ensure_dir(out_root / "anatomy_cascade")
    full_roi_inputs = _ensure_dir(out_root / "full_pipeline_roi_inputs")
    baseline_pred_dir = _ensure_dir(out_root / "baseline_predictions")
    baseline_pred_post_dir = _ensure_dir(out_root / "baseline_predictions_global_lcc")
    full_pred_raw_dir = _ensure_dir(out_root / "full_pipeline_predictions_raw")
    full_pred_constrained_dir = _ensure_dir(out_root / "full_pipeline_predictions_anatomy_constrained")
    full_pred_post_dir = _ensure_dir(out_root / "full_pipeline_predictions_global_lcc")
    baseline_eval_dir = _ensure_dir(out_root / "baseline_eval")
    full_eval_dir = _ensure_dir(out_root / "full_pipeline_eval")

    _copy_selected_case_range(full_ct_input, selected_full_ct_dir, args.case_min, args.case_max)
    _copy_selected_case_range(roi_input, selected_roi_dir, args.case_min, args.case_max)
    _run_semantic_predict(
        args.nnunetv2_py,
        dataset_root,
        task_dir,
        selected_roi_dir,
        baseline_pred_dir,
        args.semantic_trainer,
        args.semantic_config,
        args.semantic_plans,
        args.semantic_fold,
        args.semantic_checkpoint,
        bool(args.quiet_subprocess),
        logs_dir / "baseline_semantic.log",
        bool(args.semantic_disable_tta),
        required_overlay_files=required_overlay_files,
    )
    baseline_eval_input = baseline_pred_dir
    if bool(args.apply_global_lcc):
        keep_only_global_largest_component(baseline_pred_dir, baseline_pred_post_dir)
        baseline_eval_input = baseline_pred_post_dir
    _run_custom_validation(baseline_eval_input, gt_dir, baseline_eval_dir, validation_method_name)

    _run_anatomy_inference(args, dataset_root, selected_full_ct_dir, anatomy_lowres_out, anatomy_cascade_out, logs_dir)
    _generate_fracsegnet_style_roi_inputs(selected_full_ct_dir, anatomy_cascade_out, full_roi_inputs)
    _run_semantic_predict(
        args.nnunetv2_py,
        dataset_root,
        task_dir,
        full_roi_inputs,
        full_pred_raw_dir,
        args.semantic_trainer,
        args.semantic_config,
        args.semantic_plans,
        args.semantic_fold,
        args.semantic_checkpoint,
        bool(args.quiet_subprocess),
        logs_dir / "full_pipeline_semantic.log",
        bool(args.semantic_disable_tta),
        required_overlay_files=required_overlay_files,
    )
    _apply_anatomy_constraint(full_pred_raw_dir, anatomy_cascade_out, full_pred_constrained_dir)
    full_eval_input = full_pred_constrained_dir
    if bool(args.apply_global_lcc):
        keep_only_global_largest_component(full_pred_constrained_dir, full_pred_post_dir)
        full_eval_input = full_pred_post_dir
    _run_custom_validation(full_eval_input, gt_dir, full_eval_dir, validation_method_name)

    _write_comparison_report(baseline_eval_dir, full_eval_dir, out_root / "comparison_report.md")
    _write_case_delta_csv(baseline_eval_dir, full_eval_dir, out_root / "case_level_delta.csv")

    run_config = {
        "case_min": int(args.case_min),
        "case_max": int(args.case_max),
        "selected_full_ct_dir": str(selected_full_ct_dir),
        "selected_roi_dir": str(selected_roi_dir),
        "anatomy_lowres_out": str(anatomy_lowres_out),
        "anatomy_cascade_out": str(anatomy_cascade_out),
        "full_roi_inputs": str(full_roi_inputs),
        "baseline_pred_dir": str(baseline_pred_dir),
        "baseline_eval_input": str(baseline_eval_input),
        "full_pred_raw_dir": str(full_pred_raw_dir),
        "full_pred_constrained_dir": str(full_pred_constrained_dir),
        "full_eval_input": str(full_eval_input),
        "baseline_eval_dir": str(baseline_eval_dir),
        "full_eval_dir": str(full_eval_dir),
        "comparison_report": str(out_root / "comparison_report.md"),
        "case_level_delta_csv": str(out_root / "case_level_delta.csv"),
        "semantic_trainer": str(args.semantic_trainer),
        "validation_method_name": str(validation_method_name),
        "apply_global_lcc": bool(args.apply_global_lcc),
    }
    (out_root / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    print(f"Done. Results under: {out_root}")
