from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt, generate_binary_structure

from training.runtime.experiment_runtime import resolve_dataset_name, resolve_model_root
from training.runtime.project_paths import DATASET_ID, get_project_paths


ROI_TO_BONE = {
    "SA": "Sacrum",
    "LI": "LeftHip",
    "RI": "RightHip",
}

CASE_METRIC_COLUMNS = [
    "Main_DSC",
    "Main_HD95",
    "Main_ASSD",
    "Main_LDSC10",
    "Other_DSC",
    "Other_HD95",
    "Other_ASSD",
    "Other_LDSC10",
    "All_DSC",
    "All_HD95",
    "All_ASSD",
    "All_LDSC10",
]


def crop_to_bbox(a: np.ndarray, b: np.ndarray, margin: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    mask = a | b
    if np.sum(mask) == 0:
        return a, b
    coords = np.argwhere(mask)
    min_c = coords.min(axis=0)
    max_c = coords.max(axis=0)
    slices = []
    for i in range(3):
        start = max(0, int(min_c[i]) - int(margin))
        stop = min(mask.shape[i], int(max_c[i]) + int(margin) + 1)
        slices.append(slice(start, stop))
    slices_t = tuple(slices)
    return a[slices_t], b[slices_t]


def crop_to_bbox_multi(masks: Sequence[np.ndarray | None], margin: int = 50) -> List[np.ndarray | None]:
    if not masks:
        return []
    combined = masks[0]
    if combined is None:
        return list(masks)
    for mask in masks[1:]:
        if mask is not None:
            combined = combined | mask
    if np.sum(combined) == 0:
        return list(masks)
    coords = np.argwhere(combined)
    min_c = coords.min(axis=0)
    max_c = coords.max(axis=0)
    slices = []
    for i in range(3):
        start = max(0, int(min_c[i]) - int(margin))
        stop = min(combined.shape[i], int(max_c[i]) + int(margin) + 1)
        slices.append(slice(start, stop))
    slices_t = tuple(slices)
    return [mask[slices_t] if mask is not None else None for mask in masks]


def binary_dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.bool_)
    b = b.astype(np.bool_)
    inter = np.sum(a & b)
    denom = np.sum(a) + np.sum(b)
    return float(2.0 * inter / denom) if denom > 0 else 1.0


def hd95(a: np.ndarray, b: np.ndarray, spacing_zyx: Sequence[float]) -> float:
    a = a.astype(np.bool_)
    b = b.astype(np.bool_)
    if np.sum(a) == 0 and np.sum(b) == 0:
        return 0.0
    if np.sum(a) == 0 or np.sum(b) == 0:
        return float("inf")
    a, b = crop_to_bbox(a, b, margin=2)
    structure = generate_binary_structure(3, 1)
    sa = a ^ binary_erosion(a, structure=structure, border_value=0)
    sb = b ^ binary_erosion(b, structure=structure, border_value=0)
    dt_b = distance_transform_edt(~b, sampling=spacing_zyx)
    dt_a = distance_transform_edt(~a, sampling=spacing_zyx)
    distances = np.concatenate([dt_b[sa], dt_a[sb]])
    if distances.size == 0:
        return 0.0
    return float(np.percentile(distances, 95))


def assd(a: np.ndarray, b: np.ndarray, spacing_zyx: Sequence[float]) -> float:
    a = a.astype(np.bool_)
    b = b.astype(np.bool_)
    if np.sum(a) == 0 and np.sum(b) == 0:
        return 0.0
    if np.sum(a) == 0 or np.sum(b) == 0:
        return float("inf")
    a, b = crop_to_bbox(a, b, margin=2)
    structure = generate_binary_structure(3, 1)
    sa = a ^ binary_erosion(a, structure=structure, border_value=0)
    sb = b ^ binary_erosion(b, structure=structure, border_value=0)
    dt_b = distance_transform_edt(~b, sampling=spacing_zyx)
    dt_a = distance_transform_edt(~a, sampling=spacing_zyx)
    distances = np.concatenate([dt_b[sa], dt_a[sb]])
    if distances.size == 0:
        return 0.0
    return float(np.mean(distances))


def extract_surface_voxels(mask: np.ndarray) -> np.ndarray:
    structure = generate_binary_structure(3, 1)
    return mask ^ binary_erosion(mask, structure=structure, border_value=0)


def local_dice_10mm(
    a: np.ndarray,
    b: np.ndarray,
    spacing_zyx: Sequence[float],
    adjacent_mask: np.ndarray | None = None,
) -> float:
    a = a.astype(np.bool_)
    b = b.astype(np.bool_)
    if np.sum(a) == 0:
        return 1.0 if np.sum(b) == 0 else 0.0

    if adjacent_mask is not None:
        adjacent_mask = adjacent_mask.astype(np.bool_)
        if np.sum(adjacent_mask) > 0:
            c_a, c_b, c_adj = crop_to_bbox_multi([a, b, adjacent_mask], margin=50)
        else:
            c_a, c_b = crop_to_bbox(a, b, margin=50)
            c_adj = None
    else:
        c_a, c_b = crop_to_bbox(a, b, margin=50)
        c_adj = None

    gt_surface = extract_surface_voxels(c_a)
    voxel_size = float(np.mean(spacing_zyx))

    if c_adj is not None and np.sum(c_adj) > 0:
        dil_iter_adj = int(round(15.0 / voxel_size))
        adj_dilated = binary_dilation(c_adj, iterations=dil_iter_adj)
        close_fracture_surface = gt_surface & adj_dilated
        region_seed = close_fracture_surface if np.sum(close_fracture_surface) > 0 else gt_surface
    else:
        region_seed = gt_surface

    dilation_voxels = int(round(10.0 / voxel_size))
    local_region = binary_dilation(region_seed, iterations=dilation_voxels)
    pred_local = c_b & local_region
    gt_local = c_a & local_region
    if np.sum(gt_local) == 0:
        return 0.0
    return float(2.0 * np.sum(pred_local & gt_local) / (np.sum(pred_local) + np.sum(gt_local)))


def filter_small_fragments(binary_mask: np.ndarray, spacing_zyx: Sequence[float], min_vol_cm3: float = 1.0) -> np.ndarray:
    if np.sum(binary_mask) == 0:
        return binary_mask
    voxel_vol_mm3 = float(spacing_zyx[0]) * float(spacing_zyx[1]) * float(spacing_zyx[2])
    voxel_vol_cm3 = voxel_vol_mm3 / 1000.0
    mask_uint8 = binary_mask.astype(np.uint8)
    sitk_img = sitk.GetImageFromArray(mask_uint8)
    cc = sitk.ConnectedComponentImageFilter()
    lbl_img = cc.Execute(sitk_img)
    lbl_arr = sitk.GetArrayFromImage(lbl_img)
    num_labels = cc.GetObjectCount()
    if num_labels == 0:
        return binary_mask
    new_mask = np.zeros_like(binary_mask, dtype=np.bool_)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(lbl_img)
    for label_id in range(1, num_labels + 1):
        num_voxels = stats.GetNumberOfPixels(label_id)
        vol_cm3 = num_voxels * voxel_vol_cm3
        if vol_cm3 >= min_vol_cm3:
            new_mask |= lbl_arr == label_id
    return new_mask


def same_geometry(a: sitk.Image, b: sitk.Image, tol: float = 1e-5) -> bool:
    if a.GetSize() != b.GetSize():
        return False
    for av, bv in zip(a.GetSpacing(), b.GetSpacing()):
        if abs(float(av) - float(bv)) > tol:
            return False
    for av, bv in zip(a.GetOrigin(), b.GetOrigin()):
        if abs(float(av) - float(bv)) > tol:
            return False
    for av, bv in zip(a.GetDirection(), b.GetDirection()):
        if abs(float(av) - float(bv)) > tol:
            return False
    return True


def load_prediction_aligned(pred_path: Path, reference_img: sitk.Image) -> sitk.Image:
    pred_img = sitk.ReadImage(str(pred_path))
    if same_geometry(pred_img, reference_img):
        return pred_img
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetOutputPixelType(pred_img.GetPixelID())
    return resampler.Execute(pred_img)


def parse_case_name(case_name: str) -> Tuple[int, str]:
    stem = case_name.replace(".nii.gz", "")
    parts = stem.split("_")
    if len(parts) != 3 or parts[0] != "Frac":
        raise RuntimeError(f"Unexpected case name format: {case_name}")
    return int(parts[1]), parts[2]


def round_metric(value: float) -> float:
    return round(float(value), 4) if math.isfinite(float(value)) else float(value)


def get_roi_gt_masks(gt_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    labels = np.unique(gt_arr)
    labels = labels[labels > 0]
    if labels.size == 0:
        zero = np.zeros_like(gt_arr, dtype=np.bool_)
        return zero, zero, {"gt_main_label": None, "gt_other_labels": [], "gt_num_labels": 0}

    if np.any(gt_arr == 1):
        main = gt_arr == 1
        other = gt_arr >= 2
        other_labels = sorted(int(label) for label in labels if int(label) >= 2)
        return main, other, {
            "gt_main_label": 1,
            "gt_other_labels": other_labels,
            "gt_num_labels": int(len(labels)),
        }

    sizes = [(int(label), int(np.sum(gt_arr == label))) for label in labels]
    sizes.sort(key=lambda item: item[1], reverse=True)
    main_label = int(sizes[0][0])
    main = gt_arr == main_label
    other = np.zeros_like(gt_arr, dtype=np.bool_)
    other_labels: List[int] = []
    for label, _ in sizes[1:]:
        other |= gt_arr == int(label)
        other_labels.append(int(label))

    return main, other, {
        "gt_main_label": main_label,
        "gt_other_labels": other_labels,
        "gt_num_labels": int(len(sizes)),
    }


def get_pred_main_other(pred_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    unique_labels = np.unique(pred_arr)
    unique_labels = unique_labels[unique_labels > 0]
    if unique_labels.size == 0:
        zero = np.zeros_like(pred_arr, dtype=np.bool_)
        return zero, zero, {"pred_main_label": None, "pred_num_labels": 0, "pred_labels": []}

    if np.any(pred_arr == 1):
        pred_labels = [int(label) for label in sorted(unique_labels.tolist())]
        main = pred_arr == 1
        other = pred_arr >= 2
        return main, other, {
            "pred_main_label": 1,
            "pred_num_labels": int(len(pred_labels)),
            "pred_labels": pred_labels,
        }

    if len(unique_labels) > 1:
        label_sizes = [(int(label), int(np.sum(pred_arr == label))) for label in unique_labels]
        label_sizes.sort(key=lambda item: item[1], reverse=True)
        main_label = int(label_sizes[0][0])
        main = pred_arr == main_label
        other = np.zeros_like(main, dtype=np.bool_)
        for label, _ in label_sizes[1:]:
            other |= pred_arr == int(label)
        return main, other, {
            "pred_main_label": main_label,
            "pred_num_labels": int(len(label_sizes)),
            "pred_labels": [int(label) for label, _ in label_sizes],
        }

    bin_mask = pred_arr > 0
    cc = sitk.ConnectedComponentImageFilter()
    lab = cc.Execute(sitk.Cast(sitk.GetImageFromArray(bin_mask.astype(np.uint8)), sitk.sitkUInt8))
    lab_arr = sitk.GetArrayFromImage(lab)
    comps = np.unique(lab_arr)
    comps = comps[comps > 0]
    if comps.size == 0:
        zero = np.zeros_like(pred_arr, dtype=np.bool_)
        return zero, zero, {
            "pred_main_label": int(unique_labels[0]),
            "pred_num_labels": 1,
            "pred_labels": [int(unique_labels[0])],
        }

    comp_sizes = [(int(comp), int(np.sum(lab_arr == comp))) for comp in comps]
    comp_sizes.sort(key=lambda item: item[1], reverse=True)
    main_comp = int(comp_sizes[0][0])
    main = lab_arr == main_comp
    other = np.zeros_like(main, dtype=np.bool_)
    for comp, _ in comp_sizes[1:]:
        other |= lab_arr == int(comp)
    return main, other, {
        "pred_main_label": int(unique_labels[0]),
        "pred_num_labels": 1,
        "pred_labels": [int(unique_labels[0])],
    }


def compute_case_metrics(pred_path: Path, gt_path: Path) -> Dict[str, object]:
    case_name = pred_path.name
    case_id, roi_tag = parse_case_name(case_name)
    bone_name = ROI_TO_BONE.get(roi_tag, roi_tag)

    gt_img = sitk.ReadImage(str(gt_path))
    pred_img = load_prediction_aligned(pred_path, gt_img)
    gt_arr = sitk.GetArrayFromImage(gt_img)
    pred_arr = sitk.GetArrayFromImage(pred_img)
    spacing_xyz = gt_img.GetSpacing()
    spacing_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))

    gt_main, gt_other, gt_info = get_roi_gt_masks(gt_arr)
    pred_main, pred_other, pred_info = get_pred_main_other(pred_arr)
    pred_other = filter_small_fragments(pred_other, spacing_zyx, min_vol_cm3=1.0)

    all_gt = gt_arr > 0
    all_pred = pred_main | pred_other

    record: Dict[str, object] = {
        "CaseName": case_name.replace(".nii.gz", ""),
        "CaseID": case_id,
        "ROI": roi_tag,
        "Bone": bone_name,
        "SpacingZ": round(float(spacing_zyx[0]), 6),
        "SpacingY": round(float(spacing_zyx[1]), 6),
        "SpacingX": round(float(spacing_zyx[2]), 6),
        "GT_Main_Label": gt_info["gt_main_label"],
        "GT_Other_Labels": ",".join(str(value) for value in gt_info["gt_other_labels"]),
        "GT_Num_Labels": int(gt_info["gt_num_labels"]),
        "Pred_Main_Label": pred_info["pred_main_label"],
        "Pred_Labels": ",".join(str(value) for value in pred_info["pred_labels"]),
        "Pred_Num_Labels": int(pred_info["pred_num_labels"]),
        "GT_Main_Voxels": int(np.sum(gt_main)),
        "GT_Other_Voxels": int(np.sum(gt_other)),
        "Pred_Main_Voxels": int(np.sum(pred_main)),
        "Pred_Other_Voxels": int(np.sum(pred_other)),
        "GT_All_Voxels": int(np.sum(all_gt)),
        "Pred_All_Voxels": int(np.sum(all_pred)),
        "PredictionPath": str(pred_path),
        "GTPath": str(gt_path),
    }

    record["Main_DSC"] = round_metric(binary_dice(gt_main, pred_main))
    record["Main_HD95"] = round_metric(hd95(gt_main, pred_main, spacing_zyx))
    record["Main_ASSD"] = round_metric(assd(gt_main, pred_main, spacing_zyx))
    record["Main_LDSC10"] = round_metric(local_dice_10mm(gt_main, pred_main, spacing_zyx, adjacent_mask=gt_other))

    record["Other_DSC"] = round_metric(binary_dice(gt_other, pred_other))
    record["Other_HD95"] = round_metric(hd95(gt_other, pred_other, spacing_zyx))
    record["Other_ASSD"] = round_metric(assd(gt_other, pred_other, spacing_zyx))
    record["Other_LDSC10"] = round_metric(local_dice_10mm(gt_other, pred_other, spacing_zyx, adjacent_mask=gt_main))

    record["All_DSC"] = round_metric(binary_dice(all_gt, all_pred))
    record["All_HD95"] = round_metric(hd95(all_gt, all_pred, spacing_zyx))
    record["All_ASSD"] = round_metric(assd(all_gt, all_pred, spacing_zyx))
    record["All_LDSC10"] = round_metric(local_dice_10mm(all_gt, all_pred, spacing_zyx))
    return record


def is_finite_number(value: object) -> bool:
    if isinstance(value, (int, float, np.floating, np.integer)):
        return math.isfinite(float(value))
    return False


def build_aggregate_rows(records: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: List[Tuple[str, List[Dict[str, object]]]] = [
        ("Overall", list(records)),
        ("Ilium", [row for row in records if row["Bone"] in ("LeftHip", "RightHip")]),
        ("Sacrum", [row for row in records if row["Bone"] == "Sacrum"]),
        ("LeftHip", [row for row in records if row["Bone"] == "LeftHip"]),
        ("RightHip", [row for row in records if row["Bone"] == "RightHip"]),
    ]

    rows: List[Dict[str, object]] = []
    for group_name, group_records in grouped:
        row: Dict[str, object] = {"Group": group_name, "NumCases": len(group_records)}
        for metric in CASE_METRIC_COLUMNS:
            vals = [record[metric] for record in group_records if is_finite_number(record.get(metric))]
            if vals:
                arr = np.asarray(vals, dtype=np.float64)
                row[f"{metric}_mean"] = round(float(arr.mean()), 4)
                row[f"{metric}_std"] = round(float(arr.std(ddof=0)), 4)
                row[f"{metric}_valid_n"] = int(arr.size)
            else:
                row[f"{metric}_mean"] = ""
                row[f"{metric}_std"] = ""
                row[f"{metric}_valid_n"] = 0
        rows.append(row)
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_markdown(
    path: Path,
    records: Sequence[Dict[str, object]],
    method_name: str,
) -> None:
    groups = {
        "Overall": list(records),
        "Ilium": [record for record in records if record["Bone"] in ("LeftHip", "RightHip")],
        "Sacrum": [record for record in records if record["Bone"] == "Sacrum"],
        "LeftHip": [record for record in records if record["Bone"] == "LeftHip"],
        "RightHip": [record for record in records if record["Bone"] == "RightHip"],
    }

    existing_image_lines: List[str] = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("![") or stripped.startswith("<img"):
                existing_image_lines.append(stripped)

    def format_metric(values: Iterable[object], as_percent: bool = False) -> str:
        vals = [float(value) for value in values if is_finite_number(value)]
        if not vals:
            return "N/A"
        arr = np.asarray(vals, dtype=np.float64)
        scale = 100.0 if as_percent else 1.0
        mean_val = float(arr.mean()) * scale
        std_val = float(arr.std(ddof=0)) * scale
        return f"{mean_val:.2f} +- {std_val:.2f}"

    def group_metric(group_name: str, metric_name: str, as_percent: bool = False) -> str:
        return format_metric((record[metric_name] for record in groups[group_name]), as_percent=as_percent)

    lines: List[str] = []
    lines.append("# Custom Validation Summary")
    lines.append("")
    lines.append("Dataset503 ROI adaptation of the fracture-stage metrics.")
    lines.append("Main = largest fragment in one ROI case; Other = union of the remaining fragments.")
    lines.append("Predicted Other uses the same 1 cm^3 small-fragment filter as the pipeline.")
    lines.append("")
    lines.append(f"Method: {method_name}")
    lines.append(f"Total cases: {len(records)}")
    lines.append("")
    if existing_image_lines:
        lines.extend(existing_image_lines)
        lines.append("")

    lines.append("## Aggregate Table")
    lines.append("")
    lines.append("| Method | Metric | I-main | I-other | S-main | S-other | All |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    paper_rows = [
        (
            "DSC",
            group_metric("Ilium", "Main_DSC", as_percent=True),
            group_metric("Ilium", "Other_DSC", as_percent=True),
            group_metric("Sacrum", "Main_DSC", as_percent=True),
            group_metric("Sacrum", "Other_DSC", as_percent=True),
            group_metric("Overall", "All_DSC", as_percent=True),
        ),
        (
            "LDSC",
            group_metric("Ilium", "Main_LDSC10", as_percent=True),
            group_metric("Ilium", "Other_LDSC10", as_percent=True),
            group_metric("Sacrum", "Main_LDSC10", as_percent=True),
            group_metric("Sacrum", "Other_LDSC10", as_percent=True),
            group_metric("Overall", "All_LDSC10", as_percent=True),
        ),
        (
            "HD95",
            group_metric("Ilium", "Main_HD95"),
            group_metric("Ilium", "Other_HD95"),
            group_metric("Sacrum", "Main_HD95"),
            group_metric("Sacrum", "Other_HD95"),
            group_metric("Overall", "All_HD95"),
        ),
        (
            "ASSD",
            group_metric("Ilium", "Main_ASSD"),
            group_metric("Ilium", "Other_ASSD"),
            group_metric("Sacrum", "Main_ASSD"),
            group_metric("Sacrum", "Other_ASSD"),
            group_metric("Overall", "All_ASSD"),
        ),
    ]
    for idx, (metric, i_main, i_other, s_main, s_other, all_val) in enumerate(paper_rows):
        method_cell = method_name if idx == 0 else ""
        lines.append(f"| {method_cell} | {metric} | {i_main} | {i_other} | {s_main} | {s_other} | {all_val} |")

    worst_cases = sorted(records, key=lambda record: float(record["All_DSC"]))[:10]
    lines.append("")
    lines.append("## Worst Cases By All_DSC")
    lines.append("")
    lines.append("| Rank | CaseName | Bone | All_DSC | Main_DSC | Other_DSC | GT_Num_Labels | Pred_Num_Labels |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for idx, row in enumerate(worst_cases, start=1):
        lines.append(
            f"| {idx} | {row['CaseName']} | {row['Bone']} | {row['All_DSC']} | {row['Main_DSC']} | "
            f"{row['Other_DSC']} | {row['GT_Num_Labels']} | {row['Pred_Num_Labels']} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_validation(pred_dir: Path, gt_dir: Path, output_dir: Path, method_name: str) -> None:
    pred_dir = pred_dir.resolve()
    gt_dir = gt_dir.resolve()
    output_dir = output_dir.resolve()

    if not pred_dir.is_dir():
        raise RuntimeError(f"Prediction directory not found: {pred_dir}")
    if not gt_dir.is_dir():
        raise RuntimeError(f"GT directory not found: {gt_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    pred_files = sorted(pred_dir.glob("*.nii.gz"))
    if not pred_files:
        raise RuntimeError(f"No prediction files found in {pred_dir}")

    records: List[Dict[str, object]] = []
    skipped: List[Dict[str, object]] = []
    for pred_path in pred_files:
        gt_path = gt_dir / pred_path.name
        if not gt_path.is_file():
            skipped.append({"CaseName": pred_path.name, "Reason": f"Missing GT: {gt_path}"})
            continue
        try:
            records.append(compute_case_metrics(pred_path, gt_path))
        except Exception as exc:
            skipped.append({"CaseName": pred_path.name, "Reason": str(exc)})

    if not records:
        raise RuntimeError("No valid cases were evaluated.")

    records_sorted = sorted(records, key=lambda record: (int(record["CaseID"]), str(record["ROI"])))
    aggregate_rows = build_aggregate_rows(records_sorted)
    worst_rows = sorted(records_sorted, key=lambda record: float(record["All_DSC"]))[:20]

    write_csv(output_dir / "per_case_metrics.csv", records_sorted)
    write_csv(output_dir / "aggregate_by_group.csv", aggregate_rows)
    write_csv(output_dir / "worst_cases_by_all_dsc.csv", worst_rows)
    if skipped:
        write_csv(output_dir / "skipped_cases.csv", skipped)

    config = {
        "pred_dir": str(pred_dir),
        "gt_dir": str(gt_dir),
        "output_dir": str(output_dir),
        "method_name": method_name,
        "num_prediction_files": len(pred_files),
        "num_evaluated_cases": len(records_sorted),
        "num_skipped_cases": len(skipped),
        "notes": [
            "Metrics are aligned across Dataset503 ROI experiments.",
            "Main uses label 1 when present, otherwise the largest positive component in the ROI label map.",
            "Other uses the union of the remaining positive fragments.",
        ],
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    write_summary_markdown(output_dir / "summary_table.md", records_sorted, method_name=method_name)

    print(f"Evaluated {len(records_sorted)} cases.")
    print(f"Results saved to: {output_dir}")


def build_default_validation_paths(
    raw_base: Path,
    results: Path,
    task_id: int,
    trainer: str,
    plans: str,
    network: str,
    fold: str,
    checkpoint: str,
) -> tuple[Path, Path, Path]:
    v2_raw_root = get_project_paths(raw_base).nnunet_raw_root
    dataset_name = resolve_dataset_name(v2_raw_root, int(task_id))
    dataset_dir = v2_raw_root / dataset_name
    gt_candidates = [dataset_dir / "labelsTs", dataset_dir / "labelsTr"]
    gt_dir = next((path for path in gt_candidates if path.is_dir()), gt_candidates[0])

    model_root = resolve_model_root(results, dataset_name, trainer, plans, network)
    fold_name = "fold_all" if str(fold).lower() == "all" else f"fold_{fold}"
    checkpoint_tag = Path(str(checkpoint)).stem
    pred_dir = model_root / fold_name / f"predictions_labelsTs_{checkpoint_tag}"
    output_dir = model_root / fold_name / f"custom_validation_labelsTs_{checkpoint_tag}"
    return pred_dir, gt_dir, output_dir


def main(default_method_name: str, default_trainer: str) -> None:
    default_paths = get_project_paths()
    parser = argparse.ArgumentParser(
        description="Custom validation for Dataset503 ROI semantic predictions."
    )
    parser.add_argument("--raw_base", type=Path, default=default_paths.data_root)
    parser.add_argument("--results", type=Path, default=default_paths.nnunet_results_root)
    parser.add_argument("--task_id", type=int, default=DATASET_ID)
    parser.add_argument("--network", type=str, default="3d_fullres")
    parser.add_argument("--trainer", type=str, default=default_trainer)
    parser.add_argument("--plans", type=str, default="nnUNetPlans")
    parser.add_argument("--fold", type=str, default="all")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_final.pth")
    parser.add_argument("--pred_dir", type=Path, default=None)
    parser.add_argument("--gt_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--method_name", type=str, default=default_method_name)
    args = parser.parse_args()

    default_pred_dir, default_gt_dir, default_output_dir = build_default_validation_paths(
        raw_base=args.raw_base.resolve(),
        results=args.results.resolve(),
        task_id=int(args.task_id),
        trainer=str(args.trainer),
        plans=str(args.plans),
        network=str(args.network),
        fold=str(args.fold),
        checkpoint=str(args.checkpoint),
    )

    run_validation(
        pred_dir=args.pred_dir.resolve() if args.pred_dir else default_pred_dir,
        gt_dir=args.gt_dir.resolve() if args.gt_dir else default_gt_dir,
        output_dir=args.output_dir.resolve() if args.output_dir else default_output_dir,
        method_name=str(args.method_name),
    )


if __name__ == "__main__":
    main(default_method_name="Dataset503_ROI", default_trainer="nnUNetTrainer")
