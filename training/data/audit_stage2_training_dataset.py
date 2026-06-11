from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import SimpleITK as sitk

from training.runtime.project_paths import (
    DATASET_BUILD_SCHEMA_VERSION,
    SECOND_STAGE_IMAGE_GENERATION_MODE,
    SECOND_STAGE_IMAGE_SEMANTICS,
)


def is_real_nifti(path: Path) -> bool:
    return path.name.endswith(".nii.gz") and not path.name.startswith("._")


def case_stem_from_label_path(path: Path) -> str:
    if not is_real_nifti(path):
        raise RuntimeError(f"Not a supported label path: {path}")
    return path.name[: -len(".nii.gz")]


def case_stem_from_image_name(name: str) -> str:
    if name.endswith(".nii.gz"):
        stem = name[: -len(".nii.gz")]
        if stem.endswith("_0000"):
            return stem[:-5]
        return stem
    if name.endswith(".npz"):
        return name[: -len(".npz")]
    if name.endswith(".pkl"):
        return name[: -len(".pkl")]
    raise RuntimeError(f"Unsupported image file name: {name}")


def parse_case_name(case_stem: str) -> tuple[int, str]:
    parts = case_stem.split("_")
    if len(parts) != 3 or parts[0] != "Frac":
        raise RuntimeError(f"Unexpected stage-2 per-bone case name format: {case_stem}")
    return int(parts[1]), parts[2]


def collect_label_cases(label_dir: Path) -> list[str]:
    if not label_dir.is_dir():
        return []
    return sorted(case_stem_from_label_path(path) for path in label_dir.glob("*.nii.gz") if is_real_nifti(path))


def collect_image_cases(image_dir: Path) -> list[str]:
    if not image_dir.is_dir():
        return []
    stems = set()
    for path in image_dir.iterdir():
        if path.name.startswith("._"):
            continue
        if path.suffix not in (".npz", ".pkl", ".gz"):
            continue
        if path.name.endswith(".nii.gz") or path.suffix in (".npz", ".pkl"):
            stems.add(case_stem_from_image_name(path.name))
    return sorted(stems)


def iter_cases_with_bad_channels(image_dir: Path, expected_channels: int) -> Iterable[tuple[str, int]]:
    for case_stem in collect_image_cases(image_dir):
        npz_path = image_dir / f"{case_stem}.npz"
        if npz_path.is_file():
            with np.load(npz_path, mmap_mode="r") as npz:
                if "data" not in npz:
                    yield case_stem, -1
                    continue
                num_channels = int(npz["data"].shape[0])
        else:
            nifti_channels = sorted(image_dir.glob(f"{case_stem}_[0-9][0-9][0-9][0-9].nii.gz"))
            num_channels = len([path for path in nifti_channels if is_real_nifti(path)])
        if num_channels != expected_channels:
            yield case_stem, num_channels


def check_label_values(label_paths: list[Path]) -> tuple[list[str], dict[str, list[int]]]:
    bad_cases: dict[str, list[int]] = {}
    union_values: set[int] = set()
    for label_path in label_paths:
        arr = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path)))
        values = sorted(int(value) for value in np.unique(arr))
        union_values.update(values)
        if any(value not in range(0, 4) for value in values):
            bad_cases[label_path.name] = values
    return [str(value) for value in sorted(union_values)], bad_cases


def _case_records_by_name(raw_build_manifest: dict | None) -> dict[str, dict[str, object]]:
    if not raw_build_manifest:
        return {}
    records = raw_build_manifest.get("case_records", [])
    if not isinstance(records, list):
        return {}
    out: dict[str, dict[str, object]] = {}
    for record in records:
        if isinstance(record, dict) and isinstance(record.get("case"), str):
            out[str(record["case"])] = record
    return out


def check_masked_ct_consistency(
    image_dir: Path,
    label_dir: Path,
    raw_build_manifest: dict | None,
) -> tuple[dict[str, object], dict[str, object]]:
    summary: dict[str, object] = {
        "cases": 0,
        "total_voxels": 0,
        "mask_voxels": 0,
        "outside_mask_voxels": 0,
        "outside_mask_nonzero_voxels": 0,
        "zero_voxels": 0,
    }
    bad_cases: dict[str, object] = {}
    case_records = _case_records_by_name(raw_build_manifest)

    for label_path in sorted(label_dir.glob("*.nii.gz")):
        if not is_real_nifti(label_path):
            continue
        case_stem = case_stem_from_label_path(label_path)
        image_path = image_dir / f"{case_stem}_0000.nii.gz"
        if not image_path.is_file():
            bad_cases[case_stem] = "missing image"
            continue
        record = case_records.get(case_stem)
        if record is None:
            bad_cases[case_stem] = "missing case record in raw build manifest"
            continue
        anatomy_path = Path(str(record.get("stage1_anatomy_label", "")))
        if not anatomy_path.is_file():
            bad_cases[case_stem] = f"missing stage-1 anatomy label: {anatomy_path}"
            continue
        target_anatomy_label = int(record.get("target_anatomy_label", -1))
        if target_anatomy_label < 0:
            bad_cases[case_stem] = f"invalid target anatomy label: {record.get('target_anatomy_label')!r}"
            continue

        image = sitk.ReadImage(str(image_path))
        anatomy = sitk.ReadImage(str(anatomy_path))
        image_arr = sitk.GetArrayFromImage(image)
        anatomy_arr = sitk.GetArrayFromImage(anatomy)
        if image_arr.shape != anatomy_arr.shape:
            bad_cases[case_stem] = f"shape mismatch image={image_arr.shape} anatomy={anatomy_arr.shape}"
            continue

        outside_mask = anatomy_arr != target_anatomy_label
        outside_nonzero = int(np.count_nonzero(image_arr[outside_mask] != 0))
        if outside_nonzero:
            bad_cases[case_stem] = {
                "outside_stage1_anatomy_mask_nonzero_voxels": outside_nonzero,
                "stage1_anatomy_label": str(anatomy_path),
                "target_anatomy_label": target_anatomy_label,
            }

        total_voxels = int(image_arr.size)
        mask_voxels = int(np.count_nonzero(anatomy_arr == target_anatomy_label))
        outside_voxels = int(np.count_nonzero(outside_mask))
        summary["cases"] = int(summary["cases"]) + 1
        summary["total_voxels"] = int(summary["total_voxels"]) + total_voxels
        summary["mask_voxels"] = int(summary["mask_voxels"]) + mask_voxels
        summary["outside_mask_voxels"] = int(summary["outside_mask_voxels"]) + outside_voxels
        summary["outside_mask_nonzero_voxels"] = int(summary["outside_mask_nonzero_voxels"]) + outside_nonzero
        summary["zero_voxels"] = int(summary["zero_voxels"]) + int(np.count_nonzero(image_arr == 0))

    total_voxels = int(summary["total_voxels"])
    outside_voxels = int(summary["outside_mask_voxels"])
    summary["mask_fraction"] = float(int(summary["mask_voxels"]) / total_voxels) if total_voxels else 0.0
    summary["zero_fraction"] = float(int(summary["zero_voxels"]) / total_voxels) if total_voxels else 0.0
    summary["outside_mask_nonzero_fraction"] = (
        float(int(summary["outside_mask_nonzero_voxels"]) / outside_voxels) if outside_voxels else 0.0
    )
    return summary, bad_cases


def raw_build_manifest_for_preprocess(manifest: dict) -> dict:
    return {
        "dataset_build_schema_version": manifest.get("dataset_build_schema_version"),
        "image_semantics": manifest.get("image_semantics"),
        "image_generation_mode": manifest.get("image_generation_mode"),
        "mask_definition": manifest.get("mask_definition"),
        "outside_mask_value": manifest.get("outside_mask_value"),
        "full_ct_dataset_dir": manifest.get("full_ct_dataset_dir"),
        "full_ct_images": manifest.get("full_ct_images"),
        "stage1_anatomy_label_dir": manifest.get("stage1_anatomy_label_dir"),
        "fracture_label_source_root": manifest.get("fracture_label_source_root"),
        "roi_to_anatomy_label": manifest.get("roi_to_anatomy_label"),
        "label_mapping": manifest.get("label_mapping"),
        "remap_policy": manifest.get("remap_policy"),
        "train_cases": manifest.get("train_cases"),
        "test_cases": manifest.get("test_cases"),
        "case_records_count": manifest.get("case_records_count"),
        "case_records_digest": manifest.get("case_records_digest"),
        "raw_total_voxels": manifest.get("raw_total_voxels"),
        "raw_mask_voxels": manifest.get("raw_mask_voxels"),
        "raw_label_voxels": manifest.get("raw_label_voxels"),
        "raw_mask_fraction": manifest.get("raw_mask_fraction"),
    }


def raw_build_manifest_digest(raw_build_manifest: dict) -> str:
    payload = json.dumps(raw_build_manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def collect_preprocessed_cases(preprocessed_config_dir: Path) -> list[str]:
    if not preprocessed_config_dir.is_dir():
        return []
    cases = []
    for path in sorted(preprocessed_config_dir.glob("*.pkl")):
        if path.name.startswith("._"):
            continue
        cases.append(path.stem)
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit stage-2 masked-CT dataset split/remap/preprocessing consistency.")
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--preprocessed_dataset_dir", type=Path, required=True)
    parser.add_argument("--network", type=str, default="3d_fullres")
    parser.add_argument("--train_patient_max", type=int, default=100)
    parser.add_argument("--fold", type=str, default="all")
    parser.add_argument("--output_json", type=Path, default=None)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    preprocessed_dataset_dir = args.preprocessed_dataset_dir.resolve()
    labels_tr = dataset_dir / "labelsTr"
    labels_ts = dataset_dir / "labelsTs"
    images_tr = dataset_dir / "imagesTr"
    images_ts = dataset_dir / "imagesTs"

    summary: dict[str, object] = {
        "dataset_dir": str(dataset_dir),
        "preprocessed_dataset_dir": str(preprocessed_dataset_dir),
        "network": str(args.network),
        "fold": str(args.fold),
        "errors": [],
        "warnings": [],
    }
    errors: list[str] = summary["errors"]  # type: ignore[assignment]
    warnings: list[str] = summary["warnings"]  # type: ignore[assignment]

    raw_dataset_json_path = dataset_dir / "dataset.json"
    if not raw_dataset_json_path.is_file():
        raise RuntimeError(f"Missing raw dataset.json: {raw_dataset_json_path}")
    raw_dataset_json = json.loads(raw_dataset_json_path.read_text(encoding="utf-8"))
    summary["raw_dataset_json"] = raw_dataset_json

    raw_build_manifest_path = dataset_dir / "dataset_build_manifest.json"
    if not raw_build_manifest_path.is_file():
        errors.append(f"Missing raw dataset build manifest: {raw_build_manifest_path}")
        raw_build_manifest = None
    else:
        raw_build_manifest = json.loads(raw_build_manifest_path.read_text(encoding="utf-8"))
        summary["raw_dataset_build_manifest"] = raw_build_manifest
        if raw_build_manifest.get("dataset_build_schema_version") != DATASET_BUILD_SCHEMA_VERSION:
            errors.append(
                "Raw dataset build schema mismatch: "
                f"expected {DATASET_BUILD_SCHEMA_VERSION}, got {raw_build_manifest.get('dataset_build_schema_version')}"
            )
        if raw_build_manifest.get("image_semantics") != SECOND_STAGE_IMAGE_SEMANTICS:
            errors.append(
                "Raw dataset image semantics mismatch: "
                f"expected {SECOND_STAGE_IMAGE_SEMANTICS}, got {raw_build_manifest.get('image_semantics')}"
            )
        if raw_build_manifest.get("image_generation_mode") != SECOND_STAGE_IMAGE_GENERATION_MODE:
            errors.append(
                "Raw dataset image generation mode mismatch: "
                f"expected {SECOND_STAGE_IMAGE_GENERATION_MODE}, got {raw_build_manifest.get('image_generation_mode')}"
            )

    expected_labels = {
        "background": 0,
        "main fracture segment": 1,
        "secondary fragments": 2,
    }
    if raw_dataset_json.get("labels") != expected_labels:
        errors.append(f"Raw dataset.json labels mismatch: expected {expected_labels}, got {raw_dataset_json.get('labels')}")

    expected_channels = int(len(raw_dataset_json.get("channel_names", {})))
    if expected_channels <= 0:
        errors.append("Raw dataset.json channel_names is empty.")

    train_cases = collect_label_cases(labels_tr)
    test_cases = collect_label_cases(labels_ts)
    summary["num_train_cases"] = len(train_cases)
    summary["num_test_cases"] = len(test_cases)

    train_patients = sorted({parse_case_name(case)[0] for case in train_cases})
    test_patients = sorted({parse_case_name(case)[0] for case in test_cases})
    summary["train_patients"] = train_patients
    summary["test_patients"] = test_patients

    if any(patient_id > int(args.train_patient_max) for patient_id in train_patients):
        errors.append("labelsTr contains patient ids above train_patient_max.")
    if any(patient_id <= int(args.train_patient_max) for patient_id in test_patients):
        errors.append("labelsTs contains patient ids that should be in train.")

    if train_patients and (min(train_patients) != 1 or max(train_patients) != int(args.train_patient_max)):
        warnings.append(
            f"Train patient id range is {min(train_patients)}..{max(train_patients)}, expected 1..{int(args.train_patient_max)}."
        )
    if test_patients and min(test_patients) != int(args.train_patient_max) + 1:
        warnings.append(
            f"Test patients start at {min(test_patients)}, expected {int(args.train_patient_max) + 1}."
        )

    train_image_cases = collect_image_cases(images_tr)
    test_image_cases = collect_image_cases(images_ts)
    if train_image_cases != train_cases:
        errors.append("imagesTr cases do not match labelsTr cases.")
    if test_image_cases != test_cases:
        errors.append("imagesTs cases do not match labelsTs cases.")

    bad_train_channels = list(iter_cases_with_bad_channels(images_tr, expected_channels))
    bad_test_channels = list(iter_cases_with_bad_channels(images_ts, expected_channels))
    if bad_train_channels:
        errors.append(f"imagesTr channel mismatch for cases: {bad_train_channels[:10]}")
    if bad_test_channels:
        errors.append(f"imagesTs channel mismatch for cases: {bad_test_channels[:10]}")

    label_union, bad_label_cases = check_label_values(
        [path for path in sorted(labels_tr.glob("*.nii.gz")) if is_real_nifti(path)]
        + [path for path in sorted(labels_ts.glob("*.nii.gz")) if is_real_nifti(path)]
    )
    summary["label_value_union"] = label_union
    if bad_label_cases:
        errors.append(f"Found labels outside {{0,1,2}} in cases: {dict(list(bad_label_cases.items())[:10])}")

    masked_train_summary, bad_train_masked_cases = check_masked_ct_consistency(images_tr, labels_tr, raw_build_manifest)
    masked_test_summary, bad_test_masked_cases = check_masked_ct_consistency(images_ts, labels_ts, raw_build_manifest)
    summary["masked_ct_train"] = masked_train_summary
    summary["masked_ct_test"] = masked_test_summary
    if bad_train_masked_cases:
        errors.append(
            "imagesTr contains nonzero voxels outside the stage-1 anatomy-derived FracSegNet bone mask for cases: "
            f"{dict(list(bad_train_masked_cases.items())[:10])}"
        )
    if bad_test_masked_cases:
        errors.append(
            "imagesTs contains nonzero voxels outside the stage-1 anatomy-derived FracSegNet bone mask for cases: "
            f"{dict(list(bad_test_masked_cases.items())[:10])}"
        )

    pre_dataset_json_path = preprocessed_dataset_dir / "dataset.json"
    if not pre_dataset_json_path.is_file():
        errors.append(f"Missing preprocessed dataset.json: {pre_dataset_json_path}")
        pre_dataset_json = None
    else:
        pre_dataset_json = json.loads(pre_dataset_json_path.read_text(encoding="utf-8"))
        summary["preprocessed_dataset_json"] = pre_dataset_json
        if pre_dataset_json != raw_dataset_json:
            errors.append("Preprocessed dataset.json does not match raw dataset.json.")

    preprocess_manifest_path = preprocessed_dataset_dir / "fracsegnet_disMap_preprocess.json"
    if not preprocess_manifest_path.is_file():
        errors.append(f"Missing FracSegNet preprocessing manifest: {preprocess_manifest_path}")
    elif raw_build_manifest is not None:
        preprocess_manifest = json.loads(preprocess_manifest_path.read_text(encoding="utf-8"))
        summary["preprocess_manifest"] = preprocess_manifest
        expected_raw_build = raw_build_manifest_for_preprocess(raw_build_manifest)
        expected_digest = raw_build_manifest_digest(expected_raw_build)
        if preprocess_manifest.get("raw_dataset_build") != expected_raw_build:
            errors.append("Preprocessed manifest raw_dataset_build does not match current raw stage-2 dataset build.")
        if preprocess_manifest.get("raw_dataset_build_digest") != expected_digest:
            errors.append("Preprocessed manifest raw_dataset_build_digest does not match current raw stage-2 dataset build.")

    gt_segmentations_dir = preprocessed_dataset_dir / "gt_segmentations"
    gt_cases = collect_label_cases(gt_segmentations_dir)
    summary["num_preprocessed_gt_segmentations"] = len(gt_cases)
    if gt_cases and gt_cases != train_cases:
        errors.append("Preprocessed gt_segmentations cases do not match labelsTr cases.")

    config_dir = preprocessed_dataset_dir / f"nnUNetPlans_{args.network}"
    preprocessed_cases = collect_preprocessed_cases(config_dir)
    summary["num_preprocessed_cases"] = len(preprocessed_cases)
    if preprocessed_cases and preprocessed_cases != train_cases:
        errors.append(
            f"Preprocessed case identifiers do not match labelsTr cases. preprocessed={len(preprocessed_cases)} labelsTr={len(train_cases)}"
        )

    splits_file = preprocessed_dataset_dir / "splits_final.json"
    if splits_file.is_file():
        splits = json.loads(splits_file.read_text(encoding="utf-8"))
        split_case_union = sorted(
            {
                case
                for split in splits
                for key in ("train", "val")
                for case in split.get(key, [])
            }
        )
        summary["num_split_cases"] = len(split_case_union)
        if str(args.fold).lower() == "all":
            if split_case_union != train_cases:
                warnings.append("splits_final.json does not match current labelsTr, but fold=all ignores splits.")
        else:
            if split_case_union != train_cases:
                errors.append("splits_final.json does not match current labelsTr.")
    else:
        warnings.append("splits_final.json not found.")

    output_json = args.output_json.resolve() if args.output_json else preprocessed_dataset_dir / f"audit_stage2_masked_ct_{args.fold}.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"[audit] train_cases={len(train_cases)} test_cases={len(test_cases)} label_values={label_union}")
    print(f"[audit] train_patients={train_patients[:3]}...{train_patients[-3:] if train_patients else []}")
    print(f"[audit] test_patients={test_patients[:3]}...{test_patients[-3:] if test_patients else []}")
    print(f"[audit] report={output_json}")
    if warnings:
        for warning in warnings:
            print(f"[audit][warn] {warning}")
    if errors:
        for error in errors:
            print(f"[audit][error] {error}")
        raise SystemExit(1)
    print("[audit] OK")


if __name__ == "__main__":
    main()
