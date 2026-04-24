from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import SimpleITK as sitk


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
        raise RuntimeError(f"Unexpected ROI case name format: {case_stem}")
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
    parser = argparse.ArgumentParser(description="Audit Dataset503 split/remap/preprocessing consistency.")
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

    expected_labels = {
        "background": 0,
        "main fracture segment": 1,
        "segment 2": 2,
        "segment 3": 3,
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
        errors.append(f"Found labels outside {{0,1,2,3}} in cases: {dict(list(bad_label_cases.items())[:10])}")

    pre_dataset_json_path = preprocessed_dataset_dir / "dataset.json"
    if not pre_dataset_json_path.is_file():
        errors.append(f"Missing preprocessed dataset.json: {pre_dataset_json_path}")
        pre_dataset_json = None
    else:
        pre_dataset_json = json.loads(pre_dataset_json_path.read_text(encoding="utf-8"))
        summary["preprocessed_dataset_json"] = pre_dataset_json
        if pre_dataset_json != raw_dataset_json:
            errors.append("Preprocessed dataset.json does not match raw dataset.json.")

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

    output_json = args.output_json.resolve() if args.output_json else preprocessed_dataset_dir / f"audit_dataset503_{args.fold}.json"
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
