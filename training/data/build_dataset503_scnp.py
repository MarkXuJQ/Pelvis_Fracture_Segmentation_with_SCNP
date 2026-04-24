from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from training.runtime.project_paths import (
    DATASET_ID,
    DATASET_NAME,
    ROI_BONES,
    TRAIN_PATIENT_END,
    VALID_TEST_PATIENT_END,
    build_dataset_json,
    get_project_paths,
    parse_case_identifier,
    strip_nii_gz,
)


def _clear_directory(directory: Path) -> None:
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)


def _collect_source_cases(source_dir: Path) -> dict[str, Path]:
    case_map: dict[str, Path] = {}
    for file_path in sorted(source_dir.glob("*.nii.gz")):
        case_identifier = strip_nii_gz(file_path.name)
        case_map[case_identifier] = file_path
    return case_map


def _validate_roi_source_labels(label_arr: np.ndarray, label_src: Path) -> np.ndarray:
    unique_values = np.unique(label_arr)
    rounded_values = np.rint(unique_values).astype(np.int64)

    if not np.allclose(unique_values, rounded_values):
        raise RuntimeError(
            f"Source label {label_src} contains non-integer values: {unique_values.tolist()}"
        )

    unique_values = rounded_values.tolist()
    if any(value < 0 for value in unique_values):
        raise RuntimeError(
            f"Source label {label_src} contains negative values: {unique_values}"
        )

    foreground_values = [value for value in unique_values if value > 0]
    if not foreground_values:
        raise RuntimeError(
            f"Source label {label_src} does not contain any foreground fragment labels."
        )

    if 1 not in foreground_values:
        raise RuntimeError(
            "FracSegNet-style ROI labels must encode the main fracture segment as 1 and "
            f"additional fragments as values > 1. Offending file: {label_src} "
            f"(unique labels: {unique_values})"
        )

    return np.rint(label_arr).astype(np.int64, copy=False)


def _remap_label_array(label_arr: np.ndarray) -> np.ndarray:
    # FracSegNet fracture-stage targets are fixed to 4 classes:
    # 0 background / 1 main fracture segment / 2 segment 2 / 3 segment 3.
    # Our source ROI labels can contain more than three foreground fragments,
    # so any source fragment id >= 4 is collapsed into class 3.
    return np.clip(label_arr, 0, 3).astype(np.uint8, copy=False)


def _remap_label_image(label_src: Path, label_dst: Path) -> None:
    label_img = sitk.ReadImage(str(label_src))
    label_arr = sitk.GetArrayFromImage(label_img)
    label_arr = _validate_roi_source_labels(label_arr, label_src)
    remapped = _remap_label_array(label_arr)

    remapped_img = sitk.GetImageFromArray(remapped)
    remapped_img.CopyInformation(label_img)
    sitk.WriteImage(remapped_img, str(label_dst))


def _copy_case(image_src: Path, label_src: Path, image_dst_dir: Path, label_dst_dir: Path) -> None:
    image_dst_dir.mkdir(parents=True, exist_ok=True)
    label_dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_src, image_dst_dir / f"{strip_nii_gz(image_src.name)}_0000.nii.gz")
    _remap_label_image(label_src, label_dst_dir / label_src.name)


def _collect_built_image_cases(directory: Path) -> set[str]:
    cases: set[str] = set()
    for file_path in directory.glob("*_0000.nii.gz"):
        case_name = strip_nii_gz(file_path.name)
        if case_name.endswith("_0000"):
            case_name = case_name[: -len("_0000")]
        cases.add(case_name)
    return cases


def _collect_built_label_cases(directory: Path) -> set[str]:
    return {strip_nii_gz(file_path.name) for file_path in directory.glob("*.nii.gz")}


def _raw_dataset_matches_expected(raw_dataset_dir: Path, train_cases: list[str], test_cases: list[str]) -> bool:
    dataset_json_path = raw_dataset_dir / "dataset.json"
    if not dataset_json_path.is_file():
        return False

    expected_dataset_json = build_dataset_json(num_training=len(train_cases))
    try:
        existing_dataset_json = json.loads(dataset_json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if existing_dataset_json != expected_dataset_json:
        return False

    expected = {
        "imagesTr": set(train_cases),
        "labelsTr": set(train_cases),
        "imagesTs": set(test_cases),
        "labelsTs": set(test_cases),
    }
    actual = {
        "imagesTr": _collect_built_image_cases(raw_dataset_dir / "imagesTr"),
        "labelsTr": _collect_built_label_cases(raw_dataset_dir / "labelsTr"),
        "imagesTs": _collect_built_image_cases(raw_dataset_dir / "imagesTs"),
        "labelsTs": _collect_built_label_cases(raw_dataset_dir / "labelsTs"),
    }
    return actual == expected


def _expected_case_names() -> tuple[list[str], list[str]]:
    train_cases = [
        f"Frac_{patient_id:04d}_{bone_code}"
        for patient_id in range(1, TRAIN_PATIENT_END + 1)
        for bone_code in ROI_BONES
    ]
    test_cases = [
        f"Frac_{patient_id:04d}_{bone_code}"
        for patient_id in range(TRAIN_PATIENT_END + 1, VALID_TEST_PATIENT_END + 1)
        for bone_code in ROI_BONES
    ]
    return train_cases, test_cases


def build_raw_dataset(data_root: str | Path | None = None, reset_existing: bool = True) -> Path:
    paths = get_project_paths(data_root)
    source_images = _collect_source_cases(paths.source_images) if paths.source_images.is_dir() else {}
    source_labels = _collect_source_cases(paths.source_labels) if paths.source_labels.is_dir() else {}
    raw_dataset_dir = paths.raw_dataset_dir

    if not source_images or not source_labels:
        expected_train_cases, expected_test_cases = _expected_case_names()
        if _raw_dataset_matches_expected(raw_dataset_dir, expected_train_cases, expected_test_cases):
            print(
                "[dataset] source/ directory missing or incomplete, "
                f"reusing existing raw dataset at {raw_dataset_dir}"
            )
            return raw_dataset_dir

    if not source_images:
        raise RuntimeError(f"No source CT images were found in {paths.source_images}")
    if not source_labels:
        raise RuntimeError(f"No source labels were found in {paths.source_labels}")
    if set(source_images) != set(source_labels):
        missing_images = sorted(set(source_labels) - set(source_images))
        missing_labels = sorted(set(source_images) - set(source_labels))
        raise RuntimeError(
            "Source images and labels do not match.\n"
            f"Missing images for labels: {missing_images[:10]}\n"
            f"Missing labels for images: {missing_labels[:10]}"
        )

    case_records: list[tuple[str, Path, Path, bool, int]] = []
    for case_identifier in sorted(source_images):
        patient_id, bone_code = parse_case_identifier(case_identifier)
        if bone_code not in ("LI", "RI", "SA"):
            raise RuntimeError(f"Unexpected ROI bone code in case {case_identifier}: {bone_code}")
        if patient_id < 1 or patient_id > VALID_TEST_PATIENT_END:
            raise RuntimeError(
                f"Patient id outside supported range [1, {VALID_TEST_PATIENT_END}] in case {case_identifier}"
            )

        is_train = patient_id <= TRAIN_PATIENT_END
        case_records.append((case_identifier, source_images[case_identifier], source_labels[case_identifier], is_train, patient_id))

    train_cases = [case_identifier for case_identifier, _, _, is_train, _ in case_records if is_train]
    test_cases = [case_identifier for case_identifier, _, _, is_train, _ in case_records if not is_train]
    train_patients = {patient_id for _, _, _, is_train, patient_id in case_records if is_train}
    test_patients = {patient_id for _, _, _, is_train, patient_id in case_records if not is_train}

    if not reset_existing and _raw_dataset_matches_expected(raw_dataset_dir, train_cases, test_cases):
        print(f"[dataset] reusing existing raw dataset at {raw_dataset_dir}")
        return raw_dataset_dir

    if reset_existing:
        _clear_directory(raw_dataset_dir / "imagesTr")
        _clear_directory(raw_dataset_dir / "imagesTs")
        _clear_directory(raw_dataset_dir / "labelsTr")
        _clear_directory(raw_dataset_dir / "labelsTs")
    else:
        (raw_dataset_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
        (raw_dataset_dir / "imagesTs").mkdir(parents=True, exist_ok=True)
        (raw_dataset_dir / "labelsTr").mkdir(parents=True, exist_ok=True)
        (raw_dataset_dir / "labelsTs").mkdir(parents=True, exist_ok=True)

    for case_identifier, image_src, label_src, is_train, _ in case_records:
        if is_train:
            _copy_case(image_src, label_src, raw_dataset_dir / "imagesTr", raw_dataset_dir / "labelsTr")
        else:
            _copy_case(image_src, label_src, raw_dataset_dir / "imagesTs", raw_dataset_dir / "labelsTs")

    dataset_json = build_dataset_json(num_training=len(train_cases))
    (raw_dataset_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=4) + "\n", encoding="utf-8")

    manifest = {
        "dataset_id": DATASET_ID,
        "dataset_name": DATASET_NAME,
        "data_root": str(paths.data_root),
        "source_images": str(paths.source_images),
        "source_labels": str(paths.source_labels),
        "source_label_requirement": "ROI fracture instance labels with main fracture segment encoded as 1 and additional fragments encoded as values 2..N",
        "label_schema": "background/main fracture segment/segment 2/segment 3",
        "label_mapping": {
            "0": "background",
            "1": "main fracture segment",
            "2": "segment 2",
            "3": "segment 3",
            "source_values_ge_4": "segment 3",
        },
        "remap_policy": "preserve 0/1/2/3 and collapse any source fragment label >= 4 into class 3 to match the official FracSegNet fracture-stage label space",
        "train_patient_range": [1, TRAIN_PATIENT_END],
        "test_patient_range": [TRAIN_PATIENT_END + 1, VALID_TEST_PATIENT_END],
        "train_patients": len(train_patients),
        "test_patients": len(test_patients),
        "train_cases": len(train_cases),
        "test_cases": len(test_cases),
    }
    (raw_dataset_dir / "dataset_build_manifest.json").write_text(
        json.dumps(manifest, indent=4) + "\n",
        encoding="utf-8",
    )
    (paths.source_root / "source_manifest.json").write_text(
        json.dumps(
            {
                "source_images": len(source_images),
                "source_labels": len(source_labels),
                "patient_range": [1, VALID_TEST_PATIENT_END],
                "patient_count": VALID_TEST_PATIENT_END,
                "roi_bones": ["LI", "RI", "SA"],
            },
            indent=4,
        )
        + "\n",
        encoding="utf-8",
    )

    return raw_dataset_dir


def main() -> None:
    defaults = get_project_paths()
    parser = argparse.ArgumentParser(
        description="Build the CT-only nnUNet raw dataset used by the SCNP experiments."
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=defaults.data_root,
        help="Project root that contains source/ and the dataset/ workspace.",
    )
    parser.add_argument(
        "--reset_existing",
        action="store_true",
        help="Rebuild the Dataset503_SCNP raw dataset directory from source data.",
    )
    parser.add_argument("--keep_existing", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    raw_dataset_dir = build_raw_dataset(data_root=args.data_root, reset_existing=bool(args.reset_existing))
    print(f"[dataset] built raw dataset at {raw_dataset_dir}")


if __name__ == "__main__":
    main()
