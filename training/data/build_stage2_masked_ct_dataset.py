from __future__ import annotations

import argparse
import hashlib
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
    DATASET_BUILD_SCHEMA_VERSION,
    DATASET_ID,
    DATASET_NAME,
    ROI_BONES,
    SECOND_STAGE_IMAGE_GENERATION_MODE,
    SECOND_STAGE_IMAGE_SEMANTICS,
    TRAIN_PATIENT_END,
    VALID_TEST_PATIENT_END,
    build_dataset_json,
    get_project_paths,
    strip_nii_gz,
)


ROI_TO_ANATOMY_LABEL = {
    "SA": 1,
    "LI": 2,
    "RI": 3,
}

FULL_LABEL_BONE_CODE_RANGES = {
    "SA": (1, 10, 0),
    "LI": (11, 20, 10),
    "RI": (21, 30, 20),
}


def _clear_directory(directory: Path) -> None:
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)


def _metadata_close(left: tuple[float, ...], right: tuple[float, ...], atol: float = 1e-4) -> bool:
    return len(left) == len(right) and all(abs(float(a) - float(b)) <= atol for a, b in zip(left, right))


def _assert_same_grid(
    reference_img: sitk.Image,
    candidate_img: sitk.Image,
    reference_src: Path,
    candidate_src: Path,
    candidate_name: str,
) -> None:
    if reference_img.GetSize() != candidate_img.GetSize():
        raise RuntimeError(
            f"Full CT and {candidate_name} sizes do not match. "
            f"ct={reference_src} size={reference_img.GetSize()} "
            f"{candidate_name}={candidate_src} size={candidate_img.GetSize()}"
        )
    if not _metadata_close(reference_img.GetSpacing(), candidate_img.GetSpacing()):
        raise RuntimeError(
            f"Full CT and {candidate_name} spacings do not match. "
            f"ct={reference_src} spacing={reference_img.GetSpacing()} "
            f"{candidate_name}={candidate_src} spacing={candidate_img.GetSpacing()}"
        )
    if not _metadata_close(reference_img.GetOrigin(), candidate_img.GetOrigin()):
        raise RuntimeError(
            f"Full CT and {candidate_name} origins do not match. "
            f"ct={reference_src} origin={reference_img.GetOrigin()} "
            f"{candidate_name}={candidate_src} origin={candidate_img.GetOrigin()}"
        )
    if not _metadata_close(reference_img.GetDirection(), candidate_img.GetDirection()):
        raise RuntimeError(
            f"Full CT and {candidate_name} directions do not match. "
            f"ct={reference_src} direction={reference_img.GetDirection()} "
            f"{candidate_name}={candidate_src} direction={candidate_img.GetDirection()}"
        )


def _case_id(patient_id: int) -> str:
    return f"{patient_id:04d}"


def _case_identifier(patient_id: int, bone_code: str) -> str:
    return f"Frac_{patient_id:04d}_{bone_code}"


def _find_first_existing(candidates: list[Path], description: str) -> Path:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError(
        f"Could not find {description}. Tried:\n"
        + "\n".join(f"- {candidate}" for candidate in candidates)
    )


def _full_ct_path(paths, patient_id: int) -> Path:
    return _find_first_existing(
        [
            paths.full_ct_images / f"FracAnatomy_{patient_id:04d}_0000.nii.gz",
            paths.full_ct_images / f"FracAnatomy_{patient_id:04d}.nii.gz",
            paths.full_ct_images / f"{patient_id:04d}_0000.nii.gz",
            paths.full_ct_images / f"{patient_id:03d}_0000.nii.gz",
        ],
        f"Dataset501 full CT for patient {patient_id:04d}",
    )


def _fracture_label_path(paths, patient_id: int) -> Path:
    return _find_first_existing(
        [
            paths.full_ct_labels / f"FracAnatomy_{patient_id:04d}.nii.gz",
            paths.full_ct_labels / f"{patient_id:04d}.nii.gz",
            paths.full_ct_labels / f"{patient_id:03d}.nii.gz",
        ],
        f"Dataset501 full-grid fracture/anatomy label for patient {patient_id:04d}",
    )


def _anatomy_label_path(paths, patient_id: int) -> Path:
    anatomy_dir = paths.stage1_anatomy_label_dir
    return _find_first_existing(
        [
            anatomy_dir / f"FracAnatomy_{patient_id:04d}.nii.gz",
            anatomy_dir / f"FracAnatomy_{patient_id:04d}.nii.gz",
            anatomy_dir / f"{patient_id:03d}.nii.gz",
            anatomy_dir / f"{patient_id:04d}.nii.gz",
            anatomy_dir / f"{patient_id:03d}_0000.nii.gz",
            anatomy_dir / f"{patient_id:04d}_0000.nii.gz",
        ],
        f"stage-1 anatomy prediction for patient {patient_id:04d}",
    )


def _validate_discrete_label(label_arr: np.ndarray, label_src: Path, label_name: str) -> np.ndarray:
    unique_values = np.unique(label_arr)
    rounded_values = np.rint(unique_values).astype(np.int64)
    if not np.allclose(unique_values, rounded_values):
        raise RuntimeError(f"{label_name} {label_src} contains non-integer values: {unique_values.tolist()}")
    unique_values = rounded_values.tolist()
    if any(value < 0 for value in unique_values):
        raise RuntimeError(f"{label_name} {label_src} contains negative values: {unique_values}")
    return np.rint(label_arr).astype(np.int64, copy=False)


def _map_full_grid_bone_fragments(
    label_arr: np.ndarray,
    bone_code: str,
    stage1_bone_mask: np.ndarray,
    label_src: Path,
) -> tuple[np.ndarray, str]:
    start, end, offset = FULL_LABEL_BONE_CODE_RANGES[bone_code]
    out = np.zeros(label_arr.shape, dtype=np.uint8)

    has_category_fragment_codes = bool(np.any(label_arr > 10))
    if has_category_fragment_codes:
        category_mask = (label_arr >= start) & (label_arr <= end)
        out[category_mask] = (label_arr[category_mask] - offset).astype(np.uint8, copy=False)
        encoding_mode = "dataset501_category_fragment_codes"
    else:
        fallback_mask = (label_arr > 0) & stage1_bone_mask
        out[fallback_mask] = (((label_arr[fallback_mask] - 1) % 10) + 1).astype(np.uint8, copy=False)
        encoding_mode = "dataset501_uncategorized_fragments_split_by_stage1_anatomy"

    if not np.any(out > 0):
        raise RuntimeError(
            f"Full-grid label {label_src} has no voxels for bone {bone_code}. "
            f"Tried encoded values {start}..{end}, then positive labels inside the stage-1 anatomy mask."
        )
    return np.clip(out, 0, 2).astype(np.uint8, copy=False), encoding_mode


def _build_case_record(paths, patient_id: int, bone_code: str) -> dict[str, object]:
    is_train = patient_id <= TRAIN_PATIENT_END
    return {
        "case": _case_identifier(patient_id, bone_code),
        "patient_id": int(patient_id),
        "bone_code": bone_code,
        "split": "train" if is_train else "test",
        "full_ct": str(_full_ct_path(paths, patient_id)),
        "stage1_anatomy_label": str(_anatomy_label_path(paths, patient_id)),
        "target_anatomy_label": int(ROI_TO_ANATOMY_LABEL[bone_code]),
        "fracture_label": str(_fracture_label_path(paths, patient_id)),
        "fracture_label_encoding": "dataset501_full_grid",
    }


def _record_digest(case_records: list[dict[str, object]]) -> str:
    payload = json.dumps(case_records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_stage2_case(
    record: dict[str, object],
    image_dst_dir: Path,
    label_dst_dir: Path,
) -> dict[str, int | float | str]:
    case_identifier = str(record["case"])
    bone_code = str(record["bone_code"])
    patient_id = int(record["patient_id"])
    target_anatomy_label = int(record["target_anatomy_label"])
    full_ct_path = Path(str(record["full_ct"]))
    anatomy_label_path = Path(str(record["stage1_anatomy_label"]))
    fracture_label_path = Path(str(record["fracture_label"]))

    ct_img = sitk.ReadImage(str(full_ct_path))
    anatomy_img = sitk.ReadImage(str(anatomy_label_path))
    fracture_label_img = sitk.ReadImage(str(fracture_label_path))
    _assert_same_grid(ct_img, anatomy_img, full_ct_path, anatomy_label_path, "stage-1 anatomy label")
    _assert_same_grid(ct_img, fracture_label_img, full_ct_path, fracture_label_path, "fracture supervision label")

    ct_arr = sitk.GetArrayFromImage(ct_img)
    anatomy_arr = sitk.GetArrayFromImage(anatomy_img)
    fracture_arr = sitk.GetArrayFromImage(fracture_label_img)
    anatomy_arr = _validate_discrete_label(anatomy_arr, anatomy_label_path, "Stage-1 anatomy label")
    fracture_arr = _validate_discrete_label(fracture_arr, fracture_label_path, "Fracture label")

    bone_mask = anatomy_arr == target_anatomy_label
    if not np.any(bone_mask):
        raise RuntimeError(
            f"Stage-1 anatomy label {anatomy_label_path} has no voxels for "
            f"{bone_code} (label {target_anatomy_label})."
        )

    masked_arr = ct_arr.copy()
    masked_arr[~bone_mask] = 0
    masked_img = sitk.GetImageFromArray(masked_arr.astype(ct_arr.dtype, copy=False))
    masked_img.CopyInformation(ct_img)

    remapped_label, label_encoding_mode = _map_full_grid_bone_fragments(
        fracture_arr,
        bone_code,
        bone_mask,
        fracture_label_path,
    )
    label_img = sitk.GetImageFromArray(remapped_label)
    label_img.CopyInformation(fracture_label_img)

    image_dst_dir.mkdir(parents=True, exist_ok=True)
    label_dst_dir.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(masked_img, str(image_dst_dir / f"{case_identifier}_0000.nii.gz"))
    sitk.WriteImage(label_img, str(label_dst_dir / f"{case_identifier}.nii.gz"))

    total_voxels = int(masked_arr.size)
    mask_voxels = int(np.count_nonzero(bone_mask))
    label_voxels = int(np.count_nonzero(remapped_label > 0))
    outside_nonzero = int(np.count_nonzero(masked_arr[~bone_mask] != 0))
    return {
        "total_voxels": total_voxels,
        "mask_voxels": mask_voxels,
        "label_voxels": label_voxels,
        "outside_mask_nonzero_voxels": outside_nonzero,
        "label_encoding_mode": label_encoding_mode,
        "mask_fraction": float(mask_voxels / total_voxels) if total_voxels else 0.0,
    }


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

    manifest_path = raw_dataset_dir / "dataset_build_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if manifest.get("dataset_build_schema_version") != DATASET_BUILD_SCHEMA_VERSION:
        return False
    if manifest.get("image_semantics") != SECOND_STAGE_IMAGE_SEMANTICS:
        return False
    if manifest.get("image_generation_mode") != SECOND_STAGE_IMAGE_GENERATION_MODE:
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


def _expected_case_records(paths) -> tuple[list[dict[str, object]], list[str], list[str]]:
    case_records: list[dict[str, object]] = []
    for patient_id in range(1, VALID_TEST_PATIENT_END + 1):
        for bone_code in ROI_BONES:
            case_records.append(_build_case_record(paths, patient_id, bone_code))

    train_cases = [str(record["case"]) for record in case_records if record["split"] == "train"]
    test_cases = [str(record["case"]) for record in case_records if record["split"] == "test"]
    return case_records, train_cases, test_cases


def build_raw_dataset(data_root: str | Path | None = None, reset_existing: bool = True) -> Path:
    paths = get_project_paths(data_root)
    raw_dataset_dir = paths.raw_dataset_dir
    case_records, train_cases, test_cases = _expected_case_records(paths)

    if not reset_existing and _raw_dataset_matches_expected(raw_dataset_dir, train_cases, test_cases):
        print(f"[dataset] reusing existing anatomy-derived masked-CT raw dataset at {raw_dataset_dir}")
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

    total_voxels = 0
    mask_voxels = 0
    label_voxels = 0
    outside_mask_nonzero_voxels = 0
    for index, record in enumerate(case_records, start=1):
        if record["split"] == "train":
            stats = _write_stage2_case(record, raw_dataset_dir / "imagesTr", raw_dataset_dir / "labelsTr")
        else:
            stats = _write_stage2_case(record, raw_dataset_dir / "imagesTs", raw_dataset_dir / "labelsTs")
        total_voxels += int(stats["total_voxels"])
        mask_voxels += int(stats["mask_voxels"])
        label_voxels += int(stats["label_voxels"])
        outside_mask_nonzero_voxels += int(stats["outside_mask_nonzero_voxels"])
        if index == 1 or index % 25 == 0 or index == len(case_records):
            print(
                "[dataset] generated anatomy-masked cases "
                f"{index}/{len(case_records)}: {record['case']}"
            )

    dataset_json = build_dataset_json(num_training=len(train_cases))
    (raw_dataset_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=4) + "\n", encoding="utf-8")

    case_records_digest = _record_digest(case_records)
    manifest = {
        "dataset_build_schema_version": DATASET_BUILD_SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "dataset_name": DATASET_NAME,
        "data_root": str(paths.data_root),
        "image_semantics": SECOND_STAGE_IMAGE_SEMANTICS,
        "image_generation_mode": SECOND_STAGE_IMAGE_GENERATION_MODE,
        "image_generation": (
            "FracSegNet second-stage input: the full raw CT grid is masked by the "
            "stage-1 anatomy segmentation label for each target bone. Fracture GT is "
            "used only as the supervision label and for disMap/SCNP loss inputs."
        ),
        "mask_definition": "stage1 anatomy prediction == target bone label",
        "outside_mask_value": 0,
        "source_images_used_directly": False,
        "forbidden_mask_source": "fracture supervision labels are never used to generate the CT input mask",
        "official_fracsegnet_reference": {
            "extract_ct_regions": "inference/extract_ct_regions.py: saveDiffFrac",
            "extract_function": "Training/fracSegNet/basicFunc.py: extractSingleFrac",
            "anatomy_labels": {
                "1": "Sacrum",
                "2": "Left Hip",
                "3": "Right Hip",
            },
        },
        "full_ct_dataset_dir": str(paths.full_ct_dataset_dir),
        "full_ct_images": str(paths.full_ct_images),
        "stage1_anatomy_label_dir": str(paths.stage1_anatomy_label_dir),
        "fracsegnet_anatomical_model_dir": str(paths.fracsegnet_anatomical_model_dir),
        "fracture_label_source_root": str(paths.full_ct_labels),
        "label_schema": "background/main fracture segment/secondary fragments",
        "label_mapping": {
            "0": "background",
            "1": "main fracture segment",
            "2": "secondary fragments",
            "source_values_ge_2": "secondary fragments",
        },
        "remap_policy": (
            "Dataset501 full-grid labels are converted per bone when category-fragment "
            "codes are present (SA 1..10, LI 11..20, RI 21..30 -> fragment ids 1..10). "
            "If a case stores uncategorized fragment ids, positive labels are split by "
            "the stage-1 anatomy mask for that target bone. Any non-main fragment id >= 2 "
            "is collapsed into class 2."
        ),
        "roi_to_anatomy_label": ROI_TO_ANATOMY_LABEL,
        "train_patient_range": [1, TRAIN_PATIENT_END],
        "test_patient_range": [TRAIN_PATIENT_END + 1, VALID_TEST_PATIENT_END],
        "train_patients": TRAIN_PATIENT_END,
        "test_patients": VALID_TEST_PATIENT_END - TRAIN_PATIENT_END,
        "train_cases": len(train_cases),
        "test_cases": len(test_cases),
        "case_records_count": len(case_records),
        "case_records_digest": case_records_digest,
        "case_records": case_records,
        "raw_total_voxels": int(total_voxels),
        "raw_mask_voxels": int(mask_voxels),
        "raw_label_voxels": int(label_voxels),
        "raw_mask_fraction": float(mask_voxels / total_voxels) if total_voxels else 0.0,
        "outside_mask_nonzero_voxels": int(outside_mask_nonzero_voxels),
    }
    (raw_dataset_dir / "dataset_build_manifest.json").write_text(
        json.dumps(manifest, indent=4) + "\n",
        encoding="utf-8",
    )
    paths.source_root.mkdir(parents=True, exist_ok=True)
    (paths.source_root / "source_manifest.json").write_text(
        json.dumps(
            {
                "image_semantics": SECOND_STAGE_IMAGE_SEMANTICS,
                "image_generation_mode": SECOND_STAGE_IMAGE_GENERATION_MODE,
                "full_ct_dataset_dir": str(paths.full_ct_dataset_dir),
                "full_ct_images": str(paths.full_ct_images),
                "stage1_anatomy_label_dir": str(paths.stage1_anatomy_label_dir),
                "fracture_label_source_root": str(paths.full_ct_labels),
                "patient_range": [1, VALID_TEST_PATIENT_END],
                "patient_count": VALID_TEST_PATIENT_END,
                "roi_bones": list(ROI_BONES),
                "case_records_count": len(case_records),
                "case_records_digest": case_records_digest,
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
        description=(
            "Build the stage-2 masked-CT nnU-Net dataset using the official FracSegNet data flow: "
            "full CT + stage-1 anatomy label -> per-bone masked CT, with fracture "
            "labels used only as supervision."
        )
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
        help="Rebuild the stage-2 masked-CT dataset from full CT and stage-1 anatomy predictions.",
    )
    parser.add_argument("--keep_existing", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    raw_dataset_dir = build_raw_dataset(data_root=args.data_root, reset_existing=bool(args.reset_existing))
    print(f"[dataset] built raw dataset at {raw_dataset_dir}")


if __name__ == "__main__":
    main()
