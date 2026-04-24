from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


DATA_ROOT_ENV_VAR = "PELVIS_SCNP_DATA_ROOT"
DEFAULT_DATA_ROOT = Path(os.environ.get(DATA_ROOT_ENV_VAR, r"D:\Code\Pelvis_SCNP")).resolve()

DATASET_ID = 503
DATASET_NAME = "Dataset503_SCNP"
TRAIN_PATIENT_END = 100
VALID_TEST_PATIENT_END = 150
ROI_BONES = ("LI", "RI", "SA")
DISMAP_FILE_SUFFIX = "_dismap.npy"
DISMAP_PAD_VALUE = 0.2
DISMAP_METADATA_DIRNAME = "_scnp_metadata"
DISMAP_SIDECAR_DIRNAME = "dismaps"

CASE_PATTERN = re.compile(r"^Frac_(\d+)_(LI|RI|SA)$")
FILE_CASE_PATTERN = re.compile(r"^Frac_(\d+)_(LI|RI|SA)\.nii\.gz$")

DATASET_LABELS = {
    "background": 0,
    "main fracture segment": 1,
    "segment 2": 2,
    "segment 3": 3,
}


@dataclass(frozen=True)
class ProjectPaths:
    data_root: Path
    dataset_root: Path
    source_root: Path
    source_images: Path
    source_labels: Path
    nnunet_raw_root: Path
    nnunet_preprocessed_root: Path
    nnunet_results_root: Path

    @property
    def raw_dataset_dir(self) -> Path:
        return self.nnunet_raw_root / DATASET_NAME


def get_project_paths(data_root: str | Path | None = None) -> ProjectPaths:
    candidate = Path(data_root).expanduser().resolve() if data_root is not None else DEFAULT_DATA_ROOT

    if (candidate / "source").is_dir():
        project_root = candidate
        dataset_root = candidate / "dataset"
    elif candidate.name == "dataset" and (candidate.parent / "source").is_dir():
        project_root = candidate.parent
        dataset_root = candidate
    else:
        project_root = candidate
        dataset_root = candidate / "dataset"

    return ProjectPaths(
        data_root=project_root,
        dataset_root=dataset_root,
        source_root=project_root / "source",
        source_images=project_root / "source" / "images",
        source_labels=project_root / "source" / "labels",
        nnunet_raw_root=dataset_root / "nnUNet_raw_data",
        nnunet_preprocessed_root=dataset_root / "nnUNet_preprocessed",
        nnunet_results_root=dataset_root / "nnUNet_results",
    )


def strip_nii_gz(file_name: str) -> str:
    if not file_name.endswith(".nii.gz"):
        raise ValueError(f"Expected a .nii.gz file name, got: {file_name}")
    return file_name[: -len(".nii.gz")]


def parse_case_identifier(case_identifier: str) -> tuple[int, str]:
    match = CASE_PATTERN.match(case_identifier)
    if not match:
        raise ValueError(f"Unexpected ROI case identifier: {case_identifier}")
    patient_id = int(match.group(1))
    bone_code = match.group(2)
    return patient_id, bone_code


def parse_case_file_name(file_name: str) -> tuple[int, str]:
    match = FILE_CASE_PATTERN.match(file_name)
    if not match:
        raise ValueError(f"Unexpected ROI case file name: {file_name}")
    patient_id = int(match.group(1))
    bone_code = match.group(2)
    return patient_id, bone_code


def split_name_for_patient(patient_id: int) -> str:
    if patient_id < 1 or patient_id > VALID_TEST_PATIENT_END:
        raise ValueError(f"Patient id outside supported range [1, {VALID_TEST_PATIENT_END}]: {patient_id}")
    return "train" if patient_id <= TRAIN_PATIENT_END else "test"


def split_name_for_case(case_identifier: str) -> str:
    patient_id, _ = parse_case_identifier(case_identifier)
    return split_name_for_patient(patient_id)


def build_dataset_json(num_training: int) -> dict:
    return {
        "channel_names": {"0": "CT"},
        "labels": DATASET_LABELS,
        "numTraining": int(num_training),
        "file_ending": ".nii.gz",
    }


def dismap_metadata_dir(configuration_dir: str | Path) -> Path:
    return Path(configuration_dir) / DISMAP_METADATA_DIRNAME


def dismap_sidecar_dir(configuration_dir: str | Path) -> Path:
    return dismap_metadata_dir(configuration_dir) / DISMAP_SIDECAR_DIRNAME


def dismap_sidecar_path(configuration_dir: str | Path, identifier: str) -> Path:
    return dismap_sidecar_dir(configuration_dir) / f"{identifier}{DISMAP_FILE_SUFFIX}"


def legacy_dismap_sidecar_path(configuration_dir: str | Path, identifier: str) -> Path:
    return Path(configuration_dir) / f"{identifier}{DISMAP_FILE_SUFFIX}"


def dismap_manifest_path(configuration_dir: str | Path) -> Path:
    return dismap_metadata_dir(configuration_dir) / "disMap_manifest.json"
