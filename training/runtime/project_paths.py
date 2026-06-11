from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


DATA_ROOT_ENV_VAR = "PELVIS_SCNP_DATA_ROOT"
DEFAULT_DATA_ROOT = Path(
    os.environ.get(DATA_ROOT_ENV_VAR, Path(__file__).resolve().parents[2])
).resolve()
FULL_CT_DATASET_DIR_ENV_VAR = "PELVIS_SCNP_FULL_CT_DATASET_DIR"
DEFAULT_FULL_CT_DATASET_DIR = Path(
    os.environ.get(
        FULL_CT_DATASET_DIR_ENV_VAR,
        str(DEFAULT_DATA_ROOT / "external" / "Dataset501_FracAnatomy"),
    )
).resolve()
STAGE1_ANATOMY_LABEL_DIR_ENV_VAR = "PELVIS_SCNP_STAGE1_ANATOMY_LABEL_DIR"
FRACSEGNET_ANATOMICAL_MODEL_DIR_ENV_VAR = "PELVIS_SCNP_FRACSEGNET_ANATOMICAL_MODEL_DIR"
DEFAULT_FRACSEGNET_ANATOMICAL_MODEL_DIR = Path(
    os.environ.get(
        FRACSEGNET_ANATOMICAL_MODEL_DIR_ENV_VAR,
        str(DEFAULT_DATA_ROOT / "external" / "FracSegNet" / "AnatomicalSegModel"),
    )
).resolve()

DATASET_ID_ENV_VAR = "PELVIS_SCNP_NNUNET_DATASET_ID"
DATASET_NAME_ENV_VAR = "PELVIS_SCNP_NNUNET_DATASET_NAME"
DATASET_ID = int(os.environ.get(DATASET_ID_ENV_VAR, "520"))
DATASET_NAME = os.environ.get(DATASET_NAME_ENV_VAR, f"Dataset{DATASET_ID:03d}_PelvisSCNP")
DATASET_BUILD_SCHEMA_VERSION = 5
SECOND_STAGE_IMAGE_SEMANTICS = "fracsegnet_masked_ct"
SECOND_STAGE_IMAGE_GENERATION_MODE = "masked_from_stage1_anatomy_label"
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
    "secondary fragments": 2,
}


@dataclass(frozen=True)
class ProjectPaths:
    data_root: Path
    dataset_root: Path
    source_root: Path
    source_images: Path
    source_labels: Path
    full_ct_dataset_dir: Path
    full_ct_images: Path
    full_ct_labels: Path
    stage1_anatomy_label_dir: Path
    fracsegnet_anatomical_model_dir: Path
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
        full_ct_dataset_dir=DEFAULT_FULL_CT_DATASET_DIR,
        full_ct_images=DEFAULT_FULL_CT_DATASET_DIR / "imagesTr",
        full_ct_labels=DEFAULT_FULL_CT_DATASET_DIR / "labelsTr",
        stage1_anatomy_label_dir=(
            Path(os.environ[STAGE1_ANATOMY_LABEL_DIR_ENV_VAR]).expanduser().resolve()
            if STAGE1_ANATOMY_LABEL_DIR_ENV_VAR in os.environ
            else dataset_root / "stage1_anatomy_predictions" / "anatomy_cascade"
        ),
        fracsegnet_anatomical_model_dir=DEFAULT_FRACSEGNET_ANATOMICAL_MODEL_DIR,
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
        raise ValueError(f"Unexpected stage-2 per-bone case identifier: {case_identifier}")
    patient_id = int(match.group(1))
    bone_code = match.group(2)
    return patient_id, bone_code


def parse_case_file_name(file_name: str) -> tuple[int, str]:
    match = FILE_CASE_PATTERN.match(file_name)
    if not match:
        raise ValueError(f"Unexpected stage-2 per-bone case file name: {file_name}")
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
