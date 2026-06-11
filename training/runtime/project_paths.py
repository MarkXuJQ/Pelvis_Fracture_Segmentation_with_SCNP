from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DATA_ROOT_ENV_VAR = "PELVIS_SCNP_DATA_ROOT"
DATASET_ID_ENV_VAR = "PELVIS_SCNP_NNUNET_DATASET_ID"
DATASET_NAME_ENV_VAR = "PELVIS_SCNP_NNUNET_DATASET_NAME"

DEFAULT_DATA_ROOT = Path(os.environ.get(DATA_ROOT_ENV_VAR, Path(__file__).resolve().parents[2])).resolve()
DATASET_ID = int(os.environ.get(DATASET_ID_ENV_VAR, "520"))
DATASET_NAME = os.environ.get(DATASET_NAME_ENV_VAR, f"Dataset{DATASET_ID:03d}_PelvisSCNP")

DISMAP_FILE_SUFFIX = "_dismap.npy"
DISMAP_PAD_VALUE = 0.2
DISMAP_METADATA_DIRNAME = "_scnp_metadata"
DISMAP_SIDECAR_DIRNAME = "dismaps"

DATASET_LABELS = {
    "background": 0,
    "main fracture segment": 1,
    "secondary fragments": 2,
}


@dataclass(frozen=True)
class ProjectPaths:
    data_root: Path
    dataset_root: Path
    nnunet_raw_root: Path
    nnunet_preprocessed_root: Path
    nnunet_results_root: Path

    @property
    def raw_dataset_dir(self) -> Path:
        return self.nnunet_raw_root / DATASET_NAME


def get_project_paths(data_root: str | Path | None = None) -> ProjectPaths:
    root = Path(data_root).expanduser().resolve() if data_root is not None else DEFAULT_DATA_ROOT
    dataset_root = root / "dataset"
    return ProjectPaths(
        data_root=root,
        dataset_root=dataset_root,
        nnunet_raw_root=dataset_root / "nnUNet_raw_data",
        nnunet_preprocessed_root=dataset_root / "nnUNet_preprocessed",
        nnunet_results_root=dataset_root / "nnUNet_results",
    )


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
