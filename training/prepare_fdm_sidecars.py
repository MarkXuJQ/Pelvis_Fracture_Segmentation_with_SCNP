from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import blosc2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.method.fdm import calculate_dis_map
from training.runtime.project_paths import (
    DATASET_ID,
    DATASET_NAME,
    DISMAP_FILE_SUFFIX,
    dismap_manifest_path,
    dismap_sidecar_path,
    get_project_paths,
    legacy_dismap_sidecar_path,
)


def _resolve_nnunet_cli(name: str) -> str:
    py_dir = Path(sys.executable).resolve().parent
    exe_names = [name]
    if os.name == "nt" and not name.lower().endswith(".exe"):
        exe_names.insert(0, f"{name}.exe")

    for exe_name in exe_names:
        for candidate_dir in (py_dir / "Scripts", py_dir):
            candidate = candidate_dir / exe_name
            if candidate.is_file():
                return str(candidate)
    return name


def _without_training_overlays(existing_pythonpath: str | None) -> str | None:
    if not existing_pythonpath:
        return existing_pythonpath

    filtered_entries: list[str] = []
    for raw_entry in existing_pythonpath.split(os.pathsep):
        if not raw_entry:
            continue
        try:
            resolved_entry = Path(raw_entry).resolve()
        except Exception:
            filtered_entries.append(raw_entry)
            continue
        if REPO_ROOT in resolved_entry.parents and resolved_entry.name == "nnunetv2_overlay":
            continue
        filtered_entries.append(raw_entry)
    return os.pathsep.join(filtered_entries) if filtered_entries else None


def _find_dataset_name(raw_root: Path, task_id: int, dataset_name: str | None = None) -> str:
    if dataset_name:
        candidate = raw_root / dataset_name
        if not candidate.is_dir():
            raise RuntimeError(f"Missing nnU-Net raw dataset directory: {candidate}")
        return dataset_name

    candidates: list[Path] = []
    for pattern in (f"Dataset{task_id:03d}_*", f"Dataset{task_id}_*"):
        candidates.extend(path for path in raw_root.glob(pattern) if path.is_dir())
    names = sorted({path.name for path in candidates})
    if len(names) != 1:
        raise RuntimeError(f"Expected exactly 1 dataset directory for id={task_id}, got: {names}")
    return names[0]


def _sync_gt_segmentations(raw_dataset_dir: Path, preprocessed_dataset_dir: Path) -> None:
    labels_tr_dir = raw_dataset_dir / "labelsTr"
    if not labels_tr_dir.is_dir():
        raise RuntimeError(f"Missing labelsTr directory: {labels_tr_dir}")

    gt_dir = preprocessed_dataset_dir / "gt_segmentations"
    gt_dir.mkdir(parents=True, exist_ok=True)

    raw_labels = sorted(path for path in labels_tr_dir.glob("*.nii.gz") if path.is_file())
    raw_label_names = {path.name for path in raw_labels}
    if not raw_labels:
        raise RuntimeError(f"No training labels found in: {labels_tr_dir}")

    for existing in sorted(gt_dir.glob("*.nii.gz")):
        if existing.name not in raw_label_names:
            existing.unlink()

    for src in raw_labels:
        shutil.copy2(src, gt_dir / src.name)


def _iter_case_identifiers(configuration_dir: Path) -> list[str]:
    identifiers = set()
    for npz_path in configuration_dir.glob("*.npz"):
        identifiers.add(npz_path.stem)
    for npy_path in configuration_dir.glob("*_seg.npy"):
        identifiers.add(npy_path.name[: -len("_seg.npy")])
    for seg_b2nd_path in configuration_dir.glob("*_seg.b2nd"):
        identifiers.add(seg_b2nd_path.name[: -len("_seg.b2nd")])
    return sorted(identifiers)


def _load_segmentation_array(configuration_dir: Path, identifier: str) -> np.ndarray:
    seg_npy = configuration_dir / f"{identifier}_seg.npy"
    if seg_npy.is_file():
        return np.load(seg_npy, mmap_mode="r")

    seg_b2nd = configuration_dir / f"{identifier}_seg.b2nd"
    if seg_b2nd.is_file():
        seg = blosc2.open(urlpath=str(seg_b2nd), mode="r")
        return np.asarray(seg[:])

    case_npz = configuration_dir / f"{identifier}.npz"
    if case_npz.is_file():
        with np.load(case_npz) as case_data:
            if "seg" not in case_data:
                raise RuntimeError(f"Missing 'seg' array in {case_npz}")
            return case_data["seg"]

    raise FileNotFoundError(f"Could not find preprocessed segmentation for {identifier} in {configuration_dir}")


def _write_dismap_sidecar(configuration_dir: Path, identifier: str, overwrite: bool = False) -> Path:
    sidecar_path = dismap_sidecar_path(configuration_dir, identifier)
    if sidecar_path.is_file() and not overwrite:
        return sidecar_path

    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    seg = _load_segmentation_array(configuration_dir, identifier)
    dis_map = calculate_dis_map(seg)
    np.save(sidecar_path, dis_map.astype(np.float32, copy=False))

    legacy_path = legacy_dismap_sidecar_path(configuration_dir, identifier)
    if legacy_path.is_file():
        legacy_path.unlink()
    return sidecar_path


def _migrate_legacy_dismap_sidecars(configuration_dir: Path) -> int:
    moved = 0
    for legacy_path in sorted(configuration_dir.glob(f"*{DISMAP_FILE_SUFFIX}")):
        identifier = legacy_path.name[: -len(DISMAP_FILE_SUFFIX)]
        target_path = dismap_sidecar_path(configuration_dir, identifier)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.is_file():
            legacy_path.unlink()
            continue
        shutil.move(str(legacy_path), str(target_path))
        moved += 1
    return moved


def _find_configuration_dirs(preprocessed_dataset_dir: Path, requested_configs: list[str] | None) -> list[Path]:
    if requested_configs:
        dirs = [preprocessed_dataset_dir / f"nnUNetPlans_{config}" for config in requested_configs]
        missing = [str(path) for path in dirs if not path.is_dir()]
        if missing:
            raise RuntimeError(f"Requested preprocessed configuration directories are missing: {missing}")
        return dirs

    discovered = []
    for child in sorted(preprocessed_dataset_dir.iterdir()):
        if child.is_dir() and child.name.startswith("nnUNetPlans_") and _iter_case_identifiers(child):
            discovered.append(child)
    return discovered


def _configuration_has_complete_dismaps(configuration_dir: Path) -> bool:
    identifiers = _iter_case_identifiers(configuration_dir)
    return bool(identifiers) and all(dismap_sidecar_path(configuration_dir, item).is_file() for item in identifiers)


def _generate_configuration_dismaps(
    configuration_dir: Path,
    max_workers: int,
    overwrite: bool = False,
) -> dict:
    identifiers = _iter_case_identifiers(configuration_dir)
    if not identifiers:
        return {"configuration": configuration_dir.name, "cases": 0, "workers": max_workers}

    migrated_legacy = _migrate_legacy_dismap_sidecars(configuration_dir)
    if max_workers <= 1:
        for identifier in identifiers:
            _write_dismap_sidecar(configuration_dir, identifier, overwrite=overwrite)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_write_dismap_sidecar, configuration_dir, identifier, overwrite)
                for identifier in identifiers
            ]
            for future in futures:
                future.result()

    manifest = {
        "configuration": configuration_dir.name,
        "cases": len(identifiers),
        "workers": int(max_workers),
        "sidecar_suffix": DISMAP_FILE_SUFFIX,
        "generator": "FDM/disMap from the three-class fracture label space",
        "label_schema": {
            "0": "background",
            "1": "main fracture segment",
            "2": "secondary fragments",
        },
        "storage_dir": "_scnp_metadata/dismaps",
        "legacy_sidecars_migrated": int(migrated_legacy),
    }
    manifest_path = dismap_manifest_path(configuration_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=4) + "\n", encoding="utf-8")
    return manifest


def _default_dismap_workers(requested: int | None) -> int:
    if requested is not None:
        return max(1, int(requested))
    cpu_count = os.cpu_count() or 1
    return max(1, min(4, cpu_count // 2 or 1))


def main() -> None:
    defaults = get_project_paths()
    parser = argparse.ArgumentParser(
        description=(
            "Generate FDM/disMap sidecars for a standard nnU-Net v2 dataset. "
            "The raw dataset must already use the three-class fracture label schema."
        )
    )
    parser.add_argument("--raw_base", type=Path, default=defaults.data_root)
    parser.add_argument("--preprocessed", type=Path, default=defaults.nnunet_preprocessed_root)
    parser.add_argument("--results", type=Path, default=defaults.nnunet_results_root)
    parser.add_argument("--task_id", type=int, default=DATASET_ID)
    parser.add_argument("--dataset_name", type=str, default=DATASET_NAME)
    parser.add_argument("--run_preprocessing", action="store_true")
    parser.add_argument("--verify_dataset_integrity", action="store_true")
    parser.add_argument("--reset_preprocess", action="store_true")
    parser.add_argument("--configs", nargs="*", default=None)
    parser.add_argument("--np", type=int, default=None)
    parser.add_argument("--dismap_workers", type=int, default=None)
    parser.add_argument("--overwrite_dismap", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    raw_base = args.raw_base.resolve()
    paths = get_project_paths(raw_base)
    raw_root = paths.nnunet_raw_root
    preprocessed = args.preprocessed.resolve()
    results = args.results.resolve()
    dataset_name = _find_dataset_name(raw_root, int(args.task_id), args.dataset_name)
    raw_dataset_dir = raw_root / dataset_name
    preprocessed_dataset_dir = preprocessed / dataset_name

    os.environ["nnUNet_raw"] = str(raw_root)
    os.environ["nnUNet_preprocessed"] = str(preprocessed)
    os.environ["nnUNet_results"] = str(results)

    preprocessed.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    if args.reset_preprocess:
        shutil.rmtree(preprocessed_dataset_dir, ignore_errors=True)

    if args.run_preprocessing:
        env = os.environ.copy()
        sanitized_pythonpath = _without_training_overlays(env.get("PYTHONPATH"))
        if sanitized_pythonpath is None:
            env.pop("PYTHONPATH", None)
        else:
            env["PYTHONPATH"] = sanitized_pythonpath

        extract_cmd = [_resolve_nnunet_cli("nnUNetv2_extract_fingerprint"), "-d", str(int(args.task_id)), "--clean"]
        if args.verify_dataset_integrity:
            extract_cmd.append("--verify_dataset_integrity")
        if args.np is not None:
            extract_cmd += ["-np", str(int(args.np))]
        if args.verbose:
            extract_cmd.append("--verbose")
        subprocess.check_call(extract_cmd, env=env)

        plan_cmd = [_resolve_nnunet_cli("nnUNetv2_plan_experiment"), "-d", str(int(args.task_id))]
        if args.verbose:
            plan_cmd.append("--verbose")
        subprocess.check_call(plan_cmd, env=env)

        preprocess_cmd = [_resolve_nnunet_cli("nnUNetv2_preprocess"), "-d", str(int(args.task_id))]
        if args.configs:
            preprocess_cmd += ["-c", *args.configs]
        if args.np is not None:
            preprocess_cmd += ["-np", str(int(args.np))]
        if args.verbose:
            preprocess_cmd.append("--verbose")
        subprocess.check_call(preprocess_cmd, env=env)

    if not preprocessed_dataset_dir.is_dir():
        raise RuntimeError(
            f"Missing preprocessed dataset: {preprocessed_dataset_dir}. "
            "Run with --run_preprocessing or run nnU-Net preprocessing first."
        )

    _sync_gt_segmentations(raw_dataset_dir, preprocessed_dataset_dir)

    configuration_dirs = _find_configuration_dirs(preprocessed_dataset_dir, args.configs)
    if not configuration_dirs:
        raise RuntimeError(f"No preprocessed configuration directories were found in {preprocessed_dataset_dir}")

    dismap_workers = _default_dismap_workers(args.dismap_workers)
    config_manifests = []
    for configuration_dir in configuration_dirs:
        overwrite = bool(args.overwrite_dismap or args.reset_preprocess or not _configuration_has_complete_dismaps(configuration_dir))
        config_manifests.append(_generate_configuration_dismaps(configuration_dir, dismap_workers, overwrite=overwrite))

    (preprocessed_dataset_dir / "scnp_fdm_sidecars.json").write_text(
        json.dumps(
            {
                "dataset_name": dataset_name,
                "dataset_id": int(args.task_id),
                "generator": "FDM/disMap sidecars for SCNP training",
                "configurations": config_manifests,
            },
            indent=4,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
