from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from training.runtime.project_paths import DISMAP_FILE_SUFFIX, dismap_sidecar_path, get_project_paths


def build_overlay_pythonpath(
    existing_pythonpath: str | None,
    overlay_root: Path,
    required_relative_paths: list[str],
    repo_root: Path | None = None,
) -> str:
    overlay_root = overlay_root.resolve()
    repo_root = repo_root.resolve() if repo_root is not None else overlay_root.parents[1]

    missing = [
        str(overlay_root / relative_path)
        for relative_path in required_relative_paths
        if not (overlay_root / relative_path).is_file()
    ]
    if missing:
        raise RuntimeError("Missing nnU-Net overlay files:\n- " + "\n- ".join(missing))

    pythonpath_entries = [str(overlay_root), str(repo_root)]
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    return os.pathsep.join(pythonpath_entries)


def resolve_train_exe(python_executable: str | None = None) -> str:
    python_executable = python_executable or sys.executable
    py_dir = Path(python_executable).resolve().parent
    exe_name = "nnUNetv2_train.exe" if os.name == "nt" else "nnUNetv2_train"
    for candidate_dir in (py_dir / "Scripts", py_dir):
        candidate = candidate_dir / exe_name
        if candidate.is_file():
            return str(candidate)
    return "nnUNetv2_train"


def resolve_predict_exe(python_executable: str | None = None) -> str:
    python_executable = python_executable or sys.executable
    py_dir = Path(python_executable).resolve().parent
    exe_name = "nnUNetv2_predict.exe" if os.name == "nt" else "nnUNetv2_predict"
    for candidate_dir in (py_dir / "Scripts", py_dir):
        candidate = candidate_dir / exe_name
        if candidate.is_file():
            return str(candidate)
    return "nnUNetv2_predict"


def resolve_dataset_name(v2_raw_root: str | Path, task_id: int) -> str:
    candidates: list[Path] = []
    v2_raw_root = Path(v2_raw_root)
    for pattern in (f"Dataset{task_id:03d}_*", f"Dataset{task_id}_*"):
        candidates.extend(path for path in v2_raw_root.glob(pattern) if path.is_dir())
    names = sorted({path.name for path in candidates})
    if len(names) != 1:
        raise RuntimeError(f"Expected exactly 1 dataset directory for id={task_id}, got: {names}")
    return names[0]


def _dataset_label_values(dataset_json: dict) -> set[int]:
    labels = dataset_json.get("labels", {})
    values: set[int] = set()
    if not isinstance(labels, dict):
        return values
    for value in labels.values():
        if isinstance(value, int):
            values.add(int(value))
    return values


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"Missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_case_identifiers(configuration_dir: Path) -> list[str]:
    identifiers = set()
    for npz_path in configuration_dir.glob("*.npz"):
        identifiers.add(npz_path.stem)
    for npy_path in configuration_dir.glob("*_seg.npy"):
        identifiers.add(npy_path.name[: -len("_seg.npy")])
    for seg_b2nd_path in configuration_dir.glob("*_seg.b2nd"):
        identifiers.add(seg_b2nd_path.name[: -len("_seg.b2nd")])
    return sorted(identifiers)


def ensure_preprocessed_dataset_matches_raw(
    raw_base: str | Path,
    preprocessed: str | Path,
    task_id: int,
    network: str = "3d_fullres",
) -> None:
    dataset_paths = get_project_paths(raw_base)
    dataset_name = resolve_dataset_name(dataset_paths.nnunet_raw_root, task_id)
    raw_dataset_dir = dataset_paths.nnunet_raw_root / dataset_name
    raw_dataset_json_path = raw_dataset_dir / "dataset.json"
    preprocessed_dataset_dir = Path(preprocessed).resolve() / dataset_name
    preprocessed_dataset_json_path = preprocessed_dataset_dir / "dataset.json"

    raw_dataset_json = _load_json(raw_dataset_json_path)
    preprocessed_dataset_json = _load_json(preprocessed_dataset_json_path)
    if raw_dataset_json != preprocessed_dataset_json:
        raise RuntimeError(
            "The preprocessed nnU-Net cache does not match the current raw dataset.json. "
            "Re-run preprocessing and FDM sidecar generation before training."
        )

    expected_values = {0, 1, 2}
    if _dataset_label_values(raw_dataset_json) != expected_values:
        raise RuntimeError(
            "This SCNP example expects a three-class dataset.json with labels "
            "{0 background, 1 main fracture segment, 2 secondary fragments}."
        )

    configuration_dir = preprocessed_dataset_dir / f"nnUNetPlans_{network}"
    if not configuration_dir.is_dir():
        raise RuntimeError(f"Missing preprocessed configuration directory: {configuration_dir}")

    case_ids = _iter_case_identifiers(configuration_dir)
    if not case_ids:
        raise RuntimeError(f"No preprocessed training cases found in: {configuration_dir}")

    missing_dismaps = [case_id for case_id in case_ids if not dismap_sidecar_path(configuration_dir, case_id).is_file()]
    if missing_dismaps:
        raise RuntimeError(
            "Missing FDM/disMap sidecars for SCNP training. "
            "Run `python training/prepare_fdm_sidecars.py --run_preprocessing` first. "
            f"Examples: {missing_dismaps[:10]}"
        )

    normalize_preprocessed_dismap_layout(preprocessed_dataset_dir)


def normalize_preprocessed_dismap_layout(preprocessed_dataset_dir: Path) -> None:
    preprocessed_dataset_dir = Path(preprocessed_dataset_dir).resolve()
    if not preprocessed_dataset_dir.is_dir():
        return

    moved = 0
    removed_duplicates = 0
    for configuration_dir in sorted(preprocessed_dataset_dir.glob("nnUNetPlans_*")):
        if not configuration_dir.is_dir():
            continue
        for legacy_path in sorted(configuration_dir.glob(f"*{DISMAP_FILE_SUFFIX}")):
            identifier = legacy_path.name[: -len(DISMAP_FILE_SUFFIX)]
            target_path = dismap_sidecar_path(configuration_dir, identifier)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if target_path.is_file():
                legacy_path.unlink()
                removed_duplicates += 1
                continue
            shutil.move(str(legacy_path), str(target_path))
            moved += 1

    if moved or removed_duplicates:
        print(
            "[runtime] normalized disMap sidecars: "
            f"moved={moved}, removed_duplicates={removed_duplicates}, "
            f"dataset={preprocessed_dataset_dir}"
        )


def resolve_n_proc_da(cli_value: int | None) -> int | None:
    if cli_value is not None:
        return int(cli_value)

    env_value = os.environ.get("nnUNet_n_proc_DA")
    if env_value is not None and str(env_value).strip() != "":
        return int(env_value)

    if os.name == "nt":
        return 6
    return None


def apply_windows_runtime_limits() -> None:
    if os.name != "nt":
        return

    defaults = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "OMP_WAIT_POLICY": "PASSIVE",
        "KMP_BLOCKTIME": "0",
    }
    applied: dict[str, str] = {}
    for key, value in defaults.items():
        if not str(os.environ.get(key, "")).strip():
            os.environ[key] = value
            applied[key] = value

    if applied:
        print("[runtime] Applied Windows thread limits: " + ", ".join(f"{key}={value}" for key, value in applied.items()))


def run_logged_command(cmd: list[str], env: dict[str, str] | None = None, log_file: str | Path | None = None) -> None:
    if not log_file:
        subprocess.check_call(cmd, env=env)
        return

    log_path = Path(log_file).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", errors="ignore") as handle:
        handle.write(" ".join(cmd) + "\n")
        handle.flush()
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            stdout_encoding = sys.stdout.encoding or "utf-8"
            safe_line = line.encode(stdout_encoding, errors="replace").decode(stdout_encoding, errors="replace")
            sys.stdout.write(safe_line)
            handle.write(line)
            handle.flush()
        raise SystemExit(process.wait())


def resolve_model_root(results: Path, dataset_name: str, trainer: str, plans: str, network: str) -> Path:
    flat_root = results / f"{trainer}__{plans}__{network}"
    if flat_root.exists():
        return flat_root
    return results / dataset_name / f"{trainer}__{plans}__{network}"
