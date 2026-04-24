from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from training.runtime.project_paths import (
    DISMAP_FILE_SUFFIX,
    dismap_sidecar_path,
    get_project_paths,
)


def build_overlay_pythonpath(
    existing_pythonpath: str | None,
    overlay_root: Path,
    required_relative_paths: list[str],
    repo_root: Path | None = None,
) -> str:
    overlay_root = overlay_root.resolve()
    repo_root = repo_root.resolve() if repo_root is not None else overlay_root.parents[1]

    missing = [str(overlay_root / relative_path) for relative_path in required_relative_paths if not (overlay_root / relative_path).is_file()]
    if missing:
        raise RuntimeError(
            "Missing nnUNet overlay files:\n- " + "\n- ".join(missing)
        )

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


def ensure_preprocessed_dataset_matches_raw(
    raw_base: str | Path,
    preprocessed: str | Path,
    task_id: int,
) -> None:
    dataset_paths = get_project_paths(raw_base)
    dataset_name = resolve_dataset_name(dataset_paths.nnunet_raw_root, task_id)
    raw_dataset_json_path = dataset_paths.nnunet_raw_root / dataset_name / "dataset.json"
    preprocessed_dataset_json_path = Path(preprocessed).resolve() / dataset_name / "dataset.json"

    if not raw_dataset_json_path.is_file():
        raise RuntimeError(f"Missing raw dataset.json: {raw_dataset_json_path}")
    if not preprocessed_dataset_json_path.is_file():
        raise RuntimeError(
            "Missing preprocessed dataset.json. "
            f"Run preprocessing first: {preprocessed_dataset_json_path}"
        )

    raw_dataset_json = json.loads(raw_dataset_json_path.read_text(encoding="utf-8"))
    preprocessed_dataset_json = json.loads(preprocessed_dataset_json_path.read_text(encoding="utf-8"))
    if raw_dataset_json != preprocessed_dataset_json:
        raise RuntimeError(
            "The preprocessed nnU-Net cache does not match the current raw dataset label schema. "
            "Re-run preprocessing with `--preprocess --reset_preprocess` before training."
        )

    normalize_preprocessed_dismap_layout(Path(preprocessed).resolve() / dataset_name)


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


def case_name_from_image_path(image_path: Path) -> str:
    name = image_path.name
    if not name.endswith(".nii.gz"):
        raise RuntimeError(f"Expected a .nii.gz image file, got: {image_path}")
    stem = name[: -len(".nii.gz")]
    if stem.endswith("_0000"):
        stem = stem[: -len("_0000")]
    return stem


def parse_patient_id(case_name: str) -> str:
    parts = case_name.split("_")
    if len(parts) < 3:
        raise RuntimeError(f"Unexpected ROI case name format: {case_name}")
    patient_id = parts[1]
    if not patient_id.isdigit():
        raise RuntimeError(f"Cannot parse numeric patient id from case name: {case_name}")
    return patient_id


def collect_case_names(raw_dataset_dir: Path) -> list[str]:
    images_tr = raw_dataset_dir / "imagesTr"
    if not images_tr.is_dir():
        raise RuntimeError(f"Missing imagesTr directory: {images_tr}")
    case_names = sorted(
        {
            case_name_from_image_path(path)
            for path in images_tr.glob("*.nii.gz")
            if path.name.endswith("_0000.nii.gz")
        }
    )
    if not case_names:
        raise RuntimeError(f"No ROI cases found in: {images_tr}")
    return case_names


def build_patient_level_splits(case_names: list[str], patient_val_count: int, seed: int) -> tuple[list[dict], dict]:
    if patient_val_count <= 0:
        raise RuntimeError(f"patient_val_count must be > 0, got: {patient_val_count}")

    patient_to_cases: dict[str, list[str]] = defaultdict(list)
    for case_name in case_names:
        patient_to_cases[parse_patient_id(case_name)].append(case_name)

    for patient_id in patient_to_cases:
        patient_to_cases[patient_id] = sorted(patient_to_cases[patient_id])

    patient_ids = sorted(patient_to_cases)
    rng = random.Random(int(seed))
    rng.shuffle(patient_ids)

    folds_patient_ids = [
        patient_ids[index : index + patient_val_count]
        for index in range(0, len(patient_ids), patient_val_count)
    ]
    if not folds_patient_ids:
        raise RuntimeError("Failed to build patient-level folds: no patients were found.")

    all_case_names = sorted(case_names)
    splits: list[dict] = []
    fold_summaries: list[dict] = []
    for fold_idx, val_patient_ids in enumerate(folds_patient_ids):
        val_patient_set = set(val_patient_ids)
        val_cases = sorted(
            case_name
            for patient_id in val_patient_ids
            for case_name in patient_to_cases[patient_id]
        )
        train_cases = sorted(
            case_name
            for patient_id in patient_ids
            if patient_id not in val_patient_set
            for case_name in patient_to_cases[patient_id]
        )
        if sorted(train_cases + val_cases) != all_case_names:
            raise RuntimeError(f"Fold {fold_idx} does not cover the full ROI case set.")

        splits.append({"train": train_cases, "val": val_cases})
        fold_summaries.append(
            {
                "fold": fold_idx,
                "train_patients": len(patient_ids) - len(val_patient_ids),
                "val_patients": len(val_patient_ids),
                "train_cases": len(train_cases),
                "val_cases": len(val_cases),
                "val_patient_ids": val_patient_ids,
                "val_case_names": val_cases,
            }
        )

    summary = {
        "split_mode": "patient_level",
        "seed": int(seed),
        "total_patients": len(patient_ids),
        "total_cases": len(all_case_names),
        "patient_val_count": int(patient_val_count),
        "num_folds": len(folds_patient_ids),
        "cases_per_patient": sorted({len(cases) for cases in patient_to_cases.values()}),
        "folds": fold_summaries,
    }
    return splits, summary


def write_patient_split_artifacts(
    preprocessed_dataset_dir: Path,
    splits: list[dict],
    summary: dict,
    selected_fold: int,
    generated_by: Path,
) -> None:
    preprocessed_dataset_dir.mkdir(parents=True, exist_ok=True)

    split_path = preprocessed_dataset_dir / "splits_final.json"
    summary_path = preprocessed_dataset_dir / "patient_level_split_summary.json"
    val_report_path = preprocessed_dataset_dir / f"validation_fold{selected_fold}_patient_level_cases.txt"

    split_path.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if selected_fold < 0 or selected_fold >= len(summary["folds"]):
        raise RuntimeError(f"Requested fold {selected_fold} outside generated range [0, {len(summary['folds']) - 1}]")

    fold_info = summary["folds"][selected_fold]
    lines = [
        f"Patient-level validation split for {preprocessed_dataset_dir.name}",
        f"Generated by: {generated_by.resolve()}",
        "",
        "Summary",
        f"- Fold: {selected_fold}",
        f"- Total patients: {summary['total_patients']}",
        f"- Validation patients: {fold_info['val_patients']}",
        f"- Validation ROI cases: {fold_info['val_cases']}",
        f"- Seed: {summary['seed']}",
        "",
        "Validation patient IDs",
        ", ".join(fold_info["val_patient_ids"]),
        "",
        "Validation ROI cases",
    ]
    lines.extend(fold_info["val_case_names"])
    val_report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[split] wrote patient-level splits: {split_path}")
    print(f"[split] wrote split summary: {summary_path}")
    print(f"[split] wrote fold {selected_fold} validation report: {val_report_path}")
    print(
        "[split] "
        f"patients={summary['total_patients']} | val_patients_per_fold={summary['patient_val_count']} "
        f"| folds={summary['num_folds']} | fold_{selected_fold}_val_cases={fold_info['val_cases']}"
    )


def prepare_patient_level_splits(
    raw_base: str | Path,
    preprocessed: str | Path,
    task_id: int,
    selected_fold: int,
    patient_val_count: int,
    seed: int,
    generated_by: Path,
) -> None:
    v2_raw_root = get_project_paths(raw_base).nnunet_raw_root
    dataset_name = resolve_dataset_name(v2_raw_root, task_id)
    raw_dataset_dir = v2_raw_root / dataset_name
    preprocessed_dataset_dir = Path(preprocessed).resolve() / dataset_name

    case_names = collect_case_names(raw_dataset_dir)
    splits, summary = build_patient_level_splits(case_names, patient_val_count, seed)
    write_patient_split_artifacts(preprocessed_dataset_dir, splits, summary, selected_fold, generated_by=generated_by)


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
        if key not in os.environ or str(os.environ.get(key, "")).strip() == "":
            os.environ[key] = value
            applied[key] = value

    if applied:
        print(
            "[runtime] Applied Windows thread limits: "
            + ", ".join(f"{key}={value}" for key, value in applied.items())
        )


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
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            handle.write(line)
            handle.flush()
        raise SystemExit(process.wait())


def build_ct_only_input_dir(source_dir: Path, target_dir: Path) -> Path:
    source_dir = source_dir.resolve()
    target_dir = target_dir.resolve()

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for ct_file in sorted(source_dir.glob("*_0000.nii.gz")):
        shutil.copy2(ct_file, target_dir / ct_file.name)
        copied += 1

    if copied == 0:
        raise RuntimeError(f"No CT channel files (*_0000.nii.gz) found in {source_dir}")
    return target_dir


def resolve_model_root(results: Path, dataset_name: str, trainer: str, plans: str, network: str) -> Path:
    flat_root = results / f"{trainer}__{plans}__{network}"
    if flat_root.exists():
        return flat_root
    return results / dataset_name / f"{trainer}__{plans}__{network}"


def safe_remove_directory(path: Path) -> None:
    if not path.exists():
        return
    if path.is_symlink():
        path.unlink()
        return
    if os.name == "nt" and path.is_dir():
        subprocess.run(
            ["cmd", "/c", "rmdir", str(path)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if not path.exists():
            return
    shutil.rmtree(path)


def prepare_flat_results_layout(
    results_root: str | Path,
    task_id: int,
    network: str,
    trainer_name: str,
    plans_name: str = "nnUNetPlans",
) -> tuple[Path, Path]:
    v2_raw_root = Path(os.environ["nnUNet_raw"]).resolve()
    dataset_name = resolve_dataset_name(v2_raw_root, task_id)
    results_root_path = Path(results_root).resolve()
    trainer_dir_name = f"{trainer_name}__{plans_name}__{network}"

    flat_model_root = results_root_path / trainer_dir_name
    dataset_dir = results_root_path / dataset_name
    nested_model_root = dataset_dir / trainer_dir_name

    flat_model_root.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if nested_model_root.exists():
        same_target = False
        try:
            same_target = nested_model_root.resolve() == flat_model_root.resolve()
        except OSError:
            same_target = False
        if not same_target:
            safe_remove_directory(nested_model_root)

    if not nested_model_root.exists():
        if os.name == "nt":
            subprocess.check_call(
                [
                    "cmd",
                    "/c",
                    "mklink",
                    "/J",
                    str(nested_model_root),
                    str(flat_model_root),
                ]
            )
        else:
            os.symlink(flat_model_root, nested_model_root, target_is_directory=True)

    print(f"[results] flat_model_root={flat_model_root}")
    print(f"[results] nested_alias={nested_model_root}")
    return flat_model_root, nested_model_root
