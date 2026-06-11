from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from training.runtime.fracsegnet_stage1_helpers import (
    _copy_model,
    _ensure_dir,
    _patch_generic_unet_lambda,
    _patch_nnunet_nd_softmax_lambda,
    _patch_trainer_lambda_identity,
    _python_cuda_available,
    _run,
    _write_plans_pkl_from_model_pkl,
)
from training.runtime.project_paths import get_project_paths


def _iter_input_cases(input_dir: Path) -> list[str]:
    cases = []
    for path in sorted(input_dir.glob("*_0000.nii.gz")):
        cases.append(path.name[: -len("_0000.nii.gz")])
    return cases


def _prediction_complete(output_dir: Path, case_stems: list[str]) -> bool:
    if not output_dir.is_dir() or not case_stems:
        return False
    return all((output_dir / f"{case}.nii.gz").is_file() for case in case_stems)


def _stage_model_paths(model_root: Path) -> dict[str, Path]:
    return {
        "lowres_model": model_root / "lowres_model" / "Ana_lowres.model",
        "lowres_pkl": model_root / "lowres_model" / "Ana_lowres.model.pkl",
        "cascade_model": model_root / "cascadeFullres_model" / "Ana_cascade_fullres.model",
        "cascade_pkl": model_root / "cascadeFullres_model" / "Ana_cascadeFullres.model.pkl",
    }


def _prepare_fracsegnet_model_layout(args: argparse.Namespace, results_folder: Path) -> tuple[Path, Path]:
    model_paths = _stage_model_paths(args.fracsegnet_anatomical_model_dir.resolve())
    for name, path in model_paths.items():
        if not path.is_file():
            raise RuntimeError(f"Missing FracSegNet anatomical model file {name}: {path}")

    low_model_dir = (
        results_folder
        / "nnUNet"
        / "3d_lowres"
        / args.ana_task_name
        / f"nnUNetTrainerV2__{args.ana_plans}"
        / "all"
    )
    cas_model_dir = (
        results_folder
        / "nnUNet"
        / "3d_cascade_fullres"
        / args.ana_task_name
        / f"nnUNetTrainerV2CascadeFullRes__{args.ana_plans}"
        / "all"
    )

    _copy_model(model_paths["lowres_model"], model_paths["lowres_pkl"], low_model_dir)
    _copy_model(model_paths["cascade_model"], model_paths["cascade_pkl"], cas_model_dir)
    _write_plans_pkl_from_model_pkl(model_paths["lowres_pkl"], low_model_dir.parent)
    _write_plans_pkl_from_model_pkl(model_paths["cascade_pkl"], cas_model_dir.parent)
    return low_model_dir, cas_model_dir


def run_stage1_anatomy(args: argparse.Namespace) -> Path:
    paths = get_project_paths(args.data_root)
    input_dir = args.input_dir.resolve()
    lowres_out = args.lowres_out.resolve()
    cascade_out = args.cascade_out.resolve()
    logs_dir = _ensure_dir(args.logs_dir.resolve())
    dataset_root = args.dataset_root.resolve()
    results_folder = dataset_root / "nnUNet_results"
    preprocessed_root = dataset_root / "nnUNet_preprocessed"

    if not input_dir.is_dir():
        raise RuntimeError(f"Full-CT input directory not found: {input_dir}")
    case_stems = _iter_input_cases(input_dir)
    if not case_stems:
        raise RuntimeError(f"No *_0000.nii.gz files found in {input_dir}")

    if args.reset:
        shutil.rmtree(lowres_out, ignore_errors=True)
        shutil.rmtree(cascade_out, ignore_errors=True)
    _ensure_dir(lowres_out)
    _ensure_dir(cascade_out)
    _ensure_dir(results_folder)
    _ensure_dir(preprocessed_root)

    if _prediction_complete(cascade_out, case_stems) and not args.reset:
        print(f"[stage1] Reusing complete anatomy predictions in {cascade_out}")
        return cascade_out

    if not _python_cuda_available(args.nnunet_py):
        raise RuntimeError(f"nnUNet v1 env has no CUDA: {args.nnunet_py}")

    _prepare_fracsegnet_model_layout(args, results_folder)
    _patch_nnunet_nd_softmax_lambda(args.nnunet_py)
    _patch_trainer_lambda_identity(args.nnunet_py)
    _patch_generic_unet_lambda(args.nnunet_py)

    env = os.environ.copy()
    env["nnUNet_raw_data_base"] = str(dataset_root / "nnUNet_raw_data")
    env["nnUNet_preprocessed"] = str(preprocessed_root)
    env["RESULTS_FOLDER"] = str(results_folder)
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

    low_cmd = [
        args.nnunet_py,
        "-m",
        "nnunet.inference.predict_simple",
        "-i",
        str(input_dir),
        "-o",
        str(lowres_out),
        "-t",
        str(int(args.ana_task_id)),
        "-m",
        "3d_lowres",
        "-f",
        "all",
        "-tr",
        "nnUNetTrainerV2",
        "-p",
        args.ana_plans,
        "-chk",
        "model_final_checkpoint",
        "--num_threads_preprocessing",
        str(int(args.num_threads_preprocessing)),
        "--num_threads_nifti_save",
        str(int(args.num_threads_nifti_save)),
    ]
    if args.disable_tta:
        low_cmd.append("--disable_tta")
    _run(low_cmd, env=env, quiet=bool(args.quiet), log_file=logs_dir / "stage1_lowres.log")

    lowres_next_stage = preprocessed_root / args.ana_task_name / "pred_next_stage"
    _ensure_dir(lowres_next_stage)
    for path in sorted(lowres_out.glob("*.nii.gz")):
        shutil.copy2(path, lowres_next_stage / path.name)

    cas_cmd = [
        args.nnunet_py,
        "-m",
        "nnunet.inference.predict_simple",
        "-i",
        str(input_dir),
        "-o",
        str(cascade_out),
        "-t",
        str(int(args.ana_task_id)),
        "-m",
        "3d_cascade_fullres",
        "-f",
        "all",
        "-tr",
        "nnUNetTrainerV2",
        "-ctr",
        "nnUNetTrainerV2CascadeFullRes",
        "-p",
        args.ana_plans,
        "-chk",
        "model_final_checkpoint",
        "-l",
        str(lowres_out),
        "--num_threads_preprocessing",
        str(int(args.num_threads_preprocessing)),
        "--num_threads_nifti_save",
        str(int(args.num_threads_nifti_save)),
    ]
    if args.disable_tta:
        cas_cmd.append("--disable_tta")
    _run(cas_cmd, env=env, quiet=bool(args.quiet), log_file=logs_dir / "stage1_cascade.log")

    if not _prediction_complete(cascade_out, case_stems):
        missing = [case for case in case_stems if not (cascade_out / f"{case}.nii.gz").is_file()]
        raise RuntimeError(f"Stage-1 anatomy inference did not produce all cases. Missing first cases: {missing[:10]}")

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(input_dir),
        "lowres_out": str(lowres_out),
        "cascade_out": str(cascade_out),
        "case_count": len(case_stems),
        "fracsegnet_anatomical_model_dir": str(args.fracsegnet_anatomical_model_dir.resolve()),
        "ana_task_id": int(args.ana_task_id),
        "ana_task_name": args.ana_task_name,
        "ana_plans": args.ana_plans,
        "disable_tta": bool(args.disable_tta),
        "label_semantics": {
            "1": "sacrum",
            "2": "left_hip",
            "3": "right_hip",
            "4": "additional anatomy class from the official model; not used for stage-2 bone extraction",
        },
    }
    (cascade_out.parent / "stage1_anatomy_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[stage1] wrote anatomy predictions: {cascade_out}")
    return cascade_out


def main() -> None:
    defaults = get_project_paths()
    parser = argparse.ArgumentParser(
        description="Run the official FracSegNet first-stage anatomical model on complete Dataset501 CTs."
    )
    parser.add_argument("--data_root", type=Path, default=defaults.data_root)
    parser.add_argument("--dataset_root", type=Path, default=defaults.dataset_root)
    parser.add_argument("--input_dir", type=Path, default=defaults.full_ct_images)
    parser.add_argument(
        "--lowres_out",
        type=Path,
        default=defaults.dataset_root / "stage1_anatomy_predictions" / "anatomy_lowres",
    )
    parser.add_argument("--cascade_out", type=Path, default=defaults.stage1_anatomy_label_dir)
    parser.add_argument(
        "--logs_dir",
        type=Path,
        default=defaults.dataset_root / "stage1_anatomy_predictions" / "logs",
    )
    parser.add_argument("--fracsegnet_anatomical_model_dir", type=Path, default=defaults.fracsegnet_anatomical_model_dir)
    parser.add_argument("--nnunet_py", type=str, default=os.environ.get("NNUNET_V1_PY", sys.executable))
    parser.add_argument("--ana_task_id", type=int, default=600)
    parser.add_argument("--ana_task_name", type=str, default="Task600_ContinueTrainCtPelvicAnatomical120")
    parser.add_argument("--ana_plans", type=str, default="nnUNetPlans_pretrained_ContinueTrainPelvicSeg")
    parser.add_argument("--disable_tta", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_threads_preprocessing", type=int, default=1)
    parser.add_argument("--num_threads_nifti_save", type=int, default=1)
    parser.add_argument("--quiet", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    run_stage1_anatomy(args)


if __name__ == "__main__":
    main()
