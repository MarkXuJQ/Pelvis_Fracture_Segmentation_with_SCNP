from __future__ import annotations

import json
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import Mapping, Sequence


def _ensure_dir(path: Path | str) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _write_log(log_file: Path, output: str) -> None:
    _ensure_dir(log_file.parent)
    log_file.write_text(output, encoding="utf-8")


def _run(
    cmd: Sequence[str | Path],
    env: Mapping[str, str] | None = None,
    quiet: bool = False,
    log_file: Path | None = None,
) -> None:
    command = [str(part) for part in cmd]
    run_kwargs = {
        "check": True,
        "text": True,
    }
    if env is not None:
        run_kwargs["env"] = dict(env)

    capture_output = bool(quiet or log_file is not None)
    if capture_output:
        run_kwargs["stdout"] = subprocess.PIPE
        run_kwargs["stderr"] = subprocess.STDOUT

    try:
        completed = subprocess.run(command, **run_kwargs)
    except subprocess.CalledProcessError as exc:
        output = exc.stdout or ""
        if log_file is not None:
            _write_log(log_file, output)
        if output and not quiet:
            print(output, end="" if output.endswith("\n") else "\n")
        raise

    if not capture_output:
        return

    output = completed.stdout or ""
    if log_file is not None:
        _write_log(log_file, output)
    if output and not quiet:
        print(output, end="" if output.endswith("\n") else "\n")


def _python_cuda_available(python_exe: str) -> bool:
    code = (
        "ok = False\n"
        "try:\n"
        "    import torch\n"
        "    ok = bool(torch.cuda.is_available())\n"
        "except Exception:\n"
        "    ok = False\n"
        "print('1' if ok else '0')\n"
    )
    try:
        result = subprocess.run(
            [python_exe, "-c", code],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except Exception:
        return False
    return result.stdout.strip().endswith("1")


def _copy_model(model_src: Path, model_pkl_src: Path, model_dir: Path) -> None:
    model_src = Path(model_src).resolve()
    model_pkl_src = Path(model_pkl_src).resolve()
    model_dir = _ensure_dir(model_dir)

    if not model_src.is_file():
        raise RuntimeError(f"Missing model file: {model_src}")
    if not model_pkl_src.is_file():
        raise RuntimeError(f"Missing model pkl file: {model_pkl_src}")

    shutil.copy2(model_src, model_dir / "model_final_checkpoint.model")
    shutil.copy2(model_pkl_src, model_dir / "model_final_checkpoint.model.pkl")

    if (model_dir / model_src.name) != model_src:
        shutil.copy2(model_src, model_dir / model_src.name)
    if (model_dir / model_pkl_src.name) != model_pkl_src:
        shutil.copy2(model_pkl_src, model_dir / model_pkl_src.name)


def _write_plans_pkl_from_model_pkl(model_pkl_path: Path, model_root: Path) -> Path:
    model_pkl_path = Path(model_pkl_path).resolve()
    model_root = _ensure_dir(model_root)
    if not model_pkl_path.is_file():
        raise RuntimeError(f"Missing model pkl file: {model_pkl_path}")

    with model_pkl_path.open("rb") as f:
        model_info = pickle.load(f)

    plans = None
    if isinstance(model_info, dict):
        if isinstance(model_info.get("plans"), dict):
            plans = model_info["plans"]
        elif "plans_per_stage" in model_info:
            plans = model_info

    if not isinstance(plans, dict):
        raise RuntimeError(f"Could not extract plans dict from: {model_pkl_path}")

    plans_path = model_root / "plans.pkl"
    with plans_path.open("wb") as f:
        pickle.dump(plans, f)
    return plans_path


def _run_inline_python(python_exe: str, code: str) -> None:
    subprocess.run([python_exe, "-c", code], check=True)


def _patch_identity_lambdas_in_package(python_exe: str, package_name: str, pattern: str) -> None:
    payload = json.dumps({"package_name": package_name, "pattern": pattern})
    code = (
        "import importlib.util\n"
        "import json\n"
        "import re\n"
        "from pathlib import Path\n"
        f"payload = json.loads({payload!r})\n"
        "spec = importlib.util.find_spec(payload['package_name'])\n"
        "if spec is None or not spec.submodule_search_locations:\n"
        "    raise RuntimeError(f\"Could not locate package: {payload['package_name']}\")\n"
        "package_dir = Path(list(spec.submodule_search_locations)[0]).resolve()\n"
        "for path in sorted(package_dir.glob(payload['pattern'])):\n"
        "    text = path.read_text(encoding='utf-8')\n"
        "    updated = text\n"
        "    if 'lambda x: x' in updated:\n"
        "        if 'def _identity_for_pickle(x):' not in updated:\n"
        "            match = re.search(r'^class\\s', updated, flags=re.MULTILINE)\n"
        "            if match is None:\n"
        "                raise RuntimeError(f\"Could not find class anchor in {path}\")\n"
        "            helper = 'def _identity_for_pickle(x):\\n    return x\\n\\n\\n'\n"
        "            updated = updated[:match.start()] + helper + updated[match.start():]\n"
        "        updated = updated.replace('lambda x: x', '_identity_for_pickle')\n"
        "        path.write_text(updated, encoding='utf-8')\n"
        "    print(path)\n"
    )
    _run_inline_python(python_exe, code)


def _patch_nnunet_nd_softmax_lambda(python_exe: str) -> None:
    code = (
        "import importlib.util\n"
        "import re\n"
        "from pathlib import Path\n"
        "spec = importlib.util.find_spec('nnunet.utilities.nd_softmax')\n"
        "if spec is None or spec.origin is None:\n"
        "    raise RuntimeError('Could not locate nnunet.utilities.nd_softmax')\n"
        "path = Path(spec.origin).resolve()\n"
        "text = path.read_text(encoding='utf-8')\n"
        "updated, count = re.subn(\n"
        "    r'^softmax_helper\\s*=\\s*lambda x:\\s*F\\.softmax\\(x,\\s*1\\)\\s*$',\n"
        "    'def softmax_helper(x):\\n    return F.softmax(x, 1)',\n"
        "    text,\n"
        "    count=1,\n"
        "    flags=re.MULTILINE,\n"
        ")\n"
        "if count:\n"
        "    path.write_text(updated, encoding='utf-8')\n"
        "print(path)\n"
    )
    _run_inline_python(python_exe, code)


def _patch_trainer_lambda_identity(python_exe: str) -> None:
    _patch_identity_lambdas_in_package(
        python_exe=python_exe,
        package_name="nnunet.training.network_training",
        pattern="nnUNetTrainer*.py",
    )


def _patch_generic_unet_lambda(python_exe: str) -> None:
    _patch_identity_lambdas_in_package(
        python_exe=python_exe,
        package_name="nnunet.network_architecture",
        pattern="generic_UNet*.py",
    )
