from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import nnunetv2


def load_site_module(relative_path: str, alias: str):
    module_name = f"_nnunetv2_site_proxy_{alias}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    for package_root in list(nnunetv2.__path__)[1:]:
        candidate = Path(package_root, *relative_path.split("/")).resolve()
        if not candidate.is_file():
            continue
        spec = importlib.util.spec_from_file_location(module_name, candidate)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    raise ImportError(f"Could not locate nnU-Net site module: {relative_path}")


def export_public(relative_path: str, alias: str, namespace: dict) -> None:
    module = load_site_module(relative_path, alias)
    names = getattr(module, "__all__", None)
    if names is None:
        names = [name for name in dir(module) if not name.startswith("_")]
    for name in names:
        namespace[name] = getattr(module, name)
    namespace["__all__"] = list(names)
