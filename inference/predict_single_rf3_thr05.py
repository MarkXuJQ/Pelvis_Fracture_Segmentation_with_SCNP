from pathlib import Path
import os
import sys

data_root = Path(__file__).resolve().parents[1]
repo_root = data_root
os.environ.setdefault("PELVIS_SCNP_DATA_ROOT", str(data_root))
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from training.run.launcher import run_named_predict


if __name__ == "__main__":
    run_named_predict("single_rf3_thr05")
