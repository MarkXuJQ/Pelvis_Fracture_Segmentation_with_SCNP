from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from training.run.launcher import configure_top_level_environment, run_named_predict


if __name__ == "__main__":
    # Compatibility shim for the canonical multi_rf3_thr03_rf5_thr05 entrypoint.
    configure_top_level_environment(__file__)
    run_named_predict("multi_rf3_thr03_rf5_thr05")
