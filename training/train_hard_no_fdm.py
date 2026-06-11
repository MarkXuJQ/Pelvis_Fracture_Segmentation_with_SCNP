from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from training.run.launcher import configure_top_level_environment, run_named_train


def main() -> None:
    configure_top_level_environment(__file__)
    run_named_train("hard_no_fdm")


if __name__ == "__main__":
    main()
