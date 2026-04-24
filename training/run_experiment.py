from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.run.experiment_registry import list_experiment_specs
from training.run.launcher import (
    configure_top_level_environment,
    run_named_compare,
    run_named_predict,
    run_named_train,
    run_named_validate,
)


ACTION_RUNNERS = {
    "train": run_named_train,
    "predict": run_named_predict,
    "validate": run_named_validate,
    "compare": run_named_compare,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified entrypoint for Pelvis_SCNP training experiments."
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    list_parser = subparsers.add_parser("list", help="List available experiments.")
    list_parser.add_argument("--for-action", dest="filter_action", choices=tuple(ACTION_RUNNERS), default=None)

    for action in ACTION_RUNNERS:
        action_parser = subparsers.add_parser(action, help=f"Run the {action} flow for one experiment.")
        choices = []
        for spec in list_experiment_specs(action):
            choices.append(spec.key)
            choices.extend(spec.aliases)
        action_parser.add_argument(
            "experiment",
            choices=choices,
            help="Experiment key.",
        )
    return parser


def _print_experiment_list(action: str | None) -> None:
    specs = list_experiment_specs(action)
    for spec in specs:
        supported_actions = [name for name in ACTION_RUNNERS if spec in list_experiment_specs(name)]
        aliases = f" | aliases: {', '.join(spec.aliases)}" if spec.aliases else ""
        print(f"{spec.key:20} {spec.label:24} actions: {', '.join(supported_actions)}{aliases}")


def main() -> None:
    configure_top_level_environment(__file__)

    parser = _build_parser()
    args, extra_args = parser.parse_known_args()

    if args.action == "list":
        _print_experiment_list(args.filter_action)
        return

    original_argv = sys.argv
    sys.argv = [original_argv[0], *extra_args]
    try:
        ACTION_RUNNERS[args.action](args.experiment)
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
