from __future__ import annotations

import argparse

from experiments.cases import get_case
from experiments.common import OUTPUT_ROOT, find_checkpoint_by_step, python_executable, run_command


def build_viewer_command(case_name: str, step: int | None = None) -> list[str]:
    case = get_case(case_name)
    exp_dir = case.experiment_output_dir

    if step is None:
        target = OUTPUT_ROOT / exp_dir
    else:
        target = find_checkpoint_by_step(exp_dir, step=step)

    cmd = [python_executable(), "viewer.py", str(target)]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run viewer by case name and optional checkpoint step.")
    parser.add_argument("--case", required=True, help="Case name defined in experiments/cases.py")
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Checkpoint step to open in viewer. If omitted, open experiment directory.",
    )
    args = parser.parse_args()

    cmd = build_viewer_command(case_name=args.case, step=args.step)
    run_command(cmd)
    print("\nViewer should be available at: http://127.0.0.1:8080\n")


if __name__ == "__main__":
    main()