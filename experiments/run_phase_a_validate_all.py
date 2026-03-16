"""
Run validate for all Phase A checkpoints (101, 501, 1001, 3001, 5001, 10000).
Use after training with --case lego_quaternion_phase_a (or other *_quaternion_phase_a case).

Usage (from project root):
    python experiments/run_phase_a_validate_all.py --case lego_quaternion_phase_a
"""
from __future__ import annotations

import argparse

from experiments.cases import get_case
from experiments.common import python_executable, run_command
from experiments.run_validate import build_validate_command

PHASE_A_VALIDATE_STEPS = (101, 501, 1001, 3001, 5001, 10000)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate all Phase A checkpoints for a given case."
    )
    parser.add_argument(
        "--case",
        required=True,
        help="Case name, e.g. lego_quaternion_phase_a (must have been trained first).",
    )
    args = parser.parse_args()

    get_case(args.case)  # validate case exists

    for step in PHASE_A_VALIDATE_STEPS:
        cmd = build_validate_command(case_name=args.case, step=step)
        run_command(cmd)


if __name__ == "__main__":
    main()
