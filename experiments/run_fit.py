from __future__ import annotations

import argparse

from experiments.cases import get_case
from experiments.common import format_save_iterations, python_executable, run_command


def build_fit_command(case_name: str, smoke: bool = False) -> list[str]:
    case = get_case(case_name)

    exp_name = case.train_exp_name
    max_steps = case.max_steps
    save_iterations = case.save_iterations

    if smoke:
        exp_name = f"{case.train_exp_name}_smoke"
        max_steps = 10
        save_iterations = ()

    cmd = [python_executable(), "main.py", "fit"]

    for cfg in case.train_configs:
        cmd.extend(["--config", cfg])

    cmd.extend([
        "--data.path", str(case.data_path),
        "-n", exp_name,
        "--max_steps", str(max_steps),
    ])

    if save_iterations:
        cmd.extend(["--save_iterations", format_save_iterations(save_iterations)])

    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fit experiment by case name.")
    parser.add_argument("--case", required=True, help="Case name defined in experiments/cases.py")
    parser.add_argument("--smoke", action="store_true", help="Run a 10-step smoke test")
    args = parser.parse_args()

    cmd = build_fit_command(case_name=args.case, smoke=args.smoke)
    run_command(cmd)


if __name__ == "__main__":
    main()