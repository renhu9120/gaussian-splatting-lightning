from __future__ import annotations

import argparse

from experiments.cases import get_case
from experiments.common import (
    find_checkpoint_by_step,
    find_training_config,
    parse_step_from_ckpt_name,
    python_executable,
    run_command,
)


def build_validate_command(case_name: str, step: int | None = None) -> list[str]:
    case = get_case(case_name)

    config_path = find_training_config(case.train_exp_name)
    ckpt_path = find_checkpoint_by_step(case.train_exp_name, step=step)

    actual_step = parse_step_from_ckpt_name(ckpt_path.name)
    if actual_step is None:
        raise RuntimeError(f"Unable to parse step from checkpoint name: {ckpt_path.name}")

    eval_name = f"{case.dataset_name}_step{actual_step:05d}_eval"

    cmd = [
        python_executable(), "main.py", "validate",
        "--config", str(config_path),
        "--data.path", str(case.data_path),
        "--ckpt_path", str(ckpt_path),
        "--model.save_val_metrics", "true",
        "--model.save_val_output", "true",
        "--model.max_save_val_output", "100",
        "--save_val",
        "--output", "outputs",
        "--name", eval_name,
    ]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run validate by case name and checkpoint step.")
    parser.add_argument("--case", required=True, help="Case name defined in experiments/cases.py")
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Checkpoint step to validate, e.g. 999 / 2999 / 4999 / 10000. "
             "If omitted, latest checkpoint is used.",
    )
    args = parser.parse_args()

    cmd = build_validate_command(case_name=args.case, step=args.step)
    run_command(cmd)


if __name__ == "__main__":
    main()