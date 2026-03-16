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
    exp_dir = case.experiment_output_dir

    try:
        config_path = find_training_config(exp_dir)
        config_args = ["--config", str(config_path)]
    except FileNotFoundError:
        config_args = []
        for cfg in case.train_configs:
            config_args.extend(["--config", cfg])

    ckpt_path = find_checkpoint_by_step(exp_dir, step=step)

    actual_step = parse_step_from_ckpt_name(ckpt_path.name)
    if actual_step is None:
        raise RuntimeError(f"Unable to parse step from checkpoint name: {ckpt_path.name}")

    # 验证结果写入 实验根目录/validate/stepXXXXX，便于与训练结果、figures 同属一实验
    eval_name = f"{exp_dir}/validate/step{actual_step:05d}"

    cmd = [
        python_executable(), "main.py", "validate",
        *config_args,
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


# Phase A 约定：需要验证的 6 个 step。须与 checkpoint 文件名中的 step 一致。
# 训练时 save_iterations=(101,501,...) 触发保存时，文件名用的是 trainer.global_step（未+1），故为 100,500,...
PHASE_A_VALIDATE_STEPS = (100, 500, 1000, 3000, 5000, 10000)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run validate by case name and checkpoint step.")
    parser.add_argument("--case", required=True, help="Case name defined in experiments/cases.py")
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Checkpoint step to validate, e.g. 101 / 501 / 10000. If omitted, latest checkpoint is used.",
    )
    parser.add_argument(
        "--all_phase_a_steps",
        action="store_true",
        help="Run validate for all Phase A steps (100, 500, 1000, 3000, 5000, 10000). Ignores --step.",
    )
    args = parser.parse_args()

    if args.all_phase_a_steps:
        for step in PHASE_A_VALIDATE_STEPS:
            cmd = build_validate_command(case_name=args.case, step=step)
            run_command(cmd)
    else:
        cmd = build_validate_command(case_name=args.case, step=args.step)
        run_command(cmd)


if __name__ == "__main__":
    main()