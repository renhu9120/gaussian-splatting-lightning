from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


def run_command(cmd: list[str], cwd: Path | None = None) -> None:
    """Run a command and stream output in real time."""
    if cwd is None:
        cwd = PROJECT_ROOT

    print("=" * 100)
    print("Running command:")
    print(" ".join(f'"{c}"' if " " in c else c for c in cmd))
    print(f"Working directory: {cwd}")
    print("=" * 100)

    subprocess.run(cmd, cwd=str(cwd), check=True)


def format_save_iterations(steps: Iterable[int]) -> str:
    """Format save iterations in the style already verified to work in your project."""
    return "[" + ",".join(str(i) for i in steps) + "]"


def parse_step_from_ckpt_name(filename: str) -> int | None:
    """
    Parse step number from checkpoint filename like:
    epoch=49-step=4999.ckpt
    """
    match = re.search(r"step=(\d+)\.ckpt$", filename)
    if match is None:
        return None
    return int(match.group(1))


def find_checkpoint_by_step(exp_name: str, step: int | None = None) -> Path:
    """
    Find checkpoint path by experiment name and step.
    If step is None, return the latest checkpoint by largest step.
    """
    ckpt_dir = OUTPUT_ROOT / exp_name / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {ckpt_dir}")

    ckpt_files = sorted(ckpt_dir.glob("*.ckpt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found in: {ckpt_dir}")

    step_to_ckpt: dict[int, Path] = {}
    for ckpt in ckpt_files:
        parsed_step = parse_step_from_ckpt_name(ckpt.name)
        if parsed_step is not None:
            step_to_ckpt[parsed_step] = ckpt

    if not step_to_ckpt:
        raise FileNotFoundError(f"No step-formatted checkpoints found in: {ckpt_dir}")

    if step is None:
        latest_step = max(step_to_ckpt)
        return step_to_ckpt[latest_step]

    if step in step_to_ckpt:
        return step_to_ckpt[step]

    available_steps = sorted(step_to_ckpt)
    raise FileNotFoundError(
        f"Checkpoint for step={step} not found under {ckpt_dir}.\n"
        f"Available steps: {available_steps}"
    )


def find_training_config(exp_name: str) -> Path:
    """
    Use the training-time saved config.yaml, which is the safest config source for validate.
    """
    config_path = OUTPUT_ROOT / exp_name / "lightning_logs" / "version_0" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found: {config_path}")
    return config_path


def python_executable() -> str:
    """Return the current Python executable."""
    return sys.executable