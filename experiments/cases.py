
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from experiments.common import PROJECT_ROOT


DATA_ROOT = Path(r"D:\Programming\Python\3DGS\datasets\nerf_synthetic")


@dataclass
class ExperimentCase:
    case_name: str
    dataset_name: str
    data_path: Path
    train_exp_name: str
    train_configs: tuple[str, ...] = ("configs/blender.yaml", "configs/gsplat_v1.yaml")
    max_steps: int = 10000
    save_iterations: tuple[int, ...] = (1000, 3000, 5000)

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT


CASES: dict[str, ExperimentCase] = {
    "chair_baseline": ExperimentCase(
        case_name="chair_baseline",
        dataset_name="chair",
        data_path=DATA_ROOT / "chair",
        train_exp_name="chair_gsplat_baseline",
        max_steps=10000,
        save_iterations=(101, 501,1001,3001,5001),
    ),
    "lego_baseline": ExperimentCase(
        case_name="lego_baseline",
        dataset_name="lego",
        data_path=DATA_ROOT / "lego",
        train_exp_name="lego_gsplat_baseline_script",
        max_steps=10000,
        save_iterations=(101, 501,1001,3001,5001),
    ),
    "ship_baseline": ExperimentCase(
        case_name="ship_baseline",
        dataset_name="ship",
        data_path=DATA_ROOT / "ship",
        train_exp_name="ship_gsplat_baseline_script",
        max_steps=10000,
        save_iterations=(101, 501,1001,3001,5001),
    ),
    "drums_baseline": ExperimentCase(
        case_name="drums_baseline",
        dataset_name="drums",
        data_path=DATA_ROOT / "drums",
        train_exp_name="drums_gsplat_baseline_script",
        max_steps=10000,
        save_iterations=(101, 501,1001,3001,5001),
    ),
}


def get_case(case_name: str) -> ExperimentCase:
    if case_name not in CASES:
        raise KeyError(f"Unknown case: {case_name}. Available cases: {list(CASES.keys())}")
    return CASES[case_name]