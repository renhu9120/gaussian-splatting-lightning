
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
    # 实验输出根目录名：方法_数据集，如 baseline_lego / phase_a_lego。不设则用 train_exp_name（向后兼容）
    output_dir: str | None = None

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def experiment_output_dir(self) -> str:
        """用于训练/验证输出路径的实验根目录名（outputs 下的第一层）。"""
        return self.output_dir if self.output_dir is not None else self.train_exp_name


CASES: dict[str, ExperimentCase] = {
    "chair_baseline": ExperimentCase(
        case_name="chair_baseline",
        dataset_name="chair",
        data_path=DATA_ROOT / "chair",
        train_exp_name="chair_gsplat_baseline",
        output_dir="baseline_chair",
        max_steps=10000,
        save_iterations=(101, 501, 1001, 3001, 5001),
    ),
    "lego_baseline": ExperimentCase(
        case_name="lego_baseline",
        dataset_name="lego",
        data_path=DATA_ROOT / "lego",
        train_exp_name="lego_gsplat_baseline_script",
        output_dir="baseline_lego",
        max_steps=10000,
        save_iterations=(101, 501, 1001, 3001, 5001),
    ),
    "ship_baseline": ExperimentCase(
        case_name="ship_baseline",
        dataset_name="ship",
        data_path=DATA_ROOT / "ship",
        train_exp_name="ship_gsplat_baseline_script",
        output_dir="baseline_ship",
        max_steps=10000,
        save_iterations=(101, 501, 1001, 3001, 5001),
    ),
    "drums_baseline": ExperimentCase(
        case_name="drums_baseline",
        dataset_name="drums",
        data_path=DATA_ROOT / "drums",
        train_exp_name="drums_gsplat_baseline_script",
        output_dir="baseline_drums",
        max_steps=10000,
        save_iterations=(101, 501, 1001, 3001, 5001),
    ),
    # ----- Phase A: Quaternion Interpretation -----
    "lego_quaternion_phase_a": ExperimentCase(
        case_name="lego_quaternion_phase_a",
        dataset_name="lego",
        data_path=DATA_ROOT / "lego",
        train_exp_name="lego_quaternion_phase_a",
        output_dir="phase_a_lego",
        train_configs=("configs/blender.yaml", "configs/gsplat_v1.yaml", "configs/quaternion_phase_a.yaml"),
        max_steps=10000,
        save_iterations=(101, 501, 1001, 3001, 5001),
    ),
    "chair_quaternion_phase_a": ExperimentCase(
        case_name="chair_quaternion_phase_a",
        dataset_name="chair",
        data_path=DATA_ROOT / "chair",
        train_exp_name="chair_quaternion_phase_a",
        output_dir="phase_a_chair",
        train_configs=("configs/blender.yaml", "configs/gsplat_v1.yaml", "configs/quaternion_phase_a.yaml"),
        max_steps=10000,
        save_iterations=(101, 501, 1001, 3001, 5001),
    ),
    # ----- Phase B-1: Pure Quaternion + Quaternion Rotor (q_tilde = u q u_bar) -----
    "lego_quaternion_phase_b1": ExperimentCase(
        case_name="lego_quaternion_phase_b1",
        dataset_name="lego",
        data_path=DATA_ROOT / "lego",
        train_exp_name="lego_quaternion_phase_b1",
        output_dir="phase_b1_lego",
        train_configs=("configs/blender.yaml", "configs/gsplat_v1.yaml", "configs/quaternion_phase_b1.yaml"),
        max_steps=10000,
        save_iterations=(101, 501, 1001, 3001, 5001),
    ),
    # ----- Phase B-2: Per-Gaussian Quaternion Rotor (q_tilde_i = u_i q_i u_i_bar) -----
    "lego_quaternion_phase_b2": ExperimentCase(
        case_name="lego_quaternion_phase_b2",
        dataset_name="lego",
        data_path=DATA_ROOT / "lego",
        train_exp_name="lego_quaternion_phase_b2",
        output_dir="phase_b2_lego",
        train_configs=("configs/blender.yaml", "configs/gsplat_v1.yaml", "configs/quaternion_phase_b2.yaml"),
        max_steps=10000,
        save_iterations=(101, 501, 1001, 3001, 5001),
    ),
    # ----- Phase B-3: View-dependent global rotor (q_tilde_i = u(d) q_i u(d)_bar) -----
    "lego_quaternion_phase_b3": ExperimentCase(
        case_name="lego_quaternion_phase_b3",
        dataset_name="lego",
        data_path=DATA_ROOT / "lego",
        train_exp_name="lego_quaternion_phase_b3",
        output_dir="phase_b3_lego",
        train_configs=("configs/blender.yaml", "configs/gsplat_v1.yaml", "configs/quaternion_phase_b3.yaml"),
        max_steps=10000,
        save_iterations=(101, 501, 1001, 3001, 5001),
    ),
    # ----- Phase C-1: Minimal Quaternion Latent Decoder (quat latent in main appearance) -----
    "lego_quaternion_phase_c1": ExperimentCase(
        case_name="lego_quaternion_phase_c1",
        dataset_name="lego",
        data_path=DATA_ROOT / "lego",
        train_exp_name="lego_quaternion_phase_c1",
        output_dir="phase_c1_lego",
        train_configs=("configs/blender.yaml", "configs/gsplat_v1.yaml", "configs/quaternion_phase_c1.yaml"),
        max_steps=10000,
        save_iterations=(101, 501, 1001, 3001, 5001),
    ),
}


def get_case(case_name: str) -> ExperimentCase:
    if case_name not in CASES:
        raise KeyError(f"Unknown case: {case_name}. Available cases: {list(CASES.keys())}")
    return CASES[case_name]