from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


# =============================================================================
# USER CONFIG：实验输出目录结构为 outputs/<EXPERIMENT_DIR>/validate/stepXXXXX/
# =============================================================================
EXPERIMENT_DIR = "phase_c1_lego"   # 方法_数据集，如 baseline_lego / phase_a_lego
IMAGE_NAME = "r_0.jpg"             # 验证输出文件名（无扩展名 + .jpg），如 r_0.jpg / r_1.jpg
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
VALIDATE_ROOT = OUTPUT_ROOT / EXPERIMENT_DIR / "validate"
SAVE_DIR = OUTPUT_ROOT / EXPERIMENT_DIR / "validate_report"
# =============================================================================


def extract_step_from_step_dirname(dirname: str) -> Optional[int]:
    """
    Parse step from dir names under validate/: step00101, step10000, etc.
    """
    m = re.match(r"^step(\d+)$", dirname)
    if m is None:
        return None
    return int(m.group(1))


def find_eval_dirs(validate_root: Path) -> List[Tuple[int, Path]]:
    """Find step dirs (step00101, step00501, ...) under validate_root."""
    if not validate_root.exists():
        return []
    results: List[Tuple[int, Path]] = []
    for p in validate_root.iterdir():
        if not p.is_dir():
            continue
        step = extract_step_from_step_dirname(p.name)
        if step is not None:
            results.append((step, p))
    results.sort(key=lambda x: x[0])
    return results


def find_metrics_csv(eval_dir: Path) -> Path:
    metrics_dir = eval_dir / "metrics"
    if not metrics_dir.exists():
        raise FileNotFoundError(f"metrics dir not found: {metrics_dir}")

    csv_files = sorted(metrics_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No csv file found in: {metrics_dir}")

    return csv_files[0]


def summarize_metrics(csv_path: Path) -> Dict[str, float]:
    df = pd.read_csv(csv_path)
    numeric_df = df.select_dtypes(include=["number"])

    if numeric_df.empty:
        raise ValueError(f"No numeric columns found in metrics csv: {csv_path}")

    summary = numeric_df.mean().to_dict()
    return {str(k): float(v) for k, v in summary.items()}


def collect_metric_table(eval_dirs: List[Tuple[int, Path]]) -> pd.DataFrame:
    rows = []
    for step, eval_dir in eval_dirs:
        csv_path = find_metrics_csv(eval_dir)
        metric_summary = summarize_metrics(csv_path)
        row = {"step": step, "eval_dir": str(eval_dir), "csv_path": str(csv_path)}
        row.update(metric_summary)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
    return df


def list_image_files(root: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]


def find_image_by_basename(eval_dir: Path, image_name: str) -> Path:
    candidates = [p for p in list_image_files(eval_dir) if p.name == image_name]
    if not candidates:
        raise FileNotFoundError(f"Image '{image_name}' not found under: {eval_dir}")
    if len(candidates) > 1:
        print(f"[WARN] Multiple matches for {image_name} under {eval_dir}, use first one.")
    return candidates[0]


def split_gt_and_pred(image_path: Path) -> Tuple[Image.Image, Image.Image]:
    """
    Current validate output is subplot(1,2):
        left  = GT
        right = predicted render
    This function splits them in half.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    if w < 2:
        raise ValueError(f"Image width too small to split: {image_path}")

    mid = w // 2
    gt = img.crop((0, 0, mid, h))
    pred = img.crop((mid, 0, w, h))
    return gt, pred


def metric_value(row: pd.Series, candidates: List[str]) -> Optional[float]:
    for c in candidates:
        if c in row.index and pd.notna(row[c]):
            return float(row[c])
    return None


def make_progress_figure(
    df: pd.DataFrame,
    eval_dirs: List[Tuple[int, Path]],
    image_name: str,
    save_path: Path,
) -> None:
    """
    Produce:
        GT | step999 | step2999 | step4999 | step10000
    """
    step_to_dir = {step: d for step, d in eval_dirs}

    # Use the first step image to obtain GT
    first_step = int(df.iloc[0]["step"])
    first_eval_dir = step_to_dir[first_step]
    first_img_path = find_image_by_basename(first_eval_dir, image_name)
    gt_img, _ = split_gt_and_pred(first_img_path)

    ncols = 1 + len(df)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4.5))

    # Column 0: GT
    axes[0].imshow(gt_img)
    axes[0].axis("off")
    axes[0].set_title("GT", fontsize=11)

    # Remaining columns: prediction at each step
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        step = int(row["step"])
        eval_dir = step_to_dir[step]
        img_path = find_image_by_basename(eval_dir, image_name)
        _, pred_img = split_gt_and_pred(img_path)

        axes[i].imshow(pred_img)
        axes[i].axis("off")

        psnr = metric_value(row, ["val/psnr", "psnr"])
        ssim = metric_value(row, ["val/ssim", "ssim"])
        lpips = metric_value(row, ["val/lpips", "lpips"])

        title_lines = [f"step={step}"]
        if psnr is not None:
            title_lines.append(f"PSNR={psnr:.3f}")
        if ssim is not None:
            title_lines.append(f"SSIM={ssim:.4f}")
        if lpips is not None:
            title_lines.append(f"LPIPS={lpips:.4f}")

        axes[i].set_title("\n".join(title_lines), fontsize=10)

    fig.suptitle(f"{EXPERIMENT_DIR}: iterative progression on {image_name}", fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_metric_curves(df: pd.DataFrame, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    step = df["step"].tolist()

    metrics_to_plot = [
        ("PSNR", ["val/psnr", "psnr"]),
        ("SSIM", ["val/ssim", "ssim"]),
        ("LPIPS", ["val/lpips", "lpips"]),
    ]

    for ax, (title, candidates) in zip(axes, metrics_to_plot):
        selected_col = None
        for c in candidates:
            if c in df.columns:
                selected_col = c
                break

        if selected_col is None:
            ax.set_title(f"{title} (not found)")
            ax.axis("off")
            continue

        values = df[selected_col].tolist()
        ax.plot(step, values, marker="o")
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    eval_dirs = find_eval_dirs(VALIDATE_ROOT)
    if not eval_dirs:
        raise FileNotFoundError(
            f"No step dirs (step00101, ...) found under '{VALIDATE_ROOT}'."
        )

    df = collect_metric_table(eval_dirs)

    # Save metric summary
    metrics_csv_path = SAVE_DIR / f"{EXPERIMENT_DIR}_metrics_summary.csv"
    df.to_csv(metrics_csv_path, index=False, encoding="utf-8-sig")

    print("=" * 100)
    print(f"[INFO] Experiment: {EXPERIMENT_DIR}")
    print(f"[INFO] Metrics summary saved to: {metrics_csv_path}")
    print("=" * 100)
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(df)

    # Metric curves
    curve_path = SAVE_DIR / f"{EXPERIMENT_DIR}_metric_curves.png"
    plot_metric_curves(df, curve_path)
    print(f"[INFO] Metric curves saved to: {curve_path}")

    # Progress figure
    progress_path = SAVE_DIR / f"{EXPERIMENT_DIR}_iterative_progress_{Path(IMAGE_NAME).stem}.png"
    make_progress_figure(df, eval_dirs, IMAGE_NAME, progress_path)
    print(f"[INFO] Progress figure saved to: {progress_path}")


if __name__ == "__main__":
    main()