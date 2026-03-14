from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


# =============================================================================
# USER CONFIG: 只改这里，然后直接在 PyCharm 点运行
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_ROOT = PROJECT_ROOT / "outputs"
SAVE_DIR = OUTPUT_ROOT / "drums_progress_report_custom"
DATASET = "drums"          # 例如: "drums" / "ship" / "chair" / "lego"
IMAGE_NAME = "000.png"          # 例如: "000.png"，如果为 None 则自动选一张公共图片
# =============================================================================


def extract_step_from_dirname(dirname: str, dataset_name: str) -> Optional[int]:
    pattern = rf"^{re.escape(dataset_name)}_step(\d+)_eval$"
    m = re.match(pattern, dirname)
    if m is None:
        return None
    return int(m.group(1))


def find_eval_dirs(output_root: Path, dataset_name: str) -> List[Tuple[int, Path]]:
    results: List[Tuple[int, Path]] = []
    if not output_root.exists():
        raise FileNotFoundError(f"Output root does not exist: {output_root}")

    for p in output_root.iterdir():
        if not p.is_dir():
            continue
        step = extract_step_from_dirname(p.name, dataset_name)
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
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files


def find_common_relative_images(eval_dirs: List[Tuple[int, Path]]) -> List[Path]:
    relative_sets = []
    for _, eval_dir in eval_dirs:
        rels = {p.relative_to(eval_dir) for p in list_image_files(eval_dir)}
        relative_sets.append(rels)

    if not relative_sets:
        return []

    common = set.intersection(*relative_sets)
    return sorted(common)


def select_target_image(
    eval_dirs: List[Tuple[int, Path]],
    image_name: Optional[str] = None,
) -> Optional[Path]:
    common_rel_images = find_common_relative_images(eval_dirs)
    if not common_rel_images:
        return None

    if image_name is not None:
        for rel in common_rel_images:
            if rel.name == image_name:
                return rel

    for rel in common_rel_images:
        name_lower = rel.name.lower()
        if "depth" not in name_lower:
            return rel

    return common_rel_images[0]


def open_image_safe(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def metric_value(row: pd.Series, candidates: List[str]) -> Optional[float]:
    for c in candidates:
        if c in row.index and pd.notna(row[c]):
            return float(row[c])
    return None


def build_progress_figure(
    df: pd.DataFrame,
    eval_dirs: List[Tuple[int, Path]],
    target_rel_image: Path,
    save_path: Path,
) -> None:
    step_to_dir = {step: d for step, d in eval_dirs}
    n = len(df)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, df.iterrows()):
        step = int(row["step"])
        eval_dir = step_to_dir[step]
        img_path = eval_dir / target_rel_image

        img = open_image_safe(img_path)
        ax.imshow(img)
        ax.axis("off")

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

        ax.set_title("\n".join(title_lines), fontsize=10)

    fig.suptitle(f"Progression on {target_rel_image.name}", fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_metric_curves(df: pd.DataFrame, save_path: Path) -> None:
    step = df["step"].tolist()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

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
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    dataset = DATASET
    output_root = OUTPUT_ROOT
    save_dir = SAVE_DIR if SAVE_DIR is not None else output_root / f"{dataset}_progress_report"

    eval_dirs = find_eval_dirs(output_root, dataset)
    if not eval_dirs:
        raise FileNotFoundError(
            f"No eval dirs found under '{output_root}' for dataset '{dataset}'.\n"
            f"Expected folders like: {dataset}_stepXXXXX_eval"
        )

    save_dir.mkdir(parents=True, exist_ok=True)

    df = collect_metric_table(eval_dirs)
    metrics_csv_path = save_dir / f"{dataset}_metrics_summary.csv"
    df.to_csv(metrics_csv_path, index=False, encoding="utf-8-sig")

    print("=" * 100)
    print(f"[INFO] Dataset: {dataset}")
    print(f"[INFO] Metrics summary saved to: {metrics_csv_path}")
    print("=" * 100)
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(df)

    curve_path = save_dir / f"{dataset}_metric_curves.png"
    plot_metric_curves(df, curve_path)
    print(f"[INFO] Metric curves saved to: {curve_path}")

    target_rel_image = select_target_image(eval_dirs, image_name=IMAGE_NAME)
    if target_rel_image is None:
        print("[WARN] No common image found across eval dirs. Metrics summary is still available.")
        return

    progress_path = save_dir / f"{dataset}_progression_{target_rel_image.stem}.png"
    build_progress_figure(df, eval_dirs, target_rel_image, progress_path)
    print(f"[INFO] Progress figure saved to: {progress_path}")
    print(f"[INFO] Selected common image: {target_rel_image}")


if __name__ == "__main__":
    main()