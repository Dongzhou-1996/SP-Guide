#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rasterio

matplotlib.use("Agg")


def parse_args():
    parser = argparse.ArgumentParser(description="Export a minimal local GCF explanation figure.")
    parser.add_argument(
        "--summary_json",
        type=str,
        default="/home/tanghx/icus2026/usc_best_unseen_gcf_birmingham_block_5_obj10_desc5.json",
    )
    parser.add_argument("--gcf_window", type=int, default=7)
    parser.add_argument(
        "--output_png",
        type=str,
        default="/home/tanghx/icus2026/usc_best_unseen_gcf_local_triplet.png",
    )
    return parser.parse_args()


def world_to_image_index(raster: rasterio.io.DatasetReader, xy: tuple[float, float]) -> tuple[int, int]:
    row, col = raster.index(xy[0], xy[1])
    row = int(np.clip(row, 0, raster.height - 1))
    col = int(np.clip(col, 0, raster.width - 1))
    return row, col


def load_ortho_paths(map_name: str) -> tuple[Path, Path]:
    repo_root = Path("/home/tanghx/VLN/refer_repo/USCNav")
    tif_candidates = [
        repo_root / "data" / "rgbd" / f"{map_name}.tif",
        repo_root / "data" / "rgbd" / "train" / f"{map_name}.tif",
        Path("/home/tanghx/VLN/data/rgbd") / f"{map_name}.tif",
        Path("/home/tanghx/VLN/data/rgbd/train") / f"{map_name}.tif",
    ]
    png_candidates = [
        repo_root / "data" / "rgbd" / f"{map_name}.png",
        repo_root / "data" / "rgbd" / "train" / f"{map_name}.png",
        Path("/home/tanghx/VLN/data/rgbd") / f"{map_name}.png",
        Path("/home/tanghx/VLN/data/rgbd/train") / f"{map_name}.png",
    ]
    tif_path = next((p for p in tif_candidates if p.exists()), None)
    png_path = next((p for p in png_candidates if p.exists()), None)
    if tif_path is None or png_path is None:
        raise FileNotFoundError(f"Could not locate ortho assets for {map_name}")
    return tif_path, png_path


def tiny_ring_jitter(cols: np.ndarray, rows: np.ndarray, radius: float = 4.5) -> tuple[np.ndarray, np.ndarray]:
    center_col = float(cols.mean())
    center_row = float(rows.mean())
    angles = np.linspace(-np.pi / 2, 3 * np.pi / 2, len(cols), endpoint=False, dtype=np.float32)
    jittered_cols = center_col + radius * np.cos(angles)
    jittered_rows = center_row + radius * np.sin(angles)
    return jittered_cols.astype(np.float32), jittered_rows.astype(np.float32)


def main():
    args = parse_args()
    summary = json.loads(Path(args.summary_json).read_text())
    map_name = summary["selected_episode"]["map_name"]
    pred_goals = summary["pred_goal_logs"]
    window_goals = pred_goals[-args.gcf_window :]

    final_goal = np.asarray(summary["selected_goal_xy"], dtype=np.float32)
    stable_eps = 0.75
    stable_start = len(pred_goals)
    for idx in range(len(pred_goals) - 1, -1, -1):
        goal = np.asarray(pred_goals[idx], dtype=np.float32)
        if np.linalg.norm(goal - final_goal) > stable_eps:
            stable_start = idx + 1
            break
    pre_count = 3
    pre_goals = pred_goals[max(0, stable_start - pre_count) : stable_start]

    tif_path, png_path = load_ortho_paths(map_name)
    rgb = cv2.cvtColor(cv2.imread(str(png_path)), cv2.COLOR_BGR2RGB)

    with rasterio.open(tif_path) as raster:
        gcf_row, gcf_col = world_to_image_index(raster, tuple(summary["selected_goal_xy"]))
        one_row, one_col = world_to_image_index(raster, tuple(pred_goals[-1]))
        window_rows_cols = [world_to_image_index(raster, tuple(point)) for point in window_goals]
        pre_rows_cols = [world_to_image_index(raster, tuple(point)) for point in pre_goals]

    center_row = int(round((gcf_row + one_row) / 2))
    center_col = int(round((gcf_col + one_col) / 2))

    # Tight crop: keep only the local area needed to explain GCF.
    rows_for_crop = [gcf_row, one_row] + [row for row, _ in window_rows_cols] + [row for row, _ in pre_rows_cols]
    cols_for_crop = [gcf_col, one_col] + [col for _, col in window_rows_cols] + [col for _, col in pre_rows_cols]
    row_min, row_max = min(rows_for_crop), max(rows_for_crop)
    col_min, col_max = min(cols_for_crop), max(cols_for_crop)
    pad_left = 26
    pad_right = 30
    pad_top = 24
    pad_bottom = 16
    top = max(0, row_min - pad_top)
    bottom = min(rgb.shape[0], row_max + pad_bottom)
    left = max(0, col_min - pad_left)
    right = min(rgb.shape[1], col_max + pad_right)
    crop = rgb[top:bottom, left:right]

    gcf_row -= top
    gcf_col -= left
    one_row -= top
    one_col -= left
    window_rows = np.asarray([row - top for row, _ in window_rows_cols], dtype=np.float32)
    window_cols = np.asarray([col - left for _, col in window_rows_cols], dtype=np.float32)
    pre_rows = np.asarray([row - top for row, _ in pre_rows_cols], dtype=np.float32)
    pre_cols = np.asarray([col - left for _, col in pre_rows_cols], dtype=np.float32)

    cluster_span = 0.0
    if len(window_rows) > 0:
        cluster_span = float(
            np.max(np.hypot(window_cols - window_cols.mean(), window_rows - window_rows.mean()))
        )
    if cluster_span < 3.0 and len(window_rows) > 1:
        display_window_cols, display_window_rows = tiny_ring_jitter(window_cols, window_rows)
        use_visibility_offsets = True
    else:
        display_window_cols = window_cols.copy()
        display_window_rows = window_rows.copy()
        use_visibility_offsets = False

    display_pre_cols = pre_cols.copy()
    display_pre_rows = pre_rows.copy()
    if len(pre_rows) > 1:
        for idx in range(1, len(display_pre_rows)):
            if np.hypot(display_pre_cols[idx] - display_pre_cols[idx - 1], display_pre_rows[idx] - display_pre_rows[idx - 1]) < 3.0:
                display_pre_cols[idx] += 5.0 * idx
                display_pre_rows[idx] -= 2.0 * idx

    one_step_cols = np.concatenate([display_pre_cols, display_window_cols]) if len(display_window_cols) else display_pre_cols
    one_step_rows = np.concatenate([display_pre_rows, display_window_rows]) if len(display_window_rows) else display_pre_rows

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.imshow(crop)

    if len(one_step_cols) > 1:
        ax.plot(one_step_cols, one_step_rows, "-", color="#6b7280", lw=1.2, alpha=0.8, zorder=2)

    if len(pre_rows):
        ax.scatter(
            display_pre_cols,
            display_pre_rows,
            marker="o",
            s=40,
            color="#4f8ef7",
            edgecolors="white",
            linewidths=0.6,
            alpha=0.68,
            zorder=4,
        )

    if len(window_rows):
        for raw_c, raw_r, disp_c, disp_r in zip(window_cols, window_rows, display_window_cols, display_window_rows):
            if use_visibility_offsets:
                ax.plot([raw_c, disp_c], [raw_r, disp_r], ":", color="#d97706", lw=0.8, alpha=0.55, zorder=3)
        ax.scatter(
            display_window_cols,
            display_window_rows,
            marker="o",
            s=40,
            color="#ffb703",
            edgecolors="black",
            linewidths=0.5,
            alpha=0.68,
            zorder=5,
        )

    ax.scatter([gcf_col], [gcf_row], marker="X", s=180, color="#ffb703", edgecolors="white", linewidths=1.0, zorder=6)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_edgecolor("#223047")

    fig.tight_layout(pad=0.1)
    out_path = Path(args.output_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=260, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
