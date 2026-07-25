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
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

matplotlib.use("Agg")


def parse_args():
    parser = argparse.ArgumentParser(description="Export the left trajectory map panel as a standalone figure.")
    parser.add_argument(
        "--summary_json",
        type=str,
        default="/home/tanghx/icus2026/usc_best_unseen_gcf_birmingham_block_5_obj10_desc5.json",
    )
    parser.add_argument("--gcf_window", type=int, default=7)
    parser.add_argument(
        "--output_png",
        type=str,
        default="/home/tanghx/icus2026/usc_best_unseen_gcf_birmingham_block_5_obj10_desc5_left_panel.png",
    )
    return parser.parse_args()


def world_to_image_index(raster: rasterio.io.DatasetReader, xy: tuple[float, float]) -> tuple[int, int]:
    row, col = raster.index(xy[0], xy[1])
    row = int(np.clip(row, 0, raster.height - 1))
    col = int(np.clip(col, 0, raster.width - 1))
    return row, col


def pose_rows_cols(raster: rasterio.io.DatasetReader, poses: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = zip(*(world_to_image_index(raster, (pose[0], pose[1])) for pose in poses))
    return np.asarray(rows), np.asarray(cols)


def point_rows_cols(raster: rasterio.io.DatasetReader, points: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = zip(*(world_to_image_index(raster, (point[0], point[1])) for point in points))
    return np.asarray(rows), np.asarray(cols)


def compute_clip_box(
    height: int,
    width: int,
    rows_list: list[np.ndarray],
    cols_list: list[np.ndarray],
    min_pad: int = 96,
    frac_pad: float = 0.18,
) -> tuple[int, int, int, int]:
    valid_rows = [rows for rows in rows_list if rows.size > 0]
    valid_cols = [cols for cols in cols_list if cols.size > 0]
    if not valid_rows or not valid_cols:
        return 0, height, 0, width

    all_rows = np.concatenate(valid_rows)
    all_cols = np.concatenate(valid_cols)
    row_min = int(all_rows.min())
    row_max = int(all_rows.max())
    col_min = int(all_cols.min())
    col_max = int(all_cols.max())

    row_span = max(row_max - row_min, 1)
    col_span = max(col_max - col_min, 1)
    row_pad = max(min_pad, int(row_span * frac_pad))
    col_pad = max(min_pad, int(col_span * frac_pad))

    top = max(0, row_min - row_pad)
    bottom = min(height, row_max + row_pad + 1)
    left = max(0, col_min - col_pad)
    right = min(width, col_max + col_pad + 1)
    return top, bottom, left, right


def main():
    args = parse_args()
    summary = json.loads(Path(args.summary_json).read_text())
    map_name = summary["selected_episode"]["map_name"]

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

    rgb = cv2.cvtColor(cv2.imread(str(png_path)), cv2.COLOR_BGR2RGB)
    raster = rasterio.open(tif_path)
    try:
        teacher = summary["teacher_trajectory"]
        traj = summary["trajectory"]
        pred_goals = summary["pred_goal_logs"]
        window_points = pred_goals[-args.gcf_window :]

        teacher_rows, teacher_cols = pose_rows_cols(raster, teacher)
        traj_rows, traj_cols = pose_rows_cols(raster, traj)
        pred_rows, pred_cols = point_rows_cols(raster, pred_goals)
        if window_points:
            w_rows, w_cols = point_rows_cols(raster, window_points)
        else:
            w_rows = w_cols = np.asarray([])

        target_row, target_col = world_to_image_index(raster, tuple(summary["target_xy"]))
        start_row, start_col = world_to_image_index(raster, tuple(summary["start_xy"]))
        selected_row, selected_col = world_to_image_index(raster, tuple(summary["selected_goal_xy"]))
        one_step_row, one_step_col = world_to_image_index(raster, tuple(pred_goals[-1]))
        last_row, last_col = world_to_image_index(raster, tuple(traj[-1][:2]))
        clip_top, clip_bottom, clip_left, clip_right = compute_clip_box(
            rgb.shape[0],
            rgb.shape[1],
            [
                teacher_rows,
                traj_rows,
                pred_rows,
                np.asarray([target_row, start_row, selected_row, last_row]),
            ],
            [
                teacher_cols,
                traj_cols,
                pred_cols,
                np.asarray([target_col, start_col, selected_col, last_col]),
            ],
        )
    finally:
        raster.close()

    rgb = rgb[clip_top:clip_bottom, clip_left:clip_right]
    teacher_rows = teacher_rows - clip_top
    teacher_cols = teacher_cols - clip_left
    traj_rows = traj_rows - clip_top
    traj_cols = traj_cols - clip_left
    pred_rows = pred_rows - clip_top
    pred_cols = pred_cols - clip_left
    w_rows = w_rows - clip_top
    w_cols = w_cols - clip_left
    target_row -= clip_top
    target_col -= clip_left
    start_row -= clip_top
    start_col -= clip_left
    selected_row -= clip_top
    selected_col -= clip_left
    one_step_row -= clip_top
    one_step_col -= clip_left
    last_row -= clip_top
    last_col -= clip_left

    fig, ax = plt.subplots(figsize=(8.6, 5.7))
    ax.imshow(rgb)
    ax.plot(teacher_cols, teacher_rows, "--", lw=2.0, color="#ffffff", alpha=0.80, label="Teacher")
    ax.plot(traj_cols, traj_rows, "-", lw=2.6, color="#23c9ff", alpha=0.95, label="Rollout")
    scatter = ax.scatter(pred_cols, pred_rows, c=np.arange(len(pred_goals)), cmap="plasma", s=34, alpha=0.85, label="Pred")
    ax.scatter([start_col], [start_row], marker="o", s=82, color="#3ddc84", edgecolors="black", linewidths=0.8, label="Start")
    ax.scatter([target_col], [target_row], marker="*", s=200, color="#ff4d6d", edgecolors="black", linewidths=0.9, label="GT")
    ax.scatter([last_col], [last_row], marker="D", s=82, color="#f9c74f", edgecolors="black", linewidths=0.8, label="Last")
    ax.scatter([one_step_col], [one_step_row], marker="s", s=96, color="#ffb703", edgecolors="black", linewidths=0.9, label="1-step")
    ax.scatter([selected_col], [selected_row], marker="X", s=176, color="#7b2cbf", edgecolors="white", linewidths=1.0, label="GCF")
    ax.plot([last_col, selected_col], [last_row, selected_row], ":", color="#7b2cbf", lw=2.0, alpha=0.85)
    ax.plot([one_step_col, selected_col], [one_step_row, selected_row], "--", color="#ffb703", lw=1.2, alpha=0.9)
    if len(window_points):
        ax.scatter(w_cols, w_rows, facecolors="none", edgecolors="#ffffff", s=108, linewidths=1.1, label=f"W={args.gcf_window}")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("")
    legend = ax.legend(
        loc="lower right",
        fontsize=6.4,
        framealpha=0.76,
        borderpad=0.22,
        labelspacing=0.16,
        handlelength=1.12,
        markerscale=0.72,
    )
    legend.get_frame().set_linewidth(0.8)

    center_col = int(round((target_col + one_step_col + selected_col) / 3))
    center_row = int(round((target_row + one_step_row + selected_row) / 3))
    half_span = 30
    ax.add_patch(
        Rectangle(
            (center_col - half_span, center_row - half_span),
            2 * half_span,
            2 * half_span,
            fill=False,
            ec="#223047",
            lw=1.0,
            alpha=0.8,
        )
    )

    inset = inset_axes(ax, width="24%", height="28%", loc="upper left", borderpad=0.95)
    inset.imshow(rgb)
    inset.scatter(pred_cols[-args.gcf_window :], pred_rows[-args.gcf_window :], c="#7c83fd", s=12, alpha=0.75)
    inset.scatter([target_col], [target_row], marker="*", s=145, color="#ff4d6d", edgecolors="black", linewidths=0.8)
    inset.scatter([one_step_col], [one_step_row], marker="s", s=80, color="#ffb703", edgecolors="black", linewidths=0.8)
    inset.scatter([selected_col], [selected_row], marker="X", s=130, color="#7b2cbf", edgecolors="white", linewidths=0.8)
    inset.plot([one_step_col, selected_col], [one_step_row, selected_row], "--", color="#ffb703", lw=1.0, alpha=0.95)
    inset.set_xlim(center_col - half_span, center_col + half_span)
    inset.set_ylim(center_row + half_span, center_row - half_span)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_title("GCF zoom", fontsize=7.3, pad=1.8)
    for spine in inset.spines.values():
        spine.set_edgecolor("#223047")
        spine.set_linewidth(1.1)

    inset.annotate(
        "GT goal",
        xy=(target_col, target_row),
        xytext=(8, -8),
        textcoords="offset points",
        fontsize=6.3,
        color="#a61e4d",
        bbox=dict(boxstyle="round,pad=0.14", facecolor="white", alpha=0.75, edgecolor="none"),
    )
    inset.annotate(
        "1-step goal",
        xy=(one_step_col, one_step_row),
        xytext=(8, 12),
        textcoords="offset points",
        fontsize=6.2,
        color="#9a6700",
        bbox=dict(boxstyle="round,pad=0.14", facecolor="white", alpha=0.75, edgecolor="none"),
    )
    inset.annotate(
        "GT goal",
        xy=(target_col, target_row),
        xytext=(-10, -12),
        textcoords="offset points",
        fontsize=6.2,
        color="#a61e4d",
        bbox=dict(boxstyle="round,pad=0.14", facecolor="white", alpha=0.75, edgecolor="none"),
    )
    inset.annotate(
        "GCF goal",
        xy=(selected_col, selected_row),
        xytext=(-58, 14),
        textcoords="offset points",
        fontsize=6.2,
        color="#5a189a",
        bbox=dict(boxstyle="round,pad=0.14", facecolor="white", alpha=0.75, edgecolor="none"),
    )

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.034, pad=0.012)
    cbar.set_label("")
    cbar.ax.set_ylabel("")

    fig.tight_layout(pad=0.25)
    out_path = Path(args.output_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
