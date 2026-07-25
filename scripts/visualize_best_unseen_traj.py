#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import rasterio
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from gsamllavanav.cityreferobject import get_city_refer_objects
from gsamllavanav.dataset.episode import Episode
from gsamllavanav.dataset.generate import generate_episodes_from_mturk_trajectories
from gsamllavanav.dataset.mturk_trajectory import load_mturk_trajectories
from gsamllavanav.defaultpaths import ORTHO_IMAGE_DIR
from gsamllavanav.evaluate_baseline_with_map import run_episodes_batch
from gsamllavanav.maps.landmark_nav_map import LandmarkNavMap
from gsamllavanav.model_registry import get_model_spec
from gsamllavanav.observation import cropclient
from gsamllavanav.parser import parse_args
from gsamllavanav.space import Point2D, Pose4D


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Candidate:
    episode_id: str
    map_name: str
    object_id: int
    description_id: int
    description: str
    final_goal_xy: tuple[float, float]
    final_distance: float
    success: bool
    sigma: float
    endpoint_dist: float
    weighted_progress: float
    final_progress: float
    rollout_steps: int
    rollout_path_length: float
    teacher_path_length: float


def parse_cli():
    parser = argparse.ArgumentParser(description="Search and visualize a strong unseen USC trajectory.")
    parser.add_argument("--split", choices=["val_seen", "val_unseen", "test_unseen"], default="val_unseen")
    parser.add_argument("--search_limit", type=int, default=128)
    parser.add_argument("--search_batch_size", type=int, default=8)
    parser.add_argument("--episode_id", type=str, default="")
    parser.add_argument("--gcf_window", type=int, default=7)
    parser.add_argument("--gcf_beta", type=float, default=3.0)
    parser.add_argument("--output_dir", type=str, default="/home/tanghx/icus2026")
    parser.add_argument("--output_stem", type=str, default="usc_best_unseen_gcf")
    cli, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    args = parse_args()
    return cli, args


def path_length(poses: list[Pose4D]) -> float:
    if len(poses) < 2:
        return 0.0
    return float(
        sum(src.xy.dist_to(dst.xy) for src, dst in zip(poses[:-1], poses[1:]))
    )


def weighted_gcf_goal(
    pred_goals: list[Point2D],
    pred_progresses: list[float],
    window: int,
    beta: float,
) -> tuple[Point2D | None, dict[str, float | int]]:
    if not pred_goals:
        return None, {
            "sigma": float("inf"),
            "variance": float("inf"),
            "endpoint_dist": float("inf"),
            "weighted_progress": 0.0,
            "window_size": 0,
        }

    goal_window = pred_goals[-window:]
    progress_window = pred_progresses[-window:] if pred_progresses else [0.0] * len(goal_window)
    if len(progress_window) < len(goal_window):
        progress_window = progress_window + [progress_window[-1] if progress_window else 0.0] * (
            len(goal_window) - len(progress_window)
        )

    goal_array = np.asarray([[goal.x, goal.y] for goal in goal_window], dtype=np.float32)
    progress_array = np.asarray(progress_window, dtype=np.float32)
    logits = beta * (progress_array - progress_array.max()) if len(progress_array) else np.zeros((len(goal_window),), dtype=np.float32)
    weights = np.exp(logits)
    weights = weights / max(float(weights.sum()), 1e-6)
    mean_xy = (weights[:, None] * goal_array).sum(axis=0)
    distances = np.linalg.norm(goal_array - mean_xy[None, :], axis=1)
    variance = float((weights * distances**2).sum())
    sigma = float(np.sqrt(max(variance, 0.0)))
    endpoint_dist = float(np.linalg.norm(goal_array[-1] - mean_xy))
    return Point2D(float(mean_xy[0]), float(mean_xy[1])), {
        "sigma": sigma,
        "variance": variance,
        "endpoint_dist": endpoint_dist,
        "weighted_progress": float((weights * progress_array).sum()) if len(progress_array) else 0.0,
        "window_size": len(goal_window),
    }


def world_to_image_index(raster: rasterio.io.DatasetReader, xy: Point2D) -> tuple[int, int]:
    row, col = raster.index(xy.x, xy.y)
    row = int(np.clip(row, 0, raster.height - 1))
    col = int(np.clip(col, 0, raster.width - 1))
    return row, col


def pose_rows_cols(raster: rasterio.io.DatasetReader, poses: list[Pose4D]) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = zip(*(world_to_image_index(raster, pose.xy) for pose in poses))
    return np.asarray(rows), np.asarray(cols)


def point_rows_cols(raster: rasterio.io.DatasetReader, points: list[Point2D]) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = zip(*(world_to_image_index(raster, point) for point in points))
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


def load_ortho_assets(map_name: str, alt_env: str) -> tuple[np.ndarray, rasterio.io.DatasetReader]:
    image_dir = ORTHO_IMAGE_DIR / alt_env if alt_env else ORTHO_IMAGE_DIR
    rgb_path = image_dir / f"{map_name}.png"
    tif_path = image_dir / f"{map_name}.tif"
    rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
    raster = rasterio.open(tif_path)
    return rgb, raster


def build_final_nav_map(args, episode: Episode, poses: list[Pose4D]) -> LandmarkNavMap:
    cropclient.load_image_cache(alt_env=args.alt_env)
    nav_map = LandmarkNavMap(
        episode.map_name,
        args.map_shape,
        args.map_pixels_per_meter,
        episode.description_landmarks,
        episode.description_target,
        episode.description_surroundings,
        args.gsam_params,
    )
    for pose in poses:
        gsam_rgb = cropclient.crop_image(episode.map_name, pose, args.gsam_rgb_shape, "rgb")
        nav_map.update_observations(pose, gsam_rgb, None, args.gsam_use_map_cache)
    return nav_map


def save_overlay_layer(
    path: Path,
    width: int,
    height: int,
    drawer,
):
    dpi = 200
    fig = plt.figure(figsize=(max(width / dpi, 1.0), max(height / dpi, 1.0)), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor((0, 0, 0, 0))
    drawer(ax)
    fig.savefig(path, dpi=dpi, transparent=True)
    plt.close(fig)


def export_separated_assets(
    assets_dir: Path,
    rgb: np.ndarray,
    teacher_rows: np.ndarray,
    teacher_cols: np.ndarray,
    traj_rows: np.ndarray,
    traj_cols: np.ndarray,
    pred_rows: np.ndarray,
    pred_cols: np.ndarray,
    w_rows: np.ndarray,
    w_cols: np.ndarray,
    start_row: int,
    start_col: int,
    target_row: int,
    target_col: int,
    last_row: int,
    last_col: int,
    selected_row: int,
    selected_col: int,
    pred_progresses: list[float],
    cli,
    args,
    candidate: Candidate,
    selected_goal: Point2D,
    episode: Episode,
    map_channels: np.ndarray,
    map_titles: list[str],
    pred_goals: list[Point2D],
    poses: list[Pose4D],
):
    assets_dir.mkdir(parents=True, exist_ok=True)
    height, width = rgb.shape[:2]

    plt.imsave(assets_dir / "01_local_ortho.png", rgb)

    save_overlay_layer(
        assets_dir / "02_teacher_path_layer.png",
        width,
        height,
        lambda ax: ax.plot(teacher_cols, teacher_rows, "--", lw=2.0, color="#ffffff", alpha=0.9),
    )
    save_overlay_layer(
        assets_dir / "03_rollout_path_layer.png",
        width,
        height,
        lambda ax: ax.plot(traj_cols, traj_rows, "-", lw=2.6, color="#23c9ff", alpha=0.95),
    )
    save_overlay_layer(
        assets_dir / "04_pred_goals_layer.png",
        width,
        height,
        lambda ax: ax.scatter(
            pred_cols,
            pred_rows,
            c=np.arange(len(pred_cols)),
            cmap="plasma",
            s=36,
            alpha=0.9,
        ),
    )
    save_overlay_layer(
        assets_dir / "05_recent_window_layer.png",
        width,
        height,
        lambda ax: ax.scatter(
            w_cols,
            w_rows,
            facecolors="none",
            edgecolors="#ffffff",
            s=120,
            linewidths=1.2,
        ),
    )
    save_overlay_layer(
        assets_dir / "06_terminal_link_layer.png",
        width,
        height,
        lambda ax: ax.plot(
            [last_col, selected_col],
            [last_row, selected_row],
            ":",
            color="#7b2cbf",
            lw=2.0,
            alpha=0.9,
        ),
    )

    def draw_keypoints(ax):
        ax.scatter([start_col], [start_row], marker="o", s=90, color="#3ddc84", edgecolors="black", linewidths=0.8)
        ax.scatter([target_col], [target_row], marker="*", s=220, color="#ff4d6d", edgecolors="black", linewidths=0.9)
        ax.scatter([last_col], [last_row], marker="D", s=90, color="#f9c74f", edgecolors="black", linewidths=0.8)
        ax.scatter([selected_col], [selected_row], marker="X", s=190, color="#7b2cbf", edgecolors="white", linewidths=1.0)

    save_overlay_layer(
        assets_dir / "07_keypoints_layer.png",
        width,
        height,
        draw_keypoints,
    )

    progress_steps = np.arange(1, len(pred_progresses) + 1)
    fig_prog, ax_prog = plt.subplots(figsize=(8.5, 3.6))
    ax_prog.plot(progress_steps, pred_progresses, color="#4361ee", lw=2.0, marker="o", markersize=3.5)
    ax_prog.axhline(args.progress_stop_val, color="#ef476f", linestyle="--", linewidth=1.4, label="Progress stop thr")
    ax_prog.set_xlim(1, max(len(pred_progresses), 1))
    ax_prog.set_ylim(0.0, max(1.0, max(pred_progresses) + 0.05 if pred_progresses else 1.0))
    ax_prog.set_xlabel("Step")
    ax_prog.set_ylabel("Pred progress")
    ax_prog.grid(alpha=0.25)
    ax_prog.legend(loc="lower right", fontsize=9)
    ax_prog.set_title(
        f"GCF(window={cli.gcf_window}, beta={cli.gcf_beta}) | "
        f"sigma={candidate.sigma:.2f}, endpoint={candidate.endpoint_dist:.2f} m"
    )
    fig_prog.tight_layout()
    fig_prog.savefig(assets_dir / "08_progress_curve.png", dpi=220, bbox_inches="tight")
    plt.close(fig_prog)

    for idx, (title, channel) in enumerate(zip(map_titles, map_channels), start=9):
        title_slug = title.lower().replace("-", "_").replace(" ", "_")
        plt.imsave(assets_dir / f"{idx:02d}_map_{title_slug}.png", channel, cmap="viridis")

    wrapped_desc = textwrap.fill(episode.target_description, width=88)
    summary_lines = [
        f"Episode ID: {episode.id}",
        f"Map: {episode.map_name}",
        f"Checkpoint: {args.checkpoint}",
        f"Model: {args.model}",
        "",
        "Instruction:",
        wrapped_desc,
        "",
        f"Teacher path length: {candidate.teacher_path_length:.3f} m",
        f"Rollout path length: {candidate.rollout_path_length:.3f} m",
        f"Rollout steps: {candidate.rollout_steps}",
        f"Success: {candidate.success}",
        f"Final distance: {candidate.final_distance:.4f} m",
        f"Final progress: {candidate.final_progress:.6f}",
        f"Weighted progress (window): {candidate.weighted_progress:.6f}",
        f"Belief sigma: {candidate.sigma:.6f}",
        f"Belief endpoint distance: {candidate.endpoint_dist:.6f} m",
        f"Selected belief goal: ({selected_goal.x:.4f}, {selected_goal.y:.4f})",
        f"Ground-truth goal: ({episode.target_position.x:.4f}, {episode.target_position.y:.4f})",
        f"Start pose xy: ({episode.start_pose.x:.4f}, {episode.start_pose.y:.4f})",
    ]
    (assets_dir / "14_instruction_and_stats.txt").write_text("\n".join(summary_lines) + "\n")

    np.savetxt(
        assets_dir / "15_pred_progress.csv",
        np.column_stack([progress_steps, np.asarray(pred_progresses, dtype=np.float32)]),
        delimiter=",",
        header="step,pred_progress",
        comments="",
    )
    np.savetxt(
        assets_dir / "16_pred_goals_xy.csv",
        np.asarray([[point.x, point.y] for point in pred_goals], dtype=np.float32),
        delimiter=",",
        header="x,y",
        comments="",
    )
    np.savetxt(
        assets_dir / "17_rollout_xy.csv",
        np.asarray([[pose.x, pose.y] for pose in poses], dtype=np.float32),
        delimiter=",",
        header="x,y",
        comments="",
    )
    np.savetxt(
        assets_dir / "18_teacher_xy.csv",
        np.asarray([[pose.x, pose.y] for pose in episode.teacher_trajectory], dtype=np.float32),
        delimiter=",",
        header="x,y",
        comments="",
    )


def render_figure(
    args,
    cli,
    episode: Episode,
    candidate: Candidate,
    poses: list[Pose4D],
    pred_goals: list[Point2D],
    pred_progresses: list[float],
    selected_goal: Point2D,
    nav_map: LandmarkNavMap,
    output_prefix: Path,
):
    rgb, raster = load_ortho_assets(episode.map_name, args.alt_env)
    try:
        teacher_rows, teacher_cols = pose_rows_cols(raster, episode.teacher_trajectory)
        traj_rows, traj_cols = pose_rows_cols(raster, poses)
        pred_rows, pred_cols = point_rows_cols(raster, pred_goals)
        window_points = pred_goals[-cli.gcf_window :]
        if window_points:
            w_rows, w_cols = point_rows_cols(raster, window_points)
        else:
            w_rows = w_cols = np.asarray([])
        target_row, target_col = world_to_image_index(raster, episode.target_position)
        start_row, start_col = world_to_image_index(raster, episode.start_pose.xy)
        selected_row, selected_col = world_to_image_index(raster, selected_goal)
        last_row, last_col = world_to_image_index(raster, poses[-1].xy)
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
    last_row -= clip_top
    last_col -= clip_left

    map_channels = nav_map.to_array()
    map_teacher_rows, map_teacher_cols = nav_map.to_rows_cols([pose.xy for pose in episode.teacher_trajectory])
    map_traj_rows, map_traj_cols = nav_map.to_rows_cols([pose.xy for pose in poses])
    map_pred_rows, map_pred_cols = nav_map.to_rows_cols(pred_goals)
    map_misc_rows, map_misc_cols = nav_map.to_rows_cols(
        [episode.target_position.xy, episode.start_pose.xy, selected_goal, poses[-1].xy]
    )
    map_clip_top, map_clip_bottom, map_clip_left, map_clip_right = compute_clip_box(
        map_channels.shape[1],
        map_channels.shape[2],
        [map_teacher_rows, map_traj_rows, map_pred_rows, map_misc_rows],
        [map_teacher_cols, map_traj_cols, map_pred_cols, map_misc_cols],
        min_pad=10,
        frac_pad=0.22,
    )
    map_channels = map_channels[:, map_clip_top:map_clip_bottom, map_clip_left:map_clip_right]
    map_titles = ["Track-View", "Track-Explored", "Landmarks", "Target", "Surroundings"]
    progress_steps = np.arange(1, len(pred_progresses) + 1)
    wrapped_desc = textwrap.fill(episode.target_description, width=76)

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(3, 4, width_ratios=[1.55, 1.55, 1.0, 1.0], height_ratios=[1.05, 1.0, 1.0])
    ax_map = fig.add_subplot(gs[:, :2])
    ax_prog = fig.add_subplot(gs[0, 2:])
    channel_grid = gs[1:, 2:].subgridspec(2, 3, wspace=0.12, hspace=0.18)
    channel_axes = [fig.add_subplot(channel_grid[i // 3, i % 3]) for i in range(6)]

    ax_map.imshow(rgb)
    ax_map.plot(teacher_cols, teacher_rows, "--", lw=2.0, color="#ffffff", alpha=0.80, label="Teacher")
    ax_map.plot(traj_cols, traj_rows, "-", lw=2.6, color="#23c9ff", alpha=0.95, label="Rollout")
    scatter = ax_map.scatter(pred_cols, pred_rows, c=np.arange(len(pred_goals)), cmap="plasma", s=36, alpha=0.85, label="Pred goals")
    ax_map.scatter([start_col], [start_row], marker="o", s=90, color="#3ddc84", edgecolors="black", linewidths=0.8, label="Start")
    ax_map.scatter([target_col], [target_row], marker="*", s=220, color="#ff4d6d", edgecolors="black", linewidths=0.9, label="GT target")
    ax_map.scatter([last_col], [last_row], marker="D", s=90, color="#f9c74f", edgecolors="black", linewidths=0.8, label="Last pose")
    ax_map.scatter([selected_col], [selected_row], marker="X", s=190, color="#7b2cbf", edgecolors="white", linewidths=1.0, label="Belief goal")
    ax_map.plot([last_col, selected_col], [last_row, selected_row], ":", color="#7b2cbf", lw=2.0, alpha=0.85)
    if window_points:
        ax_map.scatter(w_cols, w_rows, facecolors="none", edgecolors="#ffffff", s=120, linewidths=1.2, label=f"Last {cli.gcf_window}")
    ax_map.set_title(
        f"{episode.map_name} | {episode.id} | success={candidate.success} | final dist={candidate.final_distance:.2f} m",
        fontsize=12,
    )
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    ax_map.legend(loc="lower right", fontsize=9, framealpha=0.92)
    cbar = fig.colorbar(scatter, ax=ax_map, fraction=0.028, pad=0.012)
    cbar.set_label("Prediction step", rotation=90)

    ax_prog.plot(progress_steps, pred_progresses, color="#4361ee", lw=2.0, marker="o", markersize=3.5)
    ax_prog.axhline(args.progress_stop_val, color="#ef476f", linestyle="--", linewidth=1.4, label="Progress stop thr")
    ax_prog.set_xlim(1, max(len(pred_progresses), 1))
    ax_prog.set_ylim(0.0, max(1.0, max(pred_progresses) + 0.05 if pred_progresses else 1.0))
    ax_prog.set_xlabel("Step")
    ax_prog.set_ylabel("Pred progress")
    ax_prog.grid(alpha=0.25)
    ax_prog.legend(loc="lower right", fontsize=9)
    ax_prog.set_title(
        f"GCF(window={cli.gcf_window}, beta={cli.gcf_beta}) | "
        f"sigma={candidate.sigma:.2f}, endpoint={candidate.endpoint_dist:.2f} m"
    )

    for ax, title, channel in zip(channel_axes[:5], map_titles, map_channels):
        ax.imshow(channel, cmap="viridis")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    channel_axes[5].axis("off")
    channel_axes[5].text(
        0.0,
        1.0,
        "\n".join(
            [
                "Instruction:",
                wrapped_desc,
                "",
                f"Teacher path: {candidate.teacher_path_length:.1f} m",
                f"Rollout path: {candidate.rollout_path_length:.1f} m",
                f"Final progress: {candidate.final_progress:.3f}",
                f"Weighted progress (window): {candidate.weighted_progress:.3f}",
                f"Belief goal: ({selected_goal.x:.1f}, {selected_goal.y:.1f})",
                f"Target goal: ({episode.target_position.x:.1f}, {episode.target_position.y:.1f})",
            ]
        ),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )

    fig.suptitle("USC Ins-Dec + Belief Filter Trajectory Visualization", fontsize=15, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    assets_dir = output_prefix.parent / f"{output_prefix.name}_assets"
    export_separated_assets(
        assets_dir=assets_dir,
        rgb=rgb,
        teacher_rows=teacher_rows,
        teacher_cols=teacher_cols,
        traj_rows=traj_rows,
        traj_cols=traj_cols,
        pred_rows=pred_rows,
        pred_cols=pred_cols,
        w_rows=w_rows,
        w_cols=w_cols,
        start_row=start_row,
        start_col=start_col,
        target_row=target_row,
        target_col=target_col,
        last_row=last_row,
        last_col=last_col,
        selected_row=selected_row,
        selected_col=selected_col,
        pred_progresses=pred_progresses,
        cli=cli,
        args=args,
        candidate=candidate,
        selected_goal=selected_goal,
        episode=episode,
        map_channels=map_channels,
        map_titles=map_titles,
        pred_goals=pred_goals,
        poses=poses,
    )
    return assets_dir


def main():
    cli, args = parse_cli()
    args.mode = "eval"
    args.eval_batch_size = cli.search_batch_size
    args.eval_agent_mode = "progress_stop"
    args.eval_goal_selector = "raw"
    args.gcf_filter = False
    args.terminal_belief_debug = False

    model_spec = get_model_spec(args.model)
    if model_spec.pipeline != "baseline_with_map":
        raise ValueError(f"This script expects a baseline-with-map model, got: {args.model}")
    if not args.checkpoint:
        raise ValueError("--checkpoint is required")

    output_dir = Path(cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    objects = get_city_refer_objects()
    trajectories = load_mturk_trajectories(cli.split, "all", args.altitude)
    episodes = generate_episodes_from_mturk_trajectories(objects, trajectories)
    if cli.episode_id:
        episodes = [episode for episode in episodes if str(episode.id) == cli.episode_id]
        if not episodes:
            raise ValueError(f"Episode id not found in split {cli.split}: {cli.episode_id}")
    else:
        episodes = episodes[: cli.search_limit]

    model = model_spec.factory(args.map_size).to(DEVICE)
    state = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(state["predictor_state_dict"])

    pose_logs, pred_goal_logs, pred_progress_logs = run_episodes_batch(args, model, episodes, DEVICE)

    candidates: list[Candidate] = []
    selected_points: dict[str, Point2D] = {}
    for episode in episodes:
        episode_key = str(episode.id)
        pred_goals = pred_goal_logs.get(episode.id, [])
        pred_progresses = pred_progress_logs.get(episode.id, [])
        selected_goal, stats = weighted_gcf_goal(pred_goals, pred_progresses, cli.gcf_window, cli.gcf_beta)
        if selected_goal is None:
            continue
        selected_points[episode_key] = selected_goal
        final_dist = selected_goal.dist_to(episode.target_position.xy)
        poses = pose_logs.get(episode.id, [])
        candidates.append(
            Candidate(
                episode_id=episode_key,
                map_name=episode.map_name,
                object_id=episode.id[1],
                description_id=episode.id[2],
                description=episode.target_description,
                final_goal_xy=(selected_goal.x, selected_goal.y),
                final_distance=float(final_dist),
                success=bool(final_dist <= args.success_dist),
                sigma=float(stats["sigma"]),
                endpoint_dist=float(stats["endpoint_dist"]),
                weighted_progress=float(stats["weighted_progress"]),
                final_progress=float(pred_progresses[-1] if pred_progresses else 0.0),
                rollout_steps=len(poses),
                rollout_path_length=path_length(poses),
                teacher_path_length=path_length(episode.teacher_trajectory),
            )
        )

    if not candidates:
        raise RuntimeError("No candidates found; prediction logs were empty.")

    candidates.sort(
        key=lambda c: (
            0 if c.success else 1,
            c.final_distance,
            c.sigma,
            c.rollout_path_length,
        )
    )
    best = candidates[0]
    episode = next(ep for ep in episodes if str(ep.id) == best.episode_id)
    poses = pose_logs[episode.id]
    pred_goals = pred_goal_logs[episode.id]
    pred_progresses = pred_progress_logs[episode.id]
    selected_goal = selected_points[best.episode_id]
    nav_map = build_final_nav_map(args, episode, poses)

    safe_id = f"{episode.map_name}_obj{episode.id[1]}_desc{episode.id[2]}"
    output_prefix = output_dir / f"{cli.output_stem}_{safe_id}"
    assets_dir = render_figure(
        args, cli, episode, best, poses, pred_goals, pred_progresses, selected_goal, nav_map, output_prefix
    )

    summary = {
        "split": cli.split,
        "search_limit": cli.search_limit,
        "gcf_window": cli.gcf_window,
        "gcf_beta": cli.gcf_beta,
        "checkpoint": args.checkpoint,
        "model": args.model,
        "device": DEVICE,
        "selected_episode": asdict(best),
        "selected_goal_xy": [selected_goal.x, selected_goal.y],
        "target_xy": [episode.target_position.x, episode.target_position.y],
        "start_xy": [episode.start_pose.x, episode.start_pose.y],
        "top_candidates": [asdict(candidate) for candidate in candidates[:10]],
        "trajectory": [list(pose) for pose in poses],
        "pred_goal_logs": [list(point) for point in pred_goals],
        "pred_progress_logs": pred_progresses,
        "teacher_trajectory": [list(pose) for pose in episode.teacher_trajectory],
    }
    with open(output_prefix.with_suffix(".json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(assets_dir / "00_metadata.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(
        {
            "output_prefix": str(output_prefix),
            "assets_dir": str(assets_dir),
            "selected_episode": best.episode_id,
            "success": best.success,
            "final_distance": round(best.final_distance, 4),
            "rollout_steps": best.rollout_steps,
            "top3": [
                {
                    "episode_id": candidate.episode_id,
                    "success": candidate.success,
                    "final_distance": round(candidate.final_distance, 4),
                    "sigma": round(candidate.sigma, 4),
                }
                for candidate in candidates[:3]
            ],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
