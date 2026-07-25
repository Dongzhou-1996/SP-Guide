#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from gsamllavanav.cityreferobject import get_city_refer_objects
from gsamllavanav.dataset.generate import generate_episodes_from_mturk_trajectories
from gsamllavanav.dataset.mturk_trajectory import load_mturk_trajectories
from gsamllavanav.maps.landmark_nav_map import LandmarkNavMap
from gsamllavanav.observation import cropclient
from gsamllavanav.parser import parse_args
from gsamllavanav.space import Pose4D


MAP_TITLES = ["track_view", "track_explored", "landmarks", "target", "surroundings"]
DEPTH_CMAP = "gray"
MAP_CMAP = LinearSegmentedColormap.from_list("soft_gray_map", ["#454A52", "#F3F5F7"])


def parse_cli():
    parser = argparse.ArgumentParser(description="Export RGB/depth/map observations for one rollout step.")
    parser.add_argument(
        "--summary_json",
        type=str,
        default="/home/tanghx/icus2026/usc_best_unseen_gcf_birmingham_block_5_obj10_desc5.json",
    )
    parser.add_argument("--step_index", type=int, default=-1, help="0-based rollout step. -1 selects the last non-repeated pose.")
    parser.add_argument("--output_dir", type=str, default="/home/tanghx/icus2026")
    cli, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    args = parse_args()
    return cli, args


def load_episode(split: str, altitude: float, map_name: str, object_id: int, description_id: int):
    objects = get_city_refer_objects()
    trajectories = load_mturk_trajectories(split, "all", altitude)
    episodes = generate_episodes_from_mturk_trajectories(objects, trajectories)
    for episode in episodes:
        if (
            episode.map_name == map_name
            and episode.target_object.id == object_id
            and episode.description_id == description_id
        ):
            return episode
    raise ValueError(f"Episode not found: {(map_name, object_id, description_id)} in split={split}")


def select_step_index(poses: list[Pose4D], requested_index: int) -> int:
    if not poses:
        raise ValueError("No rollout poses in summary json")
    if requested_index >= 0:
        return min(requested_index, len(poses) - 1)

    last_unique = 0
    for idx in range(1, len(poses)):
        prev_pose = poses[idx - 1]
        pose = poses[idx]
        moved = np.linalg.norm(np.array(pose.xy) - np.array(prev_pose.xy)) > 1e-6
        rotated = abs(pose.yaw - prev_pose.yaw) > 1e-6
        if moved or rotated:
            last_unique = idx
    return last_unique


def save_depth_png(path: Path, depth_raw: np.ndarray, max_depth: float):
    depth_vis = depth_to_vis(depth_raw, max_depth)
    plt.figure(figsize=(4.8, 4.8))
    plt.imshow(depth_vis, cmap=DEPTH_CMAP, vmin=0.0, vmax=1.0)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0)
    plt.close()


def depth_to_vis(depth_raw: np.ndarray, max_depth: float) -> np.ndarray:
    valid = depth_raw[np.isfinite(depth_raw)]
    if valid.size == 0:
        return np.zeros_like(depth_raw, dtype=np.float32)

    lo = float(np.percentile(valid, 2.0))
    hi = float(np.percentile(valid, 98.0))
    if hi <= lo + 1e-6:
        lo = float(valid.min())
        hi = float(valid.max()) if float(valid.max()) > lo else max(lo + 1.0, float(max_depth))

    clipped = np.clip(depth_raw, lo, hi)
    norm = (clipped - lo) / max(hi - lo, 1e-6)
    return np.power(norm, 0.9).astype(np.float32)


def save_combined_panel(
    path: Path,
    rgb_model: np.ndarray,
    depth_raw: np.ndarray,
    gsam_rgb: np.ndarray,
    map_channels: np.ndarray,
    max_depth: float,
    title: str,
):
    depth_vis = depth_to_vis(depth_raw, max_depth)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.reshape(2, 4)
    axes[0, 0].imshow(rgb_model)
    axes[0, 0].set_title("RGB (model)")
    axes[0, 1].imshow(depth_vis, cmap=DEPTH_CMAP, vmin=0.0, vmax=1.0)
    axes[0, 1].set_title("Depth (model)")
    axes[0, 2].imshow(gsam_rgb)
    axes[0, 2].set_title("RGB (map update)")
    axes[0, 3].imshow(map_channels[0], cmap=MAP_CMAP, vmin=0.0, vmax=1.0)
    axes[0, 3].set_title(MAP_TITLES[0])

    for idx, channel in enumerate(map_channels[1:], start=1):
        row = 1
        col = idx - 1
        axes[row, col].imshow(channel, cmap=MAP_CMAP, vmin=0.0, vmax=1.0)
        axes[row, col].set_title(MAP_TITLES[idx])

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    cli, args = parse_cli()
    summary_path = Path(cli.summary_json)
    summary = json.loads(summary_path.read_text())

    selected = summary["selected_episode"]
    episode = load_episode(
        split=summary["split"],
        altitude=args.altitude,
        map_name=selected["map_name"],
        object_id=selected["object_id"],
        description_id=selected["description_id"],
    )
    poses = [Pose4D(*pose) for pose in summary["trajectory"]]
    step_index = select_step_index(poses, cli.step_index)
    pose = poses[step_index]

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
    for obs_pose in poses[: step_index + 1]:
        gsam_rgb_step = cropclient.crop_image(episode.map_name, obs_pose, args.gsam_rgb_shape, "rgb")
        nav_map.update_observations(obs_pose, gsam_rgb_step, None, args.gsam_use_map_cache)

    rgb_model = cropclient.crop_model_image(episode.map_name, pose, "rgb")
    depth_raw = cropclient.crop_model_image(episode.map_name, pose, "depth")[..., 0]
    gsam_rgb = cropclient.crop_image(episode.map_name, pose, args.gsam_rgb_shape, "rgb")
    map_channels = nav_map.to_array()

    step_tag = f"step{step_index + 1:02d}_of_{len(poses):02d}"
    output_dir = Path(cli.output_dir) / f"{summary_path.stem}_{step_tag}_obs"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.imsave(output_dir / "01_rgb_model.png", rgb_model)
    save_depth_png(output_dir / "02_depth_model.png", depth_raw, args.max_depth)
    plt.imsave(output_dir / "03_rgb_map_update.png", gsam_rgb)
    for idx, (title, channel) in enumerate(zip(MAP_TITLES, map_channels), start=4):
        plt.imsave(output_dir / f"{idx:02d}_map_{title}.png", channel, cmap=MAP_CMAP, vmin=0.0, vmax=1.0)

    np.save(output_dir / "depth_model.npy", depth_raw.astype(np.float32))
    np.save(output_dir / "map_channels.npy", map_channels.astype(np.float32))

    title = (
        f"{episode.map_name} | obj={selected['object_id']} desc={selected['description_id']} | "
        f"rollout step {step_index + 1}/{len(poses)}"
    )
    save_combined_panel(
        output_dir / "00_combined_panel.png",
        rgb_model=rgb_model,
        depth_raw=depth_raw,
        gsam_rgb=gsam_rgb,
        map_channels=map_channels,
        max_depth=args.max_depth,
        title=title,
    )

    info_lines = [
        f"summary_json: {summary_path}",
        f"output_dir: {output_dir}",
        f"episode_id: {episode.id}",
        f"selected_step_index_0_based: {step_index}",
        f"selected_step_1_based: {step_index + 1}",
        f"rollout_length: {len(poses)}",
        f"pose_xyzyaw: {tuple(float(v) for v in pose)}",
        "selection_rule: last non-repeated rollout pose" if cli.step_index < 0 else "selection_rule: user-specified step_index",
        "map_channels: 0=track_view, 1=track_explored, 2=landmarks, 3=target, 4=surroundings",
        f"rgb_model_shape: {tuple(int(v) for v in rgb_model.shape)}",
        f"depth_model_shape: {tuple(int(v) for v in depth_raw.shape)}",
        f"gsam_rgb_shape: {tuple(int(v) for v in gsam_rgb.shape)}",
        f"map_shape: {tuple(int(v) for v in map_channels.shape)}",
    ]
    (output_dir / "metadata.txt").write_text("\n".join(info_lines) + "\n")

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "selected_step_index_0_based": step_index,
                "selected_step_1_based": step_index + 1,
                "pose_xyzyaw": [float(v) for v in pose],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
