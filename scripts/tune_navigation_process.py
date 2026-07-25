#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import glob
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from gsamllavanav.cityreferobject import get_city_refer_objects
from gsamllavanav.dataset.generate import generate_episodes_from_mturk_trajectories
from gsamllavanav.dataset.mturk_trajectory import load_mturk_trajectories
from gsamllavanav.evaluate import eval_goal_predictor
from gsamllavanav.space import Point2D, Pose4D
from scripts.sweep_terminal_backends import SweepConfig, select_goal


SPLITS = ("val_seen", "val_unseen", "test_unseen")


@dataclass(frozen=True)
class NavigationTuneRow:
    source_json: str
    model: str
    split: str
    stop_thr: float
    backend: str
    window: int
    beta: float
    ne: float
    sr: float
    osr: float
    spl: float
    pred_ne: float
    pred_sr: float
    pred_osr: float
    mean_steps: float
    truncated_ratio: float


def tuple_key(key: str) -> tuple[str, int, int]:
    map_name, obj_id, desc_id = ast.literal_eval(key)
    return map_name, obj_id, desc_id


def infer_model_name(path: Path) -> str:
    token = "_mturk_50.0_0.2_"
    if token not in path.name:
        raise ValueError(f"Cannot infer model name from {path}")
    return path.name.split(token)[0]


def infer_split(path: Path) -> str:
    for split in SPLITS:
        if f"_{split}_" in path.name:
            return split
    raise ValueError(f"Cannot infer split from {path}")


def load_rollout(path: Path):
    with open(path, "r") as f:
        obj = json.load(f)
    trajectory_logs = {
        tuple_key(k): [Pose4D(*pose) for pose in poses]
        for k, poses in obj["trajectory_logs"].items()
    }
    pred_goal_logs = {
        tuple_key(k): [Point2D(*xy) for xy in goals]
        for k, goals in obj["pred_goal_logs"].items()
    }
    pred_progress_logs = {
        tuple_key(k): list(progresses)
        for k, progresses in obj["pred_progress_logs"].items()
    }
    return trajectory_logs, pred_goal_logs, pred_progress_logs


def planar_path_length_xy(poses: list[Pose4D]) -> float:
    if len(poses) < 2:
        return 0.0
    xy = np.asarray([[pose.x, pose.y] for pose in poses], dtype=np.float32)
    return float(np.linalg.norm(xy[1:] - xy[:-1], axis=1).sum())


def shortest_path_length_xy(episode) -> float:
    start_xy = np.asarray([episode.start_pose.x, episode.start_pose.y], dtype=np.float32)
    goal_xy = np.asarray([episode.target_position.x, episode.target_position.y], dtype=np.float32)
    return float(np.linalg.norm(goal_xy - start_xy))


def compute_spl(args, episodes, trajectory_logs: dict[tuple[str, int, int], list[Pose4D]]) -> float:
    values = []
    for episode in episodes:
        trajectory = trajectory_logs[episode.id]
        success = float(trajectory[-1].xy.dist_to(episode.target_position.xy) <= args.success_dist)
        shortest = shortest_path_length_xy(episode)
        path_len = planar_path_length_xy(trajectory)
        values.append(success * shortest / max(path_len, shortest, 1e-6))
    return float(np.mean(values)) if values else 0.0


def truncate_by_progress(
    trajectory_logs: dict,
    pred_goal_logs: dict,
    pred_progress_logs: dict,
    stop_thr: float,
):
    new_trajectories = {}
    new_goals = {}
    new_progresses = {}
    steps = []
    truncated = 0
    total = 0
    for eps_id, goals in pred_goal_logs.items():
        progresses = pred_progress_logs.get(eps_id, [])
        trajectory = trajectory_logs.get(eps_id, [])
        if not goals or not progresses or not trajectory:
            continue
        n = min(len(goals), len(progresses), len(trajectory))
        stop_idx = n - 1
        for idx, progress in enumerate(progresses[:n]):
            if progress >= stop_thr:
                stop_idx = idx
                break
        keep = stop_idx + 1
        total += 1
        truncated += int(keep < n)
        steps.append(keep)
        new_trajectories[eps_id] = trajectory[:keep]
        new_goals[eps_id] = goals[:keep]
        new_progresses[eps_id] = progresses[:keep]
    return new_trajectories, new_goals, new_progresses, float(np.mean(steps)), truncated / max(total, 1)


def append_selected_goal(trajectory_logs, pred_goal_logs, pred_progress_logs, backend: str, window: int, beta: float):
    if backend == "natural":
        return {eps_id: list(trajectory) for eps_id, trajectory in trajectory_logs.items()}

    if backend == "gcf":
        cfg = SweepConfig(method="gcf", window=window, beta=beta)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    updated = {}
    for eps_id, trajectory in trajectory_logs.items():
        copied = list(trajectory)
        selected = select_goal(cfg, pred_goal_logs.get(eps_id, []), pred_progress_logs.get(eps_id, []))
        if selected is not None and copied:
            last_pose = copied[-1]
            copied.append(Pose4D(selected.x, selected.y, last_pose.z, last_pose.yaw))
        updated[eps_id] = copied
    return updated


def build_episodes(altitude: float):
    objects = get_city_refer_objects()
    return {
        split: generate_episodes_from_mturk_trajectories(
            objects,
            load_mturk_trajectories(split, "all", altitude),
        )
        for split in SPLITS
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Tune progress-stop and terminal backend from saved rollout JSONs.")
    parser.add_argument("--input_glob", action="append", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--altitude", type=float, default=50.0)
    parser.add_argument("--success_dist", type=float, default=20.0)
    parser.add_argument("--stop_thrs", type=float, nargs="+", default=[0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75])
    parser.add_argument("--backends", nargs="+", default=["natural", "gcf"])
    parser.add_argument("--windows", type=int, nargs="+", default=[3, 4, 5, 6, 7, 8, 9])
    parser.add_argument("--betas", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0])
    return parser.parse_args()


def main():
    args = parse_args()
    paths = []
    for pattern in args.input_glob:
        paths.extend(Path(p) for p in glob.glob(pattern))
    paths = sorted(set(paths))
    if not paths:
        raise SystemExit("No rollout JSON matched.")

    eval_args = SimpleNamespace(success_dist=args.success_dist)
    episodes_by_split = build_episodes(args.altitude)
    rows: list[NavigationTuneRow] = []

    for path in paths:
        model = infer_model_name(path)
        split = infer_split(path)
        trajectory_logs, pred_goal_logs, pred_progress_logs = load_rollout(path)
        available_ids = set(trajectory_logs) & set(pred_goal_logs) & set(pred_progress_logs)
        episodes = [episode for episode in episodes_by_split[split] if episode.id in available_ids]

        for stop_thr in args.stop_thrs:
            t_traj, t_goals, t_progresses, mean_steps, truncated_ratio = truncate_by_progress(
                trajectory_logs,
                pred_goal_logs,
                pred_progress_logs,
                stop_thr,
            )
            t_episodes = [episode for episode in episodes if episode.id in t_traj]
            for backend in args.backends:
                windows = args.windows if backend == "gcf" else [0]
                betas = args.betas if backend == "gcf" else [0.0]
                for window in windows:
                    for beta in betas:
                        updated_traj = append_selected_goal(
                            t_traj,
                            t_goals,
                            t_progresses,
                            backend,
                            window,
                            beta,
                        )
                        metrics = eval_goal_predictor(
                            eval_args,
                            t_episodes,
                            updated_traj,
                            t_goals,
                            t_progresses,
                        )
                        rows.append(
                            NavigationTuneRow(
                                source_json=str(path),
                                model=model,
                                split=split,
                                stop_thr=stop_thr,
                                backend=backend,
                                window=window,
                                beta=beta,
                                ne=metrics.mean_final_pos_to_goal_dist,
                                sr=metrics.success_rate_final_pos_to_goal,
                                osr=metrics.success_rate_oracle_pos_to_goal,
                                spl=compute_spl(eval_args, t_episodes, updated_traj),
                                pred_ne=metrics.mean_final_pred_to_goal_dist,
                                pred_sr=metrics.success_rate_final_pred_to_goal,
                                pred_osr=metrics.success_rate_oracle_pred_to_goal,
                                mean_steps=mean_steps,
                                truncated_ratio=truncated_ratio,
                            )
                        )

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    print(f"Wrote {len(rows)} rows to {out}")
    for row in sorted(rows, key=lambda r: (r.sr, r.spl, -r.ne), reverse=True)[:20]:
        print(
            f"{row.model} {row.split} stop={row.stop_thr:.2f} {row.backend} "
            f"w={row.window} b={row.beta:g} NE={row.ne:.2f} "
            f"SR={row.sr*100:.2f} OSR={row.osr*100:.2f} SPL={row.spl*100:.2f} "
            f"steps={row.mean_steps:.2f}"
        )


if __name__ == "__main__":
    main()
