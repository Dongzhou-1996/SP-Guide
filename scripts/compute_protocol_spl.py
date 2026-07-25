#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from gsamllavanav.cityreferobject import get_city_refer_objects
from gsamllavanav.dataset.generate import generate_episodes_from_mturk_trajectories
from gsamllavanav.dataset.mturk_trajectory import load_mturk_trajectories
from gsamllavanav.evaluate import eval_goal_predictor
from gsamllavanav.parser import parse_args as parse_exp_args
from gsamllavanav.space import Point2D, Pose4D
from scripts.sweep_terminal_backends import SweepConfig, select_goal


SPLITS = ("val_seen", "val_unseen", "test_unseen")


@dataclass(frozen=True)
class ProtocolRow:
    model: str
    split: str
    protocol: str
    source_json: str
    ne: float
    sr: float
    osr: float
    spl: float
    pred_ne: float
    pred_sr: float
    pred_osr: float


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


def load_rollout_json(path: Path):
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


def point_to_pose(point: Point2D) -> Pose4D:
    return Pose4D(point.x, point.y, 0.0, 0.0)


def append_goal_pose(trajectory_logs, selected_goals):
    updated = {eps_id: list(trajectory) for eps_id, trajectory in trajectory_logs.items()}
    for eps_id, goal in selected_goals.items():
        if goal is not None:
            updated[eps_id].append(point_to_pose(goal))
    return updated


def select_raw_pred(pred_goal_logs, pred_progress_logs, window: int, beta: float):
    del pred_progress_logs, window, beta
    return {
        eps_id: goals[-1]
        for eps_id, goals in pred_goal_logs.items()
        if goals
    }


def select_gcf(
    pred_goal_logs,
    pred_progress_logs,
    window: int,
    beta: float,
    progress_thr: float,
    variance_thr: float,
    min_step: int,
):
    cfg = SweepConfig(
        method="gcf",
        window=window,
        beta=beta,
        progress_thr=progress_thr,
        variance_thr=variance_thr,
        min_step=min_step,
    )
    selected = {}
    for eps_id, goals in pred_goal_logs.items():
        goal = select_goal(cfg, goals, pred_progress_logs.get(eps_id, []))
        if goal is not None:
            selected[eps_id] = goal
    return selected


SELECTORS = {
    "raw_pred": select_raw_pred,
    "gcf": select_gcf,
}


def parse_cli():
    parser = argparse.ArgumentParser(description="Compute paper-protocol NE/SR/OSR/SPL from rollout JSONs.")
    parser.add_argument("--item", action="append", required=True, help="protocol:path, e.g. raw_pred:file.json")
    parser.add_argument("--gcf_window", type=int, default=5)
    parser.add_argument("--gcf_beta", type=float, default=4.0)
    parser.add_argument("--gcf_progress_thr", type=float, default=0.80)
    parser.add_argument("--gcf_variance_thr", type=float, default=400.0)
    parser.add_argument("--gcf_min_step", type=int, default=8)
    parser.add_argument("--allow_missing", action="store_true", help="Evaluate only episodes present in the rollout.")
    parser.add_argument("--output_csv", default="")
    parser.add_argument("--output_json", default="")
    cli_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    exp_args = parse_exp_args()
    return cli_args, exp_args


def main():
    cli_args, args = parse_cli()
    objects = get_city_refer_objects()
    episodes_cache = {}
    rows: list[ProtocolRow] = []

    for item in cli_args.item:
        protocol, raw_path_str = item.split(":", 1)
        if protocol not in SELECTORS:
            raise ValueError(f"Unknown protocol {protocol}; valid: {sorted(SELECTORS)}")
        raw_path = Path(raw_path_str)
        split = infer_split(raw_path)
        model = infer_model_name(raw_path)
        if split not in episodes_cache:
            episodes_cache[split] = generate_episodes_from_mturk_trajectories(
                objects,
                load_mturk_trajectories(split, "all", args.altitude),
            )
        episodes = episodes_cache[split]
        trajectory_logs, pred_goal_logs, pred_progress_logs = load_rollout_json(raw_path)
        if cli_args.allow_missing:
            common_ids = set(trajectory_logs) & set(pred_goal_logs)
            episodes = [episode for episode in episodes if episode.id in common_ids]
        if protocol == "gcf":
            selected_goals = select_gcf(
                pred_goal_logs,
                pred_progress_logs,
                cli_args.gcf_window,
                cli_args.gcf_beta,
                cli_args.gcf_progress_thr,
                cli_args.gcf_variance_thr,
                cli_args.gcf_min_step,
            )
        else:
            selected_goals = SELECTORS[protocol](
                pred_goal_logs,
                pred_progress_logs,
                cli_args.gcf_window,
                cli_args.gcf_beta,
            )
        selected_trajectories = append_goal_pose(trajectory_logs, selected_goals)
        metrics = eval_goal_predictor(
            args,
            episodes,
            selected_trajectories,
            pred_goal_logs,
            pred_progress_logs,
        )
        rows.append(
            ProtocolRow(
                model=model,
                split=split,
                protocol=protocol,
                source_json=str(raw_path),
                ne=metrics.mean_final_pos_to_goal_dist,
                sr=metrics.success_rate_final_pos_to_goal,
                osr=metrics.success_rate_oracle_pos_to_goal,
                spl=compute_spl(args, episodes, selected_trajectories),
                pred_ne=metrics.mean_final_pred_to_goal_dist,
                pred_sr=metrics.success_rate_final_pred_to_goal,
                pred_osr=metrics.success_rate_oracle_pred_to_goal,
            )
        )

    if cli_args.output_csv and rows:
        output_csv = Path(cli_args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))

    if cli_args.output_json:
        output_json = Path(cli_args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump([asdict(row) for row in rows], f, indent=2)

    for row in rows:
        print(
            f"{row.model} | {row.protocol} | {row.split} | "
            f"NE={row.ne:.2f} SR={row.sr * 100:.2f} "
            f"OSR={row.osr * 100:.2f} SPL={row.spl * 100:.2f}"
        )


if __name__ == "__main__":
    main()
