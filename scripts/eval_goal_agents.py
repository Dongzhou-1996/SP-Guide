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
from gsamllavanav.goal_selection import goal_selection_gcf, goal_selection_gdino
from gsamllavanav.parser import parse_args
from gsamllavanav.space import Point2D, Pose4D


SPLITS = ("val_seen", "val_unseen", "test_unseen")


@dataclass(frozen=True)
class AgentMetrics:
    model: str
    split: str
    selector: str
    ne: float
    sr: float
    osr: float
    spl: float


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


def parse_cli():
    cli = argparse.ArgumentParser(description="Offline agent evaluation for goal-predictor rollout JSONs.")
    cli.add_argument("--input_json", action="append", required=True, help="Raw rollout JSON path. Repeatable.")
    cli.add_argument("--selectors", nargs="+", default=["gdino", "gcf"])
    cli.add_argument("--gcf_window", type=int, default=5)
    cli.add_argument("--gcf_beta", type=float, default=4.0)
    cli.add_argument("--gcf_progress_thr", type=float, default=0.80)
    cli.add_argument("--gcf_variance_thr", type=float, default=400.0)
    cli.add_argument("--gcf_min_step", type=int, default=8)
    cli.add_argument("--gcf_arrival_dist", type=float, default=15.0)
    cli.add_argument("--output_csv", default="")
    cli.add_argument("--output_json", default="")
    args, remaining = cli.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    exp_args = parse_args()
    return args, exp_args


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


def compute_spl(args, episodes, trajectory_logs) -> float:
    values = []
    for episode in episodes:
        trajectory = trajectory_logs[episode.id]
        final_xy = trajectory[-1].xy
        success = 1.0 if final_xy.dist_to(episode.target_position.xy) <= args.success_dist else 0.0
        shortest = shortest_path_length_xy(episode)
        path_len = planar_path_length_xy(trajectory)
        denom = max(path_len, shortest, 1e-6)
        values.append(success * (shortest / denom))
    return float(np.mean(values)) if values else 0.0


def append_selected_pose(base_trajectory_logs, predicted_positions):
    updated = {eps_id: list(trajectory) for eps_id, trajectory in base_trajectory_logs.items()}
    for eps_id, pose in predicted_positions.items():
        updated[eps_id].append(pose)
    return updated


def selector_output(args, selector_name, pred_goal_logs, pred_progress_logs):
    if selector_name == "raw":
        return None
    if selector_name == "gdino":
        return goal_selection_gdino(args, pred_goal_logs)
    if selector_name == "gcf":
        return goal_selection_gcf(args, pred_goal_logs, pred_progress_logs)
    raise ValueError(f"Unknown selector: {selector_name}")


def main():
    cli_args, args = parse_cli()
    for field in (
        "gcf_window",
        "gcf_beta",
        "gcf_progress_thr",
        "gcf_variance_thr",
        "gcf_min_step",
        "gcf_arrival_dist",
    ):
        setattr(args, field, getattr(cli_args, field))
    args.terminal_belief_window = cli_args.gcf_window
    args.terminal_belief_beta = cli_args.gcf_beta
    args.terminal_belief_progress_thr = cli_args.gcf_progress_thr
    args.terminal_belief_variance_thr = cli_args.gcf_variance_thr
    args.terminal_belief_min_step = cli_args.gcf_min_step
    args.terminal_belief_arrival_dist = cli_args.gcf_arrival_dist
    wanted_selectors = tuple(cli_args.selectors)
    episodes_cache = {}
    objects = get_city_refer_objects()
    rows: list[AgentMetrics] = []

    for raw_path_str in cli_args.input_json:
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

        for selector_name in wanted_selectors:
            predicted_positions = selector_output(args, selector_name, pred_goal_logs, pred_progress_logs)
            if predicted_positions is None:
                selected_trajectories = trajectory_logs
            else:
                selected_trajectories = append_selected_pose(trajectory_logs, predicted_positions)

            metrics = eval_goal_predictor(
                args,
                episodes,
                selected_trajectories,
                pred_goal_logs,
                pred_progress_logs,
            )
            spl = compute_spl(args, episodes, selected_trajectories)
            rows.append(
                AgentMetrics(
                    model=model,
                    split=split,
                    selector=selector_name,
                    ne=metrics.mean_final_pos_to_goal_dist,
                    sr=metrics.success_rate_final_pos_to_goal,
                    osr=metrics.success_rate_oracle_pos_to_goal,
                    spl=spl,
                )
            )

    if cli_args.output_csv:
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
            f"{row.model} | {row.selector} | {row.split} | "
            f"NE={row.ne:.2f} SR={row.sr*100:.2f} OSR={row.osr*100:.2f} SPL={row.spl*100:.2f}"
        )


if __name__ == "__main__":
    main()
