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
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

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
from gsamllavanav.terminal_belief import select_terminal_goal_from_logs


SPLITS = ("val_seen", "val_unseen", "test_unseen")


@dataclass(frozen=True)
class SweepConfig:
    method: str
    window: int = 5
    beta: float = 4.0
    progress_thr: float = 0.80
    variance_thr: float = 400.0
    min_step: int = 8

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "method": self.method,
            "window": self.window,
            "beta": self.beta,
            "progress_thr": self.progress_thr,
            "variance_thr": self.variance_thr,
            "min_step": self.min_step,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline sweep for terminal rollout backends.")
    parser.add_argument(
        "--input_glob",
        action="append",
        required=True,
        help="Glob(s) for rollout JSON files. Repeatable.",
    )
    parser.add_argument(
        "--output_csv",
        default="docs/terminal_backend_sweep.csv",
        help="CSV path to write all results.",
    )
    parser.add_argument(
        "--summary_csv",
        default="docs/terminal_backend_sweep_summary.csv",
        help="CSV path to write per-config comparison summary.",
    )
    parser.add_argument("--altitude", type=float, default=50.0)
    parser.add_argument("--success_dist", type=float, default=20.0)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["raw", "gcf"],
    )
    parser.add_argument("--windows", type=int, nargs="+", default=[3, 5, 7, 9])
    parser.add_argument("--betas", type=float, nargs="+", default=[2.0, 4.0, 8.0, 12.0])
    parser.add_argument("--progress_thrs", type=float, nargs="+", default=[0.80])
    parser.add_argument("--variance_thrs", type=float, nargs="+", default=[400.0])
    parser.add_argument("--min_steps", type=int, nargs="+", default=[8])
    parser.add_argument("--target_model", default="instr_decoder_usc_with_map")
    parser.add_argument("--baseline_model", default="instr_decoder_with_map")
    parser.add_argument("--include_test", action="store_true", default=False)
    return parser.parse_args()


def infer_model_name(path: Path) -> str:
    name = path.name
    token = "_mturk_50.0_0.2_"
    if token not in name:
        raise ValueError(f"Cannot infer model from filename: {name}")
    return name.split(token)[0]


def infer_split(path: Path) -> str:
    for split in SPLITS:
        if f"_{split}_" in path.name:
            return split
    raise ValueError(f"Cannot infer split from filename: {path.name}")


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


def tuple_key(key: str) -> tuple[str, int, int]:
    map_name, obj_id, desc_id = ast.literal_eval(key)
    return map_name, obj_id, desc_id


def select_goal(config: SweepConfig, pred_goals: list[Point2D], pred_progresses: list[float]) -> Point2D | None:
    if config.method == "raw":
        return None

    if not pred_goals:
        return None

    history_len = min(len(pred_goals), len(pred_progresses)) if pred_progresses else len(pred_goals)
    if history_len <= 0:
        return None

    pred_goals = pred_goals[:history_len]
    pred_progresses = pred_progresses[:history_len] if pred_progresses else [0.0] * history_len

    if config.method == "gcf":
        terminal_args = SimpleNamespace(
            terminal_belief_window=config.window,
            terminal_belief_beta=config.beta,
            terminal_belief_progress_thr=config.progress_thr,
            terminal_belief_variance_thr=config.variance_thr,
            terminal_belief_min_step=config.min_step,
            terminal_belief_arrival_dist=0.0,
        )
        return select_terminal_goal_from_logs(terminal_args, pred_goals, pred_progresses)

    raise ValueError(f"Unknown method: {config.method}")


def build_episodes(altitude: float) -> dict[str, list]:
    objects = get_city_refer_objects()
    episodes = {}
    for split in SPLITS:
        episodes[split] = generate_episodes_from_mturk_trajectories(
            objects,
            load_mturk_trajectories(split, "all", altitude),
        )
    return episodes


def iter_configs(args: argparse.Namespace) -> Iterable[SweepConfig]:
    yielded: set[SweepConfig] = set()
    for method in args.methods:
        if method == "raw":
            config = SweepConfig(method=method)
            if config not in yielded:
                yielded.add(config)
                yield config
            continue

        if method == "gcf":
            for window, beta, progress_thr, variance_thr, min_step in product(
                args.windows,
                args.betas,
                args.progress_thrs,
                args.variance_thrs,
                args.min_steps,
            ):
                config = SweepConfig(
                    method=method,
                    window=window,
                    beta=beta,
                    progress_thr=progress_thr,
                    variance_thr=variance_thr,
                    min_step=min_step,
                )
                if config not in yielded:
                    yielded.add(config)
                    yield config
            continue
        raise ValueError(f"Unknown method: {method}")


def make_args(success_dist: float) -> SimpleNamespace:
    return SimpleNamespace(success_dist=success_dist)


def append_selected_pose(
    trajectory_logs: dict,
    pred_goal_logs: dict,
    pred_progress_logs: dict,
    config: SweepConfig,
) -> dict:
    updated = {}
    for eps_id, trajectory in trajectory_logs.items():
        copied = list(trajectory)
        pred_goals = pred_goal_logs.get(eps_id, [])
        pred_progresses = pred_progress_logs.get(eps_id, [])
        selected = select_goal(config, pred_goals, pred_progresses)
        if selected is not None and copied:
            last_pose = copied[-1]
            copied.append(Pose4D(selected.x, selected.y, last_pose.z, last_pose.yaw))
        updated[eps_id] = copied
    return updated


def config_id(config: SweepConfig) -> str:
    return "|".join(f"{k}={v}" for k, v in config.to_dict().items())


def main() -> None:
    args = parse_args()

    input_paths: list[Path] = []
    for pattern in args.input_glob:
        input_paths.extend(Path(p) for p in glob.glob(pattern))
    input_paths = sorted(set(input_paths))
    if not input_paths:
        raise SystemExit("No input JSON files matched.")

    wanted_splits = {"val_seen", "val_unseen"}
    if args.include_test:
        wanted_splits.add("test_unseen")

    rollouts = []
    for path in input_paths:
        split = infer_split(path)
        if split not in wanted_splits:
            continue
        model_name = infer_model_name(path)
        trajectory_logs, pred_goal_logs, pred_progress_logs = load_rollout_json(path)
        rollouts.append(
            {
                "path": str(path),
                "model": model_name,
                "split": split,
                "trajectory_logs": trajectory_logs,
                "pred_goal_logs": pred_goal_logs,
                "pred_progress_logs": pred_progress_logs,
            }
        )
    if not rollouts:
        raise SystemExit("No rollout files left after split filtering.")

    episodes_by_split = build_episodes(args.altitude)
    eval_args = make_args(args.success_dist)

    for rollout in rollouts:
        available_ids = (
            set(rollout["trajectory_logs"])
            & set(rollout["pred_goal_logs"])
            & set(rollout["pred_progress_logs"])
        )
        rollout["episodes"] = [
            eps for eps in episodes_by_split[rollout["split"]]
            if eps.id in available_ids
        ]

    rows: list[dict[str, object]] = []
    for config in iter_configs(args):
        cfg_id = config_id(config)
        for rollout in rollouts:
            updated_trajectory_logs = append_selected_pose(
                rollout["trajectory_logs"],
                rollout["pred_goal_logs"],
                rollout["pred_progress_logs"],
                config,
            )
            metrics = eval_goal_predictor(
                eval_args,
                rollout["episodes"],
                updated_trajectory_logs,
                rollout["pred_goal_logs"],
                rollout["pred_progress_logs"],
            )
            row = {
                "config_id": cfg_id,
                "model": rollout["model"],
                "split": rollout["split"],
                "path": rollout["path"],
                **config.to_dict(),
                **metrics.to_dict(),
            }
            rows.append(row)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = summarize_rows(rows, args.target_model, args.baseline_model)
    summary_csv = Path(args.summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    best_shared = [
        row for row in summary_rows
        if row["shared_val_splits"] >= 2 and row["target_val_sr_avg"] > row["baseline_val_sr_avg"]
    ]
    best_shared.sort(
        key=lambda row: (
            row["target_val_sr_avg"] - row["baseline_val_sr_avg"],
            row["target_val_sr_avg"],
            -row["baseline_val_sr_avg"],
        ),
        reverse=True,
    )

    print(f"Wrote {len(rows)} per-file rows to {output_csv}")
    print(f"Wrote {len(summary_rows)} summary rows to {summary_csv}")
    if best_shared:
        print("Top shared config where target beats baseline on validation average:")
        for row in best_shared[:10]:
            print(json.dumps(row, ensure_ascii=False))
    else:
        print("No shared validation config found where target beats baseline average.")


def summarize_rows(rows: list[dict[str, object]], target_model: str, baseline_model: str) -> list[dict[str, object]]:
    grouped: dict[str, dict[tuple[str, str], dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(row["config_id"], {})[(row["model"], row["split"])] = row

    summary_rows = []
    for cfg_id, metrics_by_key in grouped.items():
        config = {k: v for k, v in next(iter(metrics_by_key.values())).items() if k in SweepConfig("raw").to_dict()}

        target_val_sr = []
        baseline_val_sr = []
        target_test_sr = math.nan
        baseline_test_sr = math.nan
        shared_val_splits = 0
        per_split = {}
        for split in ("val_seen", "val_unseen", "test_unseen"):
            target = metrics_by_key.get((target_model, split))
            baseline = metrics_by_key.get((baseline_model, split))
            if target is not None:
                per_split[f"target_{split}_sr"] = target["success_rate_final_pos_to_goal"]
                per_split[f"target_{split}_pred_sr"] = target["success_rate_final_pred_to_goal"]
            if baseline is not None:
                per_split[f"baseline_{split}_sr"] = baseline["success_rate_final_pos_to_goal"]
                per_split[f"baseline_{split}_pred_sr"] = baseline["success_rate_final_pred_to_goal"]
            if split.startswith("val") and target is not None and baseline is not None:
                shared_val_splits += 1
                target_val_sr.append(target["success_rate_final_pos_to_goal"])
                baseline_val_sr.append(baseline["success_rate_final_pos_to_goal"])
            if split == "test_unseen":
                if target is not None:
                    target_test_sr = target["success_rate_final_pos_to_goal"]
                if baseline is not None:
                    baseline_test_sr = baseline["success_rate_final_pos_to_goal"]

        target_val_sr_avg = float(np.mean(target_val_sr)) if target_val_sr else math.nan
        baseline_val_sr_avg = float(np.mean(baseline_val_sr)) if baseline_val_sr else math.nan
        summary_rows.append(
            {
                "config_id": cfg_id,
                **config,
                "shared_val_splits": shared_val_splits,
                "target_val_sr_avg": target_val_sr_avg,
                "baseline_val_sr_avg": baseline_val_sr_avg,
                "target_minus_baseline_val_sr": (
                    target_val_sr_avg - baseline_val_sr_avg
                    if not math.isnan(target_val_sr_avg) and not math.isnan(baseline_val_sr_avg)
                    else math.nan
                ),
                "target_test_sr": target_test_sr,
                "baseline_test_sr": baseline_test_sr,
                **per_split,
            }
        )
    summary_rows.sort(
        key=lambda row: (
            row["target_minus_baseline_val_sr"]
            if not math.isnan(row["target_minus_baseline_val_sr"])
            else -1e9,
            row["target_val_sr_avg"] if not math.isnan(row["target_val_sr_avg"]) else -1e9,
        ),
        reverse=True,
    )
    return summary_rows


if __name__ == "__main__":
    main()
