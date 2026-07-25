#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from gsamllavanav.cityreferobject import get_city_refer_objects
from gsamllavanav.dataset.generate import generate_episodes_from_mturk_trajectories
from gsamllavanav.dataset.mturk_trajectory import load_mturk_trajectories
from gsamllavanav.evaluate import eval_goal_predictor
from gsamllavanav.parser import parse_args
from scripts.eval_goal_agents import compute_spl, infer_model_name, infer_split, load_rollout_json


def parse_cli():
    cli = argparse.ArgumentParser(description="Offline threshold sweep on raw progress-stop rollout logs.")
    cli.add_argument("--input_json", action="append", required=True, help="Raw rollout json path. Repeatable.")
    cli.add_argument("--thr_start", type=float, default=0.10)
    cli.add_argument("--thr_end", type=float, default=0.98)
    cli.add_argument("--thr_step", type=float, default=0.02)
    cli.add_argument("--select_split", choices=["val_seen", "val_unseen"], default="val_unseen")
    cli.add_argument("--select_metric", choices=["sr", "ne"], default="sr")
    cli.add_argument("--output_json", default="")
    cli.add_argument("--output_summary_json", default="")
    cli_args, remaining = cli.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    exp_args = parse_args()
    return cli_args, exp_args


def threshold_grid(start: float, end: float, step: float) -> list[float]:
    values = []
    cur = start
    while cur <= end + 1e-9:
        values.append(round(cur, 4))
        cur += step
    return values


def truncate_rollout_for_threshold(trajectory, pred_goals, pred_progresses, threshold: float, min_stop_step: int):
    stop_idx = None
    for idx, progress in enumerate(pred_progresses):
        if idx < min_stop_step:
            continue
        if progress >= threshold:
            stop_idx = idx
            break
    if stop_idx is None:
        return list(trajectory), list(pred_goals), list(pred_progresses), len(pred_progresses) - 1
    end = stop_idx + 1
    return list(trajectory[:end]), list(pred_goals[:end]), list(pred_progresses[:end]), stop_idx


def score_candidate(metric: str, row: dict) -> tuple:
    if metric == "sr":
        return (
            row["sr"],
            row["osr"],
            -row["ne"],
            -row["spl"],
            -row["threshold"],
        )
    return (
        -row["ne"],
        row["sr"],
        row["osr"],
        row["spl"],
        -row["threshold"],
    )


def main():
    cli_args, args = parse_cli()
    objects = get_city_refer_objects()
    episodes_cache = {}
    results = []

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

        for threshold in threshold_grid(cli_args.thr_start, cli_args.thr_end, cli_args.thr_step):
            truncated_trajectory_logs = {}
            truncated_goal_logs = {}
            truncated_progress_logs = {}
            stop_steps = []
            for eps in episodes:
                eps_id = eps.id
                traj, goals, progresses, stop_idx = truncate_rollout_for_threshold(
                    trajectory_logs[eps_id],
                    pred_goal_logs[eps_id],
                    pred_progress_logs[eps_id],
                    threshold,
                    args.min_stop_step,
                )
                truncated_trajectory_logs[eps_id] = traj
                truncated_goal_logs[eps_id] = goals
                truncated_progress_logs[eps_id] = progresses
                stop_steps.append(stop_idx)

            metrics = eval_goal_predictor(
                args,
                episodes,
                truncated_trajectory_logs,
                truncated_goal_logs,
                truncated_progress_logs,
            )
            spl = compute_spl(args, episodes, truncated_trajectory_logs)
            results.append(
                {
                    "model": model,
                    "split": split,
                    "threshold": threshold,
                    "ne": metrics.mean_final_pos_to_goal_dist,
                    "sr": metrics.success_rate_final_pos_to_goal,
                    "osr": metrics.success_rate_oracle_pos_to_goal,
                    "spl": spl,
                    "pred_ne": metrics.mean_final_pred_to_goal_dist,
                    "pred_sr": metrics.success_rate_final_pred_to_goal,
                    "progress_mse": metrics.mean_progress_mse,
                    "mean_stop_step": sum(stop_steps) / max(len(stop_steps), 1),
                }
            )

    summary = {}
    by_model = {}
    for row in results:
        by_model.setdefault(row["model"], {}).setdefault(row["split"], []).append(row)

    for model, split_rows in by_model.items():
        selector_rows = split_rows[cli_args.select_split]
        best = max(selector_rows, key=lambda row: score_candidate(cli_args.select_metric, row))
        chosen_thr = best["threshold"]
        summary[model] = {
            "selected_by": {
                "split": cli_args.select_split,
                "metric": cli_args.select_metric,
                "threshold": chosen_thr,
            },
            "results": {},
        }
        for split, rows in split_rows.items():
            chosen_row = next(row for row in rows if abs(row["threshold"] - chosen_thr) < 1e-9)
            summary[model]["results"][split] = chosen_row

    if cli_args.output_json:
        with open(cli_args.output_json, "w") as f:
            json.dump(results, f, indent=2)
    if cli_args.output_summary_json:
        with open(cli_args.output_summary_json, "w") as f:
            json.dump(summary, f, indent=2)

    for model, item in summary.items():
        chosen = item["selected_by"]
        print(
            f"{model}: best threshold={chosen['threshold']:.2f} "
            f"(selected on {chosen['split']} by {chosen['metric']})"
        )
        for split, row in item["results"].items():
            print(
                f"  {split}: NE={row['ne']:.2f} SR={row['sr']*100:.2f} "
                f"OSR={row['osr']*100:.2f} SPL={row['spl']*100:.2f}"
            )


if __name__ == "__main__":
    main()
