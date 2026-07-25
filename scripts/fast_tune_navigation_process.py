#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import glob
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


SPLITS = ("val_seen", "val_unseen", "test_unseen")


@dataclass(frozen=True)
class Row:
    source_json: str
    model: str
    split: str
    stop_thr: float
    backend: str
    window: int
    beta: float
    progress_thr: float
    variance_thr: float
    min_step: int
    ne: float
    sr: float
    osr: float
    spl: float
    mean_steps: float
    truncated_ratio: float


def tuple_key(key: str) -> tuple[str, int, int]:
    return ast.literal_eval(key)


def infer_model(path: Path) -> str:
    return path.name.split("_mturk_50.0_0.2_")[0]


def infer_split(path: Path) -> str:
    for split in SPLITS:
        if f"_{split}_" in path.name:
            return split
    raise ValueError(path)


def load_json(path: Path):
    obj = json.load(open(path))
    trajectories = {
        tuple_key(k): np.asarray([[p[0], p[1]] for p in poses], dtype=np.float32)
        for k, poses in obj["trajectory_logs"].items()
    }
    goals = {
        tuple_key(k): np.asarray(goals, dtype=np.float32)
        for k, goals in obj["pred_goal_logs"].items()
    }
    progresses = {
        tuple_key(k): np.asarray(progresses, dtype=np.float32)
        for k, progresses in obj["pred_progress_logs"].items()
    }
    return trajectories, goals, progresses


def build_episode_meta(altitude: float):
    objects = get_city_refer_objects()
    meta = {}
    for split in SPLITS:
        episodes = generate_episodes_from_mturk_trajectories(
            objects,
            load_mturk_trajectories(split, "all", altitude),
        )
        meta[split] = {
            eps.id: {
                "target": np.asarray([eps.target_position.x, eps.target_position.y], dtype=np.float32),
                "start": np.asarray([eps.start_pose.x, eps.start_pose.y], dtype=np.float32),
            }
            for eps in episodes
        }
    return meta


def path_prefix_lengths(xy: np.ndarray) -> np.ndarray:
    if len(xy) == 0:
        return np.zeros((0,), dtype=np.float32)
    if len(xy) == 1:
        return np.zeros((1,), dtype=np.float32)
    steps = np.linalg.norm(xy[1:] - xy[:-1], axis=1)
    return np.concatenate([np.zeros((1,), dtype=np.float32), np.cumsum(steps)])


def stop_index(progress: np.ndarray, max_len: int, thr: float) -> int:
    n = min(len(progress), max_len)
    if n <= 0:
        return -1
    hit = np.flatnonzero(progress[:n] >= thr)
    return int(hit[0]) if len(hit) else n - 1


def select_goal(
    goals: np.ndarray,
    progress: np.ndarray,
    keep: int,
    backend: str,
    window: int,
    beta: float,
    progress_thr: float,
    variance_thr: float,
    min_step: int,
):
    if backend == "natural":
        return None
    if keep - 1 < min_step:
        return None
    active_goals = goals[:keep]
    active_progress = progress[:keep]
    if len(active_goals) < window:
        return None
    window_goals = active_goals[-window:]
    window_progress = active_progress[-len(window_goals):]
    if backend == "gcf":
        logits = beta * (window_progress - np.max(window_progress))
        weights = np.exp(logits)
        weights = weights / max(float(weights.sum()), 1e-6)
        weighted_progress = float((weights * window_progress).sum())
        mean = (weights[:, None] * window_goals).sum(axis=0)
        variance = float((weights * np.linalg.norm(window_goals - mean[None, :], axis=1) ** 2).sum())
        if weighted_progress < progress_thr or variance > variance_thr:
            return None
        return mean
    raise ValueError(backend)


def evaluate_one(
    meta,
    trajectories,
    goals,
    progresses,
    success_dist: float,
    stop_thr: float,
    backend: str,
    window: int,
    beta: float,
    progress_thr: float,
    variance_thr: float,
    min_step: int,
):
    final_dists = []
    oracle_dists = []
    spl_values = []
    steps = []
    truncated = 0
    total = 0
    for eps_id, target_meta in meta.items():
        if eps_id not in trajectories or eps_id not in goals or eps_id not in progresses:
            continue
        traj = trajectories[eps_id]
        pred = goals[eps_id]
        prog = progresses[eps_id]
        max_len = min(len(traj), len(pred), len(prog))
        idx = stop_index(prog, max_len, stop_thr)
        if idx < 0:
            continue
        keep = idx + 1
        total += 1
        truncated += int(keep < max_len)
        steps.append(keep)
        target = target_meta["target"]
        start = target_meta["start"]
        base_traj = traj[:keep]
        selected = select_goal(pred, prog, keep, backend, window, beta, progress_thr, variance_thr, min_step)
        if selected is None:
            final_xy = base_traj[-1]
            eval_traj = base_traj
        else:
            final_xy = selected
            eval_traj = np.concatenate([base_traj, selected[None, :]], axis=0)

        final_dist = float(np.linalg.norm(final_xy - target))
        final_dists.append(final_dist)
        oracle_dists.append(float(np.linalg.norm(eval_traj - target[None, :], axis=1).min()))

        prefix = path_prefix_lengths(base_traj)
        path_len = float(prefix[-1]) if len(prefix) else 0.0
        if selected is not None and len(base_traj):
            path_len += float(np.linalg.norm(selected - base_traj[-1]))
        shortest = float(np.linalg.norm(target - start))
        success = float(final_dist <= success_dist)
        spl_values.append(success * shortest / max(path_len, shortest, 1e-6))

    final_dists = np.asarray(final_dists, dtype=np.float32)
    oracle_dists = np.asarray(oracle_dists, dtype=np.float32)
    return {
        "ne": float(final_dists.mean()),
        "sr": float((final_dists <= success_dist).mean()),
        "osr": float((oracle_dists <= success_dist).mean()),
        "spl": float(np.mean(spl_values)),
        "mean_steps": float(np.mean(steps)),
        "truncated_ratio": truncated / max(total, 1),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_glob", action="append", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--altitude", type=float, default=50.0)
    parser.add_argument("--success_dist", type=float, default=20.0)
    parser.add_argument("--stop_thrs", type=float, nargs="+", default=[0.65, 0.70, 0.75])
    parser.add_argument("--backends", nargs="+", default=["natural", "gcf"])
    parser.add_argument("--windows", type=int, nargs="+", default=[3, 4, 5, 6, 7])
    parser.add_argument("--betas", type=float, nargs="+", default=[2.0, 4.0, 6.0, 8.0])
    parser.add_argument("--progress_thrs", type=float, nargs="+", default=[0.80])
    parser.add_argument("--variance_thrs", type=float, nargs="+", default=[400.0])
    parser.add_argument("--min_steps", type=int, nargs="+", default=[8])
    return parser.parse_args()


def main():
    args = parse_args()
    paths = sorted({Path(p) for pattern in args.input_glob for p in glob.glob(pattern)})
    if not paths:
        raise SystemExit("No input JSON matched.")
    meta_by_split = build_episode_meta(args.altitude)
    rows: list[Row] = []
    for path in paths:
        model = infer_model(path)
        split = infer_split(path)
        trajectories, goals, progresses = load_json(path)
        for stop_thr in args.stop_thrs:
            for backend in args.backends:
                windows = args.windows if backend == "gcf" else [0]
                betas = args.betas if backend == "gcf" else [0.0]
                progress_thrs = args.progress_thrs if backend == "gcf" else [0.0]
                variance_thrs = args.variance_thrs if backend == "gcf" else [float("inf")]
                min_steps = args.min_steps if backend == "gcf" else [0]
                for window in windows:
                    for beta in betas:
                        for progress_thr in progress_thrs:
                            for variance_thr in variance_thrs:
                                for min_step in min_steps:
                                    metrics = evaluate_one(
                                        meta_by_split[split],
                                        trajectories,
                                        goals,
                                        progresses,
                                        args.success_dist,
                                        stop_thr,
                                        backend,
                                        window,
                                        beta,
                                        progress_thr,
                                        variance_thr,
                                        min_step,
                                    )
                                    rows.append(
                                        Row(
                                            str(path),
                                            model,
                                            split,
                                            stop_thr,
                                            backend,
                                            window,
                                            beta,
                                            progress_thr,
                                            variance_thr,
                                            min_step,
                                            **metrics,
                                        )
                                    )

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)

    print(f"Wrote {len(rows)} rows to {out}")
    for row in sorted(rows, key=lambda r: (r.sr, r.spl, -r.ne), reverse=True)[:20]:
        print(
            f"{row.model} {row.split} stop={row.stop_thr:.2f} {row.backend} "
            f"w={row.window} b={row.beta:g} p={row.progress_thr:g} v={row.variance_thr:g} m={row.min_step} NE={row.ne:.2f} "
            f"SR={row.sr*100:.2f} OSR={row.osr*100:.2f} SPL={row.spl*100:.2f} "
            f"steps={row.mean_steps:.2f} trunc={row.truncated_ratio*100:.1f}%"
        )


if __name__ == "__main__":
    main()
