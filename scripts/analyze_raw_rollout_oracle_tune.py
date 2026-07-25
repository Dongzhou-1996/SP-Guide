from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Iterable

import numpy as np

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gsamllavanav.cityreferobject import get_city_refer_objects
from gsamllavanav.dataset.generate import generate_episodes_from_mturk_trajectories
from gsamllavanav.dataset.mturk_trajectory import load_mturk_trajectories


MAX_MOVE_PER_ROLLOUT_STEP_M = 25.0


def _parse_floats(value: str) -> list[float]:
    return [float(v) for v in value.split(",") if v]


def _parse_ints(value: str) -> list[int]:
    return [int(v) for v in value.split(",") if v]


def _xy(point) -> np.ndarray:
    return np.asarray(point[:2], dtype=np.float64)


def _dist(a, b) -> float:
    return float(np.linalg.norm(_xy(a) - _xy(b)))


def _quantiles(values: Iterable[float], qs=(0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {str(q): float("nan") for q in qs}
    return {str(q): float(np.quantile(arr, q)) for q in qs}


def _softmax_weights(progresses: np.ndarray, beta: float) -> np.ndarray:
    logits = beta * (progresses - progresses.max())
    weights = np.exp(logits)
    return weights / max(float(weights.sum()), 1e-9)


def _window_stats(goals: np.ndarray, progresses: np.ndarray, beta: float) -> dict[str, float | np.ndarray]:
    weights = _softmax_weights(progresses, beta)
    mean_goal = (weights[:, None] * goals).sum(axis=0)
    dists = np.linalg.norm(goals - mean_goal[None, :], axis=1)
    variance = float((weights * dists**2).sum())
    return {
        "mean_goal": mean_goal,
        "variance": variance,
        "sigma": float(math.sqrt(max(variance, 0.0))),
        "endpoint_dist": float(np.linalg.norm(goals[-1] - mean_goal)),
        "weighted_progress": float((weights * progresses).sum()),
        "progress_std": float(progresses.std()),
    }


def _episode_lookup(split: str, altitude: float):
    objects = get_city_refer_objects()
    episodes = generate_episodes_from_mturk_trajectories(
        objects,
        load_mturk_trajectories(split, "all", altitude),
    )
    return {str(ep.id): ep for ep in episodes}


def _load_records(raw_json: Path, split: str, altitude: float):
    data = json.load(open(raw_json))
    episodes = _episode_lookup(split, altitude)
    records = []
    missing = 0
    for eps_id, traj in data["trajectory_logs"].items():
        ep = episodes.get(eps_id)
        if ep is None:
            missing += 1
            continue
        goals = data["pred_goal_logs"].get(eps_id, [])
        progresses = data["pred_progress_logs"].get(eps_id, [])
        n = min(len(traj), len(goals), len(progresses))
        if n == 0:
            continue
        traj_xy = np.asarray([_xy(p) for p in traj[:n]], dtype=np.float64)
        pred_goals = np.asarray([_xy(g) for g in goals[:n]], dtype=np.float64)
        pred_progress = np.asarray(progresses[:n], dtype=np.float64)
        target = np.asarray([ep.target_position.x, ep.target_position.y], dtype=np.float64)
        start = np.asarray([ep.start_pose.x, ep.start_pose.y], dtype=np.float64)
        pose_dists = np.linalg.norm(traj_xy - target[None, :], axis=1)
        pred_dists = np.linalg.norm(pred_goals - target[None, :], axis=1)
        start_dist = max(float(np.linalg.norm(start - target)), 1e-6)
        true_progress = 1.0 - np.minimum(pose_dists / start_dist, 1.0)
        oracle_step = int(np.argmin(pose_dists))
        records.append(
            {
                "eps_id": eps_id,
                "target": target,
                "start_dist": start_dist,
                "traj_xy": traj_xy,
                "pred_goals": pred_goals,
                "pred_progress": pred_progress,
                "true_progress": true_progress,
                "pose_dists": pose_dists,
                "pred_dists": pred_dists,
                "oracle_step": oracle_step,
                "oracle_dist": float(pose_dists[oracle_step]),
                "final_dist": float(pose_dists[-1]),
                "final_pred_dist": float(pred_dists[-1]),
            }
        )
    if missing:
        print(f"[warn] missing {missing} episode ids from split={split}")
    return data, records


def _summarize_raw(records: list[dict], success_dist: float) -> dict[str, float]:
    oracle_success = [r["oracle_dist"] <= success_dist for r in records]
    final_success = [r["final_dist"] <= success_dist for r in records]
    pred_success = [r["final_pred_dist"] <= success_dist for r in records]
    lost = [o and not f for o, f in zip(oracle_success, final_success)]
    return {
        "episodes": len(records),
        "final_sr": float(np.mean(final_success)),
        "oracle_sr": float(np.mean(oracle_success)),
        "final_pred_sr": float(np.mean(pred_success)),
        "lost_after_oracle_rate": float(np.mean(lost)),
        "lost_after_oracle_count": int(sum(lost)),
        "mean_final_dist": float(np.mean([r["final_dist"] for r in records])),
        "mean_oracle_dist": float(np.mean([r["oracle_dist"] for r in records])),
        "mean_final_pred_dist": float(np.mean([r["final_pred_dist"] for r in records])),
    }


def _offset_stats(records: list[dict], offsets: range, success_dist: float) -> list[dict[str, float | int]]:
    rows = []
    for offset in offsets:
        pose_dist = []
        pred_dist = []
        pred_progress = []
        true_progress = []
        valid = 0
        for r in records:
            idx = r["oracle_step"] + offset
            if idx < 0 or idx >= len(r["pose_dists"]):
                continue
            valid += 1
            pose_dist.append(float(r["pose_dists"][idx]))
            pred_dist.append(float(r["pred_dists"][idx]))
            pred_progress.append(float(r["pred_progress"][idx]))
            true_progress.append(float(r["true_progress"][idx]))
        rows.append(
            {
                "offset": offset,
                "n": valid,
                "pose_dist_mean": mean(pose_dist) if pose_dist else float("nan"),
                "pose_sr": float(np.mean(np.asarray(pose_dist) <= success_dist)) if pose_dist else float("nan"),
                "pred_dist_mean": mean(pred_dist) if pred_dist else float("nan"),
                "pred_sr": float(np.mean(np.asarray(pred_dist) <= success_dist)) if pred_dist else float("nan"),
                "pred_progress_mean": mean(pred_progress) if pred_progress else float("nan"),
                "true_progress_mean": mean(true_progress) if true_progress else float("nan"),
            }
        )
    return rows


def _oracle_window_reference(records: list[dict], windows: list[int], beta: float):
    rows = []
    for window in windows:
        variances = []
        weighted_progresses = []
        lock_goal_dists = []
        endpoint_dists = []
        for r in records:
            end = r["oracle_step"] + 1
            start = end - window
            if start < 0:
                continue
            stats = _window_stats(r["pred_goals"][start:end], r["pred_progress"][start:end], beta)
            variances.append(float(stats["variance"]))
            weighted_progresses.append(float(stats["weighted_progress"]))
            endpoint_dists.append(float(stats["endpoint_dist"]))
            lock_goal_dists.append(float(np.linalg.norm(stats["mean_goal"] - r["target"])))
        rows.append(
            {
                "window": window,
                "n": len(variances),
                "variance_q50": _quantiles(variances, (0.5,))["0.5"],
                "variance_q75": _quantiles(variances, (0.75,))["0.75"],
                "variance_q90": _quantiles(variances, (0.9,))["0.9"],
                "weighted_progress_q25": _quantiles(weighted_progresses, (0.25,))["0.25"],
                "weighted_progress_q50": _quantiles(weighted_progresses, (0.5,))["0.5"],
                "lock_goal_dist_mean": mean(lock_goal_dists) if lock_goal_dists else float("nan"),
                "lock_goal_sr": float(np.mean(np.asarray(lock_goal_dists) <= 20.0)) if lock_goal_dists else float("nan"),
                "endpoint_dist_q75": _quantiles(endpoint_dists, (0.75,))["0.75"],
            }
        )
    return rows


def _sweep(records: list[dict], args) -> list[dict[str, float | int | str]]:
    rows = []
    max_steps = max(len(r["pose_dists"]) for r in records)
    for window in args.windows:
        for beta in args.betas:
            for progress_thr in args.progress_thrs:
                for variance_thr in args.variance_thrs:
                    for min_step in args.min_steps:
                        for arrival_dist in args.arrival_dists:
                            locked = 0
                            proxy_success = 0
                            pred_success = 0
                            oracle_aligned = 0
                            late_after_oracle = 0
                            lock_steps = []
                            lock_goal_dists = []
                            current_to_lock_dists = []
                            for r in records:
                                lock_step = None
                                lock_goal = None
                                n = min(len(r["pred_goals"]), len(r["pred_progress"]), len(r["traj_xy"]))
                                for t in range(n):
                                    if t < min_step:
                                        continue
                                    start = t - window + 1
                                    if start < 0:
                                        continue
                                    stats = _window_stats(
                                        r["pred_goals"][start : t + 1],
                                        r["pred_progress"][start : t + 1],
                                        beta,
                                    )
                                    if (
                                        float(stats["weighted_progress"]) >= progress_thr
                                        and float(stats["variance"]) <= variance_thr
                                    ):
                                        lock_step = t
                                        lock_goal = stats["mean_goal"]
                                        break
                                if lock_step is None or lock_goal is None:
                                    continue
                                locked += 1
                                lock_steps.append(lock_step)
                                goal_dist = float(np.linalg.norm(lock_goal - r["target"]))
                                cur_dist = float(np.linalg.norm(r["traj_xy"][lock_step] - lock_goal))
                                lock_goal_dists.append(goal_dist)
                                current_to_lock_dists.append(cur_dist)
                                pred_ok = goal_dist <= args.success_dist
                                reachable = cur_dist <= (max_steps - 1 - lock_step) * MAX_MOVE_PER_ROLLOUT_STEP_M + arrival_dist
                                pred_success += int(pred_ok)
                                proxy_success += int(pred_ok and reachable)
                                oracle_aligned += int(abs(lock_step - int(r["oracle_step"])) <= args.oracle_step_tolerance)
                                late_after_oracle += int(lock_step > int(r["oracle_step"]) + args.oracle_step_tolerance)
                            total = max(len(records), 1)
                            rows.append(
                                {
                                    "config": f"w{window}_b{beta:g}_p{progress_thr:g}_v{variance_thr:g}_m{min_step}_a{arrival_dist:g}",
                                    "window": window,
                                    "beta": beta,
                                    "progress_thr": progress_thr,
                                    "variance_thr": variance_thr,
                                    "min_step": min_step,
                                    "arrival_dist": arrival_dist,
                                    "lock_rate": locked / total,
                                    "proxy_sr": proxy_success / total,
                                    "lock_goal_sr": pred_success / total,
                                    "oracle_aligned_rate": oracle_aligned / total,
                                    "late_after_oracle_rate": late_after_oracle / total,
                                    "mean_lock_step": mean(lock_steps) if lock_steps else float("nan"),
                                    "median_lock_step": float(np.median(lock_steps)) if lock_steps else float("nan"),
                                    "mean_lock_goal_dist": mean(lock_goal_dists) if lock_goal_dists else float("inf"),
                                    "mean_current_to_lock_dist": mean(current_to_lock_dists) if current_to_lock_dists else float("inf"),
                                }
                            )
    rows.sort(
        key=lambda r: (
            float(r["proxy_sr"]),
            float(r["lock_goal_sr"]),
            -abs(float(r["lock_rate"]) - 0.75),
            -float(r["mean_lock_goal_dist"]),
        ),
        reverse=True,
    )
    return rows


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _write_md(path: Path, raw_summary, step_q, offset_rows, oracle_rows, sweep_rows):
    lines = []
    lines.append("# Raw 009 Oracle-Step Tuning Report")
    lines.append("")
    lines.append("## Raw Rollout Metrics")
    lines.append("")
    lines.append(f"- Episodes: {raw_summary['episodes']}")
    lines.append(f"- Final SR: {_pct(raw_summary['final_sr'])}")
    lines.append(f"- Oracle SR: {_pct(raw_summary['oracle_sr'])}")
    lines.append(f"- Final Pred-SR: {_pct(raw_summary['final_pred_sr'])}")
    lines.append(f"- Lost after oracle success: {raw_summary['lost_after_oracle_count']} ({_pct(raw_summary['lost_after_oracle_rate'])})")
    lines.append(f"- Mean final/oracle/pred distance: {raw_summary['mean_final_dist']:.2f} / {raw_summary['mean_oracle_dist']:.2f} / {raw_summary['mean_final_pred_dist']:.2f}")
    lines.append("")
    lines.append("## Oracle Step Distribution")
    lines.append("")
    lines.append("| q0 | q10 | q25 | q50 | q75 | q90 | q100 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    lines.append(
        "| "
        + " | ".join(
            f"{step_q[str(q)]:.1f}"
            for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        )
        + " |"
    )
    lines.append("")
    lines.append("## Statistics Around Oracle Step")
    lines.append("")
    lines.append("| offset | n | pose SR | pred SR | pose dist | pred dist | pred progress | true progress |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in offset_rows:
        lines.append(
            f"| {r['offset']} | {r['n']} | {_pct(float(r['pose_sr']))} | {_pct(float(r['pred_sr']))} | "
            f"{float(r['pose_dist_mean']):.2f} | {float(r['pred_dist_mean']):.2f} | "
            f"{float(r['pred_progress_mean']):.3f} | {float(r['true_progress_mean']):.3f} |"
        )
    lines.append("")
    lines.append("## Oracle-Centered Window Reference")
    lines.append("")
    lines.append("| window | n | var q50 | var q75 | var q90 | progress q25 | progress q50 | lock-goal SR | lock-goal dist | endpoint q75 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in oracle_rows:
        lines.append(
            f"| {r['window']} | {r['n']} | {r['variance_q50']:.1f} | {r['variance_q75']:.1f} | "
            f"{r['variance_q90']:.1f} | {r['weighted_progress_q25']:.3f} | {r['weighted_progress_q50']:.3f} | "
            f"{_pct(r['lock_goal_sr'])} | {r['lock_goal_dist_mean']:.2f} | {r['endpoint_dist_q75']:.2f} |"
        )
    lines.append("")
    lines.append("## Top Proxy GCF Parameters")
    lines.append("")
    lines.append("| rank | config | proxy SR | lock-goal SR | lock rate | mean lock step | lock-goal dist | oracle aligned | late after oracle |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for idx, r in enumerate(sweep_rows[:20], start=1):
        lines.append(
            f"| {idx} | `{r['config']}` | {_pct(float(r['proxy_sr']))} | {_pct(float(r['lock_goal_sr']))} | "
            f"{_pct(float(r['lock_rate']))} | {float(r['mean_lock_step']):.2f} | "
            f"{float(r['mean_lock_goal_dist']):.2f} | {_pct(float(r['oracle_aligned_rate']))} | "
            f"{_pct(float(r['late_after_oracle_rate']))} |"
        )
    lines.append("")
    lines.append("Note: proxy SR is only a fast filter for parameters. Final numbers must be confirmed by online rollout because locking changes future actions.")
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-json", type=Path, required=True)
    parser.add_argument("--split", choices=["val_seen", "val_unseen", "test_unseen"], required=True)
    parser.add_argument("--altitude", type=float, default=50)
    parser.add_argument("--success-dist", type=float, default=20.0)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--windows", type=_parse_ints, default="3,4,5,6,7,8,10,12")
    parser.add_argument("--betas", type=_parse_floats, default="0,2,4,6,8")
    parser.add_argument("--progress-thrs", type=_parse_floats, default="0.55,0.6,0.65,0.7,0.75,0.8,0.85")
    parser.add_argument("--variance-thrs", type=_parse_floats, default="150,250,400,600,800,1000,1400")
    parser.add_argument("--min-steps", type=_parse_ints, default="4,5,6,7,8,10")
    parser.add_argument("--arrival-dists", type=_parse_floats, default="12,15,18,20,25")
    parser.add_argument("--oracle-step-tolerance", type=int, default=2)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _, records = _load_records(args.raw_json, args.split, args.altitude)
    raw_summary = _summarize_raw(records, args.success_dist)
    step_q = _quantiles([r["oracle_step"] for r in records])
    offset_rows = _offset_stats(records, range(-4, 5), args.success_dist)
    oracle_rows = _oracle_window_reference(records, args.windows, beta=4.0)
    sweep_rows = _sweep(records, args)

    _write_csv(args.out_dir / "oracle_offset_stats.csv", offset_rows)
    _write_csv(args.out_dir / "oracle_window_reference.csv", oracle_rows)
    _write_csv(args.out_dir / "gcf_proxy_sweep.csv", sweep_rows)
    _write_md(args.out_dir / "oracle_step_stats.md", raw_summary, step_q, offset_rows, oracle_rows, sweep_rows)
    print(args.out_dir / "oracle_step_stats.md")
    for row in sweep_rows[:10]:
        print(
            f"{row['config']} proxy_sr={float(row['proxy_sr'])*100:.2f} "
            f"lock_goal_sr={float(row['lock_goal_sr'])*100:.2f} lock={float(row['lock_rate'])*100:.2f} "
            f"step={float(row['mean_lock_step']):.2f} dist={float(row['mean_lock_goal_dist']):.2f}"
        )


if __name__ == "__main__":
    main()
