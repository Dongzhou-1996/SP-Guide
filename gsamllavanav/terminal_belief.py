from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gsamllavanav.parser import ExperimentArgs
from gsamllavanav.space import Point2D


@dataclass(frozen=True)
class TerminalBeliefStats:
    mean_goal: Point2D | None
    sigma: float
    variance: float
    endpoint_dist: float
    weighted_progress: float
    progress_std: float
    window_size: int


def terminal_belief_config_dict(args: ExperimentArgs) -> dict[str, float | int]:
    return {
        "window": int(args.terminal_belief_window),
        "beta": float(args.terminal_belief_beta),
        "progress_thr": float(args.terminal_belief_progress_thr),
        "variance_thr": float(args.terminal_belief_variance_thr),
        "min_step": int(args.terminal_belief_min_step),
        "arrival_dist": float(args.terminal_belief_arrival_dist),
    }


def terminal_belief_rejection_reason(
    args: ExperimentArgs,
    timestep: int,
    stats: TerminalBeliefStats,
) -> str | None:
    if timestep < args.terminal_belief_min_step:
        return 'timestep'
    if stats.window_size < args.terminal_belief_window:
        return 'window'
    if stats.mean_goal is None:
        return 'empty'
    if stats.weighted_progress < args.terminal_belief_progress_thr:
        return 'progress'
    if stats.variance > args.terminal_belief_variance_thr:
        return 'variance'
    return None


def _as_goal_array(goal_xys: list[Point2D]) -> np.ndarray:
    return np.asarray([[goal.x, goal.y] for goal in goal_xys], dtype=np.float32)


def _prepare_progress_array(goal_xys: list[Point2D], progresses: list[float]) -> np.ndarray:
    progress_array = np.asarray(progresses[: len(goal_xys)], dtype=np.float32)
    if len(progress_array) < len(goal_xys):
        progress_array = np.pad(
            progress_array,
            (0, len(goal_xys) - len(progress_array)),
            mode="edge" if len(progress_array) else "constant",
        )
    return progress_array


def _weighted_goal_stats(
    goal_xys: list[Point2D],
    progress_array: np.ndarray,
    weights: np.ndarray,
) -> TerminalBeliefStats:
    if not goal_xys:
        return TerminalBeliefStats(
            mean_goal=None,
            sigma=float("inf"),
            variance=float("inf"),
            endpoint_dist=float("inf"),
            weighted_progress=0.0,
            progress_std=0.0,
            window_size=0,
    )

    goal_array = _as_goal_array(goal_xys)
    mean_xy = (weights[:, None] * goal_array).sum(axis=0)
    distances = np.linalg.norm(goal_array - mean_xy[None, :], axis=1)
    variance = float((weights * distances**2).sum())
    sigma = float(np.sqrt(variance))
    endpoint_dist = float(np.linalg.norm(goal_array[-1] - mean_xy))

    return TerminalBeliefStats(
        mean_goal=Point2D(float(mean_xy[0]), float(mean_xy[1])),
        sigma=sigma,
        variance=variance,
        endpoint_dist=endpoint_dist,
        weighted_progress=float((weights * progress_array).sum()) if len(progress_array) else 0.0,
        progress_std=float(progress_array.std()) if len(progress_array) else 0.0,
        window_size=len(goal_xys),
    )


def _progress_weighted_goal_stats(
    goal_xys: list[Point2D],
    progresses: list[float],
    beta: float,
) -> TerminalBeliefStats:
    if not goal_xys:
        return _weighted_goal_stats(goal_xys, np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32))

    progress_array = _prepare_progress_array(goal_xys, progresses)
    logits = beta * (progress_array - progress_array.max()) if len(progress_array) else np.zeros((len(goal_xys),), dtype=np.float32)
    weights = np.exp(logits)
    weights = weights / max(float(weights.sum()), 1e-6)
    return _weighted_goal_stats(goal_xys, progress_array, weights)


def terminal_belief_stats(
    args: ExperimentArgs,
    goal_xys: list[Point2D],
    progresses: list[float],
) -> TerminalBeliefStats:
    return _progress_weighted_goal_stats(goal_xys, progresses, beta=args.terminal_belief_beta)


def is_confident_terminal_belief(
    args: ExperimentArgs,
    timestep: int,
    stats: TerminalBeliefStats,
) -> bool:
    return terminal_belief_rejection_reason(args, timestep, stats) is None


def fit_terminal_goal_belief_window(
    args: ExperimentArgs,
    timestep: int,
    goal_xys: list[Point2D],
    progresses: list[float],
) -> Point2D | None:
    stats = terminal_belief_stats(args, goal_xys, progresses)
    if not is_confident_terminal_belief(args, timestep, stats):
        return None
    return stats.mean_goal


def select_terminal_goal_from_logs(
    args: ExperimentArgs,
    goal_xys: list[Point2D],
    progresses: list[float],
) -> Point2D | None:
    history_len = min(len(goal_xys), len(progresses))
    if history_len < max(args.terminal_belief_window, args.terminal_belief_min_step):
        return None

    goal_xys = goal_xys[:history_len]
    progresses = progresses[:history_len]
    for end in range(history_len, args.terminal_belief_min_step - 1, -1):
        start = end - args.terminal_belief_window
        if start < 0:
            break
        window_goals = goal_xys[start:end]
        window_progresses = progresses[start:end]
        stats = terminal_belief_stats(args, window_goals, window_progresses)
        if is_confident_terminal_belief(args, end - 1, stats):
            return stats.mean_goal
    return None
