from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gsamllavanav.observation import cropclient
from gsamllavanav.space import Point2D
import gsamllavanav.terminal_belief as terminal_belief
from scripts import eval_single_split


def _hett_crop_model_image(map_name, pose, image_type):
    if image_type == "rgb":
        return cropclient.crop_image(map_name, pose, (224, 224), "rgb")
    if image_type == "depth":
        return cropclient.crop_image(map_name, pose, (256, 256), "depth")
    raise ValueError(f"Unsupported HETT-style model crop type: {image_type}")


def _goal_array(goal_xys):
    return np.asarray([[goal.x, goal.y] for goal in goal_xys], dtype=np.float32)


def _hett_terminal_belief_stats(args, goal_xys, progresses):
    if not goal_xys:
        return terminal_belief.TerminalBeliefStats(
            mean_goal=None,
            sigma=float("inf"),
            variance=float("inf"),
            endpoint_dist=float("inf"),
            weighted_progress=0.0,
            progress_std=0.0,
            window_size=0,
        )

    progress_array = np.asarray(progresses[: len(goal_xys)], dtype=np.float32)
    logits = args.terminal_belief_beta * (progress_array - progress_array.max())
    weights = np.exp(logits)
    weights = weights / max(float(weights.sum()), 1e-6)

    goals = _goal_array(goal_xys)
    mean_xy = (weights[:, None] * goals).sum(axis=0)
    distances = np.linalg.norm(goals - mean_xy[None, :], axis=1)
    variance = float((weights * distances**2).sum())
    sigma = float(np.sqrt(variance))
    endpoint_dist = float(np.linalg.norm(goals[-1] - mean_xy))
    return terminal_belief.TerminalBeliefStats(
        mean_goal=Point2D(float(mean_xy[0]), float(mean_xy[1])),
        sigma=sigma,
        variance=variance,
        endpoint_dist=endpoint_dist,
        weighted_progress=float((weights * progress_array).sum()),
        progress_std=float(progress_array.std()),
        window_size=len(goal_xys),
    )


def _hett_terminal_belief_rejection_reason(args, timestep, stats):
    if timestep < args.terminal_belief_min_step:
        return "timestep"
    if stats.window_size < args.terminal_belief_window:
        return "window"
    if stats.mean_goal is None:
        return "empty"
    if stats.weighted_progress < args.terminal_belief_progress_thr:
        return "progress"
    sigma_thr = float(np.sqrt(args.terminal_belief_variance_thr))
    if stats.sigma > sigma_thr:
        return "sigma"
    endpoint_thr = 25.0
    if stats.endpoint_dist > endpoint_thr:
        return "endpoint"
    return None


def _hett_fit_terminal_goal_belief_window(args, timestep, goal_xys, progresses):
    stats = _hett_terminal_belief_stats(args, goal_xys, progresses)
    if _hett_terminal_belief_rejection_reason(args, timestep, stats) is not None:
        return None
    return stats.mean_goal


def main():
    cropclient.crop_model_image = _hett_crop_model_image
    terminal_belief.terminal_belief_stats = _hett_terminal_belief_stats
    terminal_belief.terminal_belief_rejection_reason = _hett_terminal_belief_rejection_reason
    terminal_belief.fit_terminal_goal_belief_window = _hett_fit_terminal_goal_belief_window
    eval_single_split.main()


if __name__ == "__main__":
    main()
