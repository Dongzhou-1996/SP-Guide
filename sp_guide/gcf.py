from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .geometry import Point2D


@dataclass(frozen=True)
class GCFConfig:
    window: int = 5
    beta: float = 4.0
    progress_threshold: float = 0.80
    variance_threshold: float = 400.0
    min_step: int = 8


@dataclass(frozen=True)
class GCFState:
    locked_goal: Point2D | None
    mean_goal: Point2D | None
    variance: float
    weighted_progress: float
    lock_step: int | None


class GoalConvergenceFilter:
    """Fit a terminal goal belief from recent goal and progress predictions."""

    def __init__(self, config: GCFConfig | None = None):
        self.config = config or GCFConfig()
        self.goals: list[Point2D] = []
        self.progresses: list[float] = []
        self.locked_goal: Point2D | None = None
        self.lock_step: int | None = None

    def update(self, goal: Point2D, progress: float) -> GCFState:
        self.goals.append(goal)
        self.progresses.append(float(progress))
        step = len(self.goals) - 1

        if self.locked_goal is None:
            state = self._fit(step)
            if self._should_lock(step, state):
                self.locked_goal = state.mean_goal
                self.lock_step = step
        return self.state

    @property
    def state(self) -> GCFState:
        state = self._fit(len(self.goals) - 1)
        return GCFState(
            locked_goal=self.locked_goal,
            mean_goal=state.mean_goal,
            variance=state.variance,
            weighted_progress=state.weighted_progress,
            lock_step=self.lock_step,
        )

    @property
    def target(self) -> Point2D | None:
        return self.locked_goal

    def _fit(self, step: int) -> GCFState:
        cfg = self.config
        if not self.goals:
            return GCFState(None, None, math.inf, 0.0, self.lock_step)

        goals = self.goals[-cfg.window:]
        progress = np.asarray(self.progresses[-len(goals):], dtype=np.float32)
        xy = np.asarray([[g.x, g.y] for g in goals], dtype=np.float32)

        logits = cfg.beta * (progress - progress.max())
        weights = np.exp(logits)
        weights = weights / max(float(weights.sum()), 1e-6)
        mean_xy = (weights[:, None] * xy).sum(axis=0)
        dist2 = ((xy - mean_xy[None, :]) ** 2).sum(axis=1)
        variance = float((weights * dist2).sum())
        weighted_progress = float((weights * progress).sum())

        return GCFState(
            locked_goal=self.locked_goal,
            mean_goal=Point2D(float(mean_xy[0]), float(mean_xy[1])),
            variance=variance,
            weighted_progress=weighted_progress,
            lock_step=self.lock_step if step >= 0 else None,
        )

    def _should_lock(self, step: int, state: GCFState) -> bool:
        cfg = self.config
        if step < cfg.min_step:
            return False
        if len(self.goals) < cfg.window:
            return False
        if state.mean_goal is None:
            return False
        if state.weighted_progress < cfg.progress_threshold:
            return False
        return state.variance <= cfg.variance_threshold

