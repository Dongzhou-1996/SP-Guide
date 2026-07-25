from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from .geometry import Point2D, Pose4D, wrap_angle


class DiscreteAction(Enum):
    STOP = (0.0, 0.0, 0.0)
    MOVE_FORWARD = (5.0, 0.0, 0.0)
    TURN_LEFT = (0.0, math.pi / 6.0, 0.0)
    TURN_RIGHT = (0.0, -math.pi / 6.0, 0.0)
    GO_UP = (0.0, 0.0, 2.0)
    GO_DOWN = (0.0, 0.0, -2.0)

    @property
    def delta(self) -> tuple[float, float, float]:
        return self.value


@dataclass(frozen=True)
class ActionSelectorConfig:
    stop_distance: float = 5.0
    turn_threshold: float = math.pi / 6.0
    vertical_threshold: float = 1.0


class GoalDirectedActionSelector:
    """Convert a target point into the next discrete navigation action."""

    def __init__(self, config: ActionSelectorConfig | None = None):
        self.config = config or ActionSelectorConfig()

    def select(self, pose: Pose4D, goal: Point2D, goal_z: float | None = None) -> DiscreteAction:
        dz = 0.0 if goal_z is None else goal_z - pose.z
        dist_xy = pose.xy.distance_to(goal)
        if dist_xy < self.config.stop_distance and abs(dz) <= self.config.vertical_threshold:
            return DiscreteAction.STOP
        if dz > self.config.vertical_threshold:
            return DiscreteAction.GO_UP
        if dz < -self.config.vertical_threshold:
            return DiscreteAction.GO_DOWN

        target_yaw = math.atan2(goal.y - pose.y, goal.x - pose.x)
        d_yaw = wrap_angle(target_yaw - pose.yaw)
        if d_yaw > self.config.turn_threshold:
            return DiscreteAction.TURN_LEFT
        if d_yaw < -self.config.turn_threshold:
            return DiscreteAction.TURN_RIGHT
        return DiscreteAction.MOVE_FORWARD

    def step(self, pose: Pose4D, goal: Point2D, goal_z: float | None = None) -> tuple[Pose4D, DiscreteAction]:
        action = self.select(pose, goal, goal_z)
        return pose.moved(*action.delta), action

