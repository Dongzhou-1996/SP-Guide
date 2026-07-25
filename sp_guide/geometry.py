from __future__ import annotations

from dataclasses import dataclass
import math


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float

    def distance_to(self, other: "Point2D") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float = 0.0


@dataclass(frozen=True)
class Pose4D:
    x: float
    y: float
    z: float = 0.0
    yaw: float = 0.0

    @property
    def xy(self) -> Point2D:
        return Point2D(self.x, self.y)

    @property
    def xyz(self) -> Point3D:
        return Point3D(self.x, self.y, self.z)

    def moved(self, forward: float, d_yaw: float, dz: float) -> "Pose4D":
        yaw = wrap_angle(self.yaw + d_yaw)
        return Pose4D(
            x=self.x + forward * math.cos(yaw),
            y=self.y + forward * math.sin(yaw),
            z=self.z + dz,
            yaw=yaw,
        )

