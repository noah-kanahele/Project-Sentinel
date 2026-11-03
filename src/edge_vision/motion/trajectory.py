"""Position history, velocity estimate and predicted arc."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class TrajectoryConfig:
    history_length: int
    velocity_window: int
    arc_horizon_s: float
    arc_steps: int


class TrajectoryBuffer:
    """Fixed-length history of (timestamp, x, y) points."""

    def __init__(self, history_length: int, velocity_window: int) -> None:
        self._points: Deque[Tuple[float, float, float]] = deque(maxlen=history_length)
        self._velocity_window = velocity_window

    def add(self, timestamp_s: float, x: float, y: float) -> None:
        self._points.append((timestamp_s, x, y))

    def clear(self) -> None:
        self._points.clear()

    def __len__(self) -> int:
        return len(self._points)

    def points(self) -> List[Tuple[float, float]]:
        """Stored (x, y) centers, oldest first."""
        return [(x, y) for _, x, y in self._points]

    def velocity(self) -> Optional[Tuple[float, float]]:
        """Least-squares velocity (px/s) over the last `velocity_window` points, or None."""
        recent = list(self._points)[-self._velocity_window:]
        if len(recent) < 2:
            return None
        t = np.array([p[0] for p in recent])
        if t[-1] - t[0] <= 0:
            return None
        vx = np.polyfit(t, [p[1] for p in recent], 1)[0]
        vy = np.polyfit(t, [p[2] for p in recent], 1)[0]
        return (float(vx), float(vy))


def predicted_arc(
    position: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    horizon_s: float,
    steps: int,
) -> List[Tuple[float, float]]:
    """Constant-acceleration positions at `steps` evenly spaced times up to `horizon_s`."""
    arc = []
    for k in range(1, steps + 1):
        t = horizon_s * k / steps
        p = position + velocity * t + 0.5 * acceleration * t * t
        arc.append((float(p[0]), float(p[1])))
    return arc
