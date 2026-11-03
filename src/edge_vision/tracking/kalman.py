"""Constant-acceleration Kalman filter. State [x, y, vx, vy, ax, ay], measurement [x, y]."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class KalmanConfig:
    process_noise_jerk: float  # white-jerk spectral density, px^2/s^5
    measurement_noise_px: float
    initial_velocity_std_px_s: float
    initial_accel_std_px_s2: float
    gravity_prior_px_s2: float  # initial ay guess (image y points down)


class KalmanFilter:
    """Tracks one point. `predict(dt)` advances the state; `update(z)` folds in a measurement."""

    def __init__(self, cfg: KalmanConfig, xy: Tuple[float, float]) -> None:
        self.cfg = cfg
        self._x = np.zeros(6)
        self._x[0], self._x[1] = xy
        self._x[5] = cfg.gravity_prior_px_s2
        r2 = cfg.measurement_noise_px ** 2
        self._P = np.diag(
            [
                r2, r2,
                cfg.initial_velocity_std_px_s ** 2, cfg.initial_velocity_std_px_s ** 2,
                cfg.initial_accel_std_px_s2 ** 2, cfg.initial_accel_std_px_s2 ** 2,
            ]
        )
        self._H = np.zeros((2, 6))
        self._H[0, 0] = self._H[1, 1] = 1.0
        self._R = np.eye(2) * r2

    @staticmethod
    def _transition(dt: float) -> np.ndarray:
        F = np.eye(6)
        for axis in (0, 1):
            F[axis, axis + 2] = dt
            F[axis, axis + 4] = 0.5 * dt * dt
            F[axis + 2, axis + 4] = dt
        return F

    def _process_noise(self, dt: float) -> np.ndarray:
        """Continuous white-jerk noise, discretised over dt, per axis."""
        q = self.cfg.process_noise_jerk
        block = q * np.array(
            [
                [dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
                [dt ** 4 / 8, dt ** 3 / 3, dt ** 2 / 2],
                [dt ** 3 / 6, dt ** 2 / 2, dt],
            ]
        )
        Q = np.zeros((6, 6))
        for axis in (0, 1):
            idx = [axis, axis + 2, axis + 4]
            Q[np.ix_(idx, idx)] = block
        return Q

    def predict(self, dt: float) -> None:
        """Advance the state by dt seconds (dt <= 0 is a no-op)."""
        if dt <= 0:
            return
        F = self._transition(dt)
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + self._process_noise(dt)

    def update(self, z: Tuple[float, float]) -> None:
        """Fold in a position measurement (Joseph-form covariance update)."""
        innovation = np.asarray(z, dtype=float) - self._H @ self._x
        S = self._H @ self._P @ self._H.T + self._R
        K = self._P @ self._H.T @ np.linalg.inv(S)
        self._x = self._x + K @ innovation
        I_KH = np.eye(6) - K @ self._H
        self._P = I_KH @ self._P @ I_KH.T + K @ self._R @ K.T

    @property
    def position(self) -> np.ndarray:
        return self._x[0:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self._x[2:4].copy()

    @property
    def acceleration(self) -> np.ndarray:
        return self._x[4:6].copy()

    @property
    def covariance(self) -> np.ndarray:
        return self._P.copy()

    def position_sigma(self) -> float:
        """1-sigma uncertainty (px) of where a measurement would land: sqrt of the larger axis of H P H' + R."""
        S = self._H @ self._P @ self._H.T + self._R
        return float(np.sqrt(max(S[0, 0], S[1, 1])))

    def project(self, dt: float) -> np.ndarray:
        """Position dt seconds ahead of the current state, without changing the filter."""
        return self._x[0:2] + self._x[2:4] * dt + 0.5 * self._x[4:6] * dt * dt
