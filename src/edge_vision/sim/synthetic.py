"""Synthetic ball trajectory and detection generator (fully seeded)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from edge_vision.detection.detector import Detection

BALL_CLASS_ID = 32  # COCO "sports ball"; synthetic false positives share it so they are adoptable.


@dataclass(frozen=True)
class SyntheticConfig:
    fps: float
    num_frames: int
    frame_width: int
    frame_height: int
    initial_position: Tuple[float, float]
    initial_velocity: Tuple[float, float]
    gravity: float
    timestamp_jitter_s: float
    center_noise_px: float
    size_noise_px: float
    ball_size_px: float
    confidence_range: Tuple[float, float]
    occlusions: Tuple[Tuple[int, int], ...]
    false_positive_rate: float
    false_positive_confidence_range: Tuple[float, float]
    false_positive_size_range: Tuple[float, float]
    miss_probability: float
    seed: int


@dataclass
class SimFrame:
    """One synthetic frame: what the detector would output plus the truth."""

    frame_idx: int
    timestamp_s: float
    detections: List[Detection]
    gt_center: Tuple[float, float]
    occluded: bool  # inside a configured occlusion window
    true_detected: bool  # a (noisy) true detection is present in `detections`


@dataclass
class SyntheticSequence:
    frames: List[SimFrame]
    occlusions: List[Tuple[int, int]]  # (start_frame, length) clipped to the sequence
    config: SyntheticConfig

    def gt_centers(self) -> np.ndarray:
        return np.array([f.gt_center for f in self.frames])


def _box(cx: float, cy: float, w: float, h: float) -> Tuple[float, float, float, float]:
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def generate_sequence(cfg: SyntheticConfig) -> SyntheticSequence:
    """Generate a projectile sequence with noise, occlusions, dropouts and false positives."""
    rng = np.random.default_rng(cfg.seed)

    # Capture times: nominal grid plus optional jitter, forced strictly increasing.
    times = np.arange(cfg.num_frames) / cfg.fps
    if cfg.timestamp_jitter_s > 0:
        times = times + rng.normal(0.0, cfg.timestamp_jitter_s, cfg.num_frames)
        min_step = 1e-6
        for i in range(1, cfg.num_frames):
            times[i] = max(times[i], times[i - 1] + min_step)
        times -= times[0]

    occluded = np.zeros(cfg.num_frames, dtype=bool)
    clipped = []
    for start, length in cfg.occlusions:
        end = min(start + length, cfg.num_frames)
        if start < cfg.num_frames and end > start:
            occluded[start:end] = True
            clipped.append((start, end - start))

    p0 = np.array(cfg.initial_position, dtype=float)
    v0 = np.array(cfg.initial_velocity, dtype=float)
    accel = np.array([0.0, cfg.gravity])

    frames: List[SimFrame] = []
    for i in range(cfg.num_frames):
        t = float(times[i])
        gt = p0 + v0 * t + 0.5 * accel * t * t
        detections: List[Detection] = []

        missed = rng.random() < cfg.miss_probability
        conf = float(rng.uniform(*cfg.confidence_range))
        center = gt + rng.normal(0.0, cfg.center_noise_px, 2)
        w = max(1.0, cfg.ball_size_px + rng.normal(0.0, cfg.size_noise_px))
        h = max(1.0, cfg.ball_size_px + rng.normal(0.0, cfg.size_noise_px))
        true_detected = not occluded[i] and not missed
        if true_detected:
            detections.append(Detection(_box(center[0], center[1], w, h), conf, BALL_CLASS_ID))

        for _ in range(int(rng.poisson(cfg.false_positive_rate))):
            fx = rng.uniform(0, cfg.frame_width)
            fy = rng.uniform(0, cfg.frame_height)
            fs = rng.uniform(*cfg.false_positive_size_range)
            fconf = rng.uniform(*cfg.false_positive_confidence_range)
            detections.append(Detection(_box(fx, fy, fs, fs), float(fconf), BALL_CLASS_ID))

        frames.append(
            SimFrame(
                frame_idx=i,
                timestamp_s=t,
                detections=detections,
                gt_center=(float(gt[0]), float(gt[1])),
                occluded=bool(occluded[i]),
                true_detected=true_detected,
            )
        )

    return SyntheticSequence(frames=frames, occlusions=clipped, config=cfg)
