"""Record frames to a video file, with a sidecar of the real per-frame timestamps."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

SIDECAR_SUFFIX = ".timestamps.csv"


def sidecar_path(video_path: Union[str, Path]) -> Path:
    """`clip.mp4` -> `clip.timestamps.csv`."""
    return Path(video_path).with_suffix(SIDECAR_SUFFIX)


def load_timestamps(video_path: Union[str, Path]) -> Optional[List[float]]:
    """Timestamps from the video's sidecar file, or None if there is none."""
    path = sidecar_path(video_path)
    if not path.exists():
        return None
    with open(path, newline="") as f:
        return [float(row["timestamp_s"]) for row in csv.DictReader(f)]


class FrameRecorder:
    """Writes frames to `path`. With `sidecar=True` also records each frame's timestamp, so a replay
    keeps the real frame spacing (a video container only stores a nominal fps)."""

    def __init__(self, path: Union[str, Path], fps: float, size: Tuple[int, int], sidecar: bool) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps if fps > 0 else 30.0, size)
        if not self._writer.isOpened():
            raise IOError(f"Could not open video for writing: {self.path}")
        self._sidecar = sidecar
        self._timestamps: List[float] = []
        self.frames = 0

    def write(self, image: np.ndarray, timestamp_s: float) -> None:
        self._writer.write(image)
        self._timestamps.append(timestamp_s)
        self.frames += 1

    def close(self) -> None:
        self._writer.release()
        if self._sidecar:
            with open(sidecar_path(self.path), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["frame_idx", "timestamp_s"])
                for i, t in enumerate(self._timestamps):
                    w.writerow([i, round(t, 6)])
