"""Frame sources. Every frame carries its own timestamp; consumers never use wall-clock for dt."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

import cv2
import numpy as np

from edge_vision.recording import load_timestamps


@dataclass
class Frame:
    """One image plus the time it was captured (seconds, source clock)."""

    index: int
    timestamp_s: float
    image: np.ndarray


class FrameSource(ABC):
    """Anything that yields timestamped frames.

    Sources that know them also expose `nominal_fps` (0 if unknown) and `size` (width, height).
    """

    nominal_fps: float = 0.0
    size: Tuple[int, int] = (0, 0)
    frame_count: int = 0  # total frames if known (files), else 0

    @abstractmethod
    def read(self) -> Optional[Frame]:
        """Return the next frame, or None at end of stream."""

    @abstractmethod
    def close(self) -> None:
        """Release underlying resources."""

    def __iter__(self) -> Iterator[Frame]:
        while True:
            frame = self.read()
            if frame is None:
                return
            yield frame

    def __enter__(self) -> FrameSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class VideoFileSource(FrameSource):
    """Reads frames from a video file using the container's timestamps (or a recorded sidecar, if present)."""

    def __init__(self, path: Union[str, Path]) -> None:
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise OSError(f"Could not open video file: {path}")
        self.nominal_fps: float = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.size: Tuple[int, int] = (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._index = 0
        self._last_ts: Optional[float] = None
        # A sidecar written by FrameRecorder keeps the real frame spacing of a live recording.
        self._recorded_ts = load_timestamps(path)

    def read(self) -> Optional[Frame]:
        ok, image = self._cap.read()
        if not ok:
            return None
        if self._recorded_ts is not None and self._index < len(self._recorded_ts):
            ts = self._recorded_ts[self._index]
        else:
            ts = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        # Some containers report 0 or repeated timestamps; keep the clock strictly increasing.
        if self._last_ts is not None and ts <= self._last_ts:
            step = 1.0 / self.nominal_fps if self.nominal_fps > 0 else 0.0
            ts = self._last_ts + step
        frame = Frame(index=self._index, timestamp_s=ts, image=image)
        self._index += 1
        self._last_ts = ts
        return frame

    def close(self) -> None:
        self._cap.release()


class CameraSource(FrameSource):
    """Live camera through OpenCV (laptop webcam, USB camera).

    Timestamps are `time.monotonic()` seconds since the first frame, taken right after the
    driver hands the frame over; that is this source's own clock, not a tracking assumption.
    On the Jetson, replace this with a GStreamer source that uses the driver's buffer
    timestamps. `read()` blocks until the next frame, so it paces itself.
    """

    def __init__(
        self,
        device: Union[int, str] = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[float] = None,
    ) -> None:
        self._cap = cv2.VideoCapture(device)
        if not self._cap.isOpened():
            raise OSError(f"Could not open camera {device!r} (on macOS, allow camera access for your terminal app)")
        if width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            self._cap.set(cv2.CAP_PROP_FPS, fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # ignored by some backends; avoids a stale-frame queue
        self.nominal_fps: float = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.size: Tuple[int, int] = (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        self._index = 0
        self._t0: Optional[float] = None

    def read(self) -> Optional[Frame]:
        ok, image = self._cap.read()
        if not ok:
            return None
        now = time.monotonic()
        if self._t0 is None:
            self._t0 = now
        frame = Frame(index=self._index, timestamp_s=now - self._t0, image=image)
        self._index += 1
        return frame

    def close(self) -> None:
        self._cap.release()
