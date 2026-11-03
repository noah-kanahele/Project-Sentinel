"""Frame sources. Every frame carries its own timestamp; consumers never use wall-clock for dt."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

import cv2
import numpy as np


@dataclass
class Frame:
    """One image plus the time it was captured (seconds, source clock)."""

    index: int
    timestamp_s: float
    image: Optional[np.ndarray] = None


class FrameSource(ABC):
    """Anything that yields timestamped frames."""

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

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class VideoFileSource(FrameSource):
    """Reads frames from a video file using the container's own timestamps."""

    def __init__(self, path: Union[str, Path]) -> None:
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise IOError(f"Could not open video file: {path}")
        self.nominal_fps: float = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.size: Tuple[int, int] = (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        self._index = 0
        self._last_ts: Optional[float] = None

    def read(self) -> Optional[Frame]:
        ok, image = self._cap.read()
        if not ok:
            return None
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
    """Live camera stub. On the Jetson this will wrap GStreamer and use driver timestamps."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError("CameraSource is not implemented yet (needs the Jetson camera).")

    def read(self) -> Optional[Frame]:  # pragma: no cover
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover
        raise NotImplementedError
