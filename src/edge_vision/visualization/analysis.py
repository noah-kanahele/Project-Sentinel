"""Slow-motion analysis video: the tracker's view of the ball's movement, written to a file."""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple, Union

import cv2
import numpy as np

from edge_vision.detection.detector import Detection
from edge_vision.tracking.tracker import TrackOutput, Tracker
from edge_vision.visualization.overlay import ARC_COLOR, draw_overlay

TRAIL_COLOR = (255, 255, 0)  # BGR cyan
VELOCITY_COLOR = (0, 255, 255)  # BGR yellow
HUD_COLOR = (255, 255, 255)
HUD_BG = (0, 0, 0)
VELOCITY_ARROW_S = 0.1  # arrow length = distance travelled in this many seconds
Point = Tuple[float, float]


def _pt(p: Point) -> Tuple[int, int]:
    return (int(round(p[0])), int(round(p[1])))


class AnalysisWriter:
    """Draws each processed frame with the whole path so far and writes it at a slowed frame rate.

    The slowdown only changes the playback rate of the written file (source fps / slowdown);
    tracking itself never depends on it.
    """

    def __init__(self, path: Union[str, Path], source_fps: float, size: Tuple[int, int], slowdown: float) -> None:
        if slowdown <= 0 or source_fps <= 0:
            raise ValueError("slowdown and source_fps must be positive")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.slowdown = slowdown
        self.fps = source_fps / slowdown
        self._writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, size)  # type: ignore[attr-defined]
        if not self._writer.isOpened():
            raise OSError(f"Could not open analysis video for writing: {self.path}")
        self._trail: List[Tuple[Point, bool]] = []  # (center, observed)

    def write(self, image: np.ndarray, output: TrackOutput, detections: Sequence[Detection], tracker: Tracker) -> None:
        """Annotate a copy of `image` and append it to the video."""
        canvas = image.copy()
        center = output.center
        if center is not None:
            self._trail.append((center, output.observed))
        self._draw_trail(canvas)
        subject = tracker.subject
        history = subject.trajectory.points() if subject is not None else []
        draw_overlay(canvas, output, history, tracker.predicted_arc(), [d.bbox_xyxy for d in detections])
        self._draw_velocity(canvas, tracker, output)
        self._draw_hud(canvas, output, tracker)
        self._writer.write(canvas)

    def _draw_trail(self, canvas: np.ndarray) -> None:
        """Whole path so far: line through every point, filled dot = observed, hollow = predicted."""
        pts = [p for p, _ in self._trail]
        if len(pts) >= 2:
            cv2.polylines(canvas, [np.array([_pt(p) for p in pts], dtype=np.int32)], False, TRAIL_COLOR, 1, cv2.LINE_AA)  # type: ignore[list-item]
        for p, observed in self._trail:
            if observed:
                cv2.circle(canvas, _pt(p), 4, TRAIL_COLOR, -1, cv2.LINE_AA)
            else:
                cv2.circle(canvas, _pt(p), 5, ARC_COLOR, 1, cv2.LINE_AA)

    def _draw_velocity(self, canvas: np.ndarray, tracker: Tracker, output: TrackOutput) -> None:
        subject = tracker.subject
        if subject is None or output.center is None:
            return
        v = subject.kf.velocity
        tip = (output.center[0] + v[0] * VELOCITY_ARROW_S, output.center[1] + v[1] * VELOCITY_ARROW_S)
        cv2.arrowedLine(canvas, _pt(output.center), _pt(tip), VELOCITY_COLOR, 2, cv2.LINE_AA, tipLength=0.25)

    def _draw_hud(self, canvas: np.ndarray, output: TrackOutput, tracker: Tracker) -> None:
        subject = tracker.subject
        speed = float(np.hypot(*subject.kf.velocity)) if subject is not None and output.center is not None else 0.0
        state = output.state.value if output.state else "NO SUBJECT"
        seen = "observed" if output.observed else "predicted"
        lines = [
            f"frame {output.frame_idx}   t={output.timestamp_s:.3f}s   {1 / self.slowdown:.2f}x speed",
            f"{state}  ({seen})   speed {speed:.0f} px/s",
        ]
        for i, text in enumerate(lines):
            y = 32 + 30 * i
            cv2.putText(canvas, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, HUD_BG, 4, cv2.LINE_AA)
            cv2.putText(canvas, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, HUD_COLOR, 1, cv2.LINE_AA)

    def close(self) -> None:
        self._writer.release()

    def __enter__(self) -> AnalysisWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def play_video(path: Union[str, Path], window: str = "edge-vision analysis", quit_key: int = ord("q")) -> None:
    """Play a video file in a window at its own frame rate; press q to stop."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay_ms = max(1, int(round(1000.0 / fps))) if fps > 0 else 33
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imshow(window, frame)
            if (cv2.waitKey(delay_ms) & 0xFF) == quit_key:
                break
    finally:
        cap.release()
        cv2.destroyWindow(window)
