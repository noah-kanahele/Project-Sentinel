"""Real-time harness: capture thread + latest-frame-only mailbox + processing on the main thread.

A camera does not wait for the tracker. The capture thread publishes frames into a single
slot; if the processor has not taken the previous frame yet it is overwritten and counted as
dropped, so latency stays bounded instead of building up in a queue. Files are replayed
paced to their own timestamps to behave like a camera. Tracking still uses only the frame
timestamps for dt (a dropped frame just means a larger dt); `perf_counter` is used purely to
measure latency.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import cv2
import numpy as np

from edge_vision.camera.frame_source import Frame, FrameSource
from edge_vision.detection.detector import Detector
from edge_vision.telemetry.logger import TelemetryLogger
from edge_vision.tracking.tracker import TrackOutput, Tracker
from edge_vision.visualization.overlay import draw_overlay

WINDOW_NAME = "edge-vision realtime"
QUIT_KEY = ord("q")
POLL_S = 0.05

Item = Tuple[Frame, float]  # frame, perf_counter time at which it became available


class LatestFrame:
    """Single-slot mailbox: `put` overwrites an unconsumed frame and counts it as dropped."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._item: Optional[Item] = None
        self._finished = False
        self.captured = 0
        self.dropped = 0

    def put(self, item: Item) -> None:
        with self._cond:
            if self._item is not None:
                self.dropped += 1
            self._item = item
            self.captured += 1
            self._cond.notify()

    def finish(self) -> None:
        with self._cond:
            self._finished = True
            self._cond.notify_all()

    def get(self, timeout: float) -> Tuple[Optional[Item], bool]:
        """Return (item or None, stream_over). stream_over is True only once drained."""
        with self._cond:
            if self._item is None and not self._finished:
                self._cond.wait(timeout)
            item, self._item = self._item, None
            return item, item is None and self._finished


@dataclass
class RealtimeStats:
    captured: int = 0
    processed: int = 0
    dropped: int = 0
    deadline_misses: int = 0
    wall_s: float = 0.0
    period_ms: Optional[float] = None  # one frame period of the source, if known
    latency_ms: List[float] = field(default_factory=list)  # frame available -> tracker output
    process_ms: List[float] = field(default_factory=list)  # detect + track time only

    def summary(self) -> Dict[str, float]:
        def pct(values: List[float], q: float) -> float:
            return float(np.percentile(values, q)) if values else float("nan")

        return {
            "captured": self.captured,
            "processed": self.processed,
            "dropped": self.dropped,
            "drop_fraction": self.dropped / self.captured if self.captured else 0.0,
            "deadline_misses": self.deadline_misses,
            "deadline_miss_fraction": self.deadline_misses / self.processed if self.processed else 0.0,
            "processed_fps": self.processed / self.wall_s if self.wall_s > 0 else 0.0,
            "period_ms": self.period_ms if self.period_ms is not None else float("nan"),
            "latency_p50_ms": pct(self.latency_ms, 50),
            "latency_p95_ms": pct(self.latency_ms, 95),
            "latency_p99_ms": pct(self.latency_ms, 99),
            "latency_max_ms": max(self.latency_ms) if self.latency_ms else float("nan"),
            "process_p99_ms": pct(self.process_ms, 99),
        }

    def verdict(self, realtime_cfg: Mapping[str, Any]) -> Dict[str, bool]:
        """Pass/fail per criterion against the `realtime:` config section."""
        s = self.summary()
        return {
            "p99_latency": s["latency_p99_ms"] <= realtime_cfg["target_p99_latency_ms"],
            "drops": s["drop_fraction"] <= realtime_cfg["max_drop_fraction"],
            "deadline_misses": s["deadline_miss_fraction"] <= realtime_cfg["max_deadline_miss_fraction"],
        }


@dataclass
class RealtimeResult:
    outputs: Dict[int, TrackOutput]  # by frame index; dropped frames have no entry
    stats: RealtimeStats


class RealtimeRunner:
    """Runs source -> detector -> tracker with real-time semantics."""

    def __init__(
        self,
        source: FrameSource,
        detector: Detector,
        tracker: Tracker,
        pace: bool,
        speed: float = 1.0,
        period_s: Optional[float] = None,
        logger: Optional[TelemetryLogger] = None,
        display: bool = False,
        warmup_image: Optional[np.ndarray] = None,
    ) -> None:
        """`pace` replays a file at its own timestamps (x `speed`); leave it False for live cameras.

        `period_s` is the source's frame period used for deadline misses; defaults to the
        source's `nominal_fps` when it has one. If `warmup_image` is given, the detector runs on it
        once before the clock starts (model load / first-call cost is not steady-state latency).
        """
        if speed <= 0:
            raise ValueError("speed must be positive")
        self.source, self.detector, self.tracker = source, detector, tracker
        self.pace, self.speed, self.logger, self.display = pace, speed, logger, display
        self.warmup_image = warmup_image
        fps = getattr(source, "nominal_fps", 0.0)
        self.period_s = period_s if period_s is not None else (1.0 / fps if fps > 0 else None)
        self._box = LatestFrame()
        self._stop = threading.Event()

    def _capture(self) -> None:
        t_start = time.perf_counter()
        ts0: Optional[float] = None
        try:
            for frame in self.source:
                if self._stop.is_set():
                    break
                if ts0 is None:
                    ts0 = frame.timestamp_s
                if self.pace:
                    wait = t_start + (frame.timestamp_s - ts0) / self.speed - time.perf_counter()
                    if wait > 0:
                        time.sleep(wait)
                self._box.put((frame, time.perf_counter()))
        finally:
            self._box.finish()

    def run(self, max_frames: Optional[int] = None, duration_s: Optional[float] = None) -> RealtimeResult:
        """Process until the stream ends, `max_frames` are processed, `duration_s` passes or q is pressed."""
        stats = RealtimeStats(period_ms=self.period_s * 1000.0 if self.period_s else None)
        outputs: Dict[int, TrackOutput] = {}
        if self.warmup_image is not None:
            self.detector.detect(self.warmup_image)
        thread = threading.Thread(target=self._capture, daemon=True)
        t_begin = time.perf_counter()
        thread.start()
        if self.display:
            cv2.namedWindow(WINDOW_NAME)
        try:
            while True:
                item, over = self._box.get(POLL_S)
                if over:
                    break
                if item is None:
                    if duration_s is not None and time.perf_counter() - t_begin >= duration_s:
                        break
                    continue
                frame, t_available = item
                t0 = time.perf_counter()
                detections = self.detector.detect(frame.image)
                t1 = time.perf_counter()
                if self.tracker.needs_selection():
                    self.tracker.select_best(detections, frame.timestamp_s, frame.index)
                output = self.tracker.update(frame.index, frame.timestamp_s, detections)
                t2 = time.perf_counter()

                outputs[frame.index] = output
                stats.processed += 1
                stats.process_ms.append((t2 - t0) * 1000.0)
                stats.latency_ms.append((t2 - t_available) * 1000.0)
                if self.period_s is not None and (t2 - t0) > self.period_s:
                    stats.deadline_misses += 1
                if self.logger is not None:
                    self.logger.log_frame(output, (t1 - t0) * 1000.0, (t2 - t0) * 1000.0)

                if self.display:
                    subject = self.tracker.subject
                    history = subject.trajectory.points() if subject is not None else []
                    draw_overlay(frame.image, output, history, self.tracker.predicted_arc(), [d.bbox_xyxy for d in detections])
                    cv2.imshow(WINDOW_NAME, frame.image)
                    if (cv2.waitKey(1) & 0xFF) == QUIT_KEY:
                        break
                if max_frames is not None and stats.processed >= max_frames:
                    break
                if duration_s is not None and time.perf_counter() - t_begin >= duration_s:
                    break
        finally:
            self._stop.set()
            stats.wall_s = time.perf_counter() - t_begin
            stats.captured = self._box.captured
            stats.dropped = self._box.dropped
            if self.display:
                cv2.destroyWindow(WINDOW_NAME)
            if self.pace:  # file replay ends on its own; a live camera thread is a daemon blocked in read()
                thread.join(timeout=1.0)
        return RealtimeResult(outputs=outputs, stats=stats)
