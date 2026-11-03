"""Frame source -> detector -> tracker -> overlay/telemetry loop (shared by scripts and tests)."""
from __future__ import annotations

import time
from typing import List, Optional, Tuple

import cv2

from edge_vision.camera.frame_source import FrameSource
from edge_vision.detection.detector import Detection, Detector
from edge_vision.telemetry.logger import TelemetryLogger
from edge_vision.tracking.tracker import TrackOutput, Tracker
from edge_vision.visualization.overlay import draw_overlay

WINDOW_NAME = "edge-vision"
QUIT_KEY = ord("q")


def pick_clicked(detections: List[Detection], click: Tuple[int, int]) -> Optional[Detection]:
    """Detection whose box contains the click (nearest center wins), else None."""
    hits = [d for d in detections if d.bbox_xyxy[0] <= click[0] <= d.bbox_xyxy[2] and d.bbox_xyxy[1] <= click[1] <= d.bbox_xyxy[3]]
    if not hits:
        return None
    return min(hits, key=lambda d: (d.center[0] - click[0]) ** 2 + (d.center[1] - click[1]) ** 2)


def process_stream(
    source: FrameSource,
    detector: Detector,
    tracker: Tracker,
    logger: Optional[TelemetryLogger] = None,
    display: bool = False,
) -> List[TrackOutput]:
    """Run the whole pipeline over `source`; returns per-frame outputs.

    Subject selection: a click in the display window picks a detection; otherwise the
    highest-confidence detection is auto-selected whenever there is no live subject.
    Wall-clock is used only to measure latency, never for tracking dt.
    """
    pending_click: List[Tuple[int, int]] = []
    if display:
        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(
            WINDOW_NAME, lambda ev, x, y, *_: pending_click.append((x, y)) if ev == cv2.EVENT_LBUTTONDOWN else None
        )

    outputs: List[TrackOutput] = []
    try:
        for frame in source:
            t0 = time.perf_counter()
            detections = detector.detect(frame.image)
            t1 = time.perf_counter()

            if pending_click:
                chosen = pick_clicked(detections, pending_click[-1])
                pending_click.clear()
                if chosen is not None:
                    tracker.select(chosen, frame.timestamp_s, frame.index)
            if tracker.needs_selection():
                tracker.select_best(detections, frame.timestamp_s, frame.index)

            output = tracker.update(frame.index, frame.timestamp_s, detections)
            t2 = time.perf_counter()
            outputs.append(output)
            if logger is not None:
                logger.log_frame(output, (t1 - t0) * 1000.0, (t2 - t0) * 1000.0)

            if display:
                subject = tracker.subject
                history = subject.trajectory.points() if subject is not None else []
                draw_overlay(frame.image, output, history, tracker.predicted_arc(), [d.bbox_xyxy for d in detections])
                cv2.imshow(WINDOW_NAME, frame.image)
                if (cv2.waitKey(1) & 0xFF) == QUIT_KEY:
                    break
    finally:
        if display:
            cv2.destroyWindow(WINDOW_NAME)
    return outputs
