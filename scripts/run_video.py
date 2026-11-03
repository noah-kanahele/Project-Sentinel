#!/usr/bin/env python3
"""Video file -> detector -> tracker -> overlay -> optional CSV.

  python scripts/run_video.py clip.mp4 --config configs/default.yaml --csv logs/out.csv --display
  python scripts/run_video.py clip.mp4 --detector color     # HSV color blob, no model needed

Click a detection in the window to choose the subject; otherwise the highest-confidence
detection is auto-selected. Press q to quit. The yolo detector needs: pip install ultralytics
"""
from __future__ import annotations

import argparse
from pathlib import Path

from edge_vision.camera.frame_source import VideoFileSource
from edge_vision.config import build_config, load_config
from edge_vision.detection.color import ColorBlobDetector
from edge_vision.detection.detector import Detector, YoloDetector
from edge_vision.pipeline import process_stream
from edge_vision.telemetry.logger import TelemetryLogger
from edge_vision.tracking.state_machine import TrackState
from edge_vision.tracking.tracker import Tracker, TrackerConfig

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="path to a video file")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--detector", choices=["yolo", "color"], default="yolo", help="detector to use (default: yolo)")
    ap.add_argument("--csv", help="write per-frame telemetry here (events go to <name>.events.csv)")
    ap.add_argument("--display", action="store_true", help="show the overlay window (click to select)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    det = cfg["detection"]
    detector: Detector
    if args.detector == "color":
        detector = ColorBlobDetector.from_config(cfg)
    else:
        detector = YoloDetector(det["model"], det["class_id"], det["confidence_threshold"], det["image_size"], det["device"])

    with TelemetryLogger(args.csv) as logger, VideoFileSource(args.video) as source:
        tracker = Tracker(TrackerConfig.from_dict(cfg), event_sink=logger.log_event)
        outputs = process_stream(source, detector, tracker, logger, args.display)

    states = {s: sum(1 for o in outputs if o.state == s) for s in TrackState}
    print(f"{len(outputs)} frames; transitions logged: {len(tracker.events)}")
    print("frames per state:", {s.value: n for s, n in states.items() if n})


if __name__ == "__main__":
    main()
