#!/usr/bin/env python3
"""Video file -> detector -> tracker -> overlay -> optional CSV.

  python scripts/run_video.py clip.mp4 --config configs/default.yaml --csv logs/out.csv --display
  python scripts/run_video.py clip.mp4 --detector color     # HSV color blob, no model needed
  python scripts/run_video.py clip.mp4 --analysis logs/a.mp4 --slowdown 4 --play-analysis

Click a detection in the window to choose the subject; otherwise the highest-confidence
detection is auto-selected. Press q to quit. The yolo detector needs: pip install ultralytics
"""
from __future__ import annotations

import argparse
from pathlib import Path

from edge_vision.camera.frame_source import VideoFileSource
from edge_vision.config import load_config
from edge_vision.detection.color import ColorBlobDetector
from edge_vision.detection.detector import Detector, YoloDetector
from edge_vision.pipeline import process_stream
from edge_vision.telemetry.logger import TelemetryLogger
from edge_vision.tracking.state_machine import TrackState
from edge_vision.tracking.tracker import Tracker, TrackerConfig
from edge_vision.visualization.analysis import AnalysisWriter, play_video

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="path to a video file")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--detector", choices=["yolo", "color"], default="yolo", help="detector to use (default: yolo)")
    ap.add_argument("--csv", help="write per-frame telemetry here (events go to <name>.events.csv)")
    ap.add_argument("--analysis", help="write a slow-motion annotated analysis video to this .mp4 path")
    ap.add_argument("--slowdown", type=float, default=4.0, help="analysis video playback slowdown factor (default 4)")
    ap.add_argument("--play-analysis", action="store_true", help="play the analysis video after processing (needs --analysis)")
    ap.add_argument("--display", action="store_true", help="show the overlay window (click to select)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    det = cfg["detection"]
    detector: Detector
    if args.detector == "color":
        detector = ColorBlobDetector.from_config(cfg)
    else:
        detector = YoloDetector(det["model"], det["class_id"], det["confidence_threshold"], det["image_size"], det["device"])

    if args.play_analysis and not args.analysis:
        ap.error("--play-analysis needs --analysis PATH")

    with TelemetryLogger(args.csv) as logger, VideoFileSource(args.video) as source:
        tracker = Tracker(TrackerConfig.from_dict(cfg), event_sink=logger.log_event)
        writer = AnalysisWriter(args.analysis, source.nominal_fps, source.size, args.slowdown) if args.analysis else None
        try:
            outputs = process_stream(source, detector, tracker, logger, args.display, writer)
        finally:
            if writer is not None:
                writer.close()

    states = {s: sum(1 for o in outputs if o.state == s) for s in TrackState}
    print(f"{len(outputs)} frames; transitions logged: {len(tracker.events)}")
    print("frames per state:", {s.value: n for s, n in states.items() if n})
    if args.analysis:
        print(f"analysis video: {args.analysis} ({args.slowdown:g}x slower)")
        if args.play_analysis:
            play_video(args.analysis)


if __name__ == "__main__":
    main()
