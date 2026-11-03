#!/usr/bin/env python3
"""Real-time test: does the pipeline keep up with a camera, and how late is each result?

  python scripts/run_realtime.py data/clip.mov --detector color          # paced file replay
  python scripts/run_realtime.py data/clip.mov --detector yolo --speed 2  # stress: 2x faster than real time
  python scripts/run_realtime.py --camera 0 --detector color --duration 20 --display   # live webcam

Reports dropped frames, deadline misses and capture-to-output latency percentiles, and
PASS/FAIL against the `realtime:` thresholds in the config.

--record [PREFIX] saves the session for offline analysis: PREFIX.mp4 (raw camera frames),
PREFIX.timestamps.csv (the real per-frame timestamps; replays pick them up automatically) and
PREFIX_overlay.mp4 (what the tracker drew live). PREFIX defaults to data/webcam_<date-time>.
On macOS the first webcam use
asks for camera permission for your terminal app.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from edge_vision.camera.frame_source import CameraSource, FrameSource, VideoFileSource
from edge_vision.config import load_config
from edge_vision.detection.color import ColorBlobDetector
from edge_vision.detection.detector import Detector, YoloDetector
from edge_vision.realtime import RealtimeRunner
from edge_vision.recording import FrameRecorder
from edge_vision.telemetry.logger import TelemetryLogger
from edge_vision.tracking.tracker import Tracker, TrackerConfig

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def build_detector(name: str, cfg: dict) -> Detector:
    if name == "color":
        return ColorBlobDetector.from_config(cfg)
    d = cfg["detection"]
    return YoloDetector(d["model"], d["class_id"], d["confidence_threshold"], d["image_size"], d["device"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", nargs="?", help="video file to replay in real time (omit with --camera)")
    ap.add_argument("--camera", type=int, help="camera index (e.g. 0) instead of a file")
    ap.add_argument("--width", type=int, help="requested camera width")
    ap.add_argument("--height", type=int, help="requested camera height")
    ap.add_argument("--fps", type=float, help="requested camera fps")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--detector", choices=["yolo", "color"], default="yolo")
    ap.add_argument("--speed", type=float, default=1.0, help="file replay speed factor (2 = twice as fast as real time)")
    ap.add_argument("--duration", type=float, help="stop after this many seconds (needed for a live camera)")
    ap.add_argument("--csv", help="write per-frame telemetry here")
    ap.add_argument("--record", nargs="?", const="auto", metavar="PREFIX", help="save raw + annotated video and timestamps for offline analysis")
    ap.add_argument("--display", action="store_true", help="show the overlay window (q quits)")
    args = ap.parse_args()
    if (args.video is None) == (args.camera is None):
        ap.error("give either a video file or --camera N")
    if args.camera is not None and args.duration is None and not args.display:
        ap.error("a live camera needs --duration SECONDS (or --display and press q)")

    cfg = load_config(args.config)
    detector = build_detector(args.detector, cfg)
    source: FrameSource = (
        CameraSource(args.camera, args.width, args.height, args.fps) if args.camera is not None else VideoFileSource(args.video)
    )
    raw = overlay = None
    if args.record:
        prefix = Path(f"data/webcam_{time.strftime('%Y%m%d_%H%M%S')}" if args.record == "auto" else args.record)
        fps = getattr(source, "nominal_fps", 0.0)
        raw = FrameRecorder(prefix.with_suffix(".mp4"), fps, source.size, sidecar=True)  # type: ignore[attr-defined]
        overlay = FrameRecorder(prefix.parent / f"{prefix.name}_overlay.mp4", fps, source.size, sidecar=False)  # type: ignore[attr-defined]
    with TelemetryLogger(args.csv) as logger, source:
        tracker = Tracker(TrackerConfig.from_dict(cfg), event_sink=logger.log_event)
        runner = RealtimeRunner(
            source, detector, tracker, pace=args.camera is None, speed=args.speed, logger=logger, display=args.display,
            warmup_image=np.zeros((source.size[1], source.size[0], 3), np.uint8),  # type: ignore[attr-defined]
            record_raw=raw, record_overlay=overlay,
        )
        result = runner.run(duration_s=args.duration)

    if raw is not None:
        print(f"recorded {raw.frames} raw frames -> {raw.path} (+ timestamps sidecar), annotated -> {overlay.path}")  # type: ignore[union-attr]
    s = result.stats.summary()
    print(f"frames captured {s['captured']}, processed {s['processed']}, dropped {s['dropped']} ({100 * s['drop_fraction']:.1f}%)")
    print(f"processed fps {s['processed_fps']:.1f}   source frame period {s['period_ms']:.1f} ms   deadline misses {s['deadline_misses']}")
    print(f"latency ms  p50 {s['latency_p50_ms']:.1f}  p95 {s['latency_p95_ms']:.1f}  p99 {s['latency_p99_ms']:.1f}  max {s['latency_max_ms']:.1f}")
    print(f"process ms  p99 {s['process_p99_ms']:.1f}   (detect + track only)")
    verdict = result.stats.verdict(cfg["realtime"])
    for name, ok in verdict.items():
        print(f"  {name:<16} {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if all(verdict.values()) else 1)


if __name__ == "__main__":
    main()
