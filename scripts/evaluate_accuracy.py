#!/usr/bin/env python3
"""Score the tracker on a labeled clip: hold rate with a confidence interval, ID switches, reacquisition.

  python scripts/evaluate_accuracy.py data/clip.mov --labels data/clip.labels.csv --detector color
  python scripts/evaluate_accuracy.py data/clip.mov --labels data/clip.labels.csv --detector yolo --realtime

--realtime replays the clip paced like a camera through the real-time harness, so frames the
pipeline cannot keep up with are dropped and count as misses. Several clips can be scored
together by giving several VIDEO:LABELS pairs with --pair.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from edge_vision.camera.frame_source import VideoFileSource
from edge_vision.config import build_config, load_config
from edge_vision.detection.color import ColorBlobDetector
from edge_vision.detection.detector import Detector, YoloDetector
from edge_vision.evaluation import AccuracyConfig, AccuracyReport, load_labels, score_clip, wilson_interval
from edge_vision.pipeline import process_stream
from edge_vision.realtime import RealtimeRunner
from edge_vision.tracking.tracker import TrackOutput, Tracker, TrackerConfig

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def build_detector(name: str, cfg: dict) -> Detector:
    if name == "color":
        return ColorBlobDetector.from_config(cfg)
    d = cfg["detection"]
    return YoloDetector(d["model"], d["class_id"], d["confidence_threshold"], d["image_size"], d["device"])


def run_clip(video: str, detector: Detector, cfg: dict, realtime: bool, speed: float) -> Dict[int, TrackOutput]:
    tracker = Tracker(TrackerConfig.from_dict(cfg))
    with VideoFileSource(video) as src:
        if realtime:
            warm = np.zeros((src.size[1], src.size[0], 3), np.uint8)
            return RealtimeRunner(src, detector, tracker, pace=True, speed=speed, warmup_image=warm).run().outputs
        return {o.frame_idx: o for o in process_stream(src, detector, tracker)}


def print_report(name: str, r: AccuracyReport) -> None:
    lo, hi = r.hold_ci
    print(f"\n{name}")
    print(f"  visible frames        : {r.visible_frames} of {r.labeled_frames} labeled")
    print(f"  hold rate             : {100 * r.hold_rate:.2f}%   95% CI [{100 * lo:.2f}%, {100 * hi:.2f}%]")
    print(f"  misses                : no track {r.miss_no_track}, wrong location {r.miss_wrong_location}, dropped {r.miss_dropped}")
    if r.center_errors_px:
        e = np.array(r.center_errors_px)
        print(f"  center error (held)   : median {np.median(e):.1f} px, p95 {np.percentile(e, 95):.1f} px")
    print(f"  ids                   : {r.tracks_started} track(s) started, {r.id_switches} id switch(es)")
    print(f"  false observations    : {r.false_observed_frames} frames matched a detection while ball hidden")
    ok, total = r.reacquisition()
    print(f"  reacquisition         : {ok}/{total} hidden stretches recovered with the same id")
    for g in r.gaps:
        status = "n/a (no track before)" if not g.evaluable else (f"back after {g.frames_to_reacquire} frame(s)" if g.success else "NOT recovered")
        print(f"    hidden frames {g.start}-{g.start + g.length - 1} ({g.length}): {status}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", nargs="?")
    ap.add_argument("--labels", help="labels CSV for VIDEO (see scripts/label_clip.py)")
    ap.add_argument("--pair", action="append", default=[], metavar="VIDEO:LABELS", help="additional clip(s) to pool")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--detector", choices=["yolo", "color"], default="yolo")
    ap.add_argument("--realtime", action="store_true", help="replay paced like a camera; dropped frames count as misses")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed factor with --realtime")
    args = ap.parse_args()

    pairs: List[Tuple[str, str]] = [tuple(p.split(":", 1)) for p in args.pair]  # type: ignore[misc]
    if args.video:
        if not args.labels:
            ap.error("--labels is required with VIDEO")
        pairs.insert(0, (args.video, args.labels))
    if not pairs:
        ap.error("give VIDEO --labels FILE and/or --pair VIDEO:LABELS")

    cfg = load_config(args.config)
    acc = build_config(AccuracyConfig, cfg["accuracy"])
    detector = build_detector(args.detector, cfg)
    pooled = AccuracyReport(target_hold_rate=acc.target_hold_rate)
    for video, labels_path in pairs:
        report = score_clip(run_clip(video, detector, cfg, args.realtime, args.speed), load_labels(labels_path), acc)
        print_report(f"{video} ({'realtime' if args.realtime else 'offline'}, {args.detector})", report)
        for attr in ("labeled_frames", "visible_frames", "held", "miss_no_track", "miss_wrong_location", "miss_dropped",
                     "false_observed_frames", "id_switches", "tracks_started"):
            setattr(pooled, attr, getattr(pooled, attr) + getattr(report, attr))
        pooled.center_errors_px += report.center_errors_px
        pooled.gaps += report.gaps
    pooled.hold_ci = wilson_interval(pooled.held, pooled.visible_frames, acc.confidence_z)
    if len(pairs) > 1:
        print_report("POOLED", pooled)
    verdict = "MEETS" if pooled.meets_target_confidently else ("point estimate meets, NOT proven" if pooled.meets_target else "BELOW")
    print(f"\ntarget hold rate {100 * acc.target_hold_rate:.0f}%: {verdict} "
          f"(lower 95% bound {100 * pooled.hold_ci[0]:.2f}% over {pooled.visible_frames} visible frames)")
    sys.exit(0 if pooled.meets_target_confidently else 1)


if __name__ == "__main__":
    main()
