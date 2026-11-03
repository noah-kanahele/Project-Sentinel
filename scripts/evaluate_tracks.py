#!/usr/bin/env python3
"""Detailed metrics for one synthetic scenario pooled over seeds.

  python scripts/evaluate_tracks.py --occlusion 25 --fp-rate 0.2 --seeds 20
  python scripts/evaluate_tracks.py --set tracker.gate_base_radius_px=30 --set synthetic.miss_probability=0.1

The metric logic is `edge_vision.tracking.metrics.compute_metrics`; this script is a CLI around it.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

from edge_vision.config import load_config
from edge_vision.sim.runner import evaluate_scenario
from edge_vision.tracking.metrics import TrackMetrics, merge_metrics

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def report(m: TrackMetrics, buckets: Any) -> str:
    """Human-readable metrics summary."""
    lines = [
        f"frames evaluated      : {m.num_frames}",
        f"ID switches           : {m.id_switches}",
        f"lost-track count      : {m.lost_track_count}   (entries into TEMPORARILY_LOST)",
        f"tracks ended          : {m.tracks_ended}",
        f"false-positive adopts : {m.fp_adoption_events} events / {m.fp_adoption_frames} frames",
        f"prediction error (px) : mean {m.pred_err_mean:.2f}   p95 {m.pred_err_p95:.2f}   "
        f"during occlusion mean {m.occlusion_pred_err_mean:.2f}",
        "frames in state       : " + ", ".join(f"{k}={v}" for k, v in sorted(m.frames_in_state.items())),
    ]
    succ, total = m.reacquisition_rate()
    lines.append(f"reacquisition overall : {succ}/{total}")
    for label, (s, t) in m.reacquisition_by_bucket(buckets).items():
        lines.append(f"  occlusion {label:>6}    : {s}/{t}" + (f"  ({100.0 * s / t:.0f}%)" if t else ""))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--occlusion", type=int, nargs="+", help="occlusion lengths (frames); one occlusion each, at frame 40")
    ap.add_argument("--fp-rate", type=float)
    ap.add_argument("--set", action="append", default=[], metavar="section.key=value", help="config override (YAML value)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    overrides: Dict[str, Any] = {}
    for item in args.set:
        key, value = item.split("=", 1)
        overrides[key] = yaml.safe_load(value)
    if args.fp_rate is not None:
        overrides["synthetic.false_positive_rate"] = args.fp_rate
    if args.occlusion:
        # One run per length would hide per-bucket detail, so spread them across seeds instead.
        lengths = args.occlusion
        parts = [
            evaluate_scenario(cfg, [seed], {**overrides, "synthetic.occlusions": ((40, lengths[seed % len(lengths)]),)})
            for seed in range(args.seeds)
        ]
        metrics = merge_metrics(parts)
    else:
        metrics = evaluate_scenario(cfg, list(range(args.seeds)), overrides)
    print(report(metrics, cfg["evaluation"]["occlusion_buckets"]))


if __name__ == "__main__":
    main()
