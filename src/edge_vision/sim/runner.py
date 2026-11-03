"""Drive a Tracker over a synthetic sequence, mimicking the video runner's selection policy."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from edge_vision.config import build_config
from edge_vision.sim.synthetic import SyntheticConfig, SyntheticSequence, generate_sequence
from edge_vision.tracking.metrics import TrackMetrics, compute_metrics, merge_metrics
from edge_vision.tracking.state_machine import TransitionEvent
from edge_vision.tracking.tracker import TrackOutput, Tracker, TrackerConfig


@dataclass
class RunResult:
    outputs: List[TrackOutput]
    events: List[TransitionEvent]


def run_sequence(cfg: Mapping[str, Any], seq: SyntheticSequence, auto_select: bool = True) -> RunResult:
    """Run the tracker over `seq`.

    With `auto_select`, a new subject (highest-confidence detection) is chosen whenever there
    is none or the previous track ended, exactly as scripts/run_video.py does.
    """
    tracker = Tracker(TrackerConfig.from_dict(cfg))
    outputs: List[TrackOutput] = []
    for frame in seq.frames:
        if tracker.subject is None or (auto_select and tracker.needs_selection()):
            tracker.select_best(frame.detections, frame.timestamp_s, frame.frame_idx)
        outputs.append(tracker.update(frame.frame_idx, frame.timestamp_s, frame.detections))
    return RunResult(outputs=outputs, events=list(tracker.events))


def with_overrides(cfg: Mapping[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    """Deep-copy `cfg` and apply dotted-path overrides, e.g. {"tracker.gate_base_radius_px": 30}."""
    out = copy.deepcopy(dict(cfg))
    for path, value in overrides.items():
        section, key = path.split(".")
        out[section][key] = value
    return out


def evaluate_scenario(
    cfg: Mapping[str, Any],
    seeds: Sequence[int],
    overrides: Optional[Mapping[str, Any]] = None,
) -> TrackMetrics:
    """Run one scenario over several seeds and pool the metrics.

    `overrides` are dotted-path config overrides ("synthetic.false_positive_rate", "tracker.gate_base_radius_px", ...).
    """
    cfg = with_overrides(cfg, overrides or {})
    ev = cfg["evaluation"]
    parts = []
    for seed in seeds:
        seq = generate_sequence(build_config(SyntheticConfig, {**cfg["synthetic"], "seed": seed}))
        result = run_sequence(cfg, seq)
        parts.append(
            compute_metrics(result.outputs, seq, ev["adoption_tolerance_px"], ev["reacq_deadline_frames"])
        )
    return merge_metrics(parts)
