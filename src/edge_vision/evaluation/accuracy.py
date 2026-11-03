"""Score tracker output against labeled ground truth, with a confidence interval on the hold rate."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

from edge_vision.evaluation.labels import Label
from edge_vision.tracking.state_machine import TrackState
from edge_vision.tracking.tracker import TrackOutput


@dataclass(frozen=True)
class AccuracyConfig:
    center_tolerance_fraction: float
    min_tolerance_px: float
    reacq_deadline_frames: int
    target_hold_rate: float
    confidence_z: float


@dataclass
class GapResult:
    """A run of not-visible frames between two visible stretches."""

    start: int
    length: int
    evaluable: bool  # the tracker held the ball (with an id) just before the gap
    frames_to_reacquire: Optional[int]  # None if not reacquired within the deadline

    @property
    def success(self) -> bool:
        return self.evaluable and self.frames_to_reacquire is not None


@dataclass
class AccuracyReport:
    labeled_frames: int = 0
    visible_frames: int = 0
    held: int = 0
    miss_no_track: int = 0  # no live track at all
    miss_wrong_location: int = 0  # track exists but is not on the ball
    miss_dropped: int = 0  # frame was dropped before processing
    false_observed_frames: int = 0  # tracker matched a detection while the ball was not visible
    id_switches: int = 0
    tracks_started: int = 0
    center_errors_px: List[float] = field(default_factory=list)  # on held frames
    gaps: List[GapResult] = field(default_factory=list)
    hold_ci: Tuple[float, float] = (float("nan"), float("nan"))
    target_hold_rate: float = 0.99

    @property
    def hold_rate(self) -> float:
        return self.held / self.visible_frames if self.visible_frames else float("nan")

    @property
    def meets_target(self) -> bool:
        """Point estimate reaches the target."""
        return self.hold_rate >= self.target_hold_rate

    @property
    def meets_target_confidently(self) -> bool:
        """Even the interval's lower bound reaches the target (what you need to claim it)."""
        return self.hold_ci[0] >= self.target_hold_rate

    def reacquisition(self) -> Tuple[int, int]:
        """(successes, evaluable gaps)."""
        evaluable = [g for g in self.gaps if g.evaluable]
        return sum(g.success for g in evaluable), len(evaluable)


def wilson_interval(successes: int, n: int, z: float) -> Tuple[float, float]:
    """Wilson score interval for a proportion (well behaved near 0 and 1)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _held(out: Optional[TrackOutput], label: Label, cfg: AccuracyConfig) -> Tuple[str, Optional[float]]:
    """Classify one visible frame: ('held'|'dropped'|'no_track'|'wrong_location', center error px)."""
    if out is None:
        return "dropped", None
    center = out.center
    if center is None or out.state in (None, TrackState.TRACK_ENDED):
        return "no_track", None
    err = math.hypot(center[0] - label.cx, center[1] - label.cy)  # type: ignore[operator]
    size = label.size or 0.0
    tol = max(cfg.min_tolerance_px, cfg.center_tolerance_fraction * size)
    return ("held" if err <= tol else "wrong_location"), err


def score_clip(
    outputs: Mapping[int, TrackOutput], labels: Mapping[int, Label], cfg: AccuracyConfig
) -> AccuracyReport:
    """Score `outputs` (by frame index; missing = dropped) against `labels`."""
    report = AccuracyReport(target_hold_rate=cfg.target_hold_rate)
    frames = sorted(labels)
    report.labeled_frames = len(frames)

    held_by_frame: Dict[int, bool] = {}
    prev_id: Optional[int] = None
    seen_ids = set()
    for idx in frames:
        label, out = labels[idx], outputs.get(idx)
        if out is not None and out.track_id is not None:
            if prev_id is not None and out.track_id != prev_id:
                report.id_switches += 1
            prev_id = out.track_id
            if out.track_id not in seen_ids:
                seen_ids.add(out.track_id)
                report.tracks_started += 1
        if not label.visible:
            if out is not None and out.observed:
                report.false_observed_frames += 1
            continue
        report.visible_frames += 1
        kind, err = _held(out, label, cfg)
        held_by_frame[idx] = kind == "held"
        if kind == "held":
            report.held += 1
            report.center_errors_px.append(err)  # type: ignore[arg-type]
        elif kind == "dropped":
            report.miss_dropped += 1
        elif kind == "no_track":
            report.miss_no_track += 1
        else:
            report.miss_wrong_location += 1

    report.hold_ci = wilson_interval(report.held, report.visible_frames, cfg.confidence_z)
    report.gaps = _score_gaps(frames, labels, outputs, held_by_frame, cfg)
    return report


def _score_gaps(
    frames: List[int],
    labels: Mapping[int, Label],
    outputs: Mapping[int, TrackOutput],
    held_by_frame: Mapping[int, bool],
    cfg: AccuracyConfig,
) -> List[GapResult]:
    """For each hidden stretch between visible ones: does the same track id come back in time?"""
    gaps: List[GapResult] = []
    i = 0
    while i < len(frames):
        if labels[frames[i]].visible:
            i += 1
            continue
        j = i
        while j < len(frames) and not labels[frames[j]].visible:
            j += 1
        if i > 0 and j < len(frames):  # needs a visible frame on both sides
            before = next(
                (outputs[f].track_id for f in reversed(frames[:i]) if held_by_frame.get(f) and f in outputs), None
            )
            first_back = frames[j]
            result = None
            if before is not None:
                for f in frames[j:]:
                    if f - first_back >= cfg.reacq_deadline_frames:
                        break
                    if labels[f].visible and held_by_frame.get(f) and outputs[f].track_id == before:
                        result = f - first_back
                        break
            gaps.append(GapResult(frames[i], frames[j - 1] - frames[i] + 1, before is not None, result))
        i = j
    return gaps
