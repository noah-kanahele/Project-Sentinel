"""Tracking metrics against synthetic ground truth."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from edge_vision.tracking.state_machine import TrackState
from edge_vision.tracking.tracker import TrackOutput

if TYPE_CHECKING:  # avoids a sim <-> tracking import cycle
    from edge_vision.sim.synthetic import SyntheticSequence

NO_TRACK = "NO_TRACK"


@dataclass
class OcclusionResult:
    start: int
    length: int
    evaluable: bool  # a live track existed just before the occlusion
    success: bool  # same id observed on a true detection within the deadline after it ended


@dataclass
class TrackMetrics:
    num_frames: int = 0
    id_switches: int = 0
    lost_track_count: int = 0  # entries into TEMPORARILY_LOST
    tracks_ended: int = 0
    frames_in_state: Dict[str, int] = field(default_factory=dict)
    occlusions: List[OcclusionResult] = field(default_factory=list)
    pred_errors: List[float] = field(default_factory=list)  # px, all frames with a prediction
    occlusion_pred_errors: List[float] = field(default_factory=list)  # px, frames inside occlusions
    fp_adoption_frames: int = 0  # frames where the matched detection was far from truth
    fp_adoption_events: int = 0  # runs of consecutive such frames

    @property
    def pred_err_mean(self) -> float:
        return float(np.mean(self.pred_errors)) if self.pred_errors else float("nan")

    @property
    def pred_err_p95(self) -> float:
        return float(np.percentile(self.pred_errors, 95)) if self.pred_errors else float("nan")

    @property
    def occlusion_pred_err_mean(self) -> float:
        return float(np.mean(self.occlusion_pred_errors)) if self.occlusion_pred_errors else float("nan")

    def reacquisition_rate(self) -> Tuple[int, int]:
        """(successes, evaluable occlusions) over all occlusions."""
        evaluable = [o for o in self.occlusions if o.evaluable]
        return sum(o.success for o in evaluable), len(evaluable)

    def reacquisition_by_bucket(self, buckets: Sequence[Sequence[Any]]) -> Dict[str, Tuple[int, int]]:
        """{label: (successes, evaluable)} for occlusion-length buckets like [[1,5],[21,None]]."""
        result: Dict[str, Tuple[int, int]] = {}
        for lo, hi in buckets:
            label = bucket_label(lo, hi)
            members = [o for o in self.occlusions if o.evaluable and o.length >= lo and (hi is None or o.length <= hi)]
            result[label] = (sum(o.success for o in members), len(members))
        return result


def bucket_label(lo: int, hi: Optional[int]) -> str:
    return f"{lo}+" if hi is None else f"{lo}-{hi}"


def compute_metrics(
    outputs: Sequence[TrackOutput],
    seq: SyntheticSequence,
    adoption_tolerance_px: float,
    reacq_deadline_frames: int,
) -> TrackMetrics:
    """Score tracker `outputs` (one per frame) against the ground truth in `seq`."""
    gt = seq.gt_centers()
    m = TrackMetrics(num_frames=len(outputs))

    counts: Counter = Counter()
    prev_state: Optional[TrackState] = None
    prev_id: Optional[int] = None
    in_adoption = False
    in_occlusion = np.zeros(len(outputs), dtype=bool)
    for start, length in seq.occlusions:
        in_occlusion[start:start + length] = True

    for out, truth in zip(outputs, gt):
        counts[out.state.value if out.state else NO_TRACK] += 1

        if out.state == TrackState.TEMPORARILY_LOST and prev_state != TrackState.TEMPORARILY_LOST:
            m.lost_track_count += 1
        if out.state == TrackState.TRACK_ENDED and prev_state != TrackState.TRACK_ENDED:
            m.tracks_ended += 1
        prev_state = out.state

        if out.track_id is not None:
            if prev_id is not None and out.track_id != prev_id:
                m.id_switches += 1
            prev_id = out.track_id

        if out.pred_xy is not None:
            err = float(np.hypot(out.pred_xy[0] - truth[0], out.pred_xy[1] - truth[1]))
            m.pred_errors.append(err)
            if in_occlusion[out.frame_idx]:
                m.occlusion_pred_errors.append(err)

        adopted = False
        if out.observed and out.center is not None:
            adopted = float(np.hypot(out.center[0] - truth[0], out.center[1] - truth[1])) > adoption_tolerance_px
        if adopted:
            m.fp_adoption_frames += 1
            if not in_adoption:
                m.fp_adoption_events += 1
        in_adoption = adopted

    m.frames_in_state = dict(counts)

    for start, length in seq.occlusions:
        before = outputs[start - 1] if start >= 1 else None
        live_before = (
            before is not None and before.track_id is not None and before.state not in (None, TrackState.TRACK_ENDED)
        )
        success = False
        if live_before:
            assert before is not None and before.track_id is not None
            end = start + length
            for out in outputs[end:end + reacq_deadline_frames]:
                if out.track_id == before.track_id and out.state == TrackState.TRACKING and out.observed and out.center is not None:
                    err = np.hypot(out.center[0] - gt[out.frame_idx][0], out.center[1] - gt[out.frame_idx][1])
                    if err <= adoption_tolerance_px:
                        success = True
                        break
        m.occlusions.append(OcclusionResult(start, length, live_before, success))
    return m


def merge_metrics(parts: Sequence[TrackMetrics]) -> TrackMetrics:
    """Pool several runs (e.g. different seeds) into one TrackMetrics."""
    total = TrackMetrics()
    counts: Counter = Counter()
    for p in parts:
        total.num_frames += p.num_frames
        total.id_switches += p.id_switches
        total.lost_track_count += p.lost_track_count
        total.tracks_ended += p.tracks_ended
        counts.update(p.frames_in_state)
        total.occlusions.extend(p.occlusions)
        total.pred_errors.extend(p.pred_errors)
        total.occlusion_pred_errors.extend(p.occlusion_pred_errors)
        total.fp_adoption_frames += p.fp_adoption_frames
        total.fp_adoption_events += p.fp_adoption_events
    total.frames_in_state = dict(counts)
    return total
