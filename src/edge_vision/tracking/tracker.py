"""Single-subject tracker with Kalman-gated association and reacquisition.

Reacquisition is just association: while a track is lost, its Kalman prediction keeps
moving along the (gravity-aware) arc and its gate keeps growing. The first detection that
falls inside the gate inherits the track id. Detections outside the gate are ignored.

Two guards keep this robust on real footage. The gate is centred on both the Kalman
prediction and the last observed position: a ball that stops or reverses (a hand, a bounce)
leaves the ballistic prediction behind, but is still near where it was last seen. And a
detection whose size differs from the track's by more than `size_ratio_max` is rejected,
which discards specks and unrelated objects.

A lost track also gets a wide-area reacquisition for balls that reappear far from any
prediction (a hand carried it somewhere new; a ballistic prediction runs off to nowhere): a
detection within `max(gate_max_radius_px, reacquire_search_size_scale * object size)` of the last
observed position is adopted once it is seen
at roughly the same place twice within a few frames (the allowed travel grows with the frames
between sightings, since a blurred ball flickers in and out). Random false positives do not repeat
in place, so a single far detection is still ignored, as the gate requires.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Mapping, Optional, Sequence, Tuple

from edge_vision.config import build_config
from edge_vision.detection.detector import Detection
from edge_vision.motion.trajectory import TrajectoryBuffer, TrajectoryConfig, predicted_arc
from edge_vision.tracking.kalman import KalmanConfig, KalmanFilter
from edge_vision.tracking.state_machine import (
    EventListener,
    StateMachineConfig,
    TrackState,
    TrackStateMachine,
    TransitionEvent,
)

Point = Tuple[float, float]
Box = Tuple[float, float, float, float]


@dataclass(frozen=True)
class GateConfig:
    gate_base_radius_px: float
    gate_size_scale: float
    gate_growth_per_frame_lost_px: float
    gate_covariance_scale: float
    gate_max_radius_px: float
    size_ratio_max: float
    observed_gate_growth_size_fraction: float
    observed_gate_max_misses: int
    reacquire_confirm_size_scale: float
    reacquire_search_size_scale: float
    reacquire_confirm_window_frames: int
    size_memory_frames: int


@dataclass(frozen=True)
class TrackerConfig:
    confidence_threshold: float
    gate: GateConfig
    state_machine: StateMachineConfig
    kalman: KalmanConfig
    trajectory: TrajectoryConfig

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any]) -> TrackerConfig:
        """Assemble from the sections of a full default.yaml-style dict."""
        return cls(
            confidence_threshold=float(cfg["detection"]["confidence_threshold"]),
            gate=build_config(GateConfig, cfg["tracker"]),
            state_machine=build_config(StateMachineConfig, cfg["state_machine"]),
            kalman=build_config(KalmanConfig, cfg["kalman"]),
            trajectory=build_config(TrajectoryConfig, cfg["trajectory"]),
        )


@dataclass
class TrackOutput:
    """What the tracker reports for one frame."""

    frame_idx: int
    timestamp_s: float
    track_id: Optional[int] = None
    state: Optional[TrackState] = None
    bbox_xyxy: Optional[Box] = None  # observed box, or predicted box while unobserved
    pred_xy: Optional[Point] = None  # Kalman prior for this frame, before the measurement update
    confidence: Optional[float] = None  # of the matched detection; None if unobserved
    observed: bool = False  # True if a detection was associated this frame
    gate_radius_px: Optional[float] = None

    @property
    def center(self) -> Optional[Point]:
        if self.bbox_xyxy is None:
            return None
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class Track:
    """One tracked object: filter, state machine, size and history."""

    def __init__(
        self,
        track_id: int,
        detection: Detection,
        frame_idx: int,
        timestamp_s: float,
        cfg: TrackerConfig,
        listeners: List[EventListener],
    ) -> None:
        self.track_id = track_id
        self.class_id = detection.class_id
        self.kf = KalmanFilter(cfg.kalman, detection.center)
        self.machine = TrackStateMachine(track_id, cfg.state_machine, frame_idx, timestamp_s, listeners)
        self.size: Point = detection.size
        # Last two observed (timestamp, center) pairs, for an agile velocity estimate.
        self.observed: List[Tuple[float, Point]] = [(timestamp_s, detection.center)]
        self.pending: Optional[Tuple[int, Point]] = None  # (update count, center) of a far candidate seen earlier
        self.updates = 0  # processed frames, counted per update call
        self.size_history: Deque[float] = deque([max(detection.size)], maxlen=cfg.gate.size_memory_frames)
        self.last_timestamp_s = timestamp_s
        self.trajectory = TrajectoryBuffer(cfg.trajectory.history_length, cfg.trajectory.velocity_window)

    @property
    def ref_size(self) -> float:
        """Recent largest size (larger box side): a ball that briefly shrinks (partly out of frame,
        blurred) is still compared against how big it really is."""
        return max(self.size_history)

    @property
    def state(self) -> TrackState:
        return self.machine.state

    @property
    def misses(self) -> int:
        return self.machine.misses


def _distance(a: Point, b: Sequence[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class Tracker:
    """Tracks one selected subject. Data structures allow several tracks later."""

    def __init__(self, cfg: TrackerConfig, event_sink: Optional[Callable[[TransitionEvent], None]] = None) -> None:
        self.cfg = cfg
        self.events: List[TransitionEvent] = []
        self._listeners: List[EventListener] = [self.events.append]
        if event_sink is not None:
            self._listeners.append(event_sink)
        self._tracks: Dict[int, Track] = {}
        self._subject_id: Optional[int] = None
        self._next_id = 1

    @property
    def subject(self) -> Optional[Track]:
        return self._tracks.get(self._subject_id) if self._subject_id is not None else None

    def needs_selection(self) -> bool:
        subject = self.subject
        return subject is None or subject.state == TrackState.TRACK_ENDED

    def select(self, detection: Detection, timestamp_s: float, frame_idx: int = 0) -> Track:
        """Make `detection` the subject. Any current subject is ended. Returns the new track."""
        if self.subject is not None:
            self.subject.machine.end(frame_idx, timestamp_s, "replaced_by_new_selection")
        track = Track(self._next_id, detection, frame_idx, timestamp_s, self.cfg, self._listeners)
        self._tracks[track.track_id] = track
        self._subject_id = track.track_id
        self._next_id += 1
        return track

    def select_best(self, detections: Sequence[Detection], timestamp_s: float, frame_idx: int = 0) -> Optional[Track]:
        """Auto-select the highest-confidence detection above the threshold (ties to the larger box), if any."""
        candidates = [d for d in detections if d.confidence >= self.cfg.confidence_threshold]
        if not candidates:
            return None
        return self.select(max(candidates, key=lambda d: (round(d.confidence, 1), max(d.size))), timestamp_s, frame_idx)

    def base_radius(self, track: Track) -> float:
        """Gate floor: at least `gate_size_scale` object sizes."""
        g = self.cfg.gate
        return max(g.gate_base_radius_px, g.gate_size_scale * max(track.size))

    def gate_radius(self, track: Track) -> float:
        """Base radius + growth per missed frame + scaled Kalman uncertainty, capped."""
        g = self.cfg.gate
        radius = (
            self.base_radius(track)
            + g.gate_growth_per_frame_lost_px * track.misses
            + g.gate_covariance_scale * track.kf.position_sigma()
        )
        return min(radius, g.gate_max_radius_px)

    def _size_ok(self, track: Track, det: Detection) -> bool:
        """Detection size is within `size_ratio_max` of the track's (larger box side)."""
        limit = self.cfg.gate.size_ratio_max
        side = max(det.size)
        return any(1.0 / limit <= side / max(ref, 1e-6) <= limit for ref in (max(track.size), track.ref_size))

    def observed_extrapolation(self, track: Track, timestamp_s: float) -> Point:
        """Last observed center moved along the observed velocity to `timestamp_s`."""
        t_last, last = track.observed[-1]
        if len(track.observed) < 2:
            return last
        t_prev, prev = track.observed[0]
        span = t_last - t_prev
        if span <= 0:
            return last
        dt = timestamp_s - t_last
        return (last[0] + (last[0] - prev[0]) / span * dt, last[1] + (last[1] - prev[1]) / span * dt)

    def _associate(
        self, track: Track, predicted: Point, gate: float, detections: Sequence[Detection], timestamp_s: float
    ) -> Optional[Detection]:
        """Nearest same-class, confident, similar-sized detection inside the gate around the Kalman
        prediction, or inside the (smaller) observed-extrapolation gate. That second gate grows with
        misses more slowly than the prediction gate, ignores Kalman uncertainty and switches off after
        `observed_gate_max_misses` (abrupt hand/bounce motion recovers within a few frames; a long
        gap means a ballistic flight, where a straight-line extrapolation only invites false positives)."""
        g = self.cfg.gate
        extrap = self.observed_extrapolation(track, timestamp_s)
        use_extrap = track.misses <= g.observed_gate_max_misses
        extrap_radius = min(
            g.gate_max_radius_px, self.base_radius(track) + g.observed_gate_growth_size_fraction * max(track.size) * track.misses
        )
        best, best_dist = None, float("inf")
        for det in detections:
            if det.confidence < self.cfg.confidence_threshold or det.class_id != track.class_id:
                continue
            if not self._size_ok(track, det):
                continue
            reach = []
            to_pred = _distance(predicted, det.center)
            if to_pred <= gate:
                reach.append(to_pred)
            if use_extrap:
                to_extrap = _distance(extrap, det.center)
                if to_extrap <= extrap_radius:
                    reach.append(to_extrap)
            if reach and min(reach) < best_dist:
                best, best_dist = det, min(reach)
        return best

    def _confirm_far(self, track: Track, detections: Sequence[Detection]) -> Optional[Detection]:
        """Wide-area reacquisition: adopt a detection near the last observed position once the same
        spot is seen twice within `reacquire_confirm_window_frames` frames. Returns the detection to
        adopt, if any."""
        g = self.cfg.gate
        anchor = track.observed[-1][1]
        search_radius = max(g.gate_max_radius_px, g.reacquire_search_size_scale * track.ref_size)
        candidates = [
            d for d in detections
            if d.confidence >= self.cfg.confidence_threshold and d.class_id == track.class_id
            and self._size_ok(track, d) and _distance(anchor, d.center) <= search_radius
        ]
        if track.pending is not None and track.updates - track.pending[0] > g.reacquire_confirm_window_frames:
            track.pending = None
        if not candidates:
            return None  # a flickering ball: keep the earlier sighting until the window runs out
        if track.pending is not None:
            seen_at, seen_xy = track.pending
            radius = g.reacquire_confirm_size_scale * track.ref_size * (track.updates - seen_at)
            agreeing = [d for d in candidates if _distance(seen_xy, d.center) <= radius]
            if agreeing:
                track.pending = None
                return min(agreeing, key=lambda d: _distance(seen_xy, d.center))
        track.pending = (track.updates, min(candidates, key=lambda d: _distance(anchor, d.center)).center)
        return None

    def update(self, frame_idx: int, timestamp_s: float, detections: Sequence[Detection]) -> TrackOutput:
        """Process one frame of detections; dt comes from timestamps only."""
        track = self.subject
        if track is None:
            return TrackOutput(frame_idx, timestamp_s)
        if track.state == TrackState.TRACK_ENDED:
            return TrackOutput(frame_idx, timestamp_s, track.track_id, track.state)

        track.updates += 1
        track.kf.predict(timestamp_s - track.last_timestamp_s)
        track.last_timestamp_s = timestamp_s
        pred = track.kf.position
        predicted: Point = (float(pred[0]), float(pred[1]))
        gate = self.gate_radius(track)

        match = self._associate(track, predicted, gate, detections, timestamp_s)
        snapped = False
        if match is not None:
            track.pending = None
        elif track.state in (TrackState.TEMPORARILY_LOST, TrackState.REACQUIRING):
            match = self._confirm_far(track, detections)
            snapped = match is not None
        if match is not None:
            if snapped:  # the old motion estimate is meaningless here: restart the filter on the ball
                track.kf = KalmanFilter(self.cfg.kalman, match.center)
                track.observed = []
            else:
                track.kf.update(match.center)
            track.size = match.size
            track.size_history.append(max(match.size))
            track.observed = (track.observed + [(timestamp_s, match.center)])[-2:]
            track.machine.on_hit(frame_idx, timestamp_s)
            bbox: Optional[Box] = match.bbox_xyxy
            center = match.center
        else:
            track.machine.on_miss(frame_idx, timestamp_s)
            bbox = None
            center = predicted

        if track.state == TrackState.TRACK_ENDED:
            return TrackOutput(frame_idx, timestamp_s, track.track_id, track.state, pred_xy=predicted, gate_radius_px=gate)

        if bbox is None:
            w, h = track.size
            bbox = (center[0] - w / 2.0, center[1] - h / 2.0, center[0] + w / 2.0, center[1] + h / 2.0)
        track.trajectory.add(timestamp_s, center[0], center[1])
        return TrackOutput(
            frame_idx=frame_idx,
            timestamp_s=timestamp_s,
            track_id=track.track_id,
            state=track.state,
            bbox_xyxy=bbox,
            pred_xy=predicted,
            confidence=match.confidence if match is not None else None,
            observed=match is not None,
            gate_radius_px=gate,
        )

    def predicted_arc(self) -> List[Point]:
        """Short look-ahead arc of the subject from the Kalman state (empty if no live subject)."""
        track = self.subject
        if track is None or track.state == TrackState.TRACK_ENDED:
            return []
        t = self.cfg.trajectory
        return predicted_arc(
            track.kf.position, track.kf.velocity, track.kf.acceleration, t.arc_horizon_s, t.arc_steps
        )
