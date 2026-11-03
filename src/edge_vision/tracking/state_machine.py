"""Track lifecycle state machine.

Driven by two counters: consecutive `hits` (while unconfirmed) and consecutive `misses`
(frames without an associated detection). Thresholds on misses are cumulative:

    misses >= frames_missed_before_lost                               -> TEMPORARILY_LOST
    misses >= frames_missed_before_lost + frames_lost_before_reacquiring -> REACQUIRING
    misses >= max_frames_before_ended                                  -> TRACK_ENDED
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional


class TrackState(str, Enum):
    DETECTED = "DETECTED"
    TRACKING = "TRACKING"
    TEMPORARILY_LOST = "TEMPORARILY_LOST"
    REACQUIRING = "REACQUIRING"
    TRACK_ENDED = "TRACK_ENDED"


@dataclass(frozen=True)
class StateMachineConfig:
    frames_to_confirm: int
    frames_missed_before_lost: int
    frames_lost_before_reacquiring: int
    max_frames_before_ended: int

    def __post_init__(self) -> None:
        if self.frames_to_confirm < 1 or self.frames_missed_before_lost < 1:
            raise ValueError("frames_to_confirm and frames_missed_before_lost must be >= 1")
        if self.frames_lost_before_reacquiring < 0:
            raise ValueError("frames_lost_before_reacquiring must be >= 0")
        if self.max_frames_before_ended <= self.frames_missed_before_lost + self.frames_lost_before_reacquiring:
            raise ValueError("max_frames_before_ended must exceed lost + reacquiring thresholds")


@dataclass(frozen=True)
class TransitionEvent:
    """A state change. `from_state` is None for track creation."""

    frame_idx: int
    timestamp_s: float
    track_id: int
    from_state: Optional[TrackState]
    to_state: TrackState
    reason: str


EventListener = Callable[[TransitionEvent], None]


class TrackStateMachine:
    """State machine for a single track. Reports every transition to its listeners."""

    def __init__(
        self,
        track_id: int,
        cfg: StateMachineConfig,
        frame_idx: int,
        timestamp_s: float,
        listeners: Optional[List[EventListener]] = None,
    ) -> None:
        self.track_id = track_id
        self.cfg = cfg
        self.state = TrackState.DETECTED
        self.hits = 0
        self.misses = 0
        self._listeners = list(listeners or [])
        self._emit(frame_idx, timestamp_s, None, TrackState.DETECTED, "created")

    def _emit(
        self, frame_idx: int, timestamp_s: float, from_state: Optional[TrackState], to_state: TrackState, reason: str
    ) -> TransitionEvent:
        event = TransitionEvent(frame_idx, timestamp_s, self.track_id, from_state, to_state, reason)
        for listener in self._listeners:
            listener(event)
        return event

    def _go(self, frame_idx: int, timestamp_s: float, to_state: TrackState, reason: str) -> None:
        old = self.state
        self.state = to_state
        self._emit(frame_idx, timestamp_s, old, to_state, reason)

    def on_hit(self, frame_idx: int, timestamp_s: float) -> None:
        """A detection was associated with this track this frame."""
        if self.state == TrackState.TRACK_ENDED:
            return
        self.misses = 0
        self.hits += 1
        if self.state == TrackState.DETECTED:
            if self.hits >= self.cfg.frames_to_confirm:
                self._go(frame_idx, timestamp_s, TrackState.TRACKING, f"confirmed_after_{self.hits}_hits")
        elif self.state in (TrackState.TEMPORARILY_LOST, TrackState.REACQUIRING):
            self._go(frame_idx, timestamp_s, TrackState.TRACKING, f"reacquired_from_{self.state.value}")

    def on_miss(self, frame_idx: int, timestamp_s: float) -> None:
        """No detection was associated with this track this frame."""
        if self.state == TrackState.TRACK_ENDED:
            return
        self.misses += 1
        cfg = self.cfg
        if self.state == TrackState.DETECTED:
            if self.misses >= cfg.frames_missed_before_lost:
                self._go(frame_idx, timestamp_s, TrackState.TRACK_ENDED, f"unconfirmed_missed_{self.misses}_frames")
            return

        if self.misses >= cfg.max_frames_before_ended:
            target, reason = TrackState.TRACK_ENDED, f"missed_{self.misses}_frames_max_exceeded"
        elif self.misses >= cfg.frames_missed_before_lost + cfg.frames_lost_before_reacquiring:
            target, reason = TrackState.REACQUIRING, f"missed_{self.misses}_frames"
        elif self.misses >= cfg.frames_missed_before_lost:
            target, reason = TrackState.TEMPORARILY_LOST, f"missed_{self.misses}_frames"
        else:
            return
        if target != self.state:
            self._go(frame_idx, timestamp_s, target, reason)

    def end(self, frame_idx: int, timestamp_s: float, reason: str) -> None:
        """Force the track to end (e.g. replaced by a new selection)."""
        if self.state != TrackState.TRACK_ENDED:
            self._go(frame_idx, timestamp_s, TrackState.TRACK_ENDED, reason)
