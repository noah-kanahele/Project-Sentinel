"""CSV telemetry: one row per frame, plus a separate transition-event log."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import IO, List, Optional, Union

from edge_vision.tracking.state_machine import TransitionEvent
from edge_vision.tracking.tracker import TrackOutput

FRAME_COLUMNS = [
    "frame_idx", "timestamp_s", "track_id", "state", "cx", "cy",
    "pred_cx", "pred_cy", "confidence", "detect_latency_ms", "total_latency_ms",
]
EVENT_COLUMNS = ["frame_idx", "timestamp_s", "track_id", "from_state", "to_state", "reason"]


def _fmt(value: object) -> object:
    """Blank for missing values, rounded for floats."""
    if value is None:
        return ""
    if isinstance(value, float):
        return round(value, 4)
    return value


class TelemetryLogger:
    """Collects events in memory and optionally streams frames/events to CSV.

    If `frame_csv` is given, events go to `<frame_csv stem>.events.csv` beside it unless
    `event_csv` is set explicitly.
    """

    def __init__(
        self,
        frame_csv: Optional[Union[str, Path]] = None,
        event_csv: Optional[Union[str, Path]] = None,
    ) -> None:
        self.events: List[TransitionEvent] = []
        self._frame_file: Optional[IO[str]] = None
        self._event_file: Optional[IO[str]] = None
        self._frame_writer = None
        self._event_writer = None
        if frame_csv is not None:
            frame_path = Path(frame_csv)
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            self._frame_file = open(frame_path, "w", newline="")
            self._frame_writer = csv.writer(self._frame_file)
            self._frame_writer.writerow(FRAME_COLUMNS)
            if event_csv is None:
                event_csv = frame_path.with_suffix(".events.csv")
        if event_csv is not None:
            self._event_file = open(event_csv, "w", newline="")
            self._event_writer = csv.writer(self._event_file)
            self._event_writer.writerow(EVENT_COLUMNS)

    def log_event(self, event: TransitionEvent) -> None:
        """Record a state transition. Usable directly as a Tracker event sink."""
        self.events.append(event)
        if self._event_writer is not None:
            self._event_writer.writerow([
                event.frame_idx, _fmt(event.timestamp_s), event.track_id,
                event.from_state.value if event.from_state else "", event.to_state.value, event.reason,
            ])

    def log_frame(
        self, output: TrackOutput, detect_latency_ms: Optional[float] = None, total_latency_ms: Optional[float] = None
    ) -> None:
        """Write one per-frame row."""
        if self._frame_writer is None:
            return
        center = output.center
        pred = output.pred_xy
        self._frame_writer.writerow([
            output.frame_idx, _fmt(output.timestamp_s), _fmt(output.track_id),
            output.state.value if output.state else "",
            _fmt(center[0] if center else None), _fmt(center[1] if center else None),
            _fmt(pred[0] if pred else None), _fmt(pred[1] if pred else None),
            _fmt(output.confidence), _fmt(detect_latency_ms), _fmt(total_latency_ms),
        ])

    def close(self) -> None:
        for f in (self._frame_file, self._event_file):
            if f is not None:
                f.close()

    def __enter__(self) -> TelemetryLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
