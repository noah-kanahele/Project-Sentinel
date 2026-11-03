"""OpenCV drawing for tracker output. Drawing only; no windows, no state."""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import cv2
import numpy as np

from edge_vision.tracking.state_machine import TrackState
from edge_vision.tracking.tracker import TrackOutput

Point = Tuple[float, float]

# BGR colors per state.
STATE_COLORS: Dict[TrackState, Tuple[int, int, int]] = {
    TrackState.DETECTED: (0, 200, 255),
    TrackState.TRACKING: (0, 220, 0),
    TrackState.TEMPORARILY_LOST: (0, 140, 255),
    TrackState.REACQUIRING: (0, 0, 255),
    TrackState.TRACK_ENDED: (160, 160, 160),
}
HISTORY_COLOR = (255, 200, 0)
ARC_COLOR = (255, 0, 255)
CANDIDATE_COLOR = (200, 200, 200)


def _pt(p: Point) -> Tuple[int, int]:
    return (int(round(p[0])), int(round(p[1])))


def draw_overlay(
    image: np.ndarray,
    output: TrackOutput,
    history: Sequence[Point] = (),
    arc: Sequence[Point] = (),
    candidates: Sequence[Tuple[float, float, float, float]] = (),
) -> np.ndarray:
    """Draw (in place) the track box, id/state/confidence, history polyline, predicted arc and candidate boxes."""
    for x1, y1, x2, y2 in candidates:
        cv2.rectangle(image, _pt((x1, y1)), _pt((x2, y2)), CANDIDATE_COLOR, 1)

    if len(history) >= 2:
        cv2.polylines(image, [np.array([_pt(p) for p in history], dtype=np.int32)], False, HISTORY_COLOR, 2)  # type: ignore[list-item]
    if arc:
        cv2.polylines(image, [np.array([_pt(p) for p in arc], dtype=np.int32)], False, ARC_COLOR, 1, cv2.LINE_AA)  # type: ignore[list-item]

    if output.state is None:
        _label(image, "no subject - click a detection", (10, 24), CANDIDATE_COLOR)
        return image
    color = STATE_COLORS[output.state]
    if output.bbox_xyxy is not None:
        x1, y1, x2, y2 = output.bbox_xyxy
        thickness = 2 if output.observed else 1  # thin = predicted, not observed
        cv2.rectangle(image, _pt((x1, y1)), _pt((x2, y2)), color, thickness)
        conf = f" {output.confidence:.2f}" if output.confidence is not None else ""
        _label(image, f"ID {output.track_id} {output.state.value}{conf}", (int(x1), max(int(y1) - 6, 12)), color)
    else:
        _label(image, f"ID {output.track_id} {output.state.value}", (10, 24), color)
    if output.pred_xy is not None and not output.observed:
        cv2.circle(image, _pt(output.pred_xy), 4, ARC_COLOR, -1)
    return image


def _label(image: np.ndarray, text: str, origin: Tuple[int, int], color: Tuple[int, int, int]) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
