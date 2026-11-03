"""Ground-truth labels for a clip: where the ball is in each frame, or that it is not visible."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

from edge_vision.camera.frame_source import FrameSource
from edge_vision.detection.detector import Detector

COLUMNS = ["frame_idx", "visible", "cx", "cy", "w", "h"]


@dataclass(frozen=True)
class Label:
    """`visible` means the ball is in frame and not hidden. cx, cy, w, h are pixels (None if not visible)."""

    frame_idx: int
    visible: bool
    cx: Optional[float] = None
    cy: Optional[float] = None
    w: Optional[float] = None
    h: Optional[float] = None

    def __post_init__(self) -> None:
        if self.visible and (self.cx is None or self.cy is None):
            raise ValueError(f"frame {self.frame_idx}: a visible label needs cx and cy")

    @property
    def size(self) -> Optional[float]:
        """Larger box side, if known."""
        return max(self.w, self.h) if self.w is not None and self.h is not None else None


def save_labels(path: Union[str, Path], labels: Iterable[Label]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for lb in sorted(labels, key=lambda x: x.frame_idx):
            writer.writerow(
                [lb.frame_idx, int(lb.visible)] + ["" if v is None else round(v, 2) for v in (lb.cx, lb.cy, lb.w, lb.h)]
            )


def load_labels(path: Union[str, Path]) -> Dict[int, Label]:
    """Read a labels CSV into {frame_idx: Label}."""

    def num(text: str) -> Optional[float]:
        return float(text) if text.strip() != "" else None

    labels: Dict[int, Label] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            visible = row["visible"].strip() in ("1", "true", "True")
            idx = int(row["frame_idx"])
            labels[idx] = Label(idx, visible, num(row["cx"]), num(row["cy"]), num(row["w"]), num(row["h"]))
    return labels


def autolabel(source: FrameSource, detector: Detector) -> List[Label]:
    """First-pass labels: the highest-confidence detection per frame (review them by hand)."""
    labels: List[Label] = []
    for frame in source:
        dets = detector.detect(frame.image)
        if not dets:
            labels.append(Label(frame.index, False))
            continue
        best = max(dets, key=lambda d: d.confidence)
        (cx, cy), (w, h) = best.center, best.size
        labels.append(Label(frame.index, True, cx, cy, w, h))
    return labels
