"""Detection types and detectors. ultralytics is imported lazily, only by YoloDetector."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class Detection:
    """One detected object: pixel-space xyxy box, confidence, class id."""

    bbox_xyxy: Tuple[float, float, float, float]
    confidence: float
    class_id: int

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def size(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return (x2 - x1, y2 - y1)


class Detector(ABC):
    """Maps an image to detections."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Detection]:
        """Return detections for one BGR image."""


class YoloDetector(Detector):
    """Wraps an ultralytics YOLO model, filtered to a single class."""

    def __init__(
        self,
        model_path: str,
        class_id: int,
        confidence_threshold: float,
        image_size: int,
        device: str,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "YoloDetector needs the optional 'ultralytics' package: pip install ultralytics"
            ) from exc
        self._model = YOLO(model_path)
        self._class_id = class_id
        self._conf = confidence_threshold
        self._imgsz = image_size
        self._device = device

    def detect(self, image: np.ndarray) -> List[Detection]:
        result = self._model.predict(
            image,
            conf=self._conf,
            classes=[self._class_id],
            imgsz=self._imgsz,
            device=self._device,
            verbose=False,
        )[0]
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()
        return [
            Detection(tuple(float(v) for v in b), float(c), int(k))  # type: ignore[arg-type]
            for b, c, k in zip(xyxy, confs, classes)
        ]
