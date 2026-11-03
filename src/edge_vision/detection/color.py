"""HSV color-blob detector: finds a uniformly colored ball with plain OpenCV (no model, CPU-cheap)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Mapping

import cv2
import numpy as np

from edge_vision.config import build_config
from edge_vision.detection.detector import Detection, Detector

CIRCLE_FILL_RATIO = math.pi / 4  # circle area / bounding-square area


@dataclass(frozen=True)
class ColorDetectorConfig:
    class_id: int
    hue_min: int
    hue_max: int
    saturation_min: int
    value_min: int
    morph_kernel_px: int
    min_area_px: float
    min_fill_ratio: float
    max_aspect_ratio: float
    confident_area_px: float


class ColorBlobDetector(Detector):
    """Thresholds an HSV range, then keeps round-ish blobs as detections.

    Confidence is the blob's fill ratio relative to a perfect circle (capped at 1), scaled down
    for blobs smaller than `confident_area_px`, so a tiny perfectly-round speck cannot outrank a
    real ball when the tracker picks the highest-confidence detection. If
    `hue_min > hue_max` the hue range wraps around 0 (reds and pinks).
    """

    def __init__(self, cfg: ColorDetectorConfig) -> None:
        self.cfg = cfg
        k = max(1, cfg.morph_kernel_px)
        self._kernel = np.ones((k, k), np.uint8)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> ColorBlobDetector:
        """Build from a full default.yaml-style dict (class id comes from the `detection` section)."""
        section = {**cfg["color_detector"], "class_id": cfg["detection"]["class_id"]}
        return cls(build_config(ColorDetectorConfig, section))

    def _mask(self, image: np.ndarray) -> np.ndarray:
        c = self.cfg
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lo_sv = (c.saturation_min, c.value_min)
        if c.hue_min <= c.hue_max:
            mask = cv2.inRange(hsv, (c.hue_min, *lo_sv), (c.hue_max, 255, 255))
        else:  # wrap-around range
            low = cv2.inRange(hsv, (0, *lo_sv), (c.hue_max, 255, 255))
            high = cv2.inRange(hsv, (c.hue_min, *lo_sv), (179, 255, 255))
            mask = cv2.bitwise_or(low, high)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)

    def detect(self, image: np.ndarray) -> List[Detection]:
        c = self.cfg
        contours, _ = cv2.findContours(self._mask(image), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < c.min_area_px:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            fill = area / float(w * h)
            if fill < c.min_fill_ratio or max(w, h) / float(min(w, h)) > c.max_aspect_ratio:
                continue
            confidence = min(1.0, fill / CIRCLE_FILL_RATIO) * min(1.0, area / c.confident_area_px)
            detections.append(Detection((float(x), float(y), float(x + w), float(y + h)), confidence, c.class_id))
        return detections
