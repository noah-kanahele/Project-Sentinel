import cv2
import numpy as np
import pytest

from edge_vision.camera.frame_source import Frame, FrameSource
from edge_vision.detection.color import ColorBlobDetector
from edge_vision.pipeline import process_stream
from edge_vision.tracking.state_machine import TrackState
from edge_vision.tracking.tracker import Tracker, TrackerConfig

GREEN = (60, 200, 60)  # BGR, hue ~60 in OpenCV after conversion of a pure-ish green
SIZE = (480, 640)  # h, w


def canvas():
    return np.full((*SIZE, 3), 200, np.uint8)  # light grey background, zero saturation


def hsv_bgr(h, s=200, v=200):
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0])


@pytest.fixture
def detector(cfg):
    return ColorBlobDetector.from_config(cfg)


def test_finds_ball_center_and_class(detector, cfg):
    img = canvas()
    cv2.circle(img, (300, 200), 25, hsv_bgr(45), -1)
    dets = detector.detect(img)
    assert len(dets) == 1
    cx, cy = dets[0].center
    assert abs(cx - 300) < 3 and abs(cy - 200) < 3
    assert dets[0].class_id == cfg["detection"]["class_id"]
    assert 0.9 <= dets[0].confidence <= 1.0


def test_empty_background_has_no_detections(detector):
    assert detector.detect(canvas()) == []


def test_wrong_color_ignored(detector):
    img = canvas()
    cv2.circle(img, (300, 200), 25, hsv_bgr(120), -1)  # blue
    assert detector.detect(img) == []


def test_specks_and_elongated_blobs_rejected(detector):
    img = canvas()
    cv2.circle(img, (100, 100), 4, hsv_bgr(45), -1)  # too small
    cv2.rectangle(img, (200, 300), (400, 320), hsv_bgr(45), -1)  # long thin bar
    assert detector.detect(img) == []


def test_two_balls_give_two_detections(detector):
    img = canvas()
    cv2.circle(img, (100, 100), 25, hsv_bgr(45), -1)
    cv2.circle(img, (400, 300), 30, hsv_bgr(50), -1)
    assert len(detector.detect(img)) == 2


def test_wraparound_hue_range_for_red(cfg):
    cfg["color_detector"].update(hue_min=170, hue_max=8)
    det = ColorBlobDetector.from_config(cfg)
    img = canvas()
    cv2.circle(img, (100, 100), 25, hsv_bgr(175), -1)
    cv2.circle(img, (300, 100), 25, hsv_bgr(3), -1)
    cv2.circle(img, (200, 300), 25, hsv_bgr(45), -1)  # green, must not match
    assert len(det.detect(img)) == 2


class RenderedSource(FrameSource):
    """Frames of a green ball moving in a parabola, with a gap where it is hidden."""

    def __init__(self, hidden=range(25, 31), n=50, fps=30.0):
        self._i, self._n, self._fps, self._hidden = 0, n, fps, set(hidden)

    def read(self):
        if self._i >= self._n:
            return None
        t = self._i / self._fps
        img = canvas()
        if self._i not in self._hidden:
            x, y = 60 + 250 * t, 400 - 420 * t + 0.5 * 450 * t * t
            cv2.circle(img, (int(x), int(y)), 18, hsv_bgr(45), -1)
        frame = Frame(self._i, t, img)
        self._i += 1
        return frame

    def close(self):
        pass


def test_color_detector_feeds_tracker_through_occlusion(detector, cfg):
    tracker = Tracker(TrackerConfig.from_dict(cfg))
    outputs = process_stream(RenderedSource(), detector, tracker)
    assert {o.track_id for o in outputs} == {1}
    assert any(o.state == TrackState.TEMPORARILY_LOST for o in outputs)
    assert outputs[-1].state == TrackState.TRACKING and outputs[-1].observed


def test_tiny_speck_cannot_outrank_the_ball(detector, cfg):
    """Regression: a 16 px speck used to score 1.0 and beat a real ball (0.97) in auto-select."""
    img = canvas()
    cv2.circle(img, (300, 200), 55, hsv_bgr(45), -1)  # the ball
    cv2.rectangle(img, (100, 100), (116, 116), hsv_bgr(45), -1)  # 16 x 16 speck, fill ratio 1.0
    dets = detector.detect(img)
    assert len(dets) == 2
    ball, speck = sorted(dets, key=lambda d: d.size[0], reverse=True)
    assert ball.confidence > speck.confidence
    assert max(dets, key=lambda d: d.confidence) is ball
