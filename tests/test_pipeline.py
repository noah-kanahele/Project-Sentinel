import csv
import importlib.util

import cv2
import numpy as np
import pytest

from edge_vision.camera.frame_source import CameraSource, VideoFileSource
from edge_vision.config import build_config
from edge_vision.detection.detector import Detection, Detector, YoloDetector
from edge_vision.motion.trajectory import TrajectoryBuffer, predicted_arc
from edge_vision.pipeline import pick_clicked, process_stream
from edge_vision.sim.synthetic import SyntheticConfig, generate_sequence
from edge_vision.telemetry.logger import FRAME_COLUMNS, TelemetryLogger
from edge_vision.tracking.state_machine import TrackState
from edge_vision.tracking.tracker import TrackOutput, Tracker, TrackerConfig
from edge_vision.visualization.overlay import draw_overlay

FPS = 30.0
SIZE = (320, 240)


class ReplayDetector(Detector):
    """Returns pre-computed detections frame by frame (stands in for YOLO)."""

    def __init__(self, per_frame):
        self._per_frame = per_frame
        self._i = 0

    def detect(self, image):
        dets = self._per_frame[self._i]
        self._i += 1
        return dets


@pytest.fixture
def clip(tmp_path, cfg):
    """A small mp4 of a ball in flight, plus per-frame detections that miss during an occlusion."""
    synth = {
        **cfg["synthetic"], "fps": FPS, "num_frames": 60, "frame_width": SIZE[0], "frame_height": SIZE[1],
        "initial_position": (20.0, 200.0), "initial_velocity": (100.0, -150.0), "gravity": 150.0,
        "occlusions": ((25, 8),), "false_positive_rate": 0.0, "miss_probability": 0.0,
    }
    seq = generate_sequence(build_config(SyntheticConfig, synth))
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
    for f in seq.frames:
        img = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
        cv2.circle(img, (int(f.gt_center[0]), int(f.gt_center[1])), 6, (255, 255, 255), -1)
        writer.write(img)
    writer.release()
    return path, ReplayDetector([f.detections for f in seq.frames])


def test_video_source_timestamps_are_monotonic_and_match_container(clip):
    path, _ = clip
    with VideoFileSource(path) as src:
        frames = list(src)
    ts = [f.timestamp_s for f in frames]
    assert len(frames) == 60 and ts[0] == pytest.approx(0.0, abs=1e-3)
    assert all(b > a for a, b in zip(ts, ts[1:]))
    assert np.mean(np.diff(ts)) == pytest.approx(1 / FPS, rel=0.05)


def test_missing_video_raises(tmp_path):
    with pytest.raises(IOError):
        VideoFileSource(tmp_path / "nope.mp4")


class FakeCapture:
    """Stands in for cv2.VideoCapture so CameraSource can be tested without a camera."""

    def __init__(self, device, opened=True, frames=3):
        self._opened, self._left, self.released = opened, frames, False

    def isOpened(self):
        return self._opened

    def set(self, prop, value):
        return True

    def get(self, prop):
        return {cv2.CAP_PROP_FPS: 30.0, cv2.CAP_PROP_FRAME_WIDTH: 64.0, cv2.CAP_PROP_FRAME_HEIGHT: 48.0}.get(prop, 0.0)

    def read(self):
        if self._left == 0:
            return False, None
        self._left -= 1
        return True, np.zeros((48, 64, 3), np.uint8)

    def release(self):
        self.released = True


def test_camera_source_timestamps_start_at_zero_and_increase(monkeypatch):
    import time

    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    with CameraSource(0) as cam:
        assert cam.nominal_fps == 30.0 and cam.size == (64, 48)
        frames = []
        for f in cam:
            frames.append(f)
            time.sleep(0.002)
    assert [f.index for f in frames] == [0, 1, 2]
    assert frames[0].timestamp_s == 0.0
    assert all(b.timestamp_s > a.timestamp_s for a, b in zip(frames, frames[1:]))


def test_camera_source_unopenable_raises(monkeypatch):
    monkeypatch.setattr(cv2, "VideoCapture", lambda device: FakeCapture(device, opened=False))
    with pytest.raises(IOError, match="camera"):
        CameraSource(0)


def test_end_to_end_video_with_csv(clip, cfg, tmp_path):
    path, detector = clip
    csv_path = tmp_path / "out.csv"
    with TelemetryLogger(csv_path) as logger, VideoFileSource(path) as src:
        tracker = Tracker(TrackerConfig.from_dict(cfg), event_sink=logger.log_event)
        outputs = process_stream(src, detector, tracker, logger)
    assert len(outputs) == 60
    assert {o.track_id for o in outputs} == {1}
    assert outputs[-1].state == TrackState.TRACKING
    assert any(o.state == TrackState.TEMPORARILY_LOST for o in outputs)

    rows = list(csv.DictReader(open(csv_path)))
    assert list(rows[0].keys()) == FRAME_COLUMNS
    assert len(rows) == 60
    assert rows[10]["state"] == "TRACKING" and float(rows[10]["total_latency_ms"]) >= float(rows[10]["detect_latency_ms"])
    assert rows[28]["confidence"] == ""  # predicted frame, no matched detection
    events = list(csv.DictReader(open(tmp_path / "out.events.csv")))
    assert [e["to_state"] for e in events][:2] == ["DETECTED", "TRACKING"]


def test_pick_clicked():
    a = Detection((0, 0, 20, 20), 0.9, 32)
    b = Detection((10, 10, 40, 40), 0.9, 32)
    assert pick_clicked([a, b], (30, 30)) is b
    assert pick_clicked([a, b], (12, 12)) is a  # overlap: nearest center
    assert pick_clicked([a, b], (200, 200)) is None


def test_overlay_draws_and_handles_no_subject():
    img = np.zeros((100, 100, 3), np.uint8)
    out = TrackOutput(3, 0.1, 1, TrackState.TRACKING, (20, 20, 40, 40), (30, 30), 0.9, True)
    draw_overlay(img, out, history=[(10, 10), (20, 20), (30, 30)], arc=[(40, 40), (50, 50)])
    assert img.any()
    blank = np.zeros((100, 100, 3), np.uint8)
    draw_overlay(blank, TrackOutput(0, 0.0))
    assert blank.any()  # "no subject" hint


def test_trajectory_buffer_velocity_and_history():
    buf = TrajectoryBuffer(history_length=5, velocity_window=4)
    assert buf.velocity() is None
    for i in range(10):
        buf.add(i * 0.1, 10 * i, -5 * i)
    assert len(buf) == 5 and buf.points()[0] == (50, -25)
    vx, vy = buf.velocity()
    assert vx == pytest.approx(100.0) and vy == pytest.approx(-50.0)


def test_predicted_arc_matches_kinematics():
    arc = predicted_arc(np.array([0.0, 0.0]), np.array([10.0, -20.0]), np.array([0.0, 40.0]), 1.0, 4)
    assert arc[-1] == pytest.approx((10.0, 0.0))
    assert len(arc) == 4


def test_yolo_detector_errors_clearly_without_ultralytics():
    if importlib.util.find_spec("ultralytics") is not None:
        pytest.skip("ultralytics installed")
    with pytest.raises(ImportError, match="ultralytics"):
        YoloDetector("yolo11n.pt", 32, 0.25, 640, "cpu")


def test_progress_callback_counts_every_frame(clip, cfg):
    path, detector = clip
    seen = []
    with VideoFileSource(path) as src:
        assert src.frame_count == 60
        process_stream(src, detector, Tracker(TrackerConfig.from_dict(cfg)), progress=lambda n, fin: seen.append((n, fin)))
    assert [n for n, _ in seen[:60]] == list(range(1, 61))
    assert [fin for _, fin in seen] == [False] * 60 + [True]  # one extra call when the stream ends


def test_keyboard_interrupt_returns_partial_results(clip, cfg):
    path, detector = clip

    calls = []

    def stop_at_ten(done, finished):
        calls.append(finished)
        if done == 10:
            raise KeyboardInterrupt

    with VideoFileSource(path) as src:
        outputs = process_stream(src, detector, Tracker(TrackerConfig.from_dict(cfg)), progress=stop_at_ten)
    assert len(outputs) == 10 and True not in calls  # an interrupted run is never reported as finished
