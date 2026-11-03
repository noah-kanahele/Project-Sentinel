import cv2
import numpy as np
import pytest

from edge_vision.camera.frame_source import VideoFileSource
from edge_vision.detection.color import ColorBlobDetector
from edge_vision.realtime import RealtimeRunner
from edge_vision.recording import FrameRecorder, load_timestamps, sidecar_path
from edge_vision.tracking.tracker import Tracker, TrackerConfig
from test_color_detector import RenderedSource

SIZE = (64, 48)  # w, h


def image():
    return np.full((SIZE[1], SIZE[0], 3), 90, np.uint8)


def test_recorder_writes_video_and_sidecar_that_replay_uses(tmp_path):
    path = tmp_path / "rec.mp4"
    times = [0.0, 0.031, 0.070, 0.101, 0.140, 0.171]  # irregular, like a real camera
    rec = FrameRecorder(path, 30.0, SIZE, sidecar=True)
    for t in times:
        rec.write(image(), t)
    rec.close()
    assert rec.frames == 6 and sidecar_path(path).name == "rec.timestamps.csv"
    assert load_timestamps(path) == pytest.approx(times)
    with VideoFileSource(path) as src:
        replay = [f.timestamp_s for f in src]
    assert replay == pytest.approx(times)  # not the uniform 1/30 s the container would give


def test_video_without_sidecar_falls_back_to_container_timestamps(tmp_path):
    path = tmp_path / "plain.mp4"
    rec = FrameRecorder(path, 30.0, SIZE, sidecar=False)
    for t in range(5):
        rec.write(image(), float(t))
    rec.close()
    assert load_timestamps(path) is None
    with VideoFileSource(path) as src:
        ts = [f.timestamp_s for f in src]
    assert np.mean(np.diff(ts)) == pytest.approx(1 / 30, rel=0.1)


class SmallSource(RenderedSource):
    nominal_fps = 100.0


def test_realtime_runner_records_raw_and_overlay(cfg, tmp_path):
    raw = FrameRecorder(tmp_path / "raw.mp4", 100.0, (640, 480), sidecar=True)
    overlay = FrameRecorder(tmp_path / "ov.mp4", 100.0, (640, 480), sidecar=False)
    tracker = Tracker(TrackerConfig.from_dict(cfg))
    result = RealtimeRunner(
        SmallSource(n=30, fps=100.0), ColorBlobDetector.from_config(cfg), tracker,
        pace=True, record_raw=raw, record_overlay=overlay,
    ).run()
    assert raw.frames == result.stats.captured == 30  # every captured frame is recorded
    assert overlay.frames == result.stats.processed
    assert len(load_timestamps(tmp_path / "raw.mp4")) == 30
    cap = cv2.VideoCapture(str(tmp_path / "ov.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == overlay.frames
