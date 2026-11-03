import cv2
import pytest

from edge_vision.detection.color import ColorBlobDetector
from edge_vision.pipeline import process_stream
from edge_vision.tracking.tracker import Tracker, TrackerConfig
from edge_vision.visualization.analysis import AnalysisWriter
from test_color_detector import RenderedSource, SIZE

SOURCE_FPS = 30.0


def run(cfg, path, slowdown, n=50):
    tracker = Tracker(TrackerConfig.from_dict(cfg))
    with AnalysisWriter(path, SOURCE_FPS, (SIZE[1], SIZE[0]), slowdown) as writer:
        outputs = process_stream(RenderedSource(n=n), ColorBlobDetector.from_config(cfg), tracker, analysis=writer)
    return outputs


def test_analysis_video_has_every_frame_at_slowed_rate(cfg, tmp_path):
    path = tmp_path / "analysis.mp4"
    outputs = run(cfg, path, slowdown=4.0)
    cap = cv2.VideoCapture(str(path))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == len(outputs) == 50
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(SOURCE_FPS / 4.0, abs=0.1)
    assert (int(cap.get(3)), int(cap.get(4))) == (SIZE[1], SIZE[0])


def test_analysis_frames_are_annotated(cfg, tmp_path):
    path = tmp_path / "analysis.mp4"
    run(cfg, path, slowdown=2.0)
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 40)  # after the occlusion; trail and HUD must be drawn
    ok, frame = cap.read()
    assert ok
    top_left = frame[:70, :400]  # HUD region over a flat grey background
    assert top_left.std() > 5


def test_invalid_slowdown_rejected(tmp_path):
    with pytest.raises(ValueError):
        AnalysisWriter(tmp_path / "x.mp4", 30.0, (640, 480), 0.0)
