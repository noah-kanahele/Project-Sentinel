import time

import numpy as np
import pytest

from edge_vision.camera.frame_source import Frame, FrameSource
from edge_vision.detection.detector import Detection, Detector
from edge_vision.realtime import LatestFrame, RealtimeRunner, RealtimeStats
from edge_vision.tracking.state_machine import TrackState
from edge_vision.tracking.tracker import Tracker, TrackerConfig

FPS = 100.0


class BallSource(FrameSource):
    """Frames at FPS with timestamps; the 'image' just carries the frame index."""

    nominal_fps = FPS

    def __init__(self, n=40):
        self._i, self._n = 0, n

    def read(self):
        if self._i >= self._n:
            return None
        f = Frame(self._i, self._i / FPS, np.zeros((4, 4, 3), np.uint8))
        self._i += 1
        return f

    def close(self):
        pass


class MovingBallDetector(Detector):
    """Reports a ball moving 2 px per frame; detect() takes `delay_s`. It learns the index from call order."""

    def __init__(self, delay_s=0.0):
        self.delay_s, self.calls = delay_s, 0

    def detect(self, image):
        time.sleep(self.delay_s)
        self.calls += 1
        return [Detection((100, 100, 130, 130), 0.9, 32)]


def runner(cfg, delay_s, n=40, **kw):
    tracker = Tracker(TrackerConfig.from_dict(cfg))
    return RealtimeRunner(BallSource(n), MovingBallDetector(delay_s), tracker, pace=True, **kw)


def test_fast_detector_processes_every_frame(cfg):
    result = runner(cfg, delay_s=0.0).run()
    s = result.stats.summary()
    assert s["captured"] == s["processed"] == 40 and s["dropped"] == 0
    assert s["deadline_misses"] == 0 and s["latency_p99_ms"] < 20
    assert result.outputs[39].state == TrackState.TRACKING


def test_slow_detector_drops_frames_but_latency_stays_bounded(cfg):
    delay = 0.04  # 4 frame periods per frame at 100 FPS
    result = runner(cfg, delay_s=delay, n=60).run()
    s = result.stats.summary()
    assert s["dropped"] > 20 and s["processed"] + s["dropped"] == s["captured"] == 60
    assert s["deadline_misses"] == s["processed"]  # every frame overran its 10 ms period
    # Latest-frame-only: latency is about one detector call, not a growing backlog.
    assert s["latency_max_ms"] < 5 * delay * 1000
    ids = sorted(result.outputs)
    assert ids == sorted(set(ids)) and len(ids) == s["processed"]


def test_tracker_dt_uses_timestamps_across_dropped_frames(cfg):
    result = runner(cfg, delay_s=0.04, n=40).run()
    ts = [result.outputs[i].timestamp_s for i in sorted(result.outputs)]
    gaps = np.diff(ts)
    assert (gaps > 1.5 / FPS).any()  # some frames were skipped, so dt grew
    assert (gaps > 0).all()


def test_max_frames_stops_early(cfg):
    result = runner(cfg, delay_s=0.0, n=40).run(max_frames=5)
    assert result.stats.processed == 5


def test_latest_frame_mailbox_counts_overwrites():
    box = LatestFrame()
    for i in range(3):
        box.put((Frame(i, float(i)), 0.0))
    item, over = box.get(0.01)
    assert item[0].index == 2 and not over
    assert (box.captured, box.dropped) == (3, 2)
    box.finish()
    assert box.get(0.01) == (None, True)


def test_verdict_uses_config_thresholds(cfg):
    stats = RealtimeStats(captured=100, processed=100, dropped=0, latency_ms=[10.0] * 100, process_ms=[5.0] * 100)
    assert all(stats.verdict(cfg["realtime"]).values())
    stats.latency_ms = [500.0] * 100
    stats.dropped, stats.deadline_misses = 20, 30
    assert not any(stats.verdict(cfg["realtime"]).values())


def test_invalid_speed_rejected(cfg):
    with pytest.raises(ValueError):
        RealtimeRunner(BallSource(), MovingBallDetector(), Tracker(TrackerConfig.from_dict(cfg)), pace=True, speed=0)
