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
    # A 10 ms frame period is tight on a loaded machine, so tolerate a stray scheduler hiccup.
    assert s["captured"] == 40 and s["processed"] >= 36 and s["dropped"] <= 4
    assert s["processed"] + s["dropped"] == s["captured"]
    assert s["latency_p99_ms"] < 50
    assert result.outputs[max(result.outputs)].state == TrackState.TRACKING


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
        box.put((Frame(i, float(i), np.zeros((1, 1, 3), np.uint8)), 0.0))
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


class SlowRecorder:
    """Stands in for FrameRecorder: writing takes a while (a slow disk or encoder)."""

    def __init__(self, delay_s):
        self.delay_s, self.frames, self.closed = delay_s, 0, False

    def write(self, image, timestamp_s):
        time.sleep(self.delay_s)
        self.frames += 1

    def close(self):
        self.closed = True


class ExplodingDetector(Detector):
    def detect(self, image):
        raise RuntimeError("detector blew up")


def test_worker_exception_is_raised_to_the_caller(cfg):
    tracker = Tracker(TrackerConfig.from_dict(cfg))
    with pytest.raises(RuntimeError, match="detector blew up"):
        RealtimeRunner(BallSource(20), ExplodingDetector(), tracker, pace=True).run()


def test_dropped_frame_indices_are_reported(cfg):
    result = runner(cfg, delay_s=0.04, n=60).run()
    s = result.stats
    assert len(s.dropped_indices) == s.dropped > 0
    assert set(s.dropped_indices).isdisjoint(result.outputs)  # dropped frames were never processed
    assert all(0 <= i < 60 for i in s.dropped_indices)


def test_slow_recording_counts_toward_loop_time_and_deadlines(cfg):
    """Detection is instant, but writing the overlay takes 3 frame periods: the whole loop overruns."""
    rec = SlowRecorder(delay_s=0.03)
    result = runner(cfg, delay_s=0.0, n=40, record_overlay=rec).run()
    s = result.stats.summary()
    assert s["process_p99_ms"] < 10  # detect + track alone looks fine...
    assert s["loop_p99_ms"] >= 30  # ...the full loop does not
    assert s["deadline_misses"] > 0 and s["dropped"] > 0
    assert rec.closed and rec.frames == result.stats.processed


def test_recorders_are_closed_even_when_the_worker_fails(cfg):
    raw, overlay = SlowRecorder(0.0), SlowRecorder(0.0)
    tracker = Tracker(TrackerConfig.from_dict(cfg))
    with pytest.raises(RuntimeError):
        RealtimeRunner(BallSource(20), ExplodingDetector(), tracker, pace=True, record_raw=raw, record_overlay=overlay).run()
    assert raw.closed and overlay.closed
