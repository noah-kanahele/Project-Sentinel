import numpy as np
import pytest

from edge_vision.config import build_config
from edge_vision.detection.detector import Detection
from edge_vision.sim.runner import run_sequence
from edge_vision.sim.synthetic import SyntheticConfig, generate_sequence
from edge_vision.tracking.state_machine import TrackState as S
from edge_vision.tracking.tracker import Tracker, TrackerConfig


def sequence(cfg, **overrides):
    base = {**cfg["synthetic"], "false_positive_rate": 0.0, "miss_probability": 0.0}
    return generate_sequence(build_config(SyntheticConfig, {**base, **overrides}))


def det(cx, cy, conf=0.9, size=24.0, cls=32):
    return Detection((cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2), conf, cls)


def test_survives_occlusion_shorter_than_max_and_keeps_id(cfg):
    length = 25
    assert length < cfg["state_machine"]["max_frames_before_ended"]
    seq = sequence(cfg, occlusions=((40, length),))
    out = run_sequence(cfg, seq, auto_select=False).outputs
    assert {o.track_id for o in out} == {1}
    assert any(o.state == S.TEMPORARILY_LOST for o in out)
    assert any(o.state == S.REACQUIRING for o in out)
    after = out[40 + length + 5]
    assert after.state == S.TRACKING and after.observed and after.track_id == 1


def test_prediction_error_during_occlusion_is_bounded(cfg):
    seq = sequence(cfg, occlusions=((40, 20),))
    out = run_sequence(cfg, seq).outputs
    gt = seq.gt_centers()
    errs = [np.linalg.norm(np.array(o.pred_xy) - gt[o.frame_idx]) for o in out[40:60]]
    assert max(errs) < cfg["tracker"]["gate_max_radius_px"] / 2


def test_ends_track_past_max_and_stays_ended(cfg):
    max_frames = cfg["state_machine"]["max_frames_before_ended"]
    seq = sequence(cfg, num_frames=140, occlusions=((30, max_frames + 10),))
    result = run_sequence(cfg, seq, auto_select=False)
    out = result.outputs
    assert out[30 + max_frames - 1].state == S.TRACK_ENDED
    assert out[-1].state == S.TRACK_ENDED and out[-1].bbox_xyxy is None
    assert [e.to_state for e in result.events if e.to_state == S.TRACK_ENDED]


def test_auto_select_starts_new_id_after_end(cfg):
    max_frames = cfg["state_machine"]["max_frames_before_ended"]
    seq = sequence(cfg, num_frames=140, occlusions=((30, max_frames + 10),))
    out = run_sequence(cfg, seq, auto_select=True).outputs
    assert out[-1].track_id == 2 and out[-1].state in (S.DETECTED, S.TRACKING)


def test_false_positive_outside_gate_is_not_adopted(cfg):
    tracker = Tracker(TrackerConfig.from_dict(cfg))
    dt = 1 / 60
    pos = np.array([200.0, 500.0])
    vel = np.array([300.0, 0.0])
    tracker.select(det(*pos), 0.0, 0)
    far_fp = det(900.0, 100.0)  # nowhere near the ball's path
    outputs = []
    for i in range(1, 21):
        pos = pos + vel * dt
        visible = i <= 8 or i > 14  # ball hidden for frames 9..14
        dets = ([det(*pos)] if visible else []) + [far_fp]
        outputs.append(tracker.update(i, i * dt, dets))
    gap = outputs[8:14]
    assert all(not o.observed for o in gap)
    assert all(o.track_id == 1 for o in outputs)
    assert all(o.center is None or np.linalg.norm(np.array(o.center) - [900, 100]) > 100 for o in outputs)
    assert outputs[-1].observed and outputs[-1].state == S.TRACKING


def test_detection_inside_gate_inherits_id(cfg):
    tracker = Tracker(TrackerConfig.from_dict(cfg))
    dt = 1 / 60
    tracker.select(det(100, 100), 0.0, 0)
    for i in range(1, 6):
        tracker.update(i, i * dt, [det(100 + 5 * i, 100)])
    # Hide the ball for 8 frames, then show a detection near the extrapolated path.
    for i in range(6, 14):
        out = tracker.update(i, i * dt, [])
    out = tracker.update(14, 14 * dt, [det(100 + 5 * 14 + 10, 105)])
    assert out.observed and out.track_id == 1 and out.state == S.TRACKING


def test_gate_grows_with_missed_frames(cfg):
    tracker = Tracker(TrackerConfig.from_dict(cfg))
    tracker.select(det(100, 100), 0.0, 0)
    for i in range(1, 5):  # confirm the track so misses don't end it as unconfirmed
        tracker.update(i, i / 60, [det(100, 100)])
    radii = [tracker.update(i, i / 60, []).gate_radius_px for i in range(5, 10)]
    assert all(b > a for a, b in zip(radii, radii[1:]))


def test_select_replaces_subject_and_ignores_other_class(cfg):
    tracker = Tracker(TrackerConfig.from_dict(cfg))
    first = tracker.select(det(100, 100), 0.0, 0)
    second = tracker.select(det(400, 400), 0.1, 6)
    assert second.track_id == first.track_id + 1 and tracker.subject is second
    assert first.state == S.TRACK_ENDED
    out = tracker.update(6, 0.1, [det(400, 400, cls=0)])  # wrong class
    assert not out.observed


def test_low_confidence_detections_ignored(cfg):
    tracker = Tracker(TrackerConfig.from_dict(cfg))
    tracker.select(det(100, 100), 0.0, 0)
    low = cfg["detection"]["confidence_threshold"] / 2
    assert not tracker.update(1, 1 / 60, [det(100, 100, conf=low)]).observed


def test_dt_comes_from_timestamps_not_frame_count(cfg):
    """Same positions at half the frame rate => double the speed => different prediction."""
    def run(dt):
        tracker = Tracker(TrackerConfig.from_dict(cfg))
        tracker.select(det(100, 100), 0.0, 0)
        for i in range(1, 15):
            tracker.update(i, i * dt, [det(100 + 5 * i, 100)])
        return tracker.subject.kf.velocity[0]

    v_fast, v_slow = run(1 / 60), run(1 / 30)
    assert np.isclose(v_fast / v_slow, 2.0, rtol=0.15)


def test_no_subject_returns_empty_output(cfg):
    out = Tracker(TrackerConfig.from_dict(cfg)).update(0, 0.0, [det(1, 1)])
    assert out.track_id is None and out.state is None and out.bbox_xyxy is None


def test_gate_scales_with_object_size(cfg):
    """A big object gets a wider gate than the fixed pixel floor (a 24 px ball stays at the floor)."""
    small, big = Tracker(TrackerConfig.from_dict(cfg)), Tracker(TrackerConfig.from_dict(cfg))
    small.select(det(100, 100, size=24.0), 0.0, 0)
    big.select(det(100, 100, size=200.0), 0.0, 0)
    floor = cfg["tracker"]["gate_base_radius_px"]
    cov = cfg["tracker"]["gate_covariance_scale"] * small.subject.kf.position_sigma()
    assert small.gate_radius(small.subject) == pytest.approx(floor + cov)
    assert big.gate_radius(big.subject) >= cfg["tracker"]["gate_size_scale"] * 200.0


def test_tracker_follows_sudden_reversal_like_a_throw(cfg):
    """A hand drops the ball, then flings it upward. The smoothed Kalman velocity still says
    'falling', but the last observed velocity already says 'rising', and the ball stays tracked."""
    tracker = Tracker(TrackerConfig.from_dict(cfg))
    dt, size = 1 / 30, 100.0
    y = 300.0
    tracker.select(det(400, y, size=size), 0.0, 0)
    steps = [20.0] * 25 + [-20.0, -50.0] + [-75.0] * 8  # px per frame; reverses within two frames
    observed_after = []
    for i, step in enumerate(steps, start=1):
        y += step
        out = tracker.update(i, i * dt, [det(400, y, size=size)])
        if i > 25:
            observed_after.append(out.observed)
    assert all(observed_after) and out.track_id == 1 and out.state == S.TRACKING


def test_size_mismatch_detection_is_rejected(cfg):
    tracker = Tracker(TrackerConfig.from_dict(cfg))
    tracker.select(det(300, 300, size=100.0), 0.0, 0)
    for i in range(1, 4):
        tracker.update(i, i / 30, [det(300, 300, size=100.0)])
    speck = det(305, 300, size=16.0)  # right on the ball, but 1/6 of its size
    assert not tracker.update(4, 4 / 30, [speck]).observed
    assert tracker.update(5, 5 / 30, [det(300, 300, size=60.0)]).observed  # within 2x is fine


def test_observed_extrapolation_follows_last_two_hits(cfg):
    tracker = Tracker(TrackerConfig.from_dict(cfg))
    tracker.select(det(100, 100), 0.0, 0)
    tracker.update(0, 0.0, [det(100, 100)])
    tracker.update(1, 0.1, [det(120, 100)])
    t = tracker.subject
    assert tracker.observed_extrapolation(t, 0.2) == pytest.approx((140.0, 100.0))
    assert tracker.observed_extrapolation(t, 0.1) == pytest.approx((120.0, 100.0))
