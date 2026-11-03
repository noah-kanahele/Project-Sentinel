import math

import pytest

from edge_vision.config import build_config
from edge_vision.detection.color import ColorBlobDetector
from edge_vision.evaluation import AccuracyConfig, Label, autolabel, load_labels, save_labels, score_clip, wilson_interval
from edge_vision.sim.runner import run_sequence
from edge_vision.sim.synthetic import SyntheticConfig, generate_sequence
from test_color_detector import RenderedSource


@pytest.fixture
def acc_cfg(cfg):
    return build_config(AccuracyConfig, cfg["accuracy"])


def scenario(cfg, **overrides):
    base = {**cfg["synthetic"], "false_positive_rate": 0.0, "miss_probability": 0.0}
    seq = generate_sequence(build_config(SyntheticConfig, {**base, **overrides}))
    size = seq.config.ball_size_px
    labels = {f.frame_idx: Label(f.frame_idx, not f.occluded, *(f.gt_center + (size, size) if not f.occluded else (None,) * 4)) for f in seq.frames}
    outputs = {o.frame_idx: o for o in run_sequence(cfg, seq, auto_select=False).outputs}
    return labels, outputs


def test_wilson_interval_known_values():
    lo, hi = wilson_interval(50, 100, 1.96)
    assert lo == pytest.approx(0.404, abs=0.002) and hi == pytest.approx(0.596, abs=0.002)
    assert wilson_interval(100, 100, 1.96)[0] == pytest.approx(0.963, abs=0.002)
    assert math.isnan(wilson_interval(0, 0, 1.96)[0])


def test_clean_run_holds_ball_but_ci_lower_bound_is_below_99(cfg, acc_cfg):
    labels, outputs = scenario(cfg, occlusions=())
    r = score_clip(outputs, labels, acc_cfg)
    assert r.visible_frames == 150 and r.hold_rate > 0.97
    assert r.meets_target
    assert not r.meets_target_confidently  # 150 frames is too few to prove 99% even at a perfect score
    assert r.id_switches == 0 and r.tracks_started == 1 and r.false_observed_frames == 0


def test_occlusion_gap_is_scored_as_reacquired(cfg, acc_cfg):
    labels, outputs = scenario(cfg, occlusions=((40, 15),))
    r = score_clip(outputs, labels, acc_cfg)
    assert len(r.gaps) == 1 and r.gaps[0].length == 15 and r.gaps[0].evaluable
    assert r.reacquisition() == (1, 1) and r.gaps[0].frames_to_reacquire <= 1
    assert r.visible_frames == 135


def test_track_ended_during_long_gap_fails_reacquisition(cfg, acc_cfg):
    n = cfg["state_machine"]["max_frames_before_ended"] + 10
    labels, outputs = scenario(cfg, num_frames=150, occlusions=((30, n),))
    r = score_clip(outputs, labels, acc_cfg)
    assert r.reacquisition() == (0, 1)
    assert r.miss_no_track > 0


def test_dropped_frames_count_as_misses(cfg, acc_cfg):
    labels, outputs = scenario(cfg, occlusions=())
    for idx in range(10, 30):
        del outputs[idx]
    r = score_clip(outputs, labels, acc_cfg)
    assert r.miss_dropped == 20 and r.held == 130


def test_wrong_location_is_separated_from_no_track(cfg, acc_cfg):
    labels, outputs = scenario(cfg, occlusions=())
    shifted = {i: Label(i, lb.visible, lb.cx + 200, lb.cy, lb.w, lb.h) for i, lb in labels.items()}
    r = score_clip(outputs, shifted, acc_cfg)
    assert r.miss_wrong_location == r.visible_frames and r.held == 0 and r.miss_no_track == 0


def test_label_csv_round_trip(tmp_path):
    labels = [Label(0, True, 10.5, 20.25, 30.0, 31.0), Label(1, False), Label(2, True, 1.0, 2.0)]
    save_labels(tmp_path / "l.csv", labels)
    loaded = load_labels(tmp_path / "l.csv")
    assert loaded[0] == labels[0] and loaded[1] == labels[1] and loaded[2].w is None
    assert loaded[2].size is None and loaded[0].size == 31.0


def test_autolabel_finds_ball_and_marks_hidden_frames(cfg):
    labels = autolabel(RenderedSource(hidden=range(25, 31), n=50), ColorBlobDetector.from_config(cfg))
    assert len(labels) == 50
    assert all(not labels[i].visible for i in range(25, 31))
    assert all(labels[i].visible for i in (0, 10, 24, 31, 49))
    assert labels[0].cx == pytest.approx(60 + 9, abs=12)  # ball starts near x=60 (+ radius rounding)
