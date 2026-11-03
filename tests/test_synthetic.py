import numpy as np

from edge_vision.config import build_config
from edge_vision.sim.synthetic import SyntheticConfig, generate_sequence


def make(cfg, **overrides):
    return build_config(SyntheticConfig, {**cfg["synthetic"], **overrides})


def test_same_seed_is_reproducible(cfg):
    a = generate_sequence(make(cfg, seed=7, false_positive_rate=0.5, timestamp_jitter_s=0.001))
    b = generate_sequence(make(cfg, seed=7, false_positive_rate=0.5, timestamp_jitter_s=0.001))
    assert [f.detections for f in a.frames] == [f.detections for f in b.frames]
    assert [f.timestamp_s for f in a.frames] == [f.timestamp_s for f in b.frames]


def test_different_seed_differs(cfg):
    a = generate_sequence(make(cfg, seed=1))
    b = generate_sequence(make(cfg, seed=2))
    assert [f.detections for f in a.frames] != [f.detections for f in b.frames]


def test_ground_truth_is_parabola(cfg):
    seq = generate_sequence(make(cfg))
    c = seq.config
    f = seq.frames[30]
    t = f.timestamp_s
    assert np.isclose(f.gt_center[0], c.initial_position[0] + c.initial_velocity[0] * t)
    expected_y = c.initial_position[1] + c.initial_velocity[1] * t + 0.5 * c.gravity * t * t
    assert np.isclose(f.gt_center[1], expected_y)


def test_occlusion_window_has_no_true_detection(cfg):
    seq = generate_sequence(make(cfg, occlusions=((20, 15),), false_positive_rate=0.0, miss_probability=0.0))
    for f in seq.frames:
        if 20 <= f.frame_idx < 35:
            assert f.occluded and not f.true_detected and f.detections == []
        else:
            assert not f.occluded and f.true_detected and len(f.detections) == 1


def test_miss_probability_and_false_positive_rate(cfg):
    seq = generate_sequence(
        make(cfg, num_frames=2000, occlusions=(), miss_probability=0.2, false_positive_rate=0.5)
    )
    miss = 1 - np.mean([f.true_detected for f in seq.frames])
    fps = np.mean([len(f.detections) - int(f.true_detected) for f in seq.frames])
    assert 0.15 < miss < 0.25
    assert 0.4 < fps < 0.6


def test_timestamp_jitter_stays_monotonic(cfg):
    seq = generate_sequence(make(cfg, timestamp_jitter_s=0.01))
    ts = [f.timestamp_s for f in seq.frames]
    assert all(b > a for a, b in zip(ts, ts[1:]))
    assert not np.allclose(np.diff(ts), 1.0 / seq.config.fps)


def test_config_extends_merges_over_base(tmp_path):
    from edge_vision.config import load_config

    (tmp_path / "base.yaml").write_text("a:\n  x: 1\n  y: 2\nb: 3\n")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\na:\n  y: 20\nc: 4\n")
    assert load_config(tmp_path / "child.yaml") == {"a": {"x": 1, "y": 20}, "b": 3, "c": 4}


def test_gumball_config_inherits_default():
    from pathlib import Path

    from edge_vision.config import load_config

    root = Path(__file__).resolve().parents[1] / "configs"
    default, gumball = load_config(root / "default.yaml"), load_config(root / "gumball.yaml")
    assert gumball["detection"]["class_id"] == 49 and default["detection"]["class_id"] == 32
    assert gumball["tracker"] == default["tracker"]
