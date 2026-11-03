import numpy as np

from edge_vision.config import build_config
from edge_vision.sim.runner import evaluate_scenario, run_sequence
from edge_vision.sim.synthetic import SyntheticConfig, generate_sequence
from edge_vision.tracking.metrics import bucket_label, compute_metrics, merge_metrics


def clean_seq(cfg, **overrides):
    base = {**cfg["synthetic"], "false_positive_rate": 0.0, "miss_probability": 0.0}
    return generate_sequence(build_config(SyntheticConfig, {**base, **overrides}))


def metrics_for(cfg, seq):
    ev = cfg["evaluation"]
    out = run_sequence(cfg, seq).outputs
    return compute_metrics(out, seq, ev["adoption_tolerance_px"], ev["reacq_deadline_frames"])


def test_clean_run_has_no_switches_or_adoptions(cfg):
    m = metrics_for(cfg, clean_seq(cfg, occlusions=((40, 12),)))
    assert m.id_switches == 0 and m.fp_adoption_events == 0 and m.tracks_ended == 0
    assert m.lost_track_count == 1
    assert m.reacquisition_rate() == (1, 1)
    assert m.pred_err_mean < 5
    assert sum(m.frames_in_state.values()) == m.num_frames
    assert m.frames_in_state["TEMPORARILY_LOST"] + m.frames_in_state["REACQUIRING"] > 0


def test_occlusion_past_max_counts_as_failed_reacquisition(cfg):
    n = cfg["state_machine"]["max_frames_before_ended"] + 10
    m = metrics_for(cfg, clean_seq(cfg, num_frames=150, occlusions=((30, n),)))
    assert m.reacquisition_rate() == (0, 1)
    assert m.tracks_ended >= 1 and m.id_switches >= 1


def test_buckets(cfg):
    seqs = [clean_seq(cfg, occlusions=((40, n),)) for n in (3, 8, 15, 30)]
    merged = merge_metrics([metrics_for(cfg, s) for s in seqs])
    by_bucket = merged.reacquisition_by_bucket(cfg["evaluation"]["occlusion_buckets"])
    assert list(by_bucket) == ["1-5", "6-10", "11-20", "21+"]
    assert all(total == 1 for _, total in by_bucket.values())
    assert bucket_label(21, None) == "21+"


def test_evaluate_scenario_is_deterministic_and_applies_overrides(cfg):
    a = evaluate_scenario(cfg, [0, 1, 2], {"synthetic.false_positive_rate": 0.3})
    b = evaluate_scenario(cfg, [0, 1, 2], {"synthetic.false_positive_rate": 0.3})
    assert a.pred_errors == b.pred_errors and a.id_switches == b.id_switches
    assert cfg["synthetic"]["false_positive_rate"] != 0.3  # input config untouched
    assert len(a.occlusions) == 3


def test_false_positive_adoption_is_detected(cfg):
    """Force the tracker onto a false detection and check it is flagged."""
    seq = clean_seq(cfg, occlusions=())
    # Replace frame 20's true detection with one 200 px off, inside a huge gate.
    cfg2 = {**cfg, "tracker": {**cfg["tracker"], "gate_base_radius_px": 400}}
    f = seq.frames[20]
    d = f.detections[0]
    x1, y1, x2, y2 = d.bbox_xyxy
    f.detections[0] = type(d)((x1 + 150, y1, x2 + 150, y2), d.confidence, d.class_id)
    ev = cfg2["evaluation"]
    out = run_sequence(cfg2, seq).outputs
    m = compute_metrics(out, seq, ev["adoption_tolerance_px"], ev["reacq_deadline_frames"])
    assert m.fp_adoption_events >= 1 and np.isfinite(m.pred_err_p95)
