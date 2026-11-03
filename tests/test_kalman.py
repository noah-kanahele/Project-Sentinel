import numpy as np

from edge_vision.config import build_config
from edge_vision.tracking.kalman import KalmanConfig, KalmanFilter

DT = 1 / 60
P0 = np.array([100.0, 650.0])
V0 = np.array([450.0, -700.0])
G = np.array([0.0, 600.0])


def truth(t):
    return P0 + V0 * t + 0.5 * G * t * t


def make_filter(cfg):
    return KalmanFilter(build_config(KalmanConfig, cfg["kalman"]), tuple(truth(0.0)))


def run_noiseless(kf, steps):
    """Feed exact parabola measurements; return prediction (prior) errors."""
    errors = []
    for i in range(1, steps + 1):
        kf.predict(DT)
        errors.append(np.linalg.norm(kf.position - truth(i * DT)))
        kf.update(tuple(truth(i * DT)))
    return errors


def test_prediction_error_converges_on_noiseless_parabola(cfg):
    errors = run_noiseless(make_filter(cfg), 60)
    assert errors[-1] < 0.5
    assert max(errors[-10:]) < 0.5
    assert errors[-1] < errors[2]


def test_predict_through_gap_stays_near_parabola(cfg):
    kf = make_filter(cfg)
    run_noiseless(kf, 60)
    for i in range(61, 61 + 20):  # 20 predict-only steps, no updates
        kf.predict(DT)
    assert np.linalg.norm(kf.position - truth(80 * DT)) < 3.0


def test_covariance_grows_without_updates_and_shrinks_with_them(cfg):
    kf = make_filter(cfg)
    run_noiseless(kf, 30)
    sigma_tracking = kf.position_sigma()
    for _ in range(15):
        kf.predict(DT)
    sigma_gap = kf.position_sigma()
    assert sigma_gap > 2 * sigma_tracking
    kf.update(tuple(truth(45 * DT)))
    assert kf.position_sigma() < sigma_gap


def test_non_positive_dt_is_noop(cfg):
    kf = make_filter(cfg)
    before = kf.covariance
    kf.predict(0.0)
    kf.predict(-1.0)
    assert np.array_equal(before, kf.covariance)


def test_project_does_not_mutate(cfg):
    kf = make_filter(cfg)
    run_noiseless(kf, 20)
    pos = kf.position
    ahead = kf.project(0.1)
    assert np.array_equal(pos, kf.position)
    assert np.linalg.norm(ahead - truth(20 * DT + 0.1)) < 2.0
