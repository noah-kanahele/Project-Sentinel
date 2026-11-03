import pytest

from edge_vision.config import build_config
from edge_vision.tracking.state_machine import (
    StateMachineConfig,
    TrackState as S,
    TrackStateMachine,
)


@pytest.fixture
def sm_cfg(cfg):
    # confirm=3, lost after 3 misses, reacquiring after 3+6=9, ended at 60
    return build_config(StateMachineConfig, cfg["state_machine"])


def make(sm_cfg):
    events = []
    return TrackStateMachine(1, sm_cfg, 0, 0.0, [events.append]), events


def hit(sm, n=1, start=1):
    for i in range(n):
        sm.on_hit(start + i, (start + i) / 60)


def miss(sm, n=1, start=1):
    for i in range(n):
        sm.on_miss(start + i, (start + i) / 60)


def test_creation_emits_event(sm_cfg):
    sm, events = make(sm_cfg)
    assert sm.state == S.DETECTED
    assert events[0].from_state is None and events[0].to_state == S.DETECTED


def test_confirm_after_n_hits(sm_cfg):
    sm, events = make(sm_cfg)
    hit(sm, sm_cfg.frames_to_confirm - 1)
    assert sm.state == S.DETECTED
    hit(sm, 1, start=10)
    assert sm.state == S.TRACKING
    assert events[-1].to_state == S.TRACKING and events[-1].frame_idx == 10


def test_unconfirmed_track_ends_on_misses(sm_cfg):
    sm, _ = make(sm_cfg)
    hit(sm)
    miss(sm, sm_cfg.frames_missed_before_lost)
    assert sm.state == S.TRACK_ENDED


def test_miss_progression_thresholds(sm_cfg):
    sm, events = make(sm_cfg)
    hit(sm, sm_cfg.frames_to_confirm)
    a = sm_cfg.frames_missed_before_lost
    b = a + sm_cfg.frames_lost_before_reacquiring
    miss(sm, a - 1)
    assert sm.state == S.TRACKING
    miss(sm, 1, start=100)
    assert sm.state == S.TEMPORARILY_LOST
    miss(sm, b - a - 1, start=200)
    assert sm.state == S.TEMPORARILY_LOST
    miss(sm, 1, start=300)
    assert sm.state == S.REACQUIRING
    miss(sm, sm_cfg.max_frames_before_ended - b - 1, start=400)
    assert sm.state == S.REACQUIRING
    miss(sm, 1, start=500)
    assert sm.state == S.TRACK_ENDED
    assert [e.to_state for e in events] == [
        S.DETECTED, S.TRACKING, S.TEMPORARILY_LOST, S.REACQUIRING, S.TRACK_ENDED,
    ]


@pytest.mark.parametrize("misses", [3, 9, 30])
def test_hit_while_lost_or_reacquiring_returns_to_tracking(sm_cfg, misses):
    sm, events = make(sm_cfg)
    hit(sm, sm_cfg.frames_to_confirm)
    miss(sm, misses)
    assert sm.state in (S.TEMPORARILY_LOST, S.REACQUIRING)
    hit(sm, 1, start=999)
    assert sm.state == S.TRACKING and sm.misses == 0
    assert events[-1].reason.startswith("reacquired_from_")


def test_ended_is_terminal(sm_cfg):
    sm, events = make(sm_cfg)
    hit(sm, sm_cfg.frames_to_confirm)
    miss(sm, sm_cfg.max_frames_before_ended)
    n = len(events)
    hit(sm, 5, start=1000)
    miss(sm, 5, start=2000)
    assert sm.state == S.TRACK_ENDED and len(events) == n


def test_events_carry_frame_and_timestamp(sm_cfg):
    sm, events = make(sm_cfg)
    sm.on_hit(7, 0.5)
    sm.on_hit(8, 0.6)
    sm.on_hit(9, 0.7)
    assert (events[-1].frame_idx, events[-1].timestamp_s, events[-1].track_id) == (9, 0.7, 1)


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        StateMachineConfig(3, 3, 6, 9)
