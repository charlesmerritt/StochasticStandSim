from __future__ import annotations

from dataclasses import replace

from core.disturbances import FireDisturbance
from core.growth import GrowthConfig, Region, StandParams, _advance_internal


def _make_state(age: float = 10.0):
    params = StandParams(
        name="debug",
        age=age,
        tpa=600.0,
        region=Region.UCP,
        si25=60.0,
    )
    return params.to_state()


def test_fire_disturbance_adds_active_envelope():
    state = _make_state()
    disturbance = FireDisturbance(age=state.age, severity=0.6)
    state = replace(state, pending_disturbances=state.pending_disturbances + (disturbance,))
    cfg = GrowthConfig()

    next_state, *_ = _advance_internal(state, dt=1.0, cfg=cfg)
    assert next_state.active_envelopes, "Fire disturbance should register an active envelope"

    env_info = next_state.active_envelopes[0]
    assert env_info["type"] == "fire"
    assert env_info["severity_class"] == disturbance.get_severity_class()
    envelope = env_info["envelope"]
    assert envelope.attack_duration_years >= 0

    later_state, *_ = _advance_internal(next_state, dt=1.0, cfg=cfg)
    assert later_state.active_envelopes, "Envelope should persist at least one additional year"


def test_fire_envelope_uses_yaml_parameters():
    envelope_set = FireDisturbance.envelope_set()
    envelope = envelope_set.get_envelope("severe_50_80")
    assert envelope.attack_drop > 0.0
    assert envelope.attack_duration_years >= 5
