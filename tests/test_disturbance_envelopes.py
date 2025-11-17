from __future__ import annotations

import statistics

from core.disturbances import (
    CatastrophicDisturbanceGenerator,
    ChronicDisturbanceGenerator,
    GeneralDisturbanceGenerator,
    sample_exponential_wait,
)


def test_catastrophic_generator_produces_future_event():
    gen = CatastrophicDisturbanceGenerator(mean_interval_years=25.0)
    event = gen.sample_event(current_age=10.0)
    assert event.category == "catastrophic"
    assert event.start_age > 10.0
    assert 0.5 <= event.severity <= 0.95
    assert event.disturbance_level in {"light", "moderate", "heavy"}
    assert event.ba_loss_fraction in {0.25, 0.5, 0.8}
    assert event.tpa_loss_fraction == event.ba_loss_fraction
    assert any(abs(event.hd_loss_fraction - val) < 1e-6 for val in (0.175, 0.35, 0.56))


def test_chronic_generator_has_lower_losses():
    gen = ChronicDisturbanceGenerator(mean_interval_years=4.0)
    event = gen.sample_event(current_age=30.0)
    assert event.category == "chronic"
    assert 0.05 <= event.severity <= 0.25
    assert event.ba_loss_fraction <= 0.2
    assert event.tpa_loss_fraction <= 0.2
    assert event.hd_loss_fraction <= 0.1


def test_exponential_wait_has_expected_mean():
    mean_interval = 10.0
    samples = [sample_exponential_wait(mean_interval) for _ in range(2000)]
    sample_mean = statistics.mean(samples)
    assert abs(sample_mean - mean_interval) < mean_interval * 0.2
