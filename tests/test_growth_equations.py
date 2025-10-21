from __future__ import annotations

import math

import pytest

from core import growth


@pytest.fixture(autouse=True)
def disable_debug_prints(monkeypatch: pytest.MonkeyPatch) -> None:
    """Silence debug prints during tests to keep logs readable."""

    monkeypatch.setenv("PMRC_GROWTH_DEBUG", "0")


@pytest.mark.parametrize(
    "age, hd",
    [
        (5.0, 20.0),
        (15.0, 45.0),
        (25.0, 65.0),
        (35.0, 85.0),
    ],
)
@pytest.mark.parametrize("form", list(growth.SIForm))
def test_hd_si_roundtrip(age: float, hd: float, form: growth.SIForm) -> None:
    si = growth.si_from_hd(hd, form=form)
    recovered = growth.hd_from_si(si, form=form)
    assert math.isclose(recovered, hd, rel_tol=1e-6, abs_tol=1e-6)


def test_hd_project_identity() -> None:
    hd = 50.0
    projected = growth.hd_project(20.0, hd, 20.0)
    assert math.isclose(projected, hd, rel_tol=1e-9, abs_tol=1e-9)


def test_tpa_project_asymptote_hold() -> None:
    result = growth.tpa_project(100.0, 60.0, 10.0, 11.0)
    assert math.isclose(result, 100.0, rel_tol=0.0, abs_tol=1e-9)


def test_ba_predict_positive() -> None:
    value = growth.ba_predict(20.0, 500.0, 60.0, growth.Region.LCP)
    assert value >= 0.0


def test_step_and_run_horizon_consistency() -> None:
    initial = growth.StandState(
        age=10.0,
        tpa=600.0,
        region=growth.Region.UCP,
        si25=60.0,
        hd=growth.hd_from_si(60.0, growth.SIForm.PROJECTION),
    )
    next_state, ba = growth.step(initial)
    assert isinstance(next_state, growth.StandState)
    assert isinstance(ba, float)

    horizon = growth.run_horizon(initial, years=3.0, dt=1.0)
    assert len(horizon) == 3
    assert all(isinstance(entry[0], growth.StandState) for entry in horizon)
    assert all(isinstance(entry[1], float) for entry in horizon)
