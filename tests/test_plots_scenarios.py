from dataclasses import replace
import math

import pytest

from core.growth import PMRC1996, PMRCGrowth, StandState
from core.growth.pmrc import MerchantabilitySpec
from core.growth.types import ProductClass, Region
from plots import (
    FertilisationEvent,
    GrowthScenario,
    ScenarioSeries,
    build_scenario_series,
    run_growth_scenario,
)


@pytest.fixture(scope="module")
def growth() -> PMRCGrowth:
    return PMRCGrowth(PMRC1996(), default_region=Region.UPPER_COASTAL_PLAIN)


@pytest.fixture()
def base_state() -> StandState:
    return StandState(age=15.0, tpa=600.0, hd=45.0, region=Region.UPPER_COASTAL_PLAIN)


def test_build_scenario_series_generates_time_series(growth: PMRCGrowth, base_state: StandState) -> None:
    merch = {ProductClass.PULPWOOD: MerchantabilitySpec(d_dbh_min=4.5, t_top=4.0)}

    series = build_scenario_series(
        growth=growth,
        initial_state=base_state,
        horizon=10.0,
        step=5.0,
        merchantability=merch,
    )

    assert isinstance(series, ScenarioSeries)
    assert series.ages[0] == pytest.approx(base_state.age)
    assert series.ages[-1] == pytest.approx(base_state.age + 10.0)
    assert len(series.ages) == len(series.heights) == len(series.tpa) == len(series.basal_area)

    pulp_series = series.yields_by_product[ProductClass.PULPWOOD]
    assert len(pulp_series) == len(series.ages)
    assert all(value >= 0 for value in pulp_series)

    assert set(series.diameter_percentiles) == {25, 50, 75}
    for percentile_values in series.diameter_percentiles.values():
        assert len(percentile_values) == len(series.ages)

    assert math.isnan(series.relative_size[0])
    assert not math.isnan(series.relative_size[-1])


def test_callbacks_and_fertilisation_are_applied(growth: PMRCGrowth, base_state: StandState) -> None:
    horizon = 10.0
    step = 5.0
    region = Region.UPPER_COASTAL_PLAIN

    event = FertilisationEvent(age=18.0, n_lbs_ac=150.0, with_p=True)

    def apply_thin(state: StandState) -> StandState:
        if state.ba is None:
            return state
        return replace(state, tpa=state.tpa * 0.9, ba=state.ba * 0.85)

    scenario = GrowthScenario(
        growth=growth,
        initial_state=base_state,
        horizon=horizon,
        step=step,
        region=region,
        callbacks=[apply_thin],
        fertilisation_events=[event],
    )

    series = run_growth_scenario(scenario)

    next_age = base_state.age + step
    projected_hd = growth.project_height(base_state, next_age)
    projected_tpa = growth.project_tpa(base_state, next_age)
    projected_ba = growth.project_ba(base_state, next_age, projected_tpa, projected_hd)

    assert series.states[1].tpa == pytest.approx(projected_tpa * 0.9)
    assert series.states[1].ba == pytest.approx(projected_ba * 0.85)

    fert_index = 1
    assert series.ages[fert_index] == pytest.approx(base_state.age + step)
    assert series.fert_adjusted_hd[fert_index] > series.heights[fert_index]
    assert series.fert_adjusted_ba[fert_index] > series.basal_area[fert_index]
