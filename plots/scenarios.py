"""Growth scenario utilities for generating plotting-ready series."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Callable, Dict, List, Mapping, Optional, Sequence

from core.growth.pmrc import MerchantabilitySpec
from core.growth import PMRCGrowth, StandState
from core.growth.types import ProductClass, Region


# ---------------------------------------------------------------------------
# Data containers


ScenarioCallback = Callable[[StandState], StandState]


@dataclass(frozen=True)
class FertilisationEvent:
    """Description of a fertilisation treatment applied at a stand age."""

    age: float
    n_lbs_ac: float
    with_p: bool = False


@dataclass
class GrowthScenario:
    """Input parameters for running a simple deterministic growth scenario."""

    growth: PMRCGrowth
    initial_state: StandState
    horizon: float
    step: float
    region: Optional[Region] = None
    merchantability: Mapping[ProductClass, MerchantabilitySpec] = field(default_factory=dict)
    callbacks: Sequence[ScenarioCallback] = field(default_factory=tuple)
    fertilisation_events: Sequence[FertilisationEvent] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("Scenario horizon must be positive")
        if self.step <= 0:
            raise ValueError("Scenario step must be positive")


@dataclass
class ScenarioSeries:
    """Time-series container with the metrics needed for plotting."""

    states: List[StandState] = field(default_factory=list)
    ages: List[float] = field(default_factory=list)
    heights: List[float] = field(default_factory=list)
    tpa: List[float] = field(default_factory=list)
    basal_area: List[float] = field(default_factory=list)
    total_yield: List[float] = field(default_factory=list)
    yields_by_product: Dict[ProductClass, List[float]] = field(default_factory=dict)
    diameter_percentiles: Dict[int, List[float]] = field(default_factory=dict)
    relative_size: List[float] = field(default_factory=list)
    fert_adjusted_hd: List[float] = field(default_factory=list)
    fert_adjusted_ba: List[float] = field(default_factory=list)

    def append(
        self,
        state: StandState,
        total_yield: float,
        product_yields: Mapping[ProductClass, float],
        percentiles: Mapping[int, float],
        relative_size: float,
        fert_hd: float,
        fert_ba: float,
    ) -> None:
        self.states.append(state)
        self.ages.append(state.age)
        self.heights.append(state.hd)
        self.tpa.append(state.tpa)
        if state.ba is None:
            raise ValueError("Basal area must be populated on states recorded in the series")
        self.basal_area.append(state.ba)
        self.total_yield.append(total_yield)

        for product, value in product_yields.items():
            self.yields_by_product.setdefault(product, []).append(value)

        for percentile, value in percentiles.items():
            self.diameter_percentiles.setdefault(percentile, []).append(value)

        self.relative_size.append(relative_size)
        self.fert_adjusted_hd.append(fert_hd)
        self.fert_adjusted_ba.append(fert_ba)


# ---------------------------------------------------------------------------
# Scenario harness


def _ensure_state_region(state: StandState, region: Region, si25: float) -> StandState:
    """Make sure derived StandState fields are populated for downstream calls."""

    updated = state
    if state.region is None:
        updated = replace(updated, region=region)
    if state.si25 is None:
        updated = replace(updated, si25=si25)
    if updated.ba is None:
        raise ValueError("Basal area must be set before recording a stand state")
    return updated


def _latest_fertilisation(events: Sequence[FertilisationEvent], age: float) -> Optional[FertilisationEvent]:
    applicable = [event for event in events if event.age <= age]
    if not applicable:
        return None
    return max(applicable, key=lambda event: event.age)


def run_growth_scenario(scenario: GrowthScenario) -> ScenarioSeries:
    """Project a stand through time and collect metrics for plotting."""

    growth = scenario.growth
    region = scenario.region or scenario.initial_state.region or growth.default_region
    base_si = scenario.initial_state.si25 or growth.site_index(scenario.initial_state)

    initial_ba = scenario.initial_state.ba or growth.ba(scenario.initial_state)
    current_state = replace(
        scenario.initial_state,
        ba=initial_ba,
        si25=base_si,
        region=region,
    )

    series = ScenarioSeries()

    def _fert_adjustments(state: StandState) -> tuple[float, float]:
        event = _latest_fertilisation(scenario.fertilisation_events, state.age)
        if not event:
            return state.hd, state.ba if state.ba is not None else growth.ba(state)
        years_since = state.age - event.age
        fert_hd = growth.fert_adjusted_hd(state, years_since, event.n_lbs_ac, event.with_p)
        fert_ba = growth.fert_adjusted_ba(state, years_since, event.n_lbs_ac, event.with_p)
        return fert_hd, fert_ba

    def _record_state(prev_state: Optional[StandState], state: StandState) -> None:
        ensured_state = _ensure_state_region(state, region, base_si)
        total_yield = growth.yield_total(ensured_state)
        product_yields = {
            product: growth.yield_by_product(ensured_state, merch)
            for product, merch in scenario.merchantability.items()
        }
        percentiles = growth.diameter_percentiles(ensured_state)

        if prev_state is None:
            relative = math.nan
        else:
            prev_ba = prev_state.ba if prev_state.ba is not None else growth.ba(prev_state)
            prev_tpa = prev_state.tpa
            b_avg1 = prev_ba / prev_tpa if prev_tpa > 0 else 0.0
            b_i1 = b_avg1
            relative = growth.project_relative_size(b_avg1=b_avg1, b_i1=b_i1, age1=prev_state.age, age2=ensured_state.age, region=region)

        fert_hd, fert_ba = _fert_adjustments(ensured_state)
        series.append(ensured_state, total_yield, product_yields, percentiles, relative, fert_hd, fert_ba)

    _record_state(None, current_state)

    start_age = current_state.age
    end_age = start_age + scenario.horizon
    prev_state = current_state

    while prev_state.age < end_age - 1e-9:
        next_age = min(prev_state.age + scenario.step, end_age)
        projected_hd = growth.project_height(prev_state, next_age)
        projected_tpa = growth.project_tpa(prev_state, next_age)
        projected_ba = growth.project_ba(prev_state, next_age, projected_tpa, projected_hd)

        next_state = StandState(
            age=next_age,
            tpa=projected_tpa,
            hd=projected_hd,
            ba=projected_ba,
            si25=base_si,
            region=region,
            percent_hardwood_ba=prev_state.percent_hardwood_ba,
        )

        for callback in scenario.callbacks:
            next_state = callback(next_state)

        if next_state.ba is None:
            next_state = replace(next_state, ba=growth.ba(next_state))

        if next_state.region is None:
            next_state = replace(next_state, region=region)
        if next_state.si25 is None:
            next_state = replace(next_state, si25=base_si)

        _record_state(prev_state, next_state)
        prev_state = next_state

    return series

