"""Plotting helpers exposing scenario generation convenience APIs."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

from core.growth import PMRCGrowth, StandState
from core.growth.pmrc import MerchantabilitySpec
from core.growth.types import ProductClass, Region

from .scenarios import (
    FertilisationEvent,
    GrowthScenario,
    ScenarioCallback,
    ScenarioSeries,
    run_growth_scenario,
)


def build_scenario_series(
    *,
    growth: PMRCGrowth,
    initial_state: StandState,
    horizon: float,
    step: float,
    region: Optional[Region] = None,
    merchantability: Optional[Mapping[ProductClass, MerchantabilitySpec]] = None,
    callbacks: Optional[Sequence[ScenarioCallback]] = None,
    fertilisation_events: Optional[Sequence[FertilisationEvent]] = None,
) -> ScenarioSeries:
    """Convenience wrapper to run a :class:`GrowthScenario` and return its series."""

    scenario = GrowthScenario(
        growth=growth,
        initial_state=initial_state,
        horizon=horizon,
        step=step,
        region=region,
        merchantability=merchantability or {},
        callbacks=tuple(callbacks or ()),
        fertilisation_events=tuple(fertilisation_events or ()),
    )
    return run_growth_scenario(scenario)


__all__ = [
    "FertilisationEvent",
    "GrowthScenario",
    "ScenarioCallback",
    "ScenarioSeries",
    "build_scenario_series",
    "run_growth_scenario",
]

