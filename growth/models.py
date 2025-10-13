from __future__ import annotations
from dataclasses import replace
from StochasticStandSim.core.types import StandState
from StochasticStandSim.core.config import GrowthParams


def _chapman_richards(age: float, p: GrowthParams) -> float:
    # Volume per ha proxy; replace with PMRC when ready
    return p.a * (1.0 - pow(2.718281828, -p.b * age)) ** p.c

def step_growth(s: StandState, dt_years: float, p: GrowthParams) -> StandState:
    next_age = s.age + dt_years
    target_V = _chapman_richards(next_age, p)
    dV = max(0.0, target_V - s.volume_m3)
    newV = s.volume_m3 + dV
    newC = newV * p.carbon_factor
    # simple basal area and stocking dynamics
    newBA = max(0.0, s.basal_area_m2 + 0.5 * dV / 10.0)
    newTPH = max(100.0, s.trees_per_ha * 0.995)
    return replace(s, volume_m3=newV, carbon_tCO2e=newC, basal_area_m2=newBA, trees_per_ha=newTPH)