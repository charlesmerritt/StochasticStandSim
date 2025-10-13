from __future__ import annotations
from dataclasses import replace
from .types import StandState, ObsDict




def initial_state(cfg: "EnvConfig") -> StandState:
    return StandState(
    age=cfg.init_age,
    volume_m3=cfg.init_volume_m3,
    basal_area_m2=cfg.init_basal_area_m2,
    trees_per_ha=cfg.init_tph,
    carbon_tCO2e=cfg.init_volume_m3 * cfg.growth.carbon_factor,
    site_index=cfg.site_index,
    cash_account=0.0,
    t=0,
    )




def observe(s: StandState) -> ObsDict:
    return {
    "age": s.age,
    "volume_m3": s.volume_m3,
    "basal_area_m2": s.basal_area_m2,
    "trees_per_ha": s.trees_per_ha,
    "carbon_tCO2e": s.carbon_tCO2e,
    "site_index": s.site_index,
    }




def advance_time(s: StandState, dt_years: float) -> StandState:
    return replace(s, age=s.age + dt_years, t=s.t + 1)