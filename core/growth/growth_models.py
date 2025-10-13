"""Growth facade that bridges :mod:`core.stand_env` with PMRC utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, MutableMapping, Tuple

import numpy as np

from . import PMRCGrowth, StandState, Region, get_growth

__all__ = ["grow_one_step"]


@dataclass(frozen=True)
class _EngineKey:
    name: str
    region: Region


_ENGINE_CACHE: Dict[_EngineKey, PMRCGrowth] = {}


def _resolve_region(value: Any, default: Region) -> Region:
    if isinstance(value, Region):
        return value
    if isinstance(value, str):
        try:
            return Region[value.upper()]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Unknown region '{value}'") from exc
    return default


def _resolve_engine(params: Mapping[str, Any]) -> Tuple[PMRCGrowth, Region]:
    engine_name = str(params.get("engine", "pmrc1996"))
    region_default = Region.LOWER_COASTAL_PLAIN
    region = _resolve_region(params.get("region"), region_default)
    key = _EngineKey(engine_name, region)
    if key not in _ENGINE_CACHE:
        _ENGINE_CACHE[key] = get_growth(engine_name, region)
    return _ENGINE_CACHE[key], region


def _stand_state_from_mapping(
    growth: PMRCGrowth,
    region: Region,
    state: Mapping[str, Any],
    site_index: float,
) -> StandState:
    age = float(state.get("age", 0.0))
    tpa = max(float(state.get("tpa", 0.0)), 0.0)
    ba = state.get("basal_area")
    ba_val = float(ba) if ba is not None else None
    hd = state.get("hd")
    if hd is not None:
        hd_val = float(hd)
    else:
        # Estimate dominant height from the site index if not provided.
        hd_val = growth.eqns.hd_from_site_index(max(age, 1.0), site_index)
    percent_hw = state.get("percent_hardwood_ba")
    phwd_val = float(percent_hw) if percent_hw is not None else None
    return StandState(
        age=age,
        tpa=tpa,
        hd=hd_val,
        ba=ba_val,
        si25=site_index,
        region=region,
        percent_hardwood_ba=phwd_val,
    )


def _biomass_from_ba_height(ba: float, height: float, conversion: float) -> float:
    ba = max(ba, 0.0)
    height = max(height, 0.0)
    return ba * height * conversion


def grow_one_step(
    state: Mapping[str, Any],
    action: np.ndarray,
    params: MutableMapping[str, Any],
) -> Dict[str, Any]:
    """Advance the stand one time step using the PMRC growth engine.

    Parameters
    ----------
    state:
        Mapping describing the current stand state.
    action:
        Action vector ``[thin_pct, fert_n, fert_p_or_plant]`` with each
        component on ``[0, 1]``.
    params:
        Mutable configuration blob passed from :class:`~core.stand_env.StandEnv`.
        Expected keys include ``engine`` (default ``"pmrc1996"``), ``region``,
        ``timestep_years`` (default ``1``), fertilisation rates and planting
        intensities.  The dictionary may also be used as a scratch pad by the
        caller to persist helper state between steps.
    """

    params = params or {}
    growth, region = _resolve_engine(params)

    step_years = float(params.get("timestep_years", 1.0))
    site_index = float(params.get("site_index", state.get("site_index", 70.0)))

    conversion = float(params.get("biomass_conversion", 0.0065))
    risk_accum = float(params.get("risk_accumulation", 0.012))
    risk_thin_relief = float(params.get("risk_thinning_relief", 0.025))

    thin_pct = float(np.clip(action[0], 0.0, 1.0))
    fert_n_rate = float(np.clip(action[1], 0.0, 1.0))
    planting_share = float(np.clip(params.get("planting_share", 0.4), 0.0, 1.0))
    fert_p_rate_raw = float(np.clip(action[2], 0.0, 1.0))
    planting_fraction = fert_p_rate_raw * planting_share
    fert_p_rate = fert_p_rate_raw * (1.0 - planting_share)

    max_fert_n = float(params.get("max_fertilizer_n_lbs", 180.0))
    max_fert_p = float(params.get("max_fertilizer_p_lbs", 40.0))
    n_lbs = fert_n_rate * max_fert_n
    p_lbs = fert_p_rate * max_fert_p
    fert_with_p = p_lbs > 1e-6

    max_planting = float(params.get("max_planting_tpa", 250.0))
    planting_tpa = planting_fraction * max_planting
    seedling_biomass = planting_tpa * float(params.get("seedling_biomass_tons", 0.015))

    stand_before = _stand_state_from_mapping(growth, region, state, site_index)
    ba_before = stand_before.ba if stand_before.ba is not None else growth.ba(stand_before)
    biomass_before = float(state.get("biomass", _biomass_from_ba_height(ba_before, stand_before.hd, conversion)))

    tpa_removed = stand_before.tpa * thin_pct
    row_fraction = float(np.clip(params.get("row_thin_fraction", 0.0), 0.0, 1.0))
    row_removed = tpa_removed * row_fraction
    select_removed = max(tpa_removed - row_removed, 0.0)
    if thin_pct > 0.0:
        ba_removed = growth.estimate_ba_removed(ba_before, stand_before.tpa, row_removed, select_removed)
        ba_removed = min(max(ba_removed, 0.0), ba_before)
    else:
        ba_removed = 0.0

    tpa_after_thin = max(stand_before.tpa - tpa_removed, 0.0)
    ba_after_thin = max(ba_before - ba_removed, 0.0)

    biomass_removed = biomass_before * (ba_removed / ba_before) if ba_before > 1e-9 else biomass_before * thin_pct

    # Apply planting immediately after thinning to represent supplemental trees.
    tpa_after_planting = tpa_after_thin + planting_tpa
    ba_after_actions = ba_after_thin + planting_tpa * float(params.get("seedling_ba_factor", 0.005))

    stand_post_actions = StandState(
        age=stand_before.age,
        tpa=tpa_after_planting,
        hd=stand_before.hd,
        ba=ba_after_actions,
        si25=site_index,
        region=region,
        percent_hardwood_ba=stand_before.percent_hardwood_ba,
    )

    age_next = stand_before.age + step_years
    hd_next = growth.project_height(stand_post_actions, age_next)
    hd_next += growth.eqns.fert_response_hd(step_years, n_lbs, fert_with_p)

    tpa_next = growth.project_tpa(stand_post_actions, age_next)
    ba_next = growth.project_ba(stand_post_actions, age_next, tpa_next, hd_next)
    ba_next += growth.eqns.fert_response_ba(step_years, n_lbs, fert_with_p)
    ba_next = max(ba_next, 0.0)

    biomass_next = _biomass_from_ba_height(ba_next, hd_next, conversion)

    risk_base = float(np.clip(state.get("risk", params.get("base_risk", 0.01)), 0.0, 1.0))
    risk_adjusted = risk_base + step_years * risk_accum - thin_pct * risk_thin_relief
    risk_next = float(np.clip(risk_adjusted, 0.0, 1.0))

    years_since_fert = float(state.get("years_since_fertilization", step_years)) + step_years
    if n_lbs > 0.0 or fert_with_p:
        years_since_fert = step_years

    info: Dict[str, Any] = {
        "ba_removed": ba_removed,
        "tpa_removed": tpa_removed,
        "biomass_removed": biomass_removed,
        "planting_tpa": planting_tpa,
        "fertilizer_n_lbs": n_lbs,
        "fertilizer_p_lbs": p_lbs,
        "with_p": fert_with_p,
    }

    next_state: Dict[str, Any] = dict(state)
    next_state.update(
        {
            "age": age_next,
            "tpa": tpa_next,
            "basal_area": ba_next,
            "biomass": biomass_next,
            "hd": hd_next,
            "site_index": site_index,
            "risk": risk_next,
            "catastrophic": False,
            "thin_removed_tpa": tpa_removed,
            "thin_removed_ba": ba_removed,
            "thin_removed_biomass": biomass_removed,
            "planting_tpa": planting_tpa,
            "planting_biomass": seedling_biomass,
            "fert_n_lbs": n_lbs,
            "fert_p_lbs": p_lbs,
            "fert_with_p": fert_with_p,
            "years_since_fertilization": years_since_fert,
        }
    )

    return {"state": next_state, "info": info}

