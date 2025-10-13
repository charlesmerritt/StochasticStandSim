"""Lightweight growth helpers for :mod:`core.stand_env`."""

from __future__ import annotations

from typing import Mapping, MutableMapping, Tuple, Dict, Any

import numpy as np

__all__ = ["grow_one_step"]


def _resolve(mapping: Mapping[str, Any], key: str, default: Any) -> Any:
    return mapping[key] if key in mapping else default


def grow_one_step(
    state: Mapping[str, Any],
    action: np.ndarray,
    params: MutableMapping[str, Any],
) -> Dict[str, Any]:
    """Advance the stand state by one time step.

    The implementation intentionally keeps the dynamics simple so the environment
    remains usable in isolation.  Projects that require richer behaviour are
    expected to replace the function via dependency injection.
    """

    params = params or {}
    next_state = dict(state)

    age_increment = float(_resolve(params, "age_increment", 1.0))
    biomass_growth = float(_resolve(params, "biomass_growth", 2.0))
    tpa_decay = float(_resolve(params, "tpa_decay_factor", 1.0))
    ba_growth = float(_resolve(params, "basal_area_growth", 1.5))
    risk_increment = float(_resolve(params, "risk_increment", 0.005))
    stumpage_price = float(_resolve(params, "stumpage_price", 20.0))

    thin_pct = float(action[0])
    fert_n = float(action[1])
    fert_p = float(action[2])

    # Update age.
    next_state["age"] = float(state.get("age", 0.0)) + age_increment

    # Harvested biomass from thinning.
    biomass = float(state.get("biomass", 0.0))
    removed_biomass = biomass * thin_pct

    fertility_gain = 1.0 + fert_n * 0.5 + fert_p * 0.3
    next_state["biomass"] = max(0.0, (biomass - removed_biomass) + biomass_growth * fertility_gain)

    tpa = float(state.get("tpa", 0.0))
    next_state["tpa"] = max(0.0, tpa * (1.0 - thin_pct * tpa_decay))

    basal_area = float(state.get("basal_area", 0.0))
    next_state["basal_area"] = max(0.0, basal_area * (1.0 - thin_pct) + ba_growth * fertility_gain)

    # Risk slowly increases with stand age but is capped at 1.0.
    risk = float(state.get("risk", 0.0))
    next_state["risk"] = float(np.clip(risk + risk_increment, 0.0, 1.0))

    # Value accumulates with biomass and harvest events.
    value = float(state.get("value", 0.0))
    next_state["value"] = value + removed_biomass * stumpage_price

    info = {
        "removed_biomass": removed_biomass,
        "fertility_gain": fertility_gain,
    }

    return {"state": next_state, "info": info}
