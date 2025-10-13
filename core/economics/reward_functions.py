"""Economic reward helpers for :mod:`core.stand_env`."""

from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping

import numpy as np

__all__ = ["step_reward"]


_REGIONAL_PRICES = {
    "southeast": {
        "biomass": 17.5,  # $/green ton
        "salvage_discount": 0.55,
        "thin_cost": 6.0,
    },
    "piedmont": {
        "biomass": 18.5,
        "salvage_discount": 0.6,
        "thin_cost": 6.5,
    },
    "coastal": {
        "biomass": 16.5,
        "salvage_discount": 0.5,
        "thin_cost": 5.5,
    },
}


def _site_premium(site_index: float, baseline: float, slope: float) -> float:
    delta = site_index - baseline
    return 1.0 + slope * delta


def _resolve_region_prices(econ_cfg: Mapping[str, Any]) -> Dict[str, float]:
    region = str(econ_cfg.get("region", "southeast")).lower()
    if region not in _REGIONAL_PRICES:
        region = "southeast"
    prices = dict(_REGIONAL_PRICES[region])
    override = econ_cfg.get("price_overrides", {})
    if isinstance(override, Mapping):
        prices.update({k: float(v) for k, v in override.items()})
    return prices


def _resolve_site_index(prev_state: Mapping[str, Any], econ_cfg: Mapping[str, Any]) -> float:
    if "site_index" in econ_cfg:
        return float(econ_cfg["site_index"])
    return float(prev_state.get("site_index", 70.0))


def step_reward(
    prev_state: Mapping[str, Any],
    action: np.ndarray,
    next_state: Mapping[str, Any],
    econ_cfg: MutableMapping[str, Any],
) -> float:
    """Compute the immediate reward for the transition ``prev_state -> next_state``.

    Revenues include thinning removals, salvage value from disturbances, and
    bonuses for maintaining productive biomass.  Costs include treatments,
    planting, and carrying costs.  The helper remains deterministic to keep the
    environment suitable for unit testing.
    """

    econ_cfg = econ_cfg or {}
    prices = _resolve_region_prices(econ_cfg)

    site_index = _resolve_site_index(prev_state, econ_cfg)
    site_factor = _site_premium(site_index, baseline=70.0, slope=float(econ_cfg.get("site_index_slope", 0.0035)))
    biomass_price = prices.get("biomass", 17.5) * site_factor
    thin_cost_per_ton = prices.get("thin_cost", 6.0)

    thin_removed = float(next_state.get("thin_removed_biomass", 0.0))
    # Fallback estimate if the growth engine did not annotate removals.
    if thin_removed <= 0.0 and float(action[0]) > 0.0:
        biomass_prev = float(prev_state.get("biomass", 0.0))
        biomass_next = float(next_state.get("biomass", 0.0))
        thin_removed = max(biomass_prev - biomass_next, 0.0) * float(np.clip(action[0], 0.0, 1.0))

    thin_revenue = thin_removed * (biomass_price - thin_cost_per_ton)

    salvage_biomass = float(next_state.get("salvage_biomass", 0.0))
    salvage_price = biomass_price * float(prices.get("salvage_discount", 0.55))
    salvage_revenue = salvage_biomass * salvage_price

    planting_tpa = float(next_state.get("planting_tpa", 0.0))
    planting_cost = planting_tpa * float(econ_cfg.get("planting_cost_per_tree", 0.35))

    fert_n_lbs = float(next_state.get("fert_n_lbs", 0.0))
    fert_p_lbs = float(next_state.get("fert_p_lbs", 0.0))
    fert_cost = fert_n_lbs * float(econ_cfg.get("fert_cost_per_lb", 0.32))
    fert_cost += fert_p_lbs * float(econ_cfg.get("fert_premium_per_lb", 0.18))

    treatment_bonus = float(econ_cfg.get("fertilization_bonus", 0.0)) if fert_n_lbs > 0 else 0.0

    catastrophic_penalty = float(econ_cfg.get("catastrophe_penalty", 3500.0))
    severity = float(next_state.get("disturbance_severity", 0.0))
    catastrophe_cost = catastrophic_penalty * severity if bool(next_state.get("catastrophic", False)) else 0.0

    annual_carry = float(econ_cfg.get("annual_carrying_cost", 8.0))
    age = float(next_state.get("age", prev_state.get("age", 0.0)))
    carrying_cost = annual_carry * (age / max(econ_cfg.get("timestep_years", 1.0), 1e-6))

    biomass_reward = float(next_state.get("biomass", 0.0)) * float(econ_cfg.get("standing_biomass_bonus", 0.0))

    reward = (
        thin_revenue
        + salvage_revenue
        + treatment_bonus
        + biomass_reward
        - planting_cost
        - fert_cost
        - carrying_cost
        - catastrophe_cost
    )

    return float(reward)

