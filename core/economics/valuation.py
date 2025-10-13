"""Deterministic valuation utilities for :mod:`core.stand_env`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, MutableMapping

import numpy as np

from .reward_functions import _resolve_region_prices, _resolve_site_index, _site_premium

__all__ = ["bellman_value"]


@dataclass(frozen=True)
class _ValueContext:
    prices: Dict[str, float]
    site_index: float
    site_factor: float
    discount: float
    horizon: int


def _discount_stream(cashflows: np.ndarray, discount: float) -> float:
    periods = np.arange(1, cashflows.size + 1, dtype=np.float64)
    factors = (1.0 + discount) ** periods
    return float(np.sum(cashflows / factors))


def _build_context(state: Mapping[str, Any], horizon: int, discount: float, econ_cfg: Mapping[str, Any]) -> _ValueContext:
    prices = _resolve_region_prices(econ_cfg)
    site_index = _resolve_site_index(state, econ_cfg)
    site_factor = _site_premium(site_index, baseline=70.0, slope=float(econ_cfg.get("site_index_slope", 0.0035)))
    return _ValueContext(prices=prices, site_index=site_index, site_factor=site_factor, discount=discount, horizon=horizon)


def _expected_growth(state: Mapping[str, Any], ctx: _ValueContext, econ_cfg: Mapping[str, Any]) -> float:
    base_growth = float(econ_cfg.get("mean_annual_biomass", 2.4))
    maturity = np.clip(float(state.get("age", 0.0)) / float(econ_cfg.get("rotation_age", 28.0)), 0.0, 2.0)
    productivity = np.clip(ctx.site_index / float(econ_cfg.get("site_index_reference", 70.0)), 0.5, 1.6)
    modifier = float(econ_cfg.get("growth_modifier", 1.0))
    # Reduce growth once stands exceed the nominal rotation age.
    slowdown = np.exp(-max(maturity - 1.0, 0.0))
    return base_growth * productivity * modifier * slowdown


def _rotation_cashflows(state: Mapping[str, Any], ctx: _ValueContext, econ_cfg: Mapping[str, Any]) -> np.ndarray:
    horizon = ctx.horizon
    if horizon <= 0:
        return np.zeros(0, dtype=np.float64)

    annual_growth = _expected_growth(state, ctx, econ_cfg)
    biomass_price = ctx.prices.get("biomass", 17.5) * ctx.site_factor
    thin_cost = ctx.prices.get("thin_cost", 6.0)

    carrying = float(econ_cfg.get("annual_carrying_cost", 8.0))
    maintenance = float(econ_cfg.get("annual_maintenance_cost", 5.0))
    annual_cost = carrying + maintenance

    risk = float(np.clip(state.get("risk", econ_cfg.get("base_risk", 0.0)), 0.0, 1.0))
    salvage_discount = float(ctx.prices.get("salvage_discount", 0.55))
    salvage_eff = float(econ_cfg.get("salvage_recovery_rate", 0.55))

    flows = np.full(horizon, -annual_cost, dtype=np.float64)

    for t in range(horizon):
        stand_age = float(state.get("age", 0.0)) + (t + 1) * float(econ_cfg.get("timestep_years", 1.0))
        age_factor = np.clip(stand_age / float(econ_cfg.get("rotation_age", 28.0)), 0.0, 2.0)
        removal = max(annual_growth * (1.0 - np.exp(-age_factor)), 0.0)
        flows[t] += (biomass_price - thin_cost) * removal

    # Terminal harvest at the end of the planning horizon.
    terminal_biomass = float(state.get("biomass", 0.0)) + annual_growth * horizon
    expected_terminal = (1.0 - risk) * terminal_biomass * biomass_price
    expected_salvage = risk * terminal_biomass * biomass_price * salvage_discount * salvage_eff
    flows[-1] += expected_terminal + expected_salvage

    establishment = float(econ_cfg.get("establishment_cost", 260.0))
    if horizon > 0:
        flows[0] -= establishment

    return flows


def _land_expectation_value(state: Mapping[str, Any], ctx: _ValueContext, econ_cfg: Mapping[str, Any]) -> float:
    flows = _rotation_cashflows(state, ctx, econ_cfg)
    if flows.size == 0:
        return 0.0
    npv_rotation = _discount_stream(flows, ctx.discount)
    capitalisation = (1.0 + ctx.discount) ** ctx.horizon - 1.0
    if capitalisation <= 0:
        return npv_rotation
    return npv_rotation * ((1.0 + ctx.discount) ** ctx.horizon) / capitalisation


def bellman_value(
    state: Mapping[str, Any],
    horizon: int,
    discount: float,
    econ_cfg: MutableMapping[str, Any],
) -> float:
    """Estimate the continuation value of ``state`` over ``horizon`` periods."""

    econ_cfg = econ_cfg or {}
    horizon = max(int(horizon), 0)
    discount = max(float(discount), 1e-6)

    ctx = _build_context(state, horizon, discount, econ_cfg)
    lev = _land_expectation_value(state, ctx, econ_cfg)

    realised = float(state.get("value", 0.0))
    return realised + lev

