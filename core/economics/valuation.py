"""Deterministic valuation routines used by :mod:`core.stand_env`."""

from __future__ import annotations

from typing import Mapping, MutableMapping, Any

import numpy as np

__all__ = ["bellman_value"]


def bellman_value(
    state: Mapping[str, Any],
    horizon: int,
    discount: float,
    econ_cfg: MutableMapping[str, Any],
) -> float:
    """Compute a simplified Bellman value.

    We approximate the value by discounting the expected terminal biomass and
    accounting for a catastrophic loss probability derived from the current
    stand risk.  The helper is intentionally modest so that unit tests remain
    lightweight.
    """

    econ_cfg = econ_cfg or {}
    terminal_price = float(econ_cfg.get("terminal_price", 80.0))
    salvage_rate = float(np.clip(econ_cfg.get("salvage_rate", 0.25), 0.0, 1.0))

    biomass = float(state.get("biomass", 0.0))
    risk = float(np.clip(state.get("risk", 0.0), 0.0, 1.0))
    discount = max(discount, 1e-6)

    expected_terminal = biomass * ((1.0 - risk) * terminal_price + risk * salvage_rate * terminal_price)
    discount_factor = 1.0 / ((1.0 + discount) ** max(horizon, 0))

    return expected_terminal * discount_factor
