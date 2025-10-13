"""Utility reward helpers for the stand environment."""

from __future__ import annotations

from typing import Mapping, MutableMapping, Dict, Any

import numpy as np

__all__ = ["step_reward"]


def step_reward(
    prev_state: Mapping[str, Any],
    action: np.ndarray,
    next_state: Mapping[str, Any],
    econ_cfg: MutableMapping[str, Any],
) -> float:
    """Compute the immediate reward for a transition.

    The heuristic combines revenue from thinning, a carrying cost for holding
    inventory, and a penalty for catastrophic losses.
    """

    econ_cfg = econ_cfg or {}

    price = float(econ_cfg.get("stumpage_price", 50.0))
    holding_cost = float(econ_cfg.get("holding_cost", 1.0))
    catastrophe_penalty = float(econ_cfg.get("catastrophe_penalty", 5000.0))

    thin_pct = float(np.clip(action[0], 0.0, 1.0))
    biomass_prev = float(prev_state.get("biomass", 0.0))

    revenue = biomass_prev * thin_pct * price
    carrying_cost = holding_cost * float(next_state.get("age", 0.0))
    penalty = catastrophe_penalty if bool(next_state.get("catastrophic", False)) else 0.0

    return revenue - carrying_cost - penalty
