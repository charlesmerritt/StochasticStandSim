"""Economics helpers for reward calculation and parameter loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional

import yaml


@dataclass(frozen=True)
class EconParams:
    """Container for economic parameters used when valuing stand actions."""

    prices: Dict[str, float]
    costs: Dict[str, float]
    discount_rate: float
    raw: Mapping[str, Any]

    def price_weighted_average(self, weights: Optional[Mapping[str, float]] = None) -> float:
        """Return a weighted average over available product prices."""
        if not self.prices:
            return 0.0
        if weights:
            total = 0.0
            weight_sum = 0.0
            for key, weight in weights.items():
                if key not in self.prices:
                    continue
                total += self.prices[key] * weight
                weight_sum += weight
            if weight_sum > 0.0:
                return total / weight_sum
        # Fallback: simple mean across listed products
        return sum(self.prices.values()) / float(len(self.prices))


def _parse_numeric(value: Any) -> float:
    """Convert YAML scalars like ``25.92/ton`` into floats."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Empty string value in econ params.")
        # Drop optional currency symbols and units (after slash).
        if "/" in cleaned:
            cleaned = cleaned.split("/", 1)[0]
        cleaned = cleaned.replace("$", "").strip()
        return float(cleaned)
    raise TypeError(f"Unsupported value type {type(value)!r} for economic parameter.")


def _parse_section(data: Mapping[str, Any] | None) -> Dict[str, float]:
    if not data:
        return {}
    parsed: Dict[str, float] = {}
    for key, value in data.items():
        parsed[key] = _parse_numeric(value)
    return parsed


def load_econ_params(path: str | Path) -> EconParams:
    """Load economic parameters from ``data/econ_params.yaml`` style files."""
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Economic parameter file not found: {path_obj}")
    with path_obj.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}

    if not isinstance(payload, MutableMapping):
        raise ValueError(f"Economic parameter file {path_obj} must be a mapping.")

    prices = _parse_section(payload.get("prices"))
    costs = _parse_section(payload.get("costs"))
    discount_section = payload.get("discount", {})
    discount_rate = _parse_numeric(discount_section.get("rate_annual", 0.0)) if discount_section else 0.0

    return EconParams(prices=prices, costs=costs, discount_rate=float(discount_rate), raw=payload)


__all__ = ["EconParams", "load_econ_params"]
