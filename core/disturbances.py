from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math
import numpy as np

from .rng import rng as legacy_rng


# ----------------------------- core event model ----------------------------- #


@dataclass
class DisturbanceEvent:
    """Single disturbance applied once when the stand passes start_age."""

    start_age: float
    severity: float
    category: str
    ba_loss_fraction: float
    tpa_loss_fraction: float
    hd_loss_fraction: float
    disturbance_level: Optional[str] = None
    triggered: bool = False


def _bounded(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def make_fire_event(*, start_age: float, severity: float, seed: int | None = None) -> DisturbanceEvent:
    """Factory used by the gym env for stochastic fire events."""
    sev = _bounded(severity if severity is not None else legacy_rng(seed), 0.0, 1.0)
    # Fires mostly remove BA/TPA; moderate impact on height
    return DisturbanceEvent(
        start_age=float(start_age),
        severity=sev,
        category="fire",
        ba_loss_fraction=sev * 0.8,
        tpa_loss_fraction=sev * 0.6,
        hd_loss_fraction=sev * 0.25,
        disturbance_level="severe" if sev >= 0.7 else "moderate",
    )


def make_wind_event(*, start_age: float, severity: float, seed: int | None = None) -> DisturbanceEvent:
    """Factory used by the gym env for stochastic wind events."""
    sev = _bounded(severity if severity is not None else legacy_rng(seed), 0.0, 1.0)
    # Wind throws fewer trees than fire but can still damage crowns
    return DisturbanceEvent(
        start_age=float(start_age),
        severity=sev,
        category="wind",
        ba_loss_fraction=sev * 0.5,
        tpa_loss_fraction=sev * 0.4,
        hd_loss_fraction=sev * 0.15,
        disturbance_level="severe" if sev >= 0.6 else "chronic",
    )


# ------------------------ disturbance generators --------------------------- #


class CatastrophicDisturbanceGenerator:
    """
    Poisson-like generator for severe events that effectively reset the stand.

    mean_interval_years: expected interval between events (exponential waiting time).
    """

    def __init__(self, mean_interval_years: float = 25.0, *, severity_min: float = 0.7, severity_max: float = 1.0):
        self.mean_interval = max(1e-6, float(mean_interval_years))
        self.sev_min = severity_min
        self.sev_max = severity_max

    def sample_event(self, current_age: float, *, rng: np.random.Generator | None = None) -> DisturbanceEvent:
        rng = rng or np.random.default_rng()
        wait = rng.exponential(self.mean_interval)
        start_age = float(current_age + wait)
        sev = float(rng.uniform(self.sev_min, self.sev_max))
        # A catastrophic event is represented by near-total loss; env/policy can choose to replant.
        return DisturbanceEvent(
            start_age=start_age,
            severity=sev,
            category="catastrophic",
            ba_loss_fraction=_bounded(sev, 0.0, 1.0),
            tpa_loss_fraction=_bounded(sev, 0.0, 1.0),
            hd_loss_fraction=_bounded(sev * 0.5, 0.0, 1.0),
            disturbance_level="severe",
        )


class ChronicDisturbanceGenerator:
    """
    Higher-frequency, low-severity disturbances that behave like transition noise.
    """

    def __init__(
        self,
        mean_interval_years: float = 6.0,
        *,
        max_loss: float = 0.25,
        hd_scale: float = 0.1,
    ):
        self.mean_interval = max(1e-6, float(mean_interval_years))
        self.max_loss = max_loss
        self.hd_scale = hd_scale

    def sample_event(self, current_age: float, *, rng: np.random.Generator | None = None) -> DisturbanceEvent:
        rng = rng or np.random.default_rng()
        wait = rng.exponential(self.mean_interval)
        start_age = float(current_age + wait)
        sev = float(rng.uniform(0.0, self.max_loss))
        return DisturbanceEvent(
            start_age=start_age,
            severity=sev,
            category="chronic",
            ba_loss_fraction=sev,
            tpa_loss_fraction=sev * 0.8,
            hd_loss_fraction=sev * self.hd_scale,
            disturbance_level="chronic",
        )


__all__ = [
    "DisturbanceEvent",
    "CatastrophicDisturbanceGenerator",
    "ChronicDisturbanceGenerator",
    "make_fire_event",
    "make_wind_event",
]
