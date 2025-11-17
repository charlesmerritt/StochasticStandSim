from __future__ import annotations
from dataclasses import dataclass, field

from .rng import rng


def get_severity(seed: int | None = None) -> float:
    """Random severity in (0,1), never exactly 0 or 1."""
    val = rng(seed)
    if val <= 0.0: return 0.001
    if val >= 1.0: return 0.999
    return round(val, 3)

@dataclass
class DisturbanceEvent:
    """A single catastrophic disturbance based on Buongiorno's work. Severity is a random float in (0,1), frequency follows an exponential distribution."""
    severity: float
    frequency: float
    
