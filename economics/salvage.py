from __future__ import annotations
from StochasticStandSim.core.config import EconomicParams


def salvage_value(lost_volume_m3: float, econ: EconomicParams) -> float:
    return econ.salvage_fraction * lost_volume_m3 * econ.price_per_m3