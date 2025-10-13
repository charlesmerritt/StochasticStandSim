from __future__ import annotations
from StochasticStandSim.core.config import EconomicParams

def revenue_for_harvest(volume_m3: float, econ: EconomicParams) -> float:
    return volume_m3 * econ.price_per_m3

def thinning_revenue_cost(thin_volume_m3: float, econ: EconomicParams) -> float:
    # thinning revenue minus cost
    return thin_volume_m3 * (econ.price_per_m3 - econ.thin_cost_per_m3)