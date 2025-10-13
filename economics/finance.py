from __future__ import annotations
from StochasticStandSim.core.utils import discount_factor
from StochasticStandSim.core.config import EconomicParams

def npv_update_cash(cash: float, amount: float, t_years: float, econ: EconomicParams) -> float:
    return cash + amount * discount_factor(econ.discount_rate_annual, t_years)

def lev(npv: float, rotation_years: float, r: float) -> float:
    # Land Expectation Value with infinite identical rotations (Faustmann)
    return npv * r / (1.0 - (1.0 + r) ** (-rotation_years))