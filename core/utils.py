from __future__ import annotations
import numpy as np


def discount_factor(rate_annual: float, years: float) -> float:
    return float((1.0 + rate_annual) ** (-years))