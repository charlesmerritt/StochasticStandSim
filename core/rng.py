from __future__ import annotations
import numpy as np


class RNG:
    def __init__(self, seed: int | None):
        self._rng = np.random.default_rng(seed)
    
    def uniform(self) -> float:
        return float(self._rng.uniform())
    
    def choice(self, p: list[float]) -> int:
        return int(self._rng.choice(len(p), p=p))
    
    def normal(self, mu: float, sigma: float) -> float:
        return float(self._rng.normal(mu, sigma))
        
    def beta(self, a: float, b: float) -> float:
        return float(self._rng.beta(a, b))