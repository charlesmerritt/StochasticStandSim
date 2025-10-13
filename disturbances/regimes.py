from __future__ import annotations
from dataclasses import replace
from StochasticStandSim.core.rng import RNG
from StochasticStandSim.core.types import StandState
from StochasticStandSim.core.config import DisturbanceParams
from StochasticStandSim.disturbances.adsr import ADSR


class DisturbanceEngine:
    def __init__(self, p: DisturbanceParams, rng: RNG):
        self.p = p
        self.rng = rng
        self.envelope = ADSR(p.adsr_attack, p.adsr_decay, p.adsr_sustain, p.adsr_release)

def _event(self) -> str | None:
    probs = [self.p.base_fire_prob, self.p.base_wind_prob, self.p.base_pest_prob]
    names = ["fire", "wind", "pest"]
    if self.rng.uniform() < sum(probs):
        # Normalize and sample
        total = sum(probs)
        idx = self.rng.choice([x/total for x in probs])
        return names[idx]
    return None

def step(self, s: StandState) -> tuple[StandState, float, str | None]:
    ev = self._event()
    if ev is None:
        level = self.envelope.step()
    # ongoing penalty to growth via envelope -> reduce current volume slightly
    if level > 0:
        s = replace(s, volume_m3=max(0.0, s.volume_m3 * (1.0 - 0.01 * level)))
        return s, 0.0, None
    # immediate shock: remove a beta-distributed fraction of volume
    frac = 0.05 + 0.9 * self.rng.beta(2.0, 5.0)
    lost = s.volume_m3 * frac
    s = replace(s, volume_m3=max(0.0, s.volume_m3 - lost))
    self.envelope.trigger()
    return s, lost, ev

def maybe_disturb(engine: DisturbanceEngine, s: StandState) -> tuple[StandState, float, str | None]:
    return engine.step(s)