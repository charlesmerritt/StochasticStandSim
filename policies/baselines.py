from __future__ import annotations
from StochasticStandSim.core.types import StandState
from StochasticStandSim.core.actions import ACTIONS

def greedy_harvest_policy(s: StandState) -> int:
    # harvest when volume exceeds simple threshold
    return 5 if s.volume_m3 > 300.0 else 0 # index 5 -> harvest_replant, 0 -> noop