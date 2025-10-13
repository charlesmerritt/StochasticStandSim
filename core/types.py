from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, TypedDict


ActionName = Literal["noop", "thin", "fertilize", "pesticide", "rx_fire", "harvest_replant"]


@dataclass(slots=True)
class StandState:
    age: float # years
    volume_m3: float # standing merchantable volume
    basal_area_m2: float # basal area
    trees_per_ha: float
    carbon_tCO2e: float
    site_index: float # site productivity index
    cash_account: float # bookkeeping
    t: int # timestep index


class ObsDict(TypedDict):
    age: float
    volume_m3: float
    basal_area_m2: float
    trees_per_ha: float
    carbon_tCO2e: float
    site_index: float