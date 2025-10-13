from __future__ import annotations
from dataclasses import dataclass


@dataclass(slots=True)
class GrowthParams:
    # Simplified Chapman-Richards parameters
    a: float = 400.0
    b: float = 0.03
    c: float = 1.6
    carbon_factor: float = 0.9 # tCO2e per m3


@dataclass(slots=True)
class DisturbanceParams:
    base_fire_prob: float = 0.01
    base_wind_prob: float = 0.01
    base_pest_prob: float = 0.02
    adsr_attack: float = 0.4
    adsr_decay: float = 3.0
    adsr_sustain: float = 0.1
    adsr_release: float = 5.0


@dataclass(slots=True)
class EconomicParams:
    price_per_m3: float = 45.0
    thin_cost_per_m3: float = 10.0
    fert_cost_per_ha: float = 150.0
    pesticide_cost_per_ha: float = 90.0
    rxfire_cost_per_ha: float = 120.0
    replant_cost_per_ha: float = 800.0
    discount_rate_annual: float = 0.04
    salvage_fraction: float = 0.35


@dataclass(slots=True)
class EnvConfig:
    dt_years: float = 1.0
    horizon_years: int = 40
    init_age: float = 3.0
    init_volume_m3: float = 5.0
    init_basal_area_m2: float = 2.5
    init_tph: float = 1200.0
    site_index: float = 22.0
    seed: int | None = 7
    growth: GrowthParams = GrowthParams()
    disturbance: DisturbanceParams = DisturbanceParams()
    economics: EconomicParams = EconomicParams()