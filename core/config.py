"""Scenario configuration dataclasses.

This module defines pure data structures for scenario parameters.
No simulation logic - just configuration that can be serialized and compared.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from core.disturbances import DisturbanceParams
from core.process_noise import NoiseParams
from core.products import HarvestCosts, ProductPrices
from core.state import Region


@dataclass
class ThinningParams:
    """Parameters for mid-rotation thinning rule (Basal Area Threshold policy).
    
    Implements a BA threshold rule: at a specified age, if BA exceeds
    the threshold, thin to the target residual BA.
    
    Default values implement the standard BAT policy:
    - Trigger at age 15
    - Thin when BA > 150 ft²/ac
    - Target residual BA of 100 ft²/ac
    
    Attributes:
        trigger_age: Age at which to evaluate thinning (years)
        ba_threshold: BA threshold to trigger thinning (ft²/ac)
        residual_ba: Target BA after thinning (ft²/ac)
        thin_cost: Fixed cost of thinning operation ($/ac)
    """
    trigger_age: float = 15.0
    ba_threshold: float = 150.0
    residual_ba: float = 100.0
    thin_cost: float = 87.34


@dataclass
class ScenarioConfig:
    """Configuration for a simulation scenario.
    
    This is a pure data structure containing all parameters needed to run
    a scenario. It can be serialized, compared, and reused.
    
    Attributes:
        name: Scenario identifier
        scenario_type: "deterministic" or "stochastic"
        
        # Initial conditions
        age0: Initial stand age (years)
        tpa0: Initial trees per acre
        si25: Site index at base age 25 (ft)
        region: PMRC coefficient region
        
        # Horizon
        rotation_length: Years to simulate
        
        # Management
        thin_params: Thinning policy parameters (None = no thinning)
        
        # Stochastic parameters (ignored for deterministic)
        noise_params: Process noise parameters
        disturbance_params: Disturbance parameters
        
        # Economics
        discount_rate: Annual discount rate for NPV/LEV
        prices: Stumpage prices by product class
        costs: Harvest and regeneration costs
        
        # Monte Carlo (for stochastic scenarios)
        n_trajectories: Number of Monte Carlo trajectories
        seed: Random seed for reproducibility (None = random)
    """
    
    # Scenario metadata
    name: str
    scenario_type: Literal["deterministic", "stochastic"] = "deterministic"
    
    # Initial conditions (defaults from docs/scenario_defaults.md)
    age0: float = 5.0
    tpa0: float = 850.0
    si25: float = 80.0
    region: Region = "ucp"
    
    # Horizon
    rotation_length: int = 35
    
    # Management policy (None = no thinning by default)
    thin_params: ThinningParams | None = None
    
    # Stochastic parameters (only used if scenario_type == "stochastic")
    noise_params: NoiseParams | None = None
    disturbance_params: DisturbanceParams | None = None
    
    # Economics
    discount_rate: float = 0.05
    prices: ProductPrices | None = field(default_factory=ProductPrices)
    costs: HarvestCosts | None = field(default_factory=HarvestCosts)
    
    # Monte Carlo settings
    n_trajectories: int = 1000
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.scenario_type not in ("deterministic", "stochastic"):
            raise ValueError(f"Invalid scenario_type: {self.scenario_type}")
        if self.age0 <= 0:
            raise ValueError("age0 must be positive")
        if self.tpa0 <= 0:
            raise ValueError("tpa0 must be positive")
        if self.si25 <= 0:
            raise ValueError("si25 must be positive")
        if self.rotation_length <= 0:
            raise ValueError("rotation_length must be positive")
        if not 0 < self.discount_rate < 1:
            raise ValueError("discount_rate must be between 0 and 1")

    def with_updates(self, **kwargs) -> ScenarioConfig:
        """Create a new config with updated values.
        
        Example:
            high_risk = BASELINE.with_updates(
                name="high_risk",
                disturbance_params=DisturbanceParams(base_prob=0.10)
            )
        """
        from dataclasses import asdict
        current = asdict(self)
        current.update(kwargs)
        return ScenarioConfig(**current)
