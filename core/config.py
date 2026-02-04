"""Configuration loader for MDP, growth, and economic parameters.

Provides a unified config entry point for:
- Economic parameters (prices, costs, discount rate)
- Stochastic noise parameters (sigma bounds per variable)
- Disturbance parameters (chronic/catastrophic rates and severity)
- Risk profiles (low/med/high presets)
- MDP discretization bins and action specs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import numpy as np
import yaml

from core.economics import EconParams, load_econ_params


# ---------------------------------------------------------------------------
# Noise parameters for stochastic growth
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NoiseParams:
    """Stochastic noise parameters for growth variables.
    
    These control the variability around deterministic PMRC projections.
    Noise is applied as lognormal (for BA, HD) or binomial/normal (for TPA).
    """
    sigma_log_ba: float = 0.10      # Log-scale std dev for basal area
    sigma_log_hd: float | None = None  # Log-scale std dev for height (None = deterministic)
    sigma_tpa: float = 20.0         # Absolute std dev for TPA (if not binomial)
    use_binomial_tpa: bool = True   # Use binomial survival model for TPA
    
    def scale(self, factor: float) -> "NoiseParams":
        """Return a scaled copy of noise parameters."""
        return NoiseParams(
            sigma_log_ba=self.sigma_log_ba * factor,
            sigma_log_hd=self.sigma_log_hd * factor if self.sigma_log_hd else None,
            sigma_tpa=self.sigma_tpa * factor,
            use_binomial_tpa=self.use_binomial_tpa,
        )


# ---------------------------------------------------------------------------
# Disturbance parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DisturbanceParams:
    """Parameters controlling disturbance frequency and severity.
    
    Mild disturbances: frequent, low-impact (e.g., minor wind, insects)
    Severe disturbances: rare, high-impact (e.g., major hurricane, fire)
    """
    # Probability of mild disturbance per year
    p_mild: float = 0.02
    
    # Mean return interval for severe disturbance (years)
    severe_mean_interval: float = 25.0
    
    # Multipliers applied to TPA/HD when disturbance occurs
    mild_tpa_multiplier: float = 0.85
    mild_hd_multiplier: float = 0.95
    severe_tpa_multiplier: float = 0.40
    severe_hd_multiplier: float = 0.80
    
    # Post-severe disturbance reset values
    severe_reset_age: float = 0.5
    severe_reset_tpa: float = 700.0
    
    @property
    def p_severe_annual(self) -> float:
        """Annual probability of severe disturbance."""
        if self.severe_mean_interval <= 0:
            return 0.0
        return 1.0 - np.exp(-1.0 / self.severe_mean_interval)


# ---------------------------------------------------------------------------
# Risk profiles (presets combining noise + disturbance)
# ---------------------------------------------------------------------------

RiskLevel = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class RiskProfile:
    """Combined noise and disturbance settings for a risk scenario."""
    name: str
    noise: NoiseParams
    disturbance: DisturbanceParams
    description: str = ""


# Default risk profiles for loblolly pine simulation
# Noise is additive Gaussian on the deterministic PMRC projection
RISK_PROFILES: Dict[RiskLevel, RiskProfile] = {
    "low": RiskProfile(
        name="Low Risk",
        noise=NoiseParams(
            sigma_log_ba=0.05,
            sigma_log_hd=None,  # Deterministic height
            sigma_tpa=10.0,
            use_binomial_tpa=True,
        ),
        disturbance=DisturbanceParams(
            p_mild=0.01,
            severe_mean_interval=50.0,  # 50-year return
            mild_tpa_multiplier=0.90,
            mild_hd_multiplier=0.98,
            severe_tpa_multiplier=0.50,
            severe_hd_multiplier=0.85,
        ),
        description="Low variability, infrequent disturbances",
    ),
    "medium": RiskProfile(
        name="Medium Risk",
        noise=NoiseParams(
            sigma_log_ba=0.10,
            sigma_log_hd=0.02,
            sigma_tpa=20.0,
            use_binomial_tpa=True,
        ),
        disturbance=DisturbanceParams(
            p_mild=0.02,
            severe_mean_interval=25.0,  # 25-year return
            mild_tpa_multiplier=0.85,
            mild_hd_multiplier=0.95,
            severe_tpa_multiplier=0.40,
            severe_hd_multiplier=0.80,
        ),
        description="Moderate variability and disturbance risk",
    ),
    "high": RiskProfile(
        name="High Risk",
        noise=NoiseParams(
            sigma_log_ba=0.15,
            sigma_log_hd=0.05,
            sigma_tpa=30.0,
            use_binomial_tpa=True,
        ),
        disturbance=DisturbanceParams(
            p_mild=0.05,
            severe_mean_interval=15.0,  # 15-year return
            mild_tpa_multiplier=0.80,
            mild_hd_multiplier=0.92,
            severe_tpa_multiplier=0.30,
            severe_hd_multiplier=0.70,
        ),
        description="High variability, frequent disturbances",
    ),
}


def get_risk_profile(level: RiskLevel) -> RiskProfile:
    """Get a predefined risk profile by level."""
    return RISK_PROFILES[level]


# ---------------------------------------------------------------------------
# MDP discretization and action specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MDPDiscretization:
    """Bin edges for discretizing continuous state space."""
    age_bins: Tuple[float, ...] = (0, 5, 10, 15, 20, 25, 30, 35, 40)
    tpa_bins: Tuple[float, ...] = (100, 200, 300, 400, 500, 600)
    ba_bins: Tuple[float, ...] = (0, 40, 80, 120, 160, 200)
    
    @property
    def n_age(self) -> int:
        return len(self.age_bins) - 1
    
    @property
    def n_tpa(self) -> int:
        return len(self.tpa_bins) - 1
    
    @property
    def n_ba(self) -> int:
        return len(self.ba_bins) - 1
    
    @property
    def n_states(self) -> int:
        return self.n_age * self.n_tpa * self.n_ba
    
    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return bin arrays for use with StateDiscretizer."""
        return (
            np.array(self.age_bins),
            np.array(self.tpa_bins),
            np.array(self.ba_bins),
        )


@dataclass(frozen=True)
class ActionSpec:
    """Specification of available management actions."""
    # Action 0: No-op (do nothing)
    # Action 1: Light thin (remove ~20% BA)
    # Action 2: Heavy thin (remove ~40% BA)
    # Action 3: Harvest and replant
    
    thin_fractions: Tuple[float, ...] = (1.0, 0.80, 0.60)  # BA retention fractions
    harvest_replant: bool = True
    
    @property
    def n_actions(self) -> int:
        n = len(self.thin_fractions)
        if self.harvest_replant:
            n += 1
        return n
    
    @property
    def action_names(self) -> List[str]:
        names = ["no-op"]
        for i, frac in enumerate(self.thin_fractions[1:], 1):
            pct = int((1 - frac) * 100)
            names.append(f"thin-{pct}%")
        if self.harvest_replant:
            names.append("harvest-replant")
        return names


# ---------------------------------------------------------------------------
# Unified simulation config
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """Complete configuration for stochastic forest simulation."""
    
    # Economic parameters
    econ: EconParams
    
    # Risk profile (noise + disturbance)
    risk_profile: RiskProfile
    
    # MDP discretization
    discretization: MDPDiscretization = field(default_factory=MDPDiscretization)
    
    # Action specification
    actions: ActionSpec = field(default_factory=ActionSpec)
    
    # Simulation parameters
    dt: float = 1.0  # Time step (years)
    max_age: float = 40.0  # Maximum simulation age
    discount_rate: float | None = None  # Override econ discount rate if set
    
    @property
    def effective_discount_rate(self) -> float:
        if self.discount_rate is not None:
            return self.discount_rate
        return self.econ.discount_rate


# ---------------------------------------------------------------------------
# Config loading utilities
# ---------------------------------------------------------------------------

def load_config(
    econ_path: str | Path | None = None,
    risk_level: RiskLevel = "medium",
) -> SimulationConfig:
    """Load simulation config from files and defaults.
    
    Args:
        econ_path: Path to economic parameters YAML. If None, uses default.
        risk_level: Risk profile to use ("low", "medium", "high").
    
    Returns:
        Complete SimulationConfig ready for use.
    """
    # Default econ path
    if econ_path is None:
        econ_path = Path(__file__).parent.parent / "data" / "econ_params.yaml"
    
    econ = load_econ_params(econ_path)
    risk_profile = get_risk_profile(risk_level)
    
    return SimulationConfig(
        econ=econ,
        risk_profile=risk_profile,
    )


def load_config_from_yaml(config_path: str | Path) -> SimulationConfig:
    """Load complete config from a YAML file.
    
    Expected format:
    ```yaml
    econ_path: data/econ_params.yaml
    risk_level: medium  # or custom noise/disturbance below
    noise:
      sigma_log_ba: 0.10
      sigma_tpa: 20.0
    disturbance:
      p_mild: 0.02
      severe_mean_interval: 25.0
    discretization:
      age_bins: [0, 5, 10, 15, 20, 25, 30, 35, 40]
      tpa_bins: [100, 200, 300, 400, 500, 600]
      ba_bins: [0, 40, 80, 120, 160, 200]
    ```
    """
    path = Path(config_path)
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    
    # Load econ params
    econ_path = data.get("econ_path")
    if econ_path:
        econ_path = path.parent / econ_path
    else:
        econ_path = Path(__file__).parent.parent / "data" / "econ_params.yaml"
    econ = load_econ_params(econ_path)
    
    # Build risk profile
    if "risk_level" in data and "noise" not in data and "disturbance" not in data:
        risk_profile = get_risk_profile(data["risk_level"])
    else:
        noise_data = data.get("noise", {})
        dist_data = data.get("disturbance", {})
        noise = NoiseParams(**noise_data) if noise_data else NoiseParams()
        disturbance = DisturbanceParams(**dist_data) if dist_data else DisturbanceParams()
        risk_profile = RiskProfile(
            name=data.get("profile_name", "Custom"),
            noise=noise,
            disturbance=disturbance,
        )
    
    # Build discretization
    disc_data = data.get("discretization", {})
    if disc_data:
        discretization = MDPDiscretization(
            age_bins=tuple(disc_data.get("age_bins", MDPDiscretization.age_bins)),
            tpa_bins=tuple(disc_data.get("tpa_bins", MDPDiscretization.tpa_bins)),
            ba_bins=tuple(disc_data.get("ba_bins", MDPDiscretization.ba_bins)),
        )
    else:
        discretization = MDPDiscretization()
    
    # Build action spec
    action_data = data.get("actions", {})
    if action_data:
        actions = ActionSpec(
            thin_fractions=tuple(action_data.get("thin_fractions", ActionSpec.thin_fractions)),
            harvest_replant=action_data.get("harvest_replant", True),
        )
    else:
        actions = ActionSpec()
    
    return SimulationConfig(
        econ=econ,
        risk_profile=risk_profile,
        discretization=discretization,
        actions=actions,
        dt=data.get("dt", 1.0),
        max_age=data.get("max_age", 40.0),
        discount_rate=data.get("discount_rate"),
    )


__all__ = [
    "NoiseParams",
    "DisturbanceParams",
    "RiskLevel",
    "RiskProfile",
    "RISK_PROFILES",
    "get_risk_profile",
    "MDPDiscretization",
    "ActionSpec",
    "SimulationConfig",
    "load_config",
    "load_config_from_yaml",
]