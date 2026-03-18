"""Disturbance modeling for stochastic forest simulation.

This module handles aleatoric uncertainty from catastrophic events:
- Disturbance occurrence (Bernoulli draws)
- Severity sampling (Beta distribution)
- State variable shocks (TPA, BA, HD reductions)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from core.state import StandState


class DisturbanceType(Enum):
    """Classification of disturbance events."""
    NONE = "none"
    MILD = "mild"
    SEVERE = "severe"


@dataclass
class DisturbanceEvent:
    """Record of a disturbance occurrence and its effects.
    
    Attributes:
        occurred: Whether any disturbance occurred this period
        dtype: Type of disturbance (none, mild, severe)
        severity: Severity draw from Beta distribution (0-1 scale)
        tpa_loss: Absolute TPA reduction
        ba_loss: Absolute BA reduction
        hd_loss: Absolute HD reduction
    """
    occurred: bool
    dtype: DisturbanceType
    severity: float = 0.0
    tpa_loss: float = 0.0
    ba_loss: float = 0.0
    hd_loss: float = 0.0


@dataclass
class DisturbanceParams:
    """Parameters controlling disturbance occurrence and severity.
    
    Aleatoric uncertainty parameters for catastrophic events.
    
    Attributes:
        p_mild: Annual probability of mild disturbance
        severe_mean_interval: Mean return interval for severe events (years)
        mild_tpa_multiplier: TPA retention fraction after mild event (e.g., 0.8 = 20% loss)
        severe_tpa_multiplier: TPA retention fraction after severe event
        mild_hd_multiplier: HD retention fraction after mild event
        severe_hd_multiplier: HD retention fraction after severe event
        severe_reset_age: Age multiplier after severe event (partial stand reset)
        severe_reset_tpa: TPA to reset to after severe event
        c_tpa: Sensitivity coefficient for TPA shock (default 1.0)
        c_ba: Sensitivity coefficient for BA shock (default 1.0)
        c_hd: Sensitivity coefficient for HD shock (default 0.0, height typically unaffected)
    """
    p_mild: float = 0.02
    severe_mean_interval: float = 25.0
    mild_tpa_multiplier: float = 0.8
    severe_tpa_multiplier: float = 0.4
    mild_hd_multiplier: float = 0.95
    severe_hd_multiplier: float = 0.8
    severe_reset_age: float = 0.5
    severe_reset_tpa: float = 700.0
    c_tpa: float = 1.0
    c_ba: float = 1.0
    c_hd: float = 0.0

    @property
    def p_severe(self) -> float:
        """Annual probability of severe disturbance."""
        return 1.0 / self.severe_mean_interval if self.severe_mean_interval > 0 else 0.0


class DisturbanceModel:
    """Samples and applies disturbance shocks to stand state.
    
    Implements the aleatoric uncertainty from catastrophic events as described
    in PLANNING.md Section 4.3-4.5:
    - Bernoulli occurrence draws
    - Beta severity sampling
    - Proportional shocks to atomic state variables only
    """

    def __init__(self, params: DisturbanceParams | None = None) -> None:
        self.params: DisturbanceParams = params or DisturbanceParams()

    def sample_occurrence(self, rng: np.random.Generator) -> DisturbanceType:
        """Sample whether a disturbance occurs this period.
        
        Args:
            rng: NumPy random generator
            
        Returns:
            DisturbanceType indicating none, mild, or severe
        """
        u = rng.random()
        if u < self.params.p_severe:
            return DisturbanceType.SEVERE
        elif u < self.params.p_severe + self.params.p_mild:
            return DisturbanceType.MILD
        return DisturbanceType.NONE

    def sample_severity(
        self,
        dtype: DisturbanceType,
        rng: np.random.Generator,
    ) -> float:
        """Sample severity conditional on disturbance type.
        
        For mild events, severity is drawn from a Beta distribution centered
        around (1 - mild_multiplier). For severe events, centered around
        (1 - severe_multiplier).
        
        Args:
            dtype: Type of disturbance
            rng: NumPy random generator
            
        Returns:
            Severity in [0, 1] where 0 = no damage, 1 = total loss
        """
        if dtype == DisturbanceType.NONE:
            return 0.0
        
        if dtype == DisturbanceType.MILD:
            # Mean severity around 1 - mild_multiplier (e.g., 0.2 for 80% retention)
            mean_sev = 1.0 - self.params.mild_tpa_multiplier
        else:  # SEVERE
            mean_sev = 1.0 - self.params.severe_tpa_multiplier
        
        # Use Beta distribution with concentration kappa=12
        kappa = 12.0
        alpha = mean_sev * kappa
        beta = (1.0 - mean_sev) * kappa
        
        # Ensure valid parameters
        alpha = max(0.1, alpha)
        beta = max(0.1, beta)
        
        return float(rng.beta(alpha, beta))

    def apply_shock(
        self,
        state: StandState,
        dtype: DisturbanceType,
        severity: float,
    ) -> tuple[StandState, DisturbanceEvent]:
        """Apply disturbance shock to atomic state variables.
        
        Shocks are applied as proportional reductions:
            x_post = x_pre * (1 - c_x * severity)
        
        Only atomic variables (TPA, BA, HD) are shocked. Derived quantities
        (volume, products) should be recomputed from post-shock atomics.
        
        Args:
            state: Current stand state (will not be mutated)
            dtype: Type of disturbance
            severity: Severity draw in [0, 1]
            
        Returns:
            Tuple of (new_state, event_record)
        """
        if dtype == DisturbanceType.NONE or severity <= 0:
            return state, DisturbanceEvent(
                occurred=False,
                dtype=DisturbanceType.NONE,
            )
        
        # Calculate losses using sensitivity coefficients
        tpa_loss = state.tpa * self.params.c_tpa * severity
        ba_loss = state.ba * self.params.c_ba * severity
        hd_loss = state.hd * self.params.c_hd * severity
        
        # Apply shocks to atomic variables
        new_tpa = max(1.0, state.tpa - tpa_loss)
        new_ba = max(0.0, state.ba - ba_loss)
        new_hd = max(1.0, state.hd - hd_loss)  # Height shouldn't go below 1 ft
        
        # For severe events, optionally reset age and TPA
        new_age = state.age
        if dtype == DisturbanceType.SEVERE:
            new_age = max(1.0, state.age * self.params.severe_reset_age)
            # Optionally cap TPA at reset value for severe events
            if self.params.severe_reset_tpa > 0:
                new_tpa = min(new_tpa, self.params.severe_reset_tpa)
        
        new_state = StandState(
            age=new_age,
            hd=new_hd,
            tpa=new_tpa,
            ba=new_ba,
            si25=state.si25,
            region=state.region,
            phwd=state.phwd,
        )
        
        event = DisturbanceEvent(
            occurred=True,
            dtype=dtype,
            severity=severity,
            tpa_loss=tpa_loss,
            ba_loss=ba_loss,
            hd_loss=hd_loss,
        )
        
        return new_state, event

    def sample_and_apply(
        self,
        state: StandState,
        rng: np.random.Generator,
    ) -> tuple[StandState, DisturbanceEvent]:
        """Sample disturbance occurrence, severity, and apply shock.
        
        Convenience method that chains sample_occurrence -> sample_severity -> apply_shock.
        
        Args:
            state: Current stand state
            rng: NumPy random generator
            
        Returns:
            Tuple of (new_state, event_record)
        """
        dtype = self.sample_occurrence(rng)
        severity = self.sample_severity(dtype, rng)
        return self.apply_shock(state, dtype, severity)
