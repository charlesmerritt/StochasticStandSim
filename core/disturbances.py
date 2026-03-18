"""Disturbance modeling for stochastic forest simulation.

This module handles aleatoric uncertainty from catastrophic events per PLANNING.md
Sections 4.3-4.5:

- **Occurrence**: Annual Bernoulli trial with probability p_dist = 1/n, where n is
  the mean return interval in years (e.g., n=20 → 5% annual probability).

- **Severity**: Conditional on occurrence, severity q ~ Beta(α, β) where:
    α = m_q * κ
    β = (1 - m_q) * κ
  m_q is the mean severity (default 0.30) and κ is concentration (default 12).

- **Shock**: One-time proportional reduction to atomic state variables:
    x_post = x_pre * (1 - c_x * q)
  where c_x is the sensitivity coefficient for variable x.

This is a single generic disturbance type. Frequency varies by scenario (p_dist),
but severity is always drawn from the same Beta distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.state import StandState


@dataclass
class DisturbanceEvent:
    """Record of a disturbance occurrence and its effects.
    
    Attributes:
        occurred: Whether a disturbance occurred this period
        severity: Severity draw from Beta distribution (0-1 scale), 0 if no disturbance
        tpa_loss: Absolute TPA reduction
        ba_loss: Absolute BA reduction
        hd_loss: Absolute HD reduction
    """
    occurred: bool
    severity: float = 0.0
    tpa_loss: float = 0.0
    ba_loss: float = 0.0
    hd_loss: float = 0.0


@dataclass
class DisturbanceParams:
    """Parameters controlling disturbance occurrence and severity.
    
    Implements the disturbance model from PLANNING.md Sections 4.3-4.5:
    - Bernoulli occurrence with probability p_dist
    - Beta severity with mean m_q and concentration kappa
    - Proportional shocks to atomic variables with sensitivity coefficients
    
    Attributes:
        p_dist: Annual probability of disturbance (default 0 = no disturbances).
                Use 1/n for n-year mean return interval (e.g., 1/20 = 0.05).
        severity_mean: Mean of Beta severity distribution (m_q, default 0.30). Fixed to a moderate disturbance regime for now. "Disturbances are 30% severe on average"
        severity_kappa: Concentration of Beta distribution (κ, default 12).
                        Higher κ = less variability around the mean.
        c_tpa: Sensitivity coefficient for TPA shock (default 1.0)
        c_ba: Sensitivity coefficient for BA shock (default 1.0)
        c_hd: Sensitivity coefficient for HD shock (default 0.0, height typically unaffected)
    """
    p_dist: float = 0.0
    severity_mean: float = 0.30
    severity_kappa: float = 12.0
    c_tpa: float = 1.0
    c_ba: float = 1.0
    c_hd: float = 0.0


class DisturbanceModel:
    """Samples and applies disturbance shocks to stand state.
    
    Implements the aleatoric uncertainty from catastrophic events as described
    in PLANNING.md Sections 4.3-4.5:
    - Bernoulli occurrence with probability p_dist
    - Beta severity sampling with mean m_q and concentration κ
    - Proportional shocks to atomic state variables only
    """

    def __init__(self, params: DisturbanceParams | None = None) -> None:
        self.params: DisturbanceParams = params or DisturbanceParams()

    def sample_occurrence(self, rng: np.random.Generator) -> bool:
        """Sample whether a disturbance occurs this period.
        
        Bernoulli trial with probability p_dist.
        
        Args:
            rng: NumPy random generator
            
        Returns:
            True if disturbance occurs, False otherwise
        """
        if self.params.p_dist <= 0:
            return False
        return rng.random() < self.params.p_dist

    def sample_severity(self, rng: np.random.Generator) -> float:
        """Sample severity from Beta distribution.
        
        Severity q ~ Beta(α, β) where:
            α = m_q * κ
            β = (1 - m_q) * κ
        
        Args:
            rng: NumPy random generator
            
        Returns:
            Severity in [0, 1] where 0 = no damage, 1 = total loss
        """
        m_q = self.params.severity_mean
        kappa = self.params.severity_kappa
        
        alpha = m_q * kappa
        beta = (1.0 - m_q) * kappa
        
        # Ensure valid parameters (alpha, beta > 0)
        alpha = max(0.1, alpha)
        beta = max(0.1, beta)
        
        return float(rng.beta(alpha, beta))

    def apply_shock(
        self,
        state: StandState,
        severity: float,
    ) -> tuple[StandState, DisturbanceEvent]:
        """Apply disturbance shock to atomic state variables.
        
        Shocks are applied as proportional reductions:
            x_post = x_pre * (1 - c_x * severity)
        
        Only atomic variables (TPA, BA, HD) are shocked. Derived quantities
        (volume, products) should be recomputed from post-shock atomics.
        
        Args:
            state: Current stand state (will not be mutated)
            severity: Severity draw in [0, 1]
            
        Returns:
            Tuple of (new_state, event_record)
        """
        if severity <= 0:
            return state, DisturbanceEvent(occurred=False)
        
        # Calculate losses using sensitivity coefficients
        tpa_loss = state.tpa * self.params.c_tpa * severity
        ba_loss = state.ba * self.params.c_ba * severity
        hd_loss = state.hd * self.params.c_hd * severity
        
        # Apply shocks to atomic variables
        new_tpa = max(1.0, state.tpa - tpa_loss)
        new_ba = max(0.0, state.ba - ba_loss)
        new_hd = max(1.0, state.hd - hd_loss)  # Height shouldn't go below 1 ft
        
        new_state = StandState(
            age=state.age,  # Age unchanged by disturbance
            hd=new_hd,
            tpa=new_tpa,
            ba=new_ba,
            si25=state.si25,
            region=state.region,
            phwd=state.phwd,
        )
        
        event = DisturbanceEvent(
            occurred=True,
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
        if not self.sample_occurrence(rng):
            return state, DisturbanceEvent(occurred=False)
        
        severity = self.sample_severity(rng)
        return self.apply_shock(state, severity)
