"""Heuristic policies for comparison.

All policies implement __call__(state) -> action, where:
- state is a StandState
- action is an integer index into ActionSpec

Policies:
- NoOpPolicy: Always do nothing
- FixedRotationPolicy: Harvest at fixed age, thin at fixed ages
- ThresholdThinPolicy: Thin when BA exceeds threshold at eligible ages
- VolumeThresholdPolicy: Thin when volume >= X at age >= A
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from core.stochastic_stand import StandState
    from core.pmrc_model import PMRCModel


class Policy(Protocol):
    """Protocol for management policies."""
    
    def __call__(self, state: "StandState") -> int:
        """Return action index given current state."""
        ...


@dataclass
class NoOpPolicy:
    """Always do nothing (action 0)."""
    
    def __call__(self, state: "StandState") -> int:
        return 0


@dataclass
class FixedRotationPolicy:
    """Harvest at fixed rotation age, with optional thinning schedule.
    
    Actions:
    - 0: no-op
    - 1: light thin (20%)
    - 2: heavy thin (40%)
    - 3: harvest-replant
    """
    rotation_age: float = 30.0
    thin_ages: tuple[float, ...] = (12.0, 18.0)
    thin_action: int = 1  # Light thin by default
    harvest_action: int = 3
    age_tolerance: float = 0.5
    
    def __call__(self, state: "StandState") -> int:
        # Check for harvest
        if state.age >= self.rotation_age - self.age_tolerance:
            return self.harvest_action
        
        # Check for thinning
        for thin_age in self.thin_ages:
            if abs(state.age - thin_age) <= self.age_tolerance:
                return self.thin_action
        
        return 0  # No-op


@dataclass
class ThresholdThinPolicy:
    """Thin when BA exceeds threshold, harvest at rotation age.
    
    Rule: If age >= min_thin_age and BA >= ba_threshold, thin.
    Respects cooldown between thins.
    """
    ba_threshold: float = 120.0
    min_thin_age: float = 10.0
    rotation_age: float = 30.0
    thin_action: int = 2  # Heavy thin
    harvest_action: int = 3
    cooldown_years: float = 5.0
    
    # Track last thin age
    _last_thin_age: float | None = field(default=None, repr=False)
    
    def __call__(self, state: "StandState") -> int:
        # Check for harvest
        if state.age >= self.rotation_age:
            return self.harvest_action
        
        # Check cooldown
        if self._last_thin_age is not None:
            if state.age - self._last_thin_age < self.cooldown_years:
                return 0
        
        # Check for thinning
        if state.age >= self.min_thin_age and state.ba >= self.ba_threshold:
            self._last_thin_age = state.age
            return self.thin_action
        
        return 0


@dataclass
class VolumeThresholdPolicy:
    """Thin when volume >= threshold at eligible ages.
    
    This implements the rule: "if volume >= X at age >= A then thin Y%"
    
    Args:
        volume_threshold: Minimum volume (cuft/ac) to trigger thin
        min_age: Minimum age to consider thinning
        max_age: Maximum age to thin (harvest after this)
        thin_action: Action index for thinning (1=light, 2=heavy)
        harvest_action: Action index for harvest
        cooldown_years: Minimum years between thins
    """
    volume_threshold: float = 2000.0  # cuft/ac
    min_age: float = 12.0
    max_age: float = 28.0
    rotation_age: float = 30.0
    thin_action: int = 2
    harvest_action: int = 3
    cooldown_years: float = 5.0
    
    _last_thin_age: float | None = field(default=None, repr=False)
    _pmrc: "PMRCModel | None" = field(default=None, repr=False)
    
    def set_pmrc(self, pmrc: "PMRCModel") -> None:
        """Set PMRC model for volume calculations."""
        self._pmrc = pmrc
    
    def __call__(self, state: "StandState") -> int:
        # Check for harvest
        if state.age >= self.rotation_age:
            self._last_thin_age = None  # Reset for next rotation
            return self.harvest_action
        
        # Need PMRC for volume calculation
        if self._pmrc is None:
            # Fall back to BA-based heuristic
            volume_est = state.ba * state.hd * 0.39  # Rough TVOB estimate
        else:
            volume_est = self._pmrc.tvob(
                state.age, state.tpa, state.hd, state.ba, region=state.region
            )
        
        # Check cooldown
        if self._last_thin_age is not None:
            if state.age - self._last_thin_age < self.cooldown_years:
                return 0
        
        # Check for thinning
        if (self.min_age <= state.age <= self.max_age 
            and volume_est >= self.volume_threshold):
            self._last_thin_age = state.age
            return self.thin_action
        
        return 0
    
    def reset(self) -> None:
        """Reset policy state for new simulation."""
        self._last_thin_age = None


def create_standard_policies() -> dict[str, Policy]:
    """Create a set of standard baseline policies for comparison."""
    return {
        "no_op": NoOpPolicy(),
        "fixed_rotation_25": FixedRotationPolicy(
            rotation_age=25.0,
            thin_ages=(12.0,),
            thin_action=1,
        ),
        "fixed_rotation_30": FixedRotationPolicy(
            rotation_age=30.0,
            thin_ages=(12.0, 20.0),
            thin_action=1,
        ),
        "ba_threshold_120": ThresholdThinPolicy(
            ba_threshold=120.0,
            min_thin_age=10.0,
            rotation_age=30.0,
        ),
        "volume_2000": VolumeThresholdPolicy(
            volume_threshold=2000.0,
            min_age=12.0,
            rotation_age=30.0,
        ),
    }