"""Stochastic PMRC wrapper with size classes.

This module provides the main StochasticPMRC class that combines:
- Deterministic PMRC growth projections (from pmrc_model.py)
- Process noise on growth increments (from process_noise.py)
- Disturbance shocks (from disturbances.py)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import numpy as np

from core.actions import ActionModel, ActionType, ThinningResult
from core.config import ThinningParams
from core.disturbances import DisturbanceModel, DisturbanceParams
from core.pmrc_model import (
    DEFAULT_DBH_BOUNDS,
    PMRCModel,
    SizeClassDistribution,
    thin_smallest_first,
)
from core.process_noise import NoiseParams, ProcessNoiseModel
from core.state import StandState

# Re-export for backward compatibility
__all__ = ["StandState", "StochasticPMRC", "TransitionTrace", "thin_to_residual_ba_smallest_first"]

Action = Literal[0, 1, 2]


@dataclass
class TransitionTrace:
    """Debug trace capturing deterministic vs stochastic components of a transition.
    
    This enables explainability of stochastic transitions for thesis documentation.
    All deltas are computed as (stochastic_value - deterministic_mean).
    """
    # Deterministic means from PMRC projection
    hd_mean: float
    tpa_mean: float
    ba_mean: float
    
    # Realized stochastic values (before disturbance)
    hd_realized: float
    tpa_realized: float
    ba_realized: float
    
    # Noise deltas (realized - mean)
    delta_hd: float
    delta_tpa: float
    delta_ba: float
    
    # Disturbance info
    disturbance_label: str | None = None
    disturbance_severity: float = 0.0
    disturbance_tpa_loss: float = 0.0
    disturbance_ba_loss: float = 0.0
    disturbance_hd_loss: float = 0.0
    
    # Recruitment
    recruitment: float = 0.0
    
    # Action info
    action_type: str = "none"
    thin_ba_removed: float = 0.0
    thin_revenue: float = 0.0


def thin_to_residual_ba_smallest_first(
    state: StandState,
    residual_ba: float,
    pmrc: PMRCModel,
    dbh_bounds: np.ndarray | None = None,
) -> tuple[StandState, SizeClassDistribution]:
    """Thin a stand to residual BA by removing smallest trees first.
    
    This is the main entry point for smallest-tree-first thinning. It:
    1. Builds a Weibull-based size class distribution using PMRC coefficients
    2. Removes trees from smallest classes until target BA is reached
    3. Returns the new stand state and the post-thin distribution
    
    Args:
        state: Current stand state
        residual_ba: Target BA after thinning (ft²/ac)
        pmrc: PMRC model instance for Weibull parameter estimation
        dbh_bounds: Optional custom DBH class boundaries
    
    Returns:
        Tuple of (new_state, post_thin_distribution)
    """
    if dbh_bounds is None:
        dbh_bounds = DEFAULT_DBH_BOUNDS
    
    if residual_ba >= state.ba:
        # No thinning needed
        dist = pmrc.diameter_class_distribution(
            ba=state.ba,
            tpa=state.tpa,
            dbh_bounds=dbh_bounds,
            region=state.region,
            phwd=state.phwd,
        )
        return state, dist
    
    # Build size class distribution using PMRC coefficients
    dist_pre = pmrc.diameter_class_distribution(
        ba=state.ba,
        tpa=state.tpa,
        dbh_bounds=dbh_bounds,
        region=state.region,
        phwd=state.phwd,
    )
    
    # Calculate BA to remove
    ba_to_remove = state.ba - residual_ba
    
    # Thin from smallest classes first
    dist_post = thin_smallest_first(dist_pre, ba_to_remove)
    
    # Create new state
    new_state = StandState(
        age=state.age,
        hd=state.hd,  # Height unchanged by thinning
        tpa=max(1.0, dist_post.total_tpa),  # Ensure at least 1 TPA
        ba=max(0.0, dist_post.total_ba),
        si25=state.si25,
        region=state.region,
        phwd=state.phwd,
    )
    
    return new_state, dist_post


class StochasticPMRC:
    """Stochastic wrapper for PMRCModel with disturbances and process noise.
    
    Combines:
    - Deterministic PMRC growth projections
    - Process noise on growth increments (BA, HD)
    - Disturbance shocks to atomic state variables
    - Recruitment to smallest diameter class
    """

    def __init__(
        self,
        pmrc: PMRCModel,
        *,
        noise_params: NoiseParams | None = None,
        disturbance_params: DisturbanceParams | None = None,
        thin_params: ThinningParams | None = None,
        recruitment_alpha: tuple[float, float, float] = (1.0, -0.005, 0.02),
    ) -> None:
        self.pmrc: PMRCModel = pmrc
        self.noise_model: ProcessNoiseModel = ProcessNoiseModel(noise_params)
        self.disturbance_model: DisturbanceModel = DisturbanceModel(disturbance_params)
        self.action_model: ActionModel = ActionModel(pmrc, thin_params=thin_params)
        self.recruitment_alpha = recruitment_alpha

    @classmethod
    def from_params(
        cls,
        pmrc: PMRCModel,
        noise: NoiseParams,
        disturbance: DisturbanceParams,
    ) -> StochasticPMRC:
        """Create StochasticPMRC from param dataclasses.
        
        Args:
            pmrc: Deterministic PMRC model
            noise: NoiseParams from core.process_noise
            disturbance: DisturbanceParams from core.disturbances
        """
        return cls(
            pmrc=pmrc,
            noise_params=noise,
            disturbance_params=disturbance,
        )

    def sample_recruitment(self, state: StandState, rng: np.random.Generator) -> float:
        """Sample new trees per acre for the smallest class."""
        return self.noise_model.sample_recruitment(
            ba=state.ba,
            si25=state.si25,
            alpha=self.recruitment_alpha,
            rng=rng,
        )

    def sample_next_state(
        self,
        state: StandState,
        dt: float,
        rng: np.random.Generator,
    ) -> tuple[StandState, TransitionTrace]:
        """Sample the next stand state after dt years.
        
        Applies in order:
        1. Deterministic PMRC projection
        2. Process noise on increments
        3. Disturbance shock (if event occurs)
        4. Feasibility projection
        
        Args:
            state: Current stand state
            dt: Time step in years
            rng: NumPy random generator
            
        Returns:
            Tuple of (new_state, transition_trace)
        """
        # 1. Deterministic PMRC projection
        age2 = state.age + dt
        hd_det = self.pmrc.hd_project(state.age, state.hd, age2)
        tpa_det = self.pmrc.tpa_project(state.tpa, state.si25, state.age, age2)
        ba_det = self.pmrc.ba_project(
            state.age, state.tpa, tpa_det, state.ba, state.hd, hd_det, age2, state.region
        )
        
        # Compute deterministic increments
        delta_hd = hd_det - state.hd
        delta_ba = ba_det - state.ba
        expected_tpa_loss = state.tpa - tpa_det
        
        # 2. Apply process noise to increments
        noisy_delta_ba, noisy_delta_hd, tpa_adj, _ = self.noise_model.apply_to_increments(
            delta_ba=delta_ba,
            delta_hd=delta_hd,
            tpa=state.tpa,
            expected_tpa_loss=expected_tpa_loss,
            rng=rng,
        )
        
        # Compute realized values (before disturbance)
        hd_realized = state.hd + noisy_delta_hd
        ba_realized = state.ba + noisy_delta_ba
        tpa_realized = tpa_det + tpa_adj  # tpa_adj is deviation from expected
        
        # Add recruitment
        recruitment = self.sample_recruitment(state, rng)
        tpa_realized += recruitment
        
        # Create pre-disturbance state
        pre_dist_state = StandState(
            age=age2,
            hd=hd_realized,
            tpa=tpa_realized,
            ba=ba_realized,
            si25=state.si25,
            region=state.region,
            phwd=state.phwd,
        )
        
        # 3. Apply disturbance shock
        post_dist_state, dist_event = self.disturbance_model.sample_and_apply(
            pre_dist_state, rng
        )
        
        # 4. Feasibility projection
        final_state = self._project_feasible(post_dist_state, state.hd)
        
        # 5. Check for management actions (thinning)
        action_type = ActionType.NONE
        thin_result: ThinningResult | None = None
        if self.action_model.should_thin(final_state):
            thin_result = self.action_model.apply_thinning(final_state)
            if thin_result.occurred and thin_result.post_thin_state is not None:
                final_state = thin_result.post_thin_state
                action_type = ActionType.THIN
        
        # Build trace
        trace = TransitionTrace(
            hd_mean=hd_det,
            tpa_mean=tpa_det,
            ba_mean=ba_det,
            hd_realized=hd_realized,
            tpa_realized=tpa_realized,
            ba_realized=ba_realized,
            delta_hd=noisy_delta_hd - delta_hd,
            delta_tpa=tpa_adj,
            delta_ba=noisy_delta_ba - delta_ba,
            disturbance_label="disturbance" if dist_event.occurred else None,
            disturbance_severity=dist_event.severity,
            disturbance_tpa_loss=dist_event.tpa_loss,
            disturbance_ba_loss=dist_event.ba_loss,
            disturbance_hd_loss=dist_event.hd_loss,
            recruitment=recruitment,
            action_type=action_type.value,
            thin_ba_removed=thin_result.ba_removed if thin_result else 0.0,
            thin_revenue=thin_result.net_revenue if thin_result else 0.0,
        )
        
        return final_state, trace

    def _project_feasible(self, state: StandState, prev_hd: float) -> StandState:
        """Project state into feasible region.
        
        Enforces:
        - TPA >= 100 (PMRC lower bound)
        - BA >= 0
        - HD >= prev_hd (height non-decreasing)
        - age > 0
        """
        return StandState(
            age=max(0.1, state.age),
            hd=max(prev_hd, state.hd),  # Height can't decrease
            tpa=max(100.0, state.tpa),  # PMRC lower bound
            ba=max(0.0, state.ba),
            si25=state.si25,
            region=state.region,
            phwd=state.phwd,
        )

    def _hd_from_site(self, si25: float, age: float) -> float:
        """Compute dominant height at a given age from site index."""
        age = max(0.5, age)
        num = 1.0 - np.exp(-self.pmrc.k * age)
        den = 1.0 - np.exp(-self.pmrc.k * 25.0)
        return max(1.0, si25 * (num / den) ** self.pmrc.m)



if __name__ == "__main__":
    pmrc = PMRCModel(region="ucp")
    # Use default BAT thinning policy (age 15, BA > 150, residual 100)
    stochastic = StochasticPMRC(pmrc, thin_params=ThinningParams())
    
    # Default initial conditions: age 5, TPA 850, SI25 80
    # HD derived from SI25, BA predicted from PMRC model
    from core.state import hd_from_si25_at_age
    age0, si25, tpa0 = 5.0, 80.0, 850.0
    initial_hd = hd_from_si25_at_age(si25, age0)
    initial_ba = pmrc.ba_predict(age=age0, tpa=tpa0, hd=initial_hd, region="ucp")
    state = StandState.from_si25(age=age0, si25=si25, tpa=tpa0, ba=initial_ba, region="ucp")
    rng = np.random.default_rng(42)
    total_thin_revenue = 0.0
    
    for i in range(35):
        state, trace = stochastic.sample_next_state(state, dt=1.0, rng=rng)
        print(f"Year {i+1}: age={state.age:.1f}, hd={state.hd:.1f}, tpa={state.tpa:.0f}, ba={state.ba:.1f}")
        if trace.disturbance_label:
            print(f"  -> Disturbance: {trace.disturbance_label}")
        if trace.action_type == "thin":
            print(f"  -> THINNED: removed {trace.thin_ba_removed:.1f} BA, revenue ${trace.thin_revenue:.2f}/ac")
            total_thin_revenue += trace.thin_revenue
    
    print(f"\nTotal thinning revenue: ${total_thin_revenue:.2f}/ac")
