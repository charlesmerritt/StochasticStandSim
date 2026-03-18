"""Management actions for forest stand simulation.

This module implements silvicultural actions that can be applied to stands:
- Thinning (from below, using BA threshold rules)
- Final harvest with product breakdown

Per PLANNING.md, actions are applied to atomic state variables.
Product yields and values are computed from post-action state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from core.pmrc_model import (
    DEFAULT_DBH_BOUNDS,
    PMRCModel,
    SizeClassDistribution,
    thin_smallest_first,
)
from core.products import (
    CUFT_TO_TON,
    HarvestCosts,
    ProductDistribution,
    ProductPrices,
    estimate_product_distribution,
)
from core.state import StandState

if TYPE_CHECKING:
    from core.config import ThinningParams


class ActionType(Enum):
    """Types of management actions."""
    NONE = "none"
    THIN = "thin"
    HARVEST = "harvest"


@dataclass
class HarvestParams:
    """Parameters for final harvest.
    
    Attributes:
        min_harvest_age: Minimum age for harvest (years)
        prices: Stumpage prices by product class
        costs: Harvest and regeneration costs
    """
    min_harvest_age: float = 25.0
    prices: ProductPrices | None = None
    costs: HarvestCosts | None = None


@dataclass
class ThinningResult:
    """Result of a thinning operation.
    
    Attributes:
        occurred: Whether thinning was applied
        ba_removed: BA removed (ft²/ac)
        tpa_removed: TPA removed
        products_removed: Product distribution of removed trees
        net_revenue: Net revenue from thinning ($/ac)
        post_thin_state: Stand state after thinning
    """
    occurred: bool
    ba_removed: float = 0.0
    tpa_removed: float = 0.0
    products_removed: ProductDistribution | None = None
    net_revenue: float = 0.0
    post_thin_state: StandState | None = None


@dataclass
class HarvestResult:
    """Result of a harvest operation.
    
    Attributes:
        products: Product distribution at harvest
        gross_revenue: Gross revenue ($/ac)
        net_revenue: Net revenue after costs ($/ac)
        vol_pulp: Volume of pulpwood (cuft/ac)
        vol_cns: Volume of chip-n-saw (cuft/ac)
        vol_saw: Volume of sawtimber (cuft/ac)
    """
    products: ProductDistribution
    gross_revenue: float
    net_revenue: float
    vol_pulp: float
    vol_cns: float
    vol_saw: float


class ActionModel:
    """Applies management actions to stand state.
    
    Implements:
    - Mid-rotation BA threshold thinning (thinning from below)
    - Final harvest with product breakdown and valuation
    """

    def __init__(
        self,
        pmrc: PMRCModel,
        thin_params: ThinningParams | None = None,
        harvest_params: HarvestParams | None = None,
    ) -> None:
        self.pmrc: PMRCModel = pmrc
        self.thin_params: ThinningParams | None = thin_params  # None = no thinning
        self.harvest_params: HarvestParams = harvest_params or HarvestParams()
        self._thinned: bool = False  # Track if thinning has occurred this rotation

    def reset_rotation(self) -> None:
        """Reset rotation state (call at start of new rotation)."""
        self._thinned = False

    def should_thin(self, state: StandState) -> bool:
        """Check if thinning should occur based on BA threshold rule.
        
        Thinning occurs once per rotation at the trigger age if BA exceeds threshold.
        Returns False if thin_params is None (thinning disabled).
        
        Args:
            state: Current stand state
            
        Returns:
            True if thinning should be applied
        """
        # No thinning if params not set
        if self.thin_params is None:
            return False
        
        if self._thinned:
            return False
        
        # Check if we're at the trigger age (within 0.5 year tolerance)
        at_trigger_age = abs(state.age - self.thin_params.trigger_age) < 0.5
        
        # Check if BA exceeds threshold
        ba_exceeds = state.ba >= self.thin_params.ba_threshold
        
        return at_trigger_age and ba_exceeds

    def apply_thinning(
        self,
        state: StandState,
        prices: ProductPrices | None = None,
    ) -> ThinningResult:
        """Apply thinning from below to reach residual BA.
        
        Uses PMRC Weibull distribution to determine which trees are removed,
        starting from smallest diameter classes.
        
        Args:
            state: Current stand state
            prices: Stumpage prices (uses defaults if None)
            
        Returns:
            ThinningResult with removed products and new state
        """
        if not self.should_thin(state):
            return ThinningResult(occurred=False)
        
        # Type guard: thin_params is guaranteed non-None after should_thin() returns True
        assert self.thin_params is not None
        
        if prices is None:
            prices = ProductPrices()
        
        # Calculate BA to remove (thin to residual)
        ba_to_remove = state.ba - self.thin_params.residual_ba
        if ba_to_remove <= 0:
            return ThinningResult(occurred=False)
        
        # Get current size class distribution
        dist_pre = self.pmrc.diameter_class_distribution(
            ba=state.ba,
            tpa=state.tpa,
            dbh_bounds=DEFAULT_DBH_BOUNDS,
            region=state.region,
            phwd=state.phwd,
        )
        
        # Apply thinning from below
        dist_post = thin_smallest_first(dist_pre, ba_to_remove)
        
        # Calculate removed trees
        tpa_removed = dist_pre.total_tpa - dist_post.total_tpa
        ba_removed = dist_pre.total_ba - dist_post.total_ba
        
        # Estimate products in removed trees
        # Use PMRC merchantability for removed volume and the Weibull class
        # difference only for optional TPA/BA-by-class summaries.
        products_removed = self._estimate_removed_products(
            state, dist_pre, dist_post
        )
        
        # Calculate revenue from removed trees
        net_revenue = self._compute_thin_revenue(
            products_removed, prices, self.thin_params.thin_cost
        )
        
        # Create post-thin state
        post_thin_state = StandState(
            age=state.age,
            hd=state.hd,  # Height unchanged by thinning
            tpa=max(100.0, dist_post.total_tpa),  # PMRC lower bound
            ba=max(0.0, dist_post.total_ba),
            si25=state.si25,
            region=state.region,
            phwd=state.phwd,
        )
        
        # Mark thinning as occurred for this rotation
        self._thinned = True
        
        return ThinningResult(
            occurred=True,
            ba_removed=ba_removed,
            tpa_removed=tpa_removed,
            products_removed=products_removed,
            net_revenue=net_revenue,
            post_thin_state=post_thin_state,
        )

    def _estimate_removed_products(
        self,
        state: StandState,
        dist_pre: SizeClassDistribution,
        dist_post: SizeClassDistribution,
    ) -> ProductDistribution:
        """Estimate product distribution of removed trees."""
        # DBH bounds: [0, 6, 9, 12, 24] -> [submerch, pulp, cns, saw]
        # Indices: 0=submerch, 1=pulp, 2=cns, 3=saw
        
        tpa_removed = dist_pre.tpa_per_class - dist_post.tpa_per_class
        ba_removed = dist_pre.ba_per_class - dist_post.ba_per_class
        
        # Ensure non-negative
        tpa_removed = np.maximum(0.0, tpa_removed)
        ba_removed = np.maximum(0.0, ba_removed)
        
        # Extract merchantable classes (skip submerch at index 0)
        tpa_pulp = tpa_removed[1] if len(tpa_removed) > 1 else 0.0
        tpa_cns = tpa_removed[2] if len(tpa_removed) > 2 else 0.0
        tpa_saw = tpa_removed[3] if len(tpa_removed) > 3 else 0.0
        
        ba_pulp = ba_removed[1] if len(ba_removed) > 1 else 0.0
        ba_cns = ba_removed[2] if len(ba_removed) > 2 else 0.0
        ba_saw = ba_removed[3] if len(ba_removed) > 3 else 0.0

        tpa_total_removed = float(np.sum(tpa_removed))
        ba_total_removed = float(np.sum(ba_removed))
        if tpa_total_removed > 0.0 and ba_total_removed > 0.0:
            removed_yields = self.pmrc.product_yields(
                age=state.age,
                tpa=tpa_total_removed,
                hd=state.hd,
                ba=ba_total_removed,
                unit="TVOB",
                region=state.region,
            )
            vol_pulp = float(removed_yields.pulpwood)
            vol_cns = float(removed_yields.chip_n_saw)
            vol_saw = float(removed_yields.sawtimber)
        else:
            vol_pulp = vol_cns = vol_saw = 0.0
        
        return ProductDistribution(
            tpa_pulp=float(tpa_pulp),
            tpa_cns=float(tpa_cns),
            tpa_saw=float(tpa_saw),
            ba_pulp=float(ba_pulp),
            ba_cns=float(ba_cns),
            ba_saw=float(ba_saw),
            vol_pulp=float(vol_pulp),
            vol_cns=float(vol_cns),
            vol_saw=float(vol_saw),
        )

    def _compute_thin_revenue(
        self,
        products: ProductDistribution,
        prices: ProductPrices,
        thin_cost: float,
    ) -> float:
        """Compute net revenue from thinning."""
        tons_pulp = products.vol_pulp * CUFT_TO_TON
        tons_cns = products.vol_cns * CUFT_TO_TON
        tons_saw = products.vol_saw * CUFT_TO_TON
        
        gross = (
            tons_pulp * prices.pulpwood +
            tons_cns * prices.chip_n_saw +
            tons_saw * prices.sawtimber
        )
        
        return gross - thin_cost

    def evaluate_harvest(
        self,
        state: StandState,
        prices: ProductPrices | None = None,
        costs: HarvestCosts | None = None,
    ) -> HarvestResult:
        """Evaluate harvest value at current state.
        
        Does not modify state - just computes what harvest would yield.
        
        Args:
            state: Current stand state
            prices: Stumpage prices
            costs: Harvest and regeneration costs
            
        Returns:
            HarvestResult with product breakdown and values
        """
        if prices is None:
            prices = self.harvest_params.prices or ProductPrices()
        if costs is None:
            costs = self.harvest_params.costs or HarvestCosts()
        
        # Get product distribution
        products = estimate_product_distribution(
            pmrc=self.pmrc,
            age=state.age,
            ba=state.ba,
            tpa=state.tpa,
            hd=state.hd,
            region=state.region,
            phwd=state.phwd,
        )
        
        # Compute values
        tons_pulp = products.vol_pulp * CUFT_TO_TON
        tons_cns = products.vol_cns * CUFT_TO_TON
        tons_saw = products.vol_saw * CUFT_TO_TON
        
        gross_revenue = (
            tons_pulp * prices.pulpwood +
            tons_cns * prices.chip_n_saw +
            tons_saw * prices.sawtimber
        )
        
        net_revenue = gross_revenue - costs.total
        
        return HarvestResult(
            products=products,
            gross_revenue=gross_revenue,
            net_revenue=net_revenue,
            vol_pulp=products.vol_pulp,
            vol_cns=products.vol_cns,
            vol_saw=products.vol_saw,
        )

    def check_and_apply_action(
        self,
        state: StandState,
        prices: ProductPrices | None = None,
    ) -> tuple[StandState, ActionType, ThinningResult | None]:
        """Check if any action should be applied and apply it.
        
        This is the main entry point for action evaluation during simulation.
        Currently only implements thinning; harvest is evaluated separately
        at rotation end.
        
        Args:
            state: Current stand state
            prices: Stumpage prices
            
        Returns:
            Tuple of (new_state, action_type, result)
        """
        # Check for thinning
        if self.should_thin(state):
            result = self.apply_thinning(state, prices)
            if result.occurred and result.post_thin_state is not None:
                return result.post_thin_state, ActionType.THIN, result
        
        return state, ActionType.NONE, None
