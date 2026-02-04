"""Tests for thinning operations including smallest-tree-first."""

import numpy as np
import pytest

from core.pmrc_model import PMRCModel
from core.stochastic_stand import (
    StandState,
    SizeClassDistribution,
    size_class_distribution_from_state,
    thin_smallest_first,
    thin_to_residual_ba_smallest_first,
    apply_thin_to_state,
    DEFAULT_DBH_BOUNDS,
)


@pytest.fixture
def pmrc():
    return PMRCModel(region="ucp")


@pytest.fixture
def mature_stand():
    """A mature stand with realistic values."""
    return StandState(
        age=20,
        hd=50.0,
        tpa=400,
        ba=120.0,
        si25=70.0,
        region="ucp",
    )


class TestSizeClassDistribution:
    """Tests for Weibull-based size class distribution."""

    def test_distribution_sums_match_stand(self, mature_stand):
        """Distribution TPA and BA should sum to stand values."""
        dist = size_class_distribution_from_state(mature_stand, DEFAULT_DBH_BOUNDS)
        
        assert np.isclose(np.sum(dist.tpa_per_class), mature_stand.tpa, rtol=0.01)
        assert np.isclose(np.sum(dist.ba_per_class), mature_stand.ba, rtol=0.01)

    def test_distribution_has_correct_shape(self, mature_stand):
        """Distribution arrays should match DBH bounds."""
        dist = size_class_distribution_from_state(mature_stand, DEFAULT_DBH_BOUNDS)
        
        n_classes = len(DEFAULT_DBH_BOUNDS) - 1
        assert len(dist.tpa_per_class) == n_classes
        assert len(dist.ba_per_class) == n_classes

    def test_distribution_values_nonnegative(self, mature_stand):
        """All distribution values should be non-negative."""
        dist = size_class_distribution_from_state(mature_stand, DEFAULT_DBH_BOUNDS)
        
        assert np.all(dist.tpa_per_class >= 0)
        assert np.all(dist.ba_per_class >= 0)


class TestSmallestFirstThinning:
    """Tests for smallest-tree-first thinning logic."""

    def test_thin_achieves_target_ba(self, mature_stand):
        """Thinning should achieve the target residual BA."""
        residual_ba = 72.0  # 60% retention
        new_state, dist = thin_to_residual_ba_smallest_first(mature_stand, residual_ba)
        
        assert np.isclose(new_state.ba, residual_ba, rtol=0.01)

    def test_thin_removes_smallest_first(self, mature_stand):
        """Smallest classes should be removed before larger ones."""
        dist_pre = size_class_distribution_from_state(mature_stand, DEFAULT_DBH_BOUNDS)
        
        # Remove 40% of BA
        ba_to_remove = mature_stand.ba * 0.4
        dist_post = thin_smallest_first(dist_pre, ba_to_remove)
        
        # Smallest class should have fewer or no trees
        assert dist_post.tpa_per_class[0] <= dist_pre.tpa_per_class[0]
        
        # If we removed enough, smallest class should be empty
        if ba_to_remove >= dist_pre.ba_per_class[0]:
            assert dist_post.tpa_per_class[0] == 0

    def test_thin_increases_qmd(self, mature_stand, pmrc):
        """Smallest-first thinning should increase QMD."""
        qmd_pre = pmrc.qmd(mature_stand.tpa, mature_stand.ba)
        
        residual_ba = mature_stand.ba * 0.6
        new_state, _ = thin_to_residual_ba_smallest_first(mature_stand, residual_ba)
        
        qmd_post = pmrc.qmd(new_state.tpa, new_state.ba)
        
        # QMD should increase when removing small trees
        assert qmd_post > qmd_pre

    def test_no_thin_if_residual_exceeds_current(self, mature_stand):
        """No thinning should occur if residual BA >= current BA."""
        residual_ba = mature_stand.ba + 10.0
        new_state, _ = thin_to_residual_ba_smallest_first(mature_stand, residual_ba)
        
        assert new_state.ba == mature_stand.ba
        assert new_state.tpa == mature_stand.tpa

    def test_thin_preserves_height(self, mature_stand):
        """Thinning should not change dominant height."""
        residual_ba = 72.0
        new_state, _ = thin_to_residual_ba_smallest_first(mature_stand, residual_ba)
        
        assert new_state.hd == mature_stand.hd


class TestApplyThinToState:
    """Tests for the unified apply_thin_to_state function."""

    def test_constant_qmd_mode_preserves_qmd(self, mature_stand, pmrc):
        """Constant QMD mode should preserve QMD."""
        qmd_pre = pmrc.qmd(mature_stand.tpa, mature_stand.ba)
        
        residual_ba = mature_stand.ba * 0.6
        new_state = apply_thin_to_state(mature_stand, residual_ba, pmrc, mode="constant_qmd")
        
        qmd_post = pmrc.qmd(new_state.tpa, new_state.ba)
        
        assert np.isclose(qmd_pre, qmd_post, rtol=0.01)

    def test_smallest_first_mode_increases_qmd(self, mature_stand, pmrc):
        """Smallest-first mode should increase QMD."""
        qmd_pre = pmrc.qmd(mature_stand.tpa, mature_stand.ba)
        
        residual_ba = mature_stand.ba * 0.6
        new_state = apply_thin_to_state(mature_stand, residual_ba, pmrc, mode="smallest_first")
        
        qmd_post = pmrc.qmd(new_state.tpa, new_state.ba)
        
        assert qmd_post > qmd_pre

    def test_both_modes_achieve_target_ba(self, mature_stand, pmrc):
        """Both modes should achieve the target residual BA."""
        residual_ba = 72.0
        
        s_qmd = apply_thin_to_state(mature_stand, residual_ba, pmrc, mode="constant_qmd")
        s_small = apply_thin_to_state(mature_stand, residual_ba, pmrc, mode="smallest_first")
        
        assert np.isclose(s_qmd.ba, residual_ba, rtol=0.01)
        assert np.isclose(s_small.ba, residual_ba, rtol=0.01)

    def test_smallest_first_removes_more_trees(self, mature_stand, pmrc):
        """Smallest-first should remove more trees than constant QMD."""
        residual_ba = mature_stand.ba * 0.6
        
        s_qmd = apply_thin_to_state(mature_stand, residual_ba, pmrc, mode="constant_qmd")
        s_small = apply_thin_to_state(mature_stand, residual_ba, pmrc, mode="smallest_first")
        
        # Smallest-first removes small trees, so fewer trees remain
        assert s_small.tpa < s_qmd.tpa
