"""Tests for site index consistency in PMRC model.

Verifies that:
1. HD projection follows the SI25 curve when initial HD is consistent
2. Detects inconsistency between stated SI25 and initial HD
3. Provides helper to compute consistent initial HD from SI25
"""

import pytest
from math import exp

from core.pmrc_model import PMRCModel
from core.state import StandState


def hd_from_si25_at_age(pmrc: PMRCModel, si25: float, age: float) -> float:
    """Compute expected HD at a given age for a site index.
    
    HD(age) = SI25 * ((1 - exp(-k*age)) / (1 - exp(-k*25)))^m
    
    This is the inverse of the site index definition.
    """
    k, m = pmrc.k, pmrc.m
    ratio = (1 - exp(-k * age)) / (1 - exp(-k * 25))
    return si25 * (ratio ** m)


def si25_from_hd_at_age(pmrc: PMRCModel, hd: float, age: float) -> float:
    """Compute implied SI25 from HD at a given age.
    
    SI25 = HD / ((1 - exp(-k*age)) / (1 - exp(-k*25)))^m
    """
    k, m = pmrc.k, pmrc.m
    ratio = (1 - exp(-k * age)) / (1 - exp(-k * 25))
    return hd / (ratio ** m)


def check_si_hd_consistency(state: StandState, pmrc: PMRCModel, tolerance: float = 0.1) -> tuple[bool, float, float]:
    """Check if state's HD is consistent with its SI25.
    
    Args:
        state: Stand state to check
        pmrc: PMRC model instance
        tolerance: Relative tolerance for consistency check
        
    Returns:
        Tuple of (is_consistent, expected_hd, implied_si25)
    """
    expected_hd = hd_from_si25_at_age(pmrc, state.si25, state.age)
    implied_si25 = si25_from_hd_at_age(pmrc, state.hd, state.age)
    
    relative_error = abs(state.hd - expected_hd) / expected_hd
    is_consistent = relative_error <= tolerance
    
    return is_consistent, expected_hd, implied_si25


class TestSiteIndexConsistency:
    """Tests for site index and height projection consistency."""
    
    def test_hd_projection_follows_si_curve(self):
        """HD projection should follow the SI25 curve when starting consistent."""
        pmrc = PMRCModel(region="ucp")
        si25 = 60.0
        
        # Start with consistent HD at age 5
        age_start = 5.0
        hd_start = hd_from_si25_at_age(pmrc, si25, age_start)
        
        # Project to age 25 - should equal SI25
        hd_at_25 = pmrc.hd_project(age_start, hd_start, 25.0)
        
        assert abs(hd_at_25 - si25) < 0.01, \
            f"HD at age 25 should equal SI25={si25}, got {hd_at_25:.2f}"
    
    def test_hd_projection_intermediate_ages(self):
        """HD projection should match expected values at all ages."""
        pmrc = PMRCModel(region="ucp")
        si25 = 60.0
        
        # Start with consistent HD at age 5
        age_start = 5.0
        hd_start = hd_from_si25_at_age(pmrc, si25, age_start)
        
        # Check intermediate ages
        for target_age in [10, 15, 20, 25, 30, 35]:
            hd_projected = pmrc.hd_project(age_start, hd_start, float(target_age))
            hd_expected = hd_from_si25_at_age(pmrc, si25, float(target_age))
            
            assert abs(hd_projected - hd_expected) < 0.01, \
                f"At age {target_age}: projected {hd_projected:.2f} != expected {hd_expected:.2f}"
    
    def test_inconsistent_initial_state_detected(self):
        """Should detect when initial HD doesn't match SI25."""
        pmrc = PMRCModel(region="ucp")
        
        # Create inconsistent state: SI25=60 but HD=40 at age 5
        # HD=40 at age 5 implies SI25 ≈ 134, not 60
        state = StandState(
            age=5.0,
            hd=40.0,  # Wrong! Should be ~17.93 for SI25=60
            tpa=500.0,
            ba=80.0,
            si25=60.0,
            region="ucp",
        )
        
        is_consistent, expected_hd, implied_si25 = check_si_hd_consistency(state, pmrc)
        
        assert not is_consistent, "Should detect inconsistency"
        assert abs(expected_hd - 17.93) < 0.1, f"Expected HD should be ~17.93, got {expected_hd:.2f}"
        assert abs(implied_si25 - 133.85) < 1.0, f"Implied SI25 should be ~134, got {implied_si25:.2f}"
    
    def test_consistent_initial_state_passes(self):
        """Should pass when initial HD matches SI25."""
        pmrc = PMRCModel(region="ucp")
        
        si25 = 60.0
        age = 5.0
        correct_hd = hd_from_si25_at_age(pmrc, si25, age)
        
        state = StandState(
            age=age,
            hd=correct_hd,
            tpa=500.0,
            ba=80.0,
            si25=si25,
            region="ucp",
        )
        
        is_consistent, expected_hd, implied_si25 = check_si_hd_consistency(state, pmrc)
        
        assert is_consistent, f"Should be consistent: HD={correct_hd:.2f}, expected={expected_hd:.2f}"
        assert abs(implied_si25 - si25) < 0.01, f"Implied SI25 should equal stated SI25"
    
    def test_si25_values(self):
        """Verify SI25 curve values for common site indices."""
        pmrc = PMRCModel(region="ucp")
        
        # Test multiple site indices
        for si25 in [50, 60, 70, 80]:
            # HD at age 25 should equal SI25 by definition
            hd_at_25 = hd_from_si25_at_age(pmrc, float(si25), 25.0)
            assert abs(hd_at_25 - si25) < 0.01, \
                f"HD at age 25 should equal SI25={si25}, got {hd_at_25:.2f}"
            
            # Verify projection from age 5 to 25
            hd_at_5 = hd_from_si25_at_age(pmrc, float(si25), 5.0)
            hd_projected = pmrc.hd_project(5.0, hd_at_5, 25.0)
            assert abs(hd_projected - si25) < 0.01, \
                f"Projected HD at 25 should equal SI25={si25}, got {hd_projected:.2f}"


def create_consistent_initial_state(
    pmrc: PMRCModel,
    si25: float,
    age: float,
    tpa: float,
    region: str = "ucp",
    phwd: float = 0.0,
) -> StandState:
    """Create an initial state with HD consistent with SI25.
    
    This ensures the HD projection will follow the expected SI curve.
    BA is computed from the PMRC prediction equation.
    
    Args:
        pmrc: PMRC model instance
        si25: Site index at base age 25
        age: Initial stand age
        tpa: Initial trees per acre
        region: Geographic region
        phwd: Percent hardwood
        
    Returns:
        StandState with consistent HD and predicted BA
    """
    hd = hd_from_si25_at_age(pmrc, si25, age)
    ba = pmrc.ba_predict(age, tpa, hd, region)
    
    return StandState(
        age=age,
        hd=hd,
        tpa=tpa,
        ba=ba,
        si25=si25,
        region=region,
        phwd=phwd,
    )


if __name__ == "__main__":
    # Run quick verification
    pmrc = PMRCModel(region="ucp")
    
    print("Site Index Consistency Test")
    print("=" * 50)
    
    # Show expected HD values for SI25=60
    si25 = 60.0
    print(f"\nExpected HD at each age for SI25={si25}:")
    for age in [5, 10, 15, 20, 25, 30, 35]:
        hd = hd_from_si25_at_age(pmrc, si25, float(age))
        print(f"  Age {age}: HD = {hd:.2f} ft")
    
    # Check the current default initial state
    print("\n" + "=" * 50)
    print("Checking default initial state (age=5, hd=40, si25=60):")
    
    state = StandState(age=5.0, hd=40.0, tpa=500.0, ba=80.0, si25=60.0, region="ucp")
    is_consistent, expected_hd, implied_si25 = check_si_hd_consistency(state, pmrc)
    
    print(f"  Stated SI25: {state.si25}")
    print(f"  Current HD:  {state.hd:.2f} ft")
    print(f"  Expected HD: {expected_hd:.2f} ft (for SI25={state.si25})")
    print(f"  Implied SI25: {implied_si25:.2f} (from current HD)")
    print(f"  Consistent: {is_consistent}")
    
    if not is_consistent:
        print(f"\n  WARNING: Initial HD does not match SI25!")
        print(f"  The projection will follow SI25={implied_si25:.0f}, not SI25={state.si25}")
    
    # Show correct initial state
    print("\n" + "=" * 50)
    print("Creating consistent initial state:")
    correct_state = create_consistent_initial_state(pmrc, si25=60.0, age=5.0, tpa=500.0)
    print(f"  Age:  {correct_state.age}")
    print(f"  HD:   {correct_state.hd:.2f} ft")
    print(f"  TPA:  {correct_state.tpa}")
    print(f"  BA:   {correct_state.ba:.2f} ft²/ac")
    print(f"  SI25: {correct_state.si25}")
