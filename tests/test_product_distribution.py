"""Validation test for product distribution over stand age.

Prints the ratio of pulpwood, chip-n-saw, and sawtimber volume
for each year of a deterministic rotation.
"""

import math

from core.pmrc_model import PMRCModel
from core.products import CUFT_TO_TON, estimate_product_distribution
from core.state import StandState, hd_from_si25_at_age


# Conversion constants
CUFT_TO_DRY_TON = 0.025  # Approximate dry weight conversion
GREEN_WEIGHT_FACTOR = 1.0  # Green weight for stumpage pricing


def compute_dq(ba: float, tpa: float) -> float:
    """Compute quadratic mean diameter (inches) from BA and TPA.
    
    Dq = sqrt(BA / TPA * 144 / pi) = sqrt(BA / TPA) * 6.8
    """
    if tpa <= 0:
        return 0.0
    return math.sqrt(ba / tpa) * 6.8  # Simplified: sqrt(144/pi) ≈ 6.77


def test_product_distribution_by_year():
    """Print product distribution ratios for each year of a rotation."""
    pmrc = PMRCModel(region="ucp")
    
    # Initial conditions
    age0 = 5.0
    tpa0 = 850.0
    si25 = 80.0
    rotation_length = 35
    
    # Initialize state
    hd0 = hd_from_si25_at_age(si25, age0)
    ba0 = pmrc.ba_predict(age=age0, tpa=tpa0, hd=hd0, region="ucp")
    
    state = StandState(
        age=age0,
        hd=hd0,
        tpa=tpa0,
        ba=ba0,
        si25=si25,
        region="ucp",
    )
    
    print("\n" + "=" * 160)
    print("Standing Inventory by Year (Deterministic) - NOT extracted volume")
    print("=" * 160)
    
    # Header row 1
    print(f"{'Age':>4}  {'Dens.':>6}  {'Height':>6}  {'BArea':>6}  {'Vol ob':>8}  "
          f"{'DryWt':>7}  {'Dq':>5}  "
          f"{'Pulp':>8}  {'CNS':>8}  {'Saw':>8}  "
          f"{'GW Pulp':>8}  {'GW CNS':>8}  {'GW Saw':>8}")
    
    # Header row 2 (units)
    print(f"{'(yr)':>4}  {'(tpa)':>6}  {'(ft)':>6}  {'(ft2)':>6}  {'(ft3)':>8}  "
          f"{'(ton)':>7}  {'(in)':>5}  "
          f"{'(ft3)':>8}  {'(ft3)':>8}  {'(ft3)':>8}  "
          f"{'(ton)':>8}  {'(ton)':>8}  {'(ton)':>8}")
    
    print("-" * 160)
    
    for year in range(rotation_length + 1):
        if year > 0:
            # Project state forward one year
            age2 = state.age + 1.0
            hd2 = pmrc.hd_project(state.age, state.hd, age2)
            tpa2 = pmrc.tpa_project(state.tpa, state.si25, state.age, age2)
            ba2 = pmrc.ba_project(
                state.age, state.tpa, tpa2, state.ba, state.hd, hd2, age2, state.region
            )
            state = StandState(
                age=age2,
                hd=hd2,
                tpa=tpa2,
                ba=ba2,
                si25=si25,
                region="ucp",
            )
        
        # Get product distribution
        dist = estimate_product_distribution(
            pmrc=pmrc,
            age=state.age,
            ba=state.ba,
            tpa=state.tpa,
            hd=state.hd,
            region=state.region,
        )
        
        # Compute derived values
        vol_ob = dist.total_vol
        dry_weight = vol_ob * CUFT_TO_DRY_TON
        dq = compute_dq(state.ba, state.tpa)
        
        # Green weight by product (ton/acre) - standing inventory
        gw_pulp = dist.vol_pulp * CUFT_TO_TON
        gw_cns = dist.vol_cns * CUFT_TO_TON
        gw_saw = dist.vol_saw * CUFT_TO_TON
        
        print(f"{state.age:>4.0f}  {state.tpa:>6.0f}  {state.hd:>6.1f}  {state.ba:>6.1f}  "
              f"{vol_ob:>8.1f}  {dry_weight:>7.2f}  {dq:>5.2f}  "
              f"{dist.vol_pulp:>8.1f}  {dist.vol_cns:>8.1f}  {dist.vol_saw:>8.1f}  "
              f"{gw_pulp:>8.2f}  {gw_cns:>8.2f}  {gw_saw:>8.2f}")
    
    print("=" * 160)
    print("\nNote: All values are STANDING INVENTORY at each age, not extracted/harvested volume.")
    print("      Extraction only occurs at thinning (if applicable) and final harvest.")
    print("\nDBH thresholds: Pulpwood 6-9\", Chip-n-Saw 9-12\", Sawtimber 12\"+")
    print(f"Conversion: CUFT_TO_TON = {CUFT_TO_TON}")
    print("=" * 160)


if __name__ == "__main__":
    test_product_distribution_by_year()
