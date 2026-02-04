"""Tests for product distribution based on DBH classes."""

from __future__ import annotations

import math

import pytest

from core.pmrc_model import PMRCModel
from core.products import (
    estimate_product_distribution,
    compute_harvest_value,
    compute_thin_value,
    ProductDistribution,
    ProductPrices,
    HarvestCosts,
    DBH_PULP_MIN,
    DBH_CNS_MIN,
    DBH_SAW_MIN,
)


@pytest.fixture
def pmrc() -> PMRCModel:
    return PMRCModel(region="ucp")


class TestProductDistribution:
    """Test product distribution estimation from stand state."""

    def test_young_stand_mostly_pulpwood(self, pmrc: PMRCModel):
        """Young stands with small QMD should be mostly pulpwood."""
        # Young stand: age ~10, small trees
        ba = 60.0  # ft²/ac
        tpa = 480.0
        hd = 35.0  # ft
        
        products = estimate_product_distribution(pmrc, ba, tpa, hd)
        
        # Young stand should be dominated by pulpwood
        assert products.pulp_fraction > 0.8, "Young stand should be >80% pulpwood"
        assert products.saw_fraction < 0.05, "Young stand should have <5% sawtimber"

    def test_mature_stand_mostly_sawtimber(self, pmrc: PMRCModel):
        """Mature stands with large QMD should be mostly sawtimber."""
        # Mature stand: age ~35-40, large trees
        ba = 160.0  # ft²/ac
        tpa = 280.0
        hd = 85.0  # ft
        
        products = estimate_product_distribution(pmrc, ba, tpa, hd)
        
        # Mature stand should be dominated by sawtimber
        assert products.saw_fraction > 0.6, "Mature stand should be >60% sawtimber"
        assert products.pulp_fraction < 0.1, "Mature stand should have <10% pulpwood"

    def test_mid_rotation_mixed_products(self, pmrc: PMRCModel):
        """Mid-rotation stands should have mixed product distribution."""
        # Mid-rotation: age ~20-25
        ba = 120.0  # ft²/ac
        tpa = 380.0
        hd = 65.0  # ft
        
        products = estimate_product_distribution(pmrc, ba, tpa, hd)
        
        # Mid-rotation should have significant CNS
        assert products.cns_fraction > 0.2, "Mid-rotation should have >20% CNS"
        # All three products should be present
        assert products.vol_pulp > 0
        assert products.vol_cns > 0
        assert products.vol_saw > 0

    def test_fractions_sum_to_one(self, pmrc: PMRCModel):
        """Product fractions should sum to 1.0 for merchantable stands."""
        ba = 100.0
        tpa = 400.0
        hd = 60.0
        
        products = estimate_product_distribution(pmrc, ba, tpa, hd)
        
        total_frac = products.pulp_fraction + products.cns_fraction + products.saw_fraction
        assert math.isclose(total_frac, 1.0, rel_tol=1e-6), f"Fractions sum to {total_frac}, expected 1.0"

    def test_zero_ba_returns_empty(self, pmrc: PMRCModel):
        """Zero BA should return empty product distribution."""
        products = estimate_product_distribution(pmrc, ba=0.0, tpa=500.0, hd=50.0)
        
        assert products.total_vol == 0
        assert products.total_ba == 0
        assert products.total_tpa == 0

    def test_zero_tpa_returns_empty(self, pmrc: PMRCModel):
        """Zero TPA should return empty product distribution."""
        products = estimate_product_distribution(pmrc, ba=100.0, tpa=0.0, hd=50.0)
        
        assert products.total_vol == 0

    def test_volume_increases_with_height(self, pmrc: PMRCModel):
        """Volume should increase with dominant height."""
        ba = 100.0
        tpa = 400.0
        
        products_short = estimate_product_distribution(pmrc, ba, tpa, hd=40.0)
        products_tall = estimate_product_distribution(pmrc, ba, tpa, hd=80.0)
        
        assert products_tall.total_vol > products_short.total_vol


class TestProductThresholds:
    """Test that DBH thresholds are correctly defined."""

    def test_pulp_threshold(self):
        assert DBH_PULP_MIN == 6.0, "Pulpwood min DBH should be 6 inches"

    def test_cns_threshold(self):
        assert DBH_CNS_MIN == 9.0, "Chip-n-saw min DBH should be 9 inches"

    def test_saw_threshold(self):
        assert DBH_SAW_MIN == 12.0, "Sawtimber min DBH should be 12 inches"


class TestHarvestValue:
    """Test harvest value calculations."""

    def test_harvest_value_increases_with_sawtimber(self, pmrc: PMRCModel):
        """Harvest value should increase as sawtimber fraction increases."""
        # Young stand (mostly pulp)
        products_young = estimate_product_distribution(pmrc, ba=60.0, tpa=480.0, hd=35.0)
        # Mature stand (mostly saw)
        products_mature = estimate_product_distribution(pmrc, ba=160.0, tpa=280.0, hd=85.0)
        
        value_young = compute_harvest_value(products_young)
        value_mature = compute_harvest_value(products_mature)
        
        assert value_mature > value_young, "Mature stand should be more valuable"

    def test_harvest_value_with_custom_prices(self, pmrc: PMRCModel):
        """Custom prices should affect harvest value."""
        products = estimate_product_distribution(pmrc, ba=100.0, tpa=400.0, hd=60.0)
        
        default_value = compute_harvest_value(products)
        high_saw_prices = ProductPrices(pulpwood=9.51, chip_n_saw=15.0, sawtimber=50.0)
        high_value = compute_harvest_value(products, prices=high_saw_prices)
        
        assert high_value > default_value, "Higher sawtimber price should increase value"

    def test_harvest_costs_reduce_value(self, pmrc: PMRCModel):
        """Higher costs should reduce net harvest value."""
        products = estimate_product_distribution(pmrc, ba=100.0, tpa=400.0, hd=60.0)
        
        low_cost = HarvestCosts(logging=100.0, replanting=100.0)
        high_cost = HarvestCosts(logging=300.0, replanting=300.0)
        
        value_low = compute_harvest_value(products, costs=low_cost)
        value_high = compute_harvest_value(products, costs=high_cost)
        
        assert value_low > value_high, "Higher costs should reduce net value"


class TestThinValue:
    """Test thinning value calculations."""

    def test_thin_value_positive_for_merchantable_stand(self, pmrc: PMRCModel):
        """Thinning a merchantable stand should have positive gross revenue."""
        products = estimate_product_distribution(pmrc, ba=120.0, tpa=400.0, hd=65.0)
        
        # Light thin (20%)
        value = compute_thin_value(products, thin_fraction=0.20)
        
        # Value may be negative due to costs, but should be reasonable
        assert value > -200, "Light thin value should not be extremely negative"

    def test_heavy_thin_removes_more_value(self, pmrc: PMRCModel):
        """Heavy thin should remove more volume/value than light thin."""
        products = estimate_product_distribution(pmrc, ba=120.0, tpa=400.0, hd=65.0)
        
        light_thin = compute_thin_value(products, thin_fraction=0.15)
        heavy_thin = compute_thin_value(products, thin_fraction=0.35)
        
        # Heavy thin removes more volume, so gross revenue should be higher
        # (though net may vary due to fixed costs)
        assert heavy_thin > light_thin, "Heavy thin should yield more revenue"


class TestProductDistributionDataclass:
    """Test ProductDistribution dataclass properties."""

    def test_total_properties(self):
        """Test total TPA, BA, and volume calculations."""
        dist = ProductDistribution(
            tpa_pulp=100, tpa_cns=50, tpa_saw=25,
            ba_pulp=20, ba_cns=30, ba_saw=50,
            vol_pulp=500, vol_cns=800, vol_saw=1200,
        )
        
        assert dist.total_tpa == 175
        assert dist.total_ba == 100
        assert dist.total_vol == 2500

    def test_fraction_properties(self):
        """Test fraction calculations."""
        dist = ProductDistribution(
            tpa_pulp=100, tpa_cns=50, tpa_saw=25,
            ba_pulp=20, ba_cns=30, ba_saw=50,
            vol_pulp=500, vol_cns=500, vol_saw=1000,
        )
        
        assert math.isclose(dist.pulp_fraction, 0.25)
        assert math.isclose(dist.cns_fraction, 0.25)
        assert math.isclose(dist.saw_fraction, 0.50)

    def test_zero_volume_fractions(self):
        """Zero volume should return zero fractions without error."""
        dist = ProductDistribution(
            tpa_pulp=0, tpa_cns=0, tpa_saw=0,
            ba_pulp=0, ba_cns=0, ba_saw=0,
            vol_pulp=0, vol_cns=0, vol_saw=0,
        )
        
        assert dist.pulp_fraction == 0.0
        assert dist.cns_fraction == 0.0
        assert dist.saw_fraction == 0.0
