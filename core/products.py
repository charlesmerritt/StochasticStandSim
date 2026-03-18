"""Product distribution and valuation helpers.

Product volumes now come from the PMRC merchantability equations in
``PMRCModel.product_yields()``. Weibull diameter classes are retained only
for optional TPA/BA-by-class summaries and thinning structure logic.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.pmrc_model import PMRCModel, Region


# Standard DBH thresholds (inches)
DBH_PULP_MIN = 6.0
DBH_CNS_MIN = 9.0
DBH_SAW_MIN = 12.0
DBH_MAX = 30.0  # Upper bound for integration


@dataclass
class ProductDistribution:
    """Distribution of timber products in a stand."""
    
    # Trees per acre in each class
    tpa_pulp: float
    tpa_cns: float
    tpa_saw: float
    
    # Basal area (ft²/acre) in each class
    ba_pulp: float
    ba_cns: float
    ba_saw: float
    
    # Volume (cuft/acre) in each class from PMRC merchantability equations
    vol_pulp: float
    vol_cns: float
    vol_saw: float
    
    @property
    def total_tpa(self) -> float:
        return self.tpa_pulp + self.tpa_cns + self.tpa_saw
    
    @property
    def total_ba(self) -> float:
        return self.ba_pulp + self.ba_cns + self.ba_saw
    
    @property
    def total_vol(self) -> float:
        return self.vol_pulp + self.vol_cns + self.vol_saw
    
    @property
    def pulp_fraction(self) -> float:
        """Fraction of merchantable volume that is pulpwood."""
        total = self.total_vol
        return self.vol_pulp / total if total > 0 else 0.0
    
    @property
    def cns_fraction(self) -> float:
        """Fraction of merchantable volume that is chip-n-saw."""
        total = self.total_vol
        return self.vol_cns / total if total > 0 else 0.0
    
    @property
    def saw_fraction(self) -> float:
        """Fraction of merchantable volume that is sawtimber."""
        total = self.total_vol
        return self.vol_saw / total if total > 0 else 0.0


def estimate_product_distribution(
    pmrc: PMRCModel,
    age: float,
    ba: float,
    tpa: float,
    hd: float,
    region: Region = "ucp",
    phwd: float = 0.0,
) -> ProductDistribution:
    """Estimate product distribution from stand state.

    Product volumes are computed from PMRC merchantability equations.
    Optional class-level TPA/BA summaries are still derived from the Weibull
    approximation when it is available.
    
    Args:
        pmrc: PMRC model instance
        age: Stand age (years)
        ba: Basal area (ft²/acre)
        tpa: Trees per acre
        hd: Dominant height (ft)
        region: Geographic region
        phwd: Percent hardwood (0-100)
    
    Returns:
        ProductDistribution with TPA, BA, and volume by product class
    """
    if age <= 0 or ba <= 0 or tpa <= 0 or hd <= 0:
        return ProductDistribution(
            tpa_pulp=0, tpa_cns=0, tpa_saw=0,
            ba_pulp=0, ba_cns=0, ba_saw=0,
            vol_pulp=0, vol_cns=0, vol_saw=0,
        )

    # PMRC merchantability equations are the canonical product-volume source.
    pmrc_yields = pmrc.product_yields(
        age=age,
        tpa=tpa,
        hd=hd,
        ba=ba,
        unit="TVOB",
        region=region,
    )

    # Keep the Weibull pathway only for optional diameter-structure summaries.
    dbh_bounds = [0.0, DBH_PULP_MIN, DBH_CNS_MIN, DBH_SAW_MIN, DBH_MAX]
    tpa_pulp = tpa_cns = tpa_saw = 0.0
    ba_pulp = ba_cns = ba_saw = 0.0
    try:
        dist = pmrc.diameter_class_distribution(
            ba=ba,
            tpa=tpa,
            dbh_bounds=dbh_bounds,
            region=region,
            phwd=phwd,
        )
        _, tpa_pulp, tpa_cns, tpa_saw = dist.tpa_per_class
        _, ba_pulp, ba_cns, ba_saw = dist.ba_per_class
    except (ValueError, RuntimeError):
        pass
    
    return ProductDistribution(
        tpa_pulp=float(tpa_pulp),
        tpa_cns=float(tpa_cns),
        tpa_saw=float(tpa_saw),
        ba_pulp=float(ba_pulp),
        ba_cns=float(ba_cns),
        ba_saw=float(ba_saw),
        vol_pulp=float(pmrc_yields.pulpwood),
        vol_cns=float(pmrc_yields.chip_n_saw),
        vol_saw=float(pmrc_yields.sawtimber),
    )


@dataclass
class ProductPrices:
    """Stumpage prices by product class ($/ton)."""
    pulpwood: float = 9.51
    chip_n_saw: float = 23.51
    sawtimber: float = 27.82


@dataclass
class HarvestCosts:
    """Harvest and regeneration costs ($/acre)."""
    logging: float = 150.0
    replanting: float = 150.80
    
    @property
    def total(self) -> float:
        return self.logging + self.replanting


# Conversion factor: cubic feet to tons (green weight)
CUFT_TO_TON = 0.031


def compute_harvest_value(
    products: ProductDistribution,
    prices: ProductPrices | None = None,
    costs: HarvestCosts | None = None,
) -> float:
    """Compute net harvest value from product distribution.
    
    Args:
        products: Product distribution from estimate_product_distribution
        prices: Stumpage prices by product class
        costs: Harvest and regeneration costs
    
    Returns:
        Net harvest value ($/acre)
    """
    if prices is None:
        prices = ProductPrices()
    if costs is None:
        costs = HarvestCosts()
    
    # Convert volume to tons
    tons_pulp = products.vol_pulp * CUFT_TO_TON
    tons_cns = products.vol_cns * CUFT_TO_TON
    tons_saw = products.vol_saw * CUFT_TO_TON
    
    # Revenue by product class
    revenue = (
        tons_pulp * prices.pulpwood +
        tons_cns * prices.chip_n_saw +
        tons_saw * prices.sawtimber
    )
    
    return revenue - costs.total


def compute_thin_value(
    products: ProductDistribution,
    thin_fraction: float,
    prices: ProductPrices | None = None,
    thin_cost: float = 87.34,
) -> float:
    """Compute net thinning value.
    
    Thinning removes smaller trees first (from below), so the removed
    volume is weighted toward pulpwood.
    
    Args:
        products: Product distribution before thinning
        thin_fraction: Fraction of BA to remove (0-1)
        prices: Stumpage prices
        thin_cost: Fixed thinning cost ($/acre)
    
    Returns:
        Net thinning value ($/acre)
    """
    if prices is None:
        prices = ProductPrices()
    
    # Thinning from below: remove pulpwood first, then CNS
    # Simplified model: removed volume is 80% pulp, 20% CNS for light thin
    # and 60% pulp, 30% CNS, 10% saw for heavy thin
    total_vol = products.total_vol
    removed_vol = total_vol * thin_fraction
    
    if thin_fraction <= 0.25:
        # Light thin - mostly pulpwood
        pulp_frac = 0.80
        cns_frac = 0.20
        saw_frac = 0.0
    else:
        # Heavy thin - some larger trees
        pulp_frac = 0.60
        cns_frac = 0.30
        saw_frac = 0.10
    
    tons_pulp = removed_vol * pulp_frac * CUFT_TO_TON
    tons_cns = removed_vol * cns_frac * CUFT_TO_TON
    tons_saw = removed_vol * saw_frac * CUFT_TO_TON
    
    revenue = (
        tons_pulp * prices.pulpwood +
        tons_cns * prices.chip_n_saw +
        tons_saw * prices.sawtimber
    )
    
    return revenue - thin_cost
