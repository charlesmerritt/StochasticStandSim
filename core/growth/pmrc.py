from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Protocol

from .types import ProductClass, Region
from . import pmrc_coeffs as coeffs


# --- Data containers ----------------------------------------------------------

@dataclass(frozen=True)
class StandState:
    """Whole-stand state at a given age."""
    age: float                 # years
    tpa: float                 # trees per acre
    hd: float                  # dominant height, ft
    ba: Optional[float] = None # basal area, ft^2/ac (None if unknown)
    si25: Optional[float] = None  # site index, base age 25, ft
    region: Optional[Region] = None
    percent_hardwood_ba: Optional[float] = None  # PHWD (% of pine BA), Piedmont only


@dataclass(frozen=True)
class MerchantabilitySpec:
    """Defines a merchantable cut by thresholds (outside bark)."""
    d_dbh_min: float  # inches
    t_top: float      # inches


class PMRCEqns(Protocol):
    """Protocol for PMRC equation set implementations. All functions are stateless."""

    # Site/height
    def site_index_from_hd(self, age: float, hd: float) -> float: ...
    def hd_from_site_index(self, age: float, si25: float) -> float: ...
    def project_hd(self, age1: float, hd1: float, age2: float, si25: float) -> float: ...

    # Survival
    def project_tpa(self, age1: float, tpa1: float, age2: float, si25: float) -> float: ...

    # Basal area
    def predict_ba(self, age: float, tpa: float, hd: float, region: Region,
                   percent_hardwood_ba: Optional[float] = None) -> float: ...
    def project_ba(self, age1: float, ba1: float, tpa1: float, hd1: float,
                   age2: float, tpa2: float, hd2: float, region: Region,
                   percent_hardwood_ba: Optional[float] = None) -> float: ...

    # Yield (TVOB, TVIB, GWOB, DWIB)
    def predict_yield(self, age: float, tpa: float, hd: float, ba: float, region: Region,
                      unit: str) -> float: ...

    # Product breakdown
    def merchantable_fraction(self, total_yield: float, d_dbh_min: float, t_top: float,
                              tpa: float, hd: float, ba: float, region: Region,
                              unit: str) -> float: ...

    # Percentiles/diameter distribution
    def diameter_percentiles(self, ba: float, tpa: float, region: Region,
                             percent_hardwood_ba: Optional[float] = None
                             ) -> Mapping[int, float]: ...

    # Relative size projection (stand table projection kernel)
    def project_relative_size(self, b_avg1: float, b_i1: float,
                              age1: float, age2: float, region: Region) -> float: ...

    # Height|DBH curve
    def height_given_dbh(self, hd: float, dq: float, dbh: float, region: Region) -> float: ...

    # Thinning helpers
    def competition_index(self, ba_thinned: float, ba_unthinned: float, region: Region) -> float: ...
    def estimate_ba_removed(self, ba_before: float, tpa_before: float,
                            tpa_removed_row: float, tpa_removed_select: float) -> float: ...
    def project_thin_response_ba(self, ba_unthin2: float, ci1: float,
                                 age1: float, age2: float, region: Region) -> float: ...

    # Fertilization response
    def fert_response_hd(self, years_since_treat: float, n_lbs_ac: float, with_p: bool) -> float: ...
    def fert_response_ba(self, years_since_treat: float, n_lbs_ac: float, with_p: bool) -> float: ...


@dataclass
class PMRCGrowth:
    """Stable facade for growth ops used by the simulator."""
    eqns: PMRCEqns
    default_region: Region

    # --- Height / site --------------------------------------------------------
    def site_index(self, s: StandState) -> float:
        si = s.si25 if s.si25 is not None else self.eqns.site_index_from_hd(s.age, s.hd)
        return si

    def project_height(self, s: StandState, age2: float) -> float:
        si25 = self.site_index(s)
        return self.eqns.project_hd(s.age, s.hd, age2, si25)

    # --- Survival -------------------------------------------------------------
    def project_tpa(self, s: StandState, age2: float) -> float:
        si25 = self.site_index(s)
        return self.eqns.project_tpa(s.age, s.tpa, age2, si25)

    # --- Basal area -----------------------------------------------------------
    def ba(self, s: StandState) -> float:
        if s.ba is not None:
            return s.ba
        region = s.region or self.default_region
        return self.eqns.predict_ba(
            age=s.age, tpa=s.tpa, hd=s.hd, region=region,
            percent_hardwood_ba=s.percent_hardwood_ba
        )

    def project_ba(self, s1: StandState, s2_age: float, tpa2: float, hd2: float) -> float:
        region = s1.region or self.default_region
        if s1.ba is None:
            ba1 = self.ba(s1)
        else:
            ba1 = s1.ba
        return self.eqns.project_ba(
            age1=s1.age, ba1=ba1, tpa1=s1.tpa, hd1=s1.hd,
            age2=s2_age, tpa2=tpa2, hd2=hd2, region=region,
            percent_hardwood_ba=s1.percent_hardwood_ba
        )

    # --- Yield and merchandising ---------------------------------------------
    def yield_total(self, s: StandState, unit: str = "TVOB") -> float:
        region = s.region or self.default_region
        ba = self.ba(s)
        return self.eqns.predict_yield(
            age=s.age, tpa=s.tpa, hd=s.hd, ba=ba, region=region, unit=unit
        )

    def yield_by_product(self, s: StandState, merch: MerchantabilitySpec,
                         unit: str = "TVOB") -> float:
        region = s.region or self.default_region
        total = self.yield_total(s, unit=unit)
        ba = self.ba(s)
        return self.eqns.merchantable_fraction(
            total_yield=total, d_dbh_min=merch.d_dbh_min, t_top=merch.t_top,
            tpa=s.tpa, hd=s.hd, ba=ba, region=region, unit=unit
        )

    # --- Diameter distribution ------------------------------------------------
    def diameter_percentiles(self, s: StandState) -> Mapping[int, float]:
        region = s.region or self.default_region
        ba = self.ba(s)
        return self.eqns.diameter_percentiles(
            ba=ba, tpa=s.tpa, region=region, percent_hardwood_ba=s.percent_hardwood_ba
        )

    # --- Stand-table projection primitive ------------------------------------
    def project_relative_size(self, b_avg1: float, b_i1: float,
                              age1: float, age2: float, region: Optional[Region] = None) -> float:
        return self.eqns.project_relative_size(
            b_avg1=b_avg1, b_i1=b_i1, age1=age1, age2=age2, region=region or self.default_region
        )

    # --- Height|DBH curve -----------------------------------------------------
    def height_given_dbh(self, s: StandState, dq: float, dbh: float) -> float:
        region = s.region or self.default_region
        return self.eqns.height_given_dbh(hd=s.hd, dq=dq, dbh=dbh, region=region)

    def competition_index(self, ba_thinned: float, ba_unthinned: float,
                           region: Optional[Region] = None) -> float:
        return self.eqns.competition_index(
            ba_thinned=ba_thinned,
            ba_unthinned=ba_unthinned,
            region=region or self.default_region,
        )

    # --- Thinning helpers -----------------------------------------------------
    def estimate_ba_removed(self, ba_before: float, tpa_before: float,
                            tpa_removed_row: float, tpa_removed_select: float) -> float:
        return self.eqns.estimate_ba_removed(ba_before, tpa_before,
                                             tpa_removed_row, tpa_removed_select)

    def project_thin_response_ba(self, ba_unthinned_future: float, ci_initial: float,
                                 age1: float, age2: float, region: Optional[Region] = None) -> float:
        return self.eqns.project_thin_response_ba(ba_unthinned_future, ci_initial,
                                                  age1, age2, region or self.default_region)

    # --- Fertilization add-ons ------------------------------------------------
    def fert_adjusted_hd(self, s: StandState, years_since_treat: float,
                         n_lbs_ac: float, with_p: bool) -> float:
        return s.hd + self.eqns.fert_response_hd(years_since_treat, n_lbs_ac, with_p)

    def fert_adjusted_ba(self, s: StandState, years_since_treat: float,
                         n_lbs_ac: float, with_p: bool) -> float:
        base_ba = self.ba(s)
        return base_ba + self.eqns.fert_response_ba(years_since_treat, n_lbs_ac, with_p)


# --- Adapter skeleton (PMRC 1996) ------------------------------------

@dataclass
class PMRC1996(PMRCEqns):
    """Implementation of the PMRC Technical Report 1996-1 models."""

    # ---- Height/site ----
    def project_hd(self, age1: float, hd1: float, age2: float, si25: float) -> float:
        """Project dominant height using the Chapman–Richards formulation."""

        alpha = coeffs.HEIGHT_SITE_PARAMETERS["alpha"]
        m = coeffs.HEIGHT_SITE_PARAMETERS["m"]

        def _g(age: float) -> float:
            return 1.0 - math.exp(-alpha * age)

        if age2 <= age1:
            return hd1
        ratio = _g(age2) / _g(age1)
        return hd1 * ratio ** m

    def site_index_from_hd(self, age: float, hd: float) -> float:
        alpha = coeffs.HEIGHT_SITE_PARAMETERS["alpha"]
        m = coeffs.HEIGHT_SITE_PARAMETERS["m"]
        base = coeffs.HEIGHT_SITE_PARAMETERS["ratio_base"]

        g_age = 1.0 - math.exp(-alpha * age)
        return hd * (base / g_age) ** m

    def hd_from_site_index(self, age: float, si25: float) -> float:
        alpha = coeffs.HEIGHT_SITE_PARAMETERS["alpha"]
        m = coeffs.HEIGHT_SITE_PARAMETERS["m"]
        base = coeffs.HEIGHT_SITE_PARAMETERS["ratio_base"]

        g_age = 1.0 - math.exp(-alpha * age)
        return si25 * (base / g_age) ** (-m)

    # ---- Survival ----
    def project_tpa(self, age1: float, tpa1: float, age2: float, si25: float) -> float:
        params = coeffs.TPA_PARAMETERS
        asym = params["asymptote"]

        if age2 <= age1:
            return tpa1
        if tpa1 <= asym:
            return tpa1

        b = params["b"]
        c = params["c"]
        d = params["d"]

        term = (tpa1 - asym) ** (-b) + (c ** 2) * si25 * (age2 ** d - age1 ** d)
        return asym + term ** (-1.0 / b)

    # ---- Basal area ----
    def predict_ba(self, age: float, tpa: float, hd: float, region: Region,
                   percent_hardwood_ba: Optional[float] = None) -> float:
        if age <= 0 or tpa <= 0 or hd <= 0:
            raise ValueError("Age, TPA and HD must be positive for BA prediction")

        region_key = region
        if region == Region.PIEDMONT and percent_hardwood_ba is None:
            percent_hardwood_ba = 0.0

        if region == Region.PIEDMONT:
            coeff = coeffs.BA_PREDICT[Region.PIEDMONT]
        else:
            coeff = coeffs.BA_PREDICT[region_key]

        ln_tpa = math.log(tpa)
        ln_hd = math.log(hd)
        inv_age = coeff.inv_age / age
        ln_tpa_over_age = ln_tpa / age
        ln_hd_over_age = ln_hd / age

        ln_ba = (
            coeff.intercept
            + inv_age
            + coeff.ln_tpa * ln_tpa
            + coeff.ln_hd * ln_hd
            + coeff.ln_tpa_over_age * ln_tpa_over_age
            + coeff.ln_hd_over_age * ln_hd_over_age
        )

        if region == Region.PIEDMONT and percent_hardwood_ba is not None:
            ln_ba += coeffs.HARDWOOD_ADJUSTMENT["coeff"] * percent_hardwood_ba

        return math.exp(ln_ba)

    def project_ba(self, age1: float, ba1: float, tpa1: float, hd1: float,
                   age2: float, tpa2: float, hd2: float, region: Region,
                   percent_hardwood_ba: Optional[float] = None) -> float:
        if age2 <= age1:
            return ba1

        if ba1 <= 0 or tpa1 <= 0 or hd1 <= 0 or tpa2 <= 0 or hd2 <= 0:
            raise ValueError("Inputs must be positive for BA projection")

        region_key = region if region != Region.PIEDMONT else Region.UPPER_COASTAL_PLAIN
        coeff = coeffs.BA_PROJECT[region_key]

        ln_ba1 = math.log(ba1)
        ln_tpa1 = math.log(tpa1)
        ln_hd1 = math.log(hd1)
        ln_tpa2 = math.log(tpa2)
        ln_hd2 = math.log(hd2)

        ln_ba2 = (
            ln_ba1
            + coeff.inv_age * ((1.0 / age2) - (1.0 / age1))
            + coeff.ln_tpa * (ln_tpa2 - ln_tpa1)
            + coeff.ln_hd * (ln_hd2 - ln_hd1)
            + coeff.ln_tpa_over_age * ((ln_tpa2 / age2) - (ln_tpa1 / age1))
            + coeff.ln_hd_over_age * ((ln_hd2 / age2) - (ln_hd1 / age1))
        )

        ba2 = math.exp(ln_ba2)

        if region == Region.PIEDMONT and percent_hardwood_ba is not None:
            adjustment = coeffs.HARDWOOD_ADJUSTMENT["coeff"] * percent_hardwood_ba
            return ba2 * math.exp(adjustment)
        return ba2

    # ---- Yield ----
    def predict_yield(self, age: float, tpa: float, hd: float, ba: float, region: Region,
                      unit: str) -> float:
        if age <= 0 or tpa <= 0 or hd <= 0 or ba <= 0:
            raise ValueError("Inputs must be positive for yield prediction")

        key = (region, unit.upper())
        if key not in coeffs.YIELD_COEFFICIENTS:
            raise KeyError(f"Unsupported yield unit {unit} for region {region}")

        intercept, b_hd, b_ba, b_tpa_over_age, b_hd_over_age, b_ba_over_age, b_tpa = coeffs.YIELD_COEFFICIENTS[key]

        ln_tpa = math.log(tpa)
        ln_hd = math.log(hd)
        ln_ba = math.log(ba)

        value = (
            intercept
            + b_hd * ln_hd
            + b_ba * ln_ba
            + b_tpa_over_age * (ln_tpa / age)
            + b_hd_over_age * (ln_hd / age)
            + b_ba_over_age * (ln_ba / age)
            + b_tpa * ln_tpa
        )
        return math.exp(value)

    # ---- Product breakdown ----
    def merchantable_fraction(self, total_yield: float, d_dbh_min: float, t_top: float,
                              tpa: float, hd: float, ba: float, region: Region,
                              unit: str) -> float:
        if total_yield <= 0:
            return 0.0
        if tpa <= 0 or ba <= 0:
            raise ValueError("TPA and BA must be positive for merchandising")

        key = (region, unit.upper())
        if key not in coeffs.MERCHANTABLE_COEFFICIENTS:
            raise KeyError(f"Unsupported merchandising unit {unit} for region {region}")

        b1, b2, b3, b4, b5 = coeffs.MERCHANTABLE_COEFFICIENTS[key]

        qmd = math.sqrt((ba / tpa) / 0.005454154)
        if qmd <= 0:
            raise ValueError("Computed QMD must be positive")

        ratio_top = t_top / qmd
        ratio_dbh = d_dbh_min / qmd

        fraction = math.exp(b1 * ratio_top ** b5 + b2 * (tpa ** b3) * (ratio_dbh ** b4))
        return total_yield * fraction

    # ---- Percentiles / diameter distribution ----
    def diameter_percentiles(self, ba: float, tpa: float, region: Region,
                             percent_hardwood_ba: Optional[float] = None
                             ) -> Mapping[int, float]:
        if ba <= 0 or tpa <= 0:
            raise ValueError("BA and TPA must be positive for diameter percentiles")

        qmd = math.sqrt((ba / tpa) / 0.005454154)
        ln_qmd = math.log(qmd)
        ln_tpa = math.log(tpa)
        ln_ba = math.log(ba)
        phwd = percent_hardwood_ba or 0.0

        result: Dict[int, float] = {}
        for percentile in coeffs.DIAMETER_PERCENTILE_COEFFICIENTS[region]:
            a, b_qmd, c_tpa, d_ba, e_hw = coeffs.percentile_coeffs(region, percentile)
            ln_dp = a + b_qmd * ln_qmd + c_tpa * ln_tpa + d_ba * ln_ba + e_hw * phwd
            result[percentile] = math.exp(ln_dp)
        return result

    # ---- Relative size projection ----
    def project_relative_size(self, b_avg1: float, b_i1: float,
                              age1: float, age2: float, region: Region) -> float:
        decay = coeffs.RELATIVE_SIZE_DECAY[region]
        if age2 <= age1:
            return b_i1
        delta = age2 - age1
        return b_avg1 + (b_i1 - b_avg1) * math.exp(-decay * delta)

    # ---- Height|DBH curve ----
    def height_given_dbh(self, hd: float, dq: float, dbh: float, region: Region) -> float:
        if hd <= 0 or dq <= 0 or dbh <= 0:
            raise ValueError("HD, DQ and DBH must be positive for height estimation")

        intercept, b_hd, b_dq, b_dbh, b_ratio_dq, b_ratio_hd = coeffs.HEIGHT_DBH_COEFFICIENTS[region]

        ln_hd = math.log(hd)
        ln_dq = math.log(dq)
        ln_dbh = math.log(dbh)
        ln_ratio_dq = math.log(dbh / dq)
        ln_ratio_hd = math.log(dbh / hd)

        ln_height = (
            intercept
            + b_hd * ln_hd
            + b_dq * ln_dq
            + b_dbh * ln_dbh
            + b_ratio_dq * ln_ratio_dq
            + b_ratio_hd * ln_ratio_hd
        )
        return math.exp(ln_height)

    # ---- Thinning ----
    def competition_index(self, ba_thinned: float, ba_unthinned: float, region: Region) -> float:
        if ba_unthinned <= 0:
            raise ValueError("Unthinned basal area must be positive")
        if ba_thinned < 0:
            raise ValueError("Thinned basal area must be non-negative")
        return 1.0 - (ba_thinned / ba_unthinned)

    def estimate_ba_removed(self, ba_before: float, tpa_before: float,
                            tpa_removed_row: float, tpa_removed_select: float) -> float:
        if ba_before < 0 or tpa_before <= 0:
            raise ValueError("Basal area must be non-negative and TPA positive")

        remaining = max(tpa_before - tpa_removed_row, 1e-9)
        row_fraction = tpa_removed_row / tpa_before
        select_fraction = 0.0
        if tpa_removed_select > 0:
            select_fraction = (tpa_removed_select / remaining) ** 1.2345

        total_fraction = row_fraction + (1.0 - row_fraction) * select_fraction
        return ba_before * total_fraction

    def project_thin_response_ba(self, ba_unthin2: float, ci1: float,
                                 age1: float, age2: float, region: Region) -> float:
        if age2 <= age1:
            return ba_unthin2 * (1 - ci1)

        decay = coeffs.RELATIVE_SIZE_DECAY[region]
        ci2 = ci1 * math.exp(-decay * (age2 - age1))
        return ba_unthin2 * (1.0 - ci2)

    # ---- Fertilization response ----
    def fert_response_hd(self, years_since_treat: float, n_lbs_ac: float, with_p: bool) -> float:
        if years_since_treat < 0:
            raise ValueError("Years since treatment must be non-negative")
        n = coeffs.FERT_HEIGHT["N"] * n_lbs_ac
        p = coeffs.FERT_HEIGHT["P"] if with_p else 0.0
        k = coeffs.FERT_HEIGHT["k"]
        return (n + p) * years_since_treat * math.exp(-k * years_since_treat)

    def fert_response_ba(self, years_since_treat: float, n_lbs_ac: float, with_p: bool) -> float:
        if years_since_treat < 0:
            raise ValueError("Years since treatment must be non-negative")
        n = coeffs.FERT_BA["N"] * n_lbs_ac
        p = coeffs.FERT_BA["P"] if with_p else 0.0
        k = coeffs.FERT_BA["k"]
        return (n + p) * years_since_treat * math.exp(-k * years_since_treat)


# --- Simple registry so other modules can resolve the growth engine -----------

_registry: Dict[str, PMRCEqns] = {}

def register_equations(name: str, impl: PMRCEqns) -> None:
    _registry[name] = impl

def get_growth(name: str, default_region: Region) -> PMRCGrowth:
    impl = _registry[name]
    return PMRCGrowth(eqns=impl, default_region=default_region)
