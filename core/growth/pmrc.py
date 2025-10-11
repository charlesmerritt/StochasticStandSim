from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, Iterable, Mapping, Optional, Protocol, Tuple


# --- Public enums -------------------------------------------------------------

class Region(Enum):
    LOWER_COASTAL_PLAIN = auto()
    UPPER_COASTAL_PLAIN = auto()
    PIEDMONT = auto()


class ProductClass(Enum):
    # Define by DBH threshold d (in) and top diameter t (in), outside bark.
    PULPWOOD = auto()
    CHIP_N_SAW = auto()
    SAWTIMBER = auto()


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
    """Adapter for PMRC Technical Report 1996-1. Implements formulas and uses
    region-specific coefficients from the report."""

    # Coefficient stores by region and model
    coeffs: Dict[str, Dict[Region, Tuple[float, ...]]] = field(default_factory=dict)

    # ---- Height/site ----
    def project_hd(self, age1: float, hd1: float, age2: float, si25: float) -> float:
        # TODO: implement Chapman–Richards HD projection used in TR1996-1, eq (12).
        raise NotImplementedError("Implement project_hd per TR1996-1 eq. (12).")

    def site_index_from_hd(self, age: float, hd: float) -> float:
        # TODO: implement eq. (13) site-index formulation for soil group A.
        raise NotImplementedError("Implement site_index_from_hd per TR1996-1 eq. (13).")

    def hd_from_site_index(self, age: float, si25: float) -> float:
        # TODO: implement eq. (14) to get HD at age from SI25.
        raise NotImplementedError("Implement hd_from_site_index per TR1996-1 eq. (14).")


    # ---- Survival ----
    def project_tpa(self, age1: float, tpa1: float, age2: float, si25: float) -> float:
        # TODO: implement survival with asymptote 100 TPA, TR1996-1 eq. (15). If TPA<100, TPA_2 = TPA_1.
        raise NotImplementedError("Implement project_tpa per TR1996-1 eq. (15).")

    # ---- Basal area ----
    def predict_ba(self, age: float, tpa: float, hd: float, region: Region,
                   percent_hardwood_ba: Optional[float] = None) -> float:
        # TODO: implement BA prediction eq. (16)
        raise NotImplementedError("Implement predict_ba per TR1996-1 eq. (16).")

    def project_ba(self, age1: float, ba1: float, tpa1: float, hd1: float,
                   age2: float, tpa2: float, hd2: float, region: Region,
                   percent_hardwood_ba: Optional[float] = None) -> float:
        # TODO: implement BA projection eq. (17).
        raise NotImplementedError("Implement project_ba per TR1996-1 eq. (17).")

    def hardwood_adjustment(self, ba: float, percent_hardwood_ba: float) -> float:
        # TODO: implement hardwood BA adjustment eq. (18).
        raise NotImplementedError("Implement hardwood_adjustment per TR1996-1 eq. (18).")

    # ---- Yield ----
    def predict_yield(self, age: float, tpa: float, hd: float, ba: float, region: Region,
                      unit: str) -> float:
        if region == Region.PIEDMONT or region == Region.UCP:
            return self.predict_yield_piedmont_ucp(age, tpa, hd, ba, unit)
        elif region == Region.LCP:
            return self.predict_yield_lcp(age, tpa, hd, ba, unit)

    def predict_yield_piedmont_ucp(self, age: float, tpa: float, hd: float, ba: float, unit: str) -> float:
        # TODO: implement per acre yield prediction, eq. (19) for Piedmont+UCP.
        raise NotImplementedError("Implement predict_yield_piedmont_ucp per TR1996-1 eq. (19).")

    def predict_yield_lcp(self, age: float, tpa: float, hd: float, ba: float, unit: str) -> float:
        # TODO: implement per acre yield prediction, eq. (20) for LCP.
        raise NotImplementedError("Implement predict_yield_lcp per TR1996-1 eq. (20).")

    # ---- Product breakdown ----
    def merchantable_fraction(self, total_yield: float, d_dbh_min: float, t_top: float,
                              tpa: float, hd: float, ba: float, region: Region,
                              unit: str) -> float:
        # TODO: implement yield allocation eq. (21) with region- and unit-specific b1..b5.
        raise NotImplementedError("Implement merchantable_fraction per TR1996-1 eq. (21).")

    # ---- Percentiles / diameter distribution ----
    def diameter_percentiles(self, ba: float, tpa: float, region: Region,
                             percent_hardwood_ba: Optional[float] = None
                             ) -> Mapping[int, float]:
        # TODO: implement percentile model eq. (22)
        raise NotImplementedError("Implement diameter_percentiles per TR1996-1 eq. (22).")

    # ---- Relative size projection ----
    def project_relative_size(self, b_avg1: float, b_i1: float,
                              age1: float, age2: float, region: Region) -> float:
        # TODO: implement relative size projection eq. (23).
        raise NotImplementedError("Implement project_relative_size per TR1996-1 eq. (23).")

    # ---- Height|DBH curve ----
    def height_given_dbh(self, hd: float, dq: float, dbh: float, region: Region) -> float:
        # TODO: implement height|DBH eq. (24) with region-specific parameters.
        raise NotImplementedError("Implement height_given_dbh per TR1996-1 eq. (24).")

    # ---- Thinning ----
    def competition_index(self, ba: float, tpa: float, region: Region) -> float:
        # TODO: implement competition index eq. (26).
        raise NotImplementedError("Implement competition_index per TR1996-1 eq. (26).")

    def estimate_ba_removed(self, ba_before: float, tpa_before: float,
                            tpa_removed_row: float, tpa_removed_select: float) -> float:
        # TODO: implement thinned BA estimator eq. (25).
        raise NotImplementedError("Implement estimate_ba_removed per TR1996-1 eq. (25).")

    def project_thin_response_ba(self, ba_unthin2: float, ci1: float,
                                 age1: float, age2: float, region: Region) -> float:
        # TODO: implement CI projection eq. (27) and apply eq. (28).
        raise NotImplementedError("Implement project_thin_response_ba per TR1996-1 eqs. (27),(28).")

    # ---- Fertilization response ----
    def fert_response_hd(self, years_since_treat: float, n_lbs_ac: float, with_p: bool) -> float:
        # TODO: implement fertilizer response for HD eq. (29).
        raise NotImplementedError("Implement fert_response_hd per TR1996-1 eq. (29).")

    def fert_response_ba(self, years_since_treat: float, n_lbs_ac: float, with_p: bool) -> float:
        # TODO: implement fertilizer response for BA eq. (30).
        raise NotImplementedError("Implement fert_response_ba per TR1996-1 eq. (30).")


# --- Simple registry so other modules can resolve the growth engine -----------

_registry: Dict[str, PMRCEqns] = {}

def register_equations(name: str, impl: PMRCEqns) -> None:
    _registry[name] = impl

def get_growth(name: str, default_region: Region) -> PMRCGrowth:
    impl = _registry[name]
    return PMRCGrowth(eqns=impl, default_region=default_region)
