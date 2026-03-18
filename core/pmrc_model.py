from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple, Dict, Sequence, List
from math import exp, log, sqrt
import numpy as np

Region = Literal["ucp", "pucp", "lcp"]
Percentile = Literal[0, 25, 50, 95]

# Default DBH class bounds for thinning (inches)
# Sub-merchantable: 0-6", Pulpwood: 6-9", CNS: 9-12", Sawtimber: 12-24"
DEFAULT_DBH_BOUNDS = np.array([0.0, 6.0, 9.0, 12.0, 24.0])
YieldUnit = Literal["TVOB", "TVIB", "GWOB", "DWIB"]
ProductCoeff = Tuple[float, float, float, float, float]

@dataclass
class ProductYields:
    pulpwood: float
    chip_n_saw: float
    sawtimber: float

@dataclass
class WeibullParams:
    """Three-parameter Weibull distribution."""

    a: float  # location
    b: float  # scale
    c: float  # shape


@dataclass
class SizeClassDistribution:
    """Diameter class counts derived from a Weibull representation.
    
    This is the canonical size class distribution using PMRC tech report
    coefficients for Weibull parameter estimation.
    """

    dbh_bounds: np.ndarray
    tpa_per_class: np.ndarray
    ba_per_class: np.ndarray
    weibull_params: WeibullParams | None = None
    percentiles: Dict[Percentile, float] | None = None

    def validate(self) -> None:
        if self.dbh_bounds.ndim != 1 or self.dbh_bounds.size < 2:
            raise ValueError("dbh_bounds must be 1D with length >= 2.")
        if np.any(np.diff(self.dbh_bounds) <= 0):
            raise ValueError("dbh_bounds must be strictly increasing.")
        if np.any(self.tpa_per_class < 0.0) or np.any(self.ba_per_class < 0.0):
            raise ValueError("Class values must be non-negative.")
        if self.tpa_per_class.shape != self.ba_per_class.shape:
            raise ValueError("TPA and BA arrays must have matching shapes.")
    
    @property
    def total_tpa(self) -> float:
        return float(np.sum(self.tpa_per_class))
    
    @property
    def total_ba(self) -> float:
        return float(np.sum(self.ba_per_class))

class PMRCModel:
    """
    Implements the PMRC equations from Harrison & Borders (PMRC Technical Report 1996-1).
    Fertilization returns additive deltas that you must add to hd and ba.
    """

    # BA coefficients (Eq. 16, 17). Keys normalized to lowercase.
    _BA_COEFFS: Dict[str, Tuple[float, float, float, float, float, float]] = {
        "lcp":  (0.0,       -42.689283, 0.367244, 0.659985, 2.012724, 7.703502),
        "ucp":  (-0.855557, -36.050347, 0.299071, 0.980246, 3.309212, 3.787258),
        "pucp": (-0.855557, -36.050347, 0.299071, 0.980246, 3.309212, 3.787258),  # alias
    }

    _YIELD_COEFFS_PUCP: Dict[YieldUnit, Tuple[float, float, float, float, float, float]] = {
        "TVOB": (0.0,       0.268552, 1.368844, -7.466863, 8.934524, 3.553411),
        "TVIB": (0.0,       0.350394, 1.263708, -8.608165, 7.193937, 6.309586),
        "GWOB": (-3.818016, 0.430179, 1.276768, -8.088792, 7.428472, 5.554509),
        "DWIB": (-4.987560, 0.446433, 1.348843, -7.757842, 7.857337, 4.222016),
    }

    _YIELD_COEFFS_LCP: Dict[YieldUnit, Tuple[float, float, float, float, float, float]] = {
        "TVOB": (-1.520877, 0.200680, 1.207586, 0.703405, -5.139064, 6.744164),
        "TVIB": (-2.088857, 0.177587, 1.303770, 0.726950, -5.091474, 6.676532),
        "GWOB": (-5.175922, 0.198424, 1.232028, 0.705769, -5.129853, 6.731477),
        "DWIB": (-6.332502, 0.145815, 1.296629, 0.814967, -4.660198, 5.383589),
    }

    _PRODUCT_COEFFS_PUCP: Dict[YieldUnit, ProductCoeff] = {
        "TVOB": (-0.982648, 3.991140, -0.748261, -0.111206, 5.784780),
        "TVIB": (-1.036792, 3.900677, -0.511939, -0.046007, 5.640610),
        "GWOB": (-1.007482, 3.931373, -0.518057, -0.048385, 5.660573),
        "DWIB": (-0.934936, 4.111618, -0.590269, -0.065355, 5.596179),
    }

    _PRODUCT_COEFFS_LCP: Dict[YieldUnit, ProductCoeff] = {
        "TVOB": (-1.034486, 3.940848, -5.062955, -0.422892, 6.004646),
        "TVIB": (-1.105225, 3.878664, -4.459271, -0.404057, 5.984225),
        "GWOB": (-1.064132, 3.818683, -5.048319, -0.422117, 5.991728),
        "DWIB": (-0.963185, 4.054202, -4.540672, -0.406561, 5.962867),
    }

    PINE_PRODUCT_SPECS = {
        "pulpwood": {"top_dia": 2.0, "dbh_lim": 4.5},
        "chip_n_saw": {"top_dia": 4.0, "dbh_lim": 8.5},
        "sawtimber": {"top_dia": 8.0, "dbh_lim": 11.5},
    }


    def __init__(self, region: Region = "ucp"):
        # Chapman–Richards and site-index parameters
        self.k = 0.014452
        self.m = 0.8216
        self.c = 0.30323
        # Survival parameters
        self.p = 0.745339
        self.g = 0.0003425
        self.r = 1.97472
        # Default region
        self.region = region.lower()
        # CI recovery betas
        self.beta_ucp = 0.076472
        self.beta_lcp = 0.110521
        # Survival asymptote
        self.min_tpa_asymptote = 100.0
        # no-op debug
        self._debug = lambda *args: None

    # ---------- Height and SI ----------
    def hd_project(self, age1: float, hd1: float, age2: float) -> float:
        """
        Eq. 12
        """
        if age2 < 0 or age1 < 0:
            raise ValueError("ages must be >= 0")
        if age2 < age1:
            raise ValueError("age2 must be >= age1")
        num = 1.0 - exp(-self.k * age2)
        den = 1.0 - exp(-self.k * age1)
        if den <= 0.0:
            raise ValueError("invalid age1")
        return hd1 * (num / den) ** self.m

    def si_from_hd(self, hd: float, form: str = "projection") -> float:
        """
        Eq. 10 (PS80) and Eq. 13 (projection-consistent).
        Convert dominant height at age A to site index at base age 25 (SI25).
        """
        if form.lower() == "ps80":
            c, k, e = 0.7476, 0.05507, 1.435
            return hd * (c / (1.0 - exp(-k))) ** e
        # projection-consistent
        return hd * (self.c / (1.0 - exp(-self.k))) ** self.m

    def hd_from_si(self, si25: float, form: str = "projection") -> float:
        """
        Eq. 11 (PS80) and Eq. 14 (projection-consistent).
        Convert SI25 to dominant height at age A used in the converter.
        """
        if form.lower() == "ps80":
            c, k, e = 0.7476, 0.05507, 1.435
            return si25 * (c / (1.0 - exp(-k))) ** -e
        # projection-consistent
        return si25 * (self.c / (1.0 - exp(-self.k))) ** -self.m

    # ---------- Survival (TPA) ----------
    def tpa_project(self, tpa1: float, si25: float, age1: float, age2: float) -> float:
        """
        Replicates R: squares g. Holds TPA if <= asymptote.
        TPA2 = 100 + ( (TPA1-100)^-p + (g^2)*SI25*(A2^r - A1^r) )^(-1/p)
        """
        if age1 < 0 or age2 < 0:
            raise ValueError("ages must be >= 0")
        if age2 < age1:
            raise ValueError("age2 must be >= age1")
        if tpa1 <= 0 or si25 <= 0:
            raise ValueError("tpa1 and si25 must be positive")
        if tpa1 <= self.min_tpa_asymptote:
            return tpa1
        p, g, r = self.p, self.g, self.r
        inner = ((tpa1 - 100.0) ** -p) + (g ** 2) * si25 * ((age2 ** r) - (age1 ** r))
        if inner <= 0:
            raise ValueError("invalid survival projection: inner term <= 0")

        return 100.0 + inner ** (-(1.0 / p))

    # ---------- Basal area ----------
    def _coeffs(self, region: Region | None) -> Tuple[float, float, float, float, float, float]:
        key = (region or self.region).lower()
        if key not in self._BA_COEFFS:
            raise ValueError(f"Unsupported region '{region}'")
        return self._BA_COEFFS[key]

    def ba_predict(self, age: float, tpa: float, hd: float, region: Region | None = None) -> float:
        if age <= 0 or tpa <= 0 or hd <= 0:
            raise ValueError("age, tpa, hd must be positive")
        b0, b1, b2, b3, b4, b5 = self._coeffs(region)
        lnBA = b0 + (b1 / age) + b2 * log(tpa) + b3 * log(hd) + b4 * (log(tpa) / age) + b5 * (log(hd) / age)
        return max(0.0, float(exp(lnBA)))

    def ba_project(
        self,
        age1: float,
        tpa1: float,
        tpa2: float,
        ba1: float,
        hd1: float,
        hd2: float,
        age2: float,
        region: Region | None = None,
    ) -> float:
        if tpa1 <= 0 or hd1 <= 0 or ba1 <= 0 or age1 <= 0 or age2 <= 0:
            raise ValueError("inputs must be positive")
        if age2 < age1:
            raise ValueError("age2 must be >= age1")
        if tpa2 <= 0 or hd2 <= 0:
            raise ValueError("tpa2 and hd2 must be positive")
        b0, b1, b2, b3, b4, b5 = self._coeffs(region)
        lnBA2 = (
            log(ba1)
            + b1 * ((1.0 / age2) - (1.0 / age1))
            + b2 * (log(tpa2) - log(tpa1))
            + b3 * (log(hd2) - log(hd1))
            + b4 * ((log(tpa2) / age2) - (log(tpa1) / age1))
            + b5 * ((log(hd2) / age2) - (log(hd1) / age1))
        )
        return max(0.0, float(exp(lnBA2)))

    # ---------- Competition index ----------
    def competition_index(self, ba_after: float, ba_unthinned: float) -> float:
        if ba_unthinned <= 0:
            raise ValueError("ba_unthinned must be positive")
        return max(0.0, min(1.0, 1.0 - (ba_after / ba_unthinned)))

    def ci_project(self, ci1: float, age1: float, age2: float, region: Region | None = None) -> float:
        key = (region or self.region).lower()
        if key in ("ucp", "pucp"):
            beta = self.beta_ucp
        elif key == "lcp":
            beta = self.beta_lcp
        else:
            raise ValueError(f"Unsupported region: {key}")
        if age2 <= age1:
            raise ValueError("age2 must be > age1")
        if ci1 < 0.0:
            raise ValueError("ci1 must be >= 0")
        return max(0.0, ci1 * exp(-beta * (age2 - age1)))

    def ba_thinned(self, ba_unthinned2: float, ci2: float) -> float:
        if ba_unthinned2 < 0:
            raise ValueError("ba_unthinned2 must be >= 0")
        ci2 = max(0.0, min(1.0, ci2))
        return ba_unthinned2 * (1.0 - ci2)

    # ---------- Volume ----------
    def yield_predict(
        self,
        age: float,
        tpa: float,
        hd: float,
        ba: float,
        unit: YieldUnit,
        region: Region | None = None,
    ) -> float:
        """
        Predict whole-stand yield using PMRC yield equations.
        Returns per-acre yield in the requested unit.
        """
        if age <= 0 or tpa <= 0 or hd <= 0 or ba <= 0:
            return 0.0

        key = (region or self.region).lower()
        unit = unit.upper()  # type: ignore[assignment]
        valid_units = {"TVOB", "TVIB", "GWOB", "DWIB"}
        if unit not in valid_units:
            raise ValueError(f"Unsupported unit: {unit}")

        if key in ("ucp", "pucp"):
            b0, b1, b2, b3, b4, b5 = self._YIELD_COEFFS_PUCP[unit]
            ln_y = (
                b0
                + b1 * log(hd)
                + b2 * log(ba)
                + b3 * (log(tpa) / age)
                + b4 * (log(hd) / age)
                + b5 * (log(ba) / age)
            )
            return float(exp(ln_y))

        if key == "lcp":
            b0, b1, b2, b3, b4, b5 = self._YIELD_COEFFS_LCP[unit]
            ln_y = (
                b0
                + b1 * log(tpa)
                + b2 * log(hd)
                + b3 * log(ba)
                + b4 * (log(tpa) / age)
                + b5 * (log(ba) / age)
            )
            return float(exp(ln_y))

        raise ValueError(f"Unsupported region: {key}")
    # Deprecated, use yield_predict()
    # def tvob(self, age: float, tpa: float, hd: float, ba: float, region: Region | None = None) -> float:
    #     """
    #     TVOB coefficients implemented for UCP/PUCP only, as in your Python version.
    #     The R script often uses GWOB for products; add as needed.
    #     """
    #     key = (region or self.region).lower()
    #     if age <= 0 or tpa <= 0 or hd <= 0 or ba <= 0:
    #         return 0.0
    #     if key not in ("ucp", "pucp"):
    #         raise ValueError("TVOB implemented for UCP/PUCP only")
    #     b0, b1, b2, b3, b4, b5 = 0.0, 0.268552, 1.368844, -7.466863, 8.934524, 3.553411
    #     lnY = b0 + b1 * log(hd) + b2 * log(ba) + b3 * (log(tpa) / age) + b4 * (log(hd) / age) + b5 * (log(ba) / age)
    #     return float(exp(lnY))

    def product_yield(
        self,
        total_yield: float,
        top_dia: float,
        dbh_lim: float,
        qmd: float,
        tpa: float,
        unit: YieldUnit,
        region: Region | None = None,
    ) -> float:
        """
        PMRC merchantable yield equation.
        Computes merchantable yield for trees >= dbh_lim to top diameter top_dia.
        """
        if total_yield <= 0 or top_dia <= 0 or dbh_lim <= 0 or qmd <= 0 or tpa <= 0:
            return 0.0

        key = (region or self.region).lower()
        unit = unit.upper()  # type: ignore[assignment]
        valid_units = {"TVOB", "TVIB", "GWOB", "DWIB"}
        if unit not in valid_units:
            raise ValueError(f"Unsupported unit: {unit}")

        if key in ("ucp", "pucp"):
            b1, b2, b3, b4, b5 = self._PRODUCT_COEFFS_PUCP[unit]
        elif key == "lcp":
            b1, b2, b3, b4, b5 = self._PRODUCT_COEFFS_LCP[unit]
        else:
            raise ValueError(f"Unsupported region: {key}")

        exponent = (
            b1 * (top_dia / qmd) ** b2
            + b3 * (tpa ** b4) * (dbh_lim / qmd) ** b5
        )
        return float(total_yield * exp(exponent))

    def product_yields(
        self,
        age: float,
        tpa: float,
        hd: float,
        ba: float,
        unit: YieldUnit = "GWOB",
        region: Region | None = None,
        ) -> ProductYields:
        """
        Exclusive product yields using PMRC merchantable equations.
        Defaults to GWOB to match the R workflow.
        """
        total = self.yield_predict(age=age, tpa=tpa, hd=hd, ba=ba, unit=unit, region=region)
        qmd = self.qmd(tpa=tpa, ba=ba)
        specs = self.PINE_PRODUCT_SPECS
        
        saw = self.product_yield(
            total_yield=total,
            top_dia=specs["sawtimber"]["top_dia"],
            dbh_lim=specs["sawtimber"]["dbh_lim"],
            qmd=qmd,
            tpa=tpa,
            unit=unit,
            region=region,
        )

        chip_and_up = self.product_yield(
            total_yield=total,
            top_dia=specs["chip_n_saw"]["top_dia"],
            dbh_lim=specs["chip_n_saw"]["dbh_lim"],
            qmd=qmd,
            tpa=tpa,
            unit=unit,
            region=region,
        )

        pulp_and_up = self.product_yield(
            total_yield=total,
            top_dia=specs["pulpwood"]["top_dia"],
            dbh_lim=specs["pulpwood"]["dbh_lim"],
            qmd=qmd,
            tpa=tpa,
            unit=unit,
            region=region,
        )

        chip = max(0.0, chip_and_up - saw)
        pulp = max(0.0, pulp_and_up - chip_and_up)

        return ProductYields(
            pulpwood=pulp,
            chip_n_saw=chip,
            sawtimber=max(0.0, saw),
        )

    # ---------- Fertilization deltas (add to baseline) ----------
    @staticmethod
    def hd_fert_delta(years_since_treatment: float, N: float, P: float) -> float:
        """Eq. 29 delta HD to add to projected hd."""
        yst = max(0.0, years_since_treatment)
        return max(0.0, (0.00106 * N + 0.2506 * P) * yst * exp(-0.1096 * yst))

    @staticmethod
    def ba_fert_delta(years_since_treatment: float, N: float, P: float) -> float:
        """Eq. 30 delta BA to add to projected ba."""
        yst = max(0.0, years_since_treatment)
        return max(0.0, (0.0121 * N + 1.3639 * P) * yst * exp(-0.2635 * yst))

    # ---------- Diameter ----------
    @staticmethod
    def qmd(tpa: float, ba: float) -> float:
        if tpa <= 0 or ba < 0:
            raise ValueError("tpa must be > 0 and ba >= 0")
        return sqrt((ba / tpa) / 0.005454154)

    @staticmethod
    def ba_from_tpa_qmd(tpa: float, qmd_in: float) -> float:
        """BA ft^2/ac from TPA and QMD in inches."""
        if tpa <= 0 or qmd_in <= 0:
            return 0.0
        return 0.005454154 * tpa * (qmd_in ** 2)

    @staticmethod
    def tpa_from_ba_qmd(ba: float, qmd_in: float) -> float:
        """TPA from BA ft^2/ac and QMD in inches."""
        if ba <= 0 or qmd_in <= 0:
            return 0.0
        return ba / (0.005454154 * (qmd_in ** 2))

    # ----------------- Weibull fit -----------------
    Percentile = Literal[0, 25, 50, 95]

    # Coefficients are (a0, a1, a2) where a2 may be zero when PHWD is not used.
    WEIBULL_PERCENTILE_COEFFS: Dict[Region, Dict[Percentile, Tuple[float, float, float]]] = {
        "ucp": {
            0:  (2.374894, 0.976577, 0.0),
            25: (2.586318, 0.503910, 0.0),
            50: (2.714412, 0.485314, 0.0),
            95: (2.869722, 0.469809, 0.0),
        },
        "pucp": {
            0:  (2.374894, 0.976577, 0.0),
            25: (2.586318, 0.503910, 0.0),
            50: (2.714412, 0.485314, 0.0),
            95: (2.869722, 0.469809, 0.0),
        },
        "lcp": {
            0:  (2.168021, 0.773026, 0.0),
            25: (2.547423, 0.574370, 0.0),
            50: (2.653169, 0.513997, 0.0),
            95: (2.861802, 0.463918, 0.0),
        },
    }

    def predict_diameter_percentiles(
        self,
        ba: float,
        tpa: float,
        region: Region,
        phwd: float = 0.0,
    ) -> Dict[Percentile, float]:
        """
        Predict diameter percentiles (P0, P25, P50, P95) in inches
        from basal area and trees per acre for a given region.

        Uses log-linear models on BA/TPA and optional percent hardwood.
        """
        if ba <= 0.0 or tpa <= 0.0:
            raise ValueError("ba and tpa must be positive to predict percentiles")

        region_key = region.lower()
        if region_key not in self.WEIBULL_PERCENTILE_COEFFS:
            raise ValueError(f"Unsupported region: {region_key}")
        coeffs_by_p = self.WEIBULL_PERCENTILE_COEFFS[region_key]
        ln_ratio = log(ba / tpa)
        result: Dict[Percentile, float] = {}

        for p, (a0, a1, a2) in coeffs_by_p.items():
            ln_px = a0 + a1 * ln_ratio + a2 * phwd
            result[p] = exp(ln_px)

        return result

    # ---------- Weibull helpers ----------

    @staticmethod
    def _weibull_cdf(dbh: float, params: WeibullParams) -> float:
        if dbh <= params.a:
            return 0.0
        x = (dbh - params.a) / params.b
        if x <= 0.0:
            return 0.0
        return 1.0 - exp(-(x ** params.c))

    @staticmethod
    def fit_weibull_from_percentiles(percentiles: Dict[Percentile, float]) -> WeibullParams:
        """
        Fit a three-parameter Weibull distribution to the supplied DBH percentiles.
        """
        required = (0, 25, 50, 95)
        for key in required:
            if key not in percentiles:
                raise ValueError(f"Percentile {key} missing for Weibull fit.")

        probs = {0: 0.0, 25: 0.25, 50: 0.5, 95: 0.95}
        d25 = percentiles[25]
        d50 = percentiles[50]

        def params_for_shape(c: float) -> Tuple[float, float, float] | None:
            if c <= 0.0:
                return None
            k25 = (-log(1.0 - probs[25])) ** (1.0 / c)
            k50 = (-log(1.0 - probs[50])) ** (1.0 / c)
            denom = k50 - k25
            if denom <= 0.0:
                return None
            b = (d50 - d25) / denom
            if b <= 0.0:
                return None
            a = d25 - b * k25
            if a < 0.0:
                a = 0.0
            return a, b, c

        def shape_error(c: float) -> Tuple[float, float, float, float] | None:
            params = params_for_shape(c)
            if not params:
                return None
            a, b, c_val = params
            err = 0.0
            for p in (0, 95):
                prob = probs[p]
                if prob <= 0.0:
                    pred = a
                else:
                    pred = a + b * ((-log(1.0 - prob)) ** (1.0 / c_val))
                err += (pred - percentiles[p]) ** 2
            return err, a, b, c_val

        best: Tuple[float, float, float, float] | None = None
        for shape in np.linspace(0.5, 8.0, 150):
            cand = shape_error(shape)
            if not cand:
                continue
            if best is None or cand[0] < best[0]:
                best = cand

        if best is None:
            raise RuntimeError("Failed to fit Weibull parameters from percentiles.")

        _, a_hat, b_hat, c_hat = best
        return WeibullParams(a=float(a_hat), b=float(max(b_hat, 1e-6)), c=float(max(c_hat, 1e-6)))

    @staticmethod
    def size_class_distribution_from_weibull(
        params: WeibullParams,
        tpa: float,
        dbh_bounds: Sequence[float],
    ) -> Tuple[List[float], List[float]]:
        """
        Compute trees and basal area per size class from Weibull parameters.
        """
        bounds = list(dbh_bounds)
        if len(bounds) < 2:
            raise ValueError("dbh_bounds must include at least two values.")
        if any(bounds[i + 1] <= bounds[i] for i in range(len(bounds) - 1)):
            raise ValueError("dbh_bounds must be strictly increasing.")
        trees: List[float] = []
        basals: List[float] = []
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            f_hi = PMRCModel._weibull_cdf(hi, params)
            f_lo = PMRCModel._weibull_cdf(lo, params)
            prob = max(0.0, min(1.0, f_hi - f_lo))
            trees_per_class = tpa * prob
            trees.append(trees_per_class)
            midpoint = 0.5 * (lo + hi)
            ba_per_tree = 0.005454154 * (midpoint ** 2)
            basals.append(trees_per_class * ba_per_tree)
        return trees, basals

    def diameter_class_distribution(
        self,
        *,
        ba: float,
        tpa: float,
        dbh_bounds: Sequence[float] | None = None,
        region: Region | None = None,
        phwd: float = 0.0,
        scale_ba: bool = True,
    ) -> SizeClassDistribution:
        """
        Build a size class distribution from BA/TPA using PMRC Weibull coefficients.
        
        This is the canonical method for deriving diameter distributions. It:
        1. Predicts diameter percentiles using PMRC regression coefficients
        2. Fits a 3-parameter Weibull to those percentiles
        3. Allocates TPA across size classes based on Weibull CDF
        4. Optionally scales BA to match input (Weibull midpoint approx may not sum exactly)
        
        Args:
            ba: Basal area (ft²/ac)
            tpa: Trees per acre
            dbh_bounds: DBH class boundaries (inches). Defaults to DEFAULT_DBH_BOUNDS.
            region: PMRC region for coefficient lookup
            phwd: Percent hardwood (0-100)
            scale_ba: If True, scale BA array to match input ba
        
        Returns:
            SizeClassDistribution with TPA and BA per class
        """
        if dbh_bounds is None:
            dbh_bounds = DEFAULT_DBH_BOUNDS
        
        region_key: Region = (region or self.region).lower()  # type: ignore[assignment]
        percentiles = self.predict_diameter_percentiles(ba, tpa, region_key, phwd)
        params = self.fit_weibull_from_percentiles(percentiles)
        trees, basals = self.size_class_distribution_from_weibull(params, tpa, dbh_bounds)
        
        tpa_arr = np.asarray(trees, dtype=float)
        ba_arr = np.asarray(basals, dtype=float)
        
        # Scale BA to match actual stand BA (Weibull midpoint approximation may not sum exactly)
        if scale_ba:
            ba_sum = np.sum(ba_arr)
            if ba_sum > 0:
                ba_arr = ba_arr * (ba / ba_sum)
        
        dist = SizeClassDistribution(
            dbh_bounds=np.asarray(dbh_bounds, dtype=float),
            tpa_per_class=tpa_arr,
            ba_per_class=ba_arr,
            weibull_params=params,
            percentiles=percentiles,
        )
        dist.validate()
        return dist


@dataclass
class RowSelectionThinResult:
    """Result of row + selection thinning (R-style PMRC thinning).
    
    Attributes:
        tpa_row_removed: TPA removed by row thinning (25% of pre-thin TPA)
        tpa_select_removed: TPA removed by selection thinning (Eq. 25)
        ba_row_removed: BA removed by row thinning
        ba_select_removed: BA removed by selection thinning
        post_thin_tpa: TPA after thinning
        post_thin_ba: BA after thinning
    """
    tpa_row_removed: float
    tpa_select_removed: float
    ba_row_removed: float
    ba_select_removed: float
    post_thin_tpa: float
    post_thin_ba: float


def tpa_select_remove(
    ba_to_remove: float,
    ba_before: float,
    tpa_before: float,
    tpa_row_remove: float,
) -> float:
    """Solve for TPA removed in selection thinning (Eq. 25 rearranged).
    
    From R script:
        tpa_select_remove <- (((ba_thin*tpa_before - ba_before*tpa_row_remove) /
                              (-ba_before*tpa_row_remove + ba_before*tpa_before))
                              **(2000/2469)) * (tpa_before - tpa_row_remove)
    
    This solves for the TPA to remove via selection thinning given:
    - Total BA to remove
    - Pre-thin BA and TPA
    - TPA already removed by row thinning
    
    Args:
        ba_to_remove: Total BA to remove (ft²/ac)
        ba_before: Pre-thin BA (ft²/ac)
        tpa_before: Pre-thin TPA
        tpa_row_remove: TPA removed by row thinning
    
    Returns:
        TPA to remove via selection thinning
    """
    if ba_to_remove <= 0 or ba_before <= 0 or tpa_before <= 0:
        return 0.0
    
    # Denominator: ba_before * (tpa_before - tpa_row_remove)
    denom = ba_before * tpa_before - ba_before * tpa_row_remove
    if denom <= 0:
        return 0.0
    
    # Numerator: ba_to_remove * tpa_before - ba_before * tpa_row_remove
    numer = ba_to_remove * tpa_before - ba_before * tpa_row_remove
    
    # If numerator is negative or zero, no selection thinning needed
    if numer <= 0:
        return 0.0
    
    # Exponent from R: 2000/2469 ≈ 0.8101 (inverse of 1.2345 from Eq. 25)
    exponent = 2000.0 / 2469.0
    
    ratio = numer / denom
    if ratio <= 0:
        return 0.0
    
    tpa_select = (ratio ** exponent) * (tpa_before - tpa_row_remove)
    return max(0.0, tpa_select)


def thin_row_and_selection(
    tpa_before: float,
    ba_before: float,
    target_residual_ba: float,
    row_fraction: float = 0.25,
) -> RowSelectionThinResult:
    """Apply row + selection thinning to reach target residual BA.
    
    This implements the R-style PMRC thinning model:
    1. Row thinning removes a fixed fraction (default 25%) of TPA uniformly
    2. Selection thinning removes additional trees to reach target BA
    
    The selection thinning TPA is computed using Eq. 25 (rearranged) which
    accounts for the non-linear relationship between TPA and BA removal.
    
    Args:
        tpa_before: Pre-thin TPA
        ba_before: Pre-thin BA (ft²/ac)
        target_residual_ba: Target BA after thinning (ft²/ac)
        row_fraction: Fraction of TPA removed by row thinning (default 0.25 = 4th row)
    
    Returns:
        RowSelectionThinResult with removal details and post-thin state
    """
    if target_residual_ba >= ba_before:
        # No thinning needed
        return RowSelectionThinResult(
            tpa_row_removed=0.0,
            tpa_select_removed=0.0,
            ba_row_removed=0.0,
            ba_select_removed=0.0,
            post_thin_tpa=tpa_before,
            post_thin_ba=ba_before,
        )
    
    ba_to_remove = ba_before - target_residual_ba
    
    # Step 1: Row thinning - removes fixed fraction of TPA
    tpa_row_remove = tpa_before * row_fraction
    # Row thinning removes proportional BA (same fraction as TPA)
    ba_row_remove = ba_before * row_fraction
    
    # Check if row thinning alone exceeds target
    if ba_row_remove >= ba_to_remove:
        # Row thinning alone is sufficient (or more than needed)
        # Scale down row removal to match target
        scale = ba_to_remove / ba_row_remove
        tpa_row_remove *= scale
        ba_row_remove = ba_to_remove
        return RowSelectionThinResult(
            tpa_row_removed=tpa_row_remove,
            tpa_select_removed=0.0,
            ba_row_removed=ba_row_remove,
            ba_select_removed=0.0,
            post_thin_tpa=tpa_before - tpa_row_remove,
            post_thin_ba=ba_before - ba_row_remove,
        )
    
    # Step 2: Selection thinning for remaining BA removal
    ba_select_remove = ba_to_remove - ba_row_remove
    
    # Use Eq. 25 (rearranged) to compute TPA for selection removal
    tpa_select_remove_val = tpa_select_remove(
        ba_to_remove=ba_to_remove,
        ba_before=ba_before,
        tpa_before=tpa_before,
        tpa_row_remove=tpa_row_remove,
    )
    
    # Ensure we don't remove more TPA than available after row thin
    tpa_after_row = tpa_before - tpa_row_remove
    tpa_select_remove_val = min(tpa_select_remove_val, tpa_after_row - 1.0)
    tpa_select_remove_val = max(0.0, tpa_select_remove_val)
    
    post_thin_tpa = tpa_before - tpa_row_remove - tpa_select_remove_val
    post_thin_ba = ba_before - ba_row_remove - ba_select_remove
    
    # Ensure non-negative
    post_thin_tpa = max(1.0, post_thin_tpa)
    post_thin_ba = max(0.0, post_thin_ba)
    
    return RowSelectionThinResult(
        tpa_row_removed=tpa_row_remove,
        tpa_select_removed=tpa_select_remove_val,
        ba_row_removed=ba_row_remove,
        ba_select_removed=ba_select_remove,
        post_thin_tpa=post_thin_tpa,
        post_thin_ba=post_thin_ba,
    )


# --- Weibull-based thinning (commented out in favor of row+selection model) ---
# def thin_smallest_first(
#     dist: SizeClassDistribution,
#     target_ba_removal: float,
# ) -> SizeClassDistribution:
#     """Remove trees from smallest diameter classes first until target BA removal is met.
#     
#     This implements "low thinning" or "thinning from below" where the smallest
#     trees are removed first to reduce competition and favor larger crop trees.
#     
#     Args:
#         dist: Current size class distribution
#         target_ba_removal: BA to remove (ft²/ac)
#     
#     Returns:
#         New SizeClassDistribution after thinning
#     
#     Note:
#         The approximation assumes uniform removal within each class. Trees are
#         removed from the smallest class first, then the next smallest, etc.
#         until the target BA removal is achieved.
#     """
#     if target_ba_removal <= 0:
#         return dist
#     
#     tpa_new = dist.tpa_per_class.copy()
#     ba_new = dist.ba_per_class.copy()
#     ba_remaining = target_ba_removal
#     
#     # Remove from smallest classes first (index 0 is smallest)
#     for i in range(len(ba_new)):
#         if ba_remaining <= 0:
#             break
#         
#         class_ba = ba_new[i]
#         if class_ba <= 0:
#             continue
#         
#         if class_ba <= ba_remaining:
#             # Remove entire class
#             ba_remaining -= class_ba
#             tpa_new[i] = 0.0
#             ba_new[i] = 0.0
#         else:
#             # Partial removal from this class
#             fraction_to_remove = ba_remaining / class_ba
#             tpa_new[i] *= (1.0 - fraction_to_remove)
#             ba_new[i] *= (1.0 - fraction_to_remove)
#             ba_remaining = 0.0
#     
#     return SizeClassDistribution(
#         dbh_bounds=dist.dbh_bounds.copy(),
#         tpa_per_class=tpa_new,
#         ba_per_class=ba_new,
#         weibull_params=dist.weibull_params,
#         percentiles=dist.percentiles,
#     )


def thin_smallest_first(
    dist: SizeClassDistribution,
    target_ba_removal: float,
) -> SizeClassDistribution:
    """Stub for backward compatibility - delegates to row+selection model.
    
    This function is retained for API compatibility but now uses the
    row+selection thinning model internally.
    """
    result = thin_row_and_selection(
        tpa_before=dist.total_tpa,
        ba_before=dist.total_ba,
        target_residual_ba=dist.total_ba - target_ba_removal,
    )
    
    # Scale the per-class arrays proportionally
    if dist.total_tpa > 0:
        tpa_scale = result.post_thin_tpa / dist.total_tpa
    else:
        tpa_scale = 1.0
    if dist.total_ba > 0:
        ba_scale = result.post_thin_ba / dist.total_ba
    else:
        ba_scale = 1.0
    
    return SizeClassDistribution(
        dbh_bounds=dist.dbh_bounds.copy(),
        tpa_per_class=dist.tpa_per_class * tpa_scale,
        ba_per_class=dist.ba_per_class * ba_scale,
        weibull_params=dist.weibull_params,
        percentiles=dist.percentiles,
    )
