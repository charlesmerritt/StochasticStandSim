from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple, Dict, Sequence, List
from math import exp, log, sqrt
import numpy as np

Region = Literal["ucp", "pucp", "lcp"]
Percentile = Literal[0, 25, 50, 95]


@dataclass
class WeibullParams:
    """Three-parameter Weibull distribution."""

    a: float  # location
    b: float  # scale
    c: float  # shape

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

    def __init__(self, region: Region = "ucp"):
        # Chapman–Richards and site-index parameters
        self.k = 0.014452
        self.m = 0.8216
        self.c = 0.30323
        # Survival parameters
        self.p = 0.745339
        self.g = 0.00034252
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
        if tpa1 <= self.min_tpa_asymptote:
            return tpa1
        p, g, r = self.p, self.g, self.r
        return 100.0 + (((tpa1 - 100.0) ** -p) + (g ** 2) * si25 * ((age2 ** r) - (age1 ** r))) ** (-(1.0 / p))

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
        beta = self.beta_ucp if key in ("ucp", "pucp") else self.beta_lcp
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
    def tvob(self, age: float, tpa: float, hd: float, ba: float, region: Region | None = None) -> float:
        """
        TVOB coefficients implemented for UCP/PUCP only, as in your Python version.
        The R script often uses GWOB for products; add as needed.
        """
        key = (region or self.region).lower()
        if age <= 0 or tpa <= 0 or hd <= 0 or ba <= 0:
            return 0.0
        if key not in ("ucp", "pucp"):
            raise ValueError("TVOB implemented for UCP/PUCP only")
        b0, b1, b2, b3, b4, b5 = 0.0, 0.268552, 1.368844, -7.466863, 8.934524, 3.553411
        lnY = b0 + b1 * log(hd) + b2 * log(ba) + b3 * (log(tpa) / age) + b4 * (log(hd) / age) + b5 * (log(ba) / age)
        return float(exp(lnY))

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

        coeffs_by_p = self.WEIBULL_PERCENTILE_COEFFS[region]
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
        dbh_bounds: Sequence[float],
        region: Region | None = None,
        phwd: float = 0.0,
    ) -> Tuple[Dict[Percentile, float], WeibullParams, List[float], List[float]]:
        """
        Convenience helper to derive Weibull-driven size classes from BA/TPA.
        """
        region_key = (region or self.region).lower()
        percentiles = self.predict_diameter_percentiles(ba, tpa, region_key, phwd)
        params = self.fit_weibull_from_percentiles(percentiles)
        trees, basals = self.size_class_distribution_from_weibull(params, tpa, dbh_bounds)
        return percentiles, params, trees, basals

