from typing import Literal, Tuple, Dict
from math import exp, log, sqrt

Region = Literal["ucp", "pucp", "lcp"]

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
        self.k = 0.014452
        self.m = 0.8216
        self.c = 0.30323
        self.p = 0.745339
        self.g = 0.00034252
        self.r = 1.97472
        self.region = region.lower()
        self.beta_ucp = 0.076472
        self.beta_lcp = 0.110521
        self.min_tpa_asymptote = 100.0

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
        return hd * (self.c / (1.0 - exp(-self.k))) ** self.m

    def hd_from_si(self, si25: float, form: str = "projection") -> float:
        """
        Eq. 11 (PS80) and Eq. 14 (projection-consistent).
        Convert SI25 to dominant height at age A used in the converter.
        """
        if form.lower() == "ps80":
            c, k, e = 0.7476, 0.05507, 1.435
            return si25 * (c / (1.0 - exp(-k))) ** -e
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