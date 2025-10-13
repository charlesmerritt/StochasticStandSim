"""Coefficient tables for the PMRC 1996 growth and yield equations.

The constants were digitised from the PMRC Technical Report 1996-1 and were
previously used in the R based workflow that accompanied this repository.
Storing them in a dedicated module keeps :mod:`pmrc` cleaner and makes the
equations easier to unit test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

from .types import Region


# ---------------------------------------------------------------------------
# Core structural parameters

HEIGHT_SITE_PARAMETERS = {
    "alpha": 0.014452,  # Chapman–Richards growth rate (1/years)
    "m": 0.8216,  # shape parameter
    # Ratio of (1 - exp(-alpha * 25)) used to anchor at site index base age 25
    "ratio_base": 0.30323,
}


TPA_PARAMETERS = {
    "asymptote": 100.0,  # stems per acre
    "b": 0.745339,
    "c": 0.0003425,
    "d": 1.97472,
}


@dataclass(frozen=True)
class BasalAreaCoefficients:
    intercept: float
    inv_age: float
    ln_tpa: float
    ln_hd: float
    ln_tpa_over_age: float
    ln_hd_over_age: float


BA_PREDICT: Mapping[Region, BasalAreaCoefficients] = {
    Region.LOWER_COASTAL_PLAIN: BasalAreaCoefficients(
        intercept=0.0,
        inv_age=-42.689283,
        ln_tpa=0.367244,
        ln_hd=0.659985,
        ln_tpa_over_age=2.012724,
        ln_hd_over_age=7.703502,
    ),
    Region.UPPER_COASTAL_PLAIN: BasalAreaCoefficients(
        intercept=-0.855557,
        inv_age=-36.050347,
        ln_tpa=0.299071,
        ln_hd=0.980246,
        ln_tpa_over_age=3.309212,
        ln_hd_over_age=3.787258,
    ),
    Region.PIEDMONT: BasalAreaCoefficients(
        intercept=-0.904066,
        inv_age=-33.811815,
        ln_tpa=0.321301,
        ln_hd=0.985342,
        ln_tpa_over_age=3.381071,
        ln_hd_over_age=2.548207,
    ),
}


# Projection coefficients reuse the same structure as the prediction
BA_PROJECT = BA_PREDICT


# Hardwood basal area adjustment (Piedmont only)
HARDWOOD_ADJUSTMENT = {
    "coeff": -0.003689,
}


# ---------------------------------------------------------------------------
# Yield models – coefficients keyed by (region, unit)

YieldKey = Tuple[Region, str]


YIELD_COEFFICIENTS: Dict[YieldKey, Tuple[float, ...]] = {
    # (intercept, ln_hd, ln_ba, ln_tpa_over_age, ln_hd_over_age, ln_ba_over_age [, ln_tpa])
    (Region.PIEDMONT, "TVOB"): (0.0, 0.268552, 1.368844, -7.466863, 8.934524, 3.553411, 0.0),
    (Region.PIEDMONT, "TVIB"): (0.0, 0.350394, 1.263708, -8.608165, 7.193937, 6.309586, 0.0),
    (Region.PIEDMONT, "GWOB"): (-3.818016, 0.430179, 1.276768, -8.088792, 7.428472, 5.554509, 0.0),
    (Region.PIEDMONT, "DWIB"): (-4.98756, 0.446433, 1.348843, -7.757842, 7.857337, 4.222016, 0.0),
    (Region.UPPER_COASTAL_PLAIN, "TVOB"): (0.0, 0.268552, 1.368844, -7.466863, 8.934524, 3.553411, 0.0),
    (Region.UPPER_COASTAL_PLAIN, "TVIB"): (0.0, 0.350394, 1.263708, -8.608165, 7.193937, 6.309586, 0.0),
    (Region.UPPER_COASTAL_PLAIN, "GWOB"): (-3.818016, 0.430179, 1.276768, -8.088792, 7.428472, 5.554509, 0.0),
    (Region.UPPER_COASTAL_PLAIN, "DWIB"): (-4.98756, 0.446433, 1.348843, -7.757842, 7.857337, 4.222016, 0.0),
    (Region.LOWER_COASTAL_PLAIN, "TVOB"): (-1.520877, 1.207586, 0.703405, -5.139064, 0.0, 6.744164, 0.20068),
    (Region.LOWER_COASTAL_PLAIN, "TVIB"): (-2.088857, 1.30377, 0.72695, -5.091474, 0.0, 6.676532, 0.177587),
    (Region.LOWER_COASTAL_PLAIN, "GWOB"): (-5.175922, 1.232028, 0.705769, -5.129853, 0.0, 6.731477, 0.198424),
    (Region.LOWER_COASTAL_PLAIN, "DWIB"): (-6.332502, 1.296629, 0.814967, -4.660198, 0.0, 5.383589, 0.145815),
}


# ---------------------------------------------------------------------------
# Merchantable fractions (Equation 21)

MerchantKey = Tuple[Region, str]


MERCHANTABLE_COEFFICIENTS: Dict[MerchantKey, Tuple[float, float, float, float, float]] = {
    (Region.PIEDMONT, "TVOB"): (-0.982648, -0.748261, -0.111206, 5.784780, 3.99114),
    (Region.PIEDMONT, "TVIB"): (-1.036792, -0.511939, -0.046007, 5.64061, 3.900677),
    (Region.PIEDMONT, "GWOB"): (-1.007482, -0.518057, -0.048385, 5.660573, 3.931373),
    (Region.PIEDMONT, "DWIB"): (-0.934936, -0.590269, -0.065355, 5.596179, 4.111618),
    (Region.UPPER_COASTAL_PLAIN, "TVOB"): (-0.982648, -0.748261, -0.111206, 5.784780, 3.99114),
    (Region.UPPER_COASTAL_PLAIN, "TVIB"): (-1.036792, -0.511939, -0.046007, 5.64061, 3.900677),
    (Region.UPPER_COASTAL_PLAIN, "GWOB"): (-1.007482, -0.518057, -0.048385, 5.660573, 3.931373),
    (Region.UPPER_COASTAL_PLAIN, "DWIB"): (-0.934936, -0.590269, -0.065355, 5.596179, 4.111618),
    (Region.LOWER_COASTAL_PLAIN, "TVOB"): (-1.034486, -5.062955, -0.422892, 6.004646, 3.940848),
    (Region.LOWER_COASTAL_PLAIN, "TVIB"): (-1.105225, -4.459271, -0.404057, 5.984225, 3.878664),
    (Region.LOWER_COASTAL_PLAIN, "GWOB"): (-1.064132, -5.048319, -0.422117, 5.991728, 3.818683),
    (Region.LOWER_COASTAL_PLAIN, "DWIB"): (-0.963185, -4.540672, -0.406561, 5.962867, 4.054202),
}


# ---------------------------------------------------------------------------
# Relative size projection (Equation 23)

RELATIVE_SIZE_DECAY = {
    Region.PIEDMONT: 0.076472,
    Region.UPPER_COASTAL_PLAIN: 0.076472,
    Region.LOWER_COASTAL_PLAIN: 0.110521,
}


# ---------------------------------------------------------------------------
# Height | DBH model (Equation 24). The coefficients are ordered as
# (intercept, ln_hd, ln_dq, ln_dbh, ln_dbh_over_dq, ln_dbh_over_hd).

HEIGHT_DBH_COEFFICIENTS: Dict[Region, Tuple[float, float, float, float, float, float]] = {
    Region.PIEDMONT: (0.101604, 0.724237, 0.111096, 0.045492, -0.01297, 0.028161),
    Region.UPPER_COASTAL_PLAIN: (0.101604, 0.724237, 0.111096, 0.045492, -0.01297, 0.028161),
    Region.LOWER_COASTAL_PLAIN: (0.16647, 0.646482, 0.156121, 0.052181, -0.018886, 0.03041),
}


# ---------------------------------------------------------------------------
# Diameter percentiles (Equation 22) – coefficients for log form where
# ln(d_p) = a_p + b_p * ln(qmd) + c_p * ln(tpa) + d_p * ln(ba) + e_p * phwd

DIAMETER_PERCENTILE_COEFFICIENTS: Dict[Region, Dict[int, Tuple[float, float, float, float, float]]] = {
    Region.PIEDMONT: {
        25: (-0.104744, 0.980842, -0.044842, 0.063523, -0.002513),
        50: (-0.005976, 0.992658, -0.031691, 0.047221, -0.001742),
        75: (0.102125, 1.008421, -0.022006, 0.028433, -0.001126),
    },
    Region.UPPER_COASTAL_PLAIN: {
        25: (-0.104744, 0.980842, -0.044842, 0.063523, 0.0),
        50: (-0.005976, 0.992658, -0.031691, 0.047221, 0.0),
        75: (0.102125, 1.008421, -0.022006, 0.028433, 0.0),
    },
    Region.LOWER_COASTAL_PLAIN: {
        25: (-0.108207, 0.986185, -0.048185, 0.058668, 0.0),
        50: (-0.00781, 1.001821, -0.033663, 0.042102, 0.0),
        75: (0.101312, 1.016942, -0.021442, 0.025998, 0.0),
    },
}


# ---------------------------------------------------------------------------
# Fertilisation response

FERT_HEIGHT = {
    "N": 0.00106,
    "P": 0.2506,
    "k": 0.1096,
}

FERT_BA = {
    "N": 0.0121,
    "P": 1.3639,
    "k": 0.2635,
}


def percentile_coeffs(region: Region, percentile: int) -> Tuple[float, float, float, float, float]:
    """Helper to fetch percentile coefficients and raise a helpful error."""

    try:
        return DIAMETER_PERCENTILE_COEFFICIENTS[region][percentile]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise KeyError(f"No percentile coefficients for region={region} p={percentile}") from exc

