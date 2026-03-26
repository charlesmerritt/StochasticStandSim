"""Stand Table Projection (STP) system for loblolly pine.

Implements the PMRC 1996 Stand Table Projection methodology as described in:
    Kinane, S. (2025). Stand Table Projection. PMRC Technical Report 1996-1.

This is a separate baseline growth model from the aggregate PMRC model.
Instead of projecting only stand-level aggregates (HD, TPA, BA), STP tracks
the diameter class distribution explicitly through time.

Mathematical overview:
    1. Project stand-level HD, TPA, BA using PMRC whole-stand equations
       (Chapman-Richards HD, survival TPA, and BA projection; TR 1996-1).
    2. Allocate stand-level mortality across diameter classes using inverse-BA
       weighting (Eqs. 4-5 and 4-6, PMRC TR 2004-4): smaller trees die at
       proportionally higher rates.
    3. Project each class midpoint DBH using PMRC TR 1996-1 Eq. 23:
           ba_midpoint2_i = n2i * relative_size_i ^ ( (age2/age1)^beta )
       where beta = -0.2277 (PUCP/UCP) or -0.0525 (LCP).
    4. Reallocate surviving trees from projected diameter classes (with
       non-integer midpoints) into traditional 1-inch DBH classes using
       a uniform-distribution overlap proportionality rule.
    5. Compute per-class BA and individual tree heights using the PMRC
       height-diameter model (Eq. 24, TR 1996-1).

Active assumptions:
    - Trees are uniformly distributed within each ±0.5 inch DBH class.
    - Mortality is inversely proportional to class BA per tree (smaller → more).
    - UCP region uses PUCP coefficients for all STP-specific equations.
    - Projected class limits are ±0.5 inch around the new midpoint.
    - Projected stand TPA and BA are constrained to PMRC whole-stand output.
    - SI25 is derived from HD at the current age via Chapman-Richards curve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, log, sqrt
from typing import Sequence

import numpy as np

from core.pmrc_model import PMRCModel
from core.state import Region, si25_from_hd_at_age

# Basal area conversion constant: BA (ft²/tree) = _BA_CONST * DBH² (in²)
# Derivation: π/4 * (D/12)² = π/576 ≈ 0.005454154
_BA_CONST = 0.005454154

# Module-level PMRC model instance (stateless; all methods accept region arg)
_PMRC = PMRCModel()

# DBH midpoint projection exponents (PMRC TR 1996-1, Eq. 23)
# The exponent applied to (age2/age1) before raising relative_size to that power.
_MIDPOINT_BETA: dict[str, float] = {
    "pucp": -0.2277,
    "ucp":  -0.2277,  # UCP uses PUCP coefficients
    "lcp":  -0.0525,
}

# Height-diameter equation coefficients (PMRC TR 1996-1, Eq. 24)
# Tuple: (c1, c2, c3) for H_i = HD * c1 * (1 - c2 * exp(-c3 * (DBH_i / QMD)))
_HD_COEFFS: dict[str, tuple[float, float, float]] = {
    "pucp": (1.179240, 0.878092, 1.618723),
    "ucp":  (1.179240, 0.878092, 1.618723),  # UCP uses PUCP coefficients
    "lcp":  (1.185552, 0.949316, 1.710774),
}


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DiameterClass:
    """One 1-inch DBH class entry in a stand table.

    Attributes:
        dbh: DBH class midpoint (inches), e.g. 8.0 for the 8-inch class.
        tpa: Trees per acre in this class.
        ba:  Basal area (ft²/ac) = _BA_CONST * dbh² * tpa.
        dht: Individual tree height (ft) from H-D equation; 0.0 if not yet
             computed (e.g. when constructing from raw inventory data).
    """
    dbh: float
    tpa: float
    ba:  float
    dht: float = 0.0


@dataclass
class StandTable:
    """Stand table: ordered 1-inch DBH classes at a given stand age.

    Stores the full size-class distribution plus the stand-level dominant
    height required for forward projections and the H-D equation.

    Attributes:
        age:     Stand age (years).
        hd:      Dominant height (ft) at *age*.
        region:  PMRC coefficient region ('ucp', 'pucp', or 'lcp').
        classes: List of DiameterClass ordered by ascending DBH.
    """
    age:     float
    hd:      float
    region:  Region
    classes: list[DiameterClass] = field(default_factory=list)

    # ── Derived stand-level attributes ──────────────────────────────────────

    @property
    def si25(self) -> float:
        """Site index at base age 25, derived via Chapman-Richards curve."""
        return si25_from_hd_at_age(self.hd, self.age)

    @property
    def tpa(self) -> float:
        """Total trees per acre (sum across all classes)."""
        return sum(c.tpa for c in self.classes)

    @property
    def ba(self) -> float:
        """Total basal area ft²/ac (sum across all classes)."""
        return sum(c.ba for c in self.classes)

    @property
    def qmd(self) -> float:
        """Quadratic mean diameter (inches) derived from stand TPA and BA."""
        t, b = self.tpa, self.ba
        if t <= 0:
            return 0.0
        return sqrt((b / t) / _BA_CONST)

    # ── Constructor helpers ──────────────────────────────────────────────────

    @classmethod
    def from_arrays(
        cls,
        age: float,
        hd: float,
        region: Region,
        dbh_midpoints: Sequence[float],
        tpa_per_class: Sequence[float],
    ) -> StandTable:
        """Construct a StandTable from parallel DBH-midpoint and TPA arrays.

        BA per class is computed as ``_BA_CONST * dbh² * tpa``.  Individual
        tree heights are left at 0.0 (call :func:`compute_heights` to fill in).

        Args:
            age:           Stand age (years).
            hd:            Dominant height (ft).
            region:        PMRC region.
            dbh_midpoints: DBH class midpoints (inches), e.g. [6, 7, 8, …].
            tpa_per_class: Trees per acre in each class (parallel to
                           *dbh_midpoints*).

        Raises:
            ValueError: If the two arrays differ in length or any TPA is
                        negative.
        """
        dbh_seq = list(dbh_midpoints)
        tpa_seq = list(tpa_per_class)
        if len(dbh_seq) != len(tpa_seq):
            raise ValueError("dbh_midpoints and tpa_per_class must have equal length")
        if any(t < 0 for t in tpa_seq):
            raise ValueError("tpa_per_class must be non-negative")

        classes = [
            DiameterClass(
                dbh=float(d),
                tpa=float(t),
                ba=_BA_CONST * float(d) ** 2 * float(t),
            )
            for d, t in zip(dbh_seq, tpa_seq)
        ]
        return cls(age=age, hd=hd, region=region, classes=classes)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers – step-by-step STP algorithm
# ─────────────────────────────────────────────────────────────────────────────

def _project_stand_level(
    hd1:      float,
    tpa1:     float,
    ba1:      float,
    si25:     float,
    age1:     float,
    age2:     float,
    region:   Region,
) -> tuple[float, float, float]:
    """Project stand-level HD, TPA, and BA from age1 to age2.

    Delegates entirely to :class:`~core.pmrc_model.PMRCModel`:
      - HD via Chapman-Richards (Eq. 12, TR 1996-1)
      - TPA via self-thinning survival equation
      - BA via the region-specific BA projection (Eqs. 16-17, TR 1996-1)

    Args:
        hd1:    Dominant height at age1 (ft).
        tpa1:   Total TPA at age1.
        ba1:    Total BA at age1 (ft²/ac).
        si25:   Site index at base age 25 (ft).
        age1:   Current stand age (years).
        age2:   Projection age (years); must be ≥ age1.
        region: PMRC coefficient region.

    Returns:
        Tuple ``(hd2, tpa2, ba2)`` — projected stand-level values at age2.
    """
    hd2  = _PMRC.hd_project(age1=age1, hd1=hd1, age2=age2)
    tpa2 = _PMRC.tpa_project(tpa1=tpa1, si25=si25, age1=age1, age2=age2)
    ba2  = _PMRC.ba_project(
        age1=age1, tpa1=tpa1, tpa2=tpa2,
        ba1=ba1, hd1=hd1, hd2=hd2, age2=age2,
        region=region,
    )
    return hd2, tpa2, ba2


def _allocate_mortality(
    dbh:        np.ndarray,
    tpa:        np.ndarray,
    tpa2_total: float,
) -> np.ndarray:
    """Allocate stand-level mortality across diameter classes.

    Implements PMRC TR 2004-4 equations 4-5 and 4-6:

    **Eq. 4-5** — Inverse-BA mortality weight for class *i*::

        pi_dclass_i = (1 / b1i) / Σ_j (1 / b1j)

    This is the conditional probability that a dead tree belongs to class *i*
    given *only* the inverse-BA weights (smaller trees die more).

    **Eq. 4-6** — Mortality probability weighted by class TPA::

        pi_i = (n1i * pi_dclass_i) / Σ_j (n1j * pi_dclass_j)

    The number of surviving trees per class is then::

        n2i = n1i − total_mortality × pi_i

    where ``total_mortality = tpa1_total − tpa2_total``.

    Args:
        dbh:        DBH class midpoints (inches), shape ``(K,)``.
        tpa:        TPA in each class at time 1, shape ``(K,)``.
        tpa2_total: Total projected TPA at time 2 (scalar).

    Returns:
        ``n2i`` — TPA in each class at time 2, clipped to ≥ 0, shape ``(K,)``.
    """
    b1i     = _BA_CONST * dbh ** 2  # per-tree BA for each class
    inv_b1i = 1.0 / b1i

    # Eq. 4-5: inverse-BA class weight
    pi_dclass = inv_b1i / inv_b1i.sum()

    # Eq. 4-6: TPA-weighted conditional mortality probability
    n1i_pi = tpa * pi_dclass
    pi_i   = n1i_pi / n1i_pi.sum()

    # Total stand mortality (clipped so mortality ≥ 0)
    mortality = max(0.0, float(tpa.sum()) - tpa2_total)

    n2i = tpa - mortality * pi_i
    return np.maximum(n2i, 0.0)


def _project_midpoints(
    dbh:       np.ndarray,
    n2i:       np.ndarray,
    ba2_total: float,
    age1:      float,
    age2:      float,
    region:    Region,
) -> tuple[np.ndarray, np.ndarray]:
    """Project DBH class midpoints to age2 and allocate projected BA.

    Implements PMRC TR 1996-1 **Eq. 23** for the midpoint projection.

    **Step 1** — Compute the average BA of a surviving tree at time 1::

        avg_ba_survivor = Σ_i (b1i * n2i) / Σ_i n2i

    **Step 2** — Relative size of each class::

        relative_size_i = b1i / avg_ba_survivor

    **Step 3** — Projected unnormalised midpoint BA (Eq. 23)::

        ba_midpoint2_i = n2i × relative_size_i ^ ( (age2/age1)^beta )

    where ``beta = −0.2277`` (PUCP/UCP) or ``−0.0525`` (LCP).

    Note: the exponent ``(age2/age1)^beta`` is evaluated *first* (R's ``^``
    operator is right-associative), producing a scalar that is then used as
    the power for ``relative_size_i``.

    **Step 4** — Scale to match whole-stand BA2::

        BA_class_2_i = (ba_midpoint2_i / Σ ba_midpoint2) × ba2_total
        b2i          = BA_class_2_i / n2i
        d2i          = sqrt(b2i / _BA_CONST)

    Args:
        dbh:       DBH class midpoints at time 1 (inches), shape ``(K,)``.
        n2i:       Surviving TPA per class, shape ``(K,)``.
        ba2_total: Projected total BA at time 2 (ft²/ac).
        age1:      Current stand age (years).
        age2:      Projection age (years).
        region:    PMRC coefficient region.

    Returns:
        ``(d2i, BA_class_2)`` — projected DBH midpoints (inches) and
        projected BA per class (ft²/ac), each of shape ``(K,)``.
    """
    region_key = region.lower()
    beta       = _MIDPOINT_BETA.get(region_key, _MIDPOINT_BETA["pucp"])

    # Per-tree BA for each class at time 1
    b1i = _BA_CONST * dbh ** 2

    # BA of surviving trees in each class
    BA_class_1 = b1i * n2i
    sum_n2i    = float(n2i.sum())

    if sum_n2i <= 0:
        # No survivors – return zeros
        zeros = np.zeros_like(dbh)
        return zeros, zeros

    avg_ba_survivor = float(BA_class_1.sum()) / sum_n2i
    relative_size   = b1i / avg_ba_survivor

    # Eq. 23: exponent computed right-to-left (R `^` precedence)
    age_ratio    = age2 / age1
    scalar_exp   = age_ratio ** beta          # e.g. (35/22)^(-0.2277) ≈ 0.900
    ba_midpoint2 = n2i * (relative_size ** scalar_exp)

    sum_ba_mid = float(ba_midpoint2.sum())

    if sum_ba_mid > 0:
        BA_class_2 = (ba_midpoint2 / sum_ba_mid) * ba2_total
    else:
        BA_class_2 = np.zeros_like(ba_midpoint2)

    # Per-tree BA → projected DBH midpoint
    b2i = np.where(n2i > 0, BA_class_2 / n2i, 0.0)
    d2i = np.sqrt(np.maximum(b2i / _BA_CONST, 0.0))

    return d2i, BA_class_2


def _reallocate_to_traditional_classes(
    d2i:     np.ndarray,
    n2i:     np.ndarray,
    min_dbh: int = 5,
) -> dict[int, float]:
    """Reallocate projected diameter classes into traditional 1-inch classes.

    Trees within each projected class are assumed **uniformly distributed**
    over ``[d2i − 0.5, d2i + 0.5]``.  The proportion transferred into each
    traditional 1-inch class ``[k − 0.5, k + 0.5]`` equals the overlap
    length divided by the projected class width (always 1 inch)::

        proportion(i → k) = overlap([d2i_i ± 0.5], [k ± 0.5])

    Args:
        d2i:     Projected DBH midpoints (inches), shape ``(K,)``.
        n2i:     TPA in each projected class, shape ``(K,)``.
        min_dbh: Minimum traditional DBH class midpoint to include in output
                 (default 5 inches).

    Returns:
        Dict mapping traditional DBH midpoint (int) to TPA.  Classes with
        zero allocation are excluded.
    """
    active = n2i > 0
    if not np.any(active):
        return {}

    d_active = d2i[active]

    trad_min = max(min_dbh, int(np.floor(float(d_active.min()) - 0.5)))
    trad_max = int(np.ceil(float(d_active.max()) + 0.5)) + 1

    result: dict[int, float] = {}

    for k in range(trad_min, trad_max + 1):
        trad_lo = k - 0.5
        trad_hi = k + 0.5
        trees_k = 0.0

        for j in range(len(d2i)):
            if n2i[j] <= 0:
                continue

            proj_lo = d2i[j] - 0.5
            proj_hi = d2i[j] + 0.5

            overlap_start = max(trad_lo, proj_lo)
            overlap_end   = min(trad_hi, proj_hi)

            if overlap_end <= overlap_start:
                continue

            proportion = (overlap_end - overlap_start) / (proj_hi - proj_lo)
            trees_k   += proportion * n2i[j]

        if trees_k > 0.0:
            result[k] = trees_k

    return result


def _height_diameter(
    dbh_i:  float,
    hd:     float,
    qmd:    float,
    region: Region,
) -> float:
    """Individual tree height from the PMRC H-D model (TR 1996-1, Eq. 24).

    PUCP/UCP::

        H_i = HD × 1.179240 × (1 − 0.878092 × exp(−1.618723 × (DBH_i / QMD)))

    LCP::

        H_i = HD × 1.185552 × (1 − 0.949316 × exp(−1.710774 × (DBH_i / QMD)))

    Args:
        dbh_i:  DBH class midpoint (inches).
        hd:     Stand dominant height (ft).
        qmd:    Quadratic mean diameter (inches).
        region: PMRC coefficient region.

    Returns:
        Individual tree height (ft).  Returns 0.0 if *qmd* ≤ 0.
    """
    if qmd <= 0:
        return 0.0
    c1, c2, c3 = _HD_COEFFS.get(region.lower(), _HD_COEFFS["pucp"])
    return hd * c1 * (1.0 - c2 * exp(-c3 * (dbh_i / qmd)))


def compute_heights(stand: StandTable) -> StandTable:
    """Return a new StandTable with individual tree heights populated.

    Uses the stand's projected QMD (computed from the class-level TPA and BA)
    and dominant height to evaluate Eq. 24 for each DBH class midpoint.

    Args:
        stand: StandTable with classes whose ``dht`` may be 0.0.

    Returns:
        New StandTable with ``dht`` filled for every class that has ``tpa > 0``.
    """
    qmd = stand.qmd
    updated = [
        DiameterClass(
            dbh=c.dbh,
            tpa=c.tpa,
            ba=c.ba,
            dht=_height_diameter(c.dbh, stand.hd, qmd, stand.region)
            if c.tpa > 0 else 0.0,
        )
        for c in stand.classes
    ]
    return StandTable(age=stand.age, hd=stand.hd, region=stand.region, classes=updated)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def project_stand_table(
    stand: StandTable,
    age2:  float,
) -> StandTable:
    """Project a stand table from its current age to *age2*.

    This is the primary entry point for the Stand Table Projection system.
    The algorithm proceeds in five steps (see module docstring for details):

    1. Project stand-level HD, TPA, BA via PMRC whole-stand equations.
    2. Allocate mortality across diameter classes (inverse-BA weighting).
    3. Project each diameter class midpoint using PMRC Eq. 23.
    4. Reallocate trees into traditional 1-inch DBH classes via overlap.
    5. Compute per-class BA and individual tree heights (Eq. 24).

    The returned stand table's TPA matches the PMRC-projected TPA exactly.
    The BA may differ very slightly from the PMRC-projected BA because the
    traditional-class BA is recomputed as ``_BA_CONST × DBH² × TPA`` using
    integer midpoints rather than the continuously projected midpoints.

    Args:
        stand: Current stand table.  Must contain at least one class with
               ``tpa > 0``.
        age2:  Target stand age (years).  Must be ≥ ``stand.age``.

    Returns:
        Projected :class:`StandTable` at *age2*, including per-class heights.

    Raises:
        ValueError: If *age2* < ``stand.age`` or stand TPA is zero.
    """
    if age2 < stand.age:
        raise ValueError(
            f"age2 ({age2}) must be >= current stand age ({stand.age})"
        )
    if age2 == stand.age:
        return stand

    age1   = stand.age
    hd1    = stand.hd
    region = stand.region
    si25   = stand.si25

    dbh  = np.array([c.dbh for c in stand.classes], dtype=float)
    tpa1 = np.array([c.tpa for c in stand.classes], dtype=float)

    tpa1_total = float(tpa1.sum())
    ba1_total  = stand.ba

    if tpa1_total <= 0:
        raise ValueError("Stand has no trees (total TPA = 0)")

    # ── Step 1: Project stand-level variables ─────────────────────────────
    hd2, tpa2_total, ba2_total = _project_stand_level(
        hd1=hd1, tpa1=tpa1_total, ba1=ba1_total,
        si25=si25, age1=age1, age2=age2, region=region,
    )

    # ── Step 2: Allocate mortality across diameter classes ─────────────────
    n2i = _allocate_mortality(dbh=dbh, tpa=tpa1, tpa2_total=tpa2_total)

    # ── Step 3: Project diameter class midpoints ───────────────────────────
    d2i, _ba_class_2 = _project_midpoints(
        dbh=dbh, n2i=n2i, ba2_total=ba2_total,
        age1=age1, age2=age2, region=region,
    )

    # ── Step 4: Reallocate to traditional 1-inch DBH classes ──────────────
    trad_dist = _reallocate_to_traditional_classes(d2i=d2i, n2i=n2i)

    # ── Step 5: Build projected stand table with BA and heights ───────────
    # QMD is derived from the redistributed stand table (not PMRC-projected
    # directly) so that H-D heights are consistent with the actual table BA.
    tpa_sum = sum(trad_dist.values())
    ba_sum  = sum(
        _BA_CONST * k ** 2 * t for k, t in trad_dist.items()
    )
    qmd2 = sqrt((ba_sum / tpa_sum) / _BA_CONST) if tpa_sum > 0 else 0.0

    projected_classes: list[DiameterClass] = []
    for k, tpa_k in sorted(trad_dist.items()):
        if tpa_k <= 0:
            continue
        ba_k  = _BA_CONST * k ** 2 * tpa_k
        dht_k = _height_diameter(
            dbh_i=float(k), hd=hd2, qmd=qmd2, region=region
        )
        projected_classes.append(DiameterClass(
            dbh=float(k), tpa=tpa_k, ba=ba_k, dht=dht_k,
        ))

    return StandTable(age=age2, hd=hd2, region=region, classes=projected_classes)


def multi_step_projection(
    stand:      StandTable,
    target_age: float,
    step:       float = 1.0,
) -> list[StandTable]:
    """Project a stand table forward in annual (or custom) steps.

    Applies :func:`project_stand_table` iteratively, so each intermediate
    stand table serves as input to the next projection.  This preserves the
    diameter distribution through the full rotation rather than projecting
    in a single large jump.

    Args:
        stand:      Initial stand table.
        target_age: Final projection age (years).
        step:       Step size in years (default 1.0).  The final step may be
                    smaller than *step* if needed to land exactly on
                    *target_age*.

    Returns:
        List of :class:`StandTable` objects (one per step), **not** including
        the initial *stand*.  The last element is at *target_age*.

    Raises:
        ValueError: If *target_age* ≤ ``stand.age`` or *step* ≤ 0.
    """
    if target_age <= stand.age:
        raise ValueError(
            f"target_age ({target_age}) must be > current age ({stand.age})"
        )
    if step <= 0:
        raise ValueError("step must be positive")

    results: list[StandTable] = []
    current = stand

    while current.age < target_age:
        next_age = min(current.age + step, target_age)
        current  = project_stand_table(current, next_age)
        results.append(current)

    return results
