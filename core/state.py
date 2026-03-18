"""Stand state definition for stochastic forest simulation.

This module defines the atomic state variables for PMRC-based simulation.
See PLANNING.md Section 3 for the rationale behind this minimal state.

IMPORTANT: HD and SI25 are linked by the Chapman-Richards site index curve.
You should only specify one when creating an initial state - the other is derived.
Use StandState.from_si25() or StandState.from_hd() factory methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Literal

Region = Literal["ucp", "pucp", "lcp"]

# Chapman-Richards parameters for site index curve
_K = 0.014452
_M = 0.8216


def hd_from_si25_at_age(si25: float, age: float) -> float:
    """Compute HD at a given age from SI25 using Chapman-Richards curve.
    
    Args:
        si25: Site index at base age 25 (ft)
        age: Stand age (years)
    
    Returns:
        Dominant height (ft) at the given age
    """
    ratio = (1 - exp(-_K * age)) / (1 - exp(-_K * 25))
    return si25 * (ratio ** _M)


def si25_from_hd_at_age(hd: float, age: float) -> float:
    """Compute SI25 from HD at a given age using Chapman-Richards curve.
    
    Args:
        hd: Dominant height (ft)
        age: Stand age (years)
    
    Returns:
        Site index at base age 25 (ft)
    """
    ratio = (1 - exp(-_K * age)) / (1 - exp(-_K * 25))
    return hd / (ratio ** _M)


@dataclass
class StandState:
    """Atomic state variables for stochastic simulation.
    
    Dynamic variables (projected each step):
        age: Stand age in years
        hd: Dominant height (ft) - derived from si25 at initial age
        tpa: Trees per acre
        ba: Basal area (ft²/ac)
    
    Constant variables (fixed for stand lifetime):
        si25: Site index at base age 25
        region: PMRC coefficient region (ucp/pucp/lcp)
        phwd: Percent hardwood (optional, for Weibull distribution)
    
    IMPORTANT: HD and SI25 are linked. Use factory methods to create states:
        - StandState.from_si25() when you know the site index
        - StandState.from_hd() when you know the dominant height
    
    Derived quantities (volume, products, qmd) are NOT stored here.
    They should be computed on-demand from these atomics.
    """

    # Dynamic (projected each step)
    age: float
    hd: float
    tpa: float
    ba: float
    
    # Constant (fixed for stand lifetime)
    si25: float
    region: Region
    phwd: float = 0.0

    @classmethod
    def from_si25(
        cls,
        age: float,
        si25: float,
        tpa: float,
        ba: float,
        region: Region = "ucp",
        phwd: float = 0.0,
    ) -> StandState:
        """Create state with HD derived from SI25.
        
        This is the preferred way to create an initial state when you know
        the site index. HD is computed to be consistent with the SI25 curve.
        
        Args:
            age: Stand age in years
            si25: Site index at base age 25
            tpa: Trees per acre
            ba: Basal area (ft²/ac)
            region: PMRC coefficient region
            phwd: Percent hardwood
        """
        hd = hd_from_si25_at_age(si25, age)
        return cls(age=age, hd=hd, tpa=tpa, ba=ba, si25=si25, region=region, phwd=phwd)

    @classmethod
    def from_hd(
        cls,
        age: float,
        hd: float,
        tpa: float,
        ba: float,
        region: Region = "ucp",
        phwd: float = 0.0,
    ) -> StandState:
        """Create state with SI25 derived from HD.
        
        Use this when you know the current dominant height and want to
        derive the implied site index.
        
        Args:
            age: Stand age in years
            hd: Dominant height (ft)
            tpa: Trees per acre
            ba: Basal area (ft²/ac)
            region: PMRC coefficient region
            phwd: Percent hardwood
        """
        si25 = si25_from_hd_at_age(hd, age)
        return cls(age=age, hd=hd, tpa=tpa, ba=ba, si25=si25, region=region, phwd=phwd)
