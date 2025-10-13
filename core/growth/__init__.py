"""Growth model package initialisation."""

from .pmrc import PMRC1996, PMRCGrowth, StandState, get_growth, register_equations
from .types import ProductClass, Region

# Register the default PMRC 1996 equations for convenience when wiring up the
# simulator. The registry keeps a singleton instance so repeated lookups are
# cheap and deterministic.
register_equations("pmrc1996", PMRC1996())

__all__ = [
    "PMRC1996",
    "PMRCGrowth",
    "StandState",
    "Region",
    "ProductClass",
    "register_equations",
    "get_growth",
]

