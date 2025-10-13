from __future__ import annotations

from enum import Enum, auto


class Region(Enum):
    """Geographic regions used by the PMRC models."""

    LOWER_COASTAL_PLAIN = auto()
    UPPER_COASTAL_PLAIN = auto()
    PIEDMONT = auto()


class ProductClass(Enum):
    """Merchandising classes defined by DBH and top diameter limits."""

    PULPWOOD = auto()
    CHIP_N_SAW = auto()
    SAWTIMBER = auto()

