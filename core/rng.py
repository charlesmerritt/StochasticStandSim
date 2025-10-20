"""
Single place for RNG streams. Exposes a simple helper that returns a random
float with three-decimal precision, optional seeding on call.
"""

from __future__ import annotations

import random

_rng = random.Random()


def rng(seed: int | None = None) -> float:
    """
    Return a pseudo-random float in [0, 1] rounded to three decimals.

    Pass `seed` to reset the underlying generator before drawing.
    """

    if seed is not None:
        _rng.seed(seed)
    return round(_rng.random(), 3)
