"""
Actions for the MDP. Thinning, fertilization, planting, and harvesting.
"""

from enum import Enum


class Action(Enum):
    THINNING = "thinning"
    FERTILIZATION = "fertilization"
    PLANTING = "planting"
    HARVESTING = "harvesting"