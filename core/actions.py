from __future__ import annotations
from dataclasses import dataclass
from .types import ActionName


@dataclass(slots=True)
class Action:
    name: ActionName
    thin_frac: float = 0.0 # 0..1 of volume removed when name=="thin"


ACTIONS: tuple[Action, ...] = (
Action("noop"),
Action("thin", thin_frac=0.25),
Action("fertilize"),
Action("pesticide"),
Action("rx_fire"),
Action("harvest_replant"),
)