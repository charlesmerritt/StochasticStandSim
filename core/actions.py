from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional
from core.growth import Stand, ThinEvent
import numpy as np

# ------------------------------- pricing -------------------------------

@dataclass(frozen=True)
class Prices:
    sawtimber: float = 27.82  # $/ton
    pulpwood:  float =  9.51
    chip:      float = 23.51

@dataclass(frozen=True)
class Costs:
    thin_precomm: float = 139.22  # $/ac
    thin_harvest: float =  87.34
    salvage:      float = 150.00
    fertilize:    float =  90.97
    rx_burn:      float =  31.76
    planting:     float = 150.80

@dataclass(frozen=True)
class Discount:
    rate_annual: float = 0.05

def revenue_from_volume_tons_even_split(total_tons: float, p: Prices) -> float:
    per = max(0.0, total_tons) / 3.0
    return per * (p.sawtimber + p.pulpwood + p.chip)

VOL_TO_TON: float = 0.031  # ft^3 → tons

# ------------------------------ actions -------------------------------

class Action(Enum):
    NOOP          = "noop"
    THINNING      = "thinning"
    FERTILIZATION = "fertilization"
    PLANTING      = "planting"
    HARVESTING    = "harvesting"
    SALVAGING     = "salvaging"
    SELLING       = "selling"
    RXFIRE        = "rxfire"

@dataclass
class ActionOutcome:
    ok: bool
    reason: Optional[str] = None
    cashflow_now: float = 0.0
    npv_at0: float = 0.0
    state_flags: Dict[str, bool] = field(default_factory=dict)

@dataclass
class ActionManager:
    prices: Prices = Prices()
    costs: Costs = Costs()
    disc: Discount = Discount()
    last_thin_age: Optional[float] = None
    harvested: bool = False
    replanted: bool = False
    salvaged_recently: bool = False
    sold: bool = False
    t0_age: float = 0.0

    # --------------- constraints ---------------

    def _check_terminal(self) -> Optional[str]:
        if self.sold:
            return "Stand already sold. No further actions allowed."
        return None

    def _thin_cooldown_ok(self, current_age: float) -> bool:
        return self.last_thin_age is None or (current_age - self.last_thin_age) >= 5.0

    # --------------- discount helper ---------------

    def _npv(self, cashflow_at_age: float, age: float) -> float:
        dt = max(0.0, age - self.t0_age)
        return cashflow_at_age / ((1.0 + self.disc.rate_annual) ** dt)

    # --------------- core apply ---------------

    def apply_thinning(
        self,
        stand: Stand,
        *,
        residual_ba: float,
        min_tonnage_frac: float = 0.25,
        treat_as_precommercial_before_age: float = 10.0,
    ) -> ActionOutcome:
        if (msg := self._check_terminal()) is not None:
            return ActionOutcome(ok=False, reason=msg)

        age = stand.state.age
        if not self._thin_cooldown_ok(age):
            return ActionOutcome(ok=False, reason="Thin cooldown 5 years not satisfied.")

        model = stand.model
        pre_tpa, pre_hd, pre_ba = stand.state.tpa, stand.state.hd, stand.state.ba
        if residual_ba >= pre_ba:
            return ActionOutcome(ok=False, reason="Residual BA must be < current BA.")

        pre_vol = model.tvob(age, pre_tpa, pre_hd, pre_ba, region=stand.cfg.region)

        stand._apply_thin_event(ThinEvent(age=age, residual_ba=residual_ba))
        post_ba = stand.state.ba
        post_vol = model.tvob(age, stand.state.tpa, pre_hd, post_ba, region=stand.cfg.region)

        removed_vol = max(0.0, pre_vol - post_vol)
        if pre_vol > 0.0 and (removed_vol / pre_vol) < min_tonnage_frac:
            return ActionOutcome(ok=False, reason="Thin must remove >= 25% of standing volume.")

        removed_tons = removed_vol * VOL_TO_TON
        revenue = revenue_from_volume_tons_even_split(removed_tons, self.prices)

        if age < treat_as_precommercial_before_age:
            # Assume only 30% of normal revenue is realizable in PCT
            net = 0.3 * revenue - self.costs.thin_precomm
        else:
            net = revenue - self.costs.thin_harvest

        self.last_thin_age = age
        return ActionOutcome(ok=True, cashflow_now=net, npv_at0=self._npv(net, age))

    def apply_fertilization(self, stand: Stand, *, N: float, P: float) -> ActionOutcome:
        if (msg := self._check_terminal()) is not None:
            return ActionOutcome(ok=False, reason=msg)
        age = stand.state.age
        stand.add_fertilization(age=age, N=N, P=P)
        net = -self.costs.fertilize
        return ActionOutcome(ok=True, cashflow_now=net, npv_at0=self._npv(net, age))

    def apply_rxfire(self, stand: Stand) -> ActionOutcome:
        if (msg := self._check_terminal()) is not None:
            return ActionOutcome(ok=False, reason=msg)
        age = stand.state.age
        net = -self.costs.rx_burn
        return ActionOutcome(ok=True, cashflow_now=net, npv_at0=self._npv(net, age))

    def apply_salvage(self, stand: Stand, *, salvage_frac: float) -> ActionOutcome:
        if (msg := self._check_terminal()) is not None:
            return ActionOutcome(ok=False, reason=msg)
        salvage_frac = max(0.0, min(1.0, float(salvage_frac)))
        age = stand.state.age
        model = stand.model
        vol_now = model.tvob(age, stand.state.tpa, stand.state.hd, stand.state.ba, region=stand.cfg.region)
        sale_tons = salvage_frac * vol_now * VOL_TO_TON
        revenue = revenue_from_volume_tons_even_split(sale_tons, self.prices)
        net = revenue - self.costs.salvage
        stand.state.ba = max(0.0, stand.state.ba * (1.0 - salvage_frac))
        self.salvaged_recently = True
        return ActionOutcome(ok=True, cashflow_now=net, npv_at0=self._npv(net, age))

    def apply_harvest(self, stand: Stand) -> ActionOutcome:
        if (msg := self._check_terminal()) is not None:
            return ActionOutcome(ok=False, reason=msg)
        if self.harvested:
            return ActionOutcome(ok=False, reason="Already harvested; plant or sell.")
        age = stand.state.age
        model = stand.model
        vol_now = model.tvob(age, stand.state.tpa, stand.state.hd, stand.state.ba, region=stand.cfg.region)
        tons = vol_now * VOL_TO_TON
        revenue = revenue_from_volume_tons_even_split(tons, self.prices)
        stand.state.age = 0.0
        stand.state.tpa = 0.0
        stand.state.ba = 0.0
        stand.state.tvob = 0.0
        stand.state.si25 = None
        stand.state.ci_anchor_age = None
        stand.state.ci_anchor_value = None
        self.harvested = True
        self.salvaged_recently = True
        net = revenue
        return ActionOutcome(ok=True, cashflow_now=net, npv_at0=self._npv(net, age), state_flags={"harvested": True})

    def apply_planting(self, stand: Stand, *, tpa0: float, si25: float) -> ActionOutcome:
        if (msg := self._check_terminal()) is not None:
            return ActionOutcome(ok=False, reason=msg)
        if not (self.harvested or self.salvaged_recently):
            return ActionOutcome(ok=False, reason="Cannot plant: harvest or salvage before planting.")
        age = stand.state.age
        stand.replant(tpa=float(tpa0), si25=float(si25), initial_age=1.0)
        self.replanted = True
        self.harvested = False
        self.salvaged_recently = False
        net = -self.costs.planting
        return ActionOutcome(ok=True, cashflow_now=net, npv_at0=self._npv(net, age), state_flags={"replanted": True})

    def apply_selling(self, stand: Stand) -> ActionOutcome:
        if (msg := self._check_terminal()) is not None:
            return ActionOutcome(ok=False, reason=msg)
        if not self.harvested:
            return ActionOutcome(ok=False, reason="Cannot sell unless harvested.")
        self.sold = True
        return ActionOutcome(ok=True, cashflow_now=0.0, npv_at0=0.0, state_flags={"sold": True})

# --------------------------- RL dispatcher ---------------------------

def apply_action(
    manager: ActionManager,
    stand: Stand,
    action: Action,
    *,
    now_params: Dict[str, float] | None = None,
) -> ActionOutcome:
    now_params = now_params or {}

    if action == Action.THINNING:
        frac = float(np.clip(now_params.get("residual_frac", 0.6), 0.25, 0.95))
        residual_ba = max(0.0, stand.state.ba * frac)
        return manager.apply_thinning(stand, residual_ba=residual_ba)
    if action == Action.FERTILIZATION:
        return manager.apply_fertilization(stand, N=float(now_params.get("N", 200.0)), P=float(now_params.get("P", 1.0)))
    if action == Action.RXFIRE:
        return manager.apply_rxfire(stand)
    if action == Action.SALVAGING:
        return manager.apply_salvage(stand, salvage_frac=float(now_params.get("salvage_frac", 0.5)))
    if action is Action.SELLING and not manager.harvested:
        return ActionOutcome(ok=False, reason="Cannot sell unless harvested.")
    if action is Action.PLANTING and not manager.harvested:
        return ActionOutcome(ok=False, reason="Planting requires prior harvest.")
    if action is Action.HARVESTING and manager.harvested:
        return ActionOutcome(ok=False, reason="Already harvested; plant or sell.")
    if action == Action.HARVESTING:
        return manager.apply_harvest(stand)
    if action == Action.PLANTING:
        tpa0 = 300.0 + 600.0 * np.clip(now_params.get("tpa0", 0.5), 0.0, 1.0)
        si25 = 45.0  + 30.0 * np.clip(now_params.get("si25", 0.5), 0.0, 1.0)
        return manager.apply_planting(stand, tpa0=tpa0, si25=si25)
    if action == Action.SELLING:
        return manager.apply_selling(stand)
    if action == Action.NOOP:
        return ActionOutcome(ok=True, cashflow_now=0.0, npv_at0=0.0)
    return ActionOutcome(ok=False, reason=f"Unsupported action {action}.")
