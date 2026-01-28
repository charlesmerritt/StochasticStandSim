from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal, Tuple
from enum import Enum
import math

from core.pmrc_model import PMRCModel
from core.disturbances import DisturbanceEvent
from core.stochastic_stand import StochasticPMRC, StandState as StochStandState
import numpy as np


class Region(str, Enum):
    UCP = "ucp"
    PUCP = "pucp"
    LCP = "lcp"


class SIForm(str, Enum):
    PROJECTION = "projection"
    PS80 = "ps80"


# --------------------------- Data models ---------------------------

@dataclass
class StandConfig:
    region: Region | str = Region.UCP
    base_age: int = 25
    hold_tpa_below_asymptote: bool = True  # hold when TPA <= 100
    tpa_geometric_decay: Optional[float] = .9999
    tpa_from_ba_policy: Literal["qmd_constant", "gamma"] = "qmd_constant"
    tpa_from_ba_gamma: float = 1.0  # used when policy == "gamma"
    use_stochastic_growth: bool = False
    hd_noise: float = 0.02
    tpa_noise: float = 0.02
    ba_noise: float = 0.03
    tvob_noise: float = 0.03


@dataclass
class FertEvent:
    age: float
    N: float
    P: float


@dataclass
class ThinEvent:
    age: float
    # Target residual BA at the thin age, like R's thin*_residBA
    residual_ba: float
    row_fraction: float = 0.25
    residual_fraction: Optional[float] = None


@dataclass
class StandState:
    age: float
    tpa: float
    si25: Optional[float] = None
    hd: float = 0.0
    ba: float = 0.0
    region: Region | str = Region.UCP
    ba_unthinned: float = 0.0  # what BA would be without thinning
    tvob: float = 0.0
    ci: float = 0.0  # competition index used only during recovery
    # CI recovery anchor after a thin:
    ci_anchor_age: Optional[float] = None
    ci_anchor_value: Optional[float] = None
    # bookkeeping
    history: Dict[str, float] = field(default_factory=dict)
    fert_hd_applied: float = 0.0
    fert_ba_applied: float = 0.0
    disturbance_flag: Optional[str] = None


# --------------------------- Stand engine ---------------------------

class Stand:
    def __init__(self, init: StandState, cfg: StandConfig | None = None):
        self.cfg = cfg or StandConfig()
        # allow init.region to override config region
        region_source = init.region if getattr(init, "region", None) is not None else self.cfg.region
        region_value = region_source.value if isinstance(region_source, Region) else region_source
        region_value = str(region_value).lower()
        object.__setattr__(self.cfg, "region", region_value)
        assert 0.0 < self.cfg.tpa_geometric_decay <= 1.0
        # normalize region in both places
        self.model = PMRCModel(region=region_value)
        self._rng = np.random.default_rng()
        self._stochastic: StochasticPMRC | None = None
        if self.cfg.use_stochastic_growth:
            # Map existing noise knobs onto the stochastic PMRC wrapper
            sigma_log_hd = self.cfg.hd_noise if self.cfg.hd_noise > 0 else None
            self._stochastic = StochasticPMRC(
                self.model,
                sigma_log_ba=max(1e-6, self.cfg.ba_noise),
                sigma_tpa=max(1e-6, self.cfg.tpa_noise * 1000.0),
                sigma_log_hd=sigma_log_hd,
                use_binomial_tpa=True,
            )
        
        # Require one of {hd, si25}
        has_hd = init.hd is not None and init.hd > 0
        has_si = init.si25 is not None and init.si25 > 0
        if not (has_hd or has_si):
            raise ValueError("Stand init requires hd > 0 or si25 > 0.")

        # Resolve the missing one using the class converters
        if has_si and not has_hd:
            si25 = float(init.si25)
            hd = float(self.model.hd_from_si(si25, form="projection"))
        elif has_hd and not has_si:
            hd = float(init.hd)
            si25 = float(self.model.si_from_hd(hd, form="projection"))
        else:
            # both provided: enforce consistency
            si25 = float(init.si25)
            hd = float(self.model.hd_from_si(si25, form="projection"))

        # BA default from predictors if missing
        ba = float(init.ba) if init.ba and init.ba > 0 else float(
            self.model.ba_predict(age=init.age, tpa=init.tpa, hd=hd, region=self.cfg.region)
        )
        self.state = StandState(
            age=float(init.age),
            tpa=float(init.tpa),
            region=region_value,
            si25=float(si25),
            hd=float(hd),
            ba=float(ba),
            ba_unthinned=float(ba),  # initially same as ba
            tvob=float(init.tvob),
            ci=float(init.ci),
            fert_hd_applied=0.0,
            fert_ba_applied=0.0,
            disturbance_flag=init.disturbance_flag,
        )
        # bookkeeping
        self._fert_hd_applied = 0.0
        self._fert_ba_applied = 0.0
        # schedules
        self._fert: List[FertEvent] = []
        self._thin: List[ThinEvent] = []

    # ---------------- Schedules ----------------

    def add_fertilization(self, age: float, N: float, P: float) -> None:
        self._fert.append(FertEvent(age=age, N=N, P=P))

    def add_thin_to_residual_ba(
        self,
        age: float,
        residual_ba: float,
        row_fraction: float = 0.25,
        residual_fraction: Optional[float] = None,
    ) -> None:
        self._thin.append(
            ThinEvent(
                age=age,
                residual_ba=residual_ba,
                row_fraction=row_fraction,
                residual_fraction=residual_fraction,
            )
        )

    def _fert_cumulative_at_age(self, age: float) -> Tuple[float, float]:
        hd_total = 0.0
        ba_total = 0.0
        for ev in self._fert:
            if age >= ev.age:
                yst = age - ev.age
                hd_total += PMRCModel.hd_fert_delta(yst, ev.N, ev.P)
                ba_total += PMRCModel.ba_fert_delta(yst, ev.N, ev.P)
        return hd_total, ba_total

    # ---------------- Core step ----------------

    def step(self, dt: float = 1.0) -> StandState:
        s = self.state
        a1, a2 = s.age, s.age + dt
        if s.tpa <= 0.0 or s.ba <= 0.0:
            # advance clock only; keep zeros
            self.state = StandState(
                age=a2,
                tpa=0.0,
                region=self.cfg.region,
                si25=s.si25,
                hd=max(0.0, s.hd),
                ba=0.0,
                ba_unthinned=0.0,
                tvob=0.0,
                ci=0.0,
                ci_anchor_age=None,
                ci_anchor_value=None,
                history=dict(s.history),
                fert_hd_applied=s.fert_hd_applied,
                fert_ba_applied=s.fert_ba_applied,
                disturbance_flag=s.disturbance_flag,
            )
            return self.state
        # --- stochastic growth path ----
        if self._stochastic is not None:
            stoch_state = StochStandState(
                age=s.age,
                hd=s.hd,
                tpa=s.tpa,
                ba=s.ba,
                si25=s.si25 or 0.0,
                region=self.cfg.region,
                phwd=0.0,
            )
            next_state, level, _ = self._stochastic.sample_next_state_with_event(
                stoch_state,
                dt,
                self._rng,
            )
            hd2, ba2, tpa2 = next_state.hd, next_state.ba, next_state.tpa
            disturbance_flag = level or s.disturbance_flag

            # Apply fertilization deltas additively (respect scheduled fert events)
            cum_hd, cum_ba = self._fert_cumulative_at_age(next_state.age)
            inc_hd = cum_hd - s.fert_hd_applied
            inc_ba = cum_ba - s.fert_ba_applied
            hd2 += inc_hd
            ba2 += inc_ba
            s.fert_hd_applied = cum_hd
            s.fert_ba_applied = cum_ba

            tvob2 = self.model.tvob(next_state.age, tpa2, hd2, ba2, region=self.cfg.region)

            self.state = StandState(
                age=next_state.age,
                tpa=tpa2,
                region=self.cfg.region,
                si25=s.si25,
                hd=hd2,
                ba=ba2,
                ba_unthinned=ba2,
                tvob=tvob2,
                ci=s.ci,
                ci_anchor_age=s.ci_anchor_age,
                ci_anchor_value=s.ci_anchor_value,
                history=dict(s.history),
                fert_hd_applied=s.fert_hd_applied,
                fert_ba_applied=s.fert_ba_applied,
                disturbance_flag=disturbance_flag,
            )

            # Apply any instantaneous thin scheduled at the advanced age
            self._maybe_apply_thin_at(self.state.age)
            return self.state

        # 1) Growth projection (unthinned baseline)
        hd2 = self.model.hd_project(a1, s.hd, a2) if s.hd > 0 else 0.0

        # --- TPA: enforce monotone non-increase ---
        years = max(0.0, a2 - a1)
        cand_model = self.model.tpa_project(s.tpa, s.si25, a1, a2)

        if self.cfg.hold_tpa_below_asymptote and s.tpa <= self.model.min_tpa_asymptote:
            tpa2 = s.tpa
        else:
            tpa2 = min(s.tpa, cand_model)
            if self.cfg.tpa_geometric_decay:
                cand_geom = s.tpa * (self.cfg.tpa_geometric_decay ** years)
                tpa2 = min(tpa2, cand_geom)

        # numeric floors
        if tpa2 <= 0.0:
            tpa2 = 1e-6
        if hd2 <= 0.0:
            hd2 = 1e-6



        # BA projection depends on whether we are in CI recovery from a thin
        if s.ci_anchor_age is None:
            if s.ba <= 0.0 or s.tpa <= 0.0 or s.hd <= 0.0:
                ba2 = 0.0
                ba_unthinned2 = 0.0
            else:
                ba2 = self.model.ba_project(a1, s.tpa, tpa2, s.ba, s.hd, hd2, a2, self.cfg.region)
                ba_unthinned2 = ba2  # no thinning effect
        else:
            # Compute unthinned counterpart then scale by projected CI
            if tpa2 <= 0.0 or hd2 <= 0.0:
                ba_unthinned2 = 0.0
            else:
                ba_unthinned2 = self.model.ba_predict(age=a2, tpa=tpa2, hd=hd2, region=self.cfg.region)
            ci0 = s.ci_anchor_value or 0.0
            ci_proj = self.model.ci_project(ci1=ci0, age1=s.ci_anchor_age, age2=a2, region=self.cfg.region)
            ba2 = self.model.ba_thinned(ba_unthinned2=ba_unthinned2, ci2=ci_proj)

        tvob2 = self.model.tvob(a2, tpa2, hd2, ba2, region=self.cfg.region)

        # 2) Apply fertilization deltas additively if any events have occurred
        cum_hd, cum_ba = self._fert_cumulative_at_age(a2)
        inc_hd = cum_hd - s.fert_hd_applied
        inc_ba = cum_ba - s.fert_ba_applied
        hd2 += inc_hd
        ba2 += inc_ba
        s.fert_hd_applied = cum_hd
        s.fert_ba_applied = cum_ba

        history_out = dict(s.history)
        disturbance_flag = s.disturbance_flag
        for ev in getattr(self, "_dist", []):
            if ev.triggered or a2 < ev.start_age:
                continue
            ba2 *= max(0.0, 1.0 - ev.ba_loss_fraction)
            tpa2 *= max(0.0, 1.0 - ev.tpa_loss_fraction)
            hd2 *= max(0.0, 1.0 - ev.hd_loss_fraction)
            ba_unthinned2 = ba2
            ev.triggered = True
            if ev.disturbance_level:
                disturbance_flag = ev.disturbance_level
            history_out[f"disturbance_{ev.category}_{ev.start_age:.2f}"] = ev.severity

        tvob2 = self.model.tvob(a2, tpa2, hd2, ba2, region=self.cfg.region)

        self.state = StandState(
            age=a2,
            tpa=tpa2,
            region=self.cfg.region,
            si25=s.si25,
            hd=hd2,
            ba=ba2,
            ba_unthinned=ba_unthinned2,
            tvob=tvob2,
            ci=s.ci,
            ci_anchor_age=s.ci_anchor_age,
            ci_anchor_value=s.ci_anchor_value,
            history=history_out,
            fert_hd_applied=s.fert_hd_applied,
            fert_ba_applied=s.fert_ba_applied,
            disturbance_flag=disturbance_flag,
        )

        # Apply any instantaneous thin at end of step if scheduled at a2
        self._maybe_apply_thin_at(a2)

        return self.state

    # ---------------- Helpers ----------------

    def run_to(self, target_age: float, dt: float = 1.0) -> StandState:
        while self.state.age < target_age:
            self.step(min(dt, target_age - self.state.age))
        return self.state

    def _fert_deltas_at_age(self, age: float) -> Tuple[float, float]:
        """Sum additive deltas from all fert events with fert.age <= age."""
        hd_delta = 0.0
        ba_delta = 0.0
        for ev in self._fert:
            if age >= ev.age:
                yst = age - ev.age
                hd_delta += PMRCModel.hd_fert_delta(yst, ev.N, ev.P)
                ba_delta += PMRCModel.ba_fert_delta(yst, ev.N, ev.P)
        return hd_delta, ba_delta

    def _maybe_apply_thin_at(self, age: float) -> None:
        # Find thin events scheduled exactly at this age (float-safe compare)
        due = [ev for ev in self._thin if abs(ev.age - age) < 1e-9]
        if not due:
            return
        # Apply sequentially if multiple at same age
        for ev in due:
            self._apply_thin_event(ev)

    def _apply_thin_event(self, ev: ThinEvent) -> None:
        """
        Apply thinning:
        - Compute pre-thin BA, TPA, HD at the thin age (already the current state).
        - If residual_ba >= prethin_ba: skip.
        - Reduce BA to residual and compute TPA assuming constant QMD.
        - Compute unthinned counterpart BA_ntc and CI at thin age.
        - Record CI anchor for future recovery.
        """
        s = self.state
        pre_ba, pre_tpa, pre_hd = s.ba, s.tpa, s.hd
        if ev.residual_ba >= pre_ba:
            # cannot thin to higher BA
            return
        target_ba = ev.residual_ba
        if ev.residual_fraction is not None:
            target_ba = max(0.0, ev.residual_fraction * pre_ba)
        if target_ba >= pre_ba:
            return

        post_ba = target_ba
        # Constant QMD: assumes trees removed proportionally across diameter classes
        qmd_pre = PMRCModel.qmd(tpa=pre_tpa, ba=pre_ba)
        post_tpa = PMRCModel.tpa_from_ba_qmd(ba=post_ba, qmd_in=qmd_pre)
        # Unthinned counterpart at the same age for CI
        ba_ntc = self.model.ba_predict(age=s.age, tpa=post_tpa, hd=pre_hd, region=self.cfg.region)
        ci0 = self.model.competition_index(ba_after=post_ba, ba_unthinned=ba_ntc)

        # Set new state at thin time
        self.state = StandState(
            age=s.age,
            tpa=post_tpa,
            region=self.cfg.region,
            si25=s.si25,
            hd=pre_hd,
            ba=post_ba,
            ba_unthinned=ba_ntc,  # store what it would be without thinning
            tvob=self.model.tvob(s.age, post_tpa, pre_hd, post_ba, region=self.cfg.region),
            ci=ci0,
            ci_anchor_age=s.age,
            ci_anchor_value=ci0,
            history={**s.history, f"thin_at_{s.age:.3f}": ev.residual_ba},
            fert_hd_applied=s.fert_hd_applied,
            fert_ba_applied=s.fert_ba_applied,
            disturbance_flag=s.disturbance_flag,
        )

    def add_disturbance(self, event: DisturbanceEvent) -> None:
        if not hasattr(self, "_dist"):
            self._dist: list[DisturbanceEvent] = []
        self._dist.append(event)

    def replant(self, *, tpa: float, si25: float, initial_age: float = 1.0) -> None:
        age_init = max(0.1, float(initial_age))
        tpa_val = float(tpa)
        si25_val = float(si25)

        hd = float(self.model.hd_from_si(si25_val, form="projection"))
        ba = float(self.model.ba_predict(age=age_init, tpa=tpa_val, hd=hd, region=self.cfg.region))
        tvob = float(self.model.tvob(age_init, tpa_val, hd, ba, region=self.cfg.region))

        self.state = StandState(
            age=age_init,
            tpa=tpa_val,
            region=self.cfg.region,
            si25=si25_val,
            hd=hd,
            ba=ba,
            ba_unthinned=ba,
            tvob=tvob,
            ci=0.0,
            ci_anchor_age=None,             # clear competition anchors
            ci_anchor_value=None,
            history={},
            fert_hd_applied=0.0,
            fert_ba_applied=0.0,
            disturbance_flag=None,
        )

        # clear management and disturbance histories
        self._fert = []
        self._thin = []
        if hasattr(self, "_dist"):
            self._dist = []

        # optional: guard large first-step dt compounding if you track last-advance
        if hasattr(self, "_last_age_advanced"):
            self._last_age_advanced = age_init



    # ---------------- Convenience ----------------

    def summary(self) -> str:
        s = self.state
        ba_diff = s.ba_unthinned - s.ba
        dist = f" | Disturbance={s.disturbance_flag}" if s.disturbance_flag else ""
        return (
            f"Age {s.age:.1f} | HD={s.hd:.2f} ft | TPA={s.tpa:.1f} | "
            f"BA={s.ba:.2f} ft2/ac | BA_unthinned={s.ba_unthinned:.2f} ft2/ac ({ba_diff:+.2f}) | "
            f"TVOB={s.tvob:.1f} | CI={s.ci_anchor_value or 0.0:.3f}{dist}"
        )


# --------------------------- Module-level utilities ---------------------------

def _pmrc(region: Region | str | None = None) -> PMRCModel:
    reg = region.value if isinstance(region, Region) else region
    reg = reg or Region.UCP.value
    return PMRCModel(region=str(reg).lower())


def si_from_hd(hd: float, form: SIForm = SIForm.PROJECTION) -> float:
    return _pmrc().si_from_hd(hd, form=form.value)


def hd_from_si(si25: float, form: SIForm = SIForm.PROJECTION) -> float:
    return _pmrc().hd_from_si(si25, form=form.value)


def hd_project(age1: float, hd1: float, age2: float, *, region: Region = Region.UCP) -> float:
    return _pmrc(region).hd_project(age1, hd1, age2)


def tpa_project(tpa1: float, si25: float, age1: float, age2: float, *, region: Region = Region.UCP) -> float:
    return _pmrc(region).tpa_project(tpa1, si25, age1, age2)


def ba_predict(age: float, tpa: float, hd: float, region: Region = Region.UCP) -> float:
    return _pmrc(region).ba_predict(age, tpa, hd, region=region.value if isinstance(region, Region) else region)


def step(init: StandState, *, cfg: StandConfig | None = None, dt: float = 1.0) -> tuple[StandState, float]:
    stand = Stand(init=init, cfg=cfg)
    state = stand.step(dt)
    return state, state.ba


def run_horizon(init: StandState, *, years: float, dt: float = 1.0, cfg: StandConfig | None = None) -> List[tuple[StandState, float]]:
    stand = Stand(init=init, cfg=cfg)
    target_age = init.age + years
    out: List[tuple[StandState, float]] = []
    while stand.state.age < target_age - 1e-9:
        step_dt = min(dt, target_age - stand.state.age)
        state = stand.step(step_dt)
        out.append((state, state.ba))
    return out


if __name__ == "__main__":
    # # Instantiate two stands with different configs
    s1 = Stand(StandState(age=1, tpa=600, si25=60, hd=0, ba=0), StandConfig(region="ucp", tpa_geometric_decay=0.99))

    # Schedule
    s1.run_to(60)
    print(s1.summary())
