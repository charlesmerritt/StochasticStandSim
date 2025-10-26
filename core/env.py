from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional, Literal
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from core.growth import Stand, StandConfig, StandState
from core.disturbances import make_fire_event, make_wind_event
from core.pmrc_model import PMRCModel
from core.actions import (
    ActionManager,
    Action,
    ActionOutcome,
    apply_action,
    revenue_from_volume_tons_even_split,
)

# ----------------------- defaults -----------------------

@dataclass(frozen=True)
class EnvConfig:
    # stand init
    age0: float = 1.0
    tpa0: float = 600.0
    si25: float = 60.0
    region: str = "ucp"
    dt: float = 1.0
    horizon_years: float = 30.0
    # Disturbances
    disturbance_enabled: bool = False
    disturbance_probs: Dict[str, float] = field(default_factory=lambda: {"fire": 0.0, "wind": 0.0})
    severity_dist: Literal["uniform", "beta"] = "uniform"
    severity_beta: Tuple[float, float] = (2.0, 5.0)   # only used if severity_dist=="beta"
    rng_seed: Optional[int] = 123
    # Economics
    discount_rate: float = 0.03  # annual discount rate for NPV calculations
    growth_reward_weight: float = 0.05  # reward shaping: add small reward for volume growth
    # PPO convenience
    obs_clip: Dict[str, Tuple[float, float]] = None  # set below in __post_init__

    def __post_init__(self):
        if self.obs_clip is None:
            object.__setattr__(self, "obs_clip", {
                "age": (0.0, 60.0),
                "tpa": (0.0, 1200.0),
                "hd":  (0.0, 120.0),
                "ba":  (0.0, 250.0),
                "vol": (0.0, 8000.0),
                "ci":  (0.0, 1.0),
            })

# ------------------- RL environment --------------------

class StandMgmtEnv(gym.Env):
    """
    Observation (Box[7]):
      [age, tpa, hd, ba, vol, ci, harvested_flag]
      All clipped to EnvConfig.obs_clip and scaled to [0,1] for PPO stability.

    Action space:
      Discrete(5) id with a companion Box(3,) 'params' passed via env.step(action, params)
      IDs:
        0 = NOOP
        1 = THIN   (params[0] = residual_ba_frac in [0.25, 0.95])
        2 = FERT   (params[0] -> N in [0, 300], params[1] -> P in {0,1})
        3 = HARVEST
        4 = PLANT  (params[0] -> tpa0 in [300, 900], params[1] -> si25 in [45, 75])
        5 = SALVAGE (params[0] -> salvage_frac in [0, 1])
        6 = RXFIRE
      SB3 PPO handles kwargs via Gym wrappers; we keep step(action, params) signature simple.
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: EnvConfig | None = None):
        super().__init__()
        self.cfg = cfg or EnvConfig()
        self.rng = np.random.default_rng(self.cfg.rng_seed)
        # discrete action menu (see step)
        self.action_space = spaces.Discrete(8)

        # observation space (normalized 0..1)
        self._obs_low  = np.array([v[0] for v in self.cfg.obs_clip.values()], dtype=np.float32)
        self._obs_high = np.array([v[1] for v in self.cfg.obs_clip.values()], dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32)

        self.model = PMRCModel(region=self.cfg.region)
        from core.actions import Discount
        self.manager = ActionManager(
            t0_age=self.cfg.age0,
            disc=Discount(rate_annual=self.cfg.discount_rate)
        )
        self.stand: Stand | None = None
        self.age_end = self.cfg.age0 + self.cfg.horizon_years
        self._reset_core()

    # ---------------- core helpers ----------------

    def _reset_core(self):
        hd0 = self.model.hd_from_si(self.cfg.si25, form="projection")
        ba0 = self.model.ba_predict(self.cfg.age0, self.cfg.tpa0, hd0, region=self.cfg.region)
        self.stand = Stand(
            init=StandState(age=self.cfg.age0, tpa=self.cfg.tpa0, si25=self.cfg.si25, hd=hd0, ba=ba0),
            cfg=StandConfig(region=self.cfg.region, tpa_geometric_decay=0.99),
        )
        self.manager.last_thin_age = None
        self.manager.harvested = False
        self.manager.replanted = False
        self.manager.salvaged_recently = False
        self.manager.sold = False
        self._prev_volume = self.stand.state.tvob  # track for growth rewards

    def _obs(self) -> np.ndarray:
        s = self.stand.state
        raw = np.array([
            s.age,
            max(0.0, s.tpa),
            max(0.0, s.hd),
            max(0.0, s.ba),
            max(0.0, s.tvob),
            np.clip(s.ci_anchor_value or 0.0, 0.0, 1.0),
            1.0 if self.manager.harvested else 0.0
        ], dtype=np.float32)
        clipped = np.clip(raw[:6], self._obs_low, self._obs_high)
        scaled = (clipped - self._obs_low) / (self._obs_high - self._obs_low + 1e-9)
        return np.concatenate([scaled, raw[6:7]])  # append harvested flag

    def _advance_growth(self):
        # Freeze growth after harvest until planting
        if self.manager.harvested and not self.manager.replanted:
            # only advance age clock
            self.stand.state.age += self.cfg.dt
            return
        self.stand.step(self.cfg.dt)

    def _draw_severity(self) -> float:
        if self.cfg.severity_dist == "beta":
            a, b = self.cfg.severity_beta
            return float(self.rng.beta(a, b))
        return float(self.rng.random())

    def _maybe_add_disturbances(self) -> None:
        if not self.cfg.disturbance_enabled:
            return
        if self.stand.state.tpa <= 0.0:  # fallow after harvest
            return
        age = self.stand.state.age
        for kind, p in self.cfg.disturbance_probs.items():
            p = max(0.0, min(1.0, float(p)))
            if self.rng.random() < p:
                sev = self._draw_severity()
                seed = int(self.rng.integers(0, 2**31 - 1))
                if kind == "fire":
                    ev = make_fire_event(start_age=age, severity=sev, seed=seed)
                elif kind == "wind":
                    ev = make_wind_event(start_age=age, severity=sev, seed=seed)
                else:
                    continue
                self.stand.add_disturbance(ev)

    # ---------------- gym API ----------------

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._reset_core()
        return self._obs(), {}

    def step(self, action):
        assert self.stand is not None
        params = [0.0, 0.0]
        if isinstance(action, dict):
            sel = int(action.get("type", 0))
            params = action.get("value", [0.0, 0.0]) or [0.0, 0.0]
        else:
            sel = int(np.clip(np.asarray(action).item(), 0, 7))
        reward = 0.0
        info: Dict[str, Any] = {}

        action_labels = {
            0: "noop",
            1: "thin_40",
            2: "thin_60",
            3: "fert_n200_p1",
            4: "harvest",
            5: "plant_600_si60",
            6: "salvage_0p3",
            7: "rxfire",
        }
        info["action_id"] = sel
        info["action_label"] = action_labels.get(sel, "noop")

        action_outcome: ActionOutcome | None = None
        step_cashflow = 0.0
        step_npv = 0.0

        if sel == 0:  # NOOP
            action_outcome = ActionOutcome(ok=True, cashflow_now=0.0, npv_at0=0.0)
            step_cashflow = 0.0
            step_npv = 0.0

        elif sel == 1:  # THIN_40
            resid_ba = max(0.0, self.stand.state.ba * 0.40)
            action_outcome = apply_action(self.manager, self.stand, Action.THINNING, now_params={"residual_ba": resid_ba})
            reward += action_outcome.npv_at0
            step_cashflow = action_outcome.cashflow_now
            step_npv = action_outcome.npv_at0
            if not action_outcome.ok:
                reward -= 50.0 
            reward = np.clip(reward, -1e4, 1e4)
            info.update({"thin_ok": action_outcome.ok, "thin_reason": action_outcome.reason})

        elif sel == 2:  # THIN_60
            resid_ba = max(0.0, self.stand.state.ba * 0.60)
            action_outcome = apply_action(self.manager, self.stand, Action.THINNING, now_params={"residual_ba": resid_ba})
            reward += action_outcome.npv_at0
            step_cashflow = action_outcome.cashflow_now
            step_npv = action_outcome.npv_at0
            if not action_outcome.ok:
                reward -= 50.0
            reward = np.clip(reward, -1e4, 1e4)
            info.update({"thin_ok": action_outcome.ok, "thin_reason": action_outcome.reason})

        elif sel == 3:  # FERT_N200_P1
            N = 200.0
            P = 1.0
            action_outcome = apply_action(self.manager, self.stand, Action.FERTILIZATION, now_params={"N": N, "P": P})
            reward += action_outcome.npv_at0
            step_cashflow = action_outcome.cashflow_now
            step_npv = action_outcome.npv_at0
            reward = np.clip(reward, -1e4, 1e4)
            info.update({"fert_ok": action_outcome.ok})

        elif sel == 4:  # HARVEST
            if self.stand.state.age < 30.0:
                action_outcome = ActionOutcome(
                    ok=False,
                    reason="Harvest prohibited before age 30.",
                    cashflow_now=0.0,
                    npv_at0=0.0,
                )
                reward -= 10.0
            else:
                action_outcome = apply_action(self.manager, self.stand, Action.HARVESTING)
                reward += action_outcome.npv_at0
                step_cashflow = action_outcome.cashflow_now
                step_npv = action_outcome.npv_at0
                info.update({"harvest_ok": action_outcome.ok})
                if params and float(params[0]) >= 0.5 and action_outcome.ok:
                    replant_outcome = apply_action(
                        self.manager,
                        self.stand,
                        Action.PLANTING,
                        now_params={"tpa0": 600.0, "si25": 60.0},
                    )
                    step_cashflow += replant_outcome.cashflow_now
                    step_npv += replant_outcome.npv_at0
                    info.update({"plant_ok": replant_outcome.ok, "plant_reason": replant_outcome.reason})
            reward = np.clip(reward, -1e4, 1e4)

        elif sel == 5:  # PLANT_600_SI60
            tpa0 = 600.0
            si = 60.0
            action_outcome = apply_action(self.manager, self.stand, Action.PLANTING, now_params={"tpa0": tpa0, "si25": si})
            reward += action_outcome.npv_at0
            step_cashflow = action_outcome.cashflow_now
            step_npv = action_outcome.npv_at0
            if not action_outcome.ok:
                reward -= 25.0
            reward = np.clip(reward, -1e4, 1e4)
            info.update({"plant_ok": action_outcome.ok, "plant_reason": action_outcome.reason})
        
        elif sel == 6:  # SALVAGE_0p3
            if self.stand.state.age < 30.0:
                action_outcome = ActionOutcome(
                    ok=False,
                    reason="Salvage prohibited before age 30.",
                    cashflow_now=0.0,
                    npv_at0=0.0,
                )
                reward -= 25.0
                step_cashflow = 0.0
                step_npv = 0.0
                info.update({"salvage_ok": False, "salvage_reason": action_outcome.reason})
            elif not self.cfg.disturbance_enabled:
                action_outcome = ActionOutcome(
                    ok=False,
                    reason="Salvage not available when disturbances are disabled.",
                    cashflow_now=0.0,
                    npv_at0=0.0,
                )
                reward -= 25.0
                step_cashflow = 0.0
                step_npv = 0.0
                info.update({"salvage_ok": False, "salvage_reason": action_outcome.reason})
            else:
                frac = float(np.clip(params[0], 0.0, 1.0)) if params else 0.3
                action_outcome = apply_action(self.manager, self.stand, Action.SALVAGING, now_params={"salvage_frac": frac})
                reward += action_outcome.npv_at0
                step_cashflow = action_outcome.cashflow_now
                step_npv = action_outcome.npv_at0
                info.update({"salvage_ok": action_outcome.ok})
            reward = np.clip(reward, -1e4, 1e4)

        elif sel == 7:  # RXFIRE
            action_outcome = apply_action(self.manager, self.stand, Action.RXFIRE)
            reward += action_outcome.npv_at0
            step_cashflow = action_outcome.cashflow_now
            step_npv = action_outcome.npv_at0
            reward = np.clip(reward, -1e4, 1e4)
            info.update({"rxfire_ok": action_outcome.ok})
        else:
            action_outcome = None

        if action_outcome is None:
            action_outcome = ActionOutcome(ok=True, cashflow_now=0.0, npv_at0=0.0)

        info["cashflow_now"] = step_cashflow
        info["npv_at0"] = step_npv
        if action_outcome.reason:
            info["action_reason"] = action_outcome.reason

        # add disturbance chance and advance t
        self._maybe_add_disturbances()
        self._advance_growth()

        # reward for volume growth (encourages growing timber)
        volume_reward = 0.0
        if self.cfg.growth_reward_weight > 0.0:
            vol_now = self.stand.state.tvob
            vol_delta = vol_now - self._prev_volume
            volume_reward = self.cfg.growth_reward_weight * vol_delta
            reward += volume_reward
            self._prev_volume = vol_now

        # termination
        done = bool(self.stand.state.age >= self.age_end or self.manager.sold)
        truncated = False

        if done:
        # monetize remaining volume if not already harvested
            if not self.manager.harvested:
                vol = self.model.tvob(self.stand.state.age, self.stand.state.tpa,
                    self.stand.state.hd, self.stand.state.ba, region=self.stand.cfg.region)
                tons = vol * 0.031
                term_cash = revenue_from_volume_tons_even_split(tons, self.manager.prices)
                reward += self.manager._npv(term_cash, self.stand.state.age)

        obs = self._obs()
        reward -= 1.0
        
        info.update({
            "age": self.stand.state.age,
            "tpa": self.stand.state.tpa,
            "ba": self.stand.state.ba,
            "reward": reward,
            "volume_reward": volume_reward,
            "action": action.name if isinstance(action, Action) else action,
            "cashflow_now": action_outcome.cashflow_now,
            "npv_at0": action_outcome.npv_at0,
            "reason": action_outcome.reason,
        })

        return obs, float(reward), done, truncated, info

    def render(self):
        # Not implemented; use your plotting utilities outside the env.
        return None
