"""Gymnasium environment for thinning control under stochastic disturbances.

This module exposes :class:`StandEnv`, a lightweight wrapper around the
growth and disturbance models defined in :mod:`core.growth` and
:mod:`core.disturbances`.  The environment follows the Gymnasium API so it can
be connected directly to Stable-Baselines3 (SB3) algorithms.
"""

from __future__ import annotations

from dataclasses import replace
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .disturbances import (
    FireDisturbance,
    ThinningDisturbance,
    WindDisturbance,
)
from .growth import (
    EXAMPLE_STANDS,
    GrowthConfig,
    StandParams,
    StandState,
    step_with_log,
)
from .rng import rng as global_rng
from .economics import EconParams, load_econ_params


_DEFAULT_ECON_PATH = Path(__file__).parent.parent / "data" / "econ_params.yaml"


def _resolve_stand(stand: str | StandParams) -> StandParams:
    """Return a :class:`StandParams` from a name or pre-built instance."""
    if isinstance(stand, StandParams):
        return stand
    try:
        return EXAMPLE_STANDS[stand]
    except KeyError as exc:
        raise ValueError(f"Unknown stand '{stand}'. Available keys: {sorted(EXAMPLE_STANDS)}") from exc


def _coerce_growth_cfg(cfg: GrowthConfig | Mapping[str, Any] | None) -> GrowthConfig:
    """Build a :class:`GrowthConfig` from a mapping, or return the given instance."""
    if cfg is None:
        return GrowthConfig()
    if isinstance(cfg, GrowthConfig):
        return cfg
    return GrowthConfig(**cfg)


class ActionType(IntEnum):
    """Supported management actions."""

    NOOP = 0
    THIN = 1
    HARVEST = 2
    SELL = 3
    SALVAGE = 4
    REPLANT = 5
    FERTILIZE = 6
    RX_FIRE = 7


class StandEnv(gym.Env[np.ndarray, Dict[str, np.ndarray | int | float]]):
    """Gymnasium-compatible MDP for stochastic stand management.

    Parameters
    ----------
    stand : str | StandParams, default="ucp_baseline"
        Initial stand definition.  Either a key from
        :data:`core.growth.EXAMPLE_STANDS` or an explicit :class:`StandParams`.
    max_age : float, default=35
        Terminal age (years). Episodes terminate once the stand reaches or
        exceeds this value.
    thin_max_fraction : float, default=0.4
        Upper bound on the thinning removal fraction (0–1] applied per action.
    disturbance_probs : Mapping[str, float], optional
        Annual probability for each stochastic disturbance. Supported keys:
        ``"fire"`` and ``"wind"``.  Values outside [0, 1] raise ``ValueError``.
    reward_config : Mapping[str, float], optional
        Overrides for reward weights derived from ``econ_path``. Recognised keys
        include ``price_per_volume``, ``growth_weight``, ``thinning_cost``,
        ``thinning_cost_precommercial``, ``precommercial_age``, and
        ``discount_rate``.

        Reward at each step is computed as::

            price_per_volume * (
                growth_weight * growth_increment
                + harvested_volume
                - disturbance_loss
            )
            - thinning_cost * I[harvested_volume > 0]

        where all volume terms are expressed in ft³/ac.
    econ_path : str | Path, optional
        Location of the economic parameter YAML. Defaults to
        ``data/econ_params.yaml`` when omitted.
    growth_cfg : GrowthConfig | Mapping[str, Any], optional
        Overrides for :class:`GrowthConfig`.
    seed : int | None, optional
        Seed for the environment's own RNG.  Passing a seed also resets the
        shared RNG used by :func:`core.rng.rng` so that disturbance sampling is
        reproducible across environments.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        stand: str | StandParams = "ucp_baseline",
        *,
        max_age: float = 35.0,
        thin_max_fraction: float = 0.4,
        disturbance_probs: Optional[Mapping[str, float]] = None,
        reward_config: Optional[Mapping[str, float]] = None,
        econ_path: str | Path | None = None,
        growth_cfg: GrowthConfig | Mapping[str, Any] | None = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._base_params = _resolve_stand(stand)
        self._growth_cfg = _coerce_growth_cfg(growth_cfg)
        self._max_age = float(max_age)
        if self._max_age <= 0:
            raise ValueError("max_age must be positive.")

        self._thin_max_fraction = float(thin_max_fraction)
        if not (0.0 <= self._thin_max_fraction <= 1.0):
            raise ValueError("thin_max_fraction must be within [0, 1].")

        probs = disturbance_probs or {"fire": 0.05, "wind": 0.03}
        self._disturbance_probs = self._validate_probs(probs)

        econ_path_obj = Path(econ_path) if econ_path is not None else _DEFAULT_ECON_PATH
        self._econ_params = self._load_econ(econ_path_obj)
        reward_defaults = self._derive_reward_defaults(self._econ_params)
        if reward_config:
            reward_defaults.update(reward_config)
        self._reward_cfg = reward_defaults

        self._state: StandState | None = None
        self._last_events: Tuple[dict, ...] = ()
        self._steps = 0
        self._finance: Dict[str, float] = {"cash": 0.0, "npv": 0.0, "lev": 0.0}

        # Gymnasium RNG helper
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        # Synchronise the shared RNG used in modules relying on core.rng.
        global_rng(seed)

        # Observations: age, TPA, BA, HD, volume, competition index, active envelopes
        obs_low = np.array(
            [
                0.0,  # age
                0.0,  # tpa
                0.0,  # ba
                0.0,  # hd
                0.0,  # volume
                0.0,  # ci
                0.0,  # active envelopes
                0.0,  # growth paused flag
                0.0,  # years since replant
                -1e7,  # cumulative cash
                -1e7,  # cumulative npv
                -1e7,  # cumulative lev
            ],
            dtype=np.float32,
        )
        obs_high = np.array(
            [
                self._max_age + 10.0,  # allow some slack beyond terminal age
                2000.0,  # trees per acre
                600.0,  # basal area
                150.0,  # dominant height
                5000.0,  # volume
                1.0,  # competition index
                10.0,  # active envelopes count
                1.0,  # growth paused flag
                self._max_age + 10.0,  # years since replant
                1e7,  # cumulative cash
                1e7,  # cumulative npv
                1e7,  # cumulative lev
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = spaces.Dict(
            {
                "type": spaces.Discrete(len(ActionType)),
                "value": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
            }
        )

    # --------------------------------------------------------------------- #
    # Gymnasium API

    def seed(self, seed: Optional[int] = None) -> list[int]:
        """Reset the RNGs and return the generated seed."""
        self.np_random, actual_seed = gym.utils.seeding.np_random(seed)
        global_rng(actual_seed)
        return [actual_seed]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment state."""
        if seed is not None:
            self.seed(int(seed))

        params = options.get("stand") if options else None
        if params is not None:
            base_params = _resolve_stand(params)
        else:
            base_params = self._base_params

        self._state = base_params.to_state()
        self._state = replace(
            self._state,
            cumulative_cash=0.0,
            cumulative_discounted_cash=0.0,
            cumulative_lev=0.0,
            years_since_replant=self._state.age,
        )
        self._last_events = ()
        self._steps = 0
        self._finance = {"cash": 0.0, "npv": 0.0, "lev": 0.0}

        obs = self._state_to_obs(self._state)
        info = {"events": self._last_events}
        return obs, info

    def step(
        self, action: Dict[str, Any] | np.ndarray | Iterable[float] | float
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance the stand one year using the provided action."""
        if self._state is None:
            raise RuntimeError("Environment must be reset before calling step().")

        action_type, action_value = self._parse_action(action)
        info: Dict[str, Any] = {"actions": []}
        cash_flow = 0.0
        terminated = False

        state = self._state
        scheduled_events = list(state.pending_disturbances)
        state_updates: Dict[str, Any] = {}
        manual_events: list[dict] = []

        exogenous_events = self._sample_disturbances()
        if exogenous_events:
            info["action_blocked"] = True
            action_type = ActionType.NOOP
        scheduled_events.extend(exogenous_events)

        if action_type == ActionType.NOOP:
            info["actions"].append({"type": "noop"})
        elif action_type == ActionType.THIN:
            delta_cash, state_updates, allowed = self._handle_thin(action_value, state, scheduled_events, info)
            if not allowed:
                info["action_warn"] = "Thinning not allowed due to cooldown or growth pause."
                cash_flow = 0.0
                scheduled_events = list(state.pending_disturbances)
                state_updates = {}
                action_type = ActionType.NOOP
            else:
                cash_flow += delta_cash
        elif action_type == ActionType.HARVEST:
            delta_cash, state_updates, harvest_events, allowed = self._handle_harvest(action_value, state, info)
            if not allowed:
                info["action_warn"] = "Harvest not permitted (growth paused or other constraint)."
                cash_flow = 0.0
                scheduled_events = list(state.pending_disturbances)
                state_updates = {}
                action_type = ActionType.NOOP
            else:
                cash_flow += delta_cash
                scheduled_events.clear()
                manual_events.extend(harvest_events)
        elif action_type == ActionType.SELL:
            delta_cash, state_updates, harvest_events = self._handle_sell(state, info)
            cash_flow += delta_cash
            scheduled_events.clear()
            manual_events.extend(harvest_events)
            terminated = True
            manual_events.extend(harvest_events)
        elif action_type == ActionType.SALVAGE:
            delta_cash, state_updates, salvage_events, success = self._handle_salvage(state, info)
            if success:
                cash_flow += delta_cash
                scheduled_events.clear()
                manual_events.extend(salvage_events)
            else:
                info["action_warn"] = "Salvage not available (no recent catastrophic disturbance)."
                cash_flow = 0.0
                state_updates = {}
                scheduled_events = list(state.pending_disturbances)
        elif action_type == ActionType.REPLANT:
            delta_cash, state_updates, allowed = self._handle_replant(state, info)
            if not allowed:
                info["action_warn"] = "Replant only allowed after harvest/salvage."
                cash_flow = 0.0
                state_updates = {}
            else:
                cash_flow += delta_cash
        elif action_type == ActionType.FERTILIZE:
            delta_cash, state_updates, allowed = self._handle_fertilize(state, info)
            if not allowed:
                info["action_warn"] = "Fertilization not allowed due to cooldown or growth pause."
                cash_flow = 0.0
                state_updates = {}
            else:
                cash_flow += delta_cash
        elif action_type == ActionType.RX_FIRE:
            delta_cash, state_updates, allowed = self._handle_rx_fire(state, info)
            if not allowed:
                info["action_warn"] = "Prescribed fire not allowed while growth is paused."
                cash_flow = 0.0
                state_updates = {}
            else:
                cash_flow += delta_cash

        if scheduled_events:
            scheduled_events.sort(key=lambda ev: ev.age)
            state_updates["pending_disturbances"] = tuple(scheduled_events)

        if state_updates:
            state = replace(state, **state_updates)
            self._state = state

        next_state, _, event_logs, _, _, _ = step_with_log(self._state, dt=1.0, cfg=self._growth_cfg)

        self._steps += 1
        finance_breakdown = self._update_financials(cash_flow)

        next_state = replace(
            next_state,
            cumulative_cash=finance_breakdown["cumulative_cash"],
            cumulative_discounted_cash=finance_breakdown["npv"],
            cumulative_lev=finance_breakdown["lev"],
        )
        self._state = next_state
        self._last_events = event_logs

        terminated = terminated or bool(next_state.age >= self._max_age)
        truncated = False

        obs = self._state_to_obs(next_state)
        info["events"] = list(event_logs) + manual_events
        info["finance"] = finance_breakdown
        reward = float(np.clip(cash_flow, -1e7, 1e7))
        return obs, reward, terminated, truncated, info

    def render(self, mode: str = "human") -> Optional[str]:
        """Render by returning a concise textual snapshot."""
        if mode != "human":
            raise NotImplementedError("Only the 'human' render mode is supported.")
        if self._state is None:
            return "StandEnv(reset required)"
        state = self._state
        summary = (
            f"age={state.age:.1f} yrs | tpa={state.tpa:.1f} | ba={state.ba or 0.0:.1f} "
            f"| hd={state.hd or 0.0:.1f} | vol={state.vol_ob or 0.0:.1f} "
            f"| cash={state.cumulative_cash:.2f}"
        )
        return summary

    # ------------------------------------------------------------------ #
    # Internal helpers

    @staticmethod
    def _validate_probs(probs: Mapping[str, float]) -> Dict[str, float]:
        supported = {"fire", "wind"}
        out: Dict[str, float] = {}
        for key, value in probs.items():
            if key not in supported:
                raise ValueError(f"Unsupported disturbance '{key}'. Expected one of {sorted(supported)}.")
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"Probability for '{key}' must be between 0 and 1 inclusive.")
            out[key] = float(value)
        return out

    def _load_econ(self, path: Path) -> EconParams:
        """Load economic parameters, falling back to neutral values on failure."""
        try:
            return load_econ_params(path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Could not locate economic parameter file at {path}. "
                "Provide `econ_path` or create data/econ_params.yaml."
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive path
            import warnings

            warnings.warn(
                f"Failed to parse economic parameters from {path}: {exc}. "
                "Falling back to neutral pricing."
            )
            return EconParams(prices={}, costs={}, discount_rate=0.0, raw={})

    @staticmethod
    def _derive_reward_defaults(econ: EconParams) -> Dict[str, float]:
        """Derive baseline reward parameters from economic inputs."""
        price = econ.price_weighted_average()
        if price <= 0.0:
            price = 35.0  # fallback consistent with prior stub
        thin_cost = econ.costs.get("thin_harvest", 300.0)
        thin_cost_pre = econ.costs.get("thin_precomm", thin_cost)
        return {
            "price_per_volume": price,
            "growth_weight": 0.6,
            "thinning_cost": thin_cost,
            "thinning_cost_precommercial": thin_cost_pre,
            "precommercial_age": 10.0,
            "discount_rate": econ.discount_rate,
        }

    def _parse_action(self, action: Dict[str, Any] | np.ndarray | Iterable[float] | float) -> Tuple[ActionType, np.ndarray]:
        """Decode action from Gymnasium client."""
        if isinstance(action, Mapping):
            action_type = int(action.get("type", 0))
            value = np.asarray(action.get("value", np.zeros(2, dtype=np.float32)), dtype=np.float32)
        else:
            arr = np.asarray(action, dtype=np.float32)
            if arr.ndim == 0:
                action_type = int(np.clip(arr.item(), 0, len(ActionType) - 1))
                value = np.zeros(2, dtype=np.float32)
            else:
                action_type = int(np.clip(arr[0], 0, len(ActionType) - 1))
                if arr.shape[0] > 1:
                    value = np.asarray(arr[1:], dtype=np.float32)
                else:
                    value = np.zeros(2, dtype=np.float32)
        value = np.pad(value, (0, max(0, 2 - value.shape[0])), mode="constant")[:2]
        return ActionType(action_type), value

    def _sample_disturbances(self) -> list[FireDisturbance | WindDisturbance]:
        """Sample disturbances that occur this step."""
        events: list[FireDisturbance | WindDisturbance] = []
        current_age = self._state.age if self._state is not None else 0.0
        for kind, prob in self._disturbance_probs.items():
            if self.np_random.random() < prob:
                severity = float(self.np_random.uniform(0.001, 0.999))
                if kind == "fire":
                    events.append(FireDisturbance(age=current_age, severity=severity))
                elif kind == "wind":
                    events.append(WindDisturbance(age=current_age, severity=severity))
        return events

    def _compute_product_mix(self, si25: float, age: float) -> Dict[str, float]:
        """Return fraction per product class."""
        high_si = si25 > 70.0
        if not high_si and age < 25:
            return {"pulpwood": 0.40, "chip": 0.40, "sawtimber": 0.20}
        if not high_si and age >= 25:
            return {"pulpwood": 0.30, "chip": 0.35, "sawtimber": 0.35}
        if high_si and age < 25:
            return {"pulpwood": 0.30, "chip": 0.30, "sawtimber": 0.40}
        return {"pulpwood": 0.00, "chip": 0.30, "sawtimber": 0.70}

    def _thinning_cost(self, age: float) -> float:
        cost_pre = self._econ_params.costs.get("thin_precomm", 0.0)
        cost_harvest = self._econ_params.costs.get("thin_harvest", 0.0)
        return cost_pre if age < 10.0 else cost_harvest

    def _harvest_revenue(self, state: StandState) -> float:
        volume = state.vol_ob or 0.0
        if volume <= 0.0:
            return 0.0
        si25 = state.si25 or 0.0
        mix = self._compute_product_mix(si25, state.age)
        value = 0.0
        for product, share in mix.items():
            price = self._econ_params.prices.get(product, 0.0)
            value += volume * share * price
        return value

    def _apply_clearcut(self, state: StandState) -> Dict[str, Any]:
        updates = {
            "ba": 0.0,
            "ba_unthinned": 0.0,
            "tpa": 0.0,
            "tpa_unthinned": 0.0,
            "vol_ob": 0.0,
            "vol_ob_unthinned": 0.0,
            "ci": 0.0,
            "growth_paused": True,
            "pending_disturbances": tuple(),
            "active_envelopes": tuple(),
            "salvage_window": 0,
            "fertilizer_years_remaining": 0,
            "fertilizer_effect_strength": 0.0,
            "rx_fire_years_remaining": 0,
            "rx_fire_severity_multiplier": 1.0,
            "last_fertilize_age": None,
            "last_thin_age": None,
            "last_disturbance_age": state.age,
            "years_since_replant": None,
        }
        return updates

    def _replant_template(self) -> StandState:
        template = self._base_params.to_state()
        return template

    def _apply_replant_values(self, base: StandState) -> Dict[str, Any]:
        return {
            "age": 0.0,
            "tpa": base.tpa,
            "tpa_unthinned": base.tpa_unthinned,
            "ba": base.ba,
            "ba_unthinned": base.ba_unthinned,
            "hd": base.hd,
            "si25": base.si25,
            "ci": base.ci,
            "vol_ob": base.vol_ob,
            "vol_ob_unthinned": base.vol_ob_unthinned,
            "growth_paused": False,
            "active_envelopes": tuple(),
            "pending_disturbances": tuple(),
            "last_thin_age": None,
            "years_since_replant": 0.0,
            "fertilizer_years_remaining": 0,
            "fertilizer_effect_strength": 0.0,
            "rx_fire_years_remaining": 0,
            "rx_fire_severity_multiplier": 1.0,
            "last_fertilize_age": None,
            "last_disturbance_age": None,
            "salvage_window": 0,
        }

    def _handle_thin(
        self,
        params: np.ndarray,
        state: StandState,
        scheduled_events: list,
        info: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any], bool]:
        if state.growth_paused:
            return 0.0, {}, False
        if state.last_thin_age is not None and (state.age - state.last_thin_age) < 5.0:
            return 0.0, {}, False
        if state.years_since_replant is not None and state.years_since_replant < 5.0:
            return 0.0, {}, False
        removal_fraction = float(np.clip(params[0], 0.0, 1.0)) * self._thin_max_fraction
        removal_fraction = max(0.25, min(self._thin_max_fraction, removal_fraction))
        scheduled_events.append(ThinningDisturbance(age=state.age, removal_fraction=removal_fraction))
        info["actions"].append({"type": "thin", "fraction": removal_fraction})
        cost = self._thinning_cost(state.age)
        updates = {"last_thin_age": state.age}
        return -cost, updates, True

    def _handle_harvest(
        self,
        params: np.ndarray,
        state: StandState,
        info: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any], list, bool]:
        if state.growth_paused:
            return 0.0, {}, [], False
        harvest_revenue = self._harvest_revenue(state)
        harvest_cost = self._econ_params.costs.get("thin_harvest", 0.0)
        cash = harvest_revenue - harvest_cost
        updates = self._apply_clearcut(state)
        replant_flag = bool(params[0] > 0.5)
        events: list = [
            {
                "type": "harvest",
                "age": state.age,
                "revenue": harvest_revenue,
                "cost": harvest_cost,
            }
        ]
        info["actions"].append({"type": "harvest", "revenue": harvest_revenue, "cost": harvest_cost})
        if replant_flag:
            template = self._replant_template()
            repl_updates = self._apply_replant_values(template)
            updates.update(repl_updates)
            planting_cost = self._econ_params.costs.get("planting", 0.0)
            cash -= planting_cost
            info["actions"].append({"type": "replant", "cost": planting_cost})
        else:
            updates["years_since_replant"] = None
        return cash, updates, events, True

    def _handle_sell(
        self,
        state: StandState,
        info: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any], list]:
        harvest_revenue = self._harvest_revenue(state)
        harvest_cost = self._econ_params.costs.get("thin_harvest", 0.0)
        lev = self._finance.get("lev", 0.0)
        cash = harvest_revenue - harvest_cost + lev
        updates = self._apply_clearcut(state)
        info["actions"].append(
            {"type": "sell", "harvest_revenue": harvest_revenue, "cost": harvest_cost, "lev": lev}
        )
        events = [
            {
                "type": "sell",
                "age": state.age,
                "revenue": harvest_revenue,
                "cost": harvest_cost,
                "lev": lev,
            }
        ]
        return cash, updates, events

    def _handle_salvage(
        self,
        state: StandState,
        info: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any], list, bool]:
        if state.salvage_window <= 0:
            return 0.0, {}, [], False
        salvage_revenue = self._harvest_revenue(state)
        salvage_cost = self._econ_params.costs.get("thin_harvest", 0.0)
        cash = salvage_revenue - salvage_cost
        updates = self._apply_clearcut(state)
        updates["salvage_window"] = 0
        updates["years_since_replant"] = None
        info["actions"].append({"type": "salvage", "revenue": salvage_revenue, "cost": salvage_cost})
        events = [
            {
                "type": "salvage",
                "age": state.age,
                "revenue": salvage_revenue,
                "cost": salvage_cost,
            }
        ]
        return cash, updates, events, True

    def _handle_replant(
        self,
        state: StandState,
        info: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any], bool]:
        if not state.growth_paused:
            return 0.0, {}, False
        template = self._replant_template()
        updates = self._apply_replant_values(template)
        planting_cost = self._econ_params.costs.get("planting", 0.0)
        info["actions"].append({"type": "replant", "cost": planting_cost})
        return -planting_cost, updates, True

    def _handle_fertilize(
        self,
        state: StandState,
        info: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any], bool]:
        if state.growth_paused:
            return 0.0, {}, False
        if state.last_fertilize_age is not None and (state.age - state.last_fertilize_age) < 1.0:
            return 0.0, {}, False
        updates = {
            "fertilizer_years_remaining": 10,
            "fertilizer_effect_strength": 10.0,
            "last_fertilize_age": state.age,
        }
        cost = self._econ_params.costs.get("fertilize", 0.0)
        info["actions"].append({"type": "fertilize", "cost": cost})
        return -cost, updates, True

    def _handle_rx_fire(
        self,
        state: StandState,
        info: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any], bool]:
        if state.growth_paused:
            return 0.0, {}, False
        updates = {
            "rx_fire_years_remaining": 5,
            "rx_fire_severity_multiplier": 0.75,
            "ci": max(0.0, (state.ci or 0.0) * 0.9),
        }
        cost = self._econ_params.costs.get("rx_burn", 0.0)
        info["actions"].append({"type": "rx_fire", "cost": cost})
        return -cost, updates, True

    def _update_financials(self, cash_flow: float) -> Dict[str, float]:
        if not hasattr(self, "_finance"):
            self._finance = {"cash": 0.0, "npv": 0.0, "lev": 0.0}
        self._finance["cash"] += cash_flow
        rate = self._reward_cfg.get("discount_rate", 0.0)
        if rate > 0.0:
            period = self._steps
            discounted = cash_flow / ((1.0 + rate) ** max(1, period))
        else:
            discounted = cash_flow
        self._finance["npv"] += discounted
        rotation = getattr(self._state, "rotation_age_assumption", getattr(self._base_params, "rotation_age_assumption", 30.0))
        if rate > 0.0 and rotation > 0:
            factor = (1.0 + rate) ** rotation
            if factor - 1.0 != 0.0:
                self._finance["lev"] = self._finance["npv"] * factor / (factor - 1.0)
        elif rate == 0.0 and rotation > 0:
            self._finance["lev"] = self._finance["npv"] / max(rotation, 1.0)
        info = {
            "cash_flow": cash_flow,
            "cumulative_cash": self._finance["cash"],
            "npv": self._finance["npv"],
            "lev": self._finance["lev"],
        }
        return info

    def _state_to_obs(self, state: StandState) -> np.ndarray:
        """Convert a :class:`StandState` into the observation vector."""
        active_envelopes = float(len(state.active_envelopes))
        years_since_replant = state.years_since_replant if state.years_since_replant is not None else 0.0
        growth_paused = 1.0 if state.growth_paused else 0.0
        obs = np.array(
            [
                state.age,
                state.tpa,
                state.ba or 0.0,
                state.hd or 0.0,
                state.vol_ob or 0.0,
                state.ci or 0.0,
                active_envelopes,
                growth_paused,
                years_since_replant,
                state.cumulative_cash,
                state.cumulative_discounted_cash,
                state.cumulative_lev,
            ],
            dtype=np.float32,
        )
        return obs

    @property
    def state(self) -> StandState:
        """Return the current internal stand state."""
        if self._state is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")
        return self._state

    @property
    def events(self) -> Tuple[dict, ...]:
        """Return the most recent event logs."""
        return self._last_events


__all__ = ["StandEnv", "ActionType"]


if __name__ == "__main__":
    env = StandEnv(seed=42)
    obs, _ = env.reset()
    cumulative_reward = 0.0
    terminated = False
    while not terminated:
        action = env.action_space.sample()
        obs, reward, terminated, _, info = env.step(action)
        cumulative_reward += reward
        if terminated:
            break
    print(env.render())
    print(f"Cumulative reward: {cumulative_reward:.2f}")
