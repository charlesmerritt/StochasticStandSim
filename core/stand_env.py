from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, Mapping, MutableMapping, Optional, Tuple, Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Advanced logic is delegated to these modules.
# Implement their functions separately with matching signatures.
from core.growth.growth_models import grow_one_step  # (state, action, params) -> dict
from core.disturbance.envelopes import apply_deterministic_risk  # (state, risk_cfg) -> dict
from core.economics.reward_functions import step_reward  # (state, action, next_state, econ_cfg) -> float
from core.economics.valuation import bellman_value  # (state, horizon, discount, econ_cfg) -> float


@dataclass
class StandConfig:
    """Configuration parameters for :class:`StandEnv`.

    The dataclass captures the information required to fully specify an
    initial stand and the knobs necessary to delegate work to specialist
    growth, disturbance, and economic modules.  Default values provide a
    lightweight, deterministic environment that works out-of-the-box for unit
    tests while still allowing downstream projects to inject richer behaviour
    via the provided configuration hooks.
    """

    # --- Stand initialisation -------------------------------------------------
    initial_age: float = 5.0
    initial_tpa: float = 600.0  # trees per acre
    initial_basal_area: float = 40.0  # ft^2/acre
    initial_biomass: float = 25.0  # tons/acre
    site_index: float = 120.0
    risk_level: float = 0.01

    # --- Planning/problem definition -----------------------------------------
    horizon: int = 60
    discount_rate: float = 0.03

    # --- Action policy controls ----------------------------------------------
    min_thinning_interval: int = 5
    min_thinning_pct: float = 0.05
    max_thinning_pct: float = 0.6
    precommercial_cutoff_age: float = 15.0
    precommercial_min_pct: float = 0.1

    # --- Delegated configuration blobs ---------------------------------------
    growth_config: MutableMapping[str, Any] = field(default_factory=dict)
    disturbance_config: MutableMapping[str, Any] = field(default_factory=dict)
    economic_config: MutableMapping[str, Any] = field(default_factory=dict)

    # Optional helper hooks (useful for dependency injection in tests).
    state_encoder: Optional[Callable[[Mapping[str, Any], "StandConfig"], np.ndarray]] = None
    observation_space_factory: Optional[Callable[["StandConfig"], gym.Space]] = None
    action_space_factory: Optional[Callable[["StandConfig"], gym.Space]] = None


class StandEnv(gym.Env):
    """Gymnasium environment representing a managed forest stand."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: Optional[StandConfig] = None, seed: Optional[int] = None):
        super().__init__()

        self.config = config or StandConfig()
        self._rng = np.random.default_rng(seed)

        # Spaces ----------------------------------------------------------------
        self.observation_space = self._build_observation_space()
        self.action_space = self._build_action_space()

        # Book-keeping ----------------------------------------------------------
        self._state: Dict[str, Any] = {}
        self._step_count: int = 0
        self._terminated: bool = False
        self._truncated: bool = False
        self._last_thin_age: float = -np.inf

        if seed is not None:
            # Align with Gymnasium's seeding semantics by performing a reset.
            self.reset(seed=seed)

    # ------------------------------------------------------------------ Spaces
    def _build_observation_space(self) -> gym.Space:
        if self.config.observation_space_factory is not None:
            return self.config.observation_space_factory(self.config)

        # Age (0-horizon), biomass (>=0), density, basal area, site index,
        # accumulated risk (probability), and discounted value proxy.
        low = np.array([0.0, 0.0, 0.0, 0.0, 50.0, 0.0, 0.0], dtype=np.float32)
        high = np.array(
            [
                max(self.config.horizon * 2, self.config.initial_age + 100.0),
                1000.0,
                1200.0,
                400.0,
                200.0,
                1.0,
                1e6,
            ],
            dtype=np.float32,
        )
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _build_action_space(self) -> gym.Space:
        if self.config.action_space_factory is not None:
            return self.config.action_space_factory(self.config)

        # Action tuple: [thin_pct, fert_n, fert_p]
        low = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    # ----------------------------------------------------------------- Helpers
    def _initial_state(self) -> Dict[str, Any]:
        return {
            "age": float(self.config.initial_age),
            "tpa": float(self.config.initial_tpa),
            "basal_area": float(self.config.initial_basal_area),
            "biomass": float(self.config.initial_biomass),
            "site_index": float(self.config.site_index),
            "risk": float(self.config.risk_level),
            "value": 0.0,
            "catastrophic": False,
        }

    def _encode_state(self, state: Mapping[str, Any]) -> np.ndarray:
        if self.config.state_encoder is not None:
            return self.config.state_encoder(state, self.config)

        # Fallback deterministic encoding for tests.
        vec = np.array(
            [
                state.get("age", 0.0),
                state.get("biomass", 0.0),
                state.get("tpa", 0.0),
                state.get("basal_area", 0.0),
                state.get("site_index", 0.0),
                state.get("risk", 0.0),
                state.get("value", 0.0),
            ],
            dtype=np.float32,
        )
        return vec

    def _apply_thinning_constraints(
        self, action: np.ndarray, state: Mapping[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        adjustments: Dict[str, Any] = {}
        thin_pct = float(action[0])
        age = float(state.get("age", 0.0))

        if thin_pct < 0.0:
            thin_pct = 0.0
        if thin_pct > self.config.max_thinning_pct:
            adjustments["thin_clipped"] = thin_pct
            thin_pct = self.config.max_thinning_pct

        interval = age - self._last_thin_age
        if thin_pct > 0.0 and interval < self.config.min_thinning_interval:
            adjustments["thin_interval_violation"] = interval
            thin_pct = 0.0

        # Minimum removal depending on commercial status.
        if thin_pct > 0.0:
            if age < self.config.precommercial_cutoff_age:
                if thin_pct < self.config.precommercial_min_pct:
                    adjustments["precommercial_min_violation"] = thin_pct
                    thin_pct = 0.0
            else:
                if thin_pct < self.config.min_thinning_pct:
                    adjustments["commercial_min_violation"] = thin_pct
                    thin_pct = 0.0

        adjusted = np.array([thin_pct, float(action[1]), float(action[2])], dtype=np.float32)
        return adjusted, adjustments

    def _maybe_call_reset_helpers(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(self.config.growth_config, MutableMapping):
            helper = self.config.growth_config.get("reset")
            if callable(helper):
                state.update(helper(state, self.config))

        if isinstance(self.config.disturbance_config, MutableMapping):
            helper = self.config.disturbance_config.get("reset")
            if callable(helper):
                state.update(helper(state, self.config))

        if isinstance(self.config.economic_config, MutableMapping):
            helper = self.config.economic_config.get("reset")
            if callable(helper):
                state.update(helper(state, self.config))

        return state

    # -------------------------------------------------------------- Gym API
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._state = self._initial_state()
        self._maybe_call_reset_helpers(self._state)

        self._step_count = 0
        self._terminated = False
        self._truncated = False
        self._last_thin_age = self._state.get("age", 0.0) - self.config.min_thinning_interval

        obs = self._encode_state(self._state)
        info = {"state": dict(self._state)}
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self._terminated or self._truncated:
            raise RuntimeError("Cannot call step() once the episode has terminated")

        action = np.asarray(action, dtype=np.float32)
        if action.shape != (3,):
            raise ValueError("Action must be a 3-vector: [thin_pct, fert_N, fert_P]")

        if not self.action_space.contains(action):
            action = np.clip(action, self.action_space.low, self.action_space.high)

        adjusted_action, constraint_info = self._apply_thinning_constraints(action, self._state)

        growth_out = grow_one_step(self._state, adjusted_action, self.config.growth_config)
        if isinstance(growth_out, tuple):
            next_state_candidate, growth_info = growth_out
        elif isinstance(growth_out, Mapping):
            next_state_candidate = growth_out.get("state", growth_out)
            growth_info = growth_out.get("info", {})
        else:
            next_state_candidate = growth_out
            growth_info = {}

        risk_out = apply_deterministic_risk(next_state_candidate, self.config.disturbance_config)
        if isinstance(risk_out, tuple):
            next_state, risk_info = risk_out
        elif isinstance(risk_out, Mapping):
            next_state = risk_out.get("state", risk_out)
            risk_info = risk_out.get("info", {})
        else:
            next_state = risk_out
            risk_info = {}

        reward = step_reward(self._state, adjusted_action, next_state, self.config.economic_config)

        self._state = dict(next_state)
        self._step_count += 1

        if adjusted_action[0] > 0.0:
            self._last_thin_age = self._state.get("age", 0.0)

        catastrophic = bool(self._state.get("catastrophic", False))
        horizon_reached = self._step_count >= self.config.horizon

        self._terminated = catastrophic or horizon_reached
        self._truncated = False

        obs = self._encode_state(self._state)

        info: Dict[str, Any] = {
            "growth": growth_info,
            "disturbance": risk_info,
            "constraints": constraint_info,
            "state": dict(self._state),
            "adjusted_action": adjusted_action,
        }
        if catastrophic:
            info["event"] = "catastrophic_loss"

        return obs, float(reward), self._terminated, self._truncated, info

    # -------------------------------------------------------------- Utilities
    def render(self) -> str:
        if not self._state:
            return "StandState(<uninitialised>)"
        return (
            "StandState(age={age:.1f}, tpa={tpa:.1f}, ba={ba:.1f}, risk={risk:.3f})".format(
                age=self._state.get("age", 0.0),
                tpa=self._state.get("tpa", 0.0),
                ba=self._state.get("basal_area", 0.0),
                risk=self._state.get("risk", 0.0),
            )
        )

    def close(self) -> None:  # pragma: no cover - simple alias for compatibility.
        return None

    # Convenience accessors ----------------------------------------------------
    @property
    def state(self) -> Dict[str, Any]:
        return dict(self._state)

    def asdict(self) -> Dict[str, Any]:
        return asdict(self.config)

    def value(self, horizon: Optional[int] = None, discount: Optional[float] = None) -> float:
        """Return the Bellman value of the current stand state."""

        horizon = horizon if horizon is not None else self.config.horizon
        discount = discount if discount is not None else self.config.discount_rate
        return float(bellman_value(self._state, horizon, discount, self.config.economic_config))
