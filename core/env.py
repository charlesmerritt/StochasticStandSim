"""Gymnasium environment for thinning control under stochastic disturbances.

This module exposes :class:`StandEnv`, a lightweight wrapper around the
growth and disturbance models defined in :mod:`core.growth` and
:mod:`core.disturbances`.  The environment follows the Gymnasium API so it can
be connected directly to Stable-Baselines3 (SB3) algorithms.
"""

from __future__ import annotations

from dataclasses import replace
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


class StandEnv(gym.Env[np.ndarray, np.ndarray]):
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

        # Gymnasium RNG helper
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        # Synchronise the shared RNG used in modules relying on core.rng.
        global_rng(seed)

        # Observations: age, TPA, BA, HD, volume, competition index, active envelopes
        obs_low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array(
            [
                self._max_age + 10.0,  # allow some slack beyond terminal age
                2000.0,  # trees per acre
                600.0,  # basal area
                150.0,  # dominant height
                5000.0,  # volume
                1.0,  # competition index
                10.0,  # active envelopes count
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

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
        self._last_events = ()
        self._steps = 0

        obs = self._state_to_obs(self._state)
        info = {"events": self._last_events}
        return obs, info

    def step(
        self, action: np.ndarray | Iterable[float] | float
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance the stand one year using the provided action."""
        if self._state is None:
            raise RuntimeError("Environment must be reset before calling step().")

        raw_action = self._extract_action(action)
        removal_fraction = min(raw_action * self._thin_max_fraction, self._thin_max_fraction)

        scheduled_events = list(self._state.pending_disturbances)
        if removal_fraction > 0.0:
            scheduled_events.append(ThinningDisturbance(age=self._state.age, removal_fraction=removal_fraction))

        # Sample stochastic disturbances.
        disturbance_events = self._sample_disturbances()
        scheduled_events.extend(disturbance_events)
        if scheduled_events:
            scheduled_events.sort(key=lambda ev: ev.age)
            self._state = replace(self._state, pending_disturbances=tuple(scheduled_events))

        prev_state = self._state
        next_state, _, event_logs, _, _, _ = step_with_log(prev_state, dt=1.0, cfg=self._growth_cfg)

        self._state = next_state
        self._last_events = event_logs
        self._steps += 1

        reward, breakdown = self._compute_reward(prev_state, next_state, event_logs)

        terminated = bool(next_state.age >= self._max_age)
        truncated = False

        obs = self._state_to_obs(next_state)
        info = {"events": event_logs, "reward_breakdown": breakdown}
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
            f"| hd={state.hd or 0.0:.1f} | vol={state.vol_ob or 0.0:.1f}"
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

    def _extract_action(self, action: np.ndarray | Iterable[float] | float) -> float:
        """Convert user supplied action into a scalar in [0, 1]."""
        if isinstance(action, (float, int)):
            value = float(action)
        else:
            arr = np.asarray(action, dtype=np.float32)
            if arr.ndim == 0:
                value = float(arr.item())
            else:
                value = float(arr[0])
        return float(np.clip(value, 0.0, 1.0))

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

    def _compute_reward(
        self,
        prev_state: StandState,
        next_state: StandState,
        event_logs: Tuple[dict, ...],
    ) -> Tuple[float, Dict[str, float]]:
        """Derive per-step reward components."""
        prev_volume = prev_state.vol_ob or 0.0
        next_volume = next_state.vol_ob or 0.0
        current_volume = prev_volume
        harvested = 0.0
        disturbance_loss = 0.0

        for event in event_logs:
            before = float(event.get("vol_before", current_volume))
            after = float(event.get("vol_after", before))
            current_volume = after
            if event["type"] == "thinning":
                harvested += max(0.0, before - after)
            else:
                disturbance_loss += max(0.0, before - after)

        growth_increment = max(0.0, next_volume - current_volume)

        price = self._reward_cfg["price_per_volume"]
        growth_weight = self._reward_cfg["growth_weight"]
        precommercial_age = self._reward_cfg.get("precommercial_age", 0.0)
        thin_key = "thinning_cost_precommercial" if prev_state.age < precommercial_age else "thinning_cost"
        thinning_cost = self._reward_cfg.get(thin_key, 0.0) if harvested > 0.0 else 0.0

        value = price * (growth_weight * growth_increment + harvested - disturbance_loss)
        net_value = value - thinning_cost

        discount_rate = self._reward_cfg.get("discount_rate", 0.0)
        if discount_rate > 0.0:
            discount_factor = (1.0 / (1.0 + discount_rate)) ** self._steps
        else:
            discount_factor = 1.0
        reward = net_value * discount_factor

        breakdown = {
            "growth_increment": growth_increment,
            "harvested_volume": harvested,
            "disturbance_loss": disturbance_loss,
            "gross_value": value,
            "thinning_cost": thinning_cost,
            "net_value": net_value,
            "discount_factor": discount_factor,
        }
        return reward, breakdown

    def _state_to_obs(self, state: StandState) -> np.ndarray:
        """Convert a :class:`StandState` into the observation vector."""
        active_envelopes = float(len(state.active_envelopes))
        obs = np.array(
            [
                state.age,
                state.tpa,
                state.ba or 0.0,
                state.hd or 0.0,
                state.vol_ob or 0.0,
                state.ci or 0.0,
                active_envelopes,
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


__all__ = ["StandEnv"]


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
