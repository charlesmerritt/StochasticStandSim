"""Train Stable Baselines3 policies on StandMgmtEnv with optional rollouts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from gymnasium import ActionWrapper, spaces
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_linear_fn

from core.env import EnvConfig, StandMgmtEnv

ACTION_LABELS = [
    "noop",
    "thin_40",
    "thin_60",
    "fert_n200_p1",
    "harvest",
    "plant_600_si60",
    "salvage_0p3",
    "rxfire",
]

DISCRETE_ALGOS = {"ppo", "a2c", "dqn"}
CONTINUOUS_ALGOS = {"ddpg", "sac", "td3"}
ALL_ALGOS = sorted(DISCRETE_ALGOS | CONTINUOUS_ALGOS)


class OneHotActionWrapper(ActionWrapper):
    """Map Box actions to discrete IDs via argmax for continuous-control algorithms."""

    def __init__(self, env: StandMgmtEnv):
        if not isinstance(env.action_space, spaces.Discrete):
            raise TypeError("OneHotActionWrapper requires a discrete action space.")
        super().__init__(env)
        self.n_actions = env.action_space.n
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_actions,), dtype=np.float32)

    def action(self, action: np.ndarray) -> int:
        action_arr = np.asarray(action, dtype=np.float32)
        if action_arr.ndim == 0:
            idx = int(np.clip(int(action_arr.item()), 0, self.n_actions - 1))
        else:
            idx = int(np.argmax(action_arr))
        return idx

    def reverse_action(self, action: int) -> np.ndarray:
        arr = np.zeros(self.n_actions, dtype=np.float32)
        arr[int(action)] = 1.0
        return arr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SB3 agents on StandMgmtEnv.")
    parser.add_argument("--algo", choices=ALL_ALGOS, default="ppo", help="RL algorithm (default: ppo).")
    parser.add_argument("--ddpg", action="store_true", help="Alias for --algo ddpg.")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Total training timesteps.")
    parser.add_argument("--discount-rate", type=float, default=0.003, help="EnvConfig discount rate.")
    parser.add_argument("--growth-reward", type=float, default=0.0, help="Reward shaping weight.")
    parser.add_argument("--horizon-years", type=float, default=100.0, help="Simulation horizon.")
    parser.add_argument("--disturbances", action="store_true", help="Enable stochastic disturbances.")
    parser.add_argument("--model-path", type=str, default="trained_policy.zip", help="Path to save the policy.")
    parser.add_argument("--deterministic", action="store_true", help="Deterministic rollout prints.")
    parser.add_argument("--learning-rate", type=float, help="Initial learning rate.")
    parser.add_argument(
        "--lr-linear",
        action="store_true",
        help="Use a linear schedule from initial lr to 0 over training.",
    )
    parser.add_argument("--rollout-steps", type=int, default=200, help="Number of steps for printed rollout summary.")
    return parser.parse_args()


def make_env(cfg: EnvConfig, algo: str) -> StandMgmtEnv:
    env: StandMgmtEnv = StandMgmtEnv(cfg)
    if algo in CONTINUOUS_ALGOS:
        env = OneHotActionWrapper(env)
    return Monitor(env)


def build_model(
    algo: str,
    env: StandMgmtEnv,
    learning_rate: float | None = None,
    linear_schedule: bool = False,
) -> BaseAlgorithm:
    algo = algo.lower()
    constructors: Dict[str, type[BaseAlgorithm]] = {
        "ppo": PPO,
        "a2c": A2C,
        "dqn": DQN,
        "ddpg": DDPG,
        "sac": SAC,
        "td3": TD3,
    }
    if algo not in constructors:
        raise ValueError(f"Unsupported algorithm '{algo}'. Supported values: {', '.join(ALL_ALGOS)}")
    kwargs: Dict[str, Any] = {"verbose": 1, "device": "cpu"}
    if learning_rate is not None:
        lr_value: float | callable
        if linear_schedule:
            lr_value = get_linear_fn(learning_rate, 0.0, 1.0)
        else:
            lr_value = learning_rate
        kwargs["learning_rate"] = lr_value
    elif algo in {"dqn", "ddpg", "sac", "td3"}:
        kwargs.setdefault("learning_rate", 1e-3)
    return constructors[algo]("MlpPolicy", env, **kwargs)


def decode_action(action_raw: Any) -> int:
    arr = np.asarray(action_raw)
    if arr.ndim == 0:
        return int(arr.item())
    return int(np.argmax(arr))


def rollout_summary(model: BaseAlgorithm, env: StandMgmtEnv, steps: int, deterministic: bool) -> List[dict[str, float]]:
    obs, _ = env.reset()
    data: List[dict[str, float]] = []
    for step_idx in range(steps):
        action_raw, _ = model.predict(obs, deterministic=deterministic)
        action_id = decode_action(action_raw)
        obs, reward, done, _, info = env.step(action_id)
        data.append(
            {
                "step": step_idx,
                "age": float(info.get("age", 0.0)),
                "reward": float(reward),
                "action": action_id,
                "cashflow": float(info.get("cashflow_now") or 0.0),
            }
        )
        if done:
            break
    return data


def main() -> None:
    args = parse_args()
    algo = "ddpg" if args.ddpg else args.algo.lower()

    env_cfg = EnvConfig(
        discount_rate=args.discount_rate,
        growth_reward_weight=args.growth_reward,
        horizon_years=args.horizon_years,
        disturbance_enabled=args.disturbances,
    )

    training_env = make_env(env_cfg, algo)
    model = build_model(algo, training_env)
    model.learn(total_timesteps=args.timesteps)
    model.save(args.model_path)

    eval_env = make_env(env_cfg, algo)
    loaded_model = model.__class__.load(Path(args.model_path), env=eval_env)
    summary = rollout_summary(loaded_model, eval_env, args.rollout_steps, deterministic=args.deterministic)

    print("\n=== Rollout summary (first 10 steps) ===")
    for row in summary[:10]:
        print(
            f"Step {row['step']:3d} | age={row['age']:.1f} | action={ACTION_LABELS[row['action']]} | "
            f"reward={row['reward']:.2f} | cashflow={row['cashflow']:.2f}"
        )

    if summary:
        counts = {idx: 0 for idx in range(len(ACTION_LABELS))}
        for row in summary:
            counts[row["action"]] += 1
        print("\n=== Action frequencies ===")
        for idx, label in enumerate(ACTION_LABELS):
            print(f"  {idx} ({label}): {counts[idx]}")


if __name__ == "__main__":
    main()
