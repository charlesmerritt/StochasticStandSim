#!/usr/bin/env python3
"""Train one policy per supported algorithm and generate rollout plots."""

from __future__ import annotations

import subprocess
from pathlib import Path


ALGOS = ["ppo", "a2c", "dqn", "ddpg", "sac", "td3"]

TIMESTEPS = 150_000
LEARNING_RATE = 3e-4
MODELS_DIR = Path("rl/models")
PLOTS_DIR = Path("plots")


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def train_and_plot(algo: str) -> None:
    model_path = MODELS_DIR / f"{algo}_linear.zip"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    train_cmd = [
        "python",
        "-m",
        "rl.train",
        "--algo",
        algo,
        "--timesteps",
        str(TIMESTEPS),
        "--learning-rate",
        str(LEARNING_RATE),
        "--lr-linear",
        "--model-path",
        str(model_path),
    ]
    run(train_cmd)

    plot_path = PLOTS_DIR / f"policy_rollout_{algo}_linear.png"
    plot_cmd = [
        "python",
        "main.py",
        "--plot",
        "policy_rollout",
        "--model-path",
        str(model_path),
        "--output-dir",
        str(PLOTS_DIR),
        "--rollout-steps",
        "200",
        "--deterministic",
    ]
    run(plot_cmd)

    print(f"Saved rollout plot to {plot_path}")


def main() -> None:
    for algo in ALGOS:
        print(f"=== Training {algo} ===")
        train_and_plot(algo)


if __name__ == "__main__":
    main()
