"""Compare stochastic growth and transition matrices under different catastrophe rates."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from core.pmrc_model import PMRCModel
from core.stochastic_stand import (
    StochasticPMRC,
    StandState,
    StateDiscretizer,
    estimate_transition_matrix,
)


CONFIGS = [
    ("none", "No catastrophes (process noise only)", 1e9),
    ("low", "Low (avg 60 yrs)", 60.0),
    ("medium", "Medium (avg 25 yrs)", 25.0),
    ("high", "High (avg 10 yrs)", 10.0),
]


def simulate_mean_trajectory(
    stochastic: StochasticPMRC,
    init_state: StandState,
    *,
    years: int,
    n_runs: int,
    dt: float = 1.0,
) -> tuple[
    np.ndarray,
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    steps = int(years / dt)
    ages = np.linspace(init_state.age, init_state.age + years, steps + 1)
    hd = np.zeros((n_runs, steps + 1))
    ba = np.zeros_like(hd)
    tpa = np.zeros_like(hd)
    rng = np.random.default_rng(12345)

    for run in range(n_runs):
        state = init_state
        hd[run, 0], ba[run, 0], tpa[run, 0] = state.hd, state.ba, state.tpa
        for step in range(1, steps + 1):
            state, _, _ = stochastic.sample_next_state_with_event(state, dt, rng)
            hd[run, step], ba[run, step], tpa[run, step] = state.hd, state.ba, state.tpa
    hd_mean, hd_std = hd.mean(axis=0), hd.std(axis=0)
    ba_mean, ba_std = ba.mean(axis=0), ba.std(axis=0)
    tpa_mean, tpa_std = tpa.mean(axis=0), tpa.std(axis=0)
    return ages, (hd_mean, hd_std), (ba_mean, ba_std), (tpa_mean, tpa_std)


def plot_growth_comparison(
    ages: np.ndarray,
    series: dict[str, tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]],
    *,
    n_runs: int,
    show_sd: bool = True,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True, dpi=200)
    labels = ["Dominant height (ft)", "Basal area (ft²/ac)", "Trees per acre"]
    colors = ["#27ae60", "#e67e22", "#c0392b", "#8e44ad"]
    for idx, (name, (_, label, _)) in enumerate(zip(series.keys(), CONFIGS)):
        (hd_mean, hd_std), (ba_mean, ba_std), (tpa_mean, tpa_std) = series[name]

        # Means
        axes[0].plot(ages, hd_mean, color=colors[idx], linewidth=2.0, label=label)
        axes[1].plot(ages, ba_mean, color=colors[idx], linewidth=2.0, label=label)
        axes[2].plot(ages, tpa_mean, color=colors[idx], linewidth=2.0, label=label)

        if show_sd:
            # ±1 SD ribbons
            axes[0].fill_between(
                ages,
                hd_mean - hd_std,
                hd_mean + hd_std,
                color=colors[idx],
                alpha=0.15,
            )
            axes[1].fill_between(
                ages,
                ba_mean - ba_std,
                ba_mean + ba_std,
                color=colors[idx],
                alpha=0.15,
            )
            axes[2].fill_between(
                ages,
                tpa_mean - tpa_std,
                tpa_mean + tpa_std,
                color=colors[idx],
                alpha=0.15,
            )
    for ax, ylabel in zip(axes, labels):
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.3)
    axes[-1].set_xlabel("Age (years)")
    axes[0].legend()
    fig.suptitle(f"Mean growth trajectories under different catastrophe rates (n={n_runs})", y=0.99)
    fig.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    fig.savefig("plots/stochastic_growth_by_catastrophe_2.png", bbox_inches="tight")


def plot_transition_matrix(matrix: np.ndarray, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
    im = ax.imshow(matrix, cmap="viridis", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Next state index")
    ax.set_ylabel("Current state index")
    fig.colorbar(im, ax=ax, label="Probability")
    Path("plots").mkdir(exist_ok=True)
    fig.savefig(filename, bbox_inches="tight")


def main() -> None:
    pmrc = PMRCModel(region="ucp")
    init_state = StandState(age=5.0, hd=40.0, tpa=600.0, ba=80.0, si25=60.0, region="ucp")
    age_bins = np.array([0.0, 15.0, 30.0, 45.0, 60.0, 80.0])
    tpa_bins = np.array([0.0, 1e6])  # ignore TPA
    ba_bins = np.array([10.0, 40.0, 70.0, 100.0, 140.0, 200.0])
    discretizer = StateDiscretizer(age_bins, tpa_bins, ba_bins)

    trajectories: dict[
        str,
        tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    ] = {}
    ages_reference: np.ndarray | None = None
    n_runs = 400

    for idx, (key, label, interval) in enumerate(CONFIGS):
        stochastic = StochasticPMRC(
            pmrc,
            sigma_log_ba=0.12,
            sigma_tpa=20.0,
            sigma_log_hd=0.08,
            use_binomial_tpa=False,
            p_mild=0.0,
            severe_mean_interval=interval,
        )

        ages, hd_stats, ba_stats, tpa_stats = simulate_mean_trajectory(
            stochastic,
            init_state,
            years=30,
            n_runs=n_runs,
            dt=1.0,
        )
        trajectories[key] = (hd_stats, ba_stats, tpa_stats)
        if ages_reference is None:
            ages_reference = ages

        matrices = estimate_transition_matrix(
            stochastic,
            discretizer,
            actions=[0],
            dt=1.0,
            n_mc=2000,
            rng=np.random.default_rng(100 + idx),
            si25=60.0,
            region="ucp",
            init_state=init_state,
            steps=30,
        )
        plot_transition_matrix(
            matrices[0],
            f"Transition matrix – {label}",
            f"plots/transition_matrix_{key}.png",
        )

    if ages_reference is not None:
        plot_growth_comparison(ages_reference, trajectories, n_runs=1000, show_sd=False)


if __name__ == "__main__":
    main()
