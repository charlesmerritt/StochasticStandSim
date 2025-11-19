"""Monte Carlo growth trajectories using the stochastic PMRC model."""

from __future__ import annotations

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from core.pmrc_model import PMRCModel
from core.stochastic_stand import StochasticPMRC, StandState


def simulate_trajectories(
    stochastic: StochasticPMRC,
    init_state: StandState,
    *,
    years: int,
    n_runs: int,
    dt: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[list[float]]]:
    """Run Monte Carlo simulations returning state arrays and disturbance info."""
    steps = int(years / dt)
    ages = np.linspace(init_state.age, init_state.age + years, steps + 1)
    hd = np.zeros((n_runs, steps + 1))
    ba = np.zeros_like(hd)
    tpa = np.zeros_like(hd)
    disturbance_times: list[list[float]] = [[] for _ in range(n_runs)]
    rng = np.random.default_rng(1234)

    for run in range(n_runs):
        state = init_state
        hd[run, 0], ba[run, 0], tpa[run, 0] = state.hd, state.ba, state.tpa
        for step in range(1, steps + 1):
            state, level, event_age = stochastic.sample_next_state_with_event(state, dt, rng)
            if level == "severe" and event_age is not None:
                disturbance_times[run].append(event_age)
            hd[run, step], ba[run, step], tpa[run, step] = state.hd, state.ba, state.tpa
    return ages, hd, ba, tpa, disturbance_times


def plot_growth(
    ages: np.ndarray,
    single: tuple[np.ndarray, np.ndarray, np.ndarray, list[float]],
    ensemble_mean: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    output_path: Path,
) -> None:
    """Plot single trajectory and ensemble mean side-by-side."""
    hd_single, ba_single, tpa_single, disturbances = single
    hd_mean, ba_mean, tpa_mean = ensemble_mean

    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True, dpi=200)
    labels = ["Dominant height (ft)", "Basal area (ft²/ac)", "Trees per acre"]
    colors = ["#2c3e50", "#c0392b", "#16a085"]
    series_single = [hd_single, ba_single, tpa_single]
    series_mean = [hd_mean, ba_mean, tpa_mean]

    for row in range(3):
        axes[row, 0].plot(ages, series_single[row], color=colors[row], linewidth=2.0)
        axes[row, 0].set_ylabel(labels[row])
        axes[row, 0].grid(True, linestyle="--", alpha=0.3)
        for event_age in disturbances:
            axes[row, 0].axvline(event_age, color="#e74c3c", linestyle="--", alpha=0.5)

        axes[row, 1].plot(ages, series_mean[row], color=colors[row], linewidth=2.0)
        axes[row, 1].grid(True, linestyle="--", alpha=0.3)

    axes[-1, 0].set_xlabel("Age (years)")
    axes[-1, 1].set_xlabel("Age (years)")
    axes[0, 0].set_title("Single stochastic run")
    axes[0, 1].set_title("Mean of many runs")
    fig.suptitle("Stochastic PMRC growth trajectories", y=0.99)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")


def main() -> None:
    pmrc = PMRCModel(region="ucp")
    stochastic = StochasticPMRC(pmrc, sigma_log_ba=0.12, sigma_tpa=20.0, p_mild=0.02, severe_mean_interval=25.0)
    init = StandState(age=5.0, hd=40.0, tpa=600.0, ba=80.0, si25=60.0, region="ucp")
    ages, hd_runs, ba_runs, tpa_runs, disturbance_times = simulate_trajectories(
        stochastic,
        init,
        years=70,
        n_runs=1000,
        dt=1.0,
    )
    single = (hd_runs[0], ba_runs[0], tpa_runs[0], disturbance_times[0])
    mean_series = (hd_runs.mean(axis=0), ba_runs.mean(axis=0), tpa_runs.mean(axis=0))
    plot_growth(ages, single, mean_series, output_path=Path("plots") / "stochastic_growth.png")


if __name__ == "__main__":
    main()
