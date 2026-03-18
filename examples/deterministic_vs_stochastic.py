"""Compare deterministic PMRC growth with stochastic wrapper trajectories.

This figure demonstrates how the stochastic wrapper adds interpretable
Gaussian noise to the deterministic PMRC model, showing the envelope of
possible futures around the expected trajectory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from core.pmrc_model import PMRCModel
from core.stochastic_model import StochasticPMRC, StandState, NoiseConfig


def make_init_state(
    pmrc: PMRCModel,
    si25: float = 60.0,
    tpa0: float = 500.0,
    region: str = "ucp",
) -> StandState:
    """Create initial stand state at age 1."""
    age0 = 1.0
    hd0 = si25 * ((1.0 - np.exp(-pmrc.k * age0)) / (1.0 - np.exp(-pmrc.k * 25.0))) ** pmrc.m
    ba0 = pmrc.ba_predict(age0, tpa0, hd0, region=region)
    return StandState(age=age0, hd=hd0, tpa=tpa0, ba=ba0, si25=si25, region=region)


def simulate_deterministic(
    pmrc: PMRCModel,
    init_state: StandState,
    horizon: int,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Run deterministic PMRC projection."""
    ages, ba, tpa, hd = [init_state.age], [init_state.ba], [init_state.tpa], [init_state.hd]
    state = init_state
    for _ in range(horizon):
        age2 = state.age + 1.0
        hd2 = pmrc.hd_project(state.age, state.hd, age2)
        tpa2 = pmrc.tpa_project(state.tpa, state.si25, state.age, age2)
        ba2 = pmrc.ba_project(state.age, state.tpa, tpa2, state.ba, state.hd, hd2, age2, region=str(state.region))
        state = StandState(age=age2, hd=hd2, tpa=tpa2, ba=ba2, si25=state.si25, region=state.region)
        ages.append(state.age)
        ba.append(state.ba)
        tpa.append(state.tpa)
        hd.append(state.hd)
    return ages, ba, tpa, hd


def simulate_stochastic_ensemble(
    stoch_pmrc: StochasticPMRC,
    init_state: StandState,
    horizon: int,
    n_runs: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run stochastic ensemble (no disturbances, just process noise)."""
    ba_runs = np.zeros((n_runs, horizon + 1))
    tpa_runs = np.zeros((n_runs, horizon + 1))
    hd_runs = np.zeros((n_runs, horizon + 1))

    for run in range(n_runs):
        state = StandState(
            age=init_state.age,
            hd=init_state.hd,
            tpa=init_state.tpa,
            ba=init_state.ba,
            si25=init_state.si25,
            region=init_state.region,
        )
        ba_runs[run, 0] = state.ba
        tpa_runs[run, 0] = state.tpa
        hd_runs[run, 0] = state.hd
        for step in range(1, horizon + 1):
            state = stoch_pmrc.sample_next_state(state, dt=1.0, rng=rng)
            ba_runs[run, step] = state.ba
            tpa_runs[run, step] = state.tpa
            hd_runs[run, step] = state.hd

    return ba_runs, tpa_runs, hd_runs


def plot_comparison(
    det_ages: list[float],
    det_ba: list[float],
    det_tpa: list[float],
    det_hd: list[float],
    stoch_ba: np.ndarray,
    stoch_tpa: np.ndarray,
    stoch_hd: np.ndarray,
    output_path: Path,
) -> None:
    """Plot deterministic vs stochastic comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    ages = np.array(det_ages)

    # Panel A: Basal Area
    ax = axes[0]
    for i in range(min(50, stoch_ba.shape[0])):
        ax.plot(ages, stoch_ba[i], color="steelblue", alpha=0.12, linewidth=0.7)
    ax.fill_between(
        ages,
        np.percentile(stoch_ba, 5, axis=0),
        np.percentile(stoch_ba, 95, axis=0),
        color="steelblue",
        alpha=0.2,
        label="90% CI",
    )
    ax.plot(ages, det_ba, color="darkred", linewidth=2.5, label="Deterministic PMRC")
    ax.plot(ages, stoch_ba.mean(axis=0), color="steelblue", linewidth=2, linestyle="--", label="Stochastic mean")
    ax.set_xlabel("Stand Age (years)")
    ax.set_ylabel("Basal Area (ft²/ac)")
    ax.set_title("(A) Basal Area")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)

    # Panel B: Trees Per Acre
    ax = axes[1]
    for i in range(min(50, stoch_tpa.shape[0])):
        ax.plot(ages, stoch_tpa[i], color="forestgreen", alpha=0.12, linewidth=0.7)
    ax.fill_between(
        ages,
        np.percentile(stoch_tpa, 5, axis=0),
        np.percentile(stoch_tpa, 95, axis=0),
        color="forestgreen",
        alpha=0.2,
        label="90% CI",
    )
    ax.plot(ages, det_tpa, color="darkred", linewidth=2.5, label="Deterministic PMRC")
    ax.plot(ages, stoch_tpa.mean(axis=0), color="forestgreen", linewidth=2, linestyle="--", label="Stochastic mean")
    ax.set_xlabel("Stand Age (years)")
    ax.set_ylabel("Trees Per Acre")
    ax.set_title("(B) Survival / Mortality")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)

    # Panel C: Dominant Height
    ax = axes[2]
    for i in range(min(50, stoch_hd.shape[0])):
        ax.plot(ages, stoch_hd[i], color="darkorange", alpha=0.12, linewidth=0.7)
    ax.fill_between(
        ages,
        np.percentile(stoch_hd, 5, axis=0),
        np.percentile(stoch_hd, 95, axis=0),
        color="darkorange",
        alpha=0.2,
        label="90% CI",
    )
    ax.plot(ages, det_hd, color="darkred", linewidth=2.5, label="Deterministic PMRC")
    ax.plot(ages, stoch_hd.mean(axis=0), color="darkorange", linewidth=2, linestyle="--", label="Stochastic mean")
    ax.set_xlabel("Stand Age (years)")
    ax.set_ylabel("Dominant Height (ft)")
    ax.set_title("(C) Height Growth")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle("Deterministic PMRC vs Stochastic Wrapper (Process Noise Only)", y=1.02, fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    pmrc = PMRCModel(region="ucp")
    init_state = make_init_state(pmrc, si25=60.0, tpa0=500.0)
    horizon = 40

    # Deterministic trajectory
    det_ages, det_ba, det_tpa, det_hd = simulate_deterministic(pmrc, init_state, horizon)

    # Stochastic ensemble (disable disturbances to isolate process noise)
    stoch_pmrc = StochasticPMRC(
        pmrc,
        noise_config=NoiseConfig(ba_std=5.0, tpa_std=25.0, hd_std=1.0, clip_std=2.0),
        use_gaussian_noise=True,
        p_mild=0.0,
        severe_mean_interval=10000.0,
    )
    rng = np.random.default_rng(42)
    stoch_ba, stoch_tpa, stoch_hd = simulate_stochastic_ensemble(stoch_pmrc, init_state, horizon, n_runs=200, rng=rng)

    output_path = Path("plots") / "deterministic_vs_stochastic.png"
    plot_comparison(det_ages, det_ba, det_tpa, det_hd, stoch_ba, stoch_tpa, stoch_hd, output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
