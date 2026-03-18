"""Debug dominant height behavior under deterministic and stochastic PMRC.

This script plots dominant height vs age for a single stand, comparing:
- Deterministic PMRCModel (no noise, no catastrophes)
- StochasticPMRC wrapper with current settings (process noise + catastrophes)

Use this to diagnose cases where, for example, an si25=70 stand appears
to have HD≈140 ft at age 25.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Allow running from repo root or examples/
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from core.pmrc_model import PMRCModel
from core.stochastic_model import StochasticPMRC, StandState


def simulate_hd_trajectory_deterministic(
    pmrc: PMRCModel,
    init_state: StandState,
    *,
    years: int,
    dt: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic HD trajectory using PMRCModel.hd_project only."""
    steps = int(years / dt)
    ages = np.linspace(init_state.age, init_state.age + years, steps + 1)
    hd = np.zeros_like(ages)
    hd[0] = init_state.hd
    age = init_state.age
    h = init_state.hd
    for i in range(1, steps + 1):
        age_next = age + dt
        h = pmrc.hd_project(age1=age, hd1=h, age2=age_next)
        hd[i] = h
        age = age_next
    return ages, hd


def simulate_hd_trajectory_stochastic(
    stochastic: StochasticPMRC,
    init_state: StandState,
    *,
    years: int,
    n_runs: int,
    dt: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and SD of HD trajectory under the stochastic wrapper."""
    steps = int(years / dt)
    ages = np.linspace(init_state.age, init_state.age + years, steps + 1)
    hd = np.zeros((n_runs, steps + 1))
    rng = np.random.default_rng(12345)
    for r in range(n_runs):
        state = init_state
        hd[r, 0] = state.hd
        for i in range(1, steps + 1):
            state, _, _ = stochastic.sample_next_state_with_event(state, dt, rng)
            hd[r, i] = state.hd
    return ages, hd.mean(axis=0), hd.std(axis=0)


def main() -> None:
    pmrc = PMRCModel(region="ucp")

    # Configure the stand you want to debug
    si25 = 70.0
    init_age = 5.0
    # Option 1: specify hd directly
    #init_hd = 10.0
    # Option 2 (commented): derive hd from si25 at init_age using the same logic as _hd_from_site
    num = 1.0 - np.exp(-pmrc.k * init_age)
    den = 1.0 - np.exp(-pmrc.k * 25.0)
    init_hd = max(1.0, si25 * (num / den) ** pmrc.m)

    init_state = StandState(
        age=init_age,
        hd=init_hd,
        tpa=600.0,
        ba=80.0,
        si25=si25,
        region="ucp",
    )

    # Deterministic trajectory
    years = 40
    ages_det, hd_det = simulate_hd_trajectory_deterministic(pmrc, init_state, years=years, dt=1.0)

    # Stochastic trajectory with your current settings
    stochastic = StochasticPMRC(
        pmrc,
        sigma_log_ba=0.12,
        sigma_tpa=20.0,
        sigma_log_hd=0.08,
        use_binomial_tpa=False,
        p_mild=0.0,
        severe_mean_interval=25.0,  # adjust if you want to effectively remove catastrophes
    )
    ages_sto, hd_mean, hd_std = simulate_hd_trajectory_stochastic(
        stochastic,
        init_state,
        years=years,
        n_runs=400,
        dt=1.0,
    )

    # Sanity check: report HD at age ~25
    target_age = 25.0
    idx_det = int(round(target_age - init_age))
    print(f"Deterministic HD at age {target_age}: {hd_det[idx_det]:.2f} ft (si25={si25})")
    print(
        f"Stochastic mean HD at age {target_age}: {hd_mean[idx_det]:.2f} "+
        f"± {hd_std[idx_det]:.2f} ft (n=400)"
    )

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
    ax.plot(ages_det, hd_det, label="Deterministic PMRC hd_project", color="#34495e", linewidth=2.0)
    ax.plot(ages_sto, hd_mean, label="Stochastic mean HD", color="#e74c3c", linewidth=2.0)
    ax.fill_between(
        ages_sto,
        hd_mean - hd_std,
        hd_mean + hd_std,
        color="#e74c3c",
        alpha=0.2,
        label="Stochastic ±1 SD",
    )
    ax.axvline(25.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Dominant height (ft)")
    ax.set_title(f"Dominant height debug (si25={si25})")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    Path("plots").mkdir(exist_ok=True)
    fig.savefig("plots/hd_debug_demo.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
