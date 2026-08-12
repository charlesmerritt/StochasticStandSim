"""Compare mean growth trajectories under different catastrophe rates.

Runs 1000 Monte Carlo trajectories for each of four disturbance regimes
(no catastrophes, 30-year, 20-year, 10-year return intervals) and plots
the mean HD, BA, and TPA over a 35-year rotation.

Uses the new ScenarioConfig + run_batch framework. Initial HD is correctly
derived from SI25 via Chapman-Richards (the old script hardcoded hd=40 at
age 5 with si25=60, which inflated projected HD by ~2×).

Parameters:
    SI25 = 80 ft, TPA₀ = 850, age₀ = 5, rotation = 35 yr, region = UCP
    Process noise: λ = 1.0 (full noise, NoiseParams defaults)
    Disturbance: p_dist ∈ {0, 1/30, 1/20, 1/10}, severity Beta(3.6, 8.4)
    Thinning: none
    Seed: 42
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.config import ScenarioConfig
from core.disturbances import DisturbanceParams
from core.process_noise import NoiseParams
from core.simulate import BatchResult, run_batch
from core.viz import plot_disturbance_regime_comparison, save_figure


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

N_RUNS = 1000
SEED = 42

NOISE = NoiseParams(lambda_proc=1.0)

SCENARIOS: list[tuple[str, str, ScenarioConfig]] = [
    (
        "noise_only",
        "No catastrophes (process noise only)",
        ScenarioConfig(
            name="noise_only",
            scenario_type="stochastic",
            noise_params=NOISE,
            disturbance_params=DisturbanceParams(p_dist=0.0),
        ),
    ),
    (
        "low_30",
        "Low (avg 30 yrs)",
        ScenarioConfig(
            name="low_30",
            scenario_type="stochastic",
            noise_params=NOISE,
            disturbance_params=DisturbanceParams(p_dist=1 / 30),
        ),
    ),
    (
        "med_20",
        "Medium (avg 20 yrs)",
        ScenarioConfig(
            name="med_20",
            scenario_type="stochastic",
            noise_params=NOISE,
            disturbance_params=DisturbanceParams(p_dist=1 / 20),
        ),
    ),
    (
        "high_10",
        "High (avg 10 yrs)",
        ScenarioConfig(
            name="high_10",
            scenario_type="stochastic",
            noise_params=NOISE,
            disturbance_params=DisturbanceParams(p_dist=1 / 10),
        ),
    ),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    results: dict[str, BatchResult] = {}
    labels: dict[str, str] = {}

    for key, label, config in SCENARIOS:
        print(f"Running {label} ({N_RUNS} trajectories) ...")
        batch = run_batch(
            config,
            n_trajectories=N_RUNS,
            seed=SEED,
            store_trajectories=True,
            show_progress=True,
        )
        results[key] = batch
        labels[key] = label

    # Sanity check: print terminal HD for the no-catastrophe scenario
    noise_batch = results["noise_only"]
    assert noise_batch.trajectories is not None
    terminal_hds = [t.trajectory[-1].hd for t in noise_batch.trajectories]
    mean_hd = float(np.mean(terminal_hds))
    print(f"\nSanity check — mean terminal HD (noise only): {mean_hd:.1f} ft")
    print("  (expected ~108 ft for SI25=80 at age 40)")

    # Generate figure (QMD replaces HD since c_hd=0 means disturbances
    # don't reduce height — HD lines would overlap)
    fig = plot_disturbance_regime_comparison(
        results,
        labels=labels,
        show_sd=False,
        variables=["vol", "ba", "tpa"],
    )
    fig.suptitle(
        f"Mean growth trajectories under different catastrophe rates (n={N_RUNS})",
        y=0.99,
    )
    fig.tight_layout()

    # Save to plots/ and paper/figs/
    Path("plots").mkdir(exist_ok=True)
    Path("paper/figs").mkdir(parents=True, exist_ok=True)
    save_figure(fig, "plots/catastrophe_comparison.png", dpi=200)
    save_figure(fig, "paper/figs/catastrophe_comparison.png", dpi=200)
    plt.close(fig)
    print("\nSaved: plots/catastrophe_comparison.png")
    print("Saved: paper/figs/catastrophe_comparison.png")


if __name__ == "__main__":
    main()
