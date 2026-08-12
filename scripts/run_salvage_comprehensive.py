"""Comprehensive salvage sensitivity analysis.

Runs all new scenarios needed for two figures:
1. Threshold sensitivity: none / p25 / p50 / p75 at 50% price fraction
2. Price fraction sensitivity: none / 25% / 50% / 75% price at p75 threshold

Both figures are produced as side-by-side panels (moderate vs extreme severity).
Reuses existing raw arrays where available to avoid redundant computation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.stats import beta as beta_dist

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.config import ScenarioConfig, ThinningParams
from core.disturbances import DisturbanceParams
from core.process_noise import NoiseParams
from core.simulate import BatchResult, run_batch, run_scenario

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_TRAJECTORIES = 1000
SEED = 42
OUTPUT_DIR = Path("data/salvage_comprehensive")
FIG_DIR = REPO_ROOT / "paper" / "figs"

DEFAULT_THIN = ThinningParams(
    trigger_age=15.0, ba_threshold=150.0, residual_ba=100.0, thin_cost=87.34
)

P_DIST_LEVELS = {"dist_30": 1 / 30, "dist_20": 1 / 20, "dist_10": 1 / 10}
DIST_DISPLAY = {
    "dist_30": r"$p_{dist}$ = 1/30",
    "dist_20": r"$p_{dist}$ = 1/20",
    "dist_10": r"$p_{dist}$ = 1/10",
}

# Severity regimes
SEVERITY_REGIMES = {
    "moderate": {"mean": 0.30, "kappa": 12.0, "alpha": 3.6, "beta": 8.4,
                 "label": "Moderate — Beta(3.6, 8.4)"},
    "extreme":  {"mean": 0.50, "kappa": 6.0,  "alpha": 3.0, "beta": 3.0,
                 "label": "Extreme — Beta(3, 3)"},
}

# Compute thresholds for each regime
for regime in SEVERITY_REGIMES.values():
    a, b = regime["alpha"], regime["beta"]
    regime["p25"] = float(beta_dist.ppf(0.25, a, b))
    regime["p50"] = float(beta_dist.ppf(0.50, a, b))
    regime["p75"] = float(beta_dist.ppf(0.75, a, b))

# Existing data directories
EXISTING_DATA = {
    # (regime, dist, threshold_label, price_frac) -> directory, filename
    # Moderate severity existing data
    "moderate_p75": REPO_ROOT / "data" / "salvage_sensitivity",
    "moderate_p50": REPO_ROOT / "data" / "salvage_sensitivity_median",
    "extreme_all":  REPO_ROOT / "data" / "salvage_sensitivity_extreme",
}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _run_batch_scenario(
    name: str,
    p_dist: float,
    severity_mean: float,
    severity_kappa: float,
    salvage_enabled: bool,
    salvage_threshold: float,
    salvage_price_fraction: float,
    seed: int,
) -> BatchResult:
    config = ScenarioConfig(
        name=name,
        scenario_type="stochastic",
        noise_params=NoiseParams(lambda_proc=0.0),
        disturbance_params=DisturbanceParams(
            p_dist=p_dist,
            severity_mean=severity_mean,
            severity_kappa=severity_kappa,
        ),
        thin_params=DEFAULT_THIN,
        salvage_enabled=salvage_enabled,
        salvage_severity_threshold=salvage_threshold,
        salvage_price_fraction=salvage_price_fraction,
    )
    return run_batch(
        config, n_trajectories=N_TRAJECTORIES, seed=seed,
        store_trajectories=False, show_progress=True,
    )


def _load_existing_npvs(directory: Path, scenario: str) -> np.ndarray | None:
    path = directory / "raw_arrays" / f"{scenario}.npz"
    if path.exists():
        return np.load(path)["npvs"]
    return None


def _get_seed_for_dist(dist: str) -> int:
    """Consistent seed per disturbance regime."""
    idx = list(P_DIST_LEVELS.keys()).index(dist)
    return SEED + idx


# ---------------------------------------------------------------------------
# Run all needed scenarios, reusing existing where possible
# ---------------------------------------------------------------------------
def run_all() -> dict[str, np.ndarray]:
    """Returns dict mapping key -> npv array.

    Key format: "{regime}_{dist}_{condition}"
    condition: "none", "p25_50pct", "p50_50pct", "p75_50pct",
               "p75_25pct", "p75_75pct"
    """
    results: dict[str, np.ndarray] = {}
    raw_dir = REPO_ROOT / OUTPUT_DIR / "raw_arrays"
    _ensure_dir(raw_dir)

    for regime_name, regime in SEVERITY_REGIMES.items():
        sev_mean = regime["mean"]
        sev_kappa = regime["kappa"]

        for dist, p_dist in P_DIST_LEVELS.items():
            seed = _get_seed_for_dist(dist)

            # ---- No salvage ----
            key_none = f"{regime_name}_{dist}_none"
            npvs = _try_load_existing_no_salvage(regime_name, dist)
            if npvs is None:
                npvs = _try_load_cached(raw_dir, key_none)
            if npvs is None:
                print(f"Running: {key_none}")
                batch = _run_batch_scenario(
                    key_none, p_dist, sev_mean, sev_kappa,
                    False, 0.5, 0.50, seed,
                )
                npvs = batch.npvs
                np.savez_compressed(raw_dir / f"{key_none}.npz", npvs=npvs)
            else:
                print(f"Loaded existing: {key_none}")
            results[key_none] = npvs

            # ---- Threshold sweep (p25, p50, p75) at 50% price ----
            for pct_label in ("p25", "p50", "p75"):
                threshold = regime[pct_label]
                key = f"{regime_name}_{dist}_{pct_label}_50pct"
                npvs = _try_load_existing_salvage(regime_name, dist, pct_label, 0.50)
                if npvs is None:
                    npvs = _try_load_cached(raw_dir, key)
                if npvs is None:
                    print(f"Running: {key}")
                    batch = _run_batch_scenario(
                        key, p_dist, sev_mean, sev_kappa,
                        True, threshold, 0.50, seed,
                    )
                    npvs = batch.npvs
                    np.savez_compressed(raw_dir / f"{key}.npz", npvs=npvs)
                else:
                    print(f"Loaded existing: {key}")
                results[key] = npvs

            # ---- Price fraction sweep (25%, 75%) at p75 threshold ----
            for price_frac, price_label in [(0.25, "25pct"), (0.75, "75pct")]:
                key = f"{regime_name}_{dist}_p75_{price_label}"
                npvs = _try_load_cached(raw_dir, key)
                if npvs is None:
                    print(f"Running: {key}")
                    batch = _run_batch_scenario(
                        key, p_dist, sev_mean, sev_kappa,
                        True, regime["p75"], price_frac, seed,
                    )
                    npvs = batch.npvs
                    np.savez_compressed(raw_dir / f"{key}.npz", npvs=npvs)
                else:
                    print(f"Loaded existing: {key}")
                results[key] = npvs

    return results


def _try_load_cached(raw_dir: Path, key: str) -> np.ndarray | None:
    path = raw_dir / f"{key}.npz"
    if path.exists():
        return np.load(path)["npvs"]
    return None


def _try_load_existing_no_salvage(regime: str, dist: str) -> np.ndarray | None:
    """Try to load no-salvage from previous experiment directories."""
    if regime == "moderate":
        return _load_existing_npvs(EXISTING_DATA["moderate_p75"], dist)
    elif regime == "extreme":
        return _load_existing_npvs(EXISTING_DATA["extreme_all"], dist)
    return None


def _try_load_existing_salvage(
    regime: str, dist: str, pct_label: str, price_frac: float
) -> np.ndarray | None:
    """Try to load salvage results from previous experiment directories."""
    if price_frac != 0.50:
        return None

    if regime == "moderate":
        if pct_label == "p75":
            return _load_existing_npvs(
                EXISTING_DATA["moderate_p75"], f"{dist}_salvage"
            )
        elif pct_label == "p50":
            return _load_existing_npvs(
                EXISTING_DATA["moderate_p50"], f"{dist}_salvage"
            )
    elif regime == "extreme":
        if pct_label == "p50":
            return _load_existing_npvs(
                EXISTING_DATA["extreme_all"], f"{dist}_salvage_p50"
            )
        elif pct_label == "p75":
            return _load_existing_npvs(
                EXISTING_DATA["extreme_all"], f"{dist}_salvage_p75"
            )
    return None


# ---------------------------------------------------------------------------
# Figure 1: Threshold sensitivity (none / p25 / p50 / p75 at 50% price)
# ---------------------------------------------------------------------------
def plot_threshold_sensitivity(results: dict[str, np.ndarray], det_npv: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    COLOR_NONE = "#4C72B0"
    COLOR_P25 = "#C44E52"
    COLOR_P50 = "#55A868"
    COLOR_P75 = "#DD8452"
    box_colors = [COLOR_NONE, COLOR_P25, COLOR_P50, COLOR_P75]
    conditions = ["none", "p25_50pct", "p50_50pct", "p75_50pct"]

    for ax, (regime_name, regime) in zip(axes, SEVERITY_REGIMES.items()):
        all_data = []
        positions = []
        colors = []
        tick_positions = []
        tick_labels = []
        group_width = 5

        for gi, dist in enumerate(P_DIST_LEVELS):
            base_x = gi * group_width
            for j, (cond, color) in enumerate(zip(conditions, box_colors)):
                key = f"{regime_name}_{dist}_{cond}"
                all_data.append(results[key])
                positions.append(base_x + j)
                colors.append(color)

            tick_positions.append(base_x + 1.5)
            tick_labels.append(DIST_DISPLAY[dist])

        bp = ax.boxplot(
            all_data, positions=positions, widths=0.65, patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.8)

        ax.axhline(det_npv, color="gray", linestyle="--", linewidth=1.0, zorder=0)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=10)
        ax.set_title(regime["label"], fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.set_xlim(-0.8, (len(P_DIST_LEVELS) - 1) * group_width + 3.8)

    axes[0].set_ylabel("NPV ($/acre)", fontsize=12)

    legend_elements = [
        Patch(facecolor=COLOR_NONE, alpha=0.75, edgecolor="black", label="No salvage"),
        Patch(facecolor=COLOR_P25, alpha=0.75, edgecolor="black", label="Salvage at p25"),
        Patch(facecolor=COLOR_P50, alpha=0.75, edgecolor="black", label="Salvage at p50"),
        Patch(facecolor=COLOR_P75, alpha=0.75, edgecolor="black", label="Salvage at p75"),
        plt.Line2D([0], [0], color="gray", linestyle="--",
                   label=f"Deterministic NPV (${det_npv:,.0f})"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=5, fontsize=10,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Salvage Threshold Sensitivity (50% Salvage Price)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.14)

    out = FIG_DIR / "salvage_threshold_sensitivity.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 2: Price fraction sensitivity (none / 25% / 50% / 75% at p75 threshold)
# ---------------------------------------------------------------------------
def plot_price_sensitivity(results: dict[str, np.ndarray], det_npv: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    COLOR_NONE = "#4C72B0"
    COLOR_25 = "#C44E52"
    COLOR_50 = "#55A868"
    COLOR_75 = "#DD8452"
    box_colors = [COLOR_NONE, COLOR_25, COLOR_50, COLOR_75]
    conditions = ["none", "p75_25pct", "p75_50pct", "p75_75pct"]

    for ax, (regime_name, regime) in zip(axes, SEVERITY_REGIMES.items()):
        all_data = []
        positions = []
        colors = []
        tick_positions = []
        tick_labels = []
        group_width = 5

        for gi, dist in enumerate(P_DIST_LEVELS):
            base_x = gi * group_width
            for j, (cond, color) in enumerate(zip(conditions, box_colors)):
                key = f"{regime_name}_{dist}_{cond}"
                all_data.append(results[key])
                positions.append(base_x + j)
                colors.append(color)

            tick_positions.append(base_x + 1.5)
            tick_labels.append(DIST_DISPLAY[dist])

        bp = ax.boxplot(
            all_data, positions=positions, widths=0.65, patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.8)

        ax.axhline(det_npv, color="gray", linestyle="--", linewidth=1.0, zorder=0)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=10)
        ax.set_title(regime["label"], fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.set_xlim(-0.8, (len(P_DIST_LEVELS) - 1) * group_width + 3.8)

    axes[0].set_ylabel("NPV ($/acre)", fontsize=12)

    legend_elements = [
        Patch(facecolor=COLOR_NONE, alpha=0.75, edgecolor="black", label="No salvage"),
        Patch(facecolor=COLOR_25, alpha=0.75, edgecolor="black", label="Salvage at 25% price"),
        Patch(facecolor=COLOR_50, alpha=0.75, edgecolor="black", label="Salvage at 50% price"),
        Patch(facecolor=COLOR_75, alpha=0.75, edgecolor="black", label="Salvage at 75% price"),
        plt.Line2D([0], [0], color="gray", linestyle="--",
                   label=f"Deterministic NPV (${det_npv:,.0f})"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=5, fontsize=10,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Salvage Price Fraction Sensitivity (p75 Threshold)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.14)

    out = FIG_DIR / "salvage_price_sensitivity.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def print_summary(results: dict[str, np.ndarray]) -> None:
    print("\n" + "=" * 90)
    print("THRESHOLD SENSITIVITY (50% salvage price)")
    print("=" * 90)
    header = f"{'Regime':<10} {'Dist':<10} {'None':>10} {'p25':>10} {'p50':>10} {'p75':>10}"
    print(header)
    print("-" * 60)
    for regime_name in SEVERITY_REGIMES:
        for dist in P_DIST_LEVELS:
            vals = []
            for cond in ["none", "p25_50pct", "p50_50pct", "p75_50pct"]:
                key = f"{regime_name}_{dist}_{cond}"
                vals.append(float(np.mean(results[key])))
            print(f"{regime_name:<10} {dist:<10} " +
                  " ".join(f"${v:>8,.0f}" for v in vals))

    print("\n" + "=" * 90)
    print("PRICE FRACTION SENSITIVITY (p75 threshold)")
    print("=" * 90)
    header = f"{'Regime':<10} {'Dist':<10} {'None':>10} {'25%':>10} {'50%':>10} {'75%':>10}"
    print(header)
    print("-" * 60)
    for regime_name in SEVERITY_REGIMES:
        for dist in P_DIST_LEVELS:
            vals = []
            for cond in ["none", "p75_25pct", "p75_50pct", "p75_75pct"]:
                key = f"{regime_name}_{dist}_{cond}"
                vals.append(float(np.mean(results[key])))
            print(f"{regime_name:<10} {dist:<10} " +
                  " ".join(f"${v:>8,.0f}" for v in vals))


def main() -> None:
    _ensure_dir(REPO_ROOT / OUTPUT_DIR)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Get deterministic baseline
    det_config = ScenarioConfig(
        name="deterministic", scenario_type="deterministic", thin_params=DEFAULT_THIN,
    )
    det = run_scenario(det_config)
    det_npv = det.npv

    results = run_all()
    plot_threshold_sensitivity(results, det_npv)
    plot_price_sensitivity(results, det_npv)
    print_summary(results)

    # Save manifest
    manifest = {
        "seed": SEED,
        "n_trajectories": N_TRAJECTORIES,
        "deterministic_npv": det_npv,
        "severity_regimes": {
            name: {k: v for k, v in r.items() if k != "label"}
            for name, r in SEVERITY_REGIMES.items()
        },
        "scenarios_run": sorted(results.keys()),
    }
    (REPO_ROOT / OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
