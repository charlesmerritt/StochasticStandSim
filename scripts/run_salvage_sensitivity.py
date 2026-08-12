"""Run salvage sensitivity analysis for disturbance-only scenarios.

Compares dist_30, dist_20, dist_10 with and without salvage enabled.
Salvage trigger: p75 of Beta(3.6, 8.4) severity distribution.
Salvage pricing: 50% of normal stumpage on remaining post-disturbance volume.

Outputs are written under data/salvage_sensitivity/ following the same
patterns as data/experiment_results/.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta as beta_dist

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.config import ScenarioConfig, ThinningParams
from core.disturbances import DisturbanceParams
from core.metrics import compare_scenarios, summarize_batch
from core.process_noise import NoiseParams
from core.simulate import BatchResult, ScenarioResult, run_batch, run_scenario

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_TRAJECTORIES = 1000
SEED = 42
OUTPUT_DIR = Path("data/salvage_sensitivity")

SEVERITY_MEAN = 0.30
SEVERITY_KAPPA = 12.0
ALPHA = SEVERITY_MEAN * SEVERITY_KAPPA   # 3.6
BETA = (1 - SEVERITY_MEAN) * SEVERITY_KAPPA  # 8.4
SALVAGE_THRESHOLD = float(beta_dist.ppf(0.75, ALPHA, BETA))

DEFAULT_THIN = ThinningParams(
    trigger_age=15.0,
    ba_threshold=150.0,
    residual_ba=100.0,
    thin_cost=87.34,
)

P_DIST_LEVELS = {
    "dist_30": 1 / 30,
    "dist_20": 1 / 20,
    "dist_10": 1 / 10,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _build_scenarios() -> list[ScenarioConfig]:
    """Build 6 scenarios: 3 disturbance-only × {no salvage, salvage at p75}."""
    scenarios: list[ScenarioConfig] = []

    for label, p_dist in P_DIST_LEVELS.items():
        base = ScenarioConfig(
            name=label,
            scenario_type="stochastic",
            noise_params=NoiseParams(lambda_proc=0.0),
            disturbance_params=DisturbanceParams(
                p_dist=p_dist,
                severity_mean=SEVERITY_MEAN,
                severity_kappa=SEVERITY_KAPPA,
            ),
            thin_params=DEFAULT_THIN,
            salvage_enabled=False,
        )
        salvage = ScenarioConfig(
            name=f"{label}_salvage",
            scenario_type="stochastic",
            noise_params=NoiseParams(lambda_proc=0.0),
            disturbance_params=DisturbanceParams(
                p_dist=p_dist,
                severity_mean=SEVERITY_MEAN,
                severity_kappa=SEVERITY_KAPPA,
            ),
            thin_params=DEFAULT_THIN,
            salvage_enabled=True,
            salvage_severity_threshold=SALVAGE_THRESHOLD,
            salvage_price_fraction=0.50,
        )
        scenarios.extend([base, salvage])

    return scenarios


def _run_all(
    scenarios: list[ScenarioConfig],
) -> dict[str, BatchResult]:
    """Run all stochastic batch scenarios."""
    results: dict[str, BatchResult] = {}

    for i, config in enumerate(scenarios):
        print(f"Running {i + 1}/{len(scenarios)}: {config.name}")
        # Each scenario pair (base + salvage) shares the same seed offset
        scenario_seed = SEED + (i // 2)
        batch = run_batch(
            config,
            n_trajectories=N_TRAJECTORIES,
            seed=scenario_seed,
            store_trajectories=False,
            show_progress=True,
        )
        results[config.name] = batch

    return results


# ---------------------------------------------------------------------------
# Deterministic baseline (for downside probability)
# ---------------------------------------------------------------------------
def _get_deterministic_baseline() -> ScenarioResult:
    config = ScenarioConfig(
        name="deterministic",
        scenario_type="deterministic",
        thin_params=DEFAULT_THIN,
    )
    return run_scenario(config)


# ---------------------------------------------------------------------------
# Output: comparison CSVs
# ---------------------------------------------------------------------------
def _comparison_rows(
    comparison: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    return [{"scenario_name": name, **stats} for name, stats in comparison.items()]


def _save_comparison_csvs(
    results: dict[str, BatchResult],
    deterministic: ScenarioResult,
    output_dir: Path,
) -> None:
    all_results: dict[str, BatchResult | ScenarioResult] = {
        "deterministic": deterministic,
        **results,
    }

    for metric in ("npv", "lev"):
        baseline = deterministic.npv if metric == "npv" else deterministic.lev
        comparison = compare_scenarios(
            all_results, metric=metric, deterministic_baseline=baseline
        )
        _write_csv(
            output_dir / f"{metric}_comparison.csv",
            _comparison_rows(comparison),
        )


# ---------------------------------------------------------------------------
# Output: scenario summaries
# ---------------------------------------------------------------------------
def _save_summaries(
    results: dict[str, BatchResult],
    deterministic: ScenarioResult,
    output_dir: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    for name, batch in results.items():
        config = batch.scenario_config
        summaries = summarize_batch(batch)
        row: dict[str, Any] = {
            "scenario_name": name,
            "salvage_enabled": config.salvage_enabled,
            "salvage_threshold": (
                config.salvage_severity_threshold if config.salvage_enabled else None
            ),
            "p_dist": config.disturbance_params.p_dist if config.disturbance_params else 0,
            "mean_salvage_count": float(np.mean(batch.salvage_counts)),
            "mean_salvage_revenue": float(np.mean(batch.salvage_revenues)),
        }
        for metric, summary in summaries.items():
            for key, value in asdict(summary).items():
                row[f"{metric}_{key}"] = value
        rows.append(row)

    _write_csv(output_dir / "scenario_summaries.csv", rows)


# ---------------------------------------------------------------------------
# Output: raw arrays
# ---------------------------------------------------------------------------
def _save_raw(results: dict[str, BatchResult], output_dir: Path) -> None:
    raw_dir = output_dir / "raw_arrays"
    _ensure_dir(raw_dir)
    for name, batch in results.items():
        np.savez_compressed(
            raw_dir / f"{name}.npz",
            terminal_values=batch.terminal_values,
            npvs=batch.npvs,
            levs=batch.levs,
            thin_revenues=batch.thin_revenues,
            salvage_counts=batch.salvage_counts,
            salvage_revenues=batch.salvage_revenues,
        )


# ---------------------------------------------------------------------------
# Validation figure: paired boxplots
# ---------------------------------------------------------------------------
def _plot_validation(
    results: dict[str, BatchResult],
    deterministic: ScenarioResult,
    output_dir: Path,
) -> None:
    fig_dir = output_dir / "figures"
    _ensure_dir(fig_dir)

    base_labels = list(P_DIST_LEVELS.keys())

    fig, ax = plt.subplots(figsize=(10, 6))

    positions = []
    data = []
    tick_positions = []
    tick_labels = []
    colors_list = []

    for i, label in enumerate(base_labels):
        base_name = label
        salvage_name = f"{label}_salvage"
        pos_base = i * 3
        pos_salv = i * 3 + 1

        data.append(results[base_name].npvs)
        data.append(results[salvage_name].npvs)
        positions.extend([pos_base, pos_salv])
        colors_list.extend(["#4C72B0", "#DD8452"])

        tick_positions.append(i * 3 + 0.5)
        ri = P_DIST_LEVELS[label]
        tick_labels.append(f"p_dist = 1/{int(1/ri)}")

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.7,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.5),
    )

    for patch, color in zip(bp["boxes"], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(
        deterministic.npv,
        color="gray",
        linestyle="--",
        linewidth=1,
        label=f"Deterministic NPV (${deterministic.npv:,.0f})",
    )

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("NPV ($/acre)")
    ax.set_title(
        f"NPV: No Salvage vs Salvage (threshold = {SALVAGE_THRESHOLD:.3f}, p75)"
    )

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#4C72B0", alpha=0.7, label="No salvage"),
        Patch(facecolor="#DD8452", alpha=0.7, label="Salvage at p75"),
        plt.Line2D([0], [0], color="gray", linestyle="--", label="Deterministic"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / "salvage_npv_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Saved validation figure: {fig_dir / 'salvage_npv_comparison.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    _ensure_dir(OUTPUT_DIR)

    print(f"Salvage severity threshold (p75): {SALVAGE_THRESHOLD:.4f}")
    print(f"Beta params: alpha={ALPHA}, beta={BETA}")
    print(f"Seed: {SEED}, n_trajectories: {N_TRAJECTORIES}\n")

    scenarios = _build_scenarios()
    deterministic = _get_deterministic_baseline()
    results = _run_all(scenarios)

    # Save manifest
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "seed": SEED,
        "n_trajectories": N_TRAJECTORIES,
        "salvage_severity_threshold": SALVAGE_THRESHOLD,
        "salvage_price_fraction": 0.50,
        "beta_alpha": ALPHA,
        "beta_beta": BETA,
        "deterministic_npv": deterministic.npv,
        "deterministic_lev": deterministic.lev,
        "scenarios": [s.name for s in scenarios],
    }
    _write_json(OUTPUT_DIR / "manifest.json", manifest)

    _save_comparison_csvs(results, deterministic, OUTPUT_DIR)
    _save_summaries(results, deterministic, OUTPUT_DIR)
    _save_raw(results, OUTPUT_DIR)
    _plot_validation(results, deterministic, OUTPUT_DIR)

    # Print quick summary
    print("\n" + "=" * 70)
    print("SALVAGE SENSITIVITY RESULTS")
    print("=" * 70)
    for label in P_DIST_LEVELS:
        base = results[label]
        salv = results[f"{label}_salvage"]
        print(f"\n{label}:")
        print(f"  No salvage — mean NPV: ${np.mean(base.npvs):,.0f}, "
              f"VaR5: ${np.percentile(base.npvs, 5):,.0f}, "
              f"CVaR5: ${np.mean(base.npvs[base.npvs <= np.percentile(base.npvs, 5)]):,.0f}")
        print(f"  Salvage    — mean NPV: ${np.mean(salv.npvs):,.0f}, "
              f"VaR5: ${np.percentile(salv.npvs, 5):,.0f}, "
              f"CVaR5: ${np.mean(salv.npvs[salv.npvs <= np.percentile(salv.npvs, 5)]):,.0f}")
        print(f"  Mean salvage events: {np.mean(salv.salvage_counts):.2f}")


if __name__ == "__main__":
    main()
