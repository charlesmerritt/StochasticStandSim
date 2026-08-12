"""Run salvage sensitivity under extreme severity regime.

Severity: Beta(3, 3) — m_q=0.50, κ=6.
  Mean/median/mode = 0.50, p75 = 0.641, p95 = 0.811, σ = 0.189.

Runs dist_30, dist_20, dist_10 × {no salvage, salvage at p50, salvage at p75}.
Outputs under data/salvage_sensitivity_extreme/.
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
from matplotlib.patches import Patch
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
OUTPUT_DIR = Path("data/salvage_sensitivity_extreme")

SEVERITY_MEAN = 0.50
SEVERITY_KAPPA = 6.0
ALPHA = SEVERITY_MEAN * SEVERITY_KAPPA   # 3.0
BETA = (1 - SEVERITY_MEAN) * SEVERITY_KAPPA  # 3.0

THRESH_P50 = float(beta_dist.ppf(0.50, ALPHA, BETA))
THRESH_P75 = float(beta_dist.ppf(0.75, ALPHA, BETA))

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

SALVAGE_CONDITIONS = {
    "": None,                  # no salvage
    "_salvage_p50": THRESH_P50,
    "_salvage_p75": THRESH_P75,
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
    """Build 9 scenarios: 3 disturbance × 3 salvage conditions."""
    scenarios: list[ScenarioConfig] = []

    for label, p_dist in P_DIST_LEVELS.items():
        for suffix, threshold in SALVAGE_CONDITIONS.items():
            enabled = threshold is not None
            cfg = ScenarioConfig(
                name=f"{label}{suffix}",
                scenario_type="stochastic",
                noise_params=NoiseParams(lambda_proc=0.0),
                disturbance_params=DisturbanceParams(
                    p_dist=p_dist,
                    severity_mean=SEVERITY_MEAN,
                    severity_kappa=SEVERITY_KAPPA,
                ),
                thin_params=DEFAULT_THIN,
                salvage_enabled=enabled,
                salvage_severity_threshold=threshold if enabled else 0.5,
                salvage_price_fraction=0.50,
            )
            scenarios.append(cfg)

    return scenarios


def _run_all(
    scenarios: list[ScenarioConfig],
) -> dict[str, BatchResult]:
    results: dict[str, BatchResult] = {}

    for i, config in enumerate(scenarios):
        print(f"Running {i + 1}/{len(scenarios)}: {config.name}")
        # Same seed for each disturbance group (3 conditions share a seed)
        scenario_seed = SEED + (i // 3)
        batch = run_batch(
            config,
            n_trajectories=N_TRAJECTORIES,
            seed=scenario_seed,
            store_trajectories=False,
            show_progress=True,
        )
        results[config.name] = batch

    return results


def _get_deterministic_baseline() -> ScenarioResult:
    config = ScenarioConfig(
        name="deterministic",
        scenario_type="deterministic",
        thin_params=DEFAULT_THIN,
    )
    return run_scenario(config)


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


def _save_summaries(
    results: dict[str, BatchResult],
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


def _plot_unified(
    results: dict[str, BatchResult],
    deterministic: ScenarioResult,
    output_dir: Path,
) -> None:
    fig_dir = output_dir / "figures"
    _ensure_dir(fig_dir)

    base_labels = list(P_DIST_LEVELS.keys())

    COLOR_NONE = "#4C72B0"
    COLOR_P50 = "#55A868"
    COLOR_P75 = "#DD8452"

    fig, ax = plt.subplots(figsize=(10, 6))

    all_data = []
    positions = []
    colors = []
    tick_positions = []
    tick_labels = []
    group_width = 4

    for gi, dist in enumerate(base_labels):
        base_x = gi * group_width
        for j, (suffix, color) in enumerate([
            ("", COLOR_NONE),
            ("_salvage_p50", COLOR_P50),
            ("_salvage_p75", COLOR_P75),
        ]):
            all_data.append(results[f"{dist}{suffix}"].npvs)
            positions.append(base_x + j)
            colors.append(color)

        tick_positions.append(base_x + 1)
        ri = P_DIST_LEVELS[dist]
        tick_labels.append(f"$p_{{dist}}$ = 1/{int(1/ri)}")

    bp = ax.boxplot(
        all_data,
        positions=positions,
        widths=0.7,
        patch_artist=True,
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

    ax.axhline(deterministic.npv, color="gray", linestyle="--", linewidth=1.0, zorder=0)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=11)
    ax.set_ylabel("NPV ($/acre)", fontsize=12)
    ax.set_title(
        f"Extreme Severity — Beta(3, 3): NPV by Disturbance Regime and Salvage Threshold",
        fontsize=13,
        fontweight="bold",
    )

    legend_elements = [
        Patch(facecolor=COLOR_NONE, alpha=0.75, edgecolor="black", label="No salvage"),
        Patch(facecolor=COLOR_P50, alpha=0.75, edgecolor="black",
              label=f"Salvage at p50 (q \u2265 {THRESH_P50:.3f})"),
        Patch(facecolor=COLOR_P75, alpha=0.75, edgecolor="black",
              label=f"Salvage at p75 (q \u2265 {THRESH_P75:.3f})"),
        plt.Line2D([0], [0], color="gray", linestyle="--",
                   label=f"Deterministic NPV (${deterministic.npv:,.0f})"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10, framealpha=0.9)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_xlim(-0.8, (len(base_labels) - 1) * group_width + 2.8)

    fig.tight_layout()
    out_path = fig_dir / "salvage_npv_extreme.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    _ensure_dir(OUTPUT_DIR)

    print(f"Extreme severity distribution: Beta({ALPHA}, {BETA})")
    print(f"  Mean = {SEVERITY_MEAN}, StdDev = {(ALPHA*BETA/((ALPHA+BETA)**2*(ALPHA+BETA+1)))**0.5:.4f}")
    print(f"  p50 threshold: {THRESH_P50:.4f}")
    print(f"  p75 threshold: {THRESH_P75:.4f}")
    print(f"Seed: {SEED}, n_trajectories: {N_TRAJECTORIES}\n")

    scenarios = _build_scenarios()
    deterministic = _get_deterministic_baseline()
    results = _run_all(scenarios)

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "seed": SEED,
        "n_trajectories": N_TRAJECTORIES,
        "severity_mean": SEVERITY_MEAN,
        "severity_kappa": SEVERITY_KAPPA,
        "beta_alpha": ALPHA,
        "beta_beta": BETA,
        "salvage_threshold_p50": THRESH_P50,
        "salvage_threshold_p75": THRESH_P75,
        "salvage_price_fraction": 0.50,
        "deterministic_npv": deterministic.npv,
        "deterministic_lev": deterministic.lev,
        "scenarios": [s.name for s in scenarios],
    }
    _write_json(OUTPUT_DIR / "manifest.json", manifest)

    _save_comparison_csvs(results, deterministic, OUTPUT_DIR)
    _save_summaries(results, OUTPUT_DIR)
    _save_raw(results, OUTPUT_DIR)
    _plot_unified(results, deterministic, OUTPUT_DIR)

    # Summary table
    print("\n" + "=" * 72)
    print("EXTREME SEVERITY RESULTS — Beta(3, 3)")
    print("=" * 72)
    print(f"\n{'Scenario':<12} {'No Salvage':>12} {'p50 Salvage':>12} {'p75 Salvage':>12}  {'Salv/rot(p50)':>13} {'Salv/rot(p75)':>13}")
    print("-" * 80)
    for label in P_DIST_LEVELS:
        npv_none = np.mean(results[label].npvs)
        npv_p50 = np.mean(results[f"{label}_salvage_p50"].npvs)
        npv_p75 = np.mean(results[f"{label}_salvage_p75"].npvs)
        sc_p50 = np.mean(results[f"{label}_salvage_p50"].salvage_counts)
        sc_p75 = np.mean(results[f"{label}_salvage_p75"].salvage_counts)
        print(f"{label:<12} ${npv_none:>10,.0f} ${npv_p50:>10,.0f} ${npv_p75:>10,.0f}  {sc_p50:>13.2f} {sc_p75:>13.2f}")


if __name__ == "__main__":
    main()
