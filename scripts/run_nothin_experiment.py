"""Run the full 16-scenario experiment matrix WITHOUT thinning.

This script mirrors run_full_matrix_experiment.py but sets thin_params=None
for all scenarios, allowing comparison of thinned vs unthinned management.

Outputs are written under data/experiment_results_nothin/:
- scenario summary tables
- deterministic baseline comparisons
- raw per-scenario arrays
- disturbance path records
- summary figures
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
import numpy as np

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.config import ScenarioConfig
from core.disturbances import DisturbanceParams
from core.metrics import compare_scenarios, summarize_batch
from core.process_noise import NoiseParams
from core.scenarios import (
    LAMBDA_LEVELS,
    P_DIST_LEVELS,
    SEVERITY_KAPPA,
    SEVERITY_MEAN,
)
from core.simulate import BatchResult, ScenarioResult, run_batch_scenarios
from core.viz import (
    plot_deterministic_product_distribution,
    plot_deterministic_vs_stochastic_comparison,
    plot_disturbance_frequency_histograms,
    plot_disturbance_regime_comparison,
    plot_downside_risk,
    plot_growth_validation,
    plot_hd_debug,
    plot_product_breakdown,
    plot_scenario_boxplots,
    plot_sensitivity_heatmap,
    plot_stochastic_growth_demo,
    plot_stochastic_product_distribution,
    plot_terminal_histogram,
    save_figure,
)

N_TRAJECTORIES = 1000
SEED = 42
OUTPUT_DIR = Path("data/experiment_results_nothin")


# =============================================================================
# NO-THINNING SCENARIO DEFINITIONS
# =============================================================================

DETERMINISTIC_NOTHIN = ScenarioConfig(
    name="deterministic",
    scenario_type="deterministic",
    thin_params=None,
)

DIST_30_NOTHIN = ScenarioConfig(
    name="dist_30",
    scenario_type="stochastic",
    noise_params=NoiseParams(lambda_proc=0.0),
    disturbance_params=DisturbanceParams(
        p_dist=1 / 30, severity_mean=SEVERITY_MEAN, severity_kappa=SEVERITY_KAPPA
    ),
    thin_params=None,
)

DIST_20_NOTHIN = ScenarioConfig(
    name="dist_20",
    scenario_type="stochastic",
    noise_params=NoiseParams(lambda_proc=0.0),
    disturbance_params=DisturbanceParams(
        p_dist=1 / 20, severity_mean=SEVERITY_MEAN, severity_kappa=SEVERITY_KAPPA
    ),
    thin_params=None,
)

DIST_10_NOTHIN = ScenarioConfig(
    name="dist_10",
    scenario_type="stochastic",
    noise_params=NoiseParams(lambda_proc=0.0),
    disturbance_params=DisturbanceParams(
        p_dist=1 / 10, severity_mean=SEVERITY_MEAN, severity_kappa=SEVERITY_KAPPA
    ),
    thin_params=None,
)

NOISE_025_NOTHIN = ScenarioConfig(
    name="noise_025",
    scenario_type="stochastic",
    noise_params=NoiseParams(lambda_proc=0.25),
    disturbance_params=DisturbanceParams(p_dist=0.0),
    thin_params=None,
)

N025_D30_NOTHIN = ScenarioConfig(
    name="n025_d30",
    scenario_type="stochastic",
    noise_params=NoiseParams(lambda_proc=0.25),
    disturbance_params=DisturbanceParams(
        p_dist=1 / 30, severity_mean=SEVERITY_MEAN, severity_kappa=SEVERITY_KAPPA
    ),
    thin_params=None,
)

N025_D20_NOTHIN = ScenarioConfig(
    name="n025_d20",
    scenario_type="stochastic",
    noise_params=NoiseParams(lambda_proc=0.25),
    disturbance_params=DisturbanceParams(
        p_dist=1 / 20, severity_mean=SEVERITY_MEAN, severity_kappa=SEVERITY_KAPPA
    ),
    thin_params=None,
)

N025_D10_NOTHIN = ScenarioConfig(
    name="n025_d10",
    scenario_type="stochastic",
    noise_params=NoiseParams(lambda_proc=0.25),
    disturbance_params=DisturbanceParams(
        p_dist=1 / 10, severity_mean=SEVERITY_MEAN, severity_kappa=SEVERITY_KAPPA
    ),
    thin_params=None,
)

NOISE_050_NOTHIN = ScenarioConfig(
    name="noise_050",
    scenario_type="stochastic",
    noise_params=NoiseParams(lambda_proc=0.5),
    disturbance_params=DisturbanceParams(p_dist=0.0),
    thin_params=None,
)

N050_D30_NOTHIN = ScenarioConfig(
    name="n050_d30",
    scenario_type="stochastic",
    noise_params=NoiseParams(lambda_proc=0.5),
    disturbance_params=DisturbanceParams(
        p_dist=1 / 30, severity_mean=SEVERITY_MEAN, severity_kappa=SEVERITY_KAPPA
    ),
    thin_params=None,
)

N050_D20_NOTHIN = ScenarioConfig(
    name="n050_d20",
    scenario_type="stochastic",
    noise_params=NoiseParams(lambda_proc=0.5),
    disturbance_params=DisturbanceParams(
        p_dist=1 / 20, severity_mean=SEVERITY_MEAN, severity_kappa=SEVERITY_KAPPA
    ),
    thin_params=None,
)

N050_D10_NOTHIN = ScenarioConfig(
    name="n050_d10",
    scenario_type="stochastic",
    noise_params=NoiseParams(lambda_proc=0.5),
    disturbance_params=DisturbanceParams(
        p_dist=1 / 10, severity_mean=SEVERITY_MEAN, severity_kappa=SEVERITY_KAPPA
    ),
    thin_params=None,
)

NOISE_100_NOTHIN = ScenarioConfig(
    name="noise_100",
    scenario_type="stochastic",
    noise_params=NoiseParams(lambda_proc=1.0),
    disturbance_params=DisturbanceParams(p_dist=0.0),
    thin_params=None,
)

N100_D30_NOTHIN = ScenarioConfig(
    name="n100_d30",
    scenario_type="stochastic",
    noise_params=NoiseParams(lambda_proc=1.0),
    disturbance_params=DisturbanceParams(
        p_dist=1 / 30, severity_mean=SEVERITY_MEAN, severity_kappa=SEVERITY_KAPPA
    ),
    thin_params=None,
)

N100_D20_NOTHIN = ScenarioConfig(
    name="n100_d20",
    scenario_type="stochastic",
    noise_params=NoiseParams(lambda_proc=1.0),
    disturbance_params=DisturbanceParams(
        p_dist=1 / 20, severity_mean=SEVERITY_MEAN, severity_kappa=SEVERITY_KAPPA
    ),
    thin_params=None,
)

N100_D10_NOTHIN = ScenarioConfig(
    name="n100_d10",
    scenario_type="stochastic",
    noise_params=NoiseParams(lambda_proc=1.0),
    disturbance_params=DisturbanceParams(
        p_dist=1 / 10, severity_mean=SEVERITY_MEAN, severity_kappa=SEVERITY_KAPPA
    ),
    thin_params=None,
)

ALL_SCENARIOS_NOTHIN = [
    DETERMINISTIC_NOTHIN,
    DIST_30_NOTHIN,
    DIST_20_NOTHIN,
    DIST_10_NOTHIN,
    NOISE_025_NOTHIN,
    N025_D30_NOTHIN,
    N025_D20_NOTHIN,
    N025_D10_NOTHIN,
    NOISE_050_NOTHIN,
    N050_D30_NOTHIN,
    N050_D20_NOTHIN,
    N050_D10_NOTHIN,
    NOISE_100_NOTHIN,
    N100_D30_NOTHIN,
    N100_D20_NOTHIN,
    N100_D10_NOTHIN,
]

NOISE_ONLY_SCENARIOS_NOTHIN = [
    DETERMINISTIC_NOTHIN,
    NOISE_025_NOTHIN,
    NOISE_050_NOTHIN,
    NOISE_100_NOTHIN,
]

DIST_ONLY_SCENARIOS_NOTHIN = [
    DETERMINISTIC_NOTHIN,
    DIST_30_NOTHIN,
    DIST_20_NOTHIN,
    DIST_10_NOTHIN,
]


# =============================================================================
# HELPER FUNCTIONS (same as run_full_matrix_experiment.py)
# =============================================================================


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_and_close(fig, path: Path) -> None:
    import matplotlib.pyplot as plt

    save_figure(fig, str(path))
    plt.close(fig)


def _metric_baseline(deterministic: ScenarioResult, metric: str) -> float:
    if metric == "terminal_value":
        return (
            deterministic.terminal_yield.net_revenue
            if deterministic.terminal_yield is not None
            else 0.0
        )
    if metric == "npv":
        return deterministic.npv
    if metric == "lev":
        return deterministic.lev
    raise ValueError(f"Unknown metric: {metric}")


def _scenario_params(result: BatchResult | ScenarioResult) -> dict[str, float]:
    config = result.scenario_config
    lambda_proc = config.noise_params.lambda_proc if config.noise_params else 0.0
    p_dist = config.disturbance_params.p_dist if config.disturbance_params else 0.0
    severity_mean = (
        config.disturbance_params.severity_mean if config.disturbance_params else 0.0
    )
    severity_kappa = (
        config.disturbance_params.severity_kappa if config.disturbance_params else 0.0
    )
    return {
        "lambda_proc": lambda_proc,
        "p_dist": p_dist,
        "severity_mean": severity_mean,
        "severity_kappa": severity_kappa,
    }


def _first_disturbance_years(batch: BatchResult) -> np.ndarray:
    first_years = []
    for years in batch.disturbance_years:
        first_years.append(years[0] if years else np.nan)
    return np.asarray(first_years, dtype=float)


def _build_summary_rows(
    ordered_results: list[tuple[str, BatchResult | ScenarioResult]],
    comparisons: dict[str, dict[str, dict[str, float]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for name, result in ordered_results:
        config = (
            result.scenario_config
            if isinstance(result, BatchResult)
            else result.scenario_config
        )
        if config is None:
            raise ValueError(f"Missing scenario config for result {name}")

        row: dict[str, Any] = {
            "scenario_name": name,
            "scenario_type": config.scenario_type,
            **_scenario_params(result),
            "rotation_length": config.rotation_length,
            "discount_rate": config.discount_rate,
        }

        if isinstance(result, BatchResult):
            summaries = summarize_batch(result)
            for metric, summary in summaries.items():
                stats = asdict(summary)
                for key, value in stats.items():
                    row[f"{metric}_{key}"] = value

            disturbed_mask = result.disturbance_occurred
            disturbed_mean_severity = (
                float(np.mean(result.mean_disturbance_severities[disturbed_mask]))
                if np.any(disturbed_mask)
                else 0.0
            )
            row.update(
                {
                    "disturbance_any_probability": float(np.mean(disturbed_mask)),
                    "disturbance_count_mean": float(np.mean(result.disturbance_counts)),
                    "disturbance_count_max": int(np.max(result.disturbance_counts)),
                    "mean_disturbance_severity_conditional": disturbed_mean_severity,
                    "max_disturbance_severity_mean": float(
                        np.mean(result.max_disturbance_severities)
                    ),
                    "first_disturbance_year_mean": float(
                        np.nanmean(_first_disturbance_years(result))
                    )
                    if np.any(disturbed_mask)
                    else float("nan"),
                    "vol_pulp_mean": float(np.mean(result.vol_pulp)),
                    "vol_cns_mean": float(np.mean(result.vol_cns)),
                    "vol_saw_mean": float(np.mean(result.vol_saw)),
                    "vol_total_mean": float(np.mean(result.vol_total)),
                }
            )
        else:
            terminal_value = (
                result.terminal_yield.net_revenue if result.terminal_yield else 0.0
            )
            row.update(
                {
                    "terminal_value_mean": terminal_value,
                    "terminal_value_std": 0.0,
                    "terminal_value_median": terminal_value,
                    "terminal_value_p5": terminal_value,
                    "terminal_value_p25": terminal_value,
                    "terminal_value_p75": terminal_value,
                    "terminal_value_p95": terminal_value,
                    "terminal_value_var_5": terminal_value,
                    "terminal_value_cvar_5": terminal_value,
                    "terminal_value_min": terminal_value,
                    "terminal_value_max": terminal_value,
                    "terminal_value_n": 1,
                    "npv_mean": result.npv,
                    "npv_std": 0.0,
                    "npv_median": result.npv,
                    "npv_p5": result.npv,
                    "npv_p25": result.npv,
                    "npv_p75": result.npv,
                    "npv_p95": result.npv,
                    "npv_var_5": result.npv,
                    "npv_cvar_5": result.npv,
                    "npv_min": result.npv,
                    "npv_max": result.npv,
                    "npv_n": 1,
                    "lev_mean": result.lev,
                    "lev_std": 0.0,
                    "lev_median": result.lev,
                    "lev_p5": result.lev,
                    "lev_p25": result.lev,
                    "lev_p75": result.lev,
                    "lev_p95": result.lev,
                    "lev_var_5": result.lev,
                    "lev_cvar_5": result.lev,
                    "lev_min": result.lev,
                    "lev_max": result.lev,
                    "lev_n": 1,
                    "thin_revenue_mean": result.thin_revenue,
                    "thin_revenue_std": 0.0,
                    "thin_revenue_median": result.thin_revenue,
                    "thin_revenue_p5": result.thin_revenue,
                    "thin_revenue_p25": result.thin_revenue,
                    "thin_revenue_p75": result.thin_revenue,
                    "thin_revenue_p95": result.thin_revenue,
                    "thin_revenue_var_5": result.thin_revenue,
                    "thin_revenue_cvar_5": result.thin_revenue,
                    "thin_revenue_min": result.thin_revenue,
                    "thin_revenue_max": result.thin_revenue,
                    "thin_revenue_n": 1,
                    "disturbance_any_probability": 0.0,
                    "disturbance_count_mean": 0.0,
                    "disturbance_count_max": 0,
                    "mean_disturbance_severity_conditional": 0.0,
                    "max_disturbance_severity_mean": 0.0,
                    "first_disturbance_year_mean": float("nan"),
                    "vol_pulp_mean": (
                        result.terminal_yield.vol_pulp if result.terminal_yield else 0.0
                    ),
                    "vol_cns_mean": (
                        result.terminal_yield.vol_cns if result.terminal_yield else 0.0
                    ),
                    "vol_saw_mean": (
                        result.terminal_yield.vol_saw if result.terminal_yield else 0.0
                    ),
                    "vol_total_mean": (
                        result.terminal_yield.vol_total
                        if result.terminal_yield
                        else 0.0
                    ),
                }
            )

        for metric, metric_comparison in comparisons.items():
            stats = metric_comparison[name]
            row[f"{metric}_downside_prob_vs_deterministic"] = stats[
                "downside_prob_vs_deterministic"
            ]

        rows.append(row)

    return rows


def _comparison_rows(comparison: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    return [{"scenario_name": name, **stats} for name, stats in comparison.items()]


def _save_raw_outputs(
    output_dir: Path,
    ordered_results: list[tuple[str, BatchResult | ScenarioResult]],
) -> None:
    raw_dir = output_dir / "raw_arrays"
    path_dir = output_dir / "disturbance_paths"
    _ensure_dir(raw_dir)
    _ensure_dir(path_dir)

    for name, result in ordered_results:
        if isinstance(result, BatchResult):
            np.savez_compressed(
                raw_dir / f"{name}.npz",
                terminal_values=result.terminal_values,
                npvs=result.npvs,
                levs=result.levs,
                thin_revenues=result.thin_revenues,
                vol_pulp=result.vol_pulp,
                vol_cns=result.vol_cns,
                vol_saw=result.vol_saw,
                vol_total=result.vol_total,
                disturbance_occurred=result.disturbance_occurred.astype(int),
                disturbance_counts=result.disturbance_counts,
                mean_disturbance_severities=result.mean_disturbance_severities,
                max_disturbance_severities=result.max_disturbance_severities,
            )
            _write_json(
                path_dir / f"{name}.json",
                {
                    "disturbance_years": result.disturbance_years,
                    "disturbance_severity_paths": result.disturbance_severity_paths,
                },
            )
        else:
            _write_json(
                raw_dir / f"{name}.json",
                {
                    "npv": result.npv,
                    "lev": result.lev,
                    "thin_revenue": result.thin_revenue,
                    "terminal_value": (
                        result.terminal_yield.net_revenue
                        if result.terminal_yield is not None
                        else 0.0
                    ),
                    "disturbance_years": result.disturbance_years,
                    "disturbance_severities": result.disturbance_severities,
                },
            )


def _generate_figures(
    output_dir: Path,
    ordered_results: list[tuple[str, BatchResult | ScenarioResult]],
    deterministic: ScenarioResult,
) -> None:
    figures_dir = output_dir / "figures"
    histogram_dir = figures_dir / "histograms"
    trajectories_dir = figures_dir / "trajectories"
    diagnostics_dir = figures_dir / "diagnostics"
    products_dir = figures_dir / "products"
    _ensure_dir(figures_dir)
    _ensure_dir(histogram_dir)
    _ensure_dir(trajectories_dir)
    _ensure_dir(diagnostics_dir)
    _ensure_dir(products_dir)

    results = {name: result for name, result in ordered_results}
    stochastic_results = {
        name: result
        for name, result in ordered_results
        if isinstance(result, BatchResult)
    }
    disturbance_only_results = {
        config.name: stochastic_results[config.name]
        for config in DIST_ONLY_SCENARIOS_NOTHIN
        if config.name in stochastic_results
    }
    noise_only_results = {
        config.name: stochastic_results[config.name]
        for config in NOISE_ONLY_SCENARIOS_NOTHIN
        if config.name in stochastic_results and config.name != "deterministic"
    }
    representative_name = "n050_d20"
    representative_batch = stochastic_results[representative_name]

    _save_and_close(
        plot_scenario_boxplots(results, metric="terminal_value"),
        figures_dir / "terminal_value_boxplots.png",
    )
    _save_and_close(
        plot_scenario_boxplots(results, metric="npv"),
        figures_dir / "npv_boxplots.png",
    )
    _save_and_close(
        plot_downside_risk(stochastic_results, metric="npv"),
        figures_dir / "npv_downside_risk.png",
    )
    _save_and_close(
        plot_product_breakdown(results),
        figures_dir / "product_breakdown.png",
    )
    _save_and_close(
        plot_sensitivity_heatmap(
            results,
            lambda_levels=LAMBDA_LEVELS,
            p_dist_levels=P_DIST_LEVELS,
            metric="npv",
            stat="mean",
        ),
        figures_dir / "npv_heatmap_mean.png",
    )
    _save_and_close(
        plot_sensitivity_heatmap(
            results,
            lambda_levels=LAMBDA_LEVELS,
            p_dist_levels=P_DIST_LEVELS,
            metric="npv",
            stat="p5",
        ),
        figures_dir / "npv_heatmap_p5.png",
    )
    _save_and_close(
        plot_deterministic_vs_stochastic_comparison(deterministic, representative_batch),
        trajectories_dir / "deterministic_vs_stochastic_n050_d20.png",
    )
    _save_and_close(
        plot_disturbance_regime_comparison(
            {
                "deterministic": deterministic,
                **disturbance_only_results,
            },
            labels={
                "deterministic": "Deterministic",
                "dist_30": "1/30 disturbance",
                "dist_20": "1/20 disturbance",
                "dist_10": "1/10 disturbance",
            },
            show_sd=True,
        ),
        trajectories_dir / "disturbance_regime_comparison.png",
    )
    _save_and_close(
        plot_stochastic_growth_demo(representative_batch),
        trajectories_dir / "stochastic_growth_demo_n050_d20.png",
    )
    _save_and_close(
        plot_disturbance_frequency_histograms(disturbance_only_results),
        diagnostics_dir / "disturbance_frequency_histograms.png",
    )
    _save_and_close(
        plot_growth_validation(representative_batch, deterministic=deterministic),
        diagnostics_dir / "growth_validation_n050_d20.png",
    )
    _save_and_close(
        plot_hd_debug(deterministic, representative_batch),
        diagnostics_dir / "hd_debug_n050_d20.png",
    )
    _save_and_close(
        plot_deterministic_product_distribution(deterministic),
        products_dir / "deterministic_product_distribution.png",
    )
    _save_and_close(
        plot_stochastic_product_distribution(
            {
                "noise_025": noise_only_results["noise_025"],
                "noise_050": noise_only_results["noise_050"],
                "noise_100": noise_only_results["noise_100"],
            },
            stackplot_label="noise_050",
            revenue_breakdown_label="noise_100",
        ),
        products_dir / "stochastic_product_distribution_noise_only.png",
    )

    deterministic_terminal_value = _metric_baseline(deterministic, "terminal_value")
    for name, result in stochastic_results.items():
        _save_and_close(
            plot_terminal_histogram(
                result,
                deterministic_value=deterministic_terminal_value,
                metric="terminal_value",
            ),
            histogram_dir / f"{name}_terminal_value_histogram.png",
        )


def main() -> None:
    print("=" * 60)
    print("No-Thinning Experiment (16 scenarios)")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"N trajectories: {N_TRAJECTORIES}")
    print(f"Seed: {SEED}")
    print("=" * 60)

    _ensure_dir(OUTPUT_DIR)

    ordered_results = list(
        run_batch_scenarios(
            ALL_SCENARIOS_NOTHIN,
            n_trajectories=N_TRAJECTORIES,
            seed=SEED,
            store_trajectories=True,
            show_progress=True,
        ).items()
    )
    results = dict(ordered_results)

    deterministic = results["deterministic"]
    if not isinstance(deterministic, ScenarioResult):
        raise TypeError("Deterministic baseline did not produce a ScenarioResult")

    comparisons = {
        metric: compare_scenarios(
            results,
            metric=metric,
            deterministic_baseline=_metric_baseline(deterministic, metric),
        )
        for metric in ("terminal_value", "npv", "lev")
    }

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "seed": SEED,
        "n_trajectories": N_TRAJECTORIES,
        "experiment_type": "no_thinning",
        "scenario_names": [config.name for config in ALL_SCENARIOS_NOTHIN],
        "scenarios": [asdict(config) for config in ALL_SCENARIOS_NOTHIN],
    }
    _write_json(OUTPUT_DIR / "manifest.json", manifest)
    _write_json(
        OUTPUT_DIR / "deterministic_baseline.json",
        {
            "terminal_value": _metric_baseline(deterministic, "terminal_value"),
            "npv": deterministic.npv,
            "lev": deterministic.lev,
        },
    )

    summary_rows = _build_summary_rows(ordered_results, comparisons)
    _write_csv(OUTPUT_DIR / "scenario_summaries.csv", summary_rows)

    for metric, comparison in comparisons.items():
        _write_csv(
            OUTPUT_DIR / f"{metric}_comparison.csv",
            _comparison_rows(comparison),
        )

    _save_raw_outputs(OUTPUT_DIR, ordered_results)
    _generate_figures(OUTPUT_DIR, ordered_results, deterministic)

    print("\n" + "=" * 60)
    print("No-Thinning Experiment Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
