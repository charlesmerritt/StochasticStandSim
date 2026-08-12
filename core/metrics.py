"""Metrics and summary statistics for Monte Carlo simulation results.

This module provides functions to summarize distributions of terminal values,
NPVs, and other economic metrics from batch simulations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from core.simulate import BatchResult, ScenarioResult


@dataclass
class DistributionSummary:
    """Summary statistics for a distribution of values.
    
    Attributes:
        mean: Arithmetic mean
        std: Standard deviation
        median: 50th percentile
        p5: 5th percentile
        p25: 25th percentile (first quartile)
        p75: 75th percentile (third quartile)
        p95: 95th percentile
        var_5: Value at Risk at 5% (same as p5, loss threshold)
        cvar_5: Conditional VaR (expected value below 5th percentile)
        min: Minimum value
        max: Maximum value
        n: Number of observations
    """
    mean: float
    std: float
    median: float
    p5: float
    p25: float
    p75: float
    p95: float
    var_5: float
    cvar_5: float
    min: float
    max: float
    n: int


def summarize_distribution(values: np.ndarray) -> DistributionSummary:
    """Compute summary statistics for a distribution.
    
    Args:
        values: Array of values to summarize
    
    Returns:
        DistributionSummary with all statistics
    """
    p5 = float(np.percentile(values, 5))
    
    # CVaR: expected value of observations below the 5th percentile
    below_p5 = values[values <= p5]
    cvar_5 = float(np.mean(below_p5)) if len(below_p5) > 0 else p5
    
    return DistributionSummary(
        mean=float(np.mean(values)),
        std=float(np.std(values)),
        median=float(np.median(values)),
        p5=p5,
        p25=float(np.percentile(values, 25)),
        p75=float(np.percentile(values, 75)),
        p95=float(np.percentile(values, 95)),
        var_5=p5,
        cvar_5=cvar_5,
        min=float(np.min(values)),
        max=float(np.max(values)),
        n=len(values),
    )


def probability_below_threshold(values: np.ndarray, threshold: float) -> float:
    """Compute the share of observations strictly below a threshold."""
    if len(values) == 0:
        return 0.0
    return float(np.mean(values < threshold))


def _get_metric_values(
    result: BatchResult | ScenarioResult,
    metric: str,
) -> np.ndarray | float:
    """Extract a metric array from a batch result or scalar from a scenario result."""
    from core.simulate import BatchResult

    if isinstance(result, BatchResult):
        if metric == "terminal_value":
            return result.terminal_values
        if metric == "npv":
            return result.npvs
        if metric == "lev":
            return result.levs
        raise ValueError(f"Unknown metric: {metric}")

    if metric == "terminal_value":
        return result.terminal_yield.net_revenue if result.terminal_yield else 0.0
    if metric == "npv":
        return result.npv
    if metric == "lev":
        return result.lev
    raise ValueError(f"Unknown metric: {metric}")


def _infer_deterministic_baseline(
    results: dict[str, BatchResult | ScenarioResult],
    metric: str,
) -> float | None:
    """Infer the deterministic baseline value from a results mapping if present."""
    from core.simulate import ScenarioResult

    for result in results.values():
        if isinstance(result, ScenarioResult) and result.scenario_type == "deterministic":
            value = _get_metric_values(result, metric)
            return float(value)
    return None


def summarize_batch(batch: BatchResult) -> dict[str, DistributionSummary]:
    """Summarize all distributions from a batch result.
    
    Args:
        batch: BatchResult from run_batch()
    
    Returns:
        Dict with keys 'terminal_value', 'npv', 'lev', 'thin_revenue'
    """
    return {
        "terminal_value": summarize_distribution(batch.terminal_values),
        "npv": summarize_distribution(batch.npvs),
        "lev": summarize_distribution(batch.levs),
        "thin_revenue": summarize_distribution(batch.thin_revenues),
    }


def compare_scenarios(
    results: dict[str, BatchResult | ScenarioResult],
    metric: str = "npv",
    deterministic_baseline: float | None = None,
) -> dict[str, dict[str, float]]:
    """Compare scenarios by a specific metric.
    
    Args:
        results: Dict from run_batch_scenarios()
        metric: One of 'terminal_value', 'npv', 'lev'
        deterministic_baseline: Optional deterministic baseline threshold for
            downside probability. If omitted, inferred from deterministic result.
    
    Returns:
        Dict mapping scenario name to summary stats dict
    """
    from core.simulate import BatchResult, ScenarioResult
    
    comparison: dict[str, dict[str, float]] = {}
    baseline_value = deterministic_baseline
    if baseline_value is None:
        baseline_value = _infer_deterministic_baseline(results, metric)
    
    for name, result in results.items():
        if isinstance(result, BatchResult):
            values = np.asarray(_get_metric_values(result, metric), dtype=float)
            summary = summarize_distribution(values)
            comparison[name] = {
                "mean": summary.mean,
                "std": summary.std,
                "median": summary.median,
                "p5": summary.p5,
                "p95": summary.p95,
                "cvar_5": summary.cvar_5,
                "downside_prob_vs_deterministic": (
                    probability_below_threshold(values, baseline_value)
                    if baseline_value is not None
                    else float("nan")
                ),
                "n": summary.n,
            }
        else:
            # Deterministic scenario - single value
            value = float(_get_metric_values(result, metric))
            comparison[name] = {
                "mean": value,
                "std": 0.0,
                "median": value,
                "p5": value,
                "p95": value,
                "cvar_5": value,
                "downside_prob_vs_deterministic": 0.0,
                "n": 1,
            }
    
    return comparison


def print_comparison_table(
    comparison: dict[str, dict[str, float]],
    metric_name: str = "NPV",
) -> None:
    """Print a formatted comparison table.
    
    Args:
        comparison: Output from compare_scenarios()
        metric_name: Name to display in header
    """
    print(f"\n{'='*80}")
    print(f"Scenario Comparison: {metric_name}")
    print("=" * 80)
    print(
        f"{'Scenario':<15} {'Mean':>10} {'Std':>10} {'Median':>10} "
        f"{'P5':>10} {'P95':>10} {'CVaR5':>10} {'P<Det':>10}"
    )
    print("-" * 80)
    
    for name, stats in comparison.items():
        print(
            f"{name:<15} "
            f"${stats['mean']:>9.0f} "
            f"${stats['std']:>9.0f} "
            f"${stats['median']:>9.0f} "
            f"${stats['p5']:>9.0f} "
            f"${stats['p95']:>9.0f} "
            f"${stats['cvar_5']:>9.0f} "
            f"{stats['downside_prob_vs_deterministic']:>9.1%}"
        )


def scenario_ranking(
    comparison: dict[str, dict[str, float]],
    by: str = "mean",
    ascending: bool = False,
) -> list[tuple[str, float]]:
    """Rank scenarios by a statistic.
    
    Args:
        comparison: Output from compare_scenarios()
        by: Statistic to rank by ('mean', 'median', 'p5', 'cvar_5', etc.)
        ascending: If True, lowest values first
    
    Returns:
        List of (scenario_name, value) tuples, sorted
    """
    items = [(name, stats[by]) for name, stats in comparison.items()]
    return sorted(items, key=lambda x: x[1], reverse=not ascending)


if __name__ == "__main__":
    # Demo with synthetic data
    rng = np.random.default_rng(42)
    
    # Simulate some batch results
    values = rng.normal(1000, 200, 1000)
    summary = summarize_distribution(values)
    
    print("Distribution Summary (synthetic data):")
    print(f"  Mean:   ${summary.mean:.2f}")
    print(f"  Std:    ${summary.std:.2f}")
    print(f"  Median: ${summary.median:.2f}")
    print(f"  P5:     ${summary.p5:.2f}")
    print(f"  P95:    ${summary.p95:.2f}")
    print(f"  VaR5:   ${summary.var_5:.2f}")
    print(f"  CVaR5:  ${summary.cvar_5:.2f}")
    print(f"  Range:  ${summary.min:.2f} - ${summary.max:.2f}")
    print(f"  N:      {summary.n}")
