"""Visualization module for stochastic forest simulation.

Implements the visualization targets from PLANNING.md Section 10 and
the directly reproducible figure families from the legacy examples:
- Deterministic vs stochastic trajectory comparisons
- Disturbance frequency summaries
- Growth validation diagnostics
- Dominant-height debug views
- Deterministic and stochastic product-distribution figures
- Stochastic growth demo figures
- Disturbance-regime trajectory comparisons
- Scenario-level histograms, boxplots, downside risk, and heatmaps

All functions return matplotlib Figure objects for flexibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from core.simulate import BatchResult, ScenarioResult


def _require_trajectories(batch: BatchResult) -> list[ScenarioResult]:
    """Return stored trajectories or raise if they were not retained."""
    if batch.trajectories is None:
        raise ValueError("BatchResult must have store_trajectories=True")
    return batch.trajectories


def _compute_qmd(tpa: float, ba: float) -> float:
    """Compute quadratic mean diameter from BA and TPA."""
    if tpa <= 0 or ba <= 0:
        return 0.0
    return float(np.sqrt((ba / tpa) / 0.005454154))


def _trajectory_axis(result: ScenarioResult, attr: str = "age") -> np.ndarray:
    """Extract a trajectory axis such as age or year."""
    return np.asarray([getattr(record, attr) for record in result.trajectory], dtype=float)


def _trajectory_matrix(batch: BatchResult, variable: str) -> np.ndarray:
    """Extract a state variable matrix from stored trajectories."""
    trajectories = _require_trajectories(batch)
    n_traj = len(trajectories)
    n_steps = len(trajectories[0].trajectory)
    values = np.zeros((n_traj, n_steps))

    for i, result in enumerate(trajectories):
        for j, record in enumerate(result.trajectory):
            if variable == "qmd":
                values[i, j] = _compute_qmd(record.tpa, record.ba)
            else:
                values[i, j] = getattr(record, variable)

    return values


def _scenario_series(
    result: BatchResult | ScenarioResult,
    variable: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return x-values, means, and standard deviations for a trajectory variable."""
    from core.simulate import BatchResult

    if isinstance(result, BatchResult):
        trajectories = _require_trajectories(result)
        x = _trajectory_axis(trajectories[0], "age")
        values = _trajectory_matrix(result, variable)
        return x, np.mean(values, axis=0), np.std(values, axis=0)

    x = _trajectory_axis(result, "age")
    if variable == "qmd":
        y = np.asarray(
            [_compute_qmd(record.tpa, record.ba) for record in result.trajectory],
            dtype=float,
        )
    else:
        y = np.asarray([getattr(record, variable) for record in result.trajectory], dtype=float)
    return x, y, np.zeros_like(y)


def _build_state_from_record(result: ScenarioResult, record) -> object:
    """Reconstruct a stand state from a trajectory record for derived calculations."""
    from core.state import StandState

    return StandState(
        age=record.age,
        hd=record.hd,
        tpa=record.tpa,
        ba=record.ba,
        si25=result.initial_state.si25,
        region=result.initial_state.region,
        phwd=result.initial_state.phwd,
    )


def _product_time_series(result: ScenarioResult) -> dict[str, np.ndarray]:
    """Compute per-year product metrics from a trajectory."""
    from core.pmrc_model import PMRCModel
    from core.products import compute_harvest_value, estimate_product_distribution

    pmrc = PMRCModel(region=result.initial_state.region)
    ages = _trajectory_axis(result, "age")
    ba = np.asarray([record.ba for record in result.trajectory], dtype=float)
    tpa = np.asarray([record.tpa for record in result.trajectory], dtype=float)
    hd = np.asarray([record.hd for record in result.trajectory], dtype=float)
    qmd = np.asarray([_compute_qmd(record.tpa, record.ba) for record in result.trajectory])

    vol_pulp = np.zeros(len(result.trajectory))
    vol_cns = np.zeros(len(result.trajectory))
    vol_saw = np.zeros(len(result.trajectory))
    vol_total = np.zeros(len(result.trajectory))
    frac_pulp = np.zeros(len(result.trajectory))
    frac_cns = np.zeros(len(result.trajectory))
    frac_saw = np.zeros(len(result.trajectory))
    harvest_value = np.zeros(len(result.trajectory))

    for i, record in enumerate(result.trajectory):
        state = _build_state_from_record(result, record)
        products = estimate_product_distribution(
            pmrc=pmrc,
            age=state.age,
            ba=state.ba,
            tpa=state.tpa,
            hd=state.hd,
            region=state.region,
            phwd=state.phwd,
        )
        vol_pulp[i] = products.vol_pulp
        vol_cns[i] = products.vol_cns
        vol_saw[i] = products.vol_saw
        vol_total[i] = products.total_vol
        frac_pulp[i] = products.pulp_fraction
        frac_cns[i] = products.cns_fraction
        frac_saw[i] = products.saw_fraction
        harvest_value[i] = compute_harvest_value(products, result.prices, result.costs)

    return {
        "age": ages,
        "hd": hd,
        "ba": ba,
        "tpa": tpa,
        "qmd": qmd,
        "vol_pulp": vol_pulp,
        "vol_cns": vol_cns,
        "vol_saw": vol_saw,
        "vol_total": vol_total,
        "frac_pulp": frac_pulp,
        "frac_cns": frac_cns,
        "frac_saw": frac_saw,
        "harvest_value": harvest_value,
    }


def _batch_product_summary(batch: BatchResult) -> dict[str, np.ndarray]:
    """Aggregate derived product metrics across stochastic trajectories."""
    trajectories = _require_trajectories(batch)
    n_traj = len(trajectories)
    n_steps = len(trajectories[0].trajectory)

    qmd = np.zeros((n_traj, n_steps))
    vol_pulp = np.zeros((n_traj, n_steps))
    vol_cns = np.zeros((n_traj, n_steps))
    vol_saw = np.zeros((n_traj, n_steps))
    harvest_value = np.zeros((n_traj, n_steps))
    ba = _trajectory_matrix(batch, "ba")

    for i, result in enumerate(trajectories):
        series = _product_time_series(result)
        qmd[i, :] = series["qmd"]
        vol_pulp[i, :] = series["vol_pulp"]
        vol_cns[i, :] = series["vol_cns"]
        vol_saw[i, :] = series["vol_saw"]
        harvest_value[i, :] = series["harvest_value"]

    return {
        "age": _trajectory_axis(trajectories[0], "age"),
        "ba_mean": np.mean(ba, axis=0),
        "ba_std": np.std(ba, axis=0),
        "qmd_mean": np.mean(qmd, axis=0),
        "qmd_std": np.std(qmd, axis=0),
        "vol_pulp_mean": np.mean(vol_pulp, axis=0),
        "vol_cns_mean": np.mean(vol_cns, axis=0),
        "vol_saw_mean": np.mean(vol_saw, axis=0),
        "harvest_value_mean": np.mean(harvest_value, axis=0),
        "harvest_value_std": np.std(harvest_value, axis=0),
    }


def plot_trajectory_fan(
    deterministic: ScenarioResult,
    stochastic_batch: BatchResult,
    variable: str = "ba",
    percentiles: tuple[float, float] = (5, 95),
    figsize: tuple[float, float] = (10, 6),
) -> Figure:
    """Plot deterministic trajectory with stochastic fan (confidence ribbon).
    
    PLANNING.md Section 10.A: Shows whether stochastic simulator follows PMRC trend.
    
    Args:
        deterministic: Result from deterministic scenario
        stochastic_batch: BatchResult with stored trajectories
        variable: State variable to plot ('ba', 'tpa', 'hd', or 'vol')
        percentiles: Lower and upper percentiles for ribbon
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt
    
    if stochastic_batch.trajectories is None:
        raise ValueError("BatchResult must have store_trajectories=True")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract years
    years = [rec.year for rec in deterministic.trajectory]
    
    # Deterministic trajectory
    det_values = [getattr(rec, variable) for rec in deterministic.trajectory]
    ax.plot(years, det_values, 'k-', linewidth=2, label='Deterministic (PMRC)')
    
    # Stochastic trajectories - compute percentiles
    n_years = len(years)
    n_traj = len(stochastic_batch.trajectories)
    
    values_matrix = np.zeros((n_traj, n_years))
    for i, result in enumerate(stochastic_batch.trajectories):
        for j, rec in enumerate(result.trajectory):
            values_matrix[i, j] = getattr(rec, variable)
    
    median = np.median(values_matrix, axis=0)
    lower = np.percentile(values_matrix, percentiles[0], axis=0)
    upper = np.percentile(values_matrix, percentiles[1], axis=0)
    
    ax.plot(years, median, 'b-', linewidth=1.5, label='Stochastic median')
    ax.fill_between(years, lower, upper, alpha=0.3, color='blue',
                    label=f'{percentiles[0]}-{percentiles[1]}% interval')
    
    ax.set_xlabel('Year')
    ax.set_ylabel(variable.upper())
    ax.set_title(f'Trajectory Fan: {variable.upper()}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_terminal_histogram(
    batch: BatchResult,
    deterministic_value: float | None = None,
    metric: str = "npv",
    bins: int = 50,
    figsize: tuple[float, float] = (10, 6),
) -> Figure:
    """Plot histogram of terminal values with deterministic reference.
    
    PLANNING.md Section 10.B: Core figure showing distribution of outcomes.
    
    Args:
        batch: BatchResult from stochastic scenario
        deterministic_value: Optional deterministic baseline value
        metric: 'terminal_value', 'npv', or 'lev'
        bins: Number of histogram bins
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if metric == "terminal_value":
        values = batch.terminal_values
        xlabel = "Terminal Value ($/ac)"
    elif metric == "npv":
        values = batch.npvs
        xlabel = "NPV ($/ac)"
    elif metric == "lev":
        values = batch.levs
        xlabel = "LEV ($/ac)"
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    ax.hist(values, bins=bins, density=True, alpha=0.7, color='steelblue',
            edgecolor='white', label=f'{batch.scenario_name} (n={batch.n_trajectories})')
    
    if deterministic_value is not None:
        ax.axvline(deterministic_value, color='red', linestyle='--', linewidth=2,
                   label=f'Deterministic: ${deterministic_value:.0f}')
    
    # Add mean line
    mean_val = float(np.mean(values))
    ax.axvline(mean_val, color='green', linestyle='-', linewidth=1.5,
               label=f'Mean: ${mean_val:.0f}')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    ax.set_title(f'Distribution of {metric.upper()}: {batch.scenario_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_scenario_boxplots(
    results: dict[str, BatchResult | ScenarioResult],
    metric: str = "npv",
    figsize: tuple[float, float] = (12, 6),
) -> Figure:
    """Plot boxplots comparing scenarios.
    
    PLANNING.md Section 10.C: Compare process noise only, disturbance only, combined.
    
    Args:
        results: Dict from run_batch_scenarios()
        metric: 'terminal_value', 'npv', or 'lev'
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt
    from core.simulate import BatchResult, ScenarioResult
    
    fig, ax = plt.subplots(figsize=figsize)
    
    names = []
    data = []
    
    for name, result in results.items():
        names.append(name)
        if isinstance(result, BatchResult):
            if metric == "terminal_value":
                data.append(result.terminal_values)
            elif metric == "npv":
                data.append(result.npvs)
            elif metric == "lev":
                data.append(result.levs)
        else:
            # Deterministic - single value
            if metric == "terminal_value":
                val = result.terminal_yield.net_revenue if result.terminal_yield else 0.0
            elif metric == "npv":
                val = result.npv
            elif metric == "lev":
                val = result.lev
            else:
                val = 0.0
            data.append([val])
    
    ax.boxplot(data, labels=names)
    ax.set_ylabel(f'{metric.upper()} ($/ac)')
    ax.set_title(f'Scenario Comparison: {metric.upper()}')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_product_breakdown(
    results: dict[str, BatchResult | ScenarioResult],
    figsize: tuple[float, float] = (12, 6),
) -> Figure:
    """Plot product distribution breakdown by scenario.
    
    PLANNING.md Section 10.D: Mean share and variability by product class.
    
    Args:
        results: Dict from run_batch_scenarios()
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt
    from core.simulate import BatchResult, ScenarioResult
    
    fig, ax = plt.subplots(figsize=figsize)
    
    scenarios = list(results.keys())
    n_scenarios = len(scenarios)
    
    pulp_means = []
    cns_means = []
    saw_means = []
    
    for name in scenarios:
        result = results[name]
        if isinstance(result, BatchResult):
            pulp_means.append(float(np.mean(result.vol_pulp)))
            cns_means.append(float(np.mean(result.vol_cns)))
            saw_means.append(float(np.mean(result.vol_saw)))
        elif isinstance(result, ScenarioResult) and result.terminal_yield:
            pulp_means.append(result.terminal_yield.vol_pulp)
            cns_means.append(result.terminal_yield.vol_cns)
            saw_means.append(result.terminal_yield.vol_saw)
        else:
            pulp_means.append(0)
            cns_means.append(0)
            saw_means.append(0)
    
    x = np.arange(n_scenarios)
    width = 0.25
    
    ax.bar(x - width, pulp_means, width, label='Pulpwood', color='#8B4513')
    ax.bar(x, cns_means, width, label='Chip-n-Saw', color='#D2691E')
    ax.bar(x + width, saw_means, width, label='Sawtimber', color='#228B22')
    
    ax.set_ylabel('Volume (cuft/ac)')
    ax.set_title('Product Distribution by Scenario')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_downside_risk(
    results: dict[str, BatchResult],
    metric: str = "npv",
    figsize: tuple[float, float] = (10, 6),
) -> Figure:
    """Plot downside risk metrics (mean, P5, VaR, CVaR).
    
    PLANNING.md Section 10.E: Useful for academic risk analysis.
    
    Args:
        results: Dict of BatchResults (stochastic scenarios only)
        metric: 'terminal_value', 'npv', or 'lev'
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt
    from core.metrics import summarize_distribution
    
    fig, ax = plt.subplots(figsize=figsize)
    
    scenarios = list(results.keys())
    n = len(scenarios)
    
    means = []
    p5s = []
    cvars = []
    
    for name in scenarios:
        batch = results[name]
        if metric == "terminal_value":
            values = batch.terminal_values
        elif metric == "npv":
            values = batch.npvs
        elif metric == "lev":
            values = batch.levs
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        summary = summarize_distribution(values)
        means.append(summary.mean)
        p5s.append(summary.p5)
        cvars.append(summary.cvar_5)
    
    x = np.arange(n)
    width = 0.25
    
    ax.bar(x - width, means, width, label='Mean', color='steelblue')
    ax.bar(x, p5s, width, label='P5 (VaR)', color='orange')
    ax.bar(x + width, cvars, width, label='CVaR (5%)', color='red')
    
    ax.set_ylabel(f'{metric.upper()} ($/ac)')
    ax.set_title(f'Downside Risk Analysis: {metric.upper()}')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_sensitivity_heatmap(
    results: dict[str, BatchResult | ScenarioResult],
    lambda_levels: list[float],
    p_dist_levels: list[float],
    metric: str = "npv",
    stat: str = "mean",
    figsize: tuple[float, float] = (10, 8),
) -> Figure:
    """Plot sensitivity heatmap over λ_proc × p_dist grid.
    
    PLANNING.md Section 10.F: Very effective summary figure.
    
    Args:
        results: Dict from run_batch_scenarios() with full matrix
        lambda_levels: Process noise levels used
        p_dist_levels: Disturbance probabilities used
        metric: 'terminal_value', 'npv', or 'lev'
        stat: 'mean', 'std', 'p5', or 'cvar_5'
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt
    from core.metrics import summarize_distribution
    from core.simulate import BatchResult, ScenarioResult
    
    fig, ax = plt.subplots(figsize=figsize)
    
    n_lambda = len(lambda_levels)
    n_pdist = len(p_dist_levels)
    
    heatmap_data = np.full((n_lambda, n_pdist), np.nan)

    def _result_params(result: BatchResult | ScenarioResult) -> tuple[float, float]:
        if isinstance(result, BatchResult):
            config = result.scenario_config
        else:
            config = result.scenario_config
        lam = config.noise_params.lambda_proc if config and config.noise_params else 0.0
        p_dist = (
            config.disturbance_params.p_dist
            if config and config.disturbance_params
            else 0.0
        )
        return lam, p_dist
    
    for result in results.values():
        lam, p_dist = _result_params(result)
        if lam not in lambda_levels or p_dist not in p_dist_levels:
            continue

        i = lambda_levels.index(lam)
        j = p_dist_levels.index(p_dist)

        if isinstance(result, BatchResult):
            if metric == "terminal_value":
                values = result.terminal_values
            elif metric == "npv":
                values = result.npvs
            elif metric == "lev":
                values = result.levs
            else:
                raise ValueError(f"Unknown metric: {metric}")

            summary = summarize_distribution(values)
            value = getattr(summary, stat)
        elif metric == "npv":
            value = result.npv
        elif metric == "lev":
            value = result.lev
        else:
            value = result.terminal_yield.net_revenue if result.terminal_yield else 0.0

        heatmap_data[i, j] = value

    if np.isnan(heatmap_data).any():
        raise ValueError("Heatmap data incomplete for provided lambda and p_dist levels")
    
    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto')
    
    ax.set_xticks(np.arange(n_pdist))
    ax.set_yticks(np.arange(n_lambda))
    ax.set_xticklabels([f'{p:.3f}' for p in p_dist_levels])
    ax.set_yticklabels([f'{l:.2f}' for l in lambda_levels])
    
    ax.set_xlabel('p_dist (disturbance probability)')
    ax.set_ylabel('λ_proc (process noise)')
    ax.set_title(f'Sensitivity Heatmap: {stat.upper()} {metric.upper()}')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f'{metric.upper()} ($/ac)')
    
    # Add text annotations
    for i in range(n_lambda):
        for j in range(n_pdist):
            text = ax.text(j, i, f'${heatmap_data[i, j]:.0f}',
                          ha='center', va='center', color='black', fontsize=8)
    
    plt.tight_layout()
    return fig


def plot_deterministic_vs_stochastic_comparison(
    deterministic: ScenarioResult,
    stochastic_batch: BatchResult,
    sample_trajectories: int = 50,
    percentiles: tuple[float, float] = (5, 95),
    figsize: tuple[float, float] = (14, 4.5),
) -> Figure:
    """Plot BA, TPA, and HD deterministic baselines against stochastic ensembles."""
    import matplotlib.pyplot as plt

    trajectories = _require_trajectories(stochastic_batch)
    ages = _trajectory_axis(deterministic, "age")
    panel_specs = [
        ("ba", "Basal Area (ft²/ac)", "Basal Area", "steelblue"),
        ("tpa", "Trees Per Acre", "Survival / Mortality", "forestgreen"),
        ("hd", "Dominant Height (ft)", "Height Growth", "darkorange"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, (variable, ylabel, title, color) in zip(axes, panel_specs):
        values = _trajectory_matrix(stochastic_batch, variable)
        det_values = np.asarray(
            [getattr(record, variable) for record in deterministic.trajectory],
            dtype=float,
        )

        for i in range(min(sample_trajectories, len(trajectories))):
            ax.plot(ages, values[i], color=color, alpha=0.12, linewidth=0.7)

        ax.fill_between(
            ages,
            np.percentile(values, percentiles[0], axis=0),
            np.percentile(values, percentiles[1], axis=0),
            color=color,
            alpha=0.2,
            label=f"{percentiles[0]}-{percentiles[1]}% interval",
        )
        ax.plot(ages, det_values, color="darkred", linewidth=2.5, label="Deterministic PMRC")
        ax.plot(
            ages,
            np.mean(values, axis=0),
            color=color,
            linewidth=2.0,
            linestyle="--",
            label="Stochastic mean",
        )
        ax.set_xlabel("Stand Age (years)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    fig.suptitle("Deterministic PMRC vs Stochastic Wrapper", y=1.02, fontsize=13)
    fig.tight_layout()
    return fig


def plot_disturbance_frequency_histograms(
    results: dict[str, BatchResult],
    figsize: tuple[float, float] = (11, 4),
    normalize: bool = True,
) -> Figure:
    """Plot disturbance-count histograms for one or more stochastic scenarios."""
    import matplotlib.pyplot as plt

    if not results:
        raise ValueError("At least one batch result is required")

    scenario_names = list(results.keys())
    fig, axes = plt.subplots(1, len(scenario_names), figsize=figsize, sharey=True)
    if len(scenario_names) == 1:
        axes = [axes]

    horizon = next(iter(results.values())).scenario_config.rotation_length
    palette = ["#c0392b", "#e67e22", "#27ae60", "#2980b9", "#8e44ad"]

    for ax, name, color in zip(axes, scenario_names, palette * len(scenario_names)):
        batch = results[name]
        counts = batch.disturbance_counts.astype(int)
        bins = np.arange(0, counts.max(initial=0) + 2) - 0.5
        hist, edges = np.histogram(counts, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        heights = hist / len(counts) if normalize else hist
        ax.bar(centers, heights, width=0.8, color=color, alpha=0.8)
        ax.set_xticks(range(int(centers.max()) + 1 if len(centers) else 1))
        ax.set_xlabel(f"Disturbances in {horizon} years")
        ax.set_title(name)
        ax.grid(True, linestyle="--", alpha=0.3)

    axes[0].set_ylabel("Probability" if normalize else "Count")
    fig.suptitle(f"Disturbance occurrence over {horizon} years", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def plot_growth_validation(
    stochastic_batch: BatchResult,
    deterministic: ScenarioResult | None = None,
    trajectory_limit: int = 50,
    figsize: tuple[float, float] = (16, 10),
) -> Figure:
    """Plot growth-validation diagnostics for a stochastic batch."""
    import matplotlib.pyplot as plt

    trajectories = _require_trajectories(stochastic_batch)
    ages = _trajectory_axis(trajectories[0], "year")
    actual_ages = _trajectory_axis(trajectories[0], "age")
    hd = _trajectory_matrix(stochastic_batch, "hd")
    ba = _trajectory_matrix(stochastic_batch, "ba")
    tpa = _trajectory_matrix(stochastic_batch, "tpa")
    qmd = _trajectory_matrix(stochastic_batch, "qmd")

    config = stochastic_batch.scenario_config
    thin_params = config.thin_params
    thin_year = (
        int(round(thin_params.trigger_age - config.age0))
        if thin_params is not None
        else None
    )
    si25 = config.si25
    n_disturbed = int(np.sum(stochastic_batch.disturbance_occurred))
    n_thinned = int(
        sum(1 for result in trajectories if result.thin_occurred)
    )

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(
        f"Growth Model Validation: {stochastic_batch.n_trajectories} trajectories",
        fontsize=14,
    )

    panel_data = [
        (axes[0, 0], hd, "blue", "darkblue", "Dominant Height (ft)", "Dominant Height (HD)"),
        (axes[0, 1], ba, "green", "darkgreen", "Basal Area (ft²/ac)", "Basal Area (BA)"),
        (axes[0, 2], tpa, "orange", "darkorange", "Trees per Acre", "Trees per Acre (TPA)"),
        (axes[1, 0], qmd, "purple", "darkviolet", "QMD (inches)", "Quadratic Mean Diameter (QMD)"),
    ]

    for ax, values, line_color, mean_color, ylabel, title in panel_data:
        for i in range(min(trajectory_limit, len(trajectories))):
            ax.plot(ages, values[i], alpha=0.3, color=line_color, linewidth=0.8)
        ax.plot(ages, np.mean(values, axis=0), color=mean_color, linewidth=2, label="Mean")
        if thin_year is not None:
            ax.axvline(thin_year, color="purple", linestyle="--", alpha=0.7)
        ax.set_xlabel("Year")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, ages[-1])

    axes[0, 0].axhline(si25, color="red", linestyle="--", linewidth=2, label=f"SI25 = {si25}")
    axes[0, 0].axvline(25 - config.age0, color="green", linestyle=":", alpha=0.7, label="Base age 25")
    axes[0, 0].legend(loc="lower right")

    if thin_params is not None:
        axes[0, 1].axhline(
            thin_params.ba_threshold,
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"Thin threshold = {thin_params.ba_threshold}",
        )
        axes[0, 1].axhline(
            thin_params.residual_ba,
            color="red",
            linestyle=":",
            linewidth=1.5,
            label=f"Thin target = {thin_params.residual_ba}",
        )
        axes[0, 1].legend(loc="lower right")

    axes[1, 1].axis("off")
    year25_idx = int(np.argmin(np.abs(actual_ages - 25.0)))
    summary = f"""
Growth Model Validation Summary
═══════════════════════════════════════

Configuration:
  SI25 = {si25:.1f} ft
  Initial TPA = {config.tpa0:.0f}
  Rotation = {config.rotation_length} years

Results at age 25:
  Mean HD = {np.mean(hd[:, year25_idx]):.1f} ft
  Mean BA = {np.mean(ba[:, year25_idx]):.1f} ft²/ac
  Mean TPA = {np.mean(tpa[:, year25_idx]):.0f}
  Mean QMD = {np.mean(qmd[:, year25_idx]):.2f} in

Results at year {config.rotation_length}:
  Mean HD = {np.mean(hd[:, -1]):.1f} ft
  Mean BA = {np.mean(ba[:, -1]):.1f} ft²/ac
  Mean TPA = {np.mean(tpa[:, -1]):.0f}
  Mean QMD = {np.mean(qmd[:, -1]):.2f} in

Events:
  Trajectories thinned: {n_thinned}/{stochastic_batch.n_trajectories}
  Trajectories disturbed: {n_disturbed}/{stochastic_batch.n_trajectories}
"""
    axes[1, 1].text(
        0.05,
        0.5,
        summary,
        transform=axes[1, 1].transAxes,
        fontsize=10,
        verticalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    hd_mean = np.mean(hd, axis=0)
    axes[1, 2].fill_between(
        ages,
        np.min(hd, axis=0),
        np.max(hd, axis=0),
        alpha=0.2,
        color="blue",
        label="HD range",
    )
    axes[1, 2].plot(ages, hd_mean, color="blue", linewidth=2, label="Mean HD (stochastic)")
    if deterministic is not None:
        axes[1, 2].plot(
            _trajectory_axis(deterministic, "year"),
            np.asarray([record.hd for record in deterministic.trajectory]),
            color="red",
            linewidth=2,
            linestyle="--",
            label="Deterministic HD",
        )
    axes[1, 2].axhline(si25, color="green", linestyle=":", alpha=0.7)
    axes[1, 2].set_xlabel("Year")
    axes[1, 2].set_ylabel("Dominant Height (ft)")
    axes[1, 2].set_title("HD: Stochastic vs Deterministic")
    axes[1, 2].legend(loc="lower right")
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_xlim(0, ages[-1])

    fig.tight_layout()
    return fig


def plot_hd_debug(
    deterministic: ScenarioResult,
    stochastic_batch: BatchResult,
    target_age: float = 25.0,
    figsize: tuple[float, float] = (7, 5),
) -> Figure:
    """Plot deterministic and stochastic dominant-height trajectories for debugging."""
    import matplotlib.pyplot as plt

    ages = _trajectory_axis(deterministic, "age")
    hd_det = np.asarray([record.hd for record in deterministic.trajectory], dtype=float)
    hd = _trajectory_matrix(stochastic_batch, "hd")
    hd_mean = np.mean(hd, axis=0)
    hd_std = np.std(hd, axis=0)
    idx = int(np.argmin(np.abs(ages - target_age)))

    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    ax.plot(ages, hd_det, label="Deterministic PMRC", color="#34495e", linewidth=2.0)
    ax.plot(ages, hd_mean, label="Stochastic mean HD", color="#e74c3c", linewidth=2.0)
    ax.fill_between(
        ages,
        hd_mean - hd_std,
        hd_mean + hd_std,
        color="#e74c3c",
        alpha=0.2,
        label="Stochastic ±1 SD",
    )
    ax.axvline(target_age, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.scatter([target_age], [hd_det[idx]], color="#34495e", s=35)
    ax.scatter([target_age], [hd_mean[idx]], color="#e74c3c", s=35)
    ax.annotate(
        f"Det: {hd_det[idx]:.1f} ft\nStoch: {hd_mean[idx]:.1f} ± {hd_std[idx]:.1f} ft",
        (target_age, hd_mean[idx]),
        xytext=(target_age + 1.0, hd_mean[idx]),
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Dominant height (ft)")
    ax.set_title(f"Dominant height debug (SI25={deterministic.initial_state.si25:.0f})")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_deterministic_product_distribution(
    deterministic: ScenarioResult,
    figsize: tuple[float, float] = (14, 8),
) -> Figure:
    """Plot deterministic stand development and product distribution over time."""
    import matplotlib.pyplot as plt
    from core.products import CUFT_TO_TON, HarvestCosts, ProductPrices

    results = _product_time_series(deterministic)
    ages = results["age"]
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    ax = axes[0, 0]
    ax.plot(ages, results["ba"], "b-", label="BA (ft²/ac)", linewidth=2)
    ax.set_xlabel("Stand Age (years)")
    ax.set_ylabel("Basal Area (ft²/ac)", color="b")
    ax.tick_params(axis="y", labelcolor="b")
    ax_twin = ax.twinx()
    ax_twin.plot(ages, results["tpa"], "r--", label="TPA", linewidth=2)
    ax_twin.set_ylabel("Trees per Acre", color="r")
    ax_twin.tick_params(axis="y", labelcolor="r")
    ax.set_title("Stand Development")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(ages, results["qmd"], "k-", linewidth=2, label="QMD")
    ax.axhline(y=6, color="green", linestyle="--", alpha=0.7, label='Pulp min (6")')
    ax.axhline(y=9, color="orange", linestyle="--", alpha=0.7, label='CNS min (9")')
    ax.axhline(y=12, color="red", linestyle="--", alpha=0.7, label='Saw min (12")')
    ax.set_xlabel("Stand Age (years)")
    ax.set_ylabel("Quadratic Mean Diameter (inches)")
    ax.set_title("Diameter Growth vs Product Thresholds")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 16)

    ax = axes[0, 2]
    ax.stackplot(
        ages,
        results["vol_pulp"],
        results["vol_cns"],
        results["vol_saw"],
        labels=["Pulpwood", "Chip-n-Saw", "Sawtimber"],
        colors=["#2ecc71", "#f39c12", "#e74c3c"],
        alpha=0.8,
    )
    ax.set_xlabel("Stand Age (years)")
    ax.set_ylabel("Volume (cuft/acre)")
    ax.set_title("Volume by Product Class")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.stackplot(
        ages,
        results["frac_pulp"] * 100.0,
        results["frac_cns"] * 100.0,
        results["frac_saw"] * 100.0,
        labels=["Pulpwood", "Chip-n-Saw", "Sawtimber"],
        colors=["#2ecc71", "#f39c12", "#e74c3c"],
        alpha=0.8,
    )
    ax.set_xlabel("Stand Age (years)")
    ax.set_ylabel("Percent of Merchantable Volume")
    ax.set_title("Product Mix Over Time")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(ages, results["harvest_value"], "g-", linewidth=2)
    ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax.fill_between(
        ages,
        results["harvest_value"],
        0,
        where=results["harvest_value"] > 0,
        color="green",
        alpha=0.3,
        label="Profitable",
    )
    ax.fill_between(
        ages,
        results["harvest_value"],
        0,
        where=results["harvest_value"] <= 0,
        color="red",
        alpha=0.3,
        label="Unprofitable",
    )
    ax.set_xlabel("Stand Age (years)")
    ax.set_ylabel("Net Harvest Value ($/acre)")
    ax.set_title("Harvest Profitability")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    prices = deterministic.prices or ProductPrices()
    costs = deterministic.costs or HarvestCosts()
    val_pulp = results["vol_pulp"] * CUFT_TO_TON * prices.pulpwood
    val_cns = results["vol_cns"] * CUFT_TO_TON * prices.chip_n_saw
    val_saw = results["vol_saw"] * CUFT_TO_TON * prices.sawtimber
    ax.stackplot(
        ages,
        val_pulp,
        val_cns,
        val_saw,
        labels=["Pulpwood", "Chip-n-Saw", "Sawtimber"],
        colors=["#2ecc71", "#f39c12", "#e74c3c"],
        alpha=0.8,
    )
    ax.axhline(y=costs.total, color="k", linestyle="--", label="Harvest Cost")
    ax.set_xlabel("Stand Age (years)")
    ax.set_ylabel("Gross Revenue ($/acre)")
    ax.set_title("Revenue by Product Class")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Product Distribution Validation: SI={deterministic.initial_state.si25:.0f}, "
        f"TPA₀={deterministic.initial_state.tpa:.0f}, {deterministic.initial_state.region.upper()} Region",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_stochastic_product_distribution(
    results_by_label: dict[str, BatchResult],
    stackplot_label: str | None = None,
    revenue_breakdown_label: str | None = None,
    figsize: tuple[float, float] = (14, 8),
) -> Figure:
    """Plot stochastic product-distribution summaries for several risk labels."""
    import matplotlib.pyplot as plt
    from core.products import CUFT_TO_TON, ProductPrices

    if not results_by_label:
        raise ValueError("At least one stochastic batch is required")

    summaries = {label: _batch_product_summary(batch) for label, batch in results_by_label.items()}
    labels = list(results_by_label.keys())
    stackplot_label = stackplot_label or labels[0]
    revenue_breakdown_label = revenue_breakdown_label or labels[-1]
    palette = ["green", "orange", "red", "#2980b9", "#8e44ad"]
    colors = {label: palette[i % len(palette)] for i, label in enumerate(labels)}

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    ax = axes[0, 0]
    for label in labels:
        summary = summaries[label]
        ax.plot(summary["age"], summary["ba_mean"], color=colors[label], linewidth=2, label=label)
    ax.set_xlabel("Stand Age (years)")
    ax.set_ylabel("Basal Area (ft²/ac)")
    ax.set_title("Mean Basal Area by Scenario")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for label in labels:
        summary = summaries[label]
        ax.plot(summary["age"], summary["qmd_mean"], color=colors[label], linewidth=2, label=label)
    ax.axhline(y=6, color="gray", linestyle="--", alpha=0.7, label='Pulp min (6")')
    ax.axhline(y=9, color="gray", linestyle=":", alpha=0.7, label='CNS min (9")')
    ax.axhline(y=12, color="gray", linestyle="-.", alpha=0.7, label='Saw min (12")')
    ax.set_xlabel("Stand Age (years)")
    ax.set_ylabel("Quadratic Mean Diameter (inches)")
    ax.set_title("Mean QMD vs Product Thresholds")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 14)

    ax = axes[0, 2]
    summary = summaries[stackplot_label]
    ax.stackplot(
        summary["age"],
        summary["vol_pulp_mean"],
        summary["vol_cns_mean"],
        summary["vol_saw_mean"],
        labels=["Pulpwood", "Chip-n-Saw", "Sawtimber"],
        colors=["#2ecc71", "#f39c12", "#e74c3c"],
        alpha=0.8,
    )
    ax.set_xlabel("Stand Age (years)")
    ax.set_ylabel("Volume (cuft/acre)")
    ax.set_title(f"Volume by Product ({stackplot_label})")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for label in labels:
        summary = summaries[label]
        total_vol = (
            summary["vol_pulp_mean"] +
            summary["vol_cns_mean"] +
            summary["vol_saw_mean"]
        )
        saw_frac = np.where(total_vol > 0, summary["vol_saw_mean"] / total_vol * 100, 0)
        ax.plot(summary["age"], saw_frac, color=colors[label], linewidth=2, label=label)
    ax.set_xlabel("Stand Age (years)")
    ax.set_ylabel("Sawtimber % of Volume")
    ax.set_title("Sawtimber Fraction by Scenario")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for label in labels:
        summary = summaries[label]
        ax.plot(
            summary["age"],
            summary["harvest_value_mean"],
            color=colors[label],
            linewidth=2,
            label=label,
        )
    ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax.set_xlabel("Stand Age (years)")
    ax.set_ylabel("Net Harvest Value ($/acre)")
    ax.set_title("Mean Harvest Value by Scenario")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    summary = summaries[revenue_breakdown_label]
    prices = results_by_label[revenue_breakdown_label].scenario_config.prices or ProductPrices()
    val_pulp = summary["vol_pulp_mean"] * CUFT_TO_TON * prices.pulpwood
    val_cns = summary["vol_cns_mean"] * CUFT_TO_TON * prices.chip_n_saw
    val_saw = summary["vol_saw_mean"] * CUFT_TO_TON * prices.sawtimber
    ax.stackplot(
        summary["age"],
        val_pulp,
        val_cns,
        val_saw,
        labels=["Pulpwood", "Chip-n-Saw", "Sawtimber"],
        colors=["#2ecc71", "#f39c12", "#e74c3c"],
        alpha=0.8,
    )
    ax.axhline(
        y=(results_by_label[revenue_breakdown_label].scenario_config.costs.total),
        color="k",
        linestyle="--",
        label="Harvest Cost",
    )
    ax.set_xlabel("Stand Age (years)")
    ax.set_ylabel("Gross Revenue ($/acre)")
    ax.set_title(f"Revenue by Product ({revenue_breakdown_label})")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Product Distribution Under Stochastic Growth (n={results_by_label[labels[0]].n_trajectories} trajectories)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_stochastic_growth_demo(
    stochastic_batch: BatchResult,
    trajectory_index: int = 0,
    figsize: tuple[float, float] = (11, 8),
) -> Figure:
    """Plot a single stochastic trajectory beside the ensemble mean."""
    import matplotlib.pyplot as plt

    trajectories = _require_trajectories(stochastic_batch)
    if trajectory_index >= len(trajectories):
        raise IndexError("trajectory_index out of bounds")

    ages = _trajectory_axis(trajectories[0], "age")
    hd = _trajectory_matrix(stochastic_batch, "hd")
    ba = _trajectory_matrix(stochastic_batch, "ba")
    tpa = _trajectory_matrix(stochastic_batch, "tpa")
    single = trajectories[trajectory_index]
    disturbance_ages = [single.trajectory[year].age for year in single.disturbance_years]

    fig, axes = plt.subplots(3, 2, figsize=figsize, sharex=True, dpi=200)
    labels = ["Dominant height (ft)", "Basal area (ft²/ac)", "Trees per acre"]
    colors = ["#2c3e50", "#c0392b", "#16a085"]
    series_single = [hd[trajectory_index], ba[trajectory_index], tpa[trajectory_index]]
    series_mean = [np.mean(hd, axis=0), np.mean(ba, axis=0), np.mean(tpa, axis=0)]

    for row in range(3):
        axes[row, 0].plot(ages, series_single[row], color=colors[row], linewidth=2.0)
        axes[row, 0].set_ylabel(labels[row])
        axes[row, 0].grid(True, linestyle="--", alpha=0.3)
        for event_age in disturbance_ages:
            axes[row, 0].axvline(event_age, color="#e74c3c", linestyle="--", alpha=0.5)

        axes[row, 1].plot(ages, series_mean[row], color=colors[row], linewidth=2.0)
        axes[row, 1].grid(True, linestyle="--", alpha=0.3)

    axes[-1, 0].set_xlabel("Age (years)")
    axes[-1, 1].set_xlabel("Age (years)")
    axes[0, 0].set_title("Single stochastic run")
    axes[0, 1].set_title("Mean of many runs")
    fig.suptitle("Stochastic PMRC growth trajectories", y=0.99)
    fig.tight_layout()
    return fig


def plot_disturbance_regime_comparison(
    results: dict[str, BatchResult | ScenarioResult],
    labels: dict[str, str] | None = None,
    show_sd: bool = False,
    figsize: tuple[float, float] = (9, 10),
) -> Figure:
    """Compare mean HD, BA, and TPA trajectories across disturbance regimes."""
    import matplotlib.pyplot as plt

    if not results:
        raise ValueError("At least one scenario result is required")

    ordered_names = list(results.keys())
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True, dpi=200)
    metric_specs = [
        ("hd", "Dominant height (ft)"),
        ("ba", "Basal area (ft²/ac)"),
        ("tpa", "Trees per acre"),
    ]
    palette = ["#27ae60", "#e67e22", "#c0392b", "#8e44ad", "#2c3e50"]

    for color, name in zip(palette * len(ordered_names), ordered_names):
        result = results[name]
        display_name = labels[name] if labels is not None and name in labels else name
        for ax, (variable, ylabel) in zip(axes, metric_specs):
            x, mean_values, std_values = _scenario_series(result, variable)
            ax.plot(x, mean_values, color=color, linewidth=2.0, label=display_name)
            if show_sd and np.any(std_values > 0):
                ax.fill_between(
                    x,
                    mean_values - std_values,
                    mean_values + std_values,
                    color=color,
                    alpha=0.15,
                )
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--", alpha=0.3)

    axes[-1].set_xlabel("Age (years)")
    axes[0].legend()
    fig.suptitle("Mean growth trajectories under disturbance regimes", y=0.99)
    fig.tight_layout()
    return fig


def save_figure(fig: Figure, path: str, dpi: int = 150) -> None:
    """Save figure to file.
    
    Args:
        fig: matplotlib Figure
        path: Output path (e.g., 'plots/histogram.png')
        dpi: Resolution
    """
    fig.savefig(path, dpi=dpi, bbox_inches='tight')


if __name__ == "__main__":
    print("Visualization module loaded.")
    print("Available functions:")
    print("  - plot_trajectory_fan()")
    print("  - plot_terminal_histogram()")
    print("  - plot_scenario_boxplots()")
    print("  - plot_product_breakdown()")
    print("  - plot_downside_risk()")
    print("  - plot_sensitivity_heatmap()")
    print("  - plot_deterministic_vs_stochastic_comparison()")
    print("  - plot_disturbance_frequency_histograms()")
    print("  - plot_growth_validation()")
    print("  - plot_hd_debug()")
    print("  - plot_deterministic_product_distribution()")
    print("  - plot_stochastic_product_distribution()")
    print("  - plot_stochastic_growth_demo()")
    print("  - plot_disturbance_regime_comparison()")
    print("  - save_figure()")
