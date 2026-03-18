"""Simulation execution for forest scenarios.

This module runs simulations given a ScenarioConfig.
Separates execution logic from scenario definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from core.actions import ActionModel
from core.config import ScenarioConfig
from core.pmrc_model import PMRCModel
from core.products import (
    CUFT_TO_TON,
    HarvestCosts,
    ProductDistribution,
    ProductPrices,
    estimate_product_distribution,
)
from core.state import StandState, hd_from_si25_at_age
from core.stochastic_model import StochasticPMRC


@dataclass
class YearRecord:
    """State and events for a single year in the rotation."""
    year: int
    age: float
    hd: float
    tpa: float
    ba: float
    vol: float = 0.0
    
    # Optional event info
    disturbance: str | None = None
    disturbance_severity: float = 0.0
    disturbance_tpa_loss: float = 0.0
    disturbance_ba_loss: float = 0.0
    thinned: bool = False
    thin_ba_removed: float = 0.0
    thin_revenue: float = 0.0


@dataclass
class TerminalYield:
    """Terminal harvest yields and values."""
    # Product volumes (cuft/ac)
    vol_pulp: float
    vol_cns: float
    vol_saw: float
    vol_total: float
    
    # Product values ($/ac)
    value_pulp: float
    value_cns: float
    value_saw: float
    gross_revenue: float
    net_revenue: float  # After harvest costs
    
    # Product distribution object
    products: ProductDistribution


@dataclass
class ScenarioResult:
    """Complete result of a rotation scenario."""
    # Scenario metadata
    scenario_name: str
    scenario_type: Literal["deterministic", "stochastic"]
    rotation_length: int
    
    # Initial conditions
    initial_state: StandState
    
    # Full trajectory
    trajectory: list[YearRecord] = field(default_factory=list)
    
    # Thinning summary
    thin_occurred: bool = False
    thin_year: int | None = None
    thin_revenue: float = 0.0
    
    # Terminal yield
    terminal_yield: TerminalYield | None = None
    
    # Economic metrics
    npv: float = 0.0
    lev: float = 0.0

    # Disturbance summary
    disturbance_occurred: bool = False
    disturbance_years: list[int] = field(default_factory=list)
    disturbance_severities: list[float] = field(default_factory=list)
    disturbance_count: int = 0
    mean_disturbance_severity: float = 0.0
    max_disturbance_severity: float = 0.0
    
    # Parameters used
    discount_rate: float = 0.05
    prices: ProductPrices | None = None
    costs: HarvestCosts | None = None
    scenario_config: ScenarioConfig | None = None


def compute_npv(
    thin_revenue: float,
    thin_year: int | None,
    harvest_revenue: float,
    rotation_length: int,
    discount_rate: float = 0.05,
    establishment_cost: float = 150.80,
) -> float:
    """Compute Net Present Value for a single rotation.
    
    NPV = -C_0 + R_thin / (1+r)^t_thin + R_harvest / (1+r)^T
    """
    npv = -establishment_cost
    
    if thin_year is not None and thin_revenue > 0:
        npv += thin_revenue / ((1 + discount_rate) ** thin_year)
    
    npv += harvest_revenue / ((1 + discount_rate) ** rotation_length)
    
    return npv


def compute_lev(
    npv: float,
    rotation_length: int,
    discount_rate: float = 0.05,
) -> float:
    """Compute Land Expectation Value (bare land value).
    
    LEV = NPV * (1+r)^T / ((1+r)^T - 1)
    """
    r = discount_rate
    T = rotation_length
    factor = ((1 + r) ** T) / (((1 + r) ** T) - 1)
    return npv * factor


def _create_initial_state(config: ScenarioConfig, pmrc: PMRCModel) -> StandState:
    """Create initial stand state from config."""
    initial_hd = hd_from_si25_at_age(config.si25, config.age0)
    initial_ba = pmrc.ba_predict(
        age=config.age0,
        tpa=config.tpa0,
        hd=initial_hd,
        region=config.region,
    )
    return StandState.from_si25(
        age=config.age0,
        si25=config.si25,
        tpa=config.tpa0,
        ba=initial_ba,
        region=config.region,
    )


def _compute_terminal_yield(
    state: StandState,
    pmrc: PMRCModel,
    prices: ProductPrices,
    costs: HarvestCosts,
) -> TerminalYield:
    """Compute terminal harvest yield and values."""
    products = estimate_product_distribution(
        pmrc=pmrc,
        age=state.age,
        ba=state.ba,
        tpa=state.tpa,
        hd=state.hd,
        region=state.region,
        phwd=state.phwd,
    )
    
    tons_pulp = products.vol_pulp * CUFT_TO_TON
    tons_cns = products.vol_cns * CUFT_TO_TON
    tons_saw = products.vol_saw * CUFT_TO_TON
    
    value_pulp = tons_pulp * prices.pulpwood
    value_cns = tons_cns * prices.chip_n_saw
    value_saw = tons_saw * prices.sawtimber
    gross_revenue = value_pulp + value_cns + value_saw
    net_revenue = gross_revenue - costs.total
    
    return TerminalYield(
        vol_pulp=products.vol_pulp,
        vol_cns=products.vol_cns,
        vol_saw=products.vol_saw,
        vol_total=products.total_vol,
        value_pulp=value_pulp,
        value_cns=value_cns,
        value_saw=value_saw,
        gross_revenue=gross_revenue,
        net_revenue=net_revenue,
        products=products,
    )


def _compute_stand_volume(state: StandState, pmrc: PMRCModel) -> float:
    """Compute stand volume from atomic state variables."""
    return pmrc.yield_predict(
        age=state.age,
        tpa=state.tpa,
        hd=state.hd,
        ba=state.ba,
        unit="TVOB",
        region=state.region,
    )


def run_scenario(
    config: ScenarioConfig,
    rng: np.random.Generator | None = None,
) -> ScenarioResult:
    """Execute a scenario and return results.
    
    Args:
        config: Scenario configuration
        rng: Random generator (only used for stochastic scenarios)
    
    Returns:
        ScenarioResult with full trajectory and economic metrics
    """
    pmrc = PMRCModel(region=config.region)
    prices = config.prices or ProductPrices()
    costs = config.costs or HarvestCosts()
    
    # Create initial state
    initial_state = _create_initial_state(config, pmrc)
    
    if config.scenario_type == "deterministic":
        return _run_deterministic(config, pmrc, initial_state, prices, costs)
    else:
        if rng is None:
            rng = np.random.default_rng(config.seed)
        return _run_stochastic(config, pmrc, initial_state, prices, costs, rng)


def _run_deterministic(
    config: ScenarioConfig,
    pmrc: PMRCModel,
    initial_state: StandState,
    prices: ProductPrices,
    costs: HarvestCosts,
) -> ScenarioResult:
    """Run deterministic scenario."""
    action_model = ActionModel(pmrc, thin_params=config.thin_params)
    
    trajectory: list[YearRecord] = []
    state = initial_state
    
    # Record initial state
    trajectory.append(YearRecord(
        year=0,
        age=state.age,
        hd=state.hd,
        tpa=state.tpa,
        ba=state.ba,
        vol=_compute_stand_volume(state, pmrc),
    ))
    
    thin_occurred = False
    thin_year: int | None = None
    total_thin_revenue = 0.0
    
    for year in range(1, config.rotation_length + 1):
        # Deterministic PMRC projection
        age2 = state.age + 1.0
        hd2 = pmrc.hd_project(state.age, state.hd, age2)
        tpa2 = pmrc.tpa_project(state.tpa, state.si25, state.age, age2)
        ba2 = pmrc.ba_project(
            state.age, state.tpa, tpa2, state.ba, state.hd, hd2, age2, state.region
        )
        
        state = StandState(
            age=age2,
            hd=hd2,
            tpa=tpa2,
            ba=ba2,
            si25=state.si25,
            region=state.region,
            phwd=state.phwd,
        )
        
        # Check for thinning
        thinned = False
        thin_ba = 0.0
        thin_rev = 0.0
        
        if action_model.should_thin(state):
            result = action_model.apply_thinning(state, prices)
            if result.occurred and result.post_thin_state is not None:
                state = result.post_thin_state
                thinned = True
                thin_occurred = True
                thin_year = year
                thin_ba = result.ba_removed
                thin_rev = result.net_revenue
                total_thin_revenue += thin_rev
        
        trajectory.append(YearRecord(
            year=year,
            age=state.age,
            hd=state.hd,
            tpa=state.tpa,
            ba=state.ba,
            vol=_compute_stand_volume(state, pmrc),
            thinned=thinned,
            thin_ba_removed=thin_ba,
            thin_revenue=thin_rev,
        ))
    
    # Compute terminal yield
    terminal_yield = _compute_terminal_yield(state, pmrc, prices, costs)
    
    # Compute NPV and LEV
    npv = compute_npv(
        thin_revenue=total_thin_revenue,
        thin_year=thin_year,
        harvest_revenue=terminal_yield.net_revenue,
        rotation_length=config.rotation_length,
        discount_rate=config.discount_rate,
        establishment_cost=costs.replanting,
    )
    
    lev = compute_lev(npv, config.rotation_length, config.discount_rate)
    
    return ScenarioResult(
        scenario_name=config.name,
        scenario_type="deterministic",
        rotation_length=config.rotation_length,
        initial_state=initial_state,
        trajectory=trajectory,
        thin_occurred=thin_occurred,
        thin_year=thin_year,
        thin_revenue=total_thin_revenue,
        terminal_yield=terminal_yield,
        npv=npv,
        lev=lev,
        discount_rate=config.discount_rate,
        prices=prices,
        costs=costs,
        scenario_config=config,
    )


def _run_stochastic(
    config: ScenarioConfig,
    pmrc: PMRCModel,
    initial_state: StandState,
    prices: ProductPrices,
    costs: HarvestCosts,
    rng: np.random.Generator,
) -> ScenarioResult:
    """Run stochastic scenario (single trajectory)."""
    stochastic = StochasticPMRC(
        pmrc,
        noise_params=config.noise_params,
        disturbance_params=config.disturbance_params,
        thin_params=config.thin_params,
    )
    
    trajectory: list[YearRecord] = []
    state = initial_state
    
    # Record initial state
    trajectory.append(YearRecord(
        year=0,
        age=state.age,
        hd=state.hd,
        tpa=state.tpa,
        ba=state.ba,
        vol=_compute_stand_volume(state, pmrc),
    ))
    
    thin_occurred = False
    thin_year: int | None = None
    total_thin_revenue = 0.0
    disturbance_years: list[int] = []
    disturbance_severities: list[float] = []
    
    for year in range(1, config.rotation_length + 1):
        state, trace = stochastic.sample_next_state(state, dt=1.0, rng=rng)
        
        thinned = trace.action_type == "thin"
        if thinned:
            thin_occurred = True
            thin_year = year
            total_thin_revenue += trace.thin_revenue
        if trace.disturbance_label:
            disturbance_years.append(year)
            disturbance_severities.append(trace.disturbance_severity)
        
        trajectory.append(YearRecord(
            year=year,
            age=state.age,
            hd=state.hd,
            tpa=state.tpa,
            ba=state.ba,
            vol=_compute_stand_volume(state, pmrc),
            disturbance=trace.disturbance_label,
            disturbance_severity=trace.disturbance_severity,
            disturbance_tpa_loss=trace.disturbance_tpa_loss,
            disturbance_ba_loss=trace.disturbance_ba_loss,
            thinned=thinned,
            thin_ba_removed=trace.thin_ba_removed,
            thin_revenue=trace.thin_revenue,
        ))
    
    # Compute terminal yield
    terminal_yield = _compute_terminal_yield(state, pmrc, prices, costs)
    
    # Compute NPV and LEV
    npv = compute_npv(
        thin_revenue=total_thin_revenue,
        thin_year=thin_year,
        harvest_revenue=terminal_yield.net_revenue,
        rotation_length=config.rotation_length,
        discount_rate=config.discount_rate,
        establishment_cost=costs.replanting,
    )
    
    lev = compute_lev(npv, config.rotation_length, config.discount_rate)
    disturbance_count = len(disturbance_years)
    mean_disturbance_severity = (
        float(np.mean(disturbance_severities)) if disturbance_severities else 0.0
    )
    max_disturbance_severity = (
        float(np.max(disturbance_severities)) if disturbance_severities else 0.0
    )
    
    return ScenarioResult(
        scenario_name=config.name,
        scenario_type="stochastic",
        rotation_length=config.rotation_length,
        initial_state=initial_state,
        trajectory=trajectory,
        thin_occurred=thin_occurred,
        thin_year=thin_year,
        thin_revenue=total_thin_revenue,
        terminal_yield=terminal_yield,
        npv=npv,
        lev=lev,
        disturbance_occurred=disturbance_count > 0,
        disturbance_years=disturbance_years,
        disturbance_severities=disturbance_severities,
        disturbance_count=disturbance_count,
        mean_disturbance_severity=mean_disturbance_severity,
        max_disturbance_severity=max_disturbance_severity,
        discount_rate=config.discount_rate,
        prices=prices,
        costs=costs,
        scenario_config=config,
    )


@dataclass
class BatchResult:
    """Results from Monte Carlo batch simulation.
    
    Attributes:
        scenario_name: Name of the scenario
        n_trajectories: Number of Monte Carlo runs
        terminal_values: Array of terminal harvest values ($/ac)
        npvs: Array of NPV values ($/ac)
        levs: Array of LEV values ($/ac)
        thin_revenues: Array of thinning revenues ($/ac)
        vol_pulp/vol_cns/vol_saw/vol_total: Product volume arrays at rotation
        disturbance_occurred: Whether each trajectory had >= 1 disturbance
        disturbance_counts: Number of disturbance years per trajectory
        mean_disturbance_severities: Mean severity per trajectory
        max_disturbance_severities: Max severity per trajectory
        disturbance_years: Disturbance years for each trajectory
        disturbance_severity_paths: Severity draws for each trajectory
        scenario_config: Configuration used to generate the batch
        trajectories: Optional list of full trajectory results
    """
    scenario_name: str
    n_trajectories: int
    terminal_values: np.ndarray
    npvs: np.ndarray
    levs: np.ndarray
    thin_revenues: np.ndarray
    vol_pulp: np.ndarray
    vol_cns: np.ndarray
    vol_saw: np.ndarray
    vol_total: np.ndarray
    disturbance_occurred: np.ndarray
    disturbance_counts: np.ndarray
    mean_disturbance_severities: np.ndarray
    max_disturbance_severities: np.ndarray
    disturbance_years: list[list[int]]
    disturbance_severity_paths: list[list[float]]
    scenario_config: ScenarioConfig
    trajectories: list[ScenarioResult] | None = None


def run_batch(
    config: ScenarioConfig,
    n_trajectories: int = 1000,
    seed: int | None = None,
    store_trajectories: bool = False,
    show_progress: bool = False,
) -> BatchResult:
    """Run Monte Carlo batch simulation for a stochastic scenario.
    
    Args:
        config: Scenario configuration (should be stochastic type)
        n_trajectories: Number of Monte Carlo runs
        seed: Random seed for reproducibility
        store_trajectories: If True, store full trajectory for each run
        show_progress: If True, print progress every 100 runs
    
    Returns:
        BatchResult with arrays of terminal values, NPVs, LEVs
    
    Raises:
        ValueError: If config is deterministic (use run_scenario instead)
    """
    if config.scenario_type == "deterministic":
        raise ValueError(
            "run_batch is for stochastic scenarios. "
            "Use run_scenario for deterministic scenarios."
        )
    
    rng = np.random.default_rng(seed)
    
    terminal_values = np.zeros(n_trajectories)
    npvs = np.zeros(n_trajectories)
    levs = np.zeros(n_trajectories)
    thin_revenues = np.zeros(n_trajectories)
    vol_pulp = np.zeros(n_trajectories)
    vol_cns = np.zeros(n_trajectories)
    vol_saw = np.zeros(n_trajectories)
    vol_total = np.zeros(n_trajectories)
    disturbance_occurred = np.zeros(n_trajectories, dtype=bool)
    disturbance_counts = np.zeros(n_trajectories, dtype=int)
    mean_disturbance_severities = np.zeros(n_trajectories)
    max_disturbance_severities = np.zeros(n_trajectories)
    disturbance_years: list[list[int]] = []
    disturbance_severity_paths: list[list[float]] = []
    trajectories: list[ScenarioResult] | None = [] if store_trajectories else None
    
    for i in range(n_trajectories):
        result = run_scenario(config, rng=rng)
        
        terminal_values[i] = result.terminal_yield.net_revenue if result.terminal_yield else 0.0
        npvs[i] = result.npv
        levs[i] = result.lev
        thin_revenues[i] = result.thin_revenue
        vol_pulp[i] = (
            result.terminal_yield.vol_pulp if result.terminal_yield is not None else 0.0
        )
        vol_cns[i] = (
            result.terminal_yield.vol_cns if result.terminal_yield is not None else 0.0
        )
        vol_saw[i] = (
            result.terminal_yield.vol_saw if result.terminal_yield is not None else 0.0
        )
        vol_total[i] = (
            result.terminal_yield.vol_total if result.terminal_yield is not None else 0.0
        )
        disturbance_occurred[i] = result.disturbance_occurred
        disturbance_counts[i] = result.disturbance_count
        mean_disturbance_severities[i] = result.mean_disturbance_severity
        max_disturbance_severities[i] = result.max_disturbance_severity
        disturbance_years.append(list(result.disturbance_years))
        disturbance_severity_paths.append(list(result.disturbance_severities))
        
        if trajectories is not None:
            trajectories.append(result)
        
        if show_progress and (i + 1) % 100 == 0:
            print(f"  Completed {i + 1}/{n_trajectories} trajectories")
    
    return BatchResult(
        scenario_name=config.name,
        n_trajectories=n_trajectories,
        terminal_values=terminal_values,
        npvs=npvs,
        levs=levs,
        thin_revenues=thin_revenues,
        vol_pulp=vol_pulp,
        vol_cns=vol_cns,
        vol_saw=vol_saw,
        vol_total=vol_total,
        disturbance_occurred=disturbance_occurred,
        disturbance_counts=disturbance_counts,
        mean_disturbance_severities=mean_disturbance_severities,
        max_disturbance_severities=max_disturbance_severities,
        disturbance_years=disturbance_years,
        disturbance_severity_paths=disturbance_severity_paths,
        scenario_config=config,
        trajectories=trajectories,
    )


def run_batch_scenarios(
    scenarios: list[ScenarioConfig],
    n_trajectories: int = 1000,
    seed: int | None = None,
    store_trajectories: bool = False,
    show_progress: bool = True,
) -> dict[str, BatchResult | ScenarioResult]:
    """Run batch simulation for multiple scenarios.
    
    For stochastic scenarios, runs Monte Carlo batch.
    For deterministic scenarios, runs single trajectory.
    
    Args:
        scenarios: List of scenario configurations
        n_trajectories: Number of Monte Carlo runs per stochastic scenario
        seed: Base random seed (incremented per scenario)
        store_trajectories: If True, retain full trajectories for stochastic runs
        show_progress: If True, print progress
    
    Returns:
        Dict mapping scenario name to BatchResult or ScenarioResult
    """
    results: dict[str, BatchResult | ScenarioResult] = {}
    
    for i, config in enumerate(scenarios):
        if show_progress:
            print(f"Running scenario {i+1}/{len(scenarios)}: {config.name}")
        
        if config.scenario_type == "deterministic":
            results[config.name] = run_scenario(config)
        else:
            scenario_seed = seed + i if seed is not None else None
            results[config.name] = run_batch(
                config,
                n_trajectories=n_trajectories,
                seed=scenario_seed,
                store_trajectories=store_trajectories,
                show_progress=show_progress,
            )
    
    return results


def print_scenario_summary(result: ScenarioResult) -> None:
    """Print a formatted summary of scenario results."""
    print(f"\n{'='*60}")
    print(f"Scenario: {result.scenario_name} ({result.scenario_type})")
    print("=" * 60)
    
    print(f"\nRotation: {result.rotation_length} years")
    print(f"Initial: age={result.initial_state.age}, SI25={result.initial_state.si25}")
    
    final = result.trajectory[-1]
    print(f"\nFinal State (Year {final.year}):")
    print(f"  Age: {final.age:.1f} years")
    print(f"  HD:  {final.hd:.1f} ft")
    print(f"  TPA: {final.tpa:.0f}")
    print(f"  BA:  {final.ba:.1f} ft²/ac")
    
    if result.thin_occurred:
        print(f"\nThinning:")
        print(f"  Year: {result.thin_year}")
        print(f"  Revenue: ${result.thin_revenue:.2f}/ac")
    else:
        print("\nThinning: None")
    
    if result.terminal_yield:
        ty = result.terminal_yield
        print("\nTerminal Yield:")
        print("  Volume (cuft/ac):")
        print(f"    Pulpwood:   {ty.vol_pulp:.1f}")
        print(f"    Chip-n-Saw: {ty.vol_cns:.1f}")
        print(f"    Sawtimber:  {ty.vol_saw:.1f}")
        print(f"    Total:      {ty.vol_total:.1f}")
        print("  Value ($/ac):")
        print(f"    Pulpwood:   ${ty.value_pulp:.2f}")
        print(f"    Chip-n-Saw: ${ty.value_cns:.2f}")
        print(f"    Sawtimber:  ${ty.value_saw:.2f}")
        print(f"    Gross:      ${ty.gross_revenue:.2f}")
        print(f"    Net:        ${ty.net_revenue:.2f}")
    
    print(f"\nEconomic Metrics (r={result.discount_rate:.1%}):")
    print(f"  NPV: ${result.npv:.2f}/ac")
    print(f"  LEV: ${result.lev:.2f}/ac")
