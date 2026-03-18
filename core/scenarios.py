"""Pre-defined scenario configurations.

This module defines the full 4×4 scenario matrix for sensitivity analysis:
- λ_proc ∈ {0, 0.25, 0.5, 1.0} (process noise scaling)
- p_dist ∈ {0, 1/30, 1/20, 1/10} (disturbance frequency)

All scenarios use fixed severity parameters:
- m_q = 0.30 (moderate severity mean)
- κ = 12 (Beta concentration)

Management policies:
- ALL_SCENARIOS: No thinning (default)
- ALL_SCENARIOS_BAT: Basal Area Threshold thinning at age 15

Use with simulate.run_scenario() or simulate.run_batch() to execute.
"""

from __future__ import annotations

from core.config import ScenarioConfig, ThinningParams
from core.disturbances import DisturbanceParams
from core.process_noise import NoiseParams


# =============================================================================
# SCENARIO MATRIX PARAMETERS
# =============================================================================

# Process noise levels (λ_proc)
LAMBDA_LEVELS = [0.0, 0.25, 0.5, 1.0]

# Disturbance frequencies (p_dist = 1/n, where n is mean return interval)
P_DIST_LEVELS = [0.0, 1/30, 1/20, 1/10]

# Fixed severity parameters
SEVERITY_MEAN = 0.30  # m_q, fixed "moderate" severity regime.
SEVERITY_KAPPA = 12.0  # κ

# Default BAT thinning parameters
DEFAULT_THIN_PARAMS = ThinningParams(
    trigger_age=15.0,
    ba_threshold=150.0,
    residual_ba=100.0,
    thin_cost=87.34,
)


# =============================================================================
# INDIVIDUAL SCENARIOS (4×4 = 16 total)
# =============================================================================

# Row 1: λ_proc = 0 (deterministic growth)
DETERMINISTIC = ScenarioConfig(
    name="deterministic",
    scenario_type="deterministic",
)
"""Deterministic baseline: λ=0, p_dist=0"""

DIST_30 = ScenarioConfig(
    name="dist_30",
    scenario_type="stochastic",
    noise_params=NoiseParams(lambda_proc=0.0),
    disturbance_params=DisturbanceParams(p_dist=1/30, severity_mean=SEVERITY_MEAN, severity_kappa=SEVERITY_KAPPA),
)
"""Disturbance only (30-year return): λ=0, p_dist=1/30"""

DIST_20 = ScenarioConfig(
    name="dist_20",
    scenario_type="stochastic",
    noise_params=NoiseParams(lambda_proc=0.0),
    disturbance_params=DisturbanceParams(p_dist=1/20, severity_mean=SEVERITY_MEAN, severity_kappa=SEVERITY_KAPPA),
)
"""Disturbance only (20-year return): λ=0, p_dist=1/20"""

DIST_10 = ScenarioConfig(
    name="dist_10",
    scenario_type="stochastic",
    noise_params=NoiseParams(lambda_proc=0.0),
    disturbance_params=DisturbanceParams(p_dist=1/10, severity_mean=SEVERITY_MEAN, severity_kappa=SEVERITY_KAPPA),
)
"""Disturbance only (10-year return): λ=0, p_dist=1/10"""

# Row 2: λ_proc = 0.25 (low noise)
NOISE_025 = ScenarioConfig(
    name="noise_025",
    scenario_type="stochastic",
    noise_params=NoiseParams(lambda_proc=0.25),
    disturbance_params=DisturbanceParams(p_dist=0.0),
)
"""Low noise only: λ=0.25, p_dist=0"""

N025_D30 = ScenarioConfig(
    name="n025_d30",
    scenario_type="stochastic",
    noise_params=NoiseParams(lambda_proc=0.25),
    disturbance_params=DisturbanceParams(p_dist=1/30, severity_mean=SEVERITY_MEAN, severity_kappa=SEVERITY_KAPPA),
)
"""Low noise + 30-year disturbance: λ=0.25, p_dist=1/30"""

N025_D20 = ScenarioConfig(
    name="n025_d20",
    scenario_type="stochastic",
    noise_params=NoiseParams(lambda_proc=0.25),
    disturbance_params=DisturbanceParams(p_dist=1/20, severity_mean=SEVERITY_MEAN, severity_kappa=SEVERITY_KAPPA),
)
"""Low noise + 20-year disturbance: λ=0.25, p_dist=1/20"""

N025_D10 = ScenarioConfig(
    name="n025_d10",
    scenario_type="stochastic",
    noise_params=NoiseParams(lambda_proc=0.25),
    disturbance_params=DisturbanceParams(p_dist=1/10, severity_mean=SEVERITY_MEAN, severity_kappa=SEVERITY_KAPPA),
)
"""Low noise + 10-year disturbance: λ=0.25, p_dist=1/10"""

# Row 3: λ_proc = 0.5 (medium noise)
NOISE_050 = ScenarioConfig(
    name="noise_050",
    scenario_type="stochastic",
    noise_params=NoiseParams(lambda_proc=0.5),
    disturbance_params=DisturbanceParams(p_dist=0.0),
)
"""Medium noise only: λ=0.5, p_dist=0"""

N050_D30 = ScenarioConfig(
    name="n050_d30",
    scenario_type="stochastic",
    noise_params=NoiseParams(lambda_proc=0.5),
    disturbance_params=DisturbanceParams(p_dist=1/30, severity_mean=SEVERITY_MEAN, severity_kappa=SEVERITY_KAPPA),
)
"""Medium noise + 30-year disturbance: λ=0.5, p_dist=1/30"""

N050_D20 = ScenarioConfig(
    name="n050_d20",
    scenario_type="stochastic",
    noise_params=NoiseParams(lambda_proc=0.5),
    disturbance_params=DisturbanceParams(p_dist=1/20, severity_mean=SEVERITY_MEAN, severity_kappa=SEVERITY_KAPPA),
)
"""Medium noise + 20-year disturbance: λ=0.5, p_dist=1/20"""

N050_D10 = ScenarioConfig(
    name="n050_d10",
    scenario_type="stochastic",
    noise_params=NoiseParams(lambda_proc=0.5),
    disturbance_params=DisturbanceParams(p_dist=1/10, severity_mean=SEVERITY_MEAN, severity_kappa=SEVERITY_KAPPA),
)
"""Medium noise + 10-year disturbance: λ=0.5, p_dist=1/10"""

# Row 4: λ_proc = 1.0 (high noise)
NOISE_100 = ScenarioConfig(
    name="noise_100",
    scenario_type="stochastic",
    noise_params=NoiseParams(lambda_proc=1.0),
    disturbance_params=DisturbanceParams(p_dist=0.0),
)
"""High noise only: λ=1.0, p_dist=0"""

N100_D30 = ScenarioConfig(
    name="n100_d30",
    scenario_type="stochastic",
    noise_params=NoiseParams(lambda_proc=1.0),
    disturbance_params=DisturbanceParams(p_dist=1/30, severity_mean=SEVERITY_MEAN, severity_kappa=SEVERITY_KAPPA),
)
"""High noise + 30-year disturbance: λ=1.0, p_dist=1/30"""

N100_D20 = ScenarioConfig(
    name="n100_d20",
    scenario_type="stochastic",
    noise_params=NoiseParams(lambda_proc=1.0),
    disturbance_params=DisturbanceParams(p_dist=1/20, severity_mean=SEVERITY_MEAN, severity_kappa=SEVERITY_KAPPA),
)
"""High noise + 20-year disturbance: λ=1.0, p_dist=1/20"""

N100_D10 = ScenarioConfig(
    name="n100_d10",
    scenario_type="stochastic",
    noise_params=NoiseParams(lambda_proc=1.0),
    disturbance_params=DisturbanceParams(p_dist=1/10, severity_mean=SEVERITY_MEAN, severity_kappa=SEVERITY_KAPPA),
)
"""High noise + 10-year disturbance: λ=1.0, p_dist=1/10"""


# =============================================================================
# SCENARIO COLLECTIONS
# =============================================================================

ALL_SCENARIOS = [
    # Row 1: λ=0
    DETERMINISTIC, DIST_30, DIST_20, DIST_10,
    # Row 2: λ=0.25
    NOISE_025, N025_D30, N025_D20, N025_D10,
    # Row 3: λ=0.5
    NOISE_050, N050_D30, N050_D20, N050_D10,
    # Row 4: λ=1.0
    NOISE_100, N100_D30, N100_D20, N100_D10,
]
"""Full 4×4 scenario matrix (16 scenarios)."""

# Convenience subsets (no thinning)
NOISE_ONLY_SCENARIOS = [DETERMINISTIC, NOISE_025, NOISE_050, NOISE_100]
"""Scenarios with process noise only (p_dist=0), no thinning."""

DIST_ONLY_SCENARIOS = [DETERMINISTIC, DIST_30, DIST_20, DIST_10]
"""Scenarios with disturbance only (λ=0), no thinning."""


# =============================================================================
# BAT (BASAL AREA THRESHOLD) THINNING SCENARIOS
# =============================================================================

def _with_bat(config: ScenarioConfig) -> ScenarioConfig:
    """Create a copy of a scenario with BAT thinning enabled."""
    return ScenarioConfig(
        name=config.name,
        scenario_type=config.scenario_type,
        age0=config.age0,
        tpa0=config.tpa0,
        si25=config.si25,
        region=config.region,
        rotation_length=config.rotation_length,
        thin_params=DEFAULT_THIN_PARAMS,
        noise_params=config.noise_params,
        disturbance_params=config.disturbance_params,
        discount_rate=config.discount_rate,
        prices=config.prices,
        costs=config.costs,
        n_trajectories=config.n_trajectories,
        seed=config.seed,
    )


ALL_SCENARIOS_BAT = [_with_bat(s) for s in ALL_SCENARIOS]
"""Full 4×4 scenario matrix with BAT thinning (16 scenarios)."""

NOISE_ONLY_SCENARIOS_BAT = [_with_bat(s) for s in NOISE_ONLY_SCENARIOS]
"""Scenarios with process noise only (p_dist=0), with BAT thinning."""

DIST_ONLY_SCENARIOS_BAT = [_with_bat(s) for s in DIST_ONLY_SCENARIOS]
"""Scenarios with disturbance only (λ=0), with BAT thinning."""


def generate_scenario_matrix(
    lambda_levels: list[float] | None = None,
    p_dist_levels: list[float] | None = None,
    severity_mean: float = SEVERITY_MEAN,
    severity_kappa: float = SEVERITY_KAPPA,
) -> list[ScenarioConfig]:
    """Generate a custom scenario matrix.
    
    Args:
        lambda_levels: Process noise scaling factors (default: [0, 0.25, 0.5, 1.0])
        p_dist_levels: Disturbance probabilities (default: [0, 1/30, 1/20, 1/10])
        severity_mean: Beta severity mean (default: 0.30)
        severity_kappa: Beta concentration (default: 12)
    
    Returns:
        List of ScenarioConfig objects for all combinations
    """
    if lambda_levels is None:
        lambda_levels = LAMBDA_LEVELS
    if p_dist_levels is None:
        p_dist_levels = P_DIST_LEVELS
    
    scenarios = []
    for lam in lambda_levels:
        for p_dist in p_dist_levels:
            # Determine scenario type
            is_deterministic = lam == 0 and p_dist == 0
            scenario_type = "deterministic" if is_deterministic else "stochastic"
            
            # Generate name
            lam_str = f"n{int(lam*100):03d}" if lam > 0 else "det"
            dist_str = f"d{int(1/p_dist):02d}" if p_dist > 0 else ""
            name = f"{lam_str}_{dist_str}".rstrip("_")
            
            config = ScenarioConfig(
                name=name,
                scenario_type=scenario_type,
                noise_params=NoiseParams(lambda_proc=lam) if not is_deterministic else None,
                disturbance_params=DisturbanceParams(
                    p_dist=p_dist,
                    severity_mean=severity_mean,
                    severity_kappa=severity_kappa,
                ) if p_dist > 0 else None,
            )
            scenarios.append(config)
    
    return scenarios


if __name__ == "__main__":
    from core.simulate import run_scenario, print_scenario_summary
    
    # Run deterministic baseline
    result = run_scenario(DETERMINISTIC)
    print_scenario_summary(result)
    
    # Print scenario matrix
    print(f"\n{'='*60}")
    print("Scenario Matrix (16 scenarios):")
    print("=" * 60)
    print(f"{'Name':<15} {'λ_proc':>8} {'p_dist':>10} {'Type':<15}")
    print("-" * 60)
    for s in ALL_SCENARIOS:
        lam = s.noise_params.lambda_proc if s.noise_params else 0.0
        p_dist = s.disturbance_params.p_dist if s.disturbance_params else 0.0
        print(f"{s.name:<15} {lam:>8.2f} {p_dist:>10.4f} {s.scenario_type:<15}")
