# Scenario Defaults

This document defines the pre-configured scenarios for sensitivity analysis.

## Scenario Matrix (4×4 = 16 scenarios)

The full scenario matrix varies two parameters:
- **λ_proc** ∈ {0, 0.25, 0.5, 1.0} — process noise scaling
- **p_dist** ∈ {0, 1/30, 1/20, 1/10} — disturbance probability

Fixed severity parameters:
- **m_q** = 0.30 (moderate severity mean)
- **κ** = 12 (Beta concentration)

| λ_proc | p_dist=0 | p_dist=1/30 | p_dist=1/20 | p_dist=1/10 |
|--------|----------|-------------|-------------|-------------|
| 0 | DETERMINISTIC | DIST_30 | DIST_20 | DIST_10 |
| 0.25 | NOISE_025 | N025_D30 | N025_D20 | N025_D10 |
| 0.5 | NOISE_050 | N050_D30 | N050_D20 | N050_D10 |
| 1.0 | NOISE_100 | N100_D30 | N100_D20 | N100_D10 |

## Initial Stand Conditions

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| `age0` | 5.0 | years | Initial stand age |
| `tpa0` | 850 | trees/acre | Initial planting density |
| `si25` | 80 | ft | Site index at base age 25 |
| `region` | "ucp" | - | Upper Coastal Plain (PMRC coefficient region) |

## Planning Horizon

| Parameter | Value | Unit |
|-----------|-------|------|
| `rotation_length` | 35 | years |
| `dt` | 1.0 | years |

## Thinning Policy (BAT)

| Parameter | Value | Unit |
|-----------|-------|------|
| `trigger_age` | 15 | years |
| `ba_threshold` | 150 | ft²/ac |
| `residual_ba` | 100 | ft²/ac |

## Usage

### Run Single Scenario

```python
from core.scenarios import DETERMINISTIC, NOISE_050, N050_D20
from core.simulate import run_scenario, run_batch

# Deterministic
result = run_scenario(DETERMINISTIC)

# Stochastic (single trajectory)
result = run_scenario(NOISE_050, rng=np.random.default_rng(42))

# Stochastic (Monte Carlo batch)
batch = run_batch(N050_D20, n_trajectories=1000, seed=42)
```

### Run All Scenarios

```python
from core.scenarios import ALL_SCENARIOS
from core.simulate import run_batch_scenarios

results = run_batch_scenarios(ALL_SCENARIOS, n_trajectories=1000, seed=42)
```

### Generate Custom Matrix

```python
from core.scenarios import generate_scenario_matrix

# Custom parameter grid
scenarios = generate_scenario_matrix(
    lambda_levels=[0.0, 0.5, 1.0],
    p_dist_levels=[0.0, 0.05, 0.10],
)
```

## Convenience Subsets

```python
from core.scenarios import NOISE_ONLY_SCENARIOS, DIST_ONLY_SCENARIOS

# Scenarios with process noise only (p_dist=0)
NOISE_ONLY_SCENARIOS = [DETERMINISTIC, NOISE_025, NOISE_050, NOISE_100]

# Scenarios with disturbance only (λ=0)
DIST_ONLY_SCENARIOS = [DETERMINISTIC, DIST_30, DIST_20, DIST_10]
```

## Related Documentation

- [Disturbance Model](disturbance_model.md) — Bernoulli + Beta severity model
- [Process Noise](process_noise.md) — Lognormal increment noise + recruitment
- [Economic Parameters](economic_parameters.md) — Prices, costs, discount rate
