# Process Noise Model

This document describes the process noise model implementation per PLANNING.md Section 4.2.

## Overview

Process noise captures year-to-year growth variability. It is applied to **growth increments** of atomic state variables, not to state levels directly.

## Multiplicative Lognormal Noise

For positive growth increments (HD, BA), noise is applied multiplicatively:

```
Δx_stoch = Δx_PMRC × exp(λ × σ_x × ε - 0.5 × (λ × σ_x)²)
```

where:
- `Δx_PMRC` = deterministic PMRC increment
- `λ` = `lambda_proc`, global noise scaling factor
- `σ_x` = variable-specific standard deviation
- `ε ~ N(0, 1)` = standard normal draw

The term `-0.5 × (λ × σ_x)²` is a **mean correction** ensuring `E[multiplier] = 1.0`.

## Parameters

```python
from core.process_noise import NoiseParams

params = NoiseParams(
    sigma_log_ba=0.14,      # BA increment noise
    sigma_log_hd=None,      # HD increment noise (None = no noise)
    sigma_tpa=30.0,         # TPA noise (if not using binomial)
    use_binomial_tpa=True,  # Use binomial mortality model
    lambda_proc=1.0,        # Global scaling (0=off, 1=full)
)
```

## TPA Mortality

TPA naturally decreases (mortality). Two options:

### Binomial Mortality (default)
Each tree has probability `p = expected_mortality / TPA` of dying:
```
actual_deaths ~ Binomial(TPA, p)
```

### Normal Noise (alternative)
Additive normal noise on TPA change:
```
TPA_noise ~ N(0, σ_tpa × λ)
```

## Recruitment (Extension)

New trees are added to the smallest diameter class via Poisson sampling:

```
R_t ~ Poisson(λ_R)
```

where:
```
λ_R = max(0, α₀ + α₁ × BA + α₂ × SI25)
```

Default coefficients: α₀ = 1.0, α₁ = -0.005, α₂ = 0.02.

**Important**: Recruitment is disabled when `lambda_proc = 0` to ensure zero-noise recovery.

## Usage

```python
from core.process_noise import ProcessNoiseModel, NoiseParams
import numpy as np

model = ProcessNoiseModel(NoiseParams(lambda_proc=0.5))
rng = np.random.default_rng(42)

# Apply noise to increments
noisy_delta_ba, noisy_delta_hd, tpa_adj, realization = model.apply_to_increments(
    delta_ba=5.0,
    delta_hd=2.0,
    tpa=500.0,
    expected_tpa_loss=10.0,
    rng=rng,
)
```

## Key Design Decisions

1. **Increment-based noise**: Noise is applied to growth increments, not state levels.

2. **Mean-corrected lognormal**: Ensures expected increment equals deterministic PMRC.

3. **λ_proc scaling**: Single parameter to scale all noise sources (0 = deterministic).

4. **Atomic variables only**: Only HD, BA, TPA receive noise. Volume/products are derived.

5. **Zero-noise recovery**: When λ_proc = 0, stochastic model exactly reproduces PMRC.
