# Disturbance Model

This document describes the disturbance model implementation per PLANNING.md Sections 4.3-4.5.

## Overview

Disturbances are modeled as a single generic event type with:
- **Bernoulli occurrence** with configurable frequency
- **Beta severity** drawn conditional on occurrence
- **Proportional shocks** to atomic state variables

## Occurrence Model

Each year, disturbance occurrence is a Bernoulli trial:

```
z_t ~ Bernoulli(p_dist)
```

where `p_dist = 1/n` and `n` is the mean return interval in years.

| Return Interval | p_dist |
|-----------------|--------|
| 10 years | 0.10 |
| 20 years | 0.05 |
| 30 years | 0.033 |
| None | 0.00 |

## Severity Model

Conditional on occurrence (`z_t = 1`), severity is drawn from a Beta distribution:

```
q_t | z_t=1 ~ Beta(α, β)
```

where:
- `α = m_q × κ`
- `β = (1 - m_q) × κ`
- `m_q` = mean severity (default 0.30)
- `κ` = concentration parameter (default 12)

Higher `κ` means less variability around the mean.

## Shock Application

Severity `q_t ∈ [0, 1]` is applied as a proportional reduction to atomic state variables:

```
x_post = x_pre × (1 - c_x × q_t)
```

where `c_x` is the sensitivity coefficient for variable `x`:

| Variable | Default c_x | Notes |
|----------|-------------|-------|
| TPA | 1.0 | Full sensitivity |
| BA | 1.0 | Full sensitivity |
| HD | 0.0 | Height typically unaffected |

## Parameters

```python
from core.disturbances import DisturbanceParams

params = DisturbanceParams(
    p_dist=0.05,           # 20-year return interval
    severity_mean=0.30,    # m_q
    severity_kappa=12.0,   # κ
    c_tpa=1.0,
    c_ba=1.0,
    c_hd=0.0,
)
```

## Usage

```python
from core.disturbances import DisturbanceModel, DisturbanceParams
import numpy as np

model = DisturbanceModel(DisturbanceParams(p_dist=0.05))
rng = np.random.default_rng(42)

# Sample and apply in one step
new_state, event = model.sample_and_apply(current_state, rng)

if event.occurred:
    print(f"Disturbance! Severity: {event.severity:.2f}")
    print(f"TPA loss: {event.tpa_loss:.0f}")
```

## Key Design Decisions

1. **Single disturbance type**: No distinction between mild/severe events. Severity is a continuous draw.

2. **Independence**: Disturbance occurrence is independent across years (no memory).

3. **Instantaneous effect**: Shocks are applied immediately, no delayed recovery.

4. **Atomic variables only**: Only TPA, BA, HD are shocked. Derived quantities (volume, products) are recomputed from post-shock atomics.

5. **Age unchanged**: Disturbances do not reset stand age.
