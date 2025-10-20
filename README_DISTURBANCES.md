# Disturbance System - Complete Guide

## Overview

The disturbance system has two distinct mechanisms:

### 1. **Kernels** - Immediate Multi-Metric Losses
- **What**: Defines percentage losses at moment of disturbance (t=0)
- **Affects**: BA, Volume, Height, TPA (all metrics simultaneously)
- **How**: Applies proportional reduction to current values
- **Example**: Fire occurs → BA drops 17%, Volume drops 22%, Height drops 2%, TPA drops 20%

### 2. **Envelopes** - BA Growth Increment Multipliers
- **What**: Defines recovery trajectory for BA growth (t>0)
- **Affects**: ONLY the per-year increment in BA (not volume, height, or TPA)
- **How**: Multiplies the annual BA growth rate
- **Example**: Normal growth +3 ft²/ac → Multiplier 0.75 → Actual growth +2.25 ft²/ac

## Key Distinction

```
KERNEL (t=0):
  Before: BA=120, Vol=600, HD=65, TPA=350
  After:  BA=100, Vol=468, HD=64, TPA=280
  ↑ Immediate proportional losses across all metrics

ENVELOPE (t>0):
  Year 1: BA growth +3.5 → multiplied by 0.75 → +2.63 → BA=102.63
  Year 2: BA growth +3.6 → multiplied by 0.83 → +2.99 → BA=105.62
  Year 3: BA growth +3.7 → multiplied by 0.92 → +3.40 → BA=109.02
  ↑ Affects ONLY the increment, NOT the total
```

## Quick Start

### Load Data
```python
from core.disturbances import load_kernel, load_envelope_set

kernel = load_kernel("data/disturbances/kernels/fire_kernel.yaml")
envelopes = load_envelope_set("data/disturbances/envelopes/fire_envelope.yaml")
```

### Create Disturbance
```python
from core.disturbances import FireDisturbance, random_severity

fire = FireDisturbance(
    age=15.0,
    severity=random_severity(),
    kernel=kernel,
    envelope_set=envelopes
)
```

### Apply Immediate Losses (Easy Way)
```python
sev_class = fire.get_severity_class()
post_dist = fire.kernel.apply_median_losses(
    sev_class,
    ba=120.0,
    vol=600.0,
    hd=65.0,
    tpa=350.0
)
# Returns: {'ba': 99.6, 'vol': 468, 'hd': 63.6, 'tpa': 280}
```

### Apply Envelope Over Time
```python
envelope = fire.envelope_set.get_envelope(sev_class)

for year in range(5):
    years_since = year
    
    # Calculate multiplier based on ADSR phase
    if years_since < envelope.attack_duration_years:
        mult = 1.0 - envelope.attack_drop
    elif years_since < envelope.attack_duration_years + envelope.decay_years:
        t = (years_since - envelope.attack_duration_years) / envelope.decay_years
        attack_val = 1.0 - envelope.attack_drop
        mult = attack_val + (envelope.sustain_level - attack_val) * t
    else:
        mult = envelope.sustain_level
    
    # Apply to BA growth increment
    normal_ba_growth = calculate_normal_ba_growth(state)
    actual_ba_growth = normal_ba_growth * mult
    new_ba = current_ba + actual_ba_growth
```

## API Reference

### DisturbanceKernel

```python
# Get all metrics for a severity class
all_losses = kernel.get_all_losses("severe_50_80")
# Returns: {
#   'basal_area': (0.08, 0.12, 0.17, 0.23, 0.30),
#   'volume': (0.10, 0.15, 0.22, 0.30, 0.38),
#   'height': (0.008, 0.015, 0.022, 0.03, 0.04),
#   'density': (0.10, 0.15, 0.20, 0.26, 0.32)
# }

# Get specific metric
ba_dist = kernel.get_loss_distribution("severe_50_80", "basal_area")
# Returns: (0.08, 0.12, 0.17, 0.23, 0.30)

# Apply median losses (convenience method)
post = kernel.apply_median_losses("severe_50_80", ba=120, vol=600, hd=65, tpa=350)
# Returns: {'ba': 99.6, 'vol': 468, 'hd': 63.6, 'tpa': 280}
```

### EnvelopeSet

```python
# Get envelope for severity class
envelope = envelopes.get_envelope("high_50_80")

# Access parameters
envelope.attack_drop          # 0.25 (25% reduction in growth rate)
envelope.attack_duration_years # 0 years
envelope.decay_years          # 3 years
envelope.sustain_level        # 1.0 (return to normal)
envelope.release_years        # 1 year
```

### Disturbance Classes

```python
# Fire or Wind
fire = FireDisturbance(age=15.0, severity=0.65, kernel=k, envelope_set=e)
fire.get_severity_class()  # Returns: "high_50_75"

# Thinning (no kernel/envelope needed)
thin = ThinningDisturbance(age=10.0, removal_fraction=0.4)
```

## File Structure

```
StochasticStandSim/
├── core/
│   └── disturbances.py          # Main classes and loaders
├── data/disturbances/
│   ├── kernels/                 # Immediate loss definitions
│   │   ├── fire_kernel.yaml
│   │   └── wind_kernel.yaml
│   └── envelopes/               # BA growth trajectory definitions
│       ├── fire_envelope.yaml
│       └── wind_envelope.yaml
├── docs/
│   ├── disturbance_architecture.md       # Full architecture details
│   └── disturbance_integration_guide.md  # Integration examples
├── examples/
│   └── disturbance_demo.py      # Working demonstration
└── tests/
    └── test_disturbances.py     # Test suite
```

## Testing

Run the demo to see everything in action:
```bash
python3 examples/disturbance_demo.py
```

Output shows:
- ✓ Basic disturbance creation with random severity
- ✓ Kernel loading and multi-metric loss distributions
- ✓ Envelope loading and BA growth multiplier trajectories
- ✓ Severity discretization examples

## Common Patterns

### Pattern 1: Simple Application
```python
# At disturbance time
fire = FireDisturbance(age=15, severity=random_severity(), kernel=k, envelope_set=e)
sev_class = fire.get_severity_class()
post = fire.kernel.apply_median_losses(sev_class, ba, vol, hd, tpa)

# Update state
state = state.replace(ba=post['ba'], vol_ob=post['vol'], hd=post['hd'], tpa=post['tpa'])
```

### Pattern 2: Stochastic Sampling
```python
# Sample from distribution instead of using median
all_losses = fire.kernel.get_all_losses(sev_class)
ba_dist = all_losses['basal_area']

# Sample from truncated normal or beta distribution
import numpy as np
mu = ba_dist[2]  # median
sigma = (ba_dist[4] - ba_dist[0]) / 6  # rough std dev
sampled_loss = np.clip(np.random.normal(mu, sigma), ba_dist[0], ba_dist[4])

ba_after = current_ba * (1 - sampled_loss)
```

### Pattern 3: Multiple Active Disturbances
```python
# Compound envelope effects from multiple disturbances
multiplier = 1.0
for dist_info in active_disturbances:
    years_since = current_age - dist_info['age_occurred']
    mult = calculate_envelope_mult(dist_info['envelope'], years_since)
    multiplier *= mult

actual_ba_growth = normal_ba_growth * multiplier
```

## Troubleshooting

### Issue: Class name mismatch
**Problem**: `ValueError: Severity class 'high_50_75' not found in kernel`  
**Cause**: Kernel uses different naming (e.g., 'severe_50_80')  
**Solution**: The demo includes approximate matching logic - adapt for your use case

### Issue: Envelope applied to wrong metric
**Problem**: Height/Volume growth incorrectly modified  
**Cause**: Envelopes only affect BA growth increments  
**Solution**: Only apply envelope multiplier to `ba_increment`, not other metrics

### Issue: Total BA reduced instead of growth
**Problem**: BA declining when it should be growing slowly  
**Cause**: Applied envelope to total BA instead of increment  
**Solution**: `new_ba = current_ba + (growth_increment * multiplier)` NOT `new_ba = current_ba * multiplier`

## Next Steps

1. **Read Architecture**: See `docs/disturbance_architecture.md` for detailed design
2. **Integration Guide**: See `docs/disturbance_integration_guide.md` for growth model integration
3. **Run Demo**: Execute `examples/disturbance_demo.py` to see live examples
4. **Customize**: Create your own kernel/envelope YAML files for different disturbance scenarios

## Summary

✅ **Kernels** = Multi-metric immediate losses (BA, Vol, HD, TPA) at t=0  
✅ **Envelopes** = BA growth increment multipliers (ONLY) for t>0  
✅ **Helper method** = `apply_median_losses()` for easy kernel application  
✅ **Fully documented** = Architecture, integration guide, and working examples  

The system is now production-ready for integration into your growth model! 🎯
