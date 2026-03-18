# Economic Parameters

This document describes the economic parameters used for valuation.

## Stumpage Prices

| Product | Price | Unit |
|---------|-------|------|
| Pulpwood | $9.51 | $/ton |
| Chip-n-Saw | $23.51 | $/ton |
| Sawtimber | $27.82 | $/ton |

## Costs

| Cost Type | Value | Unit |
|-----------|-------|------|
| Logging | $150.00 | $/acre |
| Replanting | $150.80 | $/acre |
| Thinning | $87.34 | $/acre |

## Discount Rate

- **Annual real discount rate**: 5%

Used for NPV and LEV calculations.

## Volume Conversion

- **Cubic feet to tons**: 0.025 tons/cuft

## Usage

```python
from core.products import ProductPrices, HarvestCosts

prices = ProductPrices(
    pulpwood=9.51,
    chip_n_saw=23.51,
    sawtimber=27.82,
)

costs = HarvestCosts(
    logging=150.00,
    replanting=150.80,
)
```

## NPV Calculation

```
NPV = -C_0 + R_thin / (1+r)^t_thin + R_harvest / (1+r)^T
```

where:
- `C_0` = establishment cost (replanting)
- `R_thin` = thinning revenue
- `t_thin` = thinning year
- `R_harvest` = terminal harvest revenue
- `T` = rotation length
- `r` = discount rate

## LEV Calculation

Land Expectation Value (bare land value):

```
LEV = NPV × (1+r)^T / ((1+r)^T - 1)
```

## Source

Parameters based on:
- PMRC Technical Report 2001-1 (growth model coefficients)
- Regional stumpage price reports for the Upper Coastal Plain
- Standard silvicultural practices for loblolly pine plantations
