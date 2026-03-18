# Active Assumptions

One-line statements of all modeling assumptions currently in effect.

## Time and Horizon

- Time step is 1 year (annual transitions).
- Rotation length is fixed at 35 years.
- Initial stand age is 5 years.

## State Variables

- State consists of 4 dynamic atomics (age, hd, tpa, ba) and 3 constants (si25, region, phwd).
- Derived quantities (volume, products, qmd) are computed on-demand, not stored.
- HD and SI25 are linked via Chapman-Richards curve; only one should be specified at initialization.

## Deterministic Backbone

- PMRC (Harrison & Borders 1996) provides the deterministic growth trend.
- Projection order: age → hd → tpa → ba (dependencies respected).
- Stochastic model reduces exactly to PMRC when λ_proc=0 and p_dist=0.

## Process Noise

- Multiplicative lognormal noise on BA and HD increments.
- Mean-corrected lognormal: E[multiplier] = 1.0.
- TPA mortality uses binomial model by default.
- Recruitment is Poisson-distributed, disabled when λ_proc=0.
- λ_proc scales all noise sources (0=off, 1=full).

## Disturbances

- Single generic disturbance type.
- Occurrence is annual Bernoulli with probability p_dist.
- Severity is Beta-distributed with mean m_q=0.30 and concentration κ=12.
- Shocks are proportional reductions to atomic variables: x_post = x_pre × (1 - c_x × q).
- Disturbance occurrence is independent across years (no memory).
- Disturbance effects are instantaneous (no delayed recovery).
- Age is unchanged by disturbance.

## Feasibility Constraints

- TPA ≥ 100 (PMRC lower bound).
- BA ≥ 0.
- HD is non-decreasing (height doesn't shrink).
- Age > 0.

## Management Policy

- Fixed heuristic BAT thinning policy (not optimized).
- Thinning at age 15 if BA > 150 ft²/ac, thin to 100 ft²/ac.
- Thinning from below (smallest trees removed first).

## Economics

- Stumpage prices are fixed (pulp: $9.51, CNS: $23.51, saw: $27.82 per ton).
- Costs are fixed (logging: $150, replanting: $150.80, thinning: $87.34 per acre).
- Discount rate is 5% annual real rate.
- Terminal value is product-based (sum of price × yield by class).

## Epistemic Treatment

- Stochastic kernel parameters are scenario-based, not empirically estimated.
- Uncertainty is explored via structured parameter scenarios, not posterior inference.
- m_q is fixed at 0.30 across all scenarios; only λ_proc and p_dist vary.
