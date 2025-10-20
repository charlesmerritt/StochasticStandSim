"""
All shared dataclasses and typed Protocols. Nothing else.

StandState: stand age, N, BA, H, Dq, volume, carbon, inventory flags, last_disturbance, rng_state_id.

Action: enum or typed literal. Example: {"noop","thin_light","thin_heavy","fertilize","burn_prescribed","harvest"} plus continuous knobs if needed.

Observation: alias of StandState or partial view.

Reward: scalar plus RewardComponents breakdown.

GrowthParams: PMRC coefficients and site region tags.

EconParams: prices, costs, discount rate, salvage fractions.

ADSRParams: attack, decay, sustain, release parameters per disturbance type and per state feature if needed.

JumpKernel: severity distribution spec per disturbance, parameterized by conditioning variables.

MDPConfig: horizon, gamma, step_years, action_space spec, stochastic flags.

DisturbanceEvent: type, severity, year, latent duration.

Trajectory: list of (s,a,r,s',info).
"""