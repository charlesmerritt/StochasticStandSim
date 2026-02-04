# Stochastic MDP Thesis Project Planning Document

## Problem restatement
You need a clear, data-bounded stochastic simulator with interpretable transitions from the deterministic PMRC model, plus a management layer that supports discrete rules (e.g., thin when volume/age thresholds are met). Then you want to estimate transition matrices at multiple risk profiles and solve the resulting MDPs to compare management regimes. This document captures what exists in the repo, the blockers, and a plan to close the gaps.

## Current repo context (what exists today)

### Deterministic growth core
- `core/pmrc_model.py` implements the deterministic PMRC equations (height, TPA, BA, CI, TVOB, etc.).
- `core/growth.py` wraps the deterministic PMRC model into a `Stand` class with scheduled thinning/fertilization and an optional stochastic path that delegates to `StochasticPMRC` when `StandConfig.use_stochastic_growth=True`.

### Stochastic simulator & discretization
- `core/stochastic_stand.py` provides `StochasticPMRC` with lognormal noise on BA/HD and binomial/normal noise on TPA. It also includes a thin “action” that simply scales BA/TPA and a `StateDiscretizer` plus Monte Carlo transition estimation.

### Disturbances
- `core/disturbances.py` defines “fire/wind” event factories and simple “chronic/catastrophic” generators, but this system is separate from `StochasticPMRC`. In `growth.py`, disturbances are applied as scheduled `DisturbanceEvent`s in the deterministic path, while `StochasticPMRC` applies its own disturbance logic internally.

### Management actions
- `core/actions.py` defines an `ActionManager` with thinning/harvest/plant/salvage/fert/rxfire plus simple economic accounting. It uses `Stand._apply_thin_event` which assumes constant QMD (proportional removal across diameter classes), but there is no diameter distribution or tree list logic.

### RL environment and training
- `core/env.py` implements a Gym environment with a discrete action menu and uses `ActionManager` + `Stand`. It has optional disturbance logic with Poisson timing for chronic/catastrophic events.
- `rl/train.py` implements SB3 training, but no explicit MDP solver for discretized transition matrices.

### Unimplemented placeholders / stubs
- `core/config.py`, `core/baselines.py`, `core/evaluation.py`, `core/rollout.py` are stubs.
- Tests reference disturbance configs in `data/disturbances/` that are not present.

### Data available
- `data/econ_params.yaml` contains prices, costs, and discount rate but is not wired into any config loader.
- `data/scenarios/pmrc_plots/` contains Excel files (e.g., baseline/study data) that likely hold real bounds for stochastic noise, but nothing in code loads them yet.

## Blockers and why they matter

### 1) Noise is not Gaussian and not bounded by data
- `StochasticPMRC` uses lognormal noise and binomial/normal TPA noise with fixed sigmas. There is no parameterization tied to empirical bounds, so simulated noise is not anchored to real data.

### 2) Stochastic transition path is not interpretable relative to deterministic PMRC
- There are *two* disturbance systems: one internal to `StochasticPMRC` and one via `DisturbanceEvent` in `growth.py`. This makes it hard to describe the delta from deterministic to stochastic and to ensure interpretability.

### 3) Management actions are not stable or interpretable across simulators
- The MDP transition estimator in `stochastic_stand.py` applies actions by scaling BA/TPA, which is not tied to the management actions in `core/actions.py` or the thinning logic in `core/growth.py`.
- The environment uses hard-coded action definitions and small wrappers around `ActionManager`, but this doesn’t align with MDP transition matrix creation.

### 4) No diameter distribution / tree list control
- There is a Weibull-based size class helper in `stochastic_stand.py`, but it is not used for thinning. The current thinning assumes proportional removal (constant QMD), which conflicts with your requirement to remove smallest trees first to reach target volume.

### 5) MDP transition matrices exist but are not tied to a risk profile or management spec
- The existing `estimate_transition_matrix` does MC simulation with a fixed action multiplier and no explicit risk profile config (disturbance levels, noise bounds, etc.).

### 6) No MDP solver or evaluation pipeline for the discretized matrices
- The RL environment exists, but there is no solver for discrete transition matrices (value iteration, policy iteration) and no evaluation loop for comparing policies across risk profiles.

### 7) Config and baseline evaluation are missing
- A central config facade is a stub, as are baselines/evaluation/rollout. Without config-backed inputs, the system can’t be described or reproduced easily.

## Plan of attack (smallest-change, interpretable path)

### Workstream A — Data + configuration scaffolding (minimal viable config)
1. **Inventory data sources** for noise bounds and disturbance calibration (start with `data/scenarios/pmrc_plots/*.xls(x)` and any real stand data you have locally).
2. **Implement `core/config.py` minimally** as a loader for:
   - economic parameters from `data/econ_params.yaml` (prices/costs/discount),
   - stochastic noise parameters (Gaussian mean/sigma bounds per variable),
   - disturbance parameters (chronic/catastrophic rates and severity bounds),
   - MDP discretization bins and action spec.
3. **Add a small schema validation** (pydantic or simple dataclasses) only if needed to keep changes minimal.

Deliverable: a single config entry point that lets you define risk profiles and noise bounds without touching code.

### Workstream B — Unify deterministic → stochastic transition logic
1. **Decide on a single disturbance pipeline**:
   - Either: make `StochasticPMRC` accept `DisturbanceEvent` generators and apply them the same way `Stand` does; or
   - Route stochastic transitions through `Stand` and call the deterministic step + explicit stochastic noise injection.
2. **Replace ad-hoc lognormal noise** with Gaussian noise (bounded or truncated) per variable. Use empirical bounds from your data to set mean/std and clipping rules (e.g., clip to [p5, p95] or to absolute min/max). Keep the deterministic mean as the Gaussian mean to preserve interpretability.
3. **Expose deterministic vs stochastic deltas** in a debug trace (e.g., store `delta_ba`, `delta_tpa`, `delta_hd`, `disturbance_label`) so the stochasticity is explainable in the thesis.

Deliverable: a clear, single transition function with traceable additive noise and disturbance impacts.

### Workstream C — Management actions aligned with state transitions
1. **Define a minimal `ActionSpec`** that works for both the environment and transition matrices (e.g., no-op, thin-to-residual-BA, harvest/plant). Keep the interface small and stable.
2. **Align action application**:
   - In `stochastic_stand.py`, replace the simplistic `apply_action_to_state` with a call path that uses the same logic as `Stand._apply_thin_event` (or a shared helper).
   - Ensure a single source of truth for “what thinning does” (so the transition matrix and RL env agree).
3. **Add a rule-based management policy** (baseline) that implements “if volume >= X at age A then thin Y” to support deterministic baselines and interpretability.

Deliverable: management actions that are consistent across simulator, MDP, and RL.

### Workstream D — Diameter distribution / tree list control
1. **Use `SizeClassDistribution`** (Weibull-based) to build a class-level tree list from current BA/TPA.
2. **Implement smallest-tree-first thinning**:
   - Remove trees from smallest diameter classes until the target BA or volume removal is met.
   - Recompute BA/TPA after removal and map back to the stand state.
3. **Document the approximation** (Weibull parameters, class bounds) so the MDP transition remains interpretable and repeatable.

Deliverable: thinning operations that match your stated management rule (remove smallest trees first).

### Workstream E — Risk profiles + transition matrices
1. **Define explicit risk profiles** in config (e.g., low/med/high disturbance rates + noise std bounds).
2. **Generate transition matrices per profile** using the unified stochastic transition function and consistent action spec.
3. **Validate transition rows** (row sums ~1, inspect stability) and archive matrices for repeatable experiments.

Deliverable: a small set of transition matrices keyed by risk profile.

### Workstream F — MDP solvers + evaluation
1. **Implement value iteration and policy iteration** for the discrete MDP (small, focused functions).
2. **Produce policy summaries** for each risk profile (e.g., policy table, action histograms, implied rotation ages).
3. **Compare to baselines** using a simple evaluation module (mean return, variance, disturbance losses).

Deliverable: solved MDPs and interpretable comparisons across risk levels.

### Workstream G — Tests + reproducibility
1. **Add small unit tests** for:
   - noise clipping rules,
   - action application consistency,
   - thinning with size class removal,
   - transition matrix row normalization.
2. **Backfill missing data fixtures** or skip tests cleanly where data is unavailable (as some tests already do).

Deliverable: minimal tests that prevent regression in the new stochastic pipeline.

## Immediate next steps (recommended order)
1. Implement the minimal config loader (`core/config.py`) and define 2–3 risk profiles using data-derived noise bounds.
2. Unify stochastic transitions and replace the lognormal noise with bounded Gaussian noise.
3. Implement size-class thinning and connect it to both environment actions and transition matrices.
4. Generate transition matrices for each risk profile and implement value iteration to solve them.
5. Write a short evaluation script to compare policies across risk profiles.
6. Implement the Streamlit UI for improved debugging and interactivity.

## Open questions to resolve early
- Where exactly is the “real data” for noise bounds (CSV/Excel paths, and what variables/columns)?
- Do you want Gaussian noise on *increments* (delta) or on *levels* (absolute BA/TPA/HD)?
- Should catastrophic disturbance reset the stand (as in `StochasticPMRC`) or use the event loss fractions (as in `DisturbanceEvent`)?
- What is the minimal action set for your thesis (thin/noop/harvest/plant only, or include fert/salvage)?

---
