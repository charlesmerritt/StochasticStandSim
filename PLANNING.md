Below is the most defensible Phase 1 version of your approach, keeping it simple enough to implement with no new data, while still being mathematically coherent and thesis-worthy.

---

# 1. Core modeling decision

Your Phase 1 target is:

[
p(V_T \mid s_0,\pi,\lambda_{\text{proc}},\lambda_{\text{dist}})
]

where:

* (V_T) is terminal stand value in dollars, and optionally NPV or LEV derived from that value
* (s_0) is a fixed initial stand state
* (\pi) is a fixed heuristic management policy
* (\lambda_{\text{proc}}) controls transition noise around PMRC growth
* (\lambda_{\text{dist}}) controls disturbance intensity or frequency

This is **not** yet Bayesian inference. It is a forward stochastic simulator whose uncertainty is driven by fixed parameters chosen by calibration, literature-informed judgment, and coherence checks against the deterministic PMRC baseline.

That is the correct Phase 1 framing.

---

# 2. The simplest defensible stochastic simulator

You already landed on the right factorization:

[
P(s_{t+1}\mid s_t,a_t)
======================

P(z_t\mid s_t,a_t),
P(q_t\mid z_t,s_t,a_t),
P(s_{t+1}\mid s_t,a_t,z_t,q_t)
]

with:

* (z_t \in {0,1}): disturbance occurrence
* (q_t \in [0,1]): disturbance severity
* (s_t): stand state
* (a_t): management action

For Phase 1, keep it even simpler:

* annual time step
* no delayed recovery memory
* independent yearly disturbance occurrence
* one generic disturbance type
* deterministic PMRC update as baseline trend
* multiplicative process noise on growth increments
* one-time disturbance shock applied to atomic state variables
* hard biological projection step to maintain feasibility

This is enough.

---

# 3. Recommended Phase 1 state definition

Use only the minimum variables required by your PMRC simulator. Do not store derived quantities.

## 3.1 Atomic State Variables (Minimal Sufficient Set)

From the PMRC equations, the **minimal atomic state** is:

| Variable | Type | Role |
|----------|------|------|
| `age` | Dynamic | Advances with time, required for all projections |
| `hd` | Dynamic | Dominant height, projects independently via Chapman-Richards |
| `tpa` | Dynamic | Trees per acre, projects using `si25` and `age` |
| `ba` | Dynamic | Basal area, projects using `age`, `tpa`, `hd` changes |
| `si25` | Constant | Site index at base age 25, used in TPA projection, must align with HD definition. |
| `region` | Constant | Coefficient selection (ucp/pucp/lcp) |
| `phwd` | Constant | Percent hardwood (optional, for Weibull diameter distribution) |

This gives 4 dynamic variables and 3 constants.

## 3.2 Derived Variables (Computed On-Demand)

These are **NOT stored** in state. They are computed from atomics when needed:

| Variable | Derivation |
|----------|------------|
| `qmd` | `sqrt((ba / tpa) / 0.005454154)` — quadratic mean diameter |
| `vol` | `pmrc.tvob(age, tpa, hd, ba, region)` — total volume outside bark |
| `prod` | `pmrc.diameter_class_distribution(ba, tpa, region, phwd)` — product breakdown |

Storing derived quantities risks desync when atomics are updated stochastically.

## 3.3 PMRC Projection Order

The PMRC projection sequence must follow this order:

```
age2 = age1 + dt
hd2  = hd_project(age1, hd1, age2)           # independent
tpa2 = tpa_project(tpa1, si25, age1, age2)   # uses si25, not hd
ba2  = ba_project(age1, tpa1, tpa2, ba1, hd1, hd2, age2, region)  # depends on hd and tpa changes
```

This means **stochastic noise should be applied to increments of `hd`, `tpa`, and `ba`**, not to derived quantities like volume or products.

## 3.4 Python StandState

```python
@dataclass
class StandState:
    age: float      # Dynamic - advances with time
    hd: float       # Dynamic - projects via hd_project()
    tpa: float      # Dynamic - projects via tpa_project()
    ba: float       # Dynamic - projects via ba_project()
    si25: float     # Constant - site quality, must sync with initial HD
    region: Region  # Constant - coefficient selection
    phwd: float = 0.0  # Constant - percent hardwood (optional)
```

Do not expand state unless absolutely necessary.

---

# 4. Minimal formal stochastic transition model

## 4.1 Deterministic PMRC backbone

Let the deterministic annual update be:

[
\mu_{t+1} = f(s_t,a_t)
]

where (f) is the PMRC simulator step.

This gives the mean trajectory.

---

## 4.2 Process noise on growth increments

Do **not** add noise directly to state levels first. Add noise to growth increments or survival changes.

That is more biologically defensible.

Suppose PMRC implies for a positive variable (x) an annual increment:

[
\Delta x_t^{\text{PMRC}} = x_{t+1}^{\text{det}} - x_t
]

Then define stochastic increment:

[
\Delta x_t^{\text{stoch}} = \Delta x_t^{\text{PMRC}} \cdot \exp(\lambda_{\text{proc}}\sigma_x \epsilon_{x,t})
\quad\text{with}\quad
\epsilon_{x,t}\sim \mathcal N(0,1)
]

Then:

[
x_{t+1}^{\text{pre-dist}} = x_t + \Delta x_t^{\text{stoch}}
]

This works well for positive growth increments.

For variables that may decrease naturally, you have two options:

### Option A, recommended for Phase 1

Apply noise only to the **atomic state variables** that have nonnegative annual increments:

* height increment (`hd`)
* basal area increment (`ba`)

TPA naturally decreases (mortality), so handle it separately (see below).

Volume and products are **derived** from atomics and should be recomputed after stochastic updates, not perturbed directly.

---

## 4.3 Disturbance occurrence

Let yearly disturbance occurrence be Bernoulli:

[
z_t \sim \text{Bernoulli}(p_{\text{dist}})
]

where:

[
p_{\text{dist}} = \frac{1}{n}
]

and (n \in {10,20,30}) represents expected disturbance return interval.

This is exactly your "once every (n) years on average" interpretation.

For Phase 1, keep this independent of state and action:

[
P(z_t=1\mid s_t,a_t)=p_{\text{dist}}
]

Later you can make it state-dependent.

---

## 4.4 Disturbance severity conditional on occurrence

Conditional on (z_t=1), draw severity:

[
q_t \sim \text{Beta}(\alpha,\beta)
]

where (q_t \in [0,1]) is interpreted as proportional damage intensity.

Why Beta?

* bounded on ([0,1])
* flexible
* easy to tune by mean and concentration

Parameterize it by mean severity (m_q) and concentration (\kappa):

[
\alpha = m_q \kappa,\qquad \beta=(1-m_q)\kappa
]

Then:

* higher (\kappa) means less variability around the mean
* lower (\kappa) means more variable severity

A sensible Phase 1 choice is to define several severity regimes:

* mild: (m_q = 0.15)
* moderate: (m_q = 0.30)
* severe: (m_q = 0.50)

and fix (\kappa) around 10 or 20.

This is simple and defensible.

---

## 4.5 One-time disturbance shock

Apply disturbance as an instantaneous proportional reduction to **atomic state variables only**.

For the atomic variables:

[
\text{TPA}*{t+1} = \text{TPA}*{t+1}^{\text{pre-dist}}(1-c_{\text{TPA}} q_t)
]

[
\text{BA}*{t+1} = \text{BA}*{t+1}^{\text{pre-dist}}(1-c_{\text{BA}} q_t)
]

[
\text{HD}*{t+1} = \text{HD}*{t+1}^{\text{pre-dist}}(1-c_{\text{HD}} q_t)
]

where (c_{\cdot}\in(0,1]) are variable-specific sensitivity coefficients.

For Phase 1, start with:
* (c_{\text{TPA}} = 1) — full mortality effect
* (c_{\text{BA}} = 1) — full basal area loss (consistent with TPA loss)
* (c_{\text{HD}} = 0) or small — height typically unaffected unless severe wind damage

This makes severity easy to interpret:

* (q=0.2) means roughly a 20% damage shock
* (q=0.5) means roughly a 50% damage shock

**Important:** Do NOT shock derived quantities (volume, products). Recompute them from the post-disturbance atomics.

---

## 4.6 Feasibility projection

After stochastic growth and disturbance, enforce feasibility:

[
s_{t+1}=g(\tilde s_{t+1})
]

where (\tilde s_{t+1}) is the unprojected stochastic update.

The projection (g) should enforce constraints on **atomic state variables**:

* (\text{TPA}_{t+1} \ge 100) — PMRC lower bound
* (\text{BA}_{t+1} \ge 0)
* (\text{HD}_{t+1} \ge \text{HD}_t) — height nondecreasing
* any other PMRC domain constraints

Derived quantities (volume, products) are recomputed from projected atomics and do not need separate projection.

For thesis language:

> A projection operator was applied after each stochastic transition to map the provisional post-update state back into the biologically and model-feasible region required by the PMRC system.

That sounds clean and accurate.

---

# 5. What fixed parameters do you need in Phase 1

You asked the crucial question: if there is no data, what parameters are fixed?

The answer is: Phase 1 uses **scenario parameters**, not estimated parameters.

These should be clearly presented as user-controlled or analyst-controlled scenario settings.

## 5.1 Process noise parameters

These control aleatoric variation around deterministic PMRC growth.

For each noisy incremented variable (x), define:

* (\sigma_x): baseline log-scale volatility
* (\lambda_{\text{proc}}): global process-noise multiplier

So effective noise is:

[
\sigma_x^{\text{eff}} = \lambda_{\text{proc}}\sigma_x
]

To keep things simple, use either:

### Version 1, simplest

One shared volatility:
[
\sigma_x = \sigma_{\text{proc}} \quad \text{for all } x
]

### Version 2, still simple

Separate volatilities for the **atomic variables**:

* basal area increment (`sigma_ba`)
* height increment (`sigma_hd`)

Do NOT add noise to derived quantities like volume. Volume is recomputed from atomics.

I recommend Version 2 only if your implementation already distinguishes these naturally.

Suggested Phase 1 levels:

* low: (\lambda_{\text{proc}}=0.25)
* medium: (\lambda_{\text{proc}}=0.5)
* high: (\lambda_{\text{proc}}=1.0)

The base (\sigma_x) values are chosen so that the deterministic trend still dominates.

A practical starting point:

* annual increment coefficient of variation around 5% to 15%

Since you're using lognormal multiplicative noise, choose (\sigma_x) so the spread is noticeable but not destabilizing.

## 5.2 Disturbance frequency parameters

Choose:

* low disturbance: (n=30), so (p_{\text{dist}}=1/30)
* medium disturbance: (n=20), so (p_{\text{dist}}=1/20)
* high disturbance: (n=10), so (p_{\text{dist}}=1/10)

This is interpretable and easy to explain.

## 5.3 Disturbance severity parameters

Choose a Beta severity family, for example:

* mild: mean (0.15)
* moderate: mean (0.30)
* severe: mean (0.50)

with concentration (\kappa = 12) or (20).

If you want one global disturbance-intensity slider, let it control the severity mean separately from frequency.

## 5.4 Sensitivity coefficients (for disturbance shocks)

Set for **atomic variables only**:

* (c_{\text{TPA}} = 1) — full mortality effect
* (c_{\text{BA}} = 1) — full basal area loss
* (c_{\text{HD}} = 0) — height typically unaffected (or small value for severe wind)

Do NOT define sensitivity coefficients for derived quantities (volume, products). They are recomputed from post-disturbance atomics.

---

# 6. Aleatoric vs epistemic in Phase 1

Because you have no external data, you should distinguish them like this:

## Aleatoric uncertainty

Represented explicitly by:

* yearly growth innovations
* yearly disturbance occurrence
* severity draws

This is inside the simulator.

## Epistemic uncertainty

In Phase 1, not estimated from data, but treated as **scenario uncertainty over kernel parameters**.

That means you vary:

* (\sigma_x)
* disturbance return interval (n)
* severity mean (m_q)
* severity concentration (\kappa)

and see how outputs change.

For writing:

> In the absence of empirical calibration data, epistemic uncertainty was represented through structured parameter scenarios rather than posterior inference. This allowed sensitivity analysis over plausible stochastic kernels while preserving the PMRC deterministic baseline as the central growth tendency.

That is the correct way to say it.

---

# 7. Objective formalization

You said the primary objective is terminal timber value, with optional NPV and LEV.

## 7.1 Terminal timber value

Define:

[
V_T = \sum_{k=1}^{K} p_k , y_{k,T}
]

where:

* (k) indexes product classes
* (y_{k,T}) is terminal yield in product class (k)
* (p_k) is stumpage or product price

This is your main outcome.

## 7.2 NPV

If there are intermediate costs or thinning revenues:

[
\text{NPV} = \sum_{t=0}^{T} \frac{CF_t}{(1+r)^t}
]

where (CF_t) includes:

* planting cost
* management cost
* thinning revenue
* terminal harvest revenue

If you are not modeling intermediate cashflows yet, NPV reduces to discounted terminal value minus initial costs.

## 7.3 LEV

For fixed rotation (T):

[
\text{LEV} = \frac{\text{NPV}(1+r)^T}{(1+r)^T - 1}
]

assuming identical infinite rotations and standard Faustmann framing.

You do not need to optimize rotation in Phase 1. Just report LEV for the fixed rotation.

---

# 8. Most sensible Phase 1 plan of attack

## Step 1. Freeze the deterministic baseline

You need a deterministic simulator function:

```python
s_next = pmrc_step(s_t, action_t)
```

and a deterministic policy:

```python
action_t = policy(s_t, t)
```

Run this baseline from the fixed initial state and fixed rotation length. Save:

* full state trajectory
* terminal product yields
* terminal dollar value
* NPV
* LEV

This becomes your reference path.

## Step 2. Decide which state updates receive process noise

Apply noise only to **atomic state variables**:

* basal area increment (`ba`)
* height increment (`hd`)

Do NOT add noise to derived quantities (volume, products). They are recomputed from atomics after each step.

TPA decreases via mortality; handle separately if needed.

## Step 3. Implement disturbance module

For each year:

1. draw occurrence (z_t)
2. if event occurs, draw severity (q_t)
3. apply proportional damage shock
4. project back into feasible state space

## Step 4. Implement feasibility projection

Build a single function that:

* clips or repairs impossible values
* preserves PMRC lower bounds like TPA (\ge 100)
* recomputes dependent values consistently where needed

## Step 5. Create scenario grid

Define a small grid:

### Process noise

* none: (\lambda_{\text{proc}}=0)
* low
* medium
* high

### Disturbance frequency

* none
* 1/30
* 1/20
* 1/10

### Disturbance severity

* mild
* moderate
* severe

Do not explode the grid at first. Start with 6 to 12 carefully chosen scenarios.

## Step 6. Forward simulate many trajectories

For each scenario:

* run (N) trajectories, maybe (N=1000) at first
* same (s_0)
* same policy
* same prices
* same rotation length

Collect:

* terminal value
* NPV
* LEV
* product breakdown
* whether disturbance occurred at least once
* years of disturbance
* severity path summary

## Step 7. Compare distributions to deterministic baseline

For each scenario report:

* mean
* median
* standard deviation
* 5th percentile
* 95th percentile
* downside probability relative to deterministic baseline
* VaR and CVaR if you want the academic risk view

---

# 9. Thesis-ready formalization

You can use something close to this.

## 9.1 Transition model

> Let (s_t \in \mathcal S) denote the stand state at year (t), and let (a_t=\pi(s_t,t)) denote a heuristic management action selected by a fixed rule-based policy. The deterministic PMRC simulator defines a baseline transition map
>
> [
> \mu_{t+1}=f(s_t,a_t)
> ]
>
> which is interpreted as the mean biological trend. A stochastic extension is obtained by introducing annual process noise in selected growth increments, along with an independent yearly disturbance process:
>
> [
> s_{t+1}=g!\left(\Phi(\mu_{t+1},\xi_t,z_t,q_t)\right)
> ]
>
> where (\xi_t) denotes process innovations, (z_t) denotes disturbance occurrence, (q_t) denotes disturbance severity, (\Phi) is the provisional stochastic update, and (g) is a projection operator that enforces biological and PMRC-feasible bounds.

## 9.2 Disturbance process

> Disturbance occurrence is modeled as an annual Bernoulli trial
>
> [
> z_t \sim \text{Bernoulli}(p_{\text{dist}})
> ]
>
> where (p_{\text{dist}}=1/n) corresponds to an expected recurrence interval of (n) years. Conditional on occurrence, severity is drawn from a Beta distribution
>
> [
> q_t \mid z_t=1 \sim \text{Beta}(\alpha,\beta)
> ]
>
> and interpreted as a proportional damage intensity applied instantaneously to selected stand variables.

## 9.3 Terminal value

> Terminal timber value at the fixed rotation age (T) is defined as
>
> [
> V_T=\sum_{k=1}^{K} p_k y_{k,T}
> ]
>
> where (y_{k,T}) is the terminal yield in product class (k), and (p_k) is the corresponding unit price.

## 9.4 Simulation target

> For fixed initial state (s_0), fixed policy (\pi), and fixed stochastic-kernel parameters, the object of interest is the distribution
>
> [
> p(V_T\mid s_0,\pi,\lambda_{\text{proc}},\lambda_{\text{dist}})
> ]
>
> estimated via repeated forward simulation.

## 9.5 Epistemic treatment

> Because empirical calibration data were unavailable, epistemic uncertainty was represented through structured parameter scenarios over process-noise and disturbance-kernel parameters rather than posterior estimation.

That is all solid.

---

# 10. Visualization targets

These are the most useful plots for your thesis and debugging.

## A. Deterministic vs stochastic trajectory fan plot

For key state variables:

* BA
* TPA
* merchantable volume

Plot:

* deterministic PMRC trajectory as bold line
* median stochastic trajectory
* 5 to 95% interval ribbon

This shows whether the stochastic simulator still follows the PMRC trend.

## B. Histogram or KDE of terminal value

For each scenario:

* histogram of terminal dollars
* vertical line for deterministic baseline

This is probably your core figure.

## C. Boxplots by scenario

Scenarios on x-axis, terminal value on y-axis.

Useful for comparing:

* process noise only
* disturbance only
* combined uncertainty

## D. Product distribution breakdown at rotation

For each product class:

* mean share
* variability
* deterministic reference

Could use stacked bars or violin plots by product class.

## E. Downside risk plot

Show:

* mean terminal value
* 5th percentile
* VaR
* CVaR

This is useful academically even if your main output is dollars.

## F. Sensitivity heatmap

Grid over:

* process noise level
* disturbance recurrence interval

Color by:

* expected terminal value
* standard deviation
* downside risk

Very effective summary figure.

---

# 11. Implementation tests to maintain coherence with PMRC

These are extremely important.

## Test 1. Zero-noise recovery

If:

* (\lambda_{\text{proc}}=0)
* (p_{\text{dist}}=0)

then the stochastic simulator must exactly reproduce the deterministic PMRC path.

This should be a unit test.

## Test 2. Mean-trend closeness under low process noise

For small (\lambda_{\text{proc}}), the Monte Carlo mean trajectory should remain close to the deterministic PMRC trajectory.

Not exact, because lognormal noise shifts the mean slightly, but close.

To reduce bias, you may want to center the multiplicative innovation so its mean is 1 exactly. That is:

[
\Delta x_t^{\text{stoch}}
=========================

\Delta x_t^{\text{PMRC}}
\cdot \exp\left(\lambda_{\text{proc}}\sigma_x\epsilon_{x,t} - \frac{1}{2}(\lambda_{\text{proc}}\sigma_x)^2\right)
]

This keeps the multiplicative factor mean at 1.

Use this version. It is better.

## Test 3. Feasibility invariants

After every step, check **atomic state variables**:

* TPA (\ge 100) — PMRC lower bound
* BA (\ge 0)
* HD (\ge 0) and non-decreasing (height doesn't shrink)
* age > 0
* no NaNs in any atomic variable

Derived quantities (volume, products) are recomputed from atomics and will be valid if atomics are valid.

Automatic test.

## Test 4. Disturbance frequency sanity

Across many runs, empirical disturbance frequency should approximate (p_{\text{dist}}).

For example over many stand-years:
[
\hat p \approx p_{\text{dist}}
]

## Test 5. Severity distribution sanity

Conditional on event occurrence, empirical severity should match the chosen Beta mean and variance.

## Test 6. Monotonic risk intuition

Holding everything else fixed:

* increasing process noise should usually increase dispersion in terminal value
* increasing disturbance frequency or mean severity should usually lower mean value and worsen downside tails

This is not a strict theorem but should generally hold.

## Test 7. Deterministic policy consistency

Ensure the same rule-based thinning policy fires the same way in the deterministic setting and sensibly in the stochastic setting.

If policy is threshold-based, stochasticity may change whether thresholds are crossed. That is okay, but it should be understood and logged.

---

# 12. Python model skeleton

Here is a clean module layout.

```text
forest_sim/
    state.py
    policy.py
    pmrc.py
    stochastic.py
    disturbance.py
    projection.py
    valuation.py
    simulate.py
    scenarios.py
    metrics.py
    viz.py
    tests/
```

## state.py

Define stand state dataclass with **atomic variables only**:

```python
from dataclasses import dataclass
from typing import Literal

Region = Literal["ucp", "pucp", "lcp"]

@dataclass
class StandState:
    # Dynamic (projected each step)
    age: float
    hd: float       # dominant height
    tpa: float      # trees per acre
    ba: float       # basal area
    # Constant (fixed for stand lifetime)
    si25: float     # site index at base age 25 (must match HD definition)
    region: Region  # coefficient selection
    phwd: float = 0.0  # percent hardwood (optional)
```

Do NOT store derived quantities (volume, products, qmd). Compute them on-demand from atomics.

## policy.py

Heuristic rules.

```python
class HeuristicPolicy:
    def action(self, state: StandState) -> dict:
        ...
```

Keep action representation simple:

* no-op
* thin by fraction
* harvest

## pmrc.py

Pure deterministic one-year transition.

```python
def pmrc_step(state: StandState, action: dict) -> StandState:
    ...
```

This should remain clean and testable.

## stochastic.py

Apply process noise to increments.

```python
def noisy_increment(
    det_increment: float,
    sigma: float,
    noise_scale: float,
    rng,
) -> float:
    ...
```

Use the mean-corrected lognormal factor.

## disturbance.py

Occurrence and severity.

```python
def sample_occurrence(prob: float, rng) -> bool:
    ...

def sample_severity(mean: float, concentration: float, rng) -> float:
    ...

def apply_disturbance(state: StandState, severity: float) -> StandState:
    ...
```

## projection.py

Feasibility enforcement.

```python
def project_state(state: StandState) -> StandState:
    ...
```

## valuation.py

Terminal value, NPV, LEV.

```python
def terminal_value(products: dict, prices: dict) -> float:
    ...

def npv(cashflows: list[float], discount_rate: float) -> float:
    ...

def lev(npv_value: float, rotation_length: int, discount_rate: float) -> float:
    ...
```

## simulate.py

Trajectory rollout.

```python
def simulate_rotation(
    initial_state: StandState,
    policy,
    scenario,
    prices: dict,
    rng,
) -> dict:
    ...
```

Return:

* yearly states
* yearly actions
* disturbances
* terminal value
* npv
* lev

## scenarios.py

Structured scenario definitions.

```python
from dataclasses import dataclass

@dataclass
class Scenario:
    process_noise_scale: float
    disturbance_prob: float
    severity_mean: float
    severity_concentration: float
```

## metrics.py

Aggregate Monte Carlo outputs.

```python
def summarize_terminal_values(values: list[float]) -> dict:
    ...
```

## viz.py

All plotting functions.

---

# 13. Minimal simulation algorithm

For each year (t=0,\dots,T-1):

1. compute action (a_t = \pi(s_t,t))
2. compute deterministic next state via PMRC:
   [
   \mu_{t+1}=f(s_t,a_t)
   ]
3. convert deterministic increments to stochastic increments for selected variables
4. sample disturbance occurrence (z_t)
5. if (z_t=1), sample severity (q_t) and apply one-time damage
6. project to feasible state space
7. advance to next year
8. at terminal age, compute product-based terminal value, NPV, LEV

Repeat over many Monte Carlo runs.

---

# 14. A practical first scenario set

Do not start with too many scenarios.

I would begin with these six:

1. deterministic baseline
   (\lambda_{\text{proc}}=0,; p_{\text{dist}}=0)

2. low process noise only
   (\lambda_{\text{proc}}=0.25,; p_{\text{dist}}=0)

3. medium process noise only
   (\lambda_{\text{proc}}=0.5,; p_{\text{dist}}=0)

4. disturbance only, mild
   (\lambda_{\text{proc}}=0,; p_{\text{dist}}=1/30,; m_q=0.15)

5. disturbance only, moderate
   (\lambda_{\text{proc}}=0,; p_{\text{dist}}=1/20,; m_q=0.30)

6. combined
   (\lambda_{\text{proc}}=0.5,; p_{\text{dist}}=1/20,; m_q=0.30)

Then expand later.

---

# 15. What not to do yet

Avoid these in Phase 1:

* Gibbs sampling
* posterior inference without data
* state-dependent hazard models
* delayed recovery memory
* multiple interacting disturbance types
* action optimization via RL
* too many noisy variables at once
* trying to make the kernel perfect before it runs

Your first goal is a coherent stochastic extension of PMRC, not a final calibrated ecological truth model.

---

# 16. Strong writing framing for the limitations section

You should explicitly say:

* stochastic kernel parameters are scenario-based, not empirically estimated
* disturbance occurrence is independent across years
* disturbance effects are instantaneous, not delayed
* the deterministic PMRC model is used as the central trend
* process noise is introduced in selected annual increments to preserve biological interpretability
* feasibility projection is used to keep trajectories inside PMRC-valid ranges

That will preempt criticism.

---

# 17. Best next move

The best next move is to define, on paper, the exact state variables and which increments receive stochastic noise. Once that is fixed, the rest becomes straightforward code.

I can turn this into a concrete Python scaffold next, including:

* dataclasses
* function signatures
* scenario config objects
* a rollout loop
* and the unit tests for deterministic recovery and feasibility.
