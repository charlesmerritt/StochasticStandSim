# Validation Tests

What each test checks and what counts as acceptable behavior.

## Test Suite: `tests/test_stochastic.py`

### Test 1: Zero-Noise Recovery

**What it checks**: When λ_proc=0 and p_dist=0, the stochastic simulator must exactly reproduce the deterministic PMRC trajectory.

**Acceptable behavior**: All state variables (age, hd, tpa, ba) match within relative tolerance of 1e-6 at every time step. NPV and LEV match within 1e-6.

**Why it matters**: Ensures the stochastic model is a proper extension of PMRC, not a replacement.

---

### Test 2: Mean-Trend Closeness

**What it checks**: For small λ_proc (0.25), the Monte Carlo mean trajectory should remain close to deterministic.

**Acceptable behavior**: Mean NPV from 500 trajectories is within 15% of deterministic NPV.

**Why it matters**: Lognormal noise is mean-corrected, so bias should stay modest for low noise even after applying the PMRC merchantability equations at valuation time.

---

### Test 3: Feasibility Invariants

**What it checks**: After every simulation step, atomic state variables satisfy biological constraints.

**Acceptable behavior**:
- TPA ≥ 100 (PMRC lower bound)
- BA ≥ 0
- HD non-decreasing (height doesn't shrink)
- Age > 0
- No NaN values

**Why it matters**: Ensures simulated stands remain biologically and model-feasible.

---

### Test 4: Disturbance Frequency Sanity

**What it checks**: Empirical disturbance rate matches the specified p_dist parameter.

**Acceptable behavior**: Over 1000 runs × 30 years, empirical rate is within 20% of p_dist.

**Why it matters**: Validates Bernoulli sampling is correctly implemented.

---

### Test 5: Severity Distribution Sanity

**What it checks**: Conditional on occurrence, severity draws match Beta distribution parameters.

**Acceptable behavior**:
- Empirical mean within 5% of m_q (0.30)
- Empirical variance within 20% of theoretical Beta variance

**Why it matters**: Validates Beta severity sampling is correctly parameterized.

---

### Test 6: Monotonic Risk Intuition

**What it checks**: Increasing noise/disturbance should increase dispersion and worsen downside.

**Acceptable behavior**:
- Standard deviation of NPV is monotonically increasing with λ_proc
- Mean NPV is monotonically decreasing with p_dist

**Why it matters**: Ensures model behaves as expected under risk intuition.

---

### Test 7: Deterministic Policy Consistency

**What it checks**: Thinning policy fires consistently in both deterministic and stochastic scenarios.

**Acceptable behavior**:
- Deterministic: Thinning occurs at year 10 (age 15)
- Stochastic (no disturbance): Thinning occurs in >80% of runs

**Why it matters**: Ensures management policy logic is not broken by stochastic extensions.

---

### Test 8: Comparison Metrics and Disturbance Output Tracking

**What it checks**:
- Batch outputs retain disturbance occurrence, disturbance years, and severity paths for each trajectory
- Scenario comparisons report downside probability relative to the deterministic baseline

**Acceptable behavior**:
- Batch result arrays and path lists have one entry per trajectory
- Deterministic baseline downside probability is 0
- Stochastic downside probability is bounded in [0, 1]

**Why it matters**: Ensures the experiment layer can reproduce the Step 6/7 reporting requirements from the spec without reconstructing disturbance histories after the fact.

---

## Test Suite: `tests/test_site_index.py`

### HD-SI25 Consistency

**What it checks**: Dominant height projection follows the Chapman-Richards site index curve.

**Acceptable behavior**: HD at any age matches SI25 curve within tolerance.

---

## Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test class
uv run pytest tests/test_stochastic.py::TestZeroNoiseRecovery -v

# Run with coverage
uv run pytest tests/ --cov=core --cov-report=term-missing
```

## Adding New Tests

When adding new stochastic components:
1. Add a zero-noise recovery check (component disabled → deterministic)
2. Add a sanity check (empirical statistics match parameters)
3. Add a feasibility check (constraints still hold)
4. Update this document with the new test description
