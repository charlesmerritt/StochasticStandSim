# Reward Structure Improvements

## Problem
Agent learned to only take NOOP action because:
- No terminal standing value → growing timber had no reward
- High discount rate (5%) → future harvests heavily discounted
- Short horizon (30 years) → not enough time to see harvest payoff
- All actions had negative NPV (costs without future gains)

## Solutions Implemented

### 1. ✅ Terminal Standing Value (CRITICAL)
**File**: `core/env.py` lines 241-252

Standing timber at episode end now valued at **70% of harvest revenue**, discounted to NPV.

```python
terminal_cash = 0.7 * revenue_from_volume_tons_even_split(tons, prices)
terminal_npv = manager._npv(terminal_cash, age)
reward += terminal_npv
```

**Impact**: Agent now gets reward for growing timber even without harvesting.

---

### 2. ✅ Configurable Discount Rate
**File**: `core/env.py` lines 32-33, 87-91

Added `discount_rate` parameter to `EnvConfig`:
```python
discount_rate: float = 0.05  # default, can override
```

**Recommendation**: Use 2-3% for longer-term forestry planning.

---

### 3. ✅ Growth Reward Shaping (Optional)
**File**: `core/env.py` lines 34, 230-235

Small reward for volume increases each timestep:
```python
growth_reward_weight: float = 0.0  # set to 0.01 to enable
```

**When to use**: If agent still doesn't grow timber. Start with 0.01.

---

### 4. ✅ Updated Training Config
**File**: `rl/train.py` lines 28-34

```python
env_cfg = EnvConfig(
    discount_rate=0.03,           # Lower = value future more
    growth_reward_weight=0.01,    # Encourage growth
    horizon_years=40.0,           # Longer to see harvest payoff
    disturbance_enabled=True,     # Keep stochasticity
)
```

---

## Quick Tuning Guide

### If agent still does NOOP:
1. **Increase `growth_reward_weight`** from 0.01 → 0.05
2. **Lower `discount_rate`** from 0.03 → 0.01
3. **Increase `horizon_years`** from 40 → 50

### If agent harvests too early:
1. **Decrease `growth_reward_weight`** or set to 0.0
2. **Increase terminal value multiplier** from 0.7 → 0.9 (line 234 in env.py)

### If agent never harvests:
1. **Decrease terminal value multiplier** from 0.7 → 0.5
2. Make harvest more attractive by reducing costs in `core/actions.py`

### If rewards are unstable:
1. **Adjust reward clipping** (currently ±10,000 in env.py lines 186-224)
2. **Normalize observations** better in `_obs()` method

---

## Expected Behavior After Changes

With terminal value + lower discount + longer horizon:
- Agent should **grow timber** (positive volume growth)
- Agent may **thin** to improve growth rates
- Agent should **harvest near end** of horizon to capture terminal value
- Agent may **replant** after harvest if time remains

Run training with:
```bash
uv run python -m rl.train
```

Check action frequencies in output. You should see:
- NOOP: 50-70% (waiting for growth)
- HARVEST: 10-20% (at appropriate times)
- THIN: 5-15% (if beneficial)
- PLANT: 5-10% (after harvest)
