"""MDP Analysis for Forest Stand Management Under Uncertainty.

This script implements finite and infinite horizon MDP analysis for optimal
rotation and harvest timing decisions across different risk profiles.

MDP Formulation:
- State: (age, BA) discretized
- Actions: {wait, harvest} - thinning is automatic when BA > threshold
- Discount rate: 5% (γ = 0.952)
- Finite horizon: 20-year rotation with mandatory terminal harvest
- Infinite horizon: Optimal stopping problem

Key assumptions documented for thesis:
1. Thinning from below (smallest trees first) when BA > 100 ft²/ac
2. Stand resets to age 1 after harvest
3. NPV-based rewards using product-specific prices
4. Risk profiles: low/medium/high disturbance and noise
"""

import numpy as np
import matplotlib.pyplot as plt

from core.config import get_risk_profile
from core.mdp_solver import finite_horizon_value_iteration, value_iteration
from core.pmrc_model import PMRCModel
from core.products import compute_harvest_value, estimate_product_distribution
from core.stochastic_model import (
    StateDiscretizer,
    StandState,
    StochasticPMRC,
    thin_to_residual_ba_smallest_first,
)


# =============================================================================
# MDP Configuration
# =============================================================================

ROTATION_LENGTH = 20  # years
DISCOUNT_RATE = 0.05  # 5% annual
GAMMA = 1.0 / (1.0 + DISCOUNT_RATE)  # ≈ 0.952

BA_THIN_THRESHOLD = 100.0  # ft²/ac - thin when BA exceeds this
BA_THIN_TARGET = 70.0      # ft²/ac - target BA after thinning

SI25 = 60.0
REGION = "ucp"
INITIAL_TPA = 600.0

# State discretization
AGE_BINS = np.array([0, 5, 10, 15, 20, 25, 30])
BA_BINS = np.array([0, 40, 80, 120, 160, 200])
TPA_BINS = np.array([100, 300, 500, 700])


# =============================================================================
# Transition Matrix Generation with Automatic Thinning
# =============================================================================

def generate_transitions_with_auto_thin(
    risk_level: str,
    discretizer: StateDiscretizer,
    n_samples: int = 100,
    seed: int = 42,
) -> dict[int, np.ndarray]:
    """Generate transition matrices with automatic thinning rule.
    
    Actions:
        0 = wait (grow one year, auto-thin if BA > threshold)
        1 = harvest (collect revenue, reset to young stand)
    
    Returns:
        Dict mapping action -> transition matrix P[s, s']
    """
    profile = get_risk_profile(risk_level)
    pmrc = PMRCModel(region=REGION)
    stoch = StochasticPMRC.from_config(pmrc, profile.noise, profile.disturbance)
    
    n_states = discretizer.n_states
    P_wait = np.zeros((n_states, n_states))
    P_harvest = np.zeros((n_states, n_states))
    
    rng = np.random.default_rng(seed)
    
    # Find the "reset" state (young stand after harvest)
    reset_age_idx = 0  # First age bin
    reset_ba_idx = 0   # First BA bin
    reset_tpa_idx = len(discretizer.tpa_bins) - 2  # High TPA
    reset_state_idx = discretizer.encode_from_indices(reset_age_idx, reset_tpa_idx, reset_ba_idx)
    
    for s_idx in range(n_states):
        i_age, i_tpa, i_ba = discretizer.decode(s_idx)
        
        # Get bin midpoints
        age = 0.5 * (discretizer.age_bins[i_age] + discretizer.age_bins[i_age + 1])
        tpa = 0.5 * (discretizer.tpa_bins[i_tpa] + discretizer.tpa_bins[i_tpa + 1])
        ba = 0.5 * (discretizer.ba_bins[i_ba] + discretizer.ba_bins[i_ba + 1])
        
        # Compute HD from SI and age
        k, m = pmrc.k, pmrc.m
        hd = SI25 * ((1 - np.exp(-k * max(1.0, age))) / (1 - np.exp(-k * 25.0))) ** m
        
        state = StandState(age=age, hd=hd, tpa=tpa, ba=ba, si25=SI25, region=REGION)
        
        # Action 0: Wait (with automatic thinning)
        for _ in range(n_samples):
            # Apply automatic thinning if BA > threshold
            if state.ba > BA_THIN_THRESHOLD:
                thinned_state, _ = thin_to_residual_ba_smallest_first(state, BA_THIN_TARGET)
            else:
                thinned_state = state
            
            # Grow one year with stochastic transitions
            next_state = stoch.sample_next_state(thinned_state, dt=1.0, rng=rng)
            s_next = discretizer.encode(next_state)
            P_wait[s_idx, s_next] += 1.0
        
        # Action 1: Harvest (deterministic reset)
        P_harvest[s_idx, reset_state_idx] = n_samples
    
    # Normalize rows
    for P in [P_wait, P_harvest]:
        row_sums = P.sum(axis=1, keepdims=True)
        nonzero = row_sums[:, 0] > 0
        P[nonzero] /= row_sums[nonzero]
    
    return {0: P_wait, 1: P_harvest}


# =============================================================================
# NPV Reward Function
# =============================================================================

def make_npv_reward_fn(
    discretizer: StateDiscretizer,
    pmrc: PMRCModel,
    thin_threshold: float = BA_THIN_THRESHOLD,
    thin_target: float = BA_THIN_TARGET,
) -> callable:
    """Create NPV-based reward function.
    
    Rewards:
        - Wait: Small thinning revenue if auto-thin triggered, else 0
        - Harvest: Full harvest revenue based on standing volume
    """
    age_centers = 0.5 * (discretizer.age_bins[:-1] + discretizer.age_bins[1:])
    tpa_centers = 0.5 * (discretizer.tpa_bins[:-1] + discretizer.tpa_bins[1:])
    ba_centers = 0.5 * (discretizer.ba_bins[:-1] + discretizer.ba_bins[1:])
    
    def reward_fn(s: int, a: int, s_next: int) -> float:
        i_age, i_tpa, i_ba = discretizer.decode(s)
        age = age_centers[i_age]
        tpa = tpa_centers[i_tpa]
        ba = ba_centers[i_ba]
        
        # Compute HD
        k, m = pmrc.k, pmrc.m
        hd = SI25 * ((1 - np.exp(-k * max(1.0, age))) / (1 - np.exp(-k * 25.0))) ** m
        
        if a == 1:  # Harvest
            # Full harvest revenue
            if ba > 0 and tpa > 0:
                products = estimate_product_distribution(pmrc, ba, tpa, hd, region=REGION)
                return compute_harvest_value(products)
            return 0.0
        
        else:  # Wait
            # Thinning revenue if BA > threshold
            if ba > thin_threshold:
                # Revenue from removed trees
                removed_ba = ba - thin_target
                removed_frac = removed_ba / ba
                products = estimate_product_distribution(pmrc, ba, tpa, hd, region=REGION)
                # Thinning gets lower prices (mostly pulpwood from small trees)
                thin_value = compute_harvest_value(products) * removed_frac * 0.5
                return thin_value
            return 0.0
    
    return reward_fn


def make_terminal_reward_fn(
    discretizer: StateDiscretizer,
    pmrc: PMRCModel,
) -> callable:
    """Terminal reward for finite horizon: mandatory harvest value."""
    age_centers = 0.5 * (discretizer.age_bins[:-1] + discretizer.age_bins[1:])
    tpa_centers = 0.5 * (discretizer.tpa_bins[:-1] + discretizer.tpa_bins[1:])
    ba_centers = 0.5 * (discretizer.ba_bins[:-1] + discretizer.ba_bins[1:])
    
    def terminal_fn(s: int) -> float:
        i_age, i_tpa, i_ba = discretizer.decode(s)
        age = age_centers[i_age]
        tpa = tpa_centers[i_tpa]
        ba = ba_centers[i_ba]
        
        k, m = pmrc.k, pmrc.m
        hd = SI25 * ((1 - np.exp(-k * max(1.0, age))) / (1 - np.exp(-k * 25.0))) ** m
        
        if ba > 0 and tpa > 0:
            products = estimate_product_distribution(pmrc, ba, tpa, hd, region=REGION)
            return compute_harvest_value(products)
        return 0.0
    
    return terminal_fn


# =============================================================================
# Main Analysis
# =============================================================================

def run_mdp_analysis():
    """Run complete MDP analysis across risk profiles."""
    
    pmrc = PMRCModel(region=REGION)
    discretizer = StateDiscretizer(AGE_BINS, TPA_BINS, BA_BINS)
    
    print("=" * 70)
    print("FOREST STAND MDP ANALYSIS")
    print("=" * 70)
    print(f"State space: {discretizer.n_states} states")
    print(f"  Age bins: {len(AGE_BINS)-1}, BA bins: {len(BA_BINS)-1}, TPA bins: {len(TPA_BINS)-1}")
    print(f"Discount rate: {DISCOUNT_RATE:.1%} (γ = {GAMMA:.4f})")
    print(f"Rotation length: {ROTATION_LENGTH} years")
    print(f"Auto-thin threshold: BA > {BA_THIN_THRESHOLD} → {BA_THIN_TARGET} ft²/ac")
    print()
    
    reward_fn = make_npv_reward_fn(discretizer, pmrc)
    terminal_fn = make_terminal_reward_fn(discretizer, pmrc)
    
    results = {}
    
    for risk_level in ["low", "medium", "high"]:
        print(f"--- {risk_level.upper()} RISK ---")
        
        # Generate transition matrices
        print("  Generating transition matrices...")
        matrices = generate_transitions_with_auto_thin(
            risk_level, discretizer, n_samples=100, seed=42
        )
        
        # Infinite horizon MDP
        print("  Solving infinite horizon MDP...")
        inf_result = value_iteration(matrices, reward_fn, gamma=GAMMA, theta=1e-4)
        
        # Finite horizon MDP
        print("  Solving finite horizon MDP...")
        fin_result = finite_horizon_value_iteration(
            matrices, reward_fn, terminal_fn, horizon=ROTATION_LENGTH, gamma=GAMMA
        )
        
        # Policy analysis
        inf_harvest_states = np.sum(inf_result.policy == 1)
        fin_harvest_states = np.sum(fin_result.policy == 1)
        
        print(f"  Infinite horizon: {inf_result.iterations} iterations, "
              f"harvest in {inf_harvest_states}/{discretizer.n_states} states")
        print(f"  Finite horizon: harvest in {fin_harvest_states}/{discretizer.n_states} states at t=0")
        print(f"  Mean value (infinite): ${inf_result.values.mean():.0f}")
        print(f"  Mean value (finite t=0): ${fin_result.values.mean():.0f}")
        print()
        
        results[risk_level] = {
            "matrices": matrices,
            "infinite": inf_result,
            "finite": fin_result,
        }
    
    return results, discretizer, pmrc


def plot_policy_comparison(results: dict, discretizer: StateDiscretizer, pmrc: PMRCModel):
    """Create paper-ready plots comparing policies across risk levels."""
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle("Optimal Harvest Policy by Risk Level", fontsize=14)
    
    age_centers = 0.5 * (discretizer.age_bins[:-1] + discretizer.age_bins[1:])
    ba_centers = 0.5 * (discretizer.ba_bins[:-1] + discretizer.ba_bins[1:])
    
    for col, risk_level in enumerate(["low", "medium", "high"]):
        result = results[risk_level]
        
        # Infinite horizon policy (top row)
        ax = axes[0, col]
        policy_matrix = np.zeros((len(ba_centers), len(age_centers)))
        for i_age in range(len(age_centers)):
            for i_ba in range(len(ba_centers)):
                # Use middle TPA bin
                i_tpa = len(discretizer.tpa_bins) // 2 - 1
                s_idx = discretizer.encode_from_indices(i_age, i_tpa, i_ba)
                policy_matrix[i_ba, i_age] = result["infinite"].policy[s_idx]
        
        im = ax.imshow(policy_matrix, aspect="auto", origin="lower", cmap="RdYlGn_r",
                       extent=[age_centers[0], age_centers[-1], ba_centers[0], ba_centers[-1]])
        ax.set_xlabel("Age (years)")
        ax.set_ylabel("Basal Area (ft²/ac)")
        ax.set_title(f"{risk_level.capitalize()} Risk - Infinite Horizon")
        ax.axhline(BA_THIN_THRESHOLD, color="blue", linestyle="--", alpha=0.7, label="Thin threshold")
        
        # Finite horizon policy (bottom row)
        ax = axes[1, col]
        policy_matrix = np.zeros((len(ba_centers), len(age_centers)))
        for i_age in range(len(age_centers)):
            for i_ba in range(len(ba_centers)):
                i_tpa = len(discretizer.tpa_bins) // 2 - 1
                s_idx = discretizer.encode_from_indices(i_age, i_tpa, i_ba)
                policy_matrix[i_ba, i_age] = result["finite"].policy[s_idx]
        
        im = ax.imshow(policy_matrix, aspect="auto", origin="lower", cmap="RdYlGn_r",
                       extent=[age_centers[0], age_centers[-1], ba_centers[0], ba_centers[-1]])
        ax.set_xlabel("Age (years)")
        ax.set_ylabel("Basal Area (ft²/ac)")
        ax.set_title(f"{risk_level.capitalize()} Risk - Finite Horizon ({ROTATION_LENGTH}yr)")
        ax.axhline(BA_THIN_THRESHOLD, color="blue", linestyle="--", alpha=0.7)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.6, label="Action (0=Wait, 1=Harvest)")
    
    plt.tight_layout()
    plt.savefig("plots/mdp_policy_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved: plots/mdp_policy_comparison.png")


def plot_value_functions(results: dict, discretizer: StateDiscretizer):
    """Plot value functions across risk levels."""
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Value Functions by Risk Level (Infinite Horizon)", fontsize=14)
    
    age_centers = 0.5 * (discretizer.age_bins[:-1] + discretizer.age_bins[1:])
    ba_centers = 0.5 * (discretizer.ba_bins[:-1] + discretizer.ba_bins[1:])
    
    vmin = min(results[r]["infinite"].values.min() for r in results)
    vmax = max(results[r]["infinite"].values.max() for r in results)
    
    for col, risk_level in enumerate(["low", "medium", "high"]):
        ax = axes[col]
        result = results[risk_level]
        
        value_matrix = np.zeros((len(ba_centers), len(age_centers)))
        for i_age in range(len(age_centers)):
            for i_ba in range(len(ba_centers)):
                i_tpa = len(discretizer.tpa_bins) // 2 - 1
                s_idx = discretizer.encode_from_indices(i_age, i_tpa, i_ba)
                value_matrix[i_ba, i_age] = result["infinite"].values[s_idx]
        
        im = ax.imshow(value_matrix, aspect="auto", origin="lower", cmap="viridis",
                       extent=[age_centers[0], age_centers[-1], ba_centers[0], ba_centers[-1]],
                       vmin=vmin, vmax=vmax)
        ax.set_xlabel("Age (years)")
        ax.set_ylabel("Basal Area (ft²/ac)")
        ax.set_title(f"{risk_level.capitalize()} Risk")
        ax.axhline(BA_THIN_THRESHOLD, color="white", linestyle="--", alpha=0.7)
    
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, label="Value ($)")
    
    plt.tight_layout()
    plt.savefig("plots/mdp_value_functions.png", dpi=150, bbox_inches="tight")
    print("Saved: plots/mdp_value_functions.png")


def plot_optimal_rotation_age(results: dict, discretizer: StateDiscretizer):
    """Plot implied optimal rotation age by risk level."""
    
    age_centers = 0.5 * (discretizer.age_bins[:-1] + discretizer.age_bins[1:])
    ba_centers = 0.5 * (discretizer.ba_bins[:-1] + discretizer.ba_bins[1:])
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for risk_level in ["low", "medium", "high"]:
        result = results[risk_level]
        
        # Find minimum age where harvest is optimal for each BA level
        harvest_ages = []
        for i_ba in range(len(ba_centers)):
            for i_age in range(len(age_centers)):
                i_tpa = len(discretizer.tpa_bins) // 2 - 1
                s_idx = discretizer.encode_from_indices(i_age, i_tpa, i_ba)
                if result["infinite"].policy[s_idx] == 1:  # Harvest
                    harvest_ages.append(age_centers[i_age])
                    break
            else:
                harvest_ages.append(age_centers[-1])  # Never harvest in this BA
        
        ax.plot(ba_centers, harvest_ages, marker="o", label=f"{risk_level.capitalize()} Risk")
    
    ax.set_xlabel("Basal Area (ft²/ac)")
    ax.set_ylabel("Optimal Harvest Age (years)")
    ax.set_title("Implied Optimal Rotation Age by Risk Level")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(ROTATION_LENGTH, color="gray", linestyle="--", alpha=0.5, label=f"Rotation={ROTATION_LENGTH}yr")
    
    plt.tight_layout()
    plt.savefig("plots/mdp_optimal_rotation.png", dpi=150, bbox_inches="tight")
    print("Saved: plots/mdp_optimal_rotation.png")


def print_mdp_assumptions():
    """Print MDP model assumptions for thesis documentation."""
    
    print("\n" + "=" * 70)
    print("MDP MODEL ASSUMPTIONS FOR THESIS")
    print("=" * 70)
    print("""
1. STATE SPACE
   - Continuous state (age, BA, TPA) discretized into bins
   - Age: {age_bins} years
   - BA: {ba_bins} ft²/ac
   - TPA: {tpa_bins} trees/ac
   - Total states: {n_states}

2. ACTION SPACE
   - Wait (a=0): Grow one year, apply automatic thinning if needed
   - Harvest (a=1): Clear-cut, collect revenue, reset to young stand

3. AUTOMATIC THINNING RULE
   - Triggered when BA > {thin_thresh} ft²/ac
   - Target residual BA: {thin_target} ft²/ac
   - Thinning from below (smallest trees removed first)
   - Verified: QMD increases after thinning

4. TRANSITION DYNAMICS
   - Deterministic growth: PMRC equations (Burkhart & Tomé)
   - Stochastic noise: Lognormal BA, binomial TPA survival
   - Disturbances: Mild (annual) and severe (stand-replacing)
   - Risk profiles: Low/Medium/High with different parameters

5. REWARD FUNCTION
   - NPV-based using product-specific stumpage prices
   - Products: Pulpwood, Chip-n-Saw, Sawtimber
   - Thinning revenue: 50% of harvest value (smaller trees)
   - Discount rate: {discount:.1%} annually

6. HORIZON
   - Finite: {rotation}-year rotation with mandatory terminal harvest
   - Infinite: Optimal stopping problem with discounting

7. SOLUTION METHOD
   - Value iteration (infinite horizon)
   - Backward induction (finite horizon)
   - Convergence threshold: 1e-4
""".format(
        age_bins=list(AGE_BINS),
        ba_bins=list(BA_BINS),
        tpa_bins=list(TPA_BINS),
        n_states=(len(AGE_BINS)-1) * (len(BA_BINS)-1) * (len(TPA_BINS)-1),
        thin_thresh=BA_THIN_THRESHOLD,
        thin_target=BA_THIN_TARGET,
        discount=DISCOUNT_RATE,
        rotation=ROTATION_LENGTH,
    ))


if __name__ == "__main__":
    # Print assumptions
    print_mdp_assumptions()
    
    # Run analysis
    results, discretizer, pmrc = run_mdp_analysis()
    
    # Generate plots
    plot_policy_comparison(results, discretizer, pmrc)
    plot_value_functions(results, discretizer)
    plot_optimal_rotation_age(results, discretizer)
    
    # Verify thinning is from below
    print("\n" + "=" * 70)
    print("VERIFICATION: Thinning from Below")
    print("=" * 70)
    
    test_state = StandState(age=15, hd=50.0, tpa=400, ba=120.0, si25=SI25, region=REGION)
    qmd_pre = pmrc.qmd(test_state.tpa, test_state.ba)
    
    thinned, dist = thin_to_residual_ba_smallest_first(test_state, BA_THIN_TARGET)
    qmd_post = pmrc.qmd(thinned.tpa, thinned.ba)
    
    print(f"Pre-thin:  TPA={test_state.tpa:.0f}, BA={test_state.ba:.1f}, QMD={qmd_pre:.2f}\"")
    print(f"Post-thin: TPA={thinned.tpa:.0f}, BA={thinned.ba:.1f}, QMD={qmd_post:.2f}\"")
    print(f"QMD increased: {qmd_post > qmd_pre} (confirms smallest trees removed first)")
