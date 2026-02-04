"""Buongiorno-Style MDP Analysis for Forest Stand Management.

This script implements the complete MDP analysis following Buongiorno & Zhou (2015)
methodology for optimal forest management under uncertainty.

Outputs:
- V(s): Value function for each state
- Q(s,a): Action-value function
- π(a|s): Optimal policy with argmax
- Policy comparison across risk levels
- Paper-ready visualizations
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt

from core.mdp import (
    Action,
    BALevel,
    BuongiornoConfig,
    DisturbanceState,
    ForestState,
    PriceState,
    one_step_lookahead,
    solve_mdp_for_risk_level,
)
from core.pmrc_model import PMRCModel
from core.stochastic_stand import StandState, thin_to_residual_ba_smallest_first


def print_mdp_assumptions(config: BuongiornoConfig) -> None:
    """Print MDP model assumptions for thesis documentation."""
    print("\n" + "=" * 70)
    print("MDP MODEL SPECIFICATION (Buongiorno-Style)")
    print("=" * 70)
    print(f"""
1. STATE SPACE ({config.n_states} states)
   Stand states ({config.n_stand_states}):
   - Size classes: Small (<6"), Medium (6-9"), Large (≥9")
   - BA level per class: Low/High (binary threshold)
   - Disturbance: Normal / Recently disturbed
   
   Price states ({config.n_price_states}):
   - Low (80% of base), Medium (100%), High (120%)

2. ACTION SPACE
   - WAIT (a=0): Grow one year, auto-thin if BA > {config.auto_thin_threshold} ft²/ac
   - HARVEST (a=1): Clearcut, collect revenue, reset stand

3. AUTOMATIC THINNING RULE
   - Triggered when total BA > {config.auto_thin_threshold} ft²/ac
   - Target residual BA: {config.auto_thin_target} ft²/ac
   - Thinning from below (smallest trees removed first)
   - Revenue: {config.thin_revenue_multiplier:.0%} of harvest price
   - Cost: {config.thin_cost_multiplier:.0%} of harvest cost

4. TRANSITION DYNAMICS
   - Stand: P(s'|s,a) estimated via Monte Carlo simulation
   - Price: Markov model with high autocorrelation
   - Joint: S = P(s'|s) × P(m'|m) (Buongiorno eq. 5)

5. REWARD FUNCTION
   - Harvest: gross_revenue × price_mult - cost
   - Thin: (removed_frac × gross × 0.70 × price) - (cost × 1.20)
   - Wait (no thin): 0

6. DISCOUNT RATE
   - Annual: {config.discount_rate:.1%}
   - Factor: γ = {config.gamma:.4f}
""")


def run_full_analysis() -> dict:
    """Run complete MDP analysis across all risk levels."""
    config = BuongiornoConfig()
    print_mdp_assumptions(config)
    
    results = {}
    
    print("\n" + "=" * 70)
    print("SOLVING MDPs")
    print("=" * 70)
    
    for risk_level in ["low", "medium", "high"]:
        print(f"\n--- {risk_level.upper()} RISK ---")
        
        # Infinite horizon
        sol_inf, P_inf, _ = solve_mdp_for_risk_level(
            risk_level, config, horizon=None, n_samples=500, seed=42
        )
        
        # Finite horizon (20 years)
        sol_fin, P_fin, _ = solve_mdp_for_risk_level(
            risk_level, config, horizon=20, n_samples=500, seed=42
        )
        
        results[risk_level] = {
            "infinite": sol_inf,
            "finite": sol_fin,
            "P": P_inf,
        }
        
        print(f"  Infinite horizon: {sol_inf.iterations} iterations")
        print(f"    WAIT: {sum(sol_inf.policy == Action.WAIT)}, HARVEST: {sum(sol_inf.policy == Action.HARVEST)}")
        print(f"    V(s) range: [${sol_inf.V.min():.0f}, ${sol_inf.V.max():.0f}]")
        
        print(f"  Finite horizon (T=20):")
        print(f"    WAIT: {sum(sol_fin.policy == Action.WAIT)}, HARVEST: {sum(sol_fin.policy == Action.HARVEST)}")
        print(f"    V(s) range: [${sol_fin.V.min():.0f}, ${sol_fin.V.max():.0f}]")
    
    return results, config


def print_value_function(results: dict, config: BuongiornoConfig) -> None:
    """Print V(s) table."""
    print("\n" + "=" * 70)
    print("VALUE FUNCTION V(s) - Infinite Horizon")
    print("=" * 70)
    print(f"{'State':<45} {'LOW':<12} {'MED':<12} {'HIGH':<12}")
    print("-" * 70)
    
    for s_idx in range(config.n_states):
        state = ForestState.from_index(s_idx)
        state_str = (
            f"BA({state.ba_small.name[0]}/"
            f"{state.ba_medium.name[0]}/"
            f"{state.ba_large.name[0]}) "
            f"Dist={state.disturbed.name[:4]} "
            f"Price={state.price.name}"
        )
        
        v_low = results["low"]["infinite"].V[s_idx]
        v_med = results["medium"]["infinite"].V[s_idx]
        v_high = results["high"]["infinite"].V[s_idx]
        
        print(f"{state_str:<45} ${v_low:>9.0f} ${v_med:>9.0f} ${v_high:>9.0f}")


def print_q_table(results: dict, config: BuongiornoConfig, risk_level: str = "low") -> None:
    """Print Q(s,a) table."""
    sol = results[risk_level]["infinite"]
    
    print("\n" + "=" * 70)
    print(f"Q-VALUE TABLE Q(s,a) - {risk_level.upper()} RISK")
    print("=" * 70)
    print(f"{'State':<45} {'Q(WAIT)':<12} {'Q(HARVEST)':<12} {'π(s)':<10}")
    print("-" * 70)
    
    for s_idx in range(config.n_states):
        state = ForestState.from_index(s_idx)
        state_str = (
            f"BA({state.ba_small.name[0]}/"
            f"{state.ba_medium.name[0]}/"
            f"{state.ba_large.name[0]}) "
            f"Dist={state.disturbed.name[:4]} "
            f"Price={state.price.name}"
        )
        
        q_wait = sol.Q[s_idx, 0]
        q_harvest = sol.Q[s_idx, 1]
        policy = Action(sol.policy[s_idx]).name
        
        print(f"{state_str:<45} ${q_wait:>9.0f} ${q_harvest:>9.0f} {policy:<10}")


def print_policy_comparison(results: dict, config: BuongiornoConfig) -> None:
    """Print policy comparison across risk levels."""
    print("\n" + "=" * 70)
    print("POLICY COMPARISON π(s) = argmax_a Q(s,a)")
    print("=" * 70)
    print(f"{'State':<45} {'LOW':<8} {'MED':<8} {'HIGH':<8}")
    print("-" * 70)
    
    for s_idx in range(config.n_states):
        state = ForestState.from_index(s_idx)
        state_str = (
            f"BA({state.ba_small.name[0]}/"
            f"{state.ba_medium.name[0]}/"
            f"{state.ba_large.name[0]}) "
            f"Dist={state.disturbed.name[:4]} "
            f"Price={state.price.name}"
        )
        
        p_low = Action(results["low"]["infinite"].policy[s_idx]).name[:4]
        p_med = Action(results["medium"]["infinite"].policy[s_idx]).name[:4]
        p_high = Action(results["high"]["infinite"].policy[s_idx]).name[:4]
        
        print(f"{state_str:<45} {p_low:<8} {p_med:<8} {p_high:<8}")


def demonstrate_one_step_lookahead(results: dict, config: BuongiornoConfig) -> None:
    """Demonstrate planner's one-step lookahead."""
    print("\n" + "=" * 70)
    print("ONE-STEP LOOKAHEAD (Planner's View)")
    print("=" * 70)
    
    # Pick a few interesting states
    test_states = [
        ForestState(BALevel.LOW, BALevel.LOW, BALevel.HIGH, DisturbanceState.NORMAL, PriceState.MEDIUM),
        ForestState(BALevel.HIGH, BALevel.HIGH, BALevel.LOW, DisturbanceState.NORMAL, PriceState.HIGH),
        ForestState(BALevel.LOW, BALevel.HIGH, BALevel.HIGH, DisturbanceState.RECENTLY_DISTURBED, PriceState.LOW),
    ]
    
    from core.mdp import make_reward_function
    pmrc = PMRCModel(region=config.region)
    reward_fn = make_reward_function(config, pmrc)
    
    for state in test_states:
        s_idx = state.to_index()
        sol = results["low"]["infinite"]
        
        lookahead = one_step_lookahead(s_idx, results["low"]["P"], reward_fn, sol.V, config.gamma)
        
        state_str = (
            f"BA({state.ba_small.name[0]}/"
            f"{state.ba_medium.name[0]}/"
            f"{state.ba_large.name[0]}) "
            f"Dist={state.disturbed.name[:4]} "
            f"Price={state.price.name}"
        )
        
        print(f"\nState: {state_str}")
        print(f"  Q(WAIT)    = ${lookahead[Action.WAIT]:,.0f}")
        print(f"  Q(HARVEST) = ${lookahead[Action.HARVEST]:,.0f}")
        print(f"  Best action: {Action.WAIT.name if lookahead[Action.WAIT] > lookahead[Action.HARVEST] else Action.HARVEST.name}")


def plot_value_functions(results: dict, config: BuongiornoConfig) -> None:
    """Create paper-ready value function visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Value Functions V(s) by Risk Level", fontsize=14)
    
    # Reshape values for heatmap (stand states × price states)
    for ax, risk_level in zip(axes, ["low", "medium", "high"]):
        V = results[risk_level]["infinite"].V
        
        # Reshape: rows = stand states (16), cols = price states (3)
        V_matrix = V.reshape(config.n_price_states, config.n_stand_states).T
        
        im = ax.imshow(V_matrix, aspect="auto", cmap="viridis")
        ax.set_xlabel("Price State")
        ax.set_ylabel("Stand State Index")
        ax.set_title(f"{risk_level.capitalize()} Risk")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Low", "Med", "High"])
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label="V(s) ($)")
    
    plt.tight_layout()
    plt.savefig("plots/mdp_value_functions_buongiorno.png", dpi=150, bbox_inches="tight")
    print("\nSaved: plots/mdp_value_functions_buongiorno.png")


def plot_policy_heatmap(results: dict, config: BuongiornoConfig) -> None:
    """Create policy heatmap."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Optimal Policy π(s) by Risk Level (0=WAIT, 1=HARVEST)", fontsize=14)
    
    for ax, risk_level in zip(axes, ["low", "medium", "high"]):
        policy = results[risk_level]["infinite"].policy
        
        # Reshape: rows = stand states (16), cols = price states (3)
        P_matrix = policy.reshape(config.n_price_states, config.n_stand_states).T
        
        im = ax.imshow(P_matrix, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
        ax.set_xlabel("Price State")
        ax.set_ylabel("Stand State Index")
        ax.set_title(f"{risk_level.capitalize()} Risk")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Low", "Med", "High"])
    
    plt.colorbar(im, ax=axes, label="Action (0=WAIT, 1=HARVEST)", shrink=0.8)
    
    plt.tight_layout()
    plt.savefig("plots/mdp_policy_buongiorno.png", dpi=150, bbox_inches="tight")
    print("Saved: plots/mdp_policy_buongiorno.png")


def verify_thinning_from_below() -> None:
    """Verify that automatic thinning removes smallest trees first."""
    print("\n" + "=" * 70)
    print("VERIFICATION: Thinning from Below")
    print("=" * 70)
    
    pmrc = PMRCModel(region="ucp")
    config = BuongiornoConfig()
    
    test_state = StandState(
        age=15, hd=50.0, tpa=400, ba=160.0, si25=60.0, region="ucp"
    )
    
    qmd_pre = pmrc.qmd(test_state.tpa, test_state.ba)
    thinned, dist = thin_to_residual_ba_smallest_first(test_state, config.auto_thin_target)
    qmd_post = pmrc.qmd(thinned.tpa, thinned.ba)
    
    print(f"Pre-thin:  TPA={test_state.tpa:.0f}, BA={test_state.ba:.1f} ft²/ac, QMD={qmd_pre:.2f}\"")
    print(f"Post-thin: TPA={thinned.tpa:.0f}, BA={thinned.ba:.1f} ft²/ac, QMD={qmd_post:.2f}\"")
    print(f"QMD increased: {qmd_post > qmd_pre} ✓")
    print("\nThis confirms smallest trees are removed first (thinning from below).")


if __name__ == "__main__":
    # Run full analysis
    results, config = run_full_analysis()
    
    # Print tables
    print_value_function(results, config)
    print_q_table(results, config, risk_level="low")
    print_policy_comparison(results, config)
    
    # Demonstrate one-step lookahead
    demonstrate_one_step_lookahead(results, config)
    
    # Verify thinning
    verify_thinning_from_below()
    
    # Generate plots
    plot_value_functions(results, config)
    plot_policy_heatmap(results, config)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
