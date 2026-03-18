"""MDP Paper Figures: Visualizations for Thesis.

Generates publication-ready figures for the MDP analysis:
1. Transition matrix heatmap (projected)
2. Risk-return frontier (disturbance probability vs expected return)
3. Policy regime comparison across risk levels
4. Sample stand trajectories at different risk levels
5. Categorical policy maps
"""

import sys
sys.path.insert(0, ".")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from core.config import get_risk_profile
from core.mdp import (
    Action,
    BALevel,
    BuongiornoConfig,
    BuongiornoDiscretizer,
    DisturbanceState,
    ForestState,
    PriceState,
    build_full_transition_matrix,
    estimate_stand_transitions,
    make_reward_function,
    solve_mdp_for_risk_level,
    value_iteration,
)
from core.pmrc_model import PMRCModel
from core.stochastic_model import StandState, StochasticPMRC


def plot_transition_matrix(save_path: str = "plots/mdp_transition_matrix.png") -> None:
    """Plot transition matrix heatmap for WAIT action."""
    print("Generating transition matrix visualization...")
    
    config = BuongiornoConfig()
    risk_profile = get_risk_profile("medium")
    
    # Estimate stand transitions
    stand_P = estimate_stand_transitions(config, risk_profile, n_samples=500, seed=42)
    
    # Build full transition matrix
    full_P = build_full_transition_matrix(
        stand_P, config.price_transition,
        config.n_stand_states, config.n_price_states
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Transition Matrices P(s'|s, a)", fontsize=14)
    
    # WAIT action
    ax = axes[0]
    im = ax.imshow(full_P[Action.WAIT], cmap="Blues", aspect="auto")
    ax.set_xlabel("Next State s'")
    ax.set_ylabel("Current State s")
    ax.set_title("Action: WAIT (with auto-thinning)")
    plt.colorbar(im, ax=ax, label="P(s'|s)")
    
    # HARVEST action
    ax = axes[1]
    im = ax.imshow(full_P[Action.HARVEST], cmap="Oranges", aspect="auto")
    ax.set_xlabel("Next State s'")
    ax.set_ylabel("Current State s")
    ax.set_title("Action: HARVEST (reset to young stand)")
    plt.colorbar(im, ax=ax, label="P(s'|s)")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_risk_return_frontier(save_path: str = "plots/mdp_risk_return_frontier.png") -> None:
    """Plot risk-return frontier across risk profiles."""
    print("Generating risk-return frontier...")
    
    config = BuongiornoConfig()
    pmrc = PMRCModel(region=config.region)
    
    risk_levels = ["low", "medium", "high"]
    colors = ["green", "orange", "red"]
    
    results = []
    
    for risk_level in risk_levels:
        profile = get_risk_profile(risk_level)
        
        # Solve MDP
        solution, P, _ = solve_mdp_for_risk_level(risk_level, config, n_samples=500, seed=42)
        
        # Compute expected return (mean value)
        expected_return = solution.V.mean()
        
        # Compute risk metrics
        # 1. Value variance across states
        value_variance = solution.V.var()
        value_std = solution.V.std()
        
        # 2. Disturbance probability from profile
        dist_prob = profile.disturbance.p_mild + profile.disturbance.p_severe_annual
        
        # 3. Probability of harvest action (policy aggressiveness)
        harvest_prob = (solution.policy == Action.HARVEST).mean()
        
        results.append({
            "risk_level": risk_level,
            "expected_return": expected_return,
            "value_std": value_std,
            "value_variance": value_variance,
            "disturbance_prob": dist_prob,
            "harvest_prob": harvest_prob,
        })
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Risk-Return Tradeoffs Across Risk Profiles", fontsize=14)
    
    # Plot 1: Expected Return vs Disturbance Probability
    ax = axes[0]
    for r, c in zip(results, colors):
        ax.scatter(r["disturbance_prob"], r["expected_return"], 
                   s=200, c=c, label=r["risk_level"].capitalize(), edgecolors="black", zorder=3)
    ax.set_xlabel("Annual Disturbance Probability")
    ax.set_ylabel("Expected Value V(s) ($)")
    ax.set_title("Return vs Disturbance Risk")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Expected Return vs Value Std Dev
    ax = axes[1]
    for r, c in zip(results, colors):
        ax.scatter(r["value_std"], r["expected_return"],
                   s=200, c=c, label=r["risk_level"].capitalize(), edgecolors="black", zorder=3)
    ax.set_xlabel("Value Standard Deviation ($)")
    ax.set_ylabel("Expected Value V(s) ($)")
    ax.set_title("Return vs Value Volatility")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Harvest Probability by Risk Level
    ax = axes[2]
    x = np.arange(len(risk_levels))
    bars = ax.bar(x, [r["harvest_prob"] for r in results], color=colors, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([r["risk_level"].capitalize() for r in results])
    ax.set_ylabel("Fraction of States with HARVEST Policy")
    ax.set_title("Policy Aggressiveness by Risk")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")
    
    # Add value labels on bars
    for bar, r in zip(bars, results):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{r["harvest_prob"]:.1%}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_policy_regime_comparison(save_path: str = "plots/mdp_policy_regimes.png") -> None:
    """Compare optimal policies across risk levels."""
    print("Generating policy regime comparison...")
    
    config = BuongiornoConfig()
    
    # Solve for each risk level
    solutions = {}
    for risk_level in ["low", "medium", "high"]:
        sol, _, _ = solve_mdp_for_risk_level(risk_level, config, n_samples=500, seed=42)
        solutions[risk_level] = sol
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Custom colormap: green=WAIT, red=HARVEST
    cmap = ListedColormap(["#2ecc71", "#e74c3c"])
    
    # Plot 1: Policy by Stand State (aggregated over price)
    ax = axes[0, 0]
    policy_by_stand = np.zeros((3, config.n_stand_states))  # risk_level × stand_state
    for i, risk_level in enumerate(["low", "medium", "high"]):
        sol = solutions[risk_level]
        # Aggregate: majority vote across price states
        for s_stand in range(config.n_stand_states):
            votes = [sol.policy[s_stand + p * config.n_stand_states] for p in range(config.n_price_states)]
            policy_by_stand[i, s_stand] = 1 if sum(votes) > 1.5 else 0
    
    im = ax.imshow(policy_by_stand, cmap=cmap, aspect="auto")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Low Risk", "Medium Risk", "High Risk"])
    ax.set_xlabel("Stand State Index")
    ax.set_title("Policy by Stand State (majority across prices)")
    
    # Add stand state labels
    stand_labels = []
    for s in range(config.n_stand_states):
        state = ForestState.from_index(s)
        stand_labels.append(f"{state.ba_small.name[0]}/{state.ba_medium.name[0]}/{state.ba_large.name[0]}")
    ax.set_xticks(range(0, config.n_stand_states, 2))
    ax.set_xticklabels([stand_labels[i] for i in range(0, config.n_stand_states, 2)], rotation=45, ha="right", fontsize=8)
    
    # Plot 2: Policy by Price State (aggregated over stand)
    ax = axes[0, 1]
    policy_by_price = np.zeros((3, config.n_price_states))
    for i, risk_level in enumerate(["low", "medium", "high"]):
        sol = solutions[risk_level]
        for p in range(config.n_price_states):
            votes = [sol.policy[s + p * config.n_stand_states] for s in range(config.n_stand_states)]
            policy_by_price[i, p] = sum(votes) / config.n_stand_states  # fraction harvesting
    
    x = np.arange(config.n_price_states)
    width = 0.25
    for i, (risk_level, color) in enumerate(zip(["low", "medium", "high"], ["green", "orange", "red"])):
        ax.bar(x + i*width, policy_by_price[i], width, label=risk_level.capitalize(), color=color, edgecolor="black")
    
    ax.set_xticks(x + width)
    ax.set_xticklabels(["Low Price", "Medium Price", "High Price"])
    ax.set_ylabel("Fraction Harvesting")
    ax.set_title("Harvest Probability by Price State")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")
    
    # Plot 3: Value Function Comparison
    ax = axes[1, 0]
    for risk_level, color in zip(["low", "medium", "high"], ["green", "orange", "red"]):
        sol = solutions[risk_level]
        ax.plot(range(config.n_states), sol.V, label=risk_level.capitalize(), color=color, linewidth=2)
    
    ax.set_xlabel("State Index")
    ax.set_ylabel("Value V(s) ($)")
    ax.set_title("Value Function by State")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Policy Differences
    ax = axes[1, 1]
    # Count states where policies differ
    policy_low = solutions["low"].policy
    policy_med = solutions["medium"].policy
    policy_high = solutions["high"].policy
    
    same_all = np.sum((policy_low == policy_med) & (policy_med == policy_high))
    diff_low_med = np.sum(policy_low != policy_med)
    diff_med_high = np.sum(policy_med != policy_high)
    diff_low_high = np.sum(policy_low != policy_high)
    
    categories = ["Same\n(all 3)", "Low≠Med", "Med≠High", "Low≠High"]
    values = [same_all, diff_low_med, diff_med_high, diff_low_high]
    colors_bar = ["#3498db", "#f39c12", "#e74c3c", "#9b59b6"]
    
    bars = ax.bar(categories, values, color=colors_bar, edgecolor="black")
    ax.set_ylabel("Number of States")
    ax.set_title("Policy Agreement Across Risk Levels")
    ax.grid(True, alpha=0.3, axis="y")
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(val), ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add legend for policy colors
    legend_elements = [Patch(facecolor="#2ecc71", edgecolor="black", label="WAIT"),
                       Patch(facecolor="#e74c3c", edgecolor="black", label="HARVEST")]
    fig.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_sample_trajectories(save_path: str = "plots/mdp_sample_trajectories.png") -> None:
    """Plot mean stand trajectories under different risk levels.
    
    Simulates 250 trajectories per risk level and plots the mean.
    """
    print("Generating sample trajectories (250 per risk level)...")
    
    config = BuongiornoConfig()
    pmrc = PMRCModel(region=config.region)
    
    n_trajectories = 250
    n_years = 30
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Mean Stand Trajectories by Risk Level (n=250)", fontsize=14)
    
    risk_colors = {"low": "green", "medium": "orange", "high": "red"}
    
    for col, risk_level in enumerate(["low", "medium", "high"]):
        profile = get_risk_profile(risk_level)
        stoch = StochasticPMRC.from_config(pmrc, profile.noise, profile.disturbance)
        
        # Simulate trajectories (no MDP - just growth with deterministic thinning)
        all_ba = []
        all_tpa = []
        
        for traj_idx in range(n_trajectories):
            rng = np.random.default_rng(seed=42 + traj_idx)
            
            # Start from young stand
            k, m = pmrc.k, pmrc.m
            age = 1.0
            hd = config.si25 * ((1 - np.exp(-k * age)) / (1 - np.exp(-k * 25.0))) ** m
            state = StandState(
                age=age, hd=hd, tpa=config.initial_tpa, ba=5.0,
                si25=config.si25, region=config.region
            )
            
            ba_history = [state.ba]
            tpa_history = [state.tpa]
            thinned = False
            
            for year in range(1, n_years + 1):
                # Deterministic thinning at age 15 if BA > threshold
                if year == 15 and state.ba > config.auto_thin_threshold and not thinned:
                    from core.stochastic_model import thin_to_residual_ba_smallest_first
                    state, _ = thin_to_residual_ba_smallest_first(state, config.auto_thin_target)
                    thinned = True
                
                # Grow
                state = stoch.sample_next_state(state, dt=1.0, rng=rng)
                
                ba_history.append(state.ba)
                tpa_history.append(state.tpa)
            
            all_ba.append(ba_history)
            all_tpa.append(tpa_history)
        
        all_ba = np.array(all_ba)
        all_tpa = np.array(all_tpa)
        years = np.arange(n_years + 1)
        
        color = risk_colors[risk_level]
        
        # Plot BA mean
        ax = axes[0, col]
        ba_mean = all_ba.mean(axis=0)
        ax.plot(years, ba_mean, color=color, linewidth=2.5, label="Mean")
        ax.set_xlabel("Year")
        ax.set_ylabel("Basal Area (ft²/ac)")
        ax.set_title(f"{risk_level.capitalize()} Risk")
        ax.set_ylim(0, 180)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
        
        # Plot TPA mean (bottom row)
        ax = axes[1, col]
        tpa_mean = all_tpa.mean(axis=0)
        ax.plot(years, tpa_mean, color=color, linewidth=2.5, label="Mean")
        ax.set_xlabel("Year")
        ax.set_ylabel("Trees per Acre")
        ax.set_title(f"TPA ({risk_level.capitalize()} Risk)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_categorical_policy_map(save_path: str = "plots/mdp_categorical_policy.png") -> None:
    """Create categorical policy map showing action by state components."""
    print("Generating categorical policy map...")
    
    config = BuongiornoConfig()
    solution, _, _ = solve_mdp_for_risk_level("low", config, n_samples=500, seed=42)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Optimal Policy Map: WAIT (green) vs HARVEST (red)", fontsize=14)
    
    cmap = ListedColormap(["#2ecc71", "#e74c3c"])
    
    # Create policy matrices for different views
    for price_idx, price_name in enumerate(["LOW", "MEDIUM", "HIGH"]):
        ax = axes[0, price_idx]
        
        # Matrix: rows = disturbance (2), cols = BA combination (8)
        policy_matrix = np.zeros((2, 8))
        
        for dist in range(2):
            for ba_combo in range(8):
                s_stand = ba_combo + dist * 8
                s_idx = s_stand + price_idx * config.n_stand_states
                policy_matrix[dist, ba_combo] = solution.policy[s_idx]
        
        im = ax.imshow(policy_matrix, cmap=cmap, aspect="auto")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Normal", "Disturbed"])
        ax.set_ylabel("Disturbance State")
        
        # BA combination labels
        ba_labels = []
        for i in range(8):
            s = ForestState.from_index(i)
            ba_labels.append(f"{s.ba_small.name[0]}/{s.ba_medium.name[0]}/{s.ba_large.name[0]}")
        ax.set_xticks(range(8))
        ax.set_xticklabels(ba_labels, rotation=45, ha="right", fontsize=9)
        ax.set_xlabel("BA (Small/Med/Large)")
        ax.set_title(f"Price: {price_name}")
    
    # Bottom row: Policy by Large BA level
    for large_ba in range(2):
        ax = axes[1, large_ba]
        
        # Matrix: rows = price (3), cols = small/med BA combo (4)
        policy_matrix = np.zeros((3, 4))
        
        for price_idx in range(3):
            for sm_combo in range(4):
                ba_combo = sm_combo + large_ba * 4  # Assuming large BA is bit 2
                s_stand = ba_combo  # Normal disturbance
                s_idx = s_stand + price_idx * config.n_stand_states
                policy_matrix[price_idx, sm_combo] = solution.policy[s_idx]
        
        im = ax.imshow(policy_matrix, cmap=cmap, aspect="auto")
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["Low Price", "Med Price", "High Price"])
        ax.set_ylabel("Price State")
        
        sm_labels = ["L/L", "H/L", "L/H", "H/H"]
        ax.set_xticks(range(4))
        ax.set_xticklabels(sm_labels)
        ax.set_xlabel("Small/Medium BA")
        ax.set_title(f"Large BA: {'HIGH' if large_ba else 'LOW'}")
    
    # Summary statistics in bottom right
    ax = axes[1, 2]
    ax.axis("off")
    
    n_wait = sum(solution.policy == Action.WAIT)
    n_harvest = sum(solution.policy == Action.HARVEST)
    
    summary_text = f"""
    Policy Summary (Low Risk)
    ─────────────────────────
    Total States: {config.n_states}
    
    WAIT:    {n_wait} states ({n_wait/config.n_states:.1%})
    HARVEST: {n_harvest} states ({n_harvest/config.n_states:.1%})
    
    Key Patterns:
    • Large BA HIGH → mostly HARVEST
    • Large BA LOW → mostly WAIT
    • High prices favor HARVEST
    • Disturbance shifts timing
    """
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Add legend
    legend_elements = [Patch(facecolor="#2ecc71", edgecolor="black", label="WAIT"),
                       Patch(facecolor="#e74c3c", edgecolor="black", label="HARVEST")]
    fig.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_value_function_surface(save_path: str = "plots/mdp_value_surface.png") -> None:
    """Plot value function as 3D-like surface."""
    print("Generating value function surface...")
    
    config = BuongiornoConfig()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Value Function V(s) by Risk Level", fontsize=14)
    
    for ax, risk_level in zip(axes, ["low", "medium", "high"]):
        solution, _, _ = solve_mdp_for_risk_level(risk_level, config, n_samples=500, seed=42)
        
        # Reshape: stand states × price states
        V_matrix = solution.V.reshape(config.n_price_states, config.n_stand_states).T
        
        im = ax.imshow(V_matrix, cmap="viridis", aspect="auto")
        ax.set_xlabel("Price State")
        ax.set_ylabel("Stand State Index")
        ax.set_title(f"{risk_level.capitalize()} Risk\nV range: [${solution.V.min():.0f}, ${solution.V.max():.0f}]")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Low", "Med", "High"])
        
        plt.colorbar(im, ax=ax, label="V(s) ($)")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    print("=" * 70)
    print("GENERATING MDP PAPER FIGURES")
    print("=" * 70)
    
    # Generate all figures
    plot_transition_matrix()
    plot_risk_return_frontier()
    plot_policy_regime_comparison()
    plot_sample_trajectories()
    plot_categorical_policy_map()
    plot_value_function_surface()
    
    print("\n" + "=" * 70)
    print("ALL FIGURES GENERATED")
    print("=" * 70)
    print("""
    Generated files:
    - plots/mdp_transition_matrix.png
    - plots/mdp_risk_return_frontier.png
    - plots/mdp_policy_regimes.png
    - plots/mdp_sample_trajectories.png
    - plots/mdp_categorical_policy.png
    - plots/mdp_value_surface.png
    """)
