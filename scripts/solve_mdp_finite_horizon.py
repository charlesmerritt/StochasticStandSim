"""Solve finite horizon MDP and generate deliverables.

Deliverables:
1. State transition probability table (CSV)
2. Policy representation for each risk level
3. Sample trajectory means at each risk level
4. Updated deterministic vs stochastic plot
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from core.mdp import (
    BuongiornoConfig, BuongiornoDiscretizer,
    ForestState, BALevel, TPALevel, DisturbanceState, PriceState,
    Action, make_reward_function,
)
from core.pmrc_model import PMRCModel
from core.stochastic_stand import StochasticPMRC, StandState, thin_to_residual_ba_smallest_first
from core.config import get_risk_profile, RiskProfile


# =============================================================================
# Transition Matrix Estimation (WAIT and THIN actions)
# =============================================================================

def estimate_transitions_wait_thin(
    config: BuongiornoConfig,
    risk_profile: RiskProfile,
    n_samples: int = 500,
    seed: int = 42,
) -> dict[Action, np.ndarray]:
    """Estimate transition matrices for WAIT and THIN actions.
    
    WAIT: Grow one year
    THIN: Remove 25% of BA from below, then grow one year
    """
    pmrc = PMRCModel(region="ucp")
    stoch = StochasticPMRC.from_config(pmrc, risk_profile.noise, risk_profile.disturbance)
    discretizer = BuongiornoDiscretizer(config, pmrc)
    
    n_stand = config.n_stand_states  # 40
    rng = np.random.default_rng(seed)
    
    P = {
        Action.WAIT: np.zeros((n_stand, n_stand)),
        Action.THIN: np.zeros((n_stand, n_stand)),
    }
    
    for s_idx in range(n_stand):
        # Decode state
        disturbed = DisturbanceState(s_idx // (ForestState.N_BA * ForestState.N_TPA))
        remainder = s_idx % (ForestState.N_BA * ForestState.N_TPA)
        tpa = TPALevel(remainder // ForestState.N_BA)
        ba = BALevel(remainder % ForestState.N_BA)
        
        discrete_state = ForestState(
            ba=ba, tpa=tpa, disturbed=disturbed, price=PriceState.MEDIUM
        )
        cont_state = discretizer.get_representative_continuous_state(discrete_state)
        
        for action in [Action.WAIT, Action.THIN]:
            for _ in range(n_samples):
                current = cont_state
                
                # Apply THIN action: remove 25% of BA from below
                if action == Action.THIN and current.ba > 0:
                    target_ba = current.ba * 0.75  # Keep 75%
                    current, _ = thin_to_residual_ba_smallest_first(current, target_ba)
                
                # Simulate one year of growth
                next_cont, dist_label, _, _ = stoch.sample_next_state_with_trace(
                    current, dt=1.0, rng=rng
                )
                
                # Disturbance state logic
                if disturbed == DisturbanceState.RECENTLY_DISTURBED:
                    next_disturbed = rng.random() < 0.5  # 50% recovery
                else:
                    next_disturbed = dist_label in ("mild", "severe")
                
                # Discretize
                next_discrete = discretizer.discretize(
                    next_cont, disturbed=next_disturbed, price=PriceState.MEDIUM
                )
                next_stand_idx = next_discrete.to_index() % ForestState.N_STAND
                P[action][s_idx, next_stand_idx] += 1.0
    
    # Normalize
    for action in P:
        row_sums = P[action].sum(axis=1, keepdims=True)
        nonzero = row_sums[:, 0] > 0
        P[action][nonzero] /= row_sums[nonzero]
    
    return P


def build_full_transition_matrix(
    stand_P: dict[Action, np.ndarray],
    price_transition: np.ndarray,
    n_stand: int,
    n_price: int,
) -> dict[Action, np.ndarray]:
    """Build full transition matrix (stand × price) via Kronecker product."""
    full_P = {}
    for action, P_stand in stand_P.items():
        full_P[action] = np.kron(P_stand, price_transition)
    return full_P


# =============================================================================
# Finite Horizon Backward Induction
# =============================================================================

def solve_finite_horizon_mdp(
    P: dict[Action, np.ndarray],
    reward_fn,
    terminal_reward_fn,
    gamma: float,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve finite horizon MDP via backward induction.
    
    Returns:
        V: Value function V[t, s] for each time step
        Q: Action-value Q[t, s, a]
        policy: Optimal policy[t, s]
    """
    actions = sorted(P.keys())
    n_states = P[actions[0]].shape[0]
    n_actions = len(actions)
    
    V = np.zeros((horizon + 1, n_states))
    Q = np.zeros((horizon + 1, n_states, n_actions))
    policy = np.zeros((horizon + 1, n_states), dtype=int)
    
    # Terminal values (mandatory harvest at T=30)
    for s in range(n_states):
        V[horizon, s] = terminal_reward_fn(s)
    
    # Backward induction
    for t in range(horizon - 1, -1, -1):
        for s in range(n_states):
            for ai, a in enumerate(actions):
                q_sa = 0.0
                for s_next in range(n_states):
                    if P[a][s, s_next] > 0:
                        r = reward_fn(s, a, s_next)
                        q_sa += P[a][s, s_next] * (r + gamma * V[t + 1, s_next])
                Q[t, s, ai] = q_sa
            
            V[t, s] = np.max(Q[t, s])
            policy[t, s] = actions[np.argmax(Q[t, s])]
    
    return V, Q, policy


# =============================================================================
# Trajectory Simulation
# =============================================================================

def simulate_trajectory(
    policy: np.ndarray,
    config: BuongiornoConfig,
    risk_profile: RiskProfile,
    horizon: int,
    seed: int,
) -> dict:
    """Simulate a single trajectory following the optimal policy."""
    pmrc = PMRCModel(region="ucp")
    stoch = StochasticPMRC.from_config(pmrc, risk_profile.noise, risk_profile.disturbance)
    discretizer = BuongiornoDiscretizer(config, pmrc)
    rng = np.random.default_rng(seed)
    
    # Initial state: young dense stand
    k, m = pmrc.k, pmrc.m
    age = 1.0
    hd = config.si25 * ((1 - np.exp(-k * age)) / (1 - np.exp(-k * 25.0))) ** m
    state = StandState(
        age=age, hd=hd, tpa=config.initial_tpa, ba=5.0,
        si25=config.si25, region="ucp"
    )
    
    trajectory = {
        'age': [age],
        'ba': [state.ba],
        'tpa': [state.tpa],
        'hd': [state.hd],
        'actions': [],
        'disturbed': [False],
    }
    
    disturbed = False
    price = PriceState.MEDIUM
    
    for t in range(horizon):
        # Discretize current state
        discrete = discretizer.discretize(state, disturbed=disturbed, price=price)
        s_idx = discrete.to_index()
        
        # Get action from policy
        action = Action(policy[t, s_idx])
        trajectory['actions'].append(action.name)
        
        # Apply action
        if action == Action.THIN and state.ba > 0:
            target_ba = state.ba * 0.75
            state, _ = thin_to_residual_ba_smallest_first(state, target_ba)
        
        # Simulate growth
        state, dist_label, _, _ = stoch.sample_next_state_with_trace(state, dt=1.0, rng=rng)
        
        # Update disturbance state
        if disturbed:
            disturbed = rng.random() < 0.5
        else:
            disturbed = dist_label in ("mild", "severe")
        
        # Update price (uniform random)
        price = PriceState(rng.integers(0, 3))
        
        trajectory['age'].append(state.age)
        trajectory['ba'].append(state.ba)
        trajectory['tpa'].append(state.tpa)
        trajectory['hd'].append(state.hd)
        trajectory['disturbed'].append(disturbed)
    
    return trajectory


def simulate_trajectories(
    policy: np.ndarray,
    config: BuongiornoConfig,
    risk_profile: RiskProfile,
    horizon: int,
    n_trajectories: int = 100,
    base_seed: int = 42,
) -> dict:
    """Simulate multiple trajectories and compute statistics."""
    all_ba = []
    all_tpa = []
    all_hd = []
    
    for i in range(n_trajectories):
        traj = simulate_trajectory(policy, config, risk_profile, horizon, seed=base_seed + i)
        all_ba.append(traj['ba'])
        all_tpa.append(traj['tpa'])
        all_hd.append(traj['hd'])
    
    all_ba = np.array(all_ba)
    all_tpa = np.array(all_tpa)
    all_hd = np.array(all_hd)
    
    return {
        'ages': np.arange(horizon + 1) + 1,  # 1 to 31
        'ba_mean': all_ba.mean(axis=0),
        'ba_std': all_ba.std(axis=0),
        'ba_p10': np.percentile(all_ba, 10, axis=0),
        'ba_p90': np.percentile(all_ba, 90, axis=0),
        'tpa_mean': all_tpa.mean(axis=0),
        'tpa_std': all_tpa.std(axis=0),
        'hd_mean': all_hd.mean(axis=0),
        'hd_std': all_hd.std(axis=0),
    }


# =============================================================================
# State Name Helper
# =============================================================================

def state_name(idx: int, include_price: bool = False) -> str:
    """Convert state index to readable name."""
    state = ForestState.from_index(idx)
    ba_str = ['VL', 'L', 'M', 'H', 'VH'][state.ba]
    tpa_str = ['L', 'M', 'H', 'VH'][state.tpa]
    d = 'D' if state.disturbed else 'N'
    if include_price:
        p = ['$L', '$M', '$H'][state.price]
        return f'{ba_str}{tpa_str}{d}{p}'
    return f'{ba_str}{tpa_str}{d}'


# =============================================================================
# Main
# =============================================================================

def main():
    config = BuongiornoConfig()
    pmrc = PMRCModel(region="ucp")
    
    print("=" * 70)
    print("FINITE HORIZON MDP SOLUTION")
    print("=" * 70)
    print(f"State space: {config.n_states} states (40 stand × 3 price)")
    print(f"Actions: WAIT, THIN (HARVEST is terminal only at year 30)")
    print(f"Horizon: 30 years")
    print(f"Discount rate: {config.discount_rate} (gamma = {config.gamma:.3f})")
    print()
    
    # Create output directory
    output_dir = Path("data/mdp_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Reward function
    reward_fn = make_reward_function(config, pmrc)
    
    # Terminal reward: harvest at end of rotation
    def terminal_reward(s):
        return reward_fn(s, Action.WAIT, 0)  # Use WAIT reward (0) + terminal harvest value
    
    # Actually compute terminal harvest value
    discretizer = BuongiornoDiscretizer(config, pmrc)
    from core.products import estimate_product_distribution, compute_harvest_value
    
    def terminal_harvest_reward(s):
        state = ForestState.from_index(s)
        price_mult = config.price_levels[state.price]
        disturbed_mult = config.disturbed_harvest_multiplier if state.disturbed else 1.0
        cont_state = discretizer.get_representative_continuous_state(state)
        
        if cont_state.ba <= 0 or cont_state.tpa <= 0:
            return 0.0
        
        products = estimate_product_distribution(
            pmrc, cont_state.ba, cont_state.tpa, cont_state.hd, region="ucp"
        )
        return compute_harvest_value(products) * price_mult * disturbed_mult
    
    results = {}
    
    for risk_level in ['low', 'medium', 'high']:
        print(f"\n{'='*70}")
        print(f"RISK LEVEL: {risk_level.upper()}")
        print(f"{'='*70}")
        
        profile = get_risk_profile(risk_level)
        
        # Estimate transitions
        print("Estimating transition matrices...")
        stand_P = estimate_transitions_wait_thin(config, profile, n_samples=500, seed=42)
        
        # Build full transition matrix
        full_P = build_full_transition_matrix(
            stand_P, config.price_transition,
            config.n_stand_states, config.n_price_states
        )
        
        # Save transition matrix (stand states only, WAIT action)
        stand_names = [state_name(i) for i in range(40)]
        df_wait = pd.DataFrame(stand_P[Action.WAIT], index=stand_names, columns=stand_names)
        df_wait.to_csv(output_dir / f"P_wait_{risk_level}.csv")
        
        df_thin = pd.DataFrame(stand_P[Action.THIN], index=stand_names, columns=stand_names)
        df_thin.to_csv(output_dir / f"P_thin_{risk_level}.csv")
        print(f"  Saved: {output_dir}/P_wait_{risk_level}.csv")
        print(f"  Saved: {output_dir}/P_thin_{risk_level}.csv")
        
        # Solve MDP
        print("Solving finite horizon MDP...")
        V, Q, policy = solve_finite_horizon_mdp(
            full_P, reward_fn, terminal_harvest_reward,
            gamma=config.gamma, horizon=30
        )
        
        # Save policy
        policy_df = pd.DataFrame(
            policy,
            index=[f"t={t}" for t in range(31)],
            columns=[state_name(s, include_price=True) for s in range(120)]
        )
        policy_df.to_csv(output_dir / f"policy_{risk_level}.csv")
        print(f"  Saved: {output_dir}/policy_{risk_level}.csv")
        
        # Simulate trajectories
        print("Simulating trajectories...")
        stats = simulate_trajectories(policy, config, profile, horizon=30, n_trajectories=100)
        
        results[risk_level] = {
            'V': V,
            'Q': Q,
            'policy': policy,
            'stats': stats,
            'stand_P': stand_P,
        }
        
        # Print policy summary
        print("\nPolicy Summary (at t=0, medium price, normal states):")
        print("  State      | Action | Value")
        print("  " + "-" * 35)
        for ba in BALevel:
            for tpa in [TPALevel.VERY_HIGH, TPALevel.HIGH]:
                s = ForestState(ba=ba, tpa=tpa, disturbed=DisturbanceState.NORMAL, price=PriceState.MEDIUM)
                idx = s.to_index()
                action = "THIN" if policy[0, idx] == 1 else "WAIT"
                value = V[0, idx]
                print(f"  {state_name(idx):10} | {action:6} | ${value:,.0f}")
    
    # ==========================================================================
    # DELIVERABLE 1: Transition probability table
    # ==========================================================================
    print("\n" + "=" * 70)
    print("DELIVERABLE 1: Transition Probability Tables")
    print("=" * 70)
    print(f"Saved to {output_dir}/P_wait_*.csv and P_thin_*.csv")
    
    # ==========================================================================
    # DELIVERABLE 2: Policy representation
    # ==========================================================================
    print("\n" + "=" * 70)
    print("DELIVERABLE 2: Policy Representation")
    print("=" * 70)
    
    # Create policy heatmap with discrete colors
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 10))
    
    # Discrete colormap: red=WAIT, green=THIN
    cmap = ListedColormap(['#d62728', '#2ca02c'])  # red, green
    
    for ax, risk_level in zip(axes, ['low', 'medium', 'high'], strict=False):
        policy = results[risk_level]['policy']
        
        # Extract policy for medium price, normal states (first 20 stand states)
        policy_subset = policy[:, 40:60]  # Medium price states
        
        ax.imshow(policy_subset.T, aspect='auto', cmap=cmap, vmin=0, vmax=1)
        ax.set_xlabel('Year', fontsize=11)
        ax.set_ylabel('State', fontsize=11)
        ax.set_title(f'{risk_level.upper()} Risk', fontsize=13)
        
        # Y-axis labels
        yticks = list(range(0, 20, 2))
        ax.set_yticks(yticks)
        ax.set_yticklabels([state_name(i) for i in yticks], fontsize=9)
        ax.tick_params(axis='x', labelsize=9)
    
    # Add discrete legend instead of colorbar
    legend_elements = [
        Patch(facecolor='#d62728', label='WAIT'),
        Patch(facecolor='#2ca02c', label='THIN'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=12,
               frameon=True, bbox_to_anchor=(0.5, 0.02))
    
    fig.suptitle('Optimal Policy by Risk Level\n(Medium Price, Normal States)', fontsize=14)
    plt.subplots_adjust(bottom=0.12, top=0.88, wspace=0.25)
    plt.savefig(output_dir / "policy_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/policy_heatmap.png")
    
    # ==========================================================================
    # DELIVERABLE 3: Sample trajectory means
    # ==========================================================================
    print("\n" + "=" * 70)
    print("DELIVERABLE 3: Sample Trajectory Means")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    colors = {'low': 'green', 'medium': 'orange', 'high': 'red'}
    
    for risk_level in ['low', 'medium', 'high']:
        stats = results[risk_level]['stats']
        color = colors[risk_level]
        
        # BA (no fill_between)
        axes[0, 0].plot(stats['ages'], stats['ba_mean'], color=color, label=risk_level)
        
        # TPA
        axes[0, 1].plot(stats['ages'], stats['tpa_mean'], color=color, label=risk_level)
        
        # HD
        axes[1, 0].plot(stats['ages'], stats['hd_mean'], color=color, label=risk_level)
    
    axes[0, 0].set_xlabel('Age (years)')
    axes[0, 0].set_ylabel('Basal Area (ft²/ac)')
    axes[0, 0].set_title('Mean BA by Risk Level')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Age (years)')
    axes[0, 1].set_ylabel('Trees per Acre')
    axes[0, 1].set_title('Mean TPA by Risk Level')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Age (years)')
    axes[1, 0].set_ylabel('Dominant Height (ft)')
    axes[1, 0].set_title('Mean HD by Risk Level')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Value function
    for risk_level in ['low', 'medium', 'high']:
        V = results[risk_level]['V']
        # Average value across states at each time
        mean_V = V.mean(axis=1)
        axes[1, 1].plot(range(31), mean_V, color=colors[risk_level], label=risk_level)
    
    axes[1, 1].set_xlabel('Year')
    axes[1, 1].set_ylabel('Expected Value ($)')
    axes[1, 1].set_title('Mean Value Function by Risk Level')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "trajectory_means.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/trajectory_means.png")
    
    # ==========================================================================
    # DELIVERABLE 4: Deterministic vs Stochastic plot
    # ==========================================================================
    print("\n" + "=" * 70)
    print("DELIVERABLE 4: Deterministic vs Stochastic Plot")
    print("=" * 70)
    
    # Simulate deterministic trajectory (pmrc already defined at top of main)
    
    det_ages = []
    det_ba = []
    det_tpa = []
    det_hd = []
    
    age = 1.0
    k, m = pmrc.k, pmrc.m
    hd = config.si25 * ((1 - np.exp(-k * age)) / (1 - np.exp(-k * 25.0))) ** m
    tpa = config.initial_tpa
    ba = pmrc.ba_predict(age, tpa, hd, region="ucp")
    
    for _ in range(31):
        det_ages.append(age)
        det_ba.append(ba)
        det_tpa.append(tpa)
        det_hd.append(hd)
        
        # Grow one year (deterministic)
        next_age = age + 1
        next_hd = config.si25 * ((1 - np.exp(-k * next_age)) / (1 - np.exp(-k * 25.0))) ** m
        # Use PMRC tpa_project for proper mortality modeling
        tpa = pmrc.tpa_project(tpa, config.si25, age, next_age)
        age = next_age
        hd = next_hd
        ba = pmrc.ba_predict(age, tpa, hd, region="ucp")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot deterministic
    axes[0, 0].plot(det_ages, det_ba, 'k-', linewidth=2, label='Deterministic')
    axes[0, 1].plot(det_ages, det_tpa, 'k-', linewidth=2, label='Deterministic')
    axes[1, 0].plot(det_ages, det_hd, 'k-', linewidth=2, label='Deterministic')
    
    # Plot stochastic for each risk level
    for risk_level in ['low', 'medium', 'high']:
        stats = results[risk_level]['stats']
        color = colors[risk_level]
        
        axes[0, 0].plot(stats['ages'], stats['ba_mean'], color=color, 
                        linestyle='--', label=f'Stochastic ({risk_level})')
        
        axes[0, 1].plot(stats['ages'], stats['tpa_mean'], color=color, 
                        linestyle='--', label=f'Stochastic ({risk_level})')
        
        axes[1, 0].plot(stats['ages'], stats['hd_mean'], color=color, 
                        linestyle='--', label=f'Stochastic ({risk_level})')
    
    axes[0, 0].set_xlabel('Age (years)')
    axes[0, 0].set_ylabel('Basal Area (ft²/ac)')
    axes[0, 0].set_title('Basal Area: Deterministic vs Stochastic')
    axes[0, 0].legend(loc='upper left')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Age (years)')
    axes[0, 1].set_ylabel('Trees per Acre')
    axes[0, 1].set_title('TPA: Deterministic vs Stochastic')
    axes[0, 1].legend(loc='upper right')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Age (years)')
    axes[1, 0].set_ylabel('Dominant Height (ft)')
    axes[1, 0].set_title('Dominant Height: Deterministic vs Stochastic')
    axes[1, 0].legend(loc='upper left')
    axes[1, 0].grid(True, alpha=0.3)
    
    # QMD plot
    axes[1, 1].text(0.5, 0.5, 'MDP Policy Comparison\n(see policy_heatmap.png)', 
                    ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].set_title('Policy Summary')
    axes[1, 1].axis('off')
    
    plt.suptitle('Deterministic vs Stochastic Growth Under Optimal MDP Policy', fontsize=14)
    plt.tight_layout()
    plt.savefig("plots/mdp_deterministic_vs_stochastic.png", dpi=150)
    plt.close()
    print("Saved: plots/mdp_deterministic_vs_stochastic.png")
    
    print("\n" + "=" * 70)
    print("ALL DELIVERABLES COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
