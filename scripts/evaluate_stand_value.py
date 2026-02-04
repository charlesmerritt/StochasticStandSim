"""Evaluate stand value under different risk levels.

This script simulates forest stand growth over a 30-year rotation with:
- Deterministic thinning at age 15 when BA > threshold
- Terminal harvest at year 30
- No management decisions (just value estimation)

Deliverables:
1. State transition probability table (CSV)
2. Expected stand value by risk level
3. Sample trajectory means at each risk level
4. Deterministic vs stochastic comparison plot
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from core.mdp import (
    BuongiornoConfig, BuongiornoDiscretizer,
    ForestState, BALevel, TPALevel, DisturbanceState, PriceState,
)
from core.pmrc_model import PMRCModel
from core.stochastic_stand import StochasticPMRC, StandState, thin_to_residual_ba_smallest_first
from core.config import get_risk_profile, RiskProfile
from core.products import estimate_product_distribution, compute_harvest_value


# =============================================================================
# Transition Matrix Estimation (WAIT only, with auto-thin at age 15)
# =============================================================================

def estimate_transitions(
    config: BuongiornoConfig,
    risk_profile: RiskProfile,
    n_samples: int = 500,
    seed: int = 42,
) -> np.ndarray:
    """Estimate transition matrix for stand growth (no actions).
    
    Thinning is applied deterministically at age 15 when BA > threshold.
    """
    pmrc = PMRCModel(region="ucp")
    stoch = StochasticPMRC.from_config(pmrc, risk_profile.noise, risk_profile.disturbance)
    discretizer = BuongiornoDiscretizer(config, pmrc)
    
    n_stand = config.n_stand_states  # 40
    rng = np.random.default_rng(seed)
    
    P = np.zeros((n_stand, n_stand))
    
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
        
        for _ in range(n_samples):
            current = cont_state
            
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
            P[s_idx, next_stand_idx] += 1.0
    
    # Normalize
    row_sums = P.sum(axis=1, keepdims=True)
    nonzero = row_sums[:, 0] > 0
    P[nonzero] /= row_sums[nonzero]
    
    return P


# =============================================================================
# Trajectory Simulation (with deterministic thinning at age 15)
# =============================================================================

def simulate_trajectory(
    config: BuongiornoConfig,
    risk_profile: RiskProfile,
    horizon: int,
    seed: int,
) -> dict:
    """Simulate a single trajectory with deterministic thinning at age 15."""
    pmrc = PMRCModel(region="ucp")
    stoch = StochasticPMRC.from_config(pmrc, risk_profile.noise, risk_profile.disturbance)
    rng = np.random.default_rng(seed)
    
    # Initial state: young dense stand at age 1
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
        'thinned': False,
        'disturbed': [False],
    }
    
    disturbed = False
    thinned = False
    
    for year in range(1, horizon + 1):
        # Deterministic thinning at age 15 when BA > threshold
        if year == 15 and state.ba > config.auto_thin_threshold and not thinned:
            state, _ = thin_to_residual_ba_smallest_first(state, config.auto_thin_target)
            thinned = True
        
        # Simulate growth
        state, dist_label, _, _ = stoch.sample_next_state_with_trace(state, dt=1.0, rng=rng)
        
        # Update disturbance state
        if disturbed:
            disturbed = rng.random() < 0.5
        else:
            disturbed = dist_label in ("mild", "severe")
        
        trajectory['age'].append(state.age)
        trajectory['ba'].append(state.ba)
        trajectory['tpa'].append(state.tpa)
        trajectory['hd'].append(state.hd)
        trajectory['disturbed'].append(disturbed)
    
    trajectory['thinned'] = thinned
    return trajectory


def simulate_trajectories(
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
    thin_count = 0
    
    for i in range(n_trajectories):
        traj = simulate_trajectory(config, risk_profile, horizon, seed=base_seed + i)
        all_ba.append(traj['ba'])
        all_tpa.append(traj['tpa'])
        all_hd.append(traj['hd'])
        if traj['thinned']:
            thin_count += 1
    
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
        'thin_rate': thin_count / n_trajectories,
        'all_ba': all_ba,
        'all_tpa': all_tpa,
        'all_hd': all_hd,
    }


def compute_terminal_value(
    config: BuongiornoConfig,
    pmrc: PMRCModel,
    ba: float,
    tpa: float,
    hd: float,
    disturbed: bool,
    price_mult: float = 1.0,
) -> float:
    """Compute harvest value at end of rotation."""
    if ba <= 0 or tpa <= 0:
        return 0.0
    
    disturbed_mult = config.disturbed_harvest_multiplier if disturbed else 1.0
    
    products = estimate_product_distribution(pmrc, ba, tpa, hd, region="ucp")
    return compute_harvest_value(products) * price_mult * disturbed_mult


# =============================================================================
# State Name Helper
# =============================================================================

def state_name(idx: int) -> str:
    """Convert state index to readable name."""
    disturbed = DisturbanceState(idx // (ForestState.N_BA * ForestState.N_TPA))
    remainder = idx % (ForestState.N_BA * ForestState.N_TPA)
    tpa = TPALevel(remainder // ForestState.N_BA)
    ba = BALevel(remainder % ForestState.N_BA)
    ba_str = ['VL', 'L', 'M', 'H', 'VH'][ba]
    tpa_str = ['L', 'M', 'H', 'VH'][tpa]
    d = 'D' if disturbed else 'N'
    return f'{ba_str}{tpa_str}{d}'


# =============================================================================
# Main
# =============================================================================

def main():
    config = BuongiornoConfig()
    pmrc = PMRCModel(region="ucp")
    
    print("=" * 70)
    print("STAND VALUE ESTIMATION UNDER DIFFERENT RISK LEVELS")
    print("=" * 70)
    print(f"State space: {config.n_states} states (40 stand × 3 price)")
    print(f"Horizon: 30 years")
    print(f"Thinning: Deterministic at age 15 when BA > {config.auto_thin_threshold} ft²/ac")
    print(f"Thinning target: {config.auto_thin_target} ft²/ac")
    print(f"Discount rate: {config.discount_rate} (gamma = {config.gamma:.3f})")
    print()
    
    # Create output directory
    output_dir = Path("data/mdp_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for risk_level in ['low', 'medium', 'high']:
        print(f"\n{'='*70}")
        print(f"RISK LEVEL: {risk_level.upper()}")
        print(f"{'='*70}")
        
        profile = get_risk_profile(risk_level)
        
        # Estimate transitions
        print("Estimating transition matrix...")
        P = estimate_transitions(config, profile, n_samples=500, seed=42)
        
        # Save transition matrix
        stand_names = [state_name(i) for i in range(40)]
        df = pd.DataFrame(P, index=stand_names, columns=stand_names)
        df.to_csv(output_dir / f"P_growth_{risk_level}.csv")
        print(f"  Saved: {output_dir}/P_growth_{risk_level}.csv")
        
        # Simulate trajectories
        print("Simulating trajectories...")
        stats = simulate_trajectories(config, profile, horizon=30, n_trajectories=200)
        
        # Compute terminal values for each trajectory
        terminal_values = []
        for i in range(len(stats['all_ba'])):
            ba = stats['all_ba'][i, -1]
            tpa = stats['all_tpa'][i, -1]
            hd = stats['all_hd'][i, -1]
            value = compute_terminal_value(config, pmrc, ba, tpa, hd, disturbed=False)
            terminal_values.append(value)
        
        terminal_values = np.array(terminal_values)
        
        # Discount to present value
        discount_factor = config.gamma ** 30
        pv_values = terminal_values * discount_factor
        
        results[risk_level] = {
            'stats': stats,
            'terminal_values': terminal_values,
            'pv_values': pv_values,
            'P': P,
        }
        
        print(f"\nResults:")
        print(f"  Thinning rate: {stats['thin_rate']*100:.1f}%")
        print(f"  Mean terminal BA: {stats['ba_mean'][-1]:.1f} ft²/ac")
        print(f"  Mean terminal TPA: {stats['tpa_mean'][-1]:.0f} trees/ac")
        print(f"  Mean terminal value: ${terminal_values.mean():,.0f}")
        print(f"  Std terminal value: ${terminal_values.std():,.0f}")
        print(f"  Mean present value (discounted): ${pv_values.mean():,.0f}")
    
    # ==========================================================================
    # DELIVERABLE 1: Transition probability tables
    # ==========================================================================
    print("\n" + "=" * 70)
    print("DELIVERABLE 1: Transition Probability Tables")
    print("=" * 70)
    print(f"Saved to {output_dir}/P_growth_*.csv")
    
    # ==========================================================================
    # DELIVERABLE 2: Value comparison table
    # ==========================================================================
    print("\n" + "=" * 70)
    print("DELIVERABLE 2: Stand Value by Risk Level")
    print("=" * 70)
    
    print("\n{:<10} {:>15} {:>15} {:>15} {:>12}".format(
        "Risk", "Mean Terminal", "Std Terminal", "Mean PV", "Thin Rate"))
    print("-" * 70)
    for risk_level in ['low', 'medium', 'high']:
        r = results[risk_level]
        print("{:<10} ${:>14,.0f} ${:>14,.0f} ${:>14,.0f} {:>11.1f}%".format(
            risk_level.upper(),
            r['terminal_values'].mean(),
            r['terminal_values'].std(),
            r['pv_values'].mean(),
            r['stats']['thin_rate'] * 100,
        ))
    
    # Save value summary
    value_summary = pd.DataFrame({
        'risk_level': ['low', 'medium', 'high'],
        'mean_terminal_value': [results[r]['terminal_values'].mean() for r in ['low', 'medium', 'high']],
        'std_terminal_value': [results[r]['terminal_values'].std() for r in ['low', 'medium', 'high']],
        'mean_present_value': [results[r]['pv_values'].mean() for r in ['low', 'medium', 'high']],
        'thin_rate': [results[r]['stats']['thin_rate'] for r in ['low', 'medium', 'high']],
    })
    value_summary.to_csv(output_dir / "value_summary.csv", index=False)
    print(f"\nSaved: {output_dir}/value_summary.csv")
    
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
        
        # BA
        axes[0, 0].plot(stats['ages'], stats['ba_mean'], color=color, label=risk_level)
        
        # TPA
        axes[0, 1].plot(stats['ages'], stats['tpa_mean'], color=color, label=risk_level)
        
        # HD
        axes[1, 0].plot(stats['ages'], stats['hd_mean'], color=color, label=risk_level)
    
    # Mark thinning age
    axes[0, 0].axvline(x=15, color='gray', linestyle=':', alpha=0.7, label='Thin age')
    axes[0, 1].axvline(x=15, color='gray', linestyle=':', alpha=0.7)
    axes[1, 0].axvline(x=15, color='gray', linestyle=':', alpha=0.7)
    
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
    
    # Terminal value distribution
    for risk_level in ['low', 'medium', 'high']:
        values = results[risk_level]['terminal_values']
        axes[1, 1].hist(values, bins=20, alpha=0.5, color=colors[risk_level], 
                        label=f'{risk_level} (μ=${values.mean():,.0f})')
    
    axes[1, 1].set_xlabel('Terminal Harvest Value ($)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Terminal Value Distribution by Risk Level')
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
    
    # Simulate deterministic trajectory
    det_ages = []
    det_ba = []
    det_tpa = []
    det_hd = []
    
    age = 1.0
    k, m = pmrc.k, pmrc.m
    hd = config.si25 * ((1 - np.exp(-k * age)) / (1 - np.exp(-k * 25.0))) ** m
    tpa = config.initial_tpa
    ba = pmrc.ba_predict(age, tpa, hd, region="ucp")
    
    for year in range(32):
        det_ages.append(age)
        det_ba.append(ba)
        det_tpa.append(tpa)
        det_hd.append(hd)
        
        # Deterministic thinning at age 15
        if year == 14 and ba > config.auto_thin_threshold:
            # Approximate thinning effect
            ba = config.auto_thin_target
            tpa = tpa * (config.auto_thin_target / ba) if ba > 0 else tpa
        
        # Grow one year (deterministic)
        next_age = age + 1
        next_hd = config.si25 * ((1 - np.exp(-k * next_age)) / (1 - np.exp(-k * 25.0))) ** m
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
    
    # Mark thinning age
    for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        ax.axvline(x=15, color='gray', linestyle=':', alpha=0.7)
    
    axes[0, 0].set_xlabel('Age (years)')
    axes[0, 0].set_ylabel('Basal Area (ft²/ac)')
    axes[0, 0].set_title('Basal Area: Deterministic vs Stochastic')
    axes[0, 0].legend(loc='upper left', fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Age (years)')
    axes[0, 1].set_ylabel('Trees per Acre')
    axes[0, 1].set_title('TPA: Deterministic vs Stochastic')
    axes[0, 1].legend(loc='upper right', fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Age (years)')
    axes[1, 0].set_ylabel('Dominant Height (ft)')
    axes[1, 0].set_title('Dominant Height: Deterministic vs Stochastic')
    axes[1, 0].legend(loc='upper left', fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Value comparison bar chart
    risk_levels = ['low', 'medium', 'high']
    mean_values = [results[r]['terminal_values'].mean() for r in risk_levels]
    std_values = [results[r]['terminal_values'].std() for r in risk_levels]
    
    x = np.arange(len(risk_levels))
    bars = axes[1, 1].bar(x, mean_values, yerr=std_values, capsize=5,
                          color=[colors[r] for r in risk_levels], alpha=0.7)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([r.upper() for r in risk_levels])
    axes[1, 1].set_xlabel('Risk Level')
    axes[1, 1].set_ylabel('Terminal Harvest Value ($)')
    axes[1, 1].set_title('Mean Terminal Value by Risk Level')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, mean_values):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                        f'${val:,.0f}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Stand Growth: Deterministic vs Stochastic\n(Thinning at age 15)', fontsize=14)
    plt.tight_layout()
    plt.savefig("plots/mdp_deterministic_vs_stochastic.png", dpi=150)
    plt.close()
    print("Saved: plots/mdp_deterministic_vs_stochastic.png")
    
    print("\n" + "=" * 70)
    print("ALL DELIVERABLES COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
