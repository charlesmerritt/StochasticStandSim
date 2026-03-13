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
        'severe_event_count': 0,
        'reset_event_count': 0,
        'recruitment_total': 0.0,
    }
    
    disturbed = False
    thinned = False
    severe_event_count = 0
    reset_event_count = 0
    recruitment_total = 0.0
    
    for year in range(1, horizon + 1):
        # Deterministic thinning at age 15 when BA > threshold
        if year == 15 and state.ba > config.auto_thin_threshold and not thinned:
            state, _ = thin_to_residual_ba_smallest_first(state, config.auto_thin_target)
            thinned = True
        
        # Simulate growth
        state, dist_label, event_age, trace = stoch.sample_next_state_with_trace(
            state,
            dt=1.0,
            rng=rng,
        )
        recruitment_total += trace.recruitment
        if dist_label == "severe":
            severe_event_count += 1
        if event_age is not None:
            reset_event_count += 1
        
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
    trajectory['severe_event_count'] = severe_event_count
    trajectory['reset_event_count'] = reset_event_count
    trajectory['recruitment_total'] = recruitment_total
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
    severe_event_counts = []
    reset_event_counts = []
    recruitment_totals = []
    
    for i in range(n_trajectories):
        traj = simulate_trajectory(config, risk_profile, horizon, seed=base_seed + i)
        all_ba.append(traj['ba'])
        all_tpa.append(traj['tpa'])
        all_hd.append(traj['hd'])
        severe_event_counts.append(traj['severe_event_count'])
        reset_event_counts.append(traj['reset_event_count'])
        recruitment_totals.append(traj['recruitment_total'])
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
        'severe_event_counts': np.array(severe_event_counts),
        'reset_event_counts': np.array(reset_event_counts),
        'recruitment_totals': np.array(recruitment_totals),
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


def simulate_deterministic_trajectory(
    config: BuongiornoConfig,
    pmrc: PMRCModel,
    horizon: int,
    *,
    apply_thinning: bool,
) -> dict[str, list[float]]:
    """Simulate a deterministic trajectory with optional thinning."""
    age = 1.0
    k, m = pmrc.k, pmrc.m
    hd = config.si25 * ((1 - np.exp(-k * age)) / (1 - np.exp(-k * 25.0))) ** m
    state = StandState(
        age=age,
        hd=hd,
        tpa=config.initial_tpa,
        ba=pmrc.ba_predict(age, config.initial_tpa, hd, region="ucp"),
        si25=config.si25,
        region="ucp",
    )

    ages: list[float] = []
    ba_vals: list[float] = []
    tpa_vals: list[float] = []
    hd_vals: list[float] = []
    thinned = False

    for _ in range(horizon + 1):
        ages.append(state.age)
        ba_vals.append(state.ba)
        tpa_vals.append(state.tpa)
        hd_vals.append(state.hd)

        if len(ages) == horizon + 1:
            break

        # Align with stochastic path convention: evaluate thinning before growth step.
        if apply_thinning and int(round(state.age)) == 15 and state.ba > config.auto_thin_threshold and not thinned:
            state, _ = thin_to_residual_ba_smallest_first(state, config.auto_thin_target)
            thinned = True

        age2 = state.age + 1.0
        hd2 = pmrc.hd_project(state.age, state.hd, age2)
        tpa2 = pmrc.tpa_project(state.tpa, state.si25, state.age, age2)
        ba2 = pmrc.ba_project(
            age1=state.age,
            tpa1=state.tpa,
            tpa2=tpa2,
            ba1=state.ba,
            hd1=state.hd,
            hd2=hd2,
            age2=age2,
            region="ucp",
        )
        state = StandState(
            age=age2,
            hd=hd2,
            tpa=tpa2,
            ba=ba2,
            si25=state.si25,
            region=state.region,
        )

    return {"ages": ages, "ba": ba_vals, "tpa": tpa_vals, "hd": hd_vals}


# =============================================================================
# Main
# =============================================================================

def main():
    config = BuongiornoConfig()
    pmrc = PMRCModel(region="ucp")
    n_trajectories = 1000
    
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
    tpa_diagnostics_rows = []
    
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
        stats = simulate_trajectories(
            config,
            profile,
            horizon=30,
            n_trajectories=n_trajectories,
        )
        
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
        for i in range(len(stats["all_tpa"])):
            tpa_diagnostics_rows.append(
                {
                    "risk_level": risk_level,
                    "trajectory_id": i,
                    "terminal_tpa": float(stats["all_tpa"][i, -1]),
                    "severe_event_count": int(stats["severe_event_counts"][i]),
                    "reset_event_count": int(stats["reset_event_counts"][i]),
                    "recruitment_total": float(stats["recruitment_totals"][i]),
                }
            )
        
        print(f"\nResults:")
        print(f"  Thinning rate: {stats['thin_rate']*100:.1f}%")
        print(f"  Mean terminal BA: {stats['ba_mean'][-1]:.1f} ft²/ac")
        print(f"  Mean terminal TPA: {stats['tpa_mean'][-1]:.0f} trees/ac")
        print(f"  Mean terminal value: ${terminal_values.mean():,.0f}")
        print(f"  Std terminal value: ${terminal_values.std():,.0f}")
        print(f"  Mean present value (discounted): ${pv_values.mean():,.0f}")

    tpa_diagnostics_df = pd.DataFrame(tpa_diagnostics_rows)
    tpa_diagnostics_path = output_dir / "tpa_diagnostics.csv"
    tpa_diagnostics_df.to_csv(tpa_diagnostics_path, index=False)
    print(f"\nSaved: {tpa_diagnostics_path}")

    tpa_diagnostics_summary = (
        tpa_diagnostics_df.groupby("risk_level", as_index=False)
        .agg(
            mean_terminal_tpa=("terminal_tpa", "mean"),
            mean_severe_events=("severe_event_count", "mean"),
            mean_reset_events=("reset_event_count", "mean"),
            mean_recruitment=("recruitment_total", "mean"),
        )
        .sort_values("risk_level")
    )
    tpa_diagnostics_summary_path = output_dir / "tpa_diagnostics_summary.csv"
    tpa_diagnostics_summary.to_csv(tpa_diagnostics_summary_path, index=False)
    print(f"Saved: {tpa_diagnostics_summary_path}")

    print("\nTPA Diagnostics Summary (means by risk):")
    print(
        tpa_diagnostics_summary.to_string(
            index=False,
            formatters={
                "mean_terminal_tpa": "{:.1f}".format,
                "mean_severe_events": "{:.2f}".format,
                "mean_reset_events": "{:.2f}".format,
                "mean_recruitment": "{:.1f}".format,
            },
        )
    )
    
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
    
    fig.suptitle(f'Sample Trajectory Means and Value Distributions (n={n_trajectories})', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_dir / "trajectory_means.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/trajectory_means.png")
    
    # ==========================================================================
    # DELIVERABLE 4: Deterministic vs Stochastic plot
    # ==========================================================================
    print("\n" + "=" * 70)
    print("DELIVERABLE 4: Deterministic vs Stochastic Plot")
    print("=" * 70)
    
    det_thinned = simulate_deterministic_trajectory(
        config,
        pmrc,
        horizon=30,
        apply_thinning=True,
    )
    det_unthinned = simulate_deterministic_trajectory(
        config,
        pmrc,
        horizon=30,
        apply_thinning=False,
    )
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot deterministic baselines: with and without thinning.
    axes[0, 0].plot(
        det_thinned["ages"],
        det_thinned["ba"],
        "k-",
        linewidth=2,
        label="Deterministic (thinning rule active)",
    )
    axes[0, 0].plot(
        det_unthinned["ages"],
        det_unthinned["ba"],
        color="0.45",
        linestyle="-.",
        linewidth=1.8,
        label="Deterministic (unthinned baseline)",
    )
    axes[0, 1].plot(
        det_thinned["ages"],
        det_thinned["tpa"],
        "k-",
        linewidth=2,
        label="Deterministic (thinning rule active)",
    )
    axes[0, 1].plot(
        det_unthinned["ages"],
        det_unthinned["tpa"],
        color="0.45",
        linestyle="-.",
        linewidth=1.8,
        label="Deterministic (unthinned baseline)",
    )
    axes[1, 0].plot(
        det_thinned["ages"],
        det_thinned["hd"],
        "k-",
        linewidth=2,
        label="Deterministic (thinning rule active)",
    )
    axes[1, 0].plot(
        det_unthinned["ages"],
        det_unthinned["hd"],
        color="0.45",
        linestyle="-.",
        linewidth=1.8,
        label="Deterministic (unthinned baseline)",
    )
    
    # Plot stochastic for each risk level
    for risk_level in ['low', 'medium', 'high']:
        stats = results[risk_level]['stats']
        color = colors[risk_level]
        
        axes[0, 0].plot(
            stats['ages'],
            stats['ba_mean'],
            color=color,
            linestyle='--',
            label=f"Stochastic mean ({risk_level}, n={n_trajectories})",
        )
        
        axes[0, 1].plot(
            stats['ages'],
            stats['tpa_mean'],
            color=color,
            linestyle='--',
            label=f"Stochastic mean ({risk_level}, n={n_trajectories})",
        )
        
        axes[1, 0].plot(
            stats['ages'],
            stats['hd_mean'],
            color=color,
            linestyle='--',
            label=f"Stochastic mean ({risk_level}, n={n_trajectories})",
        )
    
    # Mark thinning age.
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
    
    # Value comparison box-and-whisker chart with explicit means.
    risk_levels = ['low', 'medium', 'high']
    terminal_values_by_risk = [results[r]['terminal_values'] for r in risk_levels]
    mean_values = [vals.mean() for vals in terminal_values_by_risk]
    box = axes[1, 1].boxplot(
        terminal_values_by_risk,
        labels=[r.upper() for r in risk_levels],
        patch_artist=True,
        widths=0.6,
    )
    for box_patch, risk_level in zip(box["boxes"], risk_levels):
        box_patch.set_facecolor(colors[risk_level])
        box_patch.set_alpha(0.45)
    x_positions = np.arange(1, len(risk_levels) + 1)
    axes[1, 1].scatter(
        x_positions,
        mean_values,
        marker='D',
        s=36,
        color='black',
        label='Mean',
        zorder=3,
    )
    y_min = min(vals.min() for vals in terminal_values_by_risk)
    y_max = max(vals.max() for vals in terminal_values_by_risk)
    y_offset = max(25.0, 0.015 * (y_max - y_min))
    for x_pos, val in zip(x_positions, mean_values):
        axes[1, 1].text(
            x_pos + 0.05,
            val + y_offset,
            f"${val:,.0f}",
            ha='left',
            va='bottom',
            fontsize=9,
        )
    axes[1, 1].set_xlabel('Risk Level')
    axes[1, 1].set_ylabel('Terminal Harvest Value ($)')
    axes[1, 1].set_title('Terminal Value by Risk Level (Box-and-Whisker)')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.suptitle(
        "Stand Growth: Deterministic vs Stochastic\n"
        f"(Thinning rule: age 15, BA > {config.auto_thin_threshold:.0f} -> "
        f"{config.auto_thin_target:.0f}; stochastic n={n_trajectories})",
        fontsize=14,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("plots/mdp_deterministic_vs_stochastic.png", dpi=150)
    plt.close()
    print("Saved: plots/mdp_deterministic_vs_stochastic.png")
    
    print("\n" + "=" * 70)
    print("ALL DELIVERABLES COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
