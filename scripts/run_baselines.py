#!/usr/bin/env python3
"""Run baseline policy comparison across risk profiles.

This script evaluates heuristic management policies under different
stochastic risk scenarios and produces a summary comparison.

Usage:
    python scripts/run_baselines.py [--episodes N] [--steps N] [--seed N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from core.baselines import get_baseline_policies
from core.config import resolve_config, make_risk_profiles, SimConfig
from core.evaluation import evaluate_policies, summary_table
from core.pmrc_model import PMRCModel
from core.stochastic_stand import StandState, StochasticPMRC


def create_stochastic_pmrc(config: SimConfig) -> StochasticPMRC:
    """Create a StochasticPMRC configured for the given risk profile."""
    pmrc = PMRCModel(region=config.region)
    risk = config.risk_profile
    
    # Map risk profile to StochasticPMRC parameters
    # Convert additive noise std to approximate log-scale sigma
    sigma_log_ba = risk.noise.ba_std / 100.0  # rough approximation
    sigma_tpa = risk.noise.tpa_std
    sigma_log_hd = risk.noise.hd_std / 50.0 if risk.noise.hd_std > 0 else None
    
    return StochasticPMRC(
        pmrc,
        sigma_log_ba=sigma_log_ba,
        sigma_tpa=sigma_tpa,
        sigma_log_hd=sigma_log_hd,
        use_binomial_tpa=True,
        p_mild=risk.disturbance.chronic_prob_annual,
        severe_mean_interval=risk.disturbance.catastrophic_mean_interval,
        mild_tpa_multiplier=1.0 - risk.disturbance.chronic_tpa_loss,
        severe_tpa_multiplier=1.0 - risk.disturbance.catastrophic_tpa_loss,
        mild_hd_multiplier=1.0 - risk.disturbance.chronic_ba_loss * 0.5,  # HD less affected
        severe_hd_multiplier=1.0 - risk.disturbance.catastrophic_ba_loss * 0.5,
    )


def create_init_state(config: SimConfig) -> StandState:
    """Create initial stand state from config."""
    pmrc = PMRCModel(region=config.region)
    age = 5.0
    si25 = config.default_si25
    tpa = config.default_tpa0
    
    # Compute HD from site index
    hd = pmrc.hd_from_si(si25, form="projection")
    ba = pmrc.ba_predict(age, tpa, hd, region=config.region)
    
    return StandState(
        age=age,
        hd=hd,
        tpa=tpa,
        ba=ba,
        si25=si25,
        region=config.region,
    )


def run_comparison(
    n_episodes: int = 100,
    max_steps: int = 40,
    seed: int = 42,
) -> None:
    """Run baseline comparison across all risk profiles."""
    
    risk_profiles = make_risk_profiles()
    policies = get_baseline_policies()
    
    print("=" * 90)
    print("BASELINE POLICY COMPARISON")
    print("=" * 90)
    print(f"Episodes per policy: {n_episodes}")
    print(f"Max steps per episode: {max_steps}")
    print(f"Seed: {seed}")
    print()
    
    all_results = {}
    
    for profile_name, profile in risk_profiles.items():
        print(f"\n{'='*90}")
        print(f"RISK PROFILE: {profile_name.upper()}")
        print(f"{'='*90}")
        print(f"  Noise - BA std: {profile.noise.ba_std}, TPA std: {profile.noise.tpa_std}, HD std: {profile.noise.hd_std}")
        print(f"  Disturbance - Chronic prob: {profile.disturbance.chronic_prob_annual:.2%}, Catastrophic interval: {profile.disturbance.catastrophic_mean_interval} years")
        print()
        
        # Create config for this profile
        config = resolve_config(risk_profile_name=profile_name)
        
        # Create simulator and initial state
        stochastic_pmrc = create_stochastic_pmrc(config)
        init_state = create_init_state(config)
        
        print(f"Initial state: age={init_state.age}, BA={init_state.ba:.1f}, TPA={init_state.tpa:.0f}, HD={init_state.hd:.1f}")
        print()
        
        # Evaluate all policies
        results = evaluate_policies(
            policies,
            stochastic_pmrc,
            init_state,
            config,
            n_episodes=n_episodes,
            max_steps=max_steps,
            seed=seed,
        )
        
        all_results[profile_name] = results
        
        # Print summary table
        print(summary_table(results))
        print()
    
    # Cross-profile comparison
    print("\n" + "=" * 90)
    print("CROSS-PROFILE SUMMARY (Mean Returns)")
    print("=" * 90)
    
    # Header
    policy_names = list(policies.keys())
    header = f"{'Policy':<25}" + "".join(f"{p:>15}" for p in risk_profiles.keys())
    print(header)
    print("-" * len(header))
    
    # Rows
    for policy_name in policy_names:
        row = f"{policy_name:<25}"
        for profile_name in risk_profiles.keys():
            mean_ret = all_results[profile_name][policy_name].mean_return
            row += f"{mean_ret:>15.1f}"
        print(row)
    
    # Best policy per profile
    print()
    print("Best policy per risk profile:")
    for profile_name in risk_profiles.keys():
        best = max(all_results[profile_name].items(), key=lambda x: x[1].mean_return)
        print(f"  {profile_name}: {best[0]} (mean return: {best[1].mean_return:.1f})")


def main():
    parser = argparse.ArgumentParser(description="Run baseline policy comparison")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes per policy")
    parser.add_argument("--steps", type=int, default=40, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    run_comparison(
        n_episodes=args.episodes,
        max_steps=args.steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
