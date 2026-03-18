"""Growth Model Validation: Trajectory Analysis.

Generates validation plots showing 50 stand trajectories with:
- HD (dominant height) with SI25 reference
- BA (basal area) with thin threshold
- TPA (trees per acre)
- QMD (quadratic mean diameter)

Validates that growth model produces expected behavior:
- HD reaches SI25 at base age 25
- BA and TPA follow expected growth curves
- Thinning applied at year 15 if BA > threshold
"""

import sys
sys.path.insert(0, ".")

import matplotlib.pyplot as plt
import numpy as np

from core.config import get_risk_profile
from core.mdp import BuongiornoConfig
from core.pmrc_model import PMRCModel
from core.stochastic_model import StandState, StochasticPMRC, thin_to_residual_ba_smallest_first


def run_growth_validation(
    n_trajectories: int = 50,
    n_years: int = 35,
    thin_year: int = 15,
    save_path: str = "plots/growth_validation.png",
) -> None:
    """Run growth validation with multiple trajectories."""
    
    config = BuongiornoConfig()
    pmrc = PMRCModel(region=config.region)
    
    # Storage for trajectories
    all_hd = np.zeros((n_trajectories, n_years + 1))
    all_ba = np.zeros((n_trajectories, n_years + 1))
    all_tpa = np.zeros((n_trajectories, n_years + 1))
    all_qmd = np.zeros((n_trajectories, n_years + 1))
    
    # Track disturbance events
    disturbance_years = {i: [] for i in range(n_trajectories)}
    thin_applied = {i: False for i in range(n_trajectories)}
    
    # Run for medium risk
    risk_level = "medium"
    profile = get_risk_profile(risk_level)
    stoch = StochasticPMRC.from_config(pmrc, profile.noise, profile.disturbance)
    
    for traj_idx in range(n_trajectories):
        rng = np.random.default_rng(seed=42 + traj_idx)
        
        # Initial state: young stand
        age0 = 1.0
        hd0 = config.si25 * ((1 - np.exp(-pmrc.k * age0)) / (1 - np.exp(-pmrc.k * 25.0))) ** pmrc.m
        ba0 = pmrc.ba_predict(age0, config.initial_tpa, hd0, config.region)
        
        state = StandState(
            age=age0,
            hd=hd0,
            tpa=config.initial_tpa,
            ba=ba0,
            si25=config.si25,
            region=config.region,
        )
        
        # Record initial state
        all_hd[traj_idx, 0] = state.hd
        all_ba[traj_idx, 0] = state.ba
        all_tpa[traj_idx, 0] = state.tpa
        all_qmd[traj_idx, 0] = pmrc.qmd(state.tpa, state.ba) if state.tpa > 0 and state.ba > 0 else 0
        
        for year in range(1, n_years + 1):
            # Apply thinning ONLY at year 15 if BA > threshold
            if year == thin_year and state.ba > config.auto_thin_threshold:
                state, _ = thin_to_residual_ba_smallest_first(state, config.auto_thin_target)
                thin_applied[traj_idx] = True
            
            # Grow one year
            next_state, dist_label, _, _ = stoch.sample_next_state_with_trace(state, dt=1.0, rng=rng)
            
            if dist_label:
                disturbance_years[traj_idx].append(year)
            
            state = next_state
            
            # Record state
            all_hd[traj_idx, year] = state.hd
            all_ba[traj_idx, year] = state.ba
            all_tpa[traj_idx, year] = state.tpa
            all_qmd[traj_idx, year] = pmrc.qmd(state.tpa, state.ba) if state.tpa > 0 and state.ba > 0 else 0
    
    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"Growth Model Validation: {n_trajectories} Trajectories (Medium Risk)", fontsize=14)
    
    years = np.arange(n_years + 1)
    
    # Plot HD
    ax = axes[0, 0]
    for i in range(n_trajectories):
        ax.plot(years, all_hd[i], alpha=0.3, color="blue", linewidth=0.8)
    ax.plot(years, all_hd.mean(axis=0), color="darkblue", linewidth=2, label="Mean HD")
    ax.axhline(config.si25, color="red", linestyle="--", linewidth=2, label=f"SI25 = {config.si25}")
    ax.axvline(25, color="green", linestyle=":", alpha=0.7, label="Base age 25")
    # Mark HD at age 25
    hd_at_25 = all_hd[:, 25].mean() if n_years >= 25 else np.nan
    if not np.isnan(hd_at_25):
        ax.scatter([25], [hd_at_25], color="green", s=100, zorder=5, marker="o")
        ax.annotate(f"HD@25 = {hd_at_25:.1f}", (25, hd_at_25), xytext=(27, hd_at_25-5),
                   fontsize=10, color="green")
    ax.set_xlabel("Year")
    ax.set_ylabel("Dominant Height (ft)")
    ax.set_title("Dominant Height (HD)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n_years)
    
    # Plot BA
    ax = axes[0, 1]
    for i in range(n_trajectories):
        ax.plot(years, all_ba[i], alpha=0.3, color="green", linewidth=0.8)
    ax.plot(years, all_ba.mean(axis=0), color="darkgreen", linewidth=2, label="Mean BA")
    ax.axhline(config.auto_thin_threshold, color="orange", linestyle="--", linewidth=2, 
              label=f"Thin threshold = {config.auto_thin_threshold}")
    ax.axhline(config.auto_thin_target, color="red", linestyle=":", linewidth=1.5,
              label=f"Thin target = {config.auto_thin_target}")
    ax.axvline(thin_year, color="purple", linestyle="--", alpha=0.7, label=f"Thin year = {thin_year}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Basal Area (ft²/ac)")
    ax.set_title("Basal Area (BA)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n_years)
    
    # Plot TPA
    ax = axes[0, 2]
    for i in range(n_trajectories):
        ax.plot(years, all_tpa[i], alpha=0.3, color="orange", linewidth=0.8)
    ax.plot(years, all_tpa.mean(axis=0), color="darkorange", linewidth=2, label="Mean TPA")
    ax.axvline(thin_year, color="purple", linestyle="--", alpha=0.7, label=f"Thin year = {thin_year}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Trees per Acre")
    ax.set_title("Trees per Acre (TPA)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n_years)
    
    # Plot QMD
    ax = axes[1, 0]
    for i in range(n_trajectories):
        ax.plot(years, all_qmd[i], alpha=0.3, color="purple", linewidth=0.8)
    ax.plot(years, all_qmd.mean(axis=0), color="darkviolet", linewidth=2, label="Mean QMD")
    ax.axvline(thin_year, color="purple", linestyle="--", alpha=0.7, label=f"Thin year = {thin_year}")
    ax.set_xlabel("Year")
    ax.set_ylabel("QMD (inches)")
    ax.set_title("Quadratic Mean Diameter (QMD)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n_years)
    
    # Summary statistics
    ax = axes[1, 1]
    ax.axis("off")
    
    n_thinned = sum(thin_applied.values())
    n_disturbed = sum(1 for d in disturbance_years.values() if d)
    
    summary = f"""
Growth Model Validation Summary
═══════════════════════════════════════

Configuration:
  SI25 = {config.si25} ft
  Initial TPA = {config.initial_tpa}
  Thin threshold = {config.auto_thin_threshold} ft²/ac
  Thin target = {config.auto_thin_target} ft²/ac
  Thin year = {thin_year}

Results at Year 25:
  Mean HD = {all_hd[:, 25].mean():.1f} ft (expected: {config.si25})
  Mean BA = {all_ba[:, 25].mean():.1f} ft²/ac
  Mean TPA = {all_tpa[:, 25].mean():.0f}
  Mean QMD = {all_qmd[:, 25].mean():.2f} in

Results at Year {n_years}:
  Mean HD = {all_hd[:, -1].mean():.1f} ft
  Mean BA = {all_ba[:, -1].mean():.1f} ft²/ac
  Mean TPA = {all_tpa[:, -1].mean():.0f}
  Mean QMD = {all_qmd[:, -1].mean():.2f} in

Events:
  Trajectories thinned: {n_thinned}/{n_trajectories}
  Trajectories disturbed: {n_disturbed}/{n_trajectories}
"""
    ax.text(0.1, 0.5, summary, transform=ax.transAxes, fontsize=10,
           verticalalignment='center', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # HD validation: compare to deterministic
    ax = axes[1, 2]
    
    # Deterministic HD curve
    det_ages = np.linspace(1, n_years, 100)
    det_hd = config.si25 * ((1 - np.exp(-pmrc.k * det_ages)) / (1 - np.exp(-pmrc.k * 25.0))) ** pmrc.m
    
    ax.fill_between(years, all_hd.min(axis=0), all_hd.max(axis=0), alpha=0.2, color="blue", label="HD range")
    ax.plot(years, all_hd.mean(axis=0), color="blue", linewidth=2, label="Mean HD (stochastic)")
    ax.plot(det_ages, det_hd, color="red", linewidth=2, linestyle="--", label="Deterministic HD")
    ax.axhline(config.si25, color="green", linestyle=":", alpha=0.7)
    ax.scatter([25], [config.si25], color="green", s=100, zorder=5, marker="*", label=f"SI25 = {config.si25}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Dominant Height (ft)")
    ax.set_title("HD: Stochastic vs Deterministic")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n_years)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()
    
    # Print validation results
    print("\n" + "=" * 60)
    print("GROWTH VALIDATION RESULTS")
    print("=" * 60)
    print(f"SI25 = {config.si25} ft")
    print(f"Mean HD at age 25 = {all_hd[:, 25].mean():.2f} ft")
    print(f"HD error at age 25 = {abs(all_hd[:, 25].mean() - config.si25):.2f} ft ({abs(all_hd[:, 25].mean() - config.si25)/config.si25*100:.1f}%)")


if __name__ == "__main__":
    run_growth_validation()
