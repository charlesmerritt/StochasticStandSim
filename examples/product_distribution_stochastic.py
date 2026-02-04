"""Product distribution under stochastic growth with different risk levels.

This script visualizes how the DBH-based product distribution (pulpwood,
chip-n-saw, sawtimber) evolves under stochastic growth with disturbance risk.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from core.config import get_risk_profile
from core.pmrc_model import PMRCModel
from core.products import (
    estimate_product_distribution,
    ProductPrices,
    HarvestCosts,
    compute_harvest_value,
    CUFT_TO_TON,
)
from core.stochastic_stand import StochasticPMRC, StandState


def simulate_stochastic_growth(
    pmrc: PMRCModel,
    risk_level: str,
    si25: float = 60.0,
    tpa0: float = 600.0,
    region: str = "ucp",
    years: int = 30,
    n_trajectories: int = 250,
    seed: int = 42,
) -> dict:
    """Simulate stochastic stand growth and track product distribution."""
    profile = get_risk_profile(risk_level)
    stoch = StochasticPMRC.from_config(pmrc, profile.noise, profile.disturbance)
    
    all_ba = []
    all_tpa = []
    all_hd = []
    all_qmd = []
    all_vol_pulp = []
    all_vol_cns = []
    all_vol_saw = []
    all_harvest_value = []
    
    for traj_idx in range(n_trajectories):
        rng = np.random.default_rng(seed + traj_idx)
        
        # Initialize
        k, m = pmrc.k, pmrc.m
        age = 1.0
        hd = si25 * ((1.0 - np.exp(-k * age)) / (1.0 - np.exp(-k * 25.0))) ** m
        state = StandState(
            age=age, hd=hd, tpa=tpa0, ba=5.0,
            si25=si25, region=region
        )
        
        ba_hist = [state.ba]
        tpa_hist = [state.tpa]
        hd_hist = [state.hd]
        qmd_hist = [pmrc.qmd(state.tpa, state.ba)]
        vol_pulp_hist = []
        vol_cns_hist = []
        vol_saw_hist = []
        harvest_value_hist = []
        
        # Get initial product distribution
        products = estimate_product_distribution(pmrc, state.ba, state.tpa, state.hd, region=region)
        vol_pulp_hist.append(products.vol_pulp)
        vol_cns_hist.append(products.vol_cns)
        vol_saw_hist.append(products.vol_saw)
        harvest_value_hist.append(compute_harvest_value(products))
        
        for year in range(1, years + 1):
            # Grow
            state = stoch.sample_next_state(state, dt=1.0, rng=rng)
            
            ba_hist.append(state.ba)
            tpa_hist.append(state.tpa)
            hd_hist.append(state.hd)
            qmd_hist.append(pmrc.qmd(state.tpa, state.ba))
            
            # Get product distribution
            products = estimate_product_distribution(pmrc, state.ba, state.tpa, state.hd, region=region)
            vol_pulp_hist.append(products.vol_pulp)
            vol_cns_hist.append(products.vol_cns)
            vol_saw_hist.append(products.vol_saw)
            harvest_value_hist.append(compute_harvest_value(products))
        
        all_ba.append(ba_hist)
        all_tpa.append(tpa_hist)
        all_hd.append(hd_hist)
        all_qmd.append(qmd_hist)
        all_vol_pulp.append(vol_pulp_hist)
        all_vol_cns.append(vol_cns_hist)
        all_vol_saw.append(vol_saw_hist)
        all_harvest_value.append(harvest_value_hist)
    
    # Convert to arrays and compute means
    all_ba = np.array(all_ba)
    all_tpa = np.array(all_tpa)
    all_hd = np.array(all_hd)
    all_qmd = np.array(all_qmd)
    all_vol_pulp = np.array(all_vol_pulp)
    all_vol_cns = np.array(all_vol_cns)
    all_vol_saw = np.array(all_vol_saw)
    all_harvest_value = np.array(all_harvest_value)
    
    return {
        "age": np.arange(years + 1) + 1,
        "ba_mean": all_ba.mean(axis=0),
        "tpa_mean": all_tpa.mean(axis=0),
        "hd_mean": all_hd.mean(axis=0),
        "qmd_mean": all_qmd.mean(axis=0),
        "vol_pulp_mean": all_vol_pulp.mean(axis=0),
        "vol_cns_mean": all_vol_cns.mean(axis=0),
        "vol_saw_mean": all_vol_saw.mean(axis=0),
        "harvest_value_mean": all_harvest_value.mean(axis=0),
        "harvest_value_std": all_harvest_value.std(axis=0),
    }


def plot_product_distribution_stochastic(
    results_by_risk: dict,
    output_path: Path,
) -> None:
    """Create visualization of product distribution under stochastic growth."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    risk_colors = {"low": "green", "medium": "orange", "high": "red"}
    
    # Panel 1: BA by risk level
    ax1 = axes[0, 0]
    for risk_level, results in results_by_risk.items():
        ax1.plot(results["age"], results["ba_mean"], 
                 color=risk_colors[risk_level], linewidth=2, label=risk_level.capitalize())
    ax1.set_xlabel("Stand Age (years)")
    ax1.set_ylabel("Basal Area (ft²/ac)")
    ax1.set_title("Mean Basal Area by Risk Level")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: QMD over time with product thresholds
    ax2 = axes[0, 1]
    for risk_level, results in results_by_risk.items():
        ax2.plot(results["age"], results["qmd_mean"],
                 color=risk_colors[risk_level], linewidth=2, label=risk_level.capitalize())
    ax2.axhline(y=6, color="gray", linestyle="--", alpha=0.7, label="Pulp min (6\")")
    ax2.axhline(y=9, color="gray", linestyle=":", alpha=0.7, label="CNS min (9\")")
    ax2.axhline(y=12, color="gray", linestyle="-.", alpha=0.7, label="Saw min (12\")")
    ax2.set_xlabel("Stand Age (years)")
    ax2.set_ylabel("Quadratic Mean Diameter (inches)")
    ax2.set_title("Mean QMD vs Product Thresholds")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 14)
    
    # Panel 3: Volume by product class (stacked area) - LOW RISK
    ax3 = axes[0, 2]
    results = results_by_risk["low"]
    ax3.stackplot(
        results["age"],
        results["vol_pulp_mean"],
        results["vol_cns_mean"],
        results["vol_saw_mean"],
        labels=["Pulpwood", "Chip-n-Saw", "Sawtimber"],
        colors=["#2ecc71", "#f39c12", "#e74c3c"],
        alpha=0.8,
    )
    ax3.set_xlabel("Stand Age (years)")
    ax3.set_ylabel("Volume (cuft/acre)")
    ax3.set_title("Volume by Product (Low Risk)")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Product fractions over time - comparison
    ax4 = axes[1, 0]
    for risk_level, results in results_by_risk.items():
        total_vol = results["vol_pulp_mean"] + results["vol_cns_mean"] + results["vol_saw_mean"]
        saw_frac = np.where(total_vol > 0, results["vol_saw_mean"] / total_vol * 100, 0)
        ax4.plot(results["age"], saw_frac,
                 color=risk_colors[risk_level], linewidth=2, label=risk_level.capitalize())
    ax4.set_xlabel("Stand Age (years)")
    ax4.set_ylabel("Sawtimber % of Volume")
    ax4.set_title("Sawtimber Fraction by Risk Level")
    ax4.set_ylim(0, 100)
    ax4.legend(loc="lower right")
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: Harvest value over time
    ax5 = axes[1, 1]
    for risk_level, results in results_by_risk.items():
        ax5.plot(results["age"], results["harvest_value_mean"],
                 color=risk_colors[risk_level], linewidth=2, label=risk_level.capitalize())
    ax5.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax5.set_xlabel("Stand Age (years)")
    ax5.set_ylabel("Net Harvest Value ($/acre)")
    ax5.set_title("Mean Harvest Value by Risk Level")
    ax5.legend(loc="upper left")
    ax5.grid(True, alpha=0.3)
    
    # Panel 6: Value breakdown by product - HIGH RISK
    ax6 = axes[1, 2]
    results = results_by_risk["high"]
    prices = ProductPrices()
    val_pulp = results["vol_pulp_mean"] * CUFT_TO_TON * prices.pulpwood
    val_cns = results["vol_cns_mean"] * CUFT_TO_TON * prices.chip_n_saw
    val_saw = results["vol_saw_mean"] * CUFT_TO_TON * prices.sawtimber
    
    ax6.stackplot(
        results["age"],
        val_pulp,
        val_cns,
        val_saw,
        labels=["Pulpwood", "Chip-n-Saw", "Sawtimber"],
        colors=["#2ecc71", "#f39c12", "#e74c3c"],
        alpha=0.8,
    )
    ax6.axhline(y=HarvestCosts().total, color="k", linestyle="--", label="Harvest Cost")
    ax6.set_xlabel("Stand Age (years)")
    ax6.set_ylabel("Gross Revenue ($/acre)")
    ax6.set_title("Revenue by Product (High Risk)")
    ax6.legend(loc="upper left", fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    fig.suptitle(
        "Product Distribution Under Stochastic Growth (n=250 trajectories)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    pmrc = PMRCModel(region="ucp")
    
    print("Simulating stochastic growth for each risk level...")
    results_by_risk = {}
    for risk_level in ["low", "medium", "high"]:
        print(f"  {risk_level}...")
        results_by_risk[risk_level] = simulate_stochastic_growth(
            pmrc, risk_level, si25=60.0, tpa0=600.0, years=30, n_trajectories=250
        )
    
    # Create visualization - save to new filename
    output_path = Path("paper/figs") / "product_distribution_stochastic.png"
    plot_product_distribution_stochastic(results_by_risk, output_path)
    
    # Print summary table
    print("\nTerminal Harvest Value at Year 30:")
    print("=" * 50)
    for risk_level in ["low", "medium", "high"]:
        results = results_by_risk[risk_level]
        print(f"  {risk_level.capitalize():8} Risk: ${results['harvest_value_mean'][-1]:,.0f} "
              f"(±${results['harvest_value_std'][-1]:,.0f})")


if __name__ == "__main__":
    main()
