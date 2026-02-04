"""Validate product distribution as stands grow across time.

This script visualizes how the DBH-based product distribution (pulpwood,
chip-n-saw, sawtimber) evolves as a stand ages, following forestry standards.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from core.pmrc_model import PMRCModel
from core.products import (
    estimate_product_distribution,
    ProductPrices,
    HarvestCosts,
    compute_harvest_value,
    CUFT_TO_TON,
)


def simulate_stand_growth(
    pmrc: PMRCModel,
    si25: float = 60.0,
    tpa0: float = 500.0,
    region: str = "ucp",
    years: int = 40,
) -> dict:
    """Simulate deterministic stand growth and track product distribution."""
    # Initialize
    age = 5.0
    hd = si25 * ((1.0 - np.exp(-pmrc.k * age)) / (1.0 - np.exp(-pmrc.k * 25.0))) ** pmrc.m
    ba = pmrc.ba_predict(age, tpa0, hd, region=region)
    tpa = tpa0
    
    results = {
        "age": [],
        "hd": [],
        "ba": [],
        "tpa": [],
        "qmd": [],
        "vol_pulp": [],
        "vol_cns": [],
        "vol_saw": [],
        "vol_total": [],
        "frac_pulp": [],
        "frac_cns": [],
        "frac_saw": [],
        "harvest_value": [],
    }
    
    hd_prev = hd
    for year in range(years):
        current_age = age + year
        
        # Project stand state
        if year > 0:
            hd_new = pmrc.hd_project(current_age - 1, hd_prev, current_age)
            tpa_new = pmrc.tpa_project(tpa, si25, current_age - 1, current_age)
            ba = pmrc.ba_project(
                current_age - 1, tpa, tpa_new, ba, hd_prev, hd_new, current_age, region=region
            )
            tpa = tpa_new
            hd = hd_new
        
        hd_prev = hd
        qmd = pmrc.qmd(tpa, ba)
        
        # Get product distribution
        products = estimate_product_distribution(pmrc, ba, tpa, hd, region=region)
        harvest_val = compute_harvest_value(products)
        
        # Store results
        results["age"].append(current_age)
        results["hd"].append(hd)
        results["ba"].append(ba)
        results["tpa"].append(tpa)
        results["qmd"].append(qmd)
        results["vol_pulp"].append(products.vol_pulp)
        results["vol_cns"].append(products.vol_cns)
        results["vol_saw"].append(products.vol_saw)
        results["vol_total"].append(products.total_vol)
        results["frac_pulp"].append(products.pulp_fraction)
        results["frac_cns"].append(products.cns_fraction)
        results["frac_saw"].append(products.saw_fraction)
        results["harvest_value"].append(harvest_val)
    
    return results


def plot_product_distribution(results: dict, output_path: Path) -> None:
    """Create visualization of product distribution over time."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    ages = results["age"]
    
    # Panel 1: Stand metrics (BA, TPA, HD)
    ax1 = axes[0, 0]
    ax1.plot(ages, results["ba"], "b-", label="BA (ft²/ac)", linewidth=2)
    ax1.set_xlabel("Stand Age (years)")
    ax1.set_ylabel("Basal Area (ft²/ac)", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1_twin = ax1.twinx()
    ax1_twin.plot(ages, results["tpa"], "r--", label="TPA", linewidth=2)
    ax1_twin.set_ylabel("Trees per Acre", color="r")
    ax1_twin.tick_params(axis="y", labelcolor="r")
    ax1.set_title("Stand Development")
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: QMD over time with product thresholds
    ax2 = axes[0, 1]
    ax2.plot(ages, results["qmd"], "k-", linewidth=2, label="QMD")
    ax2.axhline(y=6, color="green", linestyle="--", alpha=0.7, label="Pulp min (6\")")
    ax2.axhline(y=9, color="orange", linestyle="--", alpha=0.7, label="CNS min (9\")")
    ax2.axhline(y=12, color="red", linestyle="--", alpha=0.7, label="Saw min (12\")")
    ax2.set_xlabel("Stand Age (years)")
    ax2.set_ylabel("Quadratic Mean Diameter (inches)")
    ax2.set_title("Diameter Growth vs Product Thresholds")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 16)
    
    # Panel 3: Volume by product class (stacked area)
    ax3 = axes[0, 2]
    ax3.stackplot(
        ages,
        results["vol_pulp"],
        results["vol_cns"],
        results["vol_saw"],
        labels=["Pulpwood", "Chip-n-Saw", "Sawtimber"],
        colors=["#2ecc71", "#f39c12", "#e74c3c"],
        alpha=0.8,
    )
    ax3.set_xlabel("Stand Age (years)")
    ax3.set_ylabel("Volume (cuft/acre)")
    ax3.set_title("Volume by Product Class")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Product fractions over time
    ax4 = axes[1, 0]
    ax4.stackplot(
        ages,
        [f * 100 for f in results["frac_pulp"]],
        [f * 100 for f in results["frac_cns"]],
        [f * 100 for f in results["frac_saw"]],
        labels=["Pulpwood", "Chip-n-Saw", "Sawtimber"],
        colors=["#2ecc71", "#f39c12", "#e74c3c"],
        alpha=0.8,
    )
    ax4.set_xlabel("Stand Age (years)")
    ax4.set_ylabel("Percent of Merchantable Volume")
    ax4.set_title("Product Mix Over Time")
    ax4.set_ylim(0, 100)
    ax4.legend(loc="upper right", fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: Harvest value over time
    ax5 = axes[1, 1]
    ax5.plot(ages, results["harvest_value"], "g-", linewidth=2)
    ax5.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax5.fill_between(
        ages,
        results["harvest_value"],
        0,
        where=[v > 0 for v in results["harvest_value"]],
        color="green",
        alpha=0.3,
        label="Profitable",
    )
    ax5.fill_between(
        ages,
        results["harvest_value"],
        0,
        where=[v <= 0 for v in results["harvest_value"]],
        color="red",
        alpha=0.3,
        label="Unprofitable",
    )
    ax5.set_xlabel("Stand Age (years)")
    ax5.set_ylabel("Net Harvest Value ($/acre)")
    ax5.set_title("Harvest Profitability")
    ax5.legend(loc="lower right", fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # Panel 6: Value breakdown by product
    ax6 = axes[1, 2]
    prices = ProductPrices()
    val_pulp = [v * CUFT_TO_TON * prices.pulpwood for v in results["vol_pulp"]]
    val_cns = [v * CUFT_TO_TON * prices.chip_n_saw for v in results["vol_cns"]]
    val_saw = [v * CUFT_TO_TON * prices.sawtimber for v in results["vol_saw"]]
    
    ax6.stackplot(
        ages,
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
    ax6.set_title("Revenue by Product Class")
    ax6.legend(loc="upper left", fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    fig.suptitle(
        "Product Distribution Validation: SI=60, TPA₀=500, UCP Region",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def print_product_table(results: dict) -> None:
    """Print a table of product distribution at key ages."""
    print("\nProduct Distribution by Age:")
    print("=" * 90)
    print(f"{'Age':>4} {'QMD':>6} {'BA':>6} {'TPA':>5} {'Vol_P':>8} {'Vol_C':>8} {'Vol_S':>8} {'%Pulp':>6} {'%CNS':>5} {'%Saw':>5} {'Value':>8}")
    print("-" * 90)
    
    for i, age in enumerate(results["age"]):
        if age in [5, 10, 15, 20, 25, 30, 35, 40, 45]:
            print(
                f"{age:4.0f} "
                f"{results['qmd'][i]:6.2f} "
                f"{results['ba'][i]:6.1f} "
                f"{results['tpa'][i]:5.0f} "
                f"{results['vol_pulp'][i]:8.0f} "
                f"{results['vol_cns'][i]:8.0f} "
                f"{results['vol_saw'][i]:8.0f} "
                f"{results['frac_pulp'][i]*100:6.1f} "
                f"{results['frac_cns'][i]*100:5.1f} "
                f"{results['frac_saw'][i]*100:5.1f} "
                f"{results['harvest_value'][i]:8.0f}"
            )
    print("=" * 90)


def main():
    pmrc = PMRCModel(region="ucp")
    
    # Simulate stand growth
    results = simulate_stand_growth(pmrc, si25=60.0, tpa0=500.0, years=40)
    
    # Print table
    print_product_table(results)
    
    # Create visualization
    output_path = Path("plots") / "product_distribution.png"
    plot_product_distribution(results, output_path)


if __name__ == "__main__":
    main()
