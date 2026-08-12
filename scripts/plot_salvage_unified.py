"""Produce a unified salvage sensitivity figure.

Loads raw NPV arrays from both salvage experiments (p75 and median)
and plots 3 groups × 3 boxes: no salvage, salvage at p50, salvage at p75.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]

P75_DIR = REPO_ROOT / "data" / "salvage_sensitivity"
P50_DIR = REPO_ROOT / "data" / "salvage_sensitivity_median"
OUT_DIR = REPO_ROOT / "data" / "salvage_sensitivity" / "figures"

DIST_LABELS = ["dist_30", "dist_20", "dist_10"]
DIST_DISPLAY = {
    "dist_30": r"$p_{dist}$ = 1/30",
    "dist_20": r"$p_{dist}$ = 1/20",
    "dist_10": r"$p_{dist}$ = 1/10",
}

# Load thresholds from manifests
with open(P75_DIR / "manifest.json") as f:
    p75_thresh = json.load(f)["salvage_severity_threshold"]
with open(P50_DIR / "manifest.json") as f:
    p50_thresh = json.load(f)["salvage_severity_threshold"]
with open(P75_DIR / "manifest.json") as f:
    det_npv = json.load(f)["deterministic_npv"]


def load_npvs(directory: Path, scenario: str) -> np.ndarray:
    data = np.load(directory / "raw_arrays" / f"{scenario}.npz")
    return data["npvs"]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect data: for each disturbance regime, 3 conditions
    all_data = []       # list of arrays
    positions = []      # x positions
    colors = []         # box colors
    tick_positions = []
    tick_labels = []

    COLOR_NONE = "#4C72B0"
    COLOR_P50 = "#55A868"
    COLOR_P75 = "#DD8452"

    group_width = 4  # spacing between groups

    for gi, dist in enumerate(DIST_LABELS):
        base_x = gi * group_width

        # No salvage (same in both experiments — identical seed & config)
        npv_none = load_npvs(P75_DIR, dist)
        # p50 salvage
        npv_p50 = load_npvs(P50_DIR, f"{dist}_salvage")
        # p75 salvage
        npv_p75 = load_npvs(P75_DIR, f"{dist}_salvage")

        all_data.extend([npv_none, npv_p50, npv_p75])
        positions.extend([base_x, base_x + 1, base_x + 2])
        colors.extend([COLOR_NONE, COLOR_P50, COLOR_P75])

        tick_positions.append(base_x + 1)
        tick_labels.append(DIST_DISPLAY[dist])

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    bp = ax.boxplot(
        all_data,
        positions=positions,
        widths=0.7,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)

    # Deterministic reference line
    ax.axhline(det_npv, color="gray", linestyle="--", linewidth=1.0, zorder=0)

    # Axes
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=11)
    ax.set_ylabel("NPV ($/acre)", fontsize=12)
    ax.set_title(
        "Salvage Sensitivity: NPV by Disturbance Regime and Severity Threshold",
        fontsize=13,
        fontweight="bold",
    )

    # Legend
    legend_elements = [
        Patch(facecolor=COLOR_NONE, alpha=0.75, edgecolor="black", label="No salvage"),
        Patch(facecolor=COLOR_P50, alpha=0.75, edgecolor="black",
              label=f"Salvage at p50 (q \u2265 {p50_thresh:.3f})"),
        Patch(facecolor=COLOR_P75, alpha=0.75, edgecolor="black",
              label=f"Salvage at p75 (q \u2265 {p75_thresh:.3f})"),
        plt.Line2D([0], [0], color="gray", linestyle="--",
                   label=f"Deterministic NPV (${det_npv:,.0f})"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10,
              framealpha=0.9)

    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_xlim(-0.8, (len(DIST_LABELS) - 1) * group_width + 2.8)

    fig.tight_layout()
    out_path = OUT_DIR / "salvage_npv_unified.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Print summary table
    print(f"\n{'Scenario':<12} {'No Salvage':>12} {'p50 Salvage':>12} {'p75 Salvage':>12}")
    print("-" * 52)
    for gi, dist in enumerate(DIST_LABELS):
        idx = gi * 3
        means = [float(np.mean(all_data[idx + j])) for j in range(3)]
        print(f"{dist:<12} ${means[0]:>10,.0f} ${means[1]:>10,.0f} ${means[2]:>10,.0f}")


if __name__ == "__main__":
    main()
