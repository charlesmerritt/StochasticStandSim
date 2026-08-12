"""Produce two comparison figures:

1. Severity distribution comparison: Beta(3.6, 8.4) vs Beta(3, 3)
2. Salvage NPV comparison: moderate vs extreme severity (side-by-side panels)

All figures saved to paper/figs/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.stats import beta

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "paper" / "figs"

# Moderate severity: Beta(3.6, 8.4)
MOD_ALPHA, MOD_BETA = 3.6, 8.4
MOD_P50 = float(beta.ppf(0.50, MOD_ALPHA, MOD_BETA))
MOD_P75 = float(beta.ppf(0.75, MOD_ALPHA, MOD_BETA))
MOD_P95 = float(beta.ppf(0.95, MOD_ALPHA, MOD_BETA))
MOD_MEAN = MOD_ALPHA / (MOD_ALPHA + MOD_BETA)
MOD_MODE = (MOD_ALPHA - 1) / (MOD_ALPHA + MOD_BETA - 2)

# Extreme severity: Beta(3, 3)
EXT_ALPHA, EXT_BETA = 3.0, 3.0
EXT_P50 = float(beta.ppf(0.50, EXT_ALPHA, EXT_BETA))
EXT_P75 = float(beta.ppf(0.75, EXT_ALPHA, EXT_BETA))
EXT_P95 = float(beta.ppf(0.95, EXT_ALPHA, EXT_BETA))
EXT_MEAN = EXT_ALPHA / (EXT_ALPHA + EXT_BETA)
EXT_MODE = (EXT_ALPHA - 1) / (EXT_ALPHA + EXT_BETA - 2)

# Data directories
MOD_P75_DIR = REPO_ROOT / "data" / "salvage_sensitivity"
MOD_P50_DIR = REPO_ROOT / "data" / "salvage_sensitivity_median"
EXT_DIR = REPO_ROOT / "data" / "salvage_sensitivity_extreme"

DIST_LABELS = ["dist_30", "dist_20", "dist_10"]
DIST_DISPLAY = {
    "dist_30": r"$p_{dist}$ = 1/30",
    "dist_20": r"$p_{dist}$ = 1/20",
    "dist_10": r"$p_{dist}$ = 1/10",
}

# Colors
COLOR_NONE = "#4C72B0"
COLOR_P50 = "#55A868"
COLOR_P75 = "#DD8452"


def load_npvs(directory: Path, scenario: str) -> np.ndarray:
    return np.load(directory / "raw_arrays" / f"{scenario}.npz")["npvs"]


def get_det_npv(directory: Path) -> float:
    with open(directory / "manifest.json") as f:
        return json.load(f)["deterministic_npv"]


# =========================================================================
# Figure 1: Severity distribution comparison
# =========================================================================
def plot_severity_comparison() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    x = np.linspace(0, 0.95, 500)

    for ax, (a, b, label, mean, mode, p50, p75, p95) in zip(axes, [
        (MOD_ALPHA, MOD_BETA, "Moderate: Beta(3.6, 8.4)",
         MOD_MEAN, MOD_MODE, MOD_P50, MOD_P75, MOD_P95),
        (EXT_ALPHA, EXT_BETA, "Extreme: Beta(3, 3)",
         EXT_MEAN, EXT_MODE, EXT_P50, EXT_P75, EXT_P95),
    ]):
        pdf = beta.pdf(x, a, b)
        ax.plot(x, pdf, color="#2c3e50", linewidth=2.0)
        ax.fill_between(x, pdf, alpha=0.10, color="#2c3e50")

        # Shade p50-p75
        mask_mid = (x >= p50) & (x <= p75)
        ax.fill_between(x[mask_mid], pdf[mask_mid], alpha=0.30, color=COLOR_P50)

        # Shade above p75
        mask_hi = x >= p75
        ax.fill_between(x[mask_hi], pdf[mask_hi], alpha=0.30, color=COLOR_P75)

        # Dashed lines
        lkw = dict(linewidth=1.2, linestyle="--", alpha=0.7)
        ax.axvline(mode, color="#8e44ad", **lkw)
        ax.axvline(mean, color="#2980b9", **lkw)
        ax.axvline(p50, color=COLOR_P50, **lkw)
        ax.axvline(p75, color=COLOR_P75, **lkw)
        ax.axvline(p95, color="#c0392b", **lkw)

        # Stats box
        std = (a * b / ((a + b) ** 2 * (a + b + 1))) ** 0.5
        textstr = (
            f"Mode  = {mode:.3f}\n"
            f"Mean  = {mean:.3f}\n"
            f"p50   = {p50:.3f}\n"
            f"p75   = {p75:.3f}\n"
            f"p95   = {p95:.3f}\n"
            f"\u03c3     = {std:.3f}"
        )
        props = dict(boxstyle="round,pad=0.4", facecolor="white",
                     edgecolor="gray", alpha=0.9)
        ax.text(0.97, 0.97, textstr, transform=ax.transAxes,
                fontsize=9, verticalalignment="top", horizontalalignment="right",
                bbox=props, family="monospace")

        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Disturbance Severity ($q$)", fontsize=11)
        ax.set_xlim(0, 0.95)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.2, linewidth=0.5)

    axes[0].set_ylabel("Probability Density", fontsize=11)

    # Shared legend
    legend_elements = [
        Patch(facecolor=COLOR_P50, alpha=0.30, label="p50–p75 region"),
        Patch(facecolor=COLOR_P75, alpha=0.30, label="Above p75 (salvage trigger)"),
        plt.Line2D([0], [0], color="#8e44ad", linestyle="--", label="Mode"),
        plt.Line2D([0], [0], color="#2980b9", linestyle="--", label="Mean"),
        plt.Line2D([0], [0], color=COLOR_P50, linestyle="--", label="Median (p50)"),
        plt.Line2D([0], [0], color=COLOR_P75, linestyle="--", label="p75"),
        plt.Line2D([0], [0], color="#c0392b", linestyle="--", label="p95"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=9,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Disturbance Severity Distributions", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)

    out = FIG_DIR / "severity_distribution_comparison.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# =========================================================================
# Figure 2: Salvage NPV comparison — side-by-side panels
# =========================================================================
def plot_salvage_comparison() -> None:
    det_npv = get_det_npv(MOD_P75_DIR)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    panel_data = [
        ("Moderate Severity — Beta(3.6, 8.4)",
         MOD_P75_DIR, MOD_P50_DIR, MOD_P50, MOD_P75),
        ("Extreme Severity — Beta(3, 3)",
         EXT_DIR, EXT_DIR, EXT_P50, EXT_P75),
    ]

    for ax, (title, p75_dir, p50_dir, thresh_p50, thresh_p75) in zip(axes, panel_data):
        all_data = []
        positions = []
        colors = []
        tick_positions = []
        tick_labels = []
        group_width = 4

        for gi, dist in enumerate(DIST_LABELS):
            base_x = gi * group_width

            # No salvage
            npv_none = load_npvs(p75_dir, dist)

            # p50 salvage
            if p50_dir == EXT_DIR:
                npv_p50 = load_npvs(p50_dir, f"{dist}_salvage_p50")
                npv_p75 = load_npvs(p75_dir, f"{dist}_salvage_p75")
            else:
                npv_p50 = load_npvs(p50_dir, f"{dist}_salvage")
                npv_p75 = load_npvs(p75_dir, f"{dist}_salvage")

            all_data.extend([npv_none, npv_p50, npv_p75])
            positions.extend([base_x, base_x + 1, base_x + 2])
            colors.extend([COLOR_NONE, COLOR_P50, COLOR_P75])

            tick_positions.append(base_x + 1)
            ri = {"dist_30": 30, "dist_20": 20, "dist_10": 10}[dist]
            tick_labels.append(f"$p_{{dist}}$ = 1/{ri}")

        bp = ax.boxplot(
            all_data,
            positions=positions,
            widths=0.65,
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

        ax.axhline(det_npv, color="gray", linestyle="--", linewidth=1.0, zorder=0)

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.set_xlim(-0.8, (len(DIST_LABELS) - 1) * group_width + 2.8)

    axes[0].set_ylabel("NPV ($/acre)", fontsize=12)

    # Shared legend
    legend_elements = [
        Patch(facecolor=COLOR_NONE, alpha=0.75, edgecolor="black", label="No salvage"),
        Patch(facecolor=COLOR_P50, alpha=0.75, edgecolor="black", label="Salvage at p50"),
        Patch(facecolor=COLOR_P75, alpha=0.75, edgecolor="black", label="Salvage at p75"),
        plt.Line2D([0], [0], color="gray", linestyle="--",
                   label=f"Deterministic NPV (${det_npv:,.0f})"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=10,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Salvage Sensitivity: Moderate vs Extreme Severity",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.14)

    out = FIG_DIR / "salvage_npv_comparison_panels.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_severity_comparison()
    plot_salvage_comparison()


if __name__ == "__main__":
    main()
