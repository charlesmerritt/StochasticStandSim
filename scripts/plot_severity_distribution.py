"""Plot the Beta(3.6, 8.4) severity distribution with annotated thresholds.

Produces a publication-quality figure for the thesis Disturbance Module section.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "paper" / "figs"

# Beta parameters
ALPHA = 3.6
BETA_PARAM = 8.4
MEAN = ALPHA / (ALPHA + BETA_PARAM)
MODE = (ALPHA - 1) / (ALPHA + BETA_PARAM - 2)
MEDIAN = float(beta.ppf(0.50, ALPHA, BETA_PARAM))
P75 = float(beta.ppf(0.75, ALPHA, BETA_PARAM))
P95 = float(beta.ppf(0.95, ALPHA, BETA_PARAM))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    x = np.linspace(0, 0.8, 500)
    pdf = beta.pdf(x, ALPHA, BETA_PARAM)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Main density curve
    ax.plot(x, pdf, color="#2c3e50", linewidth=2.0, label="Beta(3.6, 8.4) PDF")

    # Shade the full distribution lightly
    ax.fill_between(x, pdf, alpha=0.12, color="#2c3e50")

    # Shade p50-p75 region
    mask_p50_p75 = (x >= MEDIAN) & (x <= P75)
    ax.fill_between(x[mask_p50_p75], pdf[mask_p50_p75], alpha=0.25, color="#55A868",
                    label=f"p50–p75 region")

    # Shade above p75
    mask_above_p75 = x >= P75
    ax.fill_between(x[mask_above_p75], pdf[mask_above_p75], alpha=0.30, color="#DD8452",
                    label=f"Above p75 (salvage trigger)")

    # Vertical lines for key quantiles
    line_kwargs = dict(linewidth=1.3, linestyle="--", alpha=0.85)

    ax.axvline(MODE, color="#8e44ad", **line_kwargs)
    ax.axvline(MEAN, color="#2980b9", **line_kwargs)
    ax.axvline(MEDIAN, color="#55A868", **line_kwargs)
    ax.axvline(P75, color="#DD8452", **line_kwargs)
    ax.axvline(P95, color="#c0392b", **line_kwargs)

    # Annotate key quantiles with offset labels and arrows
    top = max(pdf) * 1.02
    annotations = [
        ("Mode",    MODE,   "#8e44ad", (-70, 30)),
        ("Mean",    MEAN,   "#2980b9", (60, 30)),
        ("p50",     MEDIAN, "#55A868", (75, 10)),
        ("p75",     P75,    "#DD8452", (45, 25)),
        ("p95",     P95,    "#c0392b", (0, 30)),
    ]

    for label, xpos, color, offset in annotations:
        ypos = float(beta.pdf(xpos, ALPHA, BETA_PARAM))
        ax.annotate(
            f"{label} = {xpos:.3f}",
            xy=(xpos, ypos),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color=color,
            ha="center",
            va="bottom",
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
        )

    ax.set_xlabel("Disturbance Severity ($q$)", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.set_title(
        r"Disturbance Severity Distribution: Beta($\alpha$=3.6, $\beta$=8.4)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlim(0, 0.75)
    ax.set_ylim(0, top * 1.15)

    # Add text box with summary stats
    textstr = (
        f"Mode = {MODE:.3f}\n"
        f"Mean = {MEAN:.3f}\n"
        f"Median = {MEDIAN:.3f}\n"
        f"Std Dev = {np.sqrt(ALPHA * BETA_PARAM / ((ALPHA + BETA_PARAM)**2 * (ALPHA + BETA_PARAM + 1))):.3f}\n"
        f"p75 = {P75:.3f}\n"
        f"p95 = {P95:.3f}"
    )
    props = dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9)
    ax.text(0.62, max(pdf) * 0.92, textstr, fontsize=9, verticalalignment="top",
            bbox=props, family="monospace")

    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_path = OUT_DIR / "severity_distribution.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
