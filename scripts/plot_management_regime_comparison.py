"""Compare NPV distributions for one disturbance scenario under different
management regimes.

Loads pre-computed raw arrays from:
  - data/experiment_results_nothin/  (no thinning, no salvage)
  - data/experiment_results/         (BAT thinning, no salvage)
  - data/salvage_sensitivity/        (no thinning, with salvage)

Produces an overlaid density + boxplot figure saved to paper/figs/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── Configuration ────────────────────────────────────────────────────────────
SCENARIO = "dist_20"          # 1/20 annual disturbance probability, λ=0
METRIC = "npvs"               # field name inside the .npz files

REGIME_SPECS = {
    "No Thin": {
        "path": REPO_ROOT / "data" / "experiment_results_nothin" / "raw_arrays" / f"{SCENARIO}.npz",
        "color": "#e74c3c",
        "det_baseline": REPO_ROOT / "data" / "experiment_results_nothin" / "deterministic_baseline.json",
    },
    "BAT Thin": {
        "path": REPO_ROOT / "data" / "experiment_results" / "raw_arrays" / f"{SCENARIO}.npz",
        "color": "#2980b9",
        "det_baseline": REPO_ROOT / "data" / "experiment_results" / "deterministic_baseline.json",
    },
    "No Thin + Salvage": {
        "path": REPO_ROOT / "data" / "salvage_sensitivity" / "raw_arrays" / f"{SCENARIO}_salvage.npz",
        "color": "#27ae60",
        "det_baseline": None,  # same stand, salvage doesn't change det baseline
    },
}

OUTPUT_PATH = REPO_ROOT / "paper" / "figs" / "management_regime_comparison.png"


# ── Helpers ──────────────────────────────────────────────────────────────────
def _load_npvs(path: Path) -> np.ndarray:
    data = np.load(path)
    return data[METRIC]


def _load_det_npv(path: Path | None) -> float | None:
    if path is None:
        return None
    with open(path) as f:
        return json.load(f)["npv"]


def _cvar(values: np.ndarray, alpha: float = 0.05) -> float:
    """Conditional Value-at-Risk at the alpha level."""
    cutoff = np.percentile(values, alpha * 100)
    tail = values[values <= cutoff]
    return float(np.mean(tail)) if len(tail) > 0 else float(cutoff)


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    # Load data
    regime_data: dict[str, np.ndarray] = {}
    det_baselines: dict[str, float | None] = {}
    colors: dict[str, str] = {}

    for label, spec in REGIME_SPECS.items():
        regime_data[label] = _load_npvs(spec["path"])
        det_baselines[label] = _load_det_npv(spec["det_baseline"])
        colors[label] = spec["color"]

    labels = list(regime_data.keys())

    # ── Figure layout: density on top, boxplot on bottom ─────────────────
    fig = plt.figure(figsize=(10, 7), dpi=200, constrained_layout=False)
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08, bottom=0.15, top=0.92)
    ax_hist = fig.add_subplot(gs[0])
    ax_box = fig.add_subplot(gs[1], sharex=ax_hist)

    # Density histograms (overlaid, semi-transparent)
    all_vals = np.concatenate(list(regime_data.values()))
    lo, hi = np.percentile(all_vals, [0.5, 99.5])
    bins = np.linspace(lo, hi, 60)

    for label in labels:
        vals = regime_data[label]
        ax_hist.hist(
            vals, bins=bins, density=True, alpha=0.45,
            color=colors[label], edgecolor="white", linewidth=0.4,
            label=label,
        )
        # Mean marker
        mean_val = float(np.mean(vals))
        ax_hist.axvline(
            mean_val, color=colors[label], linestyle="--", linewidth=1.6,
            alpha=0.85,
        )

    # Deterministic baselines
    for label in labels:
        det = det_baselines[label]
        if det is not None:
            ax_hist.axvline(
                det, color=colors[label], linestyle=":", linewidth=1.2,
                alpha=0.6,
            )

    ax_hist.set_ylabel("Density", fontsize=11)
    ax_hist.set_title(
        f"NPV Distribution Under Different Management Regimes\n"
        f"Scenario: {SCENARIO} ($p_{{dist}}$ = 1/20, $\\lambda$ = 0)",
        fontsize=13,
    )
    ax_hist.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax_hist.grid(True, linestyle="--", alpha=0.25)
    plt.setp(ax_hist.get_xticklabels(), visible=False)

    # Boxplots (horizontal, matching colors)
    bp_data = [regime_data[lbl] for lbl in labels]
    bp = ax_box.boxplot(
        bp_data, vert=False, patch_artist=True,
        widths=0.6, showfliers=False,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )
    for patch, label in zip(bp["boxes"], labels, strict=True):
        patch.set_facecolor(colors[label])
        patch.set_alpha(0.55)

    ax_box.set_yticklabels(labels, fontsize=10)
    ax_box.set_xlabel("NPV ($/ac)", fontsize=11)
    ax_box.grid(True, linestyle="--", alpha=0.25, axis="x")

    # ── Summary statistics annotation ────────────────────────────────────
    header = f"{'Regime':<22s} {'Mean':>8s} {'Median':>8s} {'P5':>8s} {'P95':>8s} {'CVaR₅':>8s}"
    sep = "─" * len(header)
    stat_lines = [header, sep]
    for label in labels:
        vals = regime_data[label]
        stat_lines.append(
            f"{label:<22s} "
            f"${np.mean(vals):>7,.0f} "
            f"${np.median(vals):>7,.0f} "
            f"${np.percentile(vals, 5):>7,.0f} "
            f"${np.percentile(vals, 95):>7,.0f} "
            f"${_cvar(vals):>7,.0f}"
        )

    fig.text(
        0.50, 0.01,
        "\n".join(stat_lines),
        ha="center", va="bottom", fontsize=8.5,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7f7f7", alpha=0.9),
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {OUTPUT_PATH}")

    # Print table
    print(f"\n{'Regime':<22s} {'Mean':>10s} {'Median':>10s} {'P5':>10s} {'P95':>10s} {'CVaR₅':>10s} {'Std':>10s}")
    print("─" * 84)
    for label in labels:
        vals = regime_data[label]
        print(
            f"{label:<22s} "
            f"${np.mean(vals):>9,.0f} "
            f"${np.median(vals):>9,.0f} "
            f"${np.percentile(vals, 5):>9,.0f} "
            f"${np.percentile(vals, 95):>9,.0f} "
            f"${_cvar(vals):>9,.0f} "
            f"${np.std(vals):>9,.0f}"
        )


if __name__ == "__main__":
    main()
