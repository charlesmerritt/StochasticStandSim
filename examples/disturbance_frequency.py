"""Disturbance configuration helper for plotting examples.

Defines two generators:
  - CatastrophicDisturbanceGenerator: rare, high-severity resets.
  - ChronicDisturbanceGenerator: frequent, low-severity transition noise.

When executed, it also produces a quick histogram sanity check so we can
verify parameterization visually.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from core.disturbances import CatastrophicDisturbanceGenerator, ChronicDisturbanceGenerator

HORIZON = 50.0
START_AGE = 5.0
REPLICATES = 1000
OUTPUT_PATH = Path("plots") / "disturbance_frequency.png"


def disturbance_config() -> Dict[str, Any]:
    """Return reusable disturbance generator instances/parameters for plots."""
    return {
        "catastrophic": CatastrophicDisturbanceGenerator(mean_interval_years=20.0),
        "chronic": ChronicDisturbanceGenerator(mean_interval_years=6.0, max_loss=0.25, hd_scale=0.1),
    }


def _event_counts(gen, *, start_age: float, horizon: float, replicates: int) -> list[int]:
    counts: list[int] = []
    for _ in range(replicates):
        age = start_age
        cutoff = start_age + horizon
        c = 0
        while age < cutoff:
            ev = gen.sample_event(age, rng=np.random.default_rng())
            if ev.start_age > cutoff:
                break
            c += 1
            age = ev.start_age
        counts.append(c)
    return counts


def main() -> None:
    gens = disturbance_config()
    cat_counts = _event_counts(gens["catastrophic"], start_age=START_AGE, horizon=HORIZON, replicates=REPLICATES)
    chronic_counts = _event_counts(gens["chronic"], start_age=START_AGE, horizon=HORIZON, replicates=REPLICATES)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=200, sharey=True)
    for ax, counts, title, color in [
        (axes[0], cat_counts, "Catastrophic", "#e74c3c"),
        (axes[1], chronic_counts, "Chronic", "#27ae60"),
    ]:
        bins = np.arange(0, max(counts + [0]) + 2) - 0.5
        hist, edges = np.histogram(counts, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.bar(centers, hist / len(counts), width=0.8, color=color, alpha=0.8)
        ax.set_xticks(range(int(centers.max()) + 1))
        ax.set_ylabel("Probability")
        ax.set_xlabel("Events in 50 years")
        ax.set_title(f"{title} disturbance frequency")
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle(
        f"Disturbance occurrence over {int(HORIZON)} years (start age {START_AGE})",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    OUTPUT_PATH.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    print(f"Saved plot to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
