"""Visualize disturbance frequency over a 50-year horizon."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from core.disturbances import (
    CatastrophicDisturbanceGenerator,
    ChronicDisturbanceGenerator,
    GeneralDisturbanceGenerator,
)

HORIZON = 50.0
START_AGE = 5.0
REPLICATES = 1000
OUTPUT_PATH = Path("plots") / "disturbance_frequency.png"


def _event_times(gen: GeneralDisturbanceGenerator, *, start_age: float, horizon: float) -> List[float]:
    times: List[float] = []
    current_age = start_age
    cutoff = start_age + horizon
    while current_age < cutoff:
        event = gen.sample_event(current_age)
        if event.start_age > cutoff:
            break
        times.append(event.start_age - start_age)
        current_age = event.start_age
    return times


def _run_replications(
    gen: GeneralDisturbanceGenerator,
    *,
    start_age: float,
    horizon: float,
    replicates: int,
) -> Tuple[List[int], List[List[float]]]:
    counts: List[int] = []
    timelines: List[List[float]] = []
    for _ in range(replicates):
        events = _event_times(gen, start_age=start_age, horizon=horizon)
        counts.append(len(events))
        timelines.append(events)
    return counts, timelines


def _plot_count_hist(ax, counts: Sequence[int], label: str, color: str) -> None:
    if not counts:
        return
    max_count = max(counts)
    bins = np.arange(0, max_count + 2) - 0.5
    hist, edges = np.histogram(counts, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    ax.bar(centers, hist / len(counts), width=0.8, color=color, alpha=0.8)
    ax.set_xticks(range(max_count + 1))
    ax.set_ylabel("Probability")
    ax.set_xlabel("Events in 50 years")
    ax.set_title(f"{label} disturbance frequency")
    ax.grid(True, linestyle="--", alpha=0.3)


def main() -> None:
    cat_gen = CatastrophicDisturbanceGenerator(mean_interval_years=20.0)
    chronic_gen = ChronicDisturbanceGenerator(mean_interval_years=6.0)

    cat_counts, _ = _run_replications(
        cat_gen, start_age=START_AGE, horizon=HORIZON, replicates=REPLICATES
    )
    chronic_counts, _ = _run_replications(
        chronic_gen, start_age=START_AGE, horizon=HORIZON, replicates=REPLICATES
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=200, sharey=True)
    _plot_count_hist(axes[0], cat_counts, "Catastrophic", "#e74c3c")
    _plot_count_hist(axes[1], chronic_counts, "Chronic", "#27ae60")

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
