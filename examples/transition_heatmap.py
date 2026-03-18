"""Generate transition matrix heatmap using stochastic PMRC"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from core.pmrc_model import PMRCModel
from core.stochastic_model import (
    StochasticPMRC,
    StandState,
    StateDiscretizer,
    estimate_transition_matrix,
)


def main() -> None:
    pmrc = PMRCModel(region="ucp")
    stochastic = StochasticPMRC(pmrc)
    age_bins = np.array([0.0, 15.0, 30.0, 45.0, 60.0, 80.0])
    tpa_bins = np.array([0.0, 1e6])  # ignore TPA
    ba_bins = np.array([10.0, 40.0, 70.0, 100.0, 140.0, 200.0])
    discretizer = StateDiscretizer(age_bins, tpa_bins, ba_bins)
    rng = np.random.default_rng(123)
    init_state = StandState(age=1.0, hd=15.0, tpa=800.0, ba=30.0, si25=75.0, region="ucp")
    matrices = estimate_transition_matrix(
        stochastic,
        discretizer,
        actions=[0],
        dt=5.0,
        n_mc=1000,
        rng=rng,
        si25=75.0,
        region="ucp",
        init_state=init_state,
        steps=100,
    )
    matrix = matrices[0]
    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
    im = ax.imshow(matrix, cmap="viridis", interpolation="nearest")
    ax.set_title("Transition probabilities (action=0)")
    ax.set_xlabel("Next state index")
    ax.set_ylabel("Current state index")
    fig.colorbar(im, ax=ax, label="Probability")
    Path("plots").mkdir(exist_ok=True)
    fig.savefig("plots/transition_matrix.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
