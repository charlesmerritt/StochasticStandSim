"""Generate MDP transition matrix heatmaps via Monte Carlo simulation.

This figure shows how transition probabilities are estimated by running
many stochastic simulations and counting state-to-state transitions,
demonstrating the core MDP construction methodology.
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
from core.stochastic_stand import (
    StochasticPMRC,
    StandState,
    StateDiscretizer,
    NoiseConfig,
    estimate_transition_matrix,
)
from core.config import make_risk_profiles


def make_init_state(
    pmrc: PMRCModel,
    si25: float = 60.0,
    tpa0: float = 500.0,
    region: str = "ucp",
) -> StandState:
    """Create initial stand state at age 5."""
    age0 = 5.0
    hd0 = si25 * ((1.0 - np.exp(-pmrc.k * age0)) / (1.0 - np.exp(-pmrc.k * 25.0))) ** pmrc.m
    ba0 = pmrc.ba_predict(age0, tpa0, hd0, region=region)
    return StandState(age=age0, hd=hd0, tpa=tpa0, ba=ba0, si25=si25, region=region)


def create_discretizer() -> StateDiscretizer:
    """Create state discretizer with finer bins for smoother transitions."""
    # Finer age bins (5-year increments) for smoother diagonal
    age_bins = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0])
    # Single TPA bin to reduce state space (focus on age x BA)
    tpa_bins = np.array([50.0, 700.0])
    # Finer BA bins for better resolution
    ba_bins = np.array([0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0])
    return StateDiscretizer(age_bins, tpa_bins, ba_bins)


def make_state_labels(discretizer: StateDiscretizer) -> list[str]:
    """Generate human-readable state labels."""
    labels = []
    age_edges = discretizer.age_bins
    tpa_edges = discretizer.tpa_bins
    ba_edges = discretizer.ba_bins
    n_age = len(age_edges) - 1
    n_tpa = len(tpa_edges) - 1
    n_ba = len(ba_edges) - 1

    for i_age in range(n_age):
        for i_tpa in range(n_tpa):
            for i_ba in range(n_ba):
                age_lo, age_hi = age_edges[i_age], age_edges[i_age + 1]
                ba_lo, ba_hi = ba_edges[i_ba], ba_edges[i_ba + 1]
                labels.append(f"A{int(age_lo)}-{int(age_hi)}\nBA{int(ba_lo)}-{int(ba_hi)}")
    return labels


def plot_transition_matrices(
    matrices: dict[int, np.ndarray],
    discretizer: StateDiscretizer,
    output_path: Path,
) -> None:
    """Plot transition matrices as heatmaps for each action.
    
    Shows age-to-age transition probabilities (marginalized over BA).
    """
    n_actions = len(matrices)
    action_names = {0: "No-op", 1: "Light Thin (20%)", 2: "Heavy Thin (40%)", 3: "Harvest"}

    age_edges = discretizer.age_bins
    n_age = len(age_edges) - 1
    n_ba = len(discretizer.ba_bins) - 1
    n_tpa = len(discretizer.tpa_bins) - 1

    fig, axes = plt.subplots(1, n_actions, figsize=(4.5 * n_actions, 4))
    if n_actions == 1:
        axes = [axes]

    for ax, (action, matrix) in zip(axes, sorted(matrices.items())):
        # Marginalize to age-to-age transitions
        age_matrix = np.zeros((n_age, n_age))
        for i_age_from in range(n_age):
            for i_age_to in range(n_age):
                total = 0.0
                count = 0
                for i_tpa in range(n_tpa):
                    for i_ba in range(n_ba):
                        s_from = i_age_from * (n_tpa * n_ba) + i_tpa * n_ba + i_ba
                        for i_tpa2 in range(n_tpa):
                            for i_ba2 in range(n_ba):
                                s_to = i_age_to * (n_tpa * n_ba) + i_tpa2 * n_ba + i_ba2
                                if s_from < matrix.shape[0] and s_to < matrix.shape[1]:
                                    total += matrix[s_from, s_to]
                                    count += 1
                if count > 0:
                    age_matrix[i_age_from, i_age_to] = total

        im = ax.imshow(age_matrix, cmap="Blues", interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(f"{action_names.get(action, f'Action {action}')}")
        ax.set_xlabel("Next Age Class")
        ax.set_ylabel("Current Age Class")

        # Age class labels
        age_labels = [f"{int(age_edges[i])}-{int(age_edges[i+1])}" for i in range(n_age)]
        ax.set_xticks(range(n_age))
        ax.set_yticks(range(n_age))
        ax.set_xticklabels(age_labels, fontsize=8, rotation=45)
        ax.set_yticklabels(age_labels, fontsize=8)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("P(age'|age,a)")

    fig.suptitle("Age-to-Age Transition Probabilities by Action", y=1.02, fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_single_matrix_annotated(
    matrix: np.ndarray,
    discretizer: StateDiscretizer,
    output_path: Path,
) -> None:
    """Plot a single transition matrix with probability annotations."""
    fig, ax = plt.subplots(figsize=(10, 8))

    n_states = matrix.shape[0]
    im = ax.imshow(matrix, cmap="YlOrRd", interpolation="nearest", vmin=0, vmax=np.max(matrix))

    # Annotate cells with probabilities > 0.01
    for i in range(n_states):
        for j in range(n_states):
            val = matrix[i, j]
            if val > 0.01:
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    ax.set_xlabel("Next State Index", fontsize=11)
    ax.set_ylabel("Current State Index", fontsize=11)
    ax.set_title("Transition Probability Matrix P(s'|s, a=no-op)\nEstimated via Monte Carlo Simulation", fontsize=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Transition Probability")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    pmrc = PMRCModel(region="ucp")
    discretizer = create_discretizer()
    init_state = make_init_state(pmrc)
    rng = np.random.default_rng(42)

    # Medium risk profile
    profiles = make_risk_profiles()
    profile = profiles["medium"]

    stoch_pmrc = StochasticPMRC(
        pmrc,
        noise_config=NoiseConfig(
            ba_std=profile.noise.ba_std,
            tpa_std=profile.noise.tpa_std,
            hd_std=profile.noise.hd_std,
            clip_std=profile.noise.clip_std,
        ),
        use_gaussian_noise=True,
        p_mild=profile.disturbance.chronic_prob_annual,
        severe_mean_interval=profile.disturbance.catastrophic_mean_interval,
    )

    print("Estimating transition matrices via Monte Carlo (this may take a minute)...")
    matrices = estimate_transition_matrix(
        stoch_pmrc,
        discretizer,
        actions=[0, 1, 2, 3],  # Include harvest
        dt=1.0,
        n_mc=5000,
        rng=rng,
        si25=60.0,
        region="ucp",
        init_state=init_state,
        steps=50,
    )

    # Plot all actions side by side
    output_path = Path("plots") / "mdp_transition_matrices.png"
    plot_transition_matrices(matrices, discretizer, output_path)
    print(f"Saved: {output_path}")

    # Plot single annotated matrix
    output_path_single = Path("plots") / "mdp_transition_matrix_annotated.png"
    plot_single_matrix_annotated(matrices[0], discretizer, output_path_single)
    print(f"Saved: {output_path_single}")


if __name__ == "__main__":
    main()
