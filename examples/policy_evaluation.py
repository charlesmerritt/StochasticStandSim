"""Evaluate baseline policies under stochastic uncertainty.

This figure shows return distributions and CVaR metrics for different
management policies, demonstrating how risk-sensitive evaluation works.
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
from core.stochastic_stand import StochasticPMRC, StandState, NoiseConfig
from core.config import SimConfig, make_risk_profiles, resolve_config
from core.baselines import get_baseline_policies
from core.evaluation import evaluate_policies, compute_cvar, PolicyEvaluation


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


def plot_return_distributions(
    evaluations: dict[str, PolicyEvaluation],
    output_path: Path,
) -> None:
    """Plot return distributions as violin plots with CVaR markers."""
    policy_names = list(evaluations.keys())
    n_policies = len(policy_names)

    fig, ax = plt.subplots(figsize=(12, 6))

    returns_data = []
    for name in policy_names:
        ev = evaluations[name]
        returns = [r.discounted_return for r in ev.episode_results]
        returns_data.append(returns)

    positions = np.arange(n_policies)
    parts = ax.violinplot(returns_data, positions=positions, showmeans=True, showmedians=True)

    # Color the violins
    colors = plt.cm.tab10(np.linspace(0, 1, n_policies))
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)

    # Add CVaR markers
    for i, name in enumerate(policy_names):
        ev = evaluations[name]
        returns = np.array([r.discounted_return for r in ev.episode_results])
        cvar_5 = compute_cvar(returns, 0.05)
        cvar_10 = compute_cvar(returns, 0.10)

        ax.scatter([i], [cvar_5], marker="v", s=100, c="red", zorder=5, label="CVaR 5%" if i == 0 else "")
        ax.scatter([i], [cvar_10], marker="^", s=100, c="orange", zorder=5, label="CVaR 10%" if i == 0 else "")

    ax.set_xticks(positions)
    ax.set_xticklabels([n.replace("_", "\n") for n in policy_names], fontsize=9)
    ax.set_ylabel("Discounted Return ($)")
    ax.set_xlabel("Policy")
    ax.set_title("Return Distributions Under Stochastic Uncertainty (Medium Risk)")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_risk_return_tradeoff(
    evaluations: dict[str, PolicyEvaluation],
    output_path: Path,
) -> None:
    """Plot mean return vs CVaR (risk-return tradeoff)."""
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, len(evaluations)))

    for i, (name, ev) in enumerate(evaluations.items()):
        ax.scatter(
            ev.cvar_5,
            ev.mean_return,
            s=150,
            c=[colors[i]],
            label=name.replace("_", " "),
            edgecolors="black",
            linewidths=1,
        )
        # Add error bars for std
        ax.errorbar(
            ev.cvar_5,
            ev.mean_return,
            yerr=ev.std_return,
            fmt="none",
            c=colors[i],
            alpha=0.5,
            capsize=3,
        )

    ax.set_xlabel("CVaR 5% (Worst-Case Return)", fontsize=11)
    ax.set_ylabel("Mean Return ($)", fontsize=11)
    ax.set_title("Risk-Return Tradeoff: Mean Return vs Downside Risk", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)

    # Add quadrant labels
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.axhline(y=np.mean(ylim), color="gray", linestyle=":", alpha=0.5)
    ax.axvline(x=np.mean(xlim), color="gray", linestyle=":", alpha=0.5)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_metrics_comparison(
    evaluations: dict[str, PolicyEvaluation],
    output_path: Path,
) -> None:
    """Plot bar chart comparing key metrics across policies."""
    policy_names = list(evaluations.keys())
    n_policies = len(policy_names)

    metrics = {
        "Mean Return": [ev.mean_return for ev in evaluations.values()],
        "CVaR 5%": [ev.cvar_5 for ev in evaluations.values()],
        "Mean Volume": [ev.mean_volume for ev in evaluations.values()],
        "Mean Thins": [ev.mean_thins for ev in evaluations.values()],
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, n_policies))

    for ax, (metric_name, values) in zip(axes, metrics.items()):
        bars = ax.bar(range(n_policies), values, color=colors)
        ax.set_xticks(range(n_policies))
        ax.set_xticklabels([n.replace("_", "\n") for n in policy_names], fontsize=8)
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

        # Highlight best
        best_idx = np.argmax(values)
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(3)

    fig.suptitle("Policy Performance Metrics Comparison", y=1.02, fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    pmrc = PMRCModel(region="ucp")
    init_state = make_init_state(pmrc)

    # Use medium risk profile
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

    config = resolve_config(risk_profile_name="medium")

    # Get baseline policies
    policies = get_baseline_policies()
    # Select a subset for clarity
    selected = ["noop", "fixed_rotation_25", "fixed_rotation_30", "threshold_ba_120", "aggressive", "conservative"]
    policies = {k: v for k, v in policies.items() if k in selected}

    print("Evaluating policies...")
    evaluations = evaluate_policies(
        policies,
        stoch_pmrc,
        init_state,
        config,
        n_episodes=200,
        max_steps=40,
        seed=42,
    )

    # Print summary
    print("\nPolicy Evaluation Summary:")
    print("-" * 80)
    for name, ev in sorted(evaluations.items(), key=lambda x: -x[1].mean_return):
        print(f"{name:<25} Mean: {ev.mean_return:>8.1f}  Std: {ev.std_return:>7.1f}  CVaR5%: {ev.cvar_5:>8.1f}")

    # Generate plots
    output_path = Path("plots") / "policy_return_distributions.png"
    plot_return_distributions(evaluations, output_path)
    print(f"\nSaved: {output_path}")

    output_path = Path("plots") / "risk_return_tradeoff.png"
    plot_risk_return_tradeoff(evaluations, output_path)
    print(f"Saved: {output_path}")

    output_path = Path("plots") / "policy_metrics_comparison.png"
    plot_metrics_comparison(evaluations, output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
