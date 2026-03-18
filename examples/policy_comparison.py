"""Compare MDP-optimal policies across risk profiles.

This figure shows how optimal management policies change under different
risk scenarios (low/medium/high disturbance rates), demonstrating
risk-sensitive decision making.
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
from core.stochastic_model import (
    StochasticPMRC,
    StandState,
    StateDiscretizer,
    NoiseConfig,
    estimate_transition_matrix,
    apply_action_to_state,
)
from core.config import make_risk_profiles
from core.mdp_solver import finite_horizon_value_iteration, MDPResult


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
    """Create state discretizer with finer bins."""
    # Finer age bins for smoother transitions
    age_bins = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0])
    # Single TPA bin to reduce state space
    tpa_bins = np.array([50.0, 700.0])
    # Finer BA bins
    ba_bins = np.array([0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0])
    return StateDiscretizer(age_bins, tpa_bins, ba_bins)


HORIZON = 40  # 40-year planning horizon (allows for ~1.5 rotations)


def make_reward_fn(discretizer: StateDiscretizer, pmrc: PMRCModel, si25: float = 60.0):
    """Create a timber-value reward function based on pure economics.
    
    NO hard constraints - let the MDP discover optimal timing through:
    - Young stands produce pulpwood (low price), mature stands produce sawtimber (high price)
    - Sawtimber fraction increases with age (sigmoid transition around age 20)
    - Costs are fixed → need sufficient volume AND maturity to be profitable
    - Risk affects expected future value → MDP adapts timing accordingly
    
    Economic parameters (from data/econ_params.yaml):
    - Sawtimber price: $27.82/ton
    - Pulpwood price: $9.51/ton
    - Thin cost: $87.34/acre
    - Harvest cost: $150/acre (logging)
    - Planting cost: $150.80/acre (site prep + planting after harvest)
    """
    age_edges = discretizer.age_bins
    ba_edges = discretizer.ba_bins
    age_centers = 0.5 * (age_edges[:-1] + age_edges[1:])
    ba_centers = 0.5 * (ba_edges[:-1] + ba_edges[1:])
    
    # Economic parameters
    pulpwood_price = 9.51  # $/ton
    sawtimber_price = 27.82  # $/ton
    thin_cost = 87.34  # $/acre
    harvest_cost = 150.0  # $/acre (logging)
    replant_cost = 150.80  # $/acre (site prep + planting)
    cuft_to_ton = 0.031

    def sawtimber_fraction(age: float) -> float:
        """Fraction of harvest volume that is sawtimber (vs pulpwood).
        
        Sigmoid transition: ~0% at age 10, ~50% at age 20, ~90% at age 30.
        This reflects that young trees are too small for sawtimber.
        """
        # Sigmoid centered at age 20, steepness 0.2
        return 1.0 / (1.0 + np.exp(-0.2 * (age - 20)))

    def reward_fn(s: int, a: int, s_next: int) -> float:
        i_age, _, i_ba = discretizer.decode(s)
        age = age_centers[i_age]
        ba = ba_centers[i_ba]
        
        # Volume proxy: increases with age and BA
        vol_cuft = ba * age * 0.5
        vol_tons = vol_cuft * cuft_to_ton
        
        if a == 0:  # no-op
            return 0.0
            
        elif a == 1:  # light thin (20% BA removal)
            # Thinning always gets pulpwood price (removing smaller trees)
            removed_tons = vol_tons * 0.20
            revenue = removed_tons * pulpwood_price
            return revenue - thin_cost
            
        elif a == 2:  # heavy thin (40% BA removal)
            removed_tons = vol_tons * 0.40
            revenue = removed_tons * pulpwood_price
            return revenue - thin_cost
            
        elif a == 3:  # harvest (clearcut + replant)
            # Harvest gets blend of sawtimber and pulpwood based on age
            saw_frac = sawtimber_fraction(age)
            blended_price = saw_frac * sawtimber_price + (1 - saw_frac) * pulpwood_price
            revenue = vol_tons * blended_price
            total_cost = harvest_cost + replant_cost
            return revenue - total_cost
            
        return 0.0

    return reward_fn


def make_terminal_reward_fn(discretizer: StateDiscretizer, pmrc: PMRCModel):
    """Terminal reward: ZERO - you must harvest to capture value.
    
    Standing timber at end of horizon has no value unless harvested.
    This forces the MDP to plan for explicit harvest actions.
    """
    def terminal_fn(s: int) -> float:
        return 0.0

    return terminal_fn


def solve_mdp_for_profile(
    profile_name: str,
    pmrc: PMRCModel,
    discretizer: StateDiscretizer,
    init_state: StandState,
    rng: np.random.Generator,
) -> tuple[MDPResult, dict[int, np.ndarray]]:
    """Estimate transitions and solve MDP for a risk profile."""
    profiles = make_risk_profiles()
    profile = profiles[profile_name]

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

    matrices = estimate_transition_matrix(
        stoch_pmrc,
        discretizer,
        actions=[0, 1, 2, 3],  # Include harvest action
        dt=1.0,
        n_mc=2000,
        rng=rng,
        si25=60.0,
        region="ucp",
        init_state=init_state,
        steps=45,
    )

    reward_fn = make_reward_fn(discretizer, pmrc)
    terminal_fn = make_terminal_reward_fn(discretizer, pmrc)
    result = finite_horizon_value_iteration(
        matrices, reward_fn, terminal_fn, horizon=HORIZON, gamma=0.95
    )
    return result, matrices


def plot_policy_comparison(
    results: dict[str, MDPResult],
    discretizer: StateDiscretizer,
    output_path: Path,
) -> None:
    """Plot optimal policies for each risk profile."""
    action_names = ["No-op", "Light Thin", "Heavy Thin", "Harvest"]
    action_colors = ["#95a5a6", "#3498db", "#e74c3c", "#2ecc71"]
    profile_names = list(results.keys())

    n_states = discretizer.n_states
    age_edges = discretizer.age_bins
    ba_edges = discretizer.ba_bins
    n_age = len(age_edges) - 1
    n_ba = len(ba_edges) - 1
    n_tpa = len(discretizer.tpa_bins) - 1

    fig, axes = plt.subplots(1, len(profile_names), figsize=(5 * len(profile_names), 5))
    if len(profile_names) == 1:
        axes = [axes]

    for ax, profile_name in zip(axes, profile_names):
        result = results[profile_name]
        policy = result.policy

        # Reshape policy to age x ba (aggregate over TPA)
        policy_grid = np.zeros((n_age, n_ba))
        for i_age in range(n_age):
            for i_ba in range(n_ba):
                # Average policy over TPA bins
                actions = []
                for i_tpa in range(n_tpa):
                    idx = i_age * (n_tpa * n_ba) + i_tpa * n_ba + i_ba
                    if idx < len(policy):
                        actions.append(policy[idx])
                if actions:
                    policy_grid[i_age, i_ba] = np.median(actions)

        im = ax.imshow(
            policy_grid.T,
            cmap=plt.cm.colors.ListedColormap(action_colors),
            aspect="auto",
            origin="lower",
            vmin=0,
            vmax=3,
        )

        ax.set_xticks(range(n_age))
        ax.set_xticklabels([f"{int(age_edges[i])}-{int(age_edges[i+1])}" for i in range(n_age)])
        ax.set_yticks(range(n_ba))
        ax.set_yticklabels([f"{int(ba_edges[i])}-{int(ba_edges[i+1])}" for i in range(n_ba)])
        ax.set_xlabel("Age Class (years)")
        ax.set_ylabel("Basal Area Class (ft²/ac)")
        ax.set_title(f"{profile_name.capitalize()} Risk\n(converged: {result.converged}, iter: {result.iterations})")

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=n) for c, n in zip(action_colors, action_names)]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Optimal Management Policy by Risk Profile", y=1.02, fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_value_functions(
    results: dict[str, MDPResult],
    discretizer: StateDiscretizer,
    output_path: Path,
) -> None:
    """Plot value functions for each risk profile."""
    profile_names = list(results.keys())
    age_edges = discretizer.age_bins
    ba_edges = discretizer.ba_bins
    n_age = len(age_edges) - 1
    n_ba = len(ba_edges) - 1
    n_tpa = len(discretizer.tpa_bins) - 1

    fig, axes = plt.subplots(1, len(profile_names), figsize=(5 * len(profile_names), 4))
    if len(profile_names) == 1:
        axes = [axes]

    vmin = min(r.values.min() for r in results.values())
    vmax = max(r.values.max() for r in results.values())

    for ax, profile_name in zip(axes, profile_names):
        result = results[profile_name]
        values = result.values

        # Reshape values to age x ba (average over TPA)
        value_grid = np.zeros((n_age, n_ba))
        for i_age in range(n_age):
            for i_ba in range(n_ba):
                vals = []
                for i_tpa in range(n_tpa):
                    idx = i_age * (n_tpa * n_ba) + i_tpa * n_ba + i_ba
                    if idx < len(values):
                        vals.append(values[idx])
                if vals:
                    value_grid[i_age, i_ba] = np.mean(vals)

        im = ax.imshow(value_grid.T, cmap="viridis", aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(n_age))
        ax.set_xticklabels([f"{int(age_edges[i])}-{int(age_edges[i+1])}" for i in range(n_age)])
        ax.set_yticks(range(n_ba))
        ax.set_yticklabels([f"{int(ba_edges[i])}-{int(ba_edges[i+1])}" for i in range(n_ba)])
        ax.set_xlabel("Age Class (years)")
        ax.set_ylabel("Basal Area Class (ft²/ac)")
        ax.set_title(f"{profile_name.capitalize()} Risk")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="V(s)")

    fig.suptitle("State Value Functions V(s) by Risk Profile", y=1.02, fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def solve_mdp_no_risk(
    pmrc: PMRCModel,
    discretizer: StateDiscretizer,
    init_state: StandState,
    rng: np.random.Generator,
) -> tuple[MDPResult, dict[int, np.ndarray]]:
    """Solve MDP with no stochastic risk (deterministic baseline)."""
    # No noise, no disturbances
    stoch_pmrc = StochasticPMRC(
        pmrc,
        noise_config=NoiseConfig(ba_std=0.0, tpa_std=0.0, hd_std=0.0, clip_std=0.0),
        use_gaussian_noise=True,
        p_mild=0.0,
        severe_mean_interval=100000.0,  # essentially never
    )

    matrices = estimate_transition_matrix(
        stoch_pmrc,
        discretizer,
        actions=[0, 1, 2, 3],
        dt=1.0,
        n_mc=1000,  # fewer needed for deterministic
        rng=rng,
        si25=60.0,
        region="ucp",
        init_state=init_state,
        steps=45,
    )

    reward_fn = make_reward_fn(discretizer, pmrc)
    terminal_fn = make_terminal_reward_fn(discretizer, pmrc)
    result = finite_horizon_value_iteration(
        matrices, reward_fn, terminal_fn, horizon=HORIZON, gamma=0.95
    )
    return result, matrices


def main() -> None:
    pmrc = PMRCModel(region="ucp")
    discretizer = create_discretizer()
    init_state = make_init_state(pmrc)

    results = {}
    
    # No risk (deterministic) baseline
    print("Solving MDP for no-risk (deterministic) baseline...")
    rng = np.random.default_rng(42)
    result, _ = solve_mdp_no_risk(pmrc, discretizer, init_state, rng)
    results["none"] = result
    print(f"  Converged: {result.converged}, Iterations: {result.iterations}")
    
    # Risk profiles
    for profile_name in ["low", "medium", "high"]:
        print(f"Solving MDP for {profile_name} risk profile...")
        rng = np.random.default_rng(42)
        result, _ = solve_mdp_for_profile(profile_name, pmrc, discretizer, init_state, rng)
        results[profile_name] = result
        print(f"  Converged: {result.converged}, Iterations: {result.iterations}")

    output_path = Path("plots") / "policy_comparison.png"
    plot_policy_comparison(results, discretizer, output_path)
    print(f"Saved: {output_path}")

    output_path_values = Path("plots") / "value_functions.png"
    plot_value_functions(results, discretizer, output_path_values)
    print(f"Saved: {output_path_values}")

    # Plot trajectories
    output_path_traj = Path("plots") / "policy_trajectories.png"
    plot_trajectories(results, discretizer, pmrc, output_path_traj)
    print(f"Saved: {output_path_traj}")


def simulate_trajectory_continuous(
    stoch_pmrc: StochasticPMRC,
    discretizer: StateDiscretizer,
    result: MDPResult,
    init_state: StandState,
    horizon: int,
    rng: np.random.Generator,
) -> tuple[list[float], list[float], list[int]]:
    """Simulate a trajectory using continuous state dynamics and time-dependent policy.
    
    Returns:
        ages: list of stand ages over time
        bas: list of basal areas over time
        actions: list of actions taken
    """
    full_policy = getattr(result, '_full_policy', None)
    if full_policy is None:
        full_policy = np.tile(result.policy, (horizon + 1, 1))
    
    pmrc = stoch_pmrc.pmrc
    ages = [init_state.age]
    bas = [init_state.ba]
    actions_taken = []
    
    state = init_state
    for t in range(horizon):
        # Encode current state to get optimal action
        s_idx = discretizer.encode(state)
        a = int(full_policy[t, s_idx])
        actions_taken.append(a)
        
        # Apply action and simulate next state
        acted = apply_action_to_state(state, a, pmrc)
        state = stoch_pmrc.sample_next_state(acted, dt=1.0, rng=rng)
        
        ages.append(state.age)
        bas.append(state.ba)
    
    return ages, bas, actions_taken


def plot_trajectories(
    results: dict[str, MDPResult],
    discretizer: StateDiscretizer,
    pmrc: PMRCModel,
    output_path: Path,
) -> None:
    """Plot optimal action trajectories over time for each risk profile."""
    action_names = {0: "No-op", 1: "Light Thin", 2: "Heavy Thin", 3: "Harvest"}
    action_colors = {0: "#808080", 1: "#1f77b4", 2: "#d62728", 3: "#2ca02c"}
    
    # We need transition matrices to simulate - recreate them
    init_state = make_init_state(pmrc)
    rng = np.random.default_rng(42)
    
    # Get initial state index
    init_idx = discretizer.encode(init_state)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    profile_names = ["none", "low", "medium", "high"]
    
    for ax, profile_name in zip(axes, profile_names):
        result = results[profile_name]
        
        # Recreate transition matrices for this profile
        if profile_name == "none":
            stoch_pmrc = StochasticPMRC(
                pmrc,
                noise_config=NoiseConfig(ba_std=0.0, tpa_std=0.0, hd_std=0.0, clip_std=0.0),
                use_gaussian_noise=True,
                p_mild=0.0,
                severe_mean_interval=100000.0,
            )
        else:
            profiles = make_risk_profiles()
            profile = profiles[profile_name]
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
        
        # Simulate trajectory using continuous dynamics
        traj_rng = np.random.default_rng(42)  # Fresh RNG for each trajectory
        ages, bas, actions_taken = simulate_trajectory_continuous(
            stoch_pmrc, discretizer, result, init_state, HORIZON, traj_rng
        )
        
        # Plot age over time
        time_steps = list(range(HORIZON + 1))
        ax.plot(time_steps, ages, 'k-', linewidth=1.5, label='Stand Age')
        
        # Mark actions with colored markers
        for t, a in enumerate(actions_taken):
            if a != 0:  # Only mark non-noop actions
                ax.axvline(x=t, color=action_colors[a], alpha=0.5, linestyle='--')
                ax.scatter([t], [ages[t]], c=action_colors[a], s=100, zorder=5,
                          label=action_names[a] if a not in [actions_taken[i] for i in range(t)] else "")
        
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Stand Age (years)")
        ax.set_title(f"{profile_name.capitalize()} Risk")
        ax.set_xlim(0, HORIZON)
        ax.set_ylim(0, 45)
        ax.grid(True, alpha=0.3)
        
        # Add action sequence as text
        action_seq = [action_names[a][:1] for a in actions_taken]  # First letter
        unique_actions = []
        for i, a in enumerate(actions_taken):
            if a != 0:
                unique_actions.append(f"t={i}: {action_names[a]}")
        if unique_actions:
            ax.text(0.02, 0.98, "\n".join(unique_actions[:5]), transform=ax.transAxes,
                   fontsize=8, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Legend
    handles = [plt.Line2D([0], [0], color=action_colors[a], linewidth=2, label=action_names[a])
               for a in [0, 1, 2, 3]]
    fig.legend(handles=handles, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.02))
    
    fig.suptitle("Optimal Management Trajectories by Risk Profile\n(Starting from age 5, BA ~50)", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
