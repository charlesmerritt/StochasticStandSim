"""Adapter layer for UI-to-core interface.

Provides a clean interface between the Streamlit UI and the core simulation/MDP modules.
Minimizes coupling and avoids modifying core modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple
import numpy as np

from core.config import (
    ActionSpec,
    DisturbanceParams,
    MDPDiscretization,
    NoiseParams,
    RiskLevel,
    RiskProfile,
    get_risk_profile,
    RISK_PROFILES,
)
from core.stochastic_stand import (
    StandState,
    StochasticPMRC,
    StateDiscretizer,
    apply_action_to_state,
    apply_harvest_replant,
    TransitionTrace,
)
from core.pmrc_model import PMRCModel
from core.baselines import (
    Policy,
    NoOpPolicy,
    FixedRotationPolicy,
    ThresholdThinPolicy,
    VolumeThresholdPolicy,
    create_standard_policies,
)
from core.mdp import (
    MDPSolution,
    BuongiornoConfig,
    BuongiornoDiscretizer,
    ForestState,
    solve_mdp_for_risk_level,
    estimate_stand_transitions,
    build_full_transition_matrix,
    make_reward_function,
    value_iteration,
    Action as MDPAction,
)


@dataclass
class StepRecord:
    """Record of a single simulation step."""
    step: int
    age: float
    tpa: float
    ba: float
    hd: float
    si25: float
    volume: float
    action: int
    action_name: str
    reward: float
    cumulative_reward: float
    disturbance: Optional[str]
    disturbance_tpa_loss: float
    disturbance_hd_loss: float
    noise_delta_ba: float
    noise_delta_tpa: float
    seed: int
    timestamp: str


@dataclass
class RolloutResult:
    """Complete rollout result with metadata."""
    steps: List[StepRecord]
    risk_level: str
    seed: int
    final_reward: float
    disturbance_params: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulatorConfig:
    """Configuration for the simulator adapter."""
    si25: float = 60.0
    region: str = "ucp"
    initial_age: float = 1.0
    initial_tpa: float = 600.0
    dt: float = 1.0
    max_age: float = 40.0
    discount_rate: float = 0.05


class SimulatorAdapter:
    """Adapter wrapping StochasticPMRC for UI interaction.
    
    Provides:
    - Step-by-step simulation with full state tracking
    - Manual disturbance triggering
    - Policy application
    - State discretization for transition visualization
    """
    
    ACTION_NAMES = {
        0: "No-op",
        1: "Light Thin (20%)",
        2: "Heavy Thin (40%)",
        3: "Harvest & Replant",
    }
    
    def __init__(
        self,
        config: SimulatorConfig,
        risk_profile: RiskProfile,
        seed: int = 42,
    ):
        self.config = config
        self.risk_profile = risk_profile
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        self.pmrc = PMRCModel(region=config.region)
        self.stoch = StochasticPMRC.from_config(
            self.pmrc,
            risk_profile.noise,
            risk_profile.disturbance,
        )
        
        self._state: Optional[StandState] = None
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._history: List[StepRecord] = []
        self._pending_disturbance: Optional[str] = None
        
    def reset(self, seed: Optional[int] = None) -> StandState:
        """Reset simulator to initial conditions."""
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        
        hd0 = self.pmrc.hd_from_si(self.config.si25, form="projection")
        ba0 = self.pmrc.ba_predict(
            age=self.config.initial_age,
            tpa=self.config.initial_tpa,
            hd=hd0,
            region=self.config.region,
        )
        
        self._state = StandState(
            age=self.config.initial_age,
            hd=hd0,
            tpa=self.config.initial_tpa,
            ba=ba0,
            si25=self.config.si25,
            region=self.config.region,
        )
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._history = []
        self._pending_disturbance = None
        
        return self._state
    
    @property
    def state(self) -> Optional[StandState]:
        return self._state
    
    @property
    def step_count(self) -> int:
        return self._step_count
    
    @property
    def cumulative_reward(self) -> float:
        return self._cumulative_reward
    
    @property
    def history(self) -> List[StepRecord]:
        return self._history
    
    @property
    def is_done(self) -> bool:
        if self._state is None:
            return True
        return self._state.age >= self.config.max_age
    
    def trigger_disturbance(self, disturbance_type: Literal["mild", "severe"]) -> None:
        """Queue a disturbance to occur at the next step."""
        self._pending_disturbance = disturbance_type
    
    def step(
        self,
        action: int,
        custom_disturbance_params: Optional[DisturbanceParams] = None,
    ) -> Tuple[StandState, float, bool, Dict[str, Any]]:
        """Execute one simulation step.
        
        Args:
            action: Action index (0=noop, 1=light thin, 2=heavy thin, 3=harvest)
            custom_disturbance_params: Override disturbance parameters for this step
            
        Returns:
            (next_state, reward, done, info)
        """
        if self._state is None:
            raise RuntimeError("Simulator not initialized. Call reset() first.")
        
        import datetime
        timestamp = datetime.datetime.now().isoformat()
        
        # Apply action
        action_spec = ActionSpec()
        acted_state = apply_action_to_state(self._state, action, self.pmrc, action_spec)
        
        # Use custom disturbance params if provided
        if custom_disturbance_params is not None:
            stoch = StochasticPMRC.from_config(
                self.pmrc,
                self.risk_profile.noise,
                custom_disturbance_params,
            )
        else:
            stoch = self.stoch
        
        # Handle pending manual disturbance
        disturbance_label = None
        trace = None
        
        if self._pending_disturbance is not None:
            # Force the disturbance
            next_state, disturbance_label, _, trace = self._apply_forced_disturbance(
                acted_state, self._pending_disturbance
            )
            self._pending_disturbance = None
        else:
            # Normal stochastic transition
            next_state, disturbance_label, _, trace = stoch.sample_next_state_with_trace(
                acted_state, self.config.dt, self.rng
            )
        
        # Compute reward (simplified: volume-based)
        volume_before = self.pmrc.tvob(
            self._state.age, self._state.tpa, self._state.hd, self._state.ba,
            region=self.config.region,
        )
        volume_after = self.pmrc.tvob(
            next_state.age, next_state.tpa, next_state.hd, next_state.ba,
            region=self.config.region,
        )
        
        # Reward = volume growth + harvest value if action=3
        reward = 0.0
        if action == 3:  # Harvest
            tons = volume_before * 0.031
            reward = tons * 25.0  # Simplified pricing
        else:
            reward = (volume_after - volume_before) * 0.01  # Growth reward
        
        self._cumulative_reward += reward
        self._step_count += 1
        
        # Record step
        record = StepRecord(
            step=self._step_count,
            age=next_state.age,
            tpa=next_state.tpa,
            ba=next_state.ba,
            hd=next_state.hd,
            si25=next_state.si25,
            volume=volume_after,
            action=action,
            action_name=self.ACTION_NAMES.get(action, f"Action {action}"),
            reward=reward,
            cumulative_reward=self._cumulative_reward,
            disturbance=disturbance_label,
            disturbance_tpa_loss=trace.disturbance_tpa_loss if trace else 0.0,
            disturbance_hd_loss=trace.disturbance_hd_loss if trace else 0.0,
            noise_delta_ba=trace.delta_ba if trace else 0.0,
            noise_delta_tpa=trace.delta_tpa if trace else 0.0,
            seed=self.seed,
            timestamp=timestamp,
        )
        self._history.append(record)
        
        self._state = next_state
        done = self.is_done
        
        info = {
            "trace": trace,
            "disturbance": disturbance_label,
            "volume": volume_after,
        }
        
        return next_state, reward, done, info
    
    def _apply_forced_disturbance(
        self,
        state: StandState,
        disturbance_type: str,
    ) -> Tuple[StandState, str, Optional[float], TransitionTrace]:
        """Apply a forced disturbance (manual trigger)."""
        age2 = state.age + self.config.dt
        
        # Get deterministic projections for trace
        hd_mean = self.pmrc.hd_project(state.age, state.hd, age2)
        tpa_mean = self.pmrc.tpa_project(state.tpa, state.si25, state.age, age2)
        ba_mean = self.pmrc.ba_project(
            state.age, state.tpa, tpa_mean, state.ba, state.hd, hd_mean, age2, state.region
        )
        
        # Apply disturbance multipliers
        if disturbance_type == "severe":
            tpa_mult = self.risk_profile.disturbance.severe_tpa_multiplier
            hd_mult = self.risk_profile.disturbance.severe_hd_multiplier
        else:  # mild
            tpa_mult = self.risk_profile.disturbance.mild_tpa_multiplier
            hd_mult = self.risk_profile.disturbance.mild_hd_multiplier
        
        tpa2 = max(10.0, tpa_mean * tpa_mult)
        hd2 = max(1.0, hd_mean * hd_mult)
        ba2 = max(1.0, ba_mean * tpa_mult)
        
        trace = TransitionTrace(
            hd_mean=hd_mean,
            tpa_mean=tpa_mean,
            ba_mean=ba_mean,
            hd_realized=hd2,
            tpa_realized=tpa2,
            ba_realized=ba2,
            delta_hd=0.0,
            delta_tpa=0.0,
            delta_ba=0.0,
            disturbance_label=disturbance_type,
            disturbance_tpa_loss=tpa_mean - tpa2,
            disturbance_hd_loss=hd_mean - hd2,
            recruitment=0.0,
        )
        
        next_state = StandState(
            age=age2,
            hd=hd2,
            tpa=tpa2,
            ba=ba2,
            si25=state.si25,
            region=state.region,
        )
        
        return next_state, disturbance_type, age2, trace
    
    def get_rollout_result(self) -> RolloutResult:
        """Package current history as a RolloutResult."""
        return RolloutResult(
            steps=list(self._history),
            risk_level=self.risk_profile.name,
            seed=self.seed,
            final_reward=self._cumulative_reward,
            disturbance_params={
                "p_mild": self.risk_profile.disturbance.p_mild,
                "severe_mean_interval": self.risk_profile.disturbance.severe_mean_interval,
            },
        )


class PolicyAdapter:
    """Adapter for applying policies and MDP solutions."""
    
    def __init__(self, pmrc: PMRCModel, config: BuongiornoConfig):
        self.pmrc = pmrc
        self.config = config
        self.discretizer = BuongiornoDiscretizer(config, pmrc)
        self._mdp_solutions: Dict[str, MDPSolution] = {}
        self._heuristic_policies = create_standard_policies()
    
    def get_available_policies(self) -> List[str]:
        """Return list of available policy names."""
        policies = list(self._heuristic_policies.keys())
        policies.extend([f"mdp_{k}" for k in self._mdp_solutions.keys()])
        return policies
    
    def solve_mdp(
        self,
        risk_level: RiskLevel,
        n_samples: int = 500,
        seed: int = 42,
    ) -> MDPSolution:
        """Solve MDP for a risk level and cache the solution."""
        cache_key = f"{risk_level}_{n_samples}_{seed}"
        if cache_key not in self._mdp_solutions:
            solution, _, _ = solve_mdp_for_risk_level(
                risk_level,
                self.config,
                n_samples=n_samples,
                seed=seed,
            )
            self._mdp_solutions[cache_key] = solution
        return self._mdp_solutions[cache_key]
    
    def get_mdp_recommendation(
        self,
        state: StandState,
        solution: MDPSolution,
        disturbed: bool = False,
    ) -> Dict[str, Any]:
        """Get MDP policy recommendation for current state.
        
        Returns:
            Dict with 'action', 'value', 'q_values', 'state_index'
        """
        from core.mdp import PriceState, DisturbanceState
        
        # Discretize state
        discrete_state = self.discretizer.discretize(
            state,
            disturbed=disturbed,
            price=PriceState.MEDIUM,
        )
        state_idx = discrete_state.to_index()
        
        # Get policy recommendation
        action = int(solution.policy[state_idx])
        value = float(solution.V[state_idx])
        q_values = {
            MDPAction.WAIT: float(solution.Q[state_idx, 0]),
        }
        if solution.Q.shape[1] > 1:
            q_values[MDPAction.HARVEST] = float(solution.Q[state_idx, 1])
        
        return {
            "action": action,
            "action_name": "WAIT" if action == 0 else "HARVEST",
            "value": value,
            "q_values": q_values,
            "state_index": state_idx,
            "discrete_state": str(discrete_state),
        }
    
    def get_heuristic_action(self, state: StandState, policy_name: str) -> int:
        """Get action from a heuristic policy."""
        if policy_name not in self._heuristic_policies:
            return 0
        policy = self._heuristic_policies[policy_name]
        if hasattr(policy, "set_pmrc"):
            policy.set_pmrc(self.pmrc)
        return policy(state)


class TransitionVisualizer:
    """Manages transition matrix visualization data."""
    
    def __init__(self, discretization: MDPDiscretization):
        self.discretization = discretization
        age_bins, tpa_bins, ba_bins = discretization.to_numpy()
        self.discretizer = StateDiscretizer(age_bins, tpa_bins, ba_bins)
        self._transition_counts: Dict[int, Dict[int, int]] = {}
        self._trajectory_paths: List[List[int]] = []
        self._current_path: List[int] = []
    
    @property
    def n_states(self) -> int:
        return self.discretizer.n_states
    
    def reset(self) -> None:
        """Clear all accumulated transition data."""
        self._transition_counts = {}
        self._trajectory_paths = []
        self._current_path = []
    
    def start_new_trajectory(self) -> None:
        """Start tracking a new trajectory."""
        if self._current_path:
            self._trajectory_paths.append(self._current_path)
        self._current_path = []
    
    def record_state(self, state: StandState) -> int:
        """Record a state visit and return its discrete index."""
        state_idx = self.discretizer.encode(state)
        self._current_path.append(state_idx)
        return state_idx
    
    def record_transition(self, from_state: StandState, to_state: StandState) -> None:
        """Record a transition between states."""
        from_idx = self.discretizer.encode(from_state)
        to_idx = self.discretizer.encode(to_state)
        
        if from_idx not in self._transition_counts:
            self._transition_counts[from_idx] = {}
        if to_idx not in self._transition_counts[from_idx]:
            self._transition_counts[from_idx][to_idx] = 0
        self._transition_counts[from_idx][to_idx] += 1
    
    def get_transition_matrix(self) -> np.ndarray:
        """Build transition matrix from accumulated counts."""
        n = self.n_states
        matrix = np.zeros((n, n))
        
        for from_idx, to_counts in self._transition_counts.items():
            total = sum(to_counts.values())
            if total > 0:
                for to_idx, count in to_counts.items():
                    matrix[from_idx, to_idx] = count / total
        
        return matrix
    
    def get_state_labels(self) -> List[str]:
        """Generate labels for each discrete state."""
        labels = []
        for idx in range(self.n_states):
            i_age, i_tpa, i_ba = self.discretizer.decode(idx)
            age_lo = self.discretizer.age_bins[i_age]
            age_hi = self.discretizer.age_bins[i_age + 1]
            tpa_lo = self.discretizer.tpa_bins[i_tpa]
            tpa_hi = self.discretizer.tpa_bins[i_tpa + 1]
            ba_lo = self.discretizer.ba_bins[i_ba]
            ba_hi = self.discretizer.ba_bins[i_ba + 1]
            labels.append(f"A{age_lo}-{age_hi} T{tpa_lo}-{tpa_hi} B{ba_lo}-{ba_hi}")
        return labels
    
    def get_trajectory_paths(self) -> List[List[int]]:
        """Return all recorded trajectory paths."""
        paths = list(self._trajectory_paths)
        if self._current_path:
            paths.append(self._current_path)
        return paths
    
    def encode_state(self, state: StandState) -> int:
        """Encode a continuous state to discrete index."""
        return self.discretizer.encode(state)
    
    def decode_state(self, idx: int) -> Tuple[int, int, int]:
        """Decode a discrete index to bin indices."""
        return self.discretizer.decode(idx)
    
    def get_bin_midpoints(self, idx: int) -> Tuple[float, float, float]:
        """Get midpoint values for a discrete state."""
        i_age, i_tpa, i_ba = self.discretizer.decode(idx)
        age = 0.5 * (self.discretizer.age_bins[i_age] + self.discretizer.age_bins[i_age + 1])
        tpa = 0.5 * (self.discretizer.tpa_bins[i_tpa] + self.discretizer.tpa_bins[i_tpa + 1])
        ba = 0.5 * (self.discretizer.ba_bins[i_ba] + self.discretizer.ba_bins[i_ba + 1])
        return age, tpa, ba


def run_batch_rollouts(
    config: SimulatorConfig,
    risk_level: RiskLevel,
    n_rollouts: int,
    policy: Optional[Callable[[StandState], int]] = None,
    base_seed: int = 42,
    max_steps: int = 40,
) -> List[RolloutResult]:
    """Run multiple rollouts for batch analysis.
    
    Args:
        config: Simulator configuration
        risk_level: Risk level to use
        n_rollouts: Number of rollouts to run
        policy: Optional policy function (state -> action). If None, uses no-op.
        base_seed: Base seed for reproducibility
        max_steps: Maximum steps per rollout
        
    Returns:
        List of RolloutResult objects
    """
    risk_profile = get_risk_profile(risk_level)
    results = []
    
    for i in range(n_rollouts):
        seed = base_seed + i
        sim = SimulatorAdapter(config, risk_profile, seed=seed)
        sim.reset()
        
        for _ in range(max_steps):
            if sim.is_done:
                break
            
            if policy is not None:
                action = policy(sim.state)
            else:
                action = 0  # No-op
            
            sim.step(action)
        
        results.append(sim.get_rollout_result())
    
    return results


def run_parameter_sweep(
    config: SimulatorConfig,
    risk_levels: List[RiskLevel],
    disturbance_probs: List[float],
    n_rollouts_per_config: int = 10,
    base_seed: int = 42,
    max_steps: int = 40,
) -> Dict[str, List[RolloutResult]]:
    """Run parameter sweep over risk levels and disturbance probabilities.
    
    Returns:
        Dict mapping config key to list of rollout results
    """
    results = {}
    
    for risk_level in risk_levels:
        base_profile = get_risk_profile(risk_level)
        
        for p_mild in disturbance_probs:
            # Create modified disturbance params
            modified_dist = DisturbanceParams(
                p_mild=p_mild,
                severe_mean_interval=base_profile.disturbance.severe_mean_interval,
                mild_tpa_multiplier=base_profile.disturbance.mild_tpa_multiplier,
                mild_hd_multiplier=base_profile.disturbance.mild_hd_multiplier,
                severe_tpa_multiplier=base_profile.disturbance.severe_tpa_multiplier,
                severe_hd_multiplier=base_profile.disturbance.severe_hd_multiplier,
            )
            modified_profile = RiskProfile(
                name=f"{risk_level}_p{p_mild:.2f}",
                noise=base_profile.noise,
                disturbance=modified_dist,
            )
            
            config_key = f"{risk_level}_pmild_{p_mild:.2f}"
            config_results = []
            
            for i in range(n_rollouts_per_config):
                seed = base_seed + i
                sim = SimulatorAdapter(config, modified_profile, seed=seed)
                sim.reset()
                
                for _ in range(max_steps):
                    if sim.is_done:
                        break
                    sim.step(0)  # No-op policy
                
                config_results.append(sim.get_rollout_result())
            
            results[config_key] = config_results
    
    return results
