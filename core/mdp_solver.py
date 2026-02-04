"""MDP solvers: value iteration and policy iteration for discrete transition matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class MDPResult:
    """Result of MDP solution."""
    values: np.ndarray          # V(s) for each state
    policy: np.ndarray          # optimal action index for each state
    iterations: int             # number of iterations to converge
    converged: bool             # whether algorithm converged


def value_iteration(
    transition_matrices: dict[int, np.ndarray],
    reward_fn: Callable[[int, int, int], float],
    gamma: float = 0.95,
    theta: float = 1e-6,
    max_iterations: int = 1000,
) -> MDPResult:
    """Solve MDP via value iteration.
    
    Args:
        transition_matrices: dict mapping action_index -> P[s, s'] transition matrix
        reward_fn: R(state, action, next_state) -> reward
        gamma: discount factor
        theta: convergence threshold
        max_iterations: maximum iterations
        
    Returns:
        MDPResult with optimal values and policy
    """
    actions = sorted(transition_matrices.keys())
    n_states = transition_matrices[actions[0]].shape[0]
    
    V = np.zeros(n_states)
    policy = np.zeros(n_states, dtype=int)
    
    for iteration in range(max_iterations):
        delta = 0.0
        V_new = np.zeros(n_states)
        
        for s in range(n_states):
            action_values = []
            for a in actions:
                P = transition_matrices[a]
                # Expected value under action a from state s
                q_sa = 0.0
                for s_next in range(n_states):
                    if P[s, s_next] > 0:
                        r = reward_fn(s, a, s_next)
                        q_sa += P[s, s_next] * (r + gamma * V[s_next])
                action_values.append(q_sa)
            
            best_value = max(action_values)
            V_new[s] = best_value
            policy[s] = actions[np.argmax(action_values)]
            delta = max(delta, abs(V_new[s] - V[s]))
        
        V = V_new
        if delta < theta:
            return MDPResult(values=V, policy=policy, iterations=iteration + 1, converged=True)
    
    return MDPResult(values=V, policy=policy, iterations=max_iterations, converged=False)


def policy_iteration(
    transition_matrices: dict[int, np.ndarray],
    reward_fn: Callable[[int, int, int], float],
    gamma: float = 0.95,
    theta: float = 1e-6,
    max_iterations: int = 100,
) -> MDPResult:
    """Solve MDP via policy iteration.
    
    Args:
        transition_matrices: dict mapping action_index -> P[s, s'] transition matrix
        reward_fn: R(state, action, next_state) -> reward
        gamma: discount factor
        theta: convergence threshold for policy evaluation
        max_iterations: maximum policy improvement iterations
        
    Returns:
        MDPResult with optimal values and policy
    """
    actions = sorted(transition_matrices.keys())
    n_states = transition_matrices[actions[0]].shape[0]
    
    # Initialize random policy
    policy = np.zeros(n_states, dtype=int)
    V = np.zeros(n_states)
    
    for iteration in range(max_iterations):
        # Policy evaluation
        V = _policy_evaluation(transition_matrices, reward_fn, policy, gamma, theta)
        
        # Policy improvement
        policy_stable = True
        for s in range(n_states):
            old_action = policy[s]
            action_values = []
            for a in actions:
                P = transition_matrices[a]
                q_sa = 0.0
                for s_next in range(n_states):
                    if P[s, s_next] > 0:
                        r = reward_fn(s, a, s_next)
                        q_sa += P[s, s_next] * (r + gamma * V[s_next])
                action_values.append(q_sa)
            
            policy[s] = actions[np.argmax(action_values)]
            if old_action != policy[s]:
                policy_stable = False
        
        if policy_stable:
            return MDPResult(values=V, policy=policy, iterations=iteration + 1, converged=True)
    
    return MDPResult(values=V, policy=policy, iterations=max_iterations, converged=False)


def _policy_evaluation(
    transition_matrices: dict[int, np.ndarray],
    reward_fn: Callable[[int, int, int], float],
    policy: np.ndarray,
    gamma: float,
    theta: float,
    max_iterations: int = 1000,
) -> np.ndarray:
    """Evaluate a fixed policy."""
    n_states = len(policy)
    V = np.zeros(n_states)
    
    for _ in range(max_iterations):
        delta = 0.0
        for s in range(n_states):
            a = policy[s]
            P = transition_matrices[a]
            v_new = 0.0
            for s_next in range(n_states):
                if P[s, s_next] > 0:
                    r = reward_fn(s, a, s_next)
                    v_new += P[s, s_next] * (r + gamma * V[s_next])
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new
        
        if delta < theta:
            break
    
    return V


def finite_horizon_value_iteration(
    transition_matrices: dict[int, np.ndarray],
    reward_fn: Callable[[int, int, int], float],
    terminal_reward_fn: Callable[[int], float],
    horizon: int,
    gamma: float = 0.95,
) -> MDPResult:
    """Solve finite-horizon MDP via backward induction.
    
    Args:
        transition_matrices: dict mapping action_index -> P[s, s'] transition matrix
        reward_fn: R(state, action, next_state) -> reward
        terminal_reward_fn: R_T(state) -> terminal reward at horizon
        horizon: number of time steps
        gamma: discount factor
        
    Returns:
        MDPResult with optimal values (at t=0) and policy (at t=0)
    """
    actions = sorted(transition_matrices.keys())
    n_states = transition_matrices[actions[0]].shape[0]
    
    # V[t, s] = value of being in state s at time t
    # Policy[t, s] = optimal action at state s at time t
    V = np.zeros((horizon + 1, n_states))
    policy = np.zeros((horizon + 1, n_states), dtype=int)
    
    # Terminal values
    for s in range(n_states):
        V[horizon, s] = terminal_reward_fn(s)
    
    # Backward induction
    for t in range(horizon - 1, -1, -1):
        for s in range(n_states):
            action_values = []
            for a in actions:
                P = transition_matrices[a]
                q_sa = 0.0
                for s_next in range(n_states):
                    if P[s, s_next] > 0:
                        r = reward_fn(s, a, s_next)
                        q_sa += P[s, s_next] * (r + gamma * V[t + 1, s_next])
                action_values.append(q_sa)
            
            best_idx = int(np.argmax(action_values))
            V[t, s] = action_values[best_idx]
            policy[t, s] = actions[best_idx]
    
    # Return values and policy at t=0, plus full time-dependent arrays
    result = MDPResult(values=V[0], policy=policy[0], iterations=horizon, converged=True)
    # Attach full arrays for trajectory simulation (not part of dataclass)
    result._full_policy = policy  # shape: (horizon+1, n_states)
    result._full_values = V  # shape: (horizon+1, n_states)
    return result


def extract_q_values(
    transition_matrices: dict[int, np.ndarray],
    reward_fn: Callable[[int, int, int], float],
    V: np.ndarray,
    gamma: float = 0.95,
) -> np.ndarray:
    """Extract Q(s, a) values from value function.
    
    Returns:
        Q: array of shape (n_states, n_actions)
    """
    actions = sorted(transition_matrices.keys())
    n_states = V.shape[0]
    n_actions = len(actions)
    
    Q = np.zeros((n_states, n_actions))
    for ai, a in enumerate(actions):
        P = transition_matrices[a]
        for s in range(n_states):
            for s_next in range(n_states):
                if P[s, s_next] > 0:
                    r = reward_fn(s, a, s_next)
                    Q[s, ai] += P[s, s_next] * (r + gamma * V[s_next])
    
    return Q


# ----------------------------- Reward Functions -----------------------------

def make_timber_reward_fn(
    discretizer,
    econ_params,
    action_spec,
    pmrc_model,
    region: str = "ucp",
    si25: float = 60.0,
) -> Callable[[int, int, int], float]:
    """Create a reward function based on timber economics.
    
    Args:
        discretizer: StateDiscretizer instance
        econ_params: EconParams from config
        action_spec: ActionSpec from config
        pmrc_model: PMRCModel instance
        region: region code
        si25: site index
        
    Returns:
        reward_fn(state_idx, action_idx, next_state_idx) -> float
    """
    # Precompute bin centers
    age_edges, tpa_edges, ba_edges = discretizer.age_bins, discretizer.tpa_bins, discretizer.ba_bins
    age_centers = 0.5 * (age_edges[:-1] + age_edges[1:])
    tpa_centers = 0.5 * (tpa_edges[:-1] + tpa_edges[1:])
    ba_centers = 0.5 * (ba_edges[:-1] + ba_edges[1:])
    
    # Price per unit volume (simplified)
    avg_price = (econ_params.sawtimber_price + econ_params.pulpwood_price + econ_params.chip_price) / 3.0
    vol_to_ton = 0.031
    
    def reward_fn(s: int, a: int, s_next: int) -> float:
        i_age, i_tpa, i_ba = discretizer.decode(s)
        age = age_centers[i_age]
        tpa = tpa_centers[i_tpa]
        ba = ba_centers[i_ba]
        
        # Compute volume at current state
        hd = pmrc_model.hd_from_si(si25, form="projection")
        vol = pmrc_model.tvob(age, tpa, hd, ba, region=region)
        
        # Action costs/revenues
        # 0=noop, 1=thin_light, 2=thin_heavy, 3=harvest
        if a == 0:
            return 0.0
        elif a == 1:  # thin_light
            removed_frac = 1.0 - action_spec.thin_light_residual_frac
            removed_vol = vol * removed_frac
            revenue = removed_vol * vol_to_ton * avg_price
            cost = econ_params.thin_harvest_cost
            return revenue - cost
        elif a == 2:  # thin_heavy
            removed_frac = 1.0 - action_spec.thin_heavy_residual_frac
            removed_vol = vol * removed_frac
            revenue = removed_vol * vol_to_ton * avg_price
            cost = econ_params.thin_harvest_cost
            return revenue - cost
        elif a == 3:  # harvest
            revenue = vol * vol_to_ton * avg_price
            return revenue
        
        return 0.0
    
    return reward_fn


def make_simple_reward_fn(
    discretizer,
    harvest_reward: float = 1000.0,
    thin_reward: float = 100.0,
    growth_reward_per_ba: float = 1.0,
) -> Callable[[int, int, int], float]:
    """Create a simple reward function for testing.
    
    Args:
        discretizer: StateDiscretizer instance
        harvest_reward: reward for harvesting
        thin_reward: reward for thinning
        growth_reward_per_ba: reward per unit BA growth
        
    Returns:
        reward_fn(state_idx, action_idx, next_state_idx) -> float
    """
    ba_edges = discretizer.ba_bins
    ba_centers = 0.5 * (ba_edges[:-1] + ba_edges[1:])
    
    def reward_fn(s: int, a: int, s_next: int) -> float:
        _, _, i_ba = discretizer.decode(s)
        _, _, i_ba_next = discretizer.decode(s_next)
        
        ba = ba_centers[i_ba]
        ba_next = ba_centers[i_ba_next]
        
        # Growth reward
        growth = max(0, ba_next - ba) * growth_reward_per_ba
        
        # Action rewards
        if a == 3:  # harvest
            return harvest_reward * (ba / 100.0)  # scale by BA
        elif a in (1, 2):  # thin
            return thin_reward * (ba / 100.0)
        
        return growth
    
    return reward_fn
