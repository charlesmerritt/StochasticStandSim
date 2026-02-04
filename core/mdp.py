"""Buongiorno-Style MDP for Forest Stand Management.

This module implements a Markov Decision Process following the methodology of
Buongiorno & Zhou (2015) "Adaptive Economic and Ecological Forest Management
Under Risk".

State Space Design:
-------------------
The state is defined by coarse categorical variables:

1. **Stand State** (18 states):
   - BA level: LOW / MEDIUM / HIGH (3 levels)
   - TPA level: LOW / MEDIUM / HIGH (3 levels)  
   - Disturbance indicator: NORMAL / RECENTLY_DISTURBED (2 levels)
   Note: HIGH BA + LOW TPA is physically impossible (3×3×2 - 2 = 16 feasible)
   
2. **Price State** (3 levels):
   - LOW / MEDIUM / HIGH stumpage prices

Combined: 18 stand states × 3 price states = 54 total MDP states
Feasible: ~48 states after removing impossible combinations

Thresholds (calibrated to typical loblolly pine, SI25=60):
- BA: LOW < 60, MEDIUM 60-120, HIGH > 120 ft²/ac
- TPA: LOW < 300, MEDIUM 300-450, HIGH > 450 trees/ac
- Thinning trigger: BA ≥ MEDIUM AND TPA = HIGH (aligns with age ~15)

Transition Estimation:
----------------------
Transitions are estimated by:
1. Running continuous stochastic growth simulations
2. Binning outcomes into discrete states
3. Counting transitions to estimate P(s'|s, a)

The joint transition factors as:
    S = [p(j|i)] = [p(s'|s) × p(m'|m)]

where stand transitions depend on actions, price transitions are exogenous.

References:
-----------
Buongiorno, J., & Zhou, M. (2015). Adaptive economic and ecological forest
management under risk. Forest Ecosystems, 2(1), 4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Literal

import numpy as np

from core.pmrc_model import PMRCModel
from core.stochastic_stand import (
    StandState,
    StochasticPMRC,
    SizeClassDistribution,
    size_class_distribution_from_state,
    thin_to_residual_ba_smallest_first,
    DEFAULT_DBH_BOUNDS,
)
from core.config import RiskProfile, get_risk_profile
from core.products import estimate_product_distribution, compute_harvest_value


# =============================================================================
# State Space Definitions
# =============================================================================

class BALevel(IntEnum):
    """Basal area level (ft²/ac)."""
    VERY_LOW = 0   # BA < 40
    LOW = 1        # 40 ≤ BA < 70
    MEDIUM = 2     # 70 ≤ BA < 100
    HIGH = 3       # 100 ≤ BA < 130
    VERY_HIGH = 4  # BA ≥ 130


class TPALevel(IntEnum):
    """Trees per acre level."""
    LOW = 0        # TPA < 200
    MEDIUM = 1     # 200 ≤ TPA < 350
    HIGH = 2       # 350 ≤ TPA < 500
    VERY_HIGH = 3  # TPA ≥ 500


class DisturbanceState(IntEnum):
    """Disturbance history indicator."""
    NORMAL = 0
    RECENTLY_DISTURBED = 1  # Within last 5 years


class PriceState(IntEnum):
    """Stumpage price level."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2


@dataclass(frozen=True)
class ForestState:
    """Discrete forest state with BA, TPA, disturbance, and price.
    
    State space: BA (5) × TPA (4) × Disturbed (2) × Price (3) = 120 states
    """
    ba: BALevel            # Total BA level (5 levels)
    tpa: TPALevel          # Trees per acre level (4 levels)
    disturbed: DisturbanceState
    price: PriceState
    
    # Constants for encoding
    N_BA = 5
    N_TPA = 4
    N_DISTURBED = 2
    N_PRICE = 3
    N_STAND = N_BA * N_TPA * N_DISTURBED  # 40
    N_TOTAL = N_STAND * N_PRICE  # 120
    
    def to_index(self) -> int:
        """Convert state to single integer index.
        
        Encoding: ba + 5*tpa + 20*disturbed + 40*price
        Total: 5 × 4 × 2 × 3 = 120 states
        """
        stand_idx = int(self.ba) + self.N_BA * int(self.tpa) + (self.N_BA * self.N_TPA) * int(self.disturbed)
        return stand_idx + self.N_STAND * int(self.price)
    
    @classmethod
    def from_index(cls, idx: int) -> "ForestState":
        """Reconstruct state from integer index."""
        price = PriceState(idx // cls.N_STAND)
        stand_idx = idx % cls.N_STAND
        disturbed = DisturbanceState(stand_idx // (cls.N_BA * cls.N_TPA))
        stand_idx = stand_idx % (cls.N_BA * cls.N_TPA)
        tpa = TPALevel(stand_idx // cls.N_BA)
        ba = BALevel(stand_idx % cls.N_BA)
        return cls(ba=ba, tpa=tpa, disturbed=disturbed, price=price)
    
    def is_feasible(self) -> bool:
        """Check if this state is physically possible."""
        # VERY_HIGH BA + LOW TPA is impossible
        if self.ba == BALevel.VERY_HIGH and self.tpa == TPALevel.LOW:
            return False
        return True
    
    def __str__(self) -> str:
        ba_str = ["VL", "L", "M", "H", "VH"][self.ba]
        tpa_str = ["L", "M", "H", "VH"][self.tpa]
        dist_str = "D" if self.disturbed else "N"
        price_str = ["$L", "$M", "$H"][self.price]
        return f"BA:{ba_str} TPA:{tpa_str} {dist_str} {price_str}"


@dataclass
class BuongiornoConfig:
    """Configuration for Buongiorno-style MDP.
    
    Attributes:
        ba_thresholds: BA boundaries for LOW/MEDIUM/HIGH classification
        tpa_thresholds: TPA boundaries for LOW/MEDIUM/HIGH classification
        price_levels: Stumpage price multipliers for low/medium/high
        price_transition: 3×3 Markov transition matrix for prices
        auto_thin_ba_threshold: BA level above which thinning is considered
        auto_thin_tpa_threshold: TPA level above which thinning triggers
        auto_thin_target_ba: Target BA after thinning
        harvest_cost_per_ton: Cost to harvest ($/ton)
        thin_cost_multiplier: Thinning costs this × harvest cost
        thin_revenue_multiplier: Thinning revenue is this × harvest revenue
        discount_rate: Annual discount rate
        si25: Site index (base age 25)
        region: PMRC region code
    """
    # BA thresholds: [VL/L, L/M, M/H, H/VH boundaries]
    # ~30 ft² wide buckets to capture annual growth
    ba_thresholds: tuple[float, float, float, float] = (40.0, 70.0, 100.0, 130.0)
    
    # TPA thresholds: [L/M, M/H, H/VH boundaries]
    # ~150 trees wide buckets
    tpa_thresholds: tuple[float, float, float] = (200.0, 350.0, 500.0)
    
    # Price state parameters
    price_levels: dict[PriceState, float] = field(default_factory=lambda: {
        PriceState.LOW: 0.8,     # 80% of base price
        PriceState.MEDIUM: 1.0,  # Base price
        PriceState.HIGH: 1.2,    # 120% of base price
    })
    
    # Price transition matrix P(m'|m) - rows sum to 1
    # Uniform random: price is i.i.d. each year
    price_transition: np.ndarray = field(default_factory=lambda: np.array([
        [0.34, 0.33, 0.33],  # From LOW
        [0.33, 0.34, 0.33],  # From MEDIUM
        [0.33, 0.33, 0.34],  # From HIGH
    ]))
    
    # Automatic thinning rule
    auto_thin_threshold: float = 150.0  # ft²/ac total BA - thin if above this
    auto_thin_target: float = 100.0     # ft²/ac after thinning
    
    # Economics
    harvest_cost_per_ton: float = 15.0   # $/ton
    thin_cost_multiplier: float = 1.20   # Thinning costs 20% more
    thin_revenue_multiplier: float = 0.70  # Thinning gets 70% of harvest price
    disturbed_harvest_multiplier: float = 0.50  # Disturbed stands get 50% of normal value (salvage)
    discount_rate: float = 0.05
    
    # Stand parameters
    si25: float = 60.0
    region: str = "ucp"
    initial_tpa: float = 600.0
    
    @property
    def gamma(self) -> float:
        """Discount factor."""
        return 1.0 / (1.0 + self.discount_rate)
    
    @property
    def n_stand_states(self) -> int:
        """Number of stand states (excluding price)."""
        return 5 * 4 * 2  # 5 BA × 4 TPA × 2 disturbance = 40
    
    @property
    def n_price_states(self) -> int:
        """Number of price states."""
        return 3
    
    @property
    def n_states(self) -> int:
        """Total number of MDP states."""
        return self.n_stand_states * self.n_price_states  # 120
    
    def classify_ba(self, ba: float) -> BALevel:
        """Classify continuous BA into discrete level."""
        if ba < self.ba_thresholds[0]:      # < 40
            return BALevel.VERY_LOW
        elif ba < self.ba_thresholds[1]:    # 40-70
            return BALevel.LOW
        elif ba < self.ba_thresholds[2]:    # 70-100
            return BALevel.MEDIUM
        elif ba < self.ba_thresholds[3]:    # 100-130
            return BALevel.HIGH
        else:                                # >= 130
            return BALevel.VERY_HIGH
    
    def classify_tpa(self, tpa: float) -> TPALevel:
        """Classify continuous TPA into discrete level."""
        if tpa < self.tpa_thresholds[0]:    # < 200
            return TPALevel.LOW
        elif tpa < self.tpa_thresholds[1]:  # 200-350
            return TPALevel.MEDIUM
        elif tpa < self.tpa_thresholds[2]:  # 350-500
            return TPALevel.HIGH
        else:                                # >= 500
            return TPALevel.VERY_HIGH
    
    def get_representative_ba(self, level: BALevel) -> float:
        """Get representative BA value for a discrete level."""
        # Midpoint of each bucket
        if level == BALevel.VERY_LOW:
            return self.ba_thresholds[0] / 2  # 20
        elif level == BALevel.LOW:
            return (self.ba_thresholds[0] + self.ba_thresholds[1]) / 2  # 55
        elif level == BALevel.MEDIUM:
            return (self.ba_thresholds[1] + self.ba_thresholds[2]) / 2  # 85
        elif level == BALevel.HIGH:
            return (self.ba_thresholds[2] + self.ba_thresholds[3]) / 2  # 115
        else:  # VERY_HIGH
            return self.ba_thresholds[3] + 20  # 150
    
    def get_representative_tpa(self, level: TPALevel) -> float:
        """Get representative TPA value for a discrete level."""
        # Midpoint of each bucket
        if level == TPALevel.LOW:
            return self.tpa_thresholds[0] / 2  # 100
        elif level == TPALevel.MEDIUM:
            return (self.tpa_thresholds[0] + self.tpa_thresholds[1]) / 2  # 275
        elif level == TPALevel.HIGH:
            return (self.tpa_thresholds[1] + self.tpa_thresholds[2]) / 2  # 425
        else:  # VERY_HIGH
            return self.tpa_thresholds[2] + 75  # 575


# =============================================================================
# State Discretization
# =============================================================================

class BuongiornoDiscretizer:
    """Discretize continuous stand state into discrete ForestState.
    
    Maps continuous (BA, TPA) → discrete ForestState by:
    1. Classifying BA as LOW/MEDIUM/HIGH
    2. Classifying TPA as LOW/MEDIUM/HIGH
    3. Tracking disturbance history
    4. Including current price state
    """
    
    def __init__(self, config: BuongiornoConfig, pmrc: PMRCModel):
        self.config = config
        self.pmrc = pmrc
    
    def discretize(
        self,
        state: StandState,
        disturbed: bool = False,
        price: PriceState = PriceState.MEDIUM,
    ) -> ForestState:
        """Convert continuous stand state to discrete ForestState.
        
        Args:
            state: Continuous stand state
            disturbed: Whether stand was recently disturbed
            price: Current price state
            
        Returns:
            Discrete ForestState
        """
        ba_level = self.config.classify_ba(state.ba)
        tpa_level = self.config.classify_tpa(state.tpa)
        
        return ForestState(
            ba=ba_level,
            tpa=tpa_level,
            disturbed=DisturbanceState.RECENTLY_DISTURBED if disturbed else DisturbanceState.NORMAL,
            price=price,
        )
    
    def state_to_index(self, state: ForestState) -> int:
        """Convert ForestState to integer index."""
        return state.to_index()
    
    def index_to_state(self, idx: int) -> ForestState:
        """Convert integer index to ForestState."""
        return ForestState.from_index(idx)
    
    def get_representative_continuous_state(
        self,
        discrete_state: ForestState,
    ) -> StandState:
        """Create a representative continuous state for a discrete state.
        
        Used for simulation-based transition estimation.
        """
        # Get representative BA and TPA values
        ba = self.config.get_representative_ba(discrete_state.ba)
        tpa = self.config.get_representative_tpa(discrete_state.tpa)
        
        # Estimate age from BA (heuristic: higher BA = older stand)
        # BA ~30 at age 6, ~100 at age 15, ~150 at age 30
        if ba < 60:
            estimated_age = 5.0 + (ba / 60.0) * 10.0  # 5-15 years
        elif ba < 120:
            estimated_age = 15.0 + ((ba - 60) / 60.0) * 10.0  # 15-25 years
        else:
            estimated_age = 25.0 + ((ba - 120) / 60.0) * 10.0  # 25-35 years
        
        # Compute HD from age and SI
        k, m = self.pmrc.k, self.pmrc.m
        hd = self.config.si25 * (
            (1 - np.exp(-k * max(1.0, estimated_age))) /
            (1 - np.exp(-k * 25.0))
        ) ** m
        
        return StandState(
            age=estimated_age,
            hd=hd,
            tpa=tpa,
            ba=ba,
            si25=self.config.si25,
            region="ucp",  # type: ignore[arg-type]
        )


# =============================================================================
# Action Space
# =============================================================================

class Action(IntEnum):
    """Management actions.
    
    WAIT: Let stand grow for one year (with deterministic thinning at age 15)
    
    Note: No explicit actions - thinning is deterministic at age 15 when BA > threshold.
    HARVEST is terminal only - applied at end of horizon (year 30).
    """
    WAIT = 0     # Grow one year (auto-thin at age 15 if BA > threshold)


# =============================================================================
# Transition Matrix Estimation
# =============================================================================

def estimate_stand_transitions(
    config: BuongiornoConfig,
    risk_profile: RiskProfile,
    n_samples: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """Estimate stand transition matrix P(s'|s, a) via simulation.
    
    For each (stand_state, action) pair:
    1. Create representative continuous state
    2. Apply action (auto-thin if needed, or harvest)
    3. Simulate one year of stochastic growth
    4. Discretize result and count transitions
    
    Args:
        config: MDP configuration
        risk_profile: Risk parameters for stochastic simulation
        n_samples: Number of Monte Carlo samples per state
        seed: Random seed
        
    Returns:
        Transition matrices: dict[Action, np.ndarray] where each matrix is
        (n_stand_states × n_stand_states)
    """
    pmrc = PMRCModel(region=config.region)
    stoch = StochasticPMRC.from_config(pmrc, risk_profile.noise, risk_profile.disturbance)
    discretizer = BuongiornoDiscretizer(config, pmrc)
    
    n_stand = config.n_stand_states
    rng = np.random.default_rng(seed)
    
    # Initialize transition matrices for each action
    P = {
        Action.WAIT: np.zeros((n_stand, n_stand)),
        Action.HARVEST: np.zeros((n_stand, n_stand)),
    }
    
    # Reset state after harvest (young stand, normal, price unchanged)
    # We'll handle price separately
    reset_state = ForestState(
        ba=BALevel.VERY_LOW,
        tpa=TPALevel.VERY_HIGH,  # Replanted at high density
        disturbed=DisturbanceState.NORMAL,
        price=PriceState.MEDIUM,  # Price handled separately
    )
    reset_stand_idx = reset_state.to_index() % ForestState.N_STAND
    
    for s_idx in range(n_stand):
        # Reconstruct discrete state (stand only, no price)
        # Encoding: ba + 5*tpa + 20*disturbed (for 5 BA levels, 4 TPA levels)
        disturbed = DisturbanceState(s_idx // (ForestState.N_BA * ForestState.N_TPA))
        remainder = s_idx % (ForestState.N_BA * ForestState.N_TPA)
        tpa = TPALevel(remainder // ForestState.N_BA)
        ba = BALevel(remainder % ForestState.N_BA)
        
        discrete_state = ForestState(
            ba=ba,
            tpa=tpa,
            disturbed=disturbed,
            price=PriceState.MEDIUM,
        )
        
        # Get representative continuous state
        cont_state = discretizer.get_representative_continuous_state(discrete_state)
        
        # Action: WAIT (with automatic thinning)
        for _ in range(n_samples):
            # Apply automatic thinning if BA > threshold
            if cont_state.ba > config.auto_thin_threshold:
                thinned, _ = thin_to_residual_ba_smallest_first(
                    cont_state, config.auto_thin_target
                )
            else:
                thinned = cont_state
            
            # Simulate one year - get disturbance label from simulator
            next_cont, dist_label, _, _ = stoch.sample_next_state_with_trace(thinned, dt=1.0, rng=rng)
            
            # Disturbance state logic:
            # - If currently disturbed, stay disturbed (persistence)
            # - If disturbance occurred this step (mild or severe), become disturbed
            # - Otherwise, recover to normal
            if disturbed == DisturbanceState.RECENTLY_DISTURBED:
                # Disturbed stands stay disturbed for at least one more step
                # (50% chance to recover each year)
                next_disturbed = rng.random() < 0.5
            else:
                # Normal stands become disturbed if disturbance occurred
                next_disturbed = dist_label in ("mild", "severe")
            
            # Discretize
            next_discrete = discretizer.discretize(
                next_cont,
                disturbed=next_disturbed,
                price=PriceState.MEDIUM,
            )
            next_stand_idx = next_discrete.to_index() % ForestState.N_STAND
            P[Action.WAIT][s_idx, next_stand_idx] += 1.0
        
        # Action: HARVEST (deterministic reset)
        P[Action.HARVEST][s_idx, reset_stand_idx] = n_samples
    
    # Normalize rows
    for action in P:
        row_sums = P[action].sum(axis=1, keepdims=True)
        nonzero = row_sums[:, 0] > 0
        P[action][nonzero] /= row_sums[nonzero]
    
    return P


def build_full_transition_matrix(
    stand_transitions: dict[Action, np.ndarray],
    price_transition: np.ndarray,
    n_stand_states: int,
    n_price_states: int,
) -> dict[Action, np.ndarray]:
    """Build full transition matrix including price dynamics.
    
    Following Buongiorno eq. (5):
        S = [p(j|i)] = [p(s'|s) × p(m'|m)]
    
    The joint transition is the Kronecker product of stand and price transitions.
    
    Args:
        stand_transitions: P(s'|s, a) for each action
        price_transition: P(m'|m) Markov price model
        n_stand_states: Number of stand states
        n_price_states: Number of price states
        
    Returns:
        Full transition matrices for each action
    """
    n_total = n_stand_states * n_price_states
    full_P = {}
    
    for action, P_stand in stand_transitions.items():
        # Kronecker product: P_full[i,j] = P_stand[s,s'] × P_price[m,m']
        # where i = s × n_price + m, j = s' × n_price + m'
        full_P[action] = np.kron(P_stand, price_transition)
    
    return full_P


# =============================================================================
# Reward Function
# =============================================================================

def make_reward_function(
    config: BuongiornoConfig,
    pmrc: PMRCModel,
) -> Callable[[int, Action, int], float]:
    """Create reward function R(s, a, s').
    
    Rewards:
    - WAIT: $0 (no immediate cash flow; thinning at age 15 is handled in simulation, not here)
    - HARVEST: Harvest revenue minus costs, scaled by price state and disturbance
    
    Note: Thinning is NOT an MDP action. It's a deterministic rule applied only at age 15
    during the transition simulation, not a decision the agent makes.
    
    Args:
        config: MDP configuration
        pmrc: PMRC model for volume calculations
        
    Returns:
        Reward function R(state_idx, action, next_state_idx) -> float
    """
    discretizer = BuongiornoDiscretizer(config, pmrc)
    
    def reward_fn(s: int, a: Action, s_next: int) -> float:
        if a == Action.WAIT:
            return 0.0  # No immediate reward for waiting
        
        # HARVEST action
        state = ForestState.from_index(s)
        price_mult = config.price_levels[state.price]
        
        # Disturbed stands have reduced value (salvage logging)
        disturbed_mult = config.disturbed_harvest_multiplier if state.disturbed == DisturbanceState.RECENTLY_DISTURBED else 1.0
        
        # Get continuous state for value calculation
        cont_state = discretizer.get_representative_continuous_state(state)
        
        if cont_state.ba <= 0 or cont_state.tpa <= 0:
            return 0.0
        
        # Compute harvest value using product distribution
        products = estimate_product_distribution(
            pmrc, cont_state.ba, cont_state.tpa, cont_state.hd,
            region="ucp",  # type: ignore[arg-type]
        )
        
        # compute_harvest_value already subtracts logging + replanting costs
        net_value = compute_harvest_value(products) * price_mult * disturbed_mult
        
        return net_value
    
    return reward_fn


# =============================================================================
# MDP Solvers
# =============================================================================

@dataclass
class MDPSolution:
    """Solution to the MDP."""
    V: np.ndarray           # Value function V(s)
    Q: np.ndarray           # Action-value function Q(s, a)
    policy: np.ndarray      # Optimal policy π(s) = argmax_a Q(s, a)
    iterations: int         # Number of iterations to converge
    converged: bool         # Whether algorithm converged


def value_iteration(
    P: dict[Action, np.ndarray],
    reward_fn: Callable[[int, Action, int], float],
    gamma: float = 0.95,
    theta: float = 1e-6,
    max_iter: int = 1000,
) -> MDPSolution:
    """Solve infinite-horizon MDP via value iteration.
    
    Args:
        P: Transition matrices P[action][s, s']
        reward_fn: R(s, a, s') reward function
        gamma: Discount factor
        theta: Convergence threshold
        max_iter: Maximum iterations
        
    Returns:
        MDPSolution with V, Q, policy
    """
    actions = sorted(P.keys())
    n_states = P[actions[0]].shape[0]
    n_actions = len(actions)
    
    V = np.zeros(n_states)
    Q = np.zeros((n_states, n_actions))
    policy = np.zeros(n_states, dtype=int)
    
    for iteration in range(max_iter):
        delta = 0.0
        
        for s in range(n_states):
            v_old = V[s]
            
            for ai, a in enumerate(actions):
                q_sa = 0.0
                for s_next in range(n_states):
                    if P[a][s, s_next] > 0:
                        r = reward_fn(s, a, s_next)
                        q_sa += P[a][s, s_next] * (r + gamma * V[s_next])
                Q[s, ai] = q_sa
            
            V[s] = np.max(Q[s])
            policy[s] = actions[np.argmax(Q[s])]
            delta = max(delta, abs(V[s] - v_old))
        
        if delta < theta:
            return MDPSolution(V=V, Q=Q, policy=policy, iterations=iteration+1, converged=True)
    
    return MDPSolution(V=V, Q=Q, policy=policy, iterations=max_iter, converged=False)


def finite_horizon_value_iteration(
    P: dict[Action, np.ndarray],
    reward_fn: Callable[[int, Action, int], float],
    terminal_reward_fn: Callable[[int], float],
    horizon: int,
    gamma: float = 0.95,
) -> MDPSolution:
    """Solve finite-horizon MDP via backward induction.
    
    Args:
        P: Transition matrices P[action][s, s']
        reward_fn: R(s, a, s') reward function
        terminal_reward_fn: R_T(s) terminal reward (mandatory harvest)
        horizon: Number of time steps
        gamma: Discount factor
        
    Returns:
        MDPSolution with V, Q, policy at t=0
    """
    actions = sorted(P.keys())
    n_states = P[actions[0]].shape[0]
    n_actions = len(actions)
    
    # V[t, s] and Q[t, s, a]
    V = np.zeros((horizon + 1, n_states))
    Q = np.zeros((horizon + 1, n_states, n_actions))
    policy = np.zeros((horizon + 1, n_states), dtype=int)
    
    # Terminal values (mandatory harvest)
    for s in range(n_states):
        V[horizon, s] = terminal_reward_fn(s)
    
    # Backward induction
    for t in range(horizon - 1, -1, -1):
        for s in range(n_states):
            for ai, a in enumerate(actions):
                q_sa = 0.0
                for s_next in range(n_states):
                    if P[a][s, s_next] > 0:
                        r = reward_fn(s, a, s_next)
                        q_sa += P[a][s, s_next] * (r + gamma * V[t + 1, s_next])
                Q[t, s, ai] = q_sa
            
            V[t, s] = np.max(Q[t, s])
            policy[t, s] = actions[np.argmax(Q[t, s])]
    
    # Return t=0 solution
    return MDPSolution(
        V=V[0],
        Q=Q[0],
        policy=policy[0],
        iterations=horizon,
        converged=True,
    )


def policy_iteration(
    P: dict[Action, np.ndarray],
    reward_fn: Callable[[int, Action, int], float],
    gamma: float = 0.95,
    theta: float = 1e-6,
    max_iter: int = 100,
) -> MDPSolution:
    """Solve MDP via policy iteration.
    
    Args:
        P: Transition matrices P[action][s, s']
        reward_fn: R(s, a, s') reward function
        gamma: Discount factor
        theta: Convergence threshold for policy evaluation
        max_iter: Maximum policy improvement iterations
        
    Returns:
        MDPSolution with V, Q, policy
    """
    actions = sorted(P.keys())
    n_states = P[actions[0]].shape[0]
    n_actions = len(actions)
    
    # Initialize with WAIT policy
    policy = np.zeros(n_states, dtype=int)
    V = np.zeros(n_states)
    Q = np.zeros((n_states, n_actions))
    
    for iteration in range(max_iter):
        # Policy evaluation
        for _ in range(1000):
            delta = 0.0
            for s in range(n_states):
                v_old = V[s]
                a = policy[s]
                v_new = 0.0
                for s_next in range(n_states):
                    if P[a][s, s_next] > 0:
                        r = reward_fn(s, a, s_next)
                        v_new += P[a][s, s_next] * (r + gamma * V[s_next])
                V[s] = v_new
                delta = max(delta, abs(V[s] - v_old))
            if delta < theta:
                break
        
        # Policy improvement
        policy_stable = True
        for s in range(n_states):
            old_action = policy[s]
            for ai, a in enumerate(actions):
                q_sa = 0.0
                for s_next in range(n_states):
                    if P[a][s, s_next] > 0:
                        r = reward_fn(s, a, s_next)
                        q_sa += P[a][s, s_next] * (r + gamma * V[s_next])
                Q[s, ai] = q_sa
            
            policy[s] = actions[np.argmax(Q[s])]
            if old_action != policy[s]:
                policy_stable = False
        
        if policy_stable:
            return MDPSolution(V=V, Q=Q, policy=policy, iterations=iteration+1, converged=True)
    
    return MDPSolution(V=V, Q=Q, policy=policy, iterations=max_iter, converged=False)


# =============================================================================
# One-Step Lookahead (Planner)
# =============================================================================

def one_step_lookahead(
    state_idx: int,
    P: dict[Action, np.ndarray],
    reward_fn: Callable[[int, Action, int], float],
    V: np.ndarray,
    gamma: float = 0.95,
) -> dict[Action, float]:
    """Compute expected value of each action from a given state.
    
    This is the planner's one-step lookahead:
        Q(s, a) = Σ_s' P(s'|s,a) [R(s,a,s') + γV(s')]
    
    Args:
        state_idx: Current state index
        P: Transition matrices
        reward_fn: Reward function
        V: Value function
        gamma: Discount factor
        
    Returns:
        Dict mapping action to expected value
    """
    actions = sorted(P.keys())
    result = {}
    
    for a in actions:
        q_sa = 0.0
        n_states = P[a].shape[0]
        for s_next in range(n_states):
            if P[a][state_idx, s_next] > 0:
                r = reward_fn(state_idx, a, s_next)
                q_sa += P[a][state_idx, s_next] * (r + gamma * V[s_next])
        result[a] = q_sa
    
    return result


# =============================================================================
# Convenience Functions
# =============================================================================

def solve_mdp_for_risk_level(
    risk_level: Literal["low", "medium", "high"],
    config: BuongiornoConfig | None = None,
    horizon: int | None = None,
    n_samples: int = 1000,
    seed: int = 42,
) -> tuple[MDPSolution, dict[Action, np.ndarray], BuongiornoConfig]:
    """Solve MDP for a given risk level.
    
    Args:
        risk_level: "low", "medium", or "high"
        config: MDP configuration (uses defaults if None)
        horizon: If provided, solve finite-horizon; else infinite-horizon
        n_samples: Monte Carlo samples for transition estimation
        seed: Random seed
        
    Returns:
        (solution, transition_matrices, config)
    """
    if config is None:
        config = BuongiornoConfig()
    
    pmrc = PMRCModel(region=config.region)
    risk_profile = get_risk_profile(risk_level)
    
    # Estimate transitions
    stand_P = estimate_stand_transitions(config, risk_profile, n_samples, seed)
    full_P = build_full_transition_matrix(
        stand_P, config.price_transition,
        config.n_stand_states, config.n_price_states
    )
    
    # Create reward function
    reward_fn = make_reward_function(config, pmrc)
    
    # Solve
    if horizon is not None:
        # Finite horizon with mandatory terminal harvest
        def terminal_fn(s: int) -> float:
            return reward_fn(s, Action.HARVEST, 0)  # Force harvest at terminal
        
        solution = finite_horizon_value_iteration(
            full_P, reward_fn, terminal_fn, horizon, config.gamma
        )
    else:
        # Infinite horizon
        solution = value_iteration(full_P, reward_fn, config.gamma)
    
    return solution, full_P, config


def print_policy_table(
    solution: MDPSolution,
    config: BuongiornoConfig,
) -> None:
    """Print human-readable policy table."""
    print("\n" + "=" * 70)
    print("OPTIMAL POLICY TABLE")
    print("=" * 70)
    print(f"{'State':<40} {'Action':<10} {'V(s)':<12}")
    print("-" * 70)
    
    for s_idx in range(config.n_states):
        state = ForestState.from_index(s_idx)
        action = Action(solution.policy[s_idx])
        value = solution.V[s_idx]
        
        state_str = (
            f"BA({state.ba_small.name[0]}/{state.ba_medium.name[0]}/{state.ba_large.name[0]}) "
            f"Dist={state.disturbed.name[:4]} Price={state.price.name}"
        )
        print(f"{state_str:<40} {action.name:<10} ${value:>10.0f}")


def print_q_table(
    solution: MDPSolution,
    config: BuongiornoConfig,
) -> None:
    """Print Q(s,a) table."""
    print("\n" + "=" * 70)
    print("Q-VALUE TABLE")
    print("=" * 70)
    print(f"{'State':<40} {'Q(WAIT)':<12} {'Q(HARVEST)':<12} {'Best':<10}")
    print("-" * 70)
    
    for s_idx in range(config.n_states):
        state = ForestState.from_index(s_idx)
        q_wait = solution.Q[s_idx, 0]
        q_harvest = solution.Q[s_idx, 1]
        best = "WAIT" if q_wait > q_harvest else "HARVEST"
        
        state_str = (
            f"BA({state.ba_small.name[0]}/{state.ba_medium.name[0]}/{state.ba_large.name[0]}) "
            f"Dist={state.disturbed.name[:4]} Price={state.price.name}"
        )
        print(f"{state_str:<40} ${q_wait:>10.0f} ${q_harvest:>10.0f} {best:<10}")
