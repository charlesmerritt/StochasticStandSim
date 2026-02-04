"""Stochastic PMRC wrapper with size classes and MDP utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Literal, Sequence, Tuple, List, Optional

import numpy as np

if TYPE_CHECKING:
    from core.config import NoiseParams, DisturbanceParams, ActionSpec

try:
    from core.pmrc_model import PMRCModel, Region, WeibullParams
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    from core.pmrc_model import PMRCModel, Region, WeibullParams

Action = Literal[0, 1, 2]


@dataclass
class StandState:
    """Continuous stand description for stochastic simulation."""

    age: float
    hd: float
    tpa: float
    ba: float
    si25: float
    region: Region
    phwd: float = 0.0


@dataclass
class TransitionTrace:
    """Debug trace capturing deterministic vs stochastic components of a transition.
    
    This enables explainability of stochastic transitions for thesis documentation.
    All deltas are computed as (stochastic_value - deterministic_mean).
    """
    # Deterministic means from PMRC projection
    hd_mean: float
    tpa_mean: float
    ba_mean: float
    
    # Realized stochastic values (before disturbance)
    hd_realized: float
    tpa_realized: float
    ba_realized: float
    
    # Noise deltas (realized - mean)
    delta_hd: float
    delta_tpa: float
    delta_ba: float
    
    # Disturbance info
    disturbance_label: str | None = None
    disturbance_tpa_loss: float = 0.0
    disturbance_hd_loss: float = 0.0
    
    # Recruitment
    recruitment: float = 0.0


@dataclass
class SizeClassDistribution:
    """Diameter class counts derived from a Weibull representation."""

    dbh_bounds: np.ndarray
    tpa_per_class: np.ndarray
    ba_per_class: np.ndarray

    def validate(self) -> None:
        if self.dbh_bounds.ndim != 1 or self.dbh_bounds.size < 2:
            raise ValueError("dbh_bounds must be 1D with length >= 2.")
        if np.any(np.diff(self.dbh_bounds) <= 0):
            raise ValueError("dbh_bounds must be strictly increasing.")
        if np.any(self.tpa_per_class < 0.0) or np.any(self.ba_per_class < 0.0):
            raise ValueError("Class values must be non-negative.")
        if self.tpa_per_class.shape != self.ba_per_class.shape:
            raise ValueError("TPA and BA arrays must have matching shapes.")


def weibull_cdf(dbh: float, params: WeibullParams) -> float:
    """CDF of the three-parameter Weibull distribution."""
    if dbh <= params.a or params.b <= 0.0 or params.c <= 0.0:
        return 0.0
    x = (dbh - params.a) / params.b
    if x <= 0.0:
        return 0.0
    return 1.0 - np.exp(-(x ** params.c))


def weibull_pdf(dbh: float, params: WeibullParams) -> float:
    """PDF of the three-parameter Weibull distribution."""
    if dbh <= params.a or params.b <= 0.0 or params.c <= 0.0:
        return 0.0
    x = (dbh - params.a) / params.b
    if x <= 0.0:
        return 0.0
    return (params.c / params.b) * (x ** (params.c - 1.0)) * np.exp(-(x ** params.c))


def weibull_from_ba_tpa(
    ba: float,
    tpa: float,
    *,
    location_fraction: float = 0.3,
    shape: float = 3.0,
) -> WeibullParams:
    """
    Heuristic mapping from basal area / TPA to Weibull parameters.
    """
    if ba <= 0.0 or tpa <= 0.0:
        return WeibullParams(a=0.0, b=1.0, c=shape)
    qmd = PMRCModel.qmd(tpa=tpa, ba=ba)
    loc = max(0.0, location_fraction * qmd)
    c = max(0.5, shape)
    # ensure the mean diameter matches qmd approximately
    # For a 3-parameter Weibull, mean of dbh^2 approx (loc^2 + 2*loc*b + b^2*gamma)
    # Use proportional scale
    b = max(0.5, 0.6 * qmd)
    return WeibullParams(a=loc, b=b, c=c)


def size_class_distribution_from_state(
    state: StandState,
    dbh_bounds: np.ndarray,
) -> SizeClassDistribution:
    """Construct a size class distribution using a heuristic Weibull.
    
    The distribution is scaled so that total TPA and BA match the stand state.
    """
    params = weibull_from_ba_tpa(state.ba, state.tpa)
    trees = []
    basals = []
    for lo, hi in zip(dbh_bounds[:-1], dbh_bounds[1:]):
        prob = max(0.0, weibull_cdf(hi, params) - weibull_cdf(lo, params))
        tpa_class = state.tpa * prob
        d_mid = 0.5 * (lo + hi)
        ba_tree = 0.005454154 * (d_mid ** 2)
        trees.append(tpa_class)
        basals.append(tpa_class * ba_tree)
    
    tpa_arr = np.asarray(trees, dtype=float)
    ba_arr = np.asarray(basals, dtype=float)
    
    # Scale BA to match actual stand BA (Weibull approximation may not sum exactly)
    ba_sum = np.sum(ba_arr)
    if ba_sum > 0:
        ba_arr = ba_arr * (state.ba / ba_sum)
    
    dist = SizeClassDistribution(
        dbh_bounds=dbh_bounds.copy(),
        tpa_per_class=tpa_arr,
        ba_per_class=ba_arr,
    )
    dist.validate()
    return dist


# Default DBH class bounds for thinning (inches)
# Sub-merchantable: 0-6", Pulpwood: 6-9", CNS: 9-12", Sawtimber: 12-24"
DEFAULT_DBH_BOUNDS = np.array([0.0, 6.0, 9.0, 12.0, 24.0])


def thin_smallest_first(
    dist: SizeClassDistribution,
    target_ba_removal: float,
) -> SizeClassDistribution:
    """Remove trees from smallest diameter classes first until target BA removal is met.
    
    This implements "low thinning" or "thinning from below" where the smallest
    trees are removed first to reduce competition and favor larger crop trees.
    
    Args:
        dist: Current size class distribution
        target_ba_removal: BA to remove (ft²/ac)
    
    Returns:
        New SizeClassDistribution after thinning
    
    Note:
        The approximation assumes uniform removal within each class. Trees are
        removed from the smallest class first, then the next smallest, etc.
        until the target BA removal is achieved.
    """
    if target_ba_removal <= 0:
        return dist
    
    tpa_new = dist.tpa_per_class.copy()
    ba_new = dist.ba_per_class.copy()
    ba_remaining = target_ba_removal
    
    # Remove from smallest classes first (index 0 is smallest)
    for i in range(len(ba_new)):
        if ba_remaining <= 0:
            break
        
        class_ba = ba_new[i]
        if class_ba <= 0:
            continue
        
        if class_ba <= ba_remaining:
            # Remove entire class
            ba_remaining -= class_ba
            tpa_new[i] = 0.0
            ba_new[i] = 0.0
        else:
            # Partial removal from this class
            fraction_to_remove = ba_remaining / class_ba
            tpa_new[i] *= (1.0 - fraction_to_remove)
            ba_new[i] *= (1.0 - fraction_to_remove)
            ba_remaining = 0.0
    
    return SizeClassDistribution(
        dbh_bounds=dist.dbh_bounds.copy(),
        tpa_per_class=tpa_new,
        ba_per_class=ba_new,
    )


def thin_to_residual_ba_smallest_first(
    state: StandState,
    residual_ba: float,
    dbh_bounds: np.ndarray | None = None,
) -> tuple[StandState, SizeClassDistribution]:
    """Thin a stand to residual BA by removing smallest trees first.
    
    This is the main entry point for smallest-tree-first thinning. It:
    1. Builds a Weibull-based size class distribution from the current state
    2. Removes trees from smallest classes until target BA is reached
    3. Returns the new stand state and the post-thin distribution
    
    Args:
        state: Current stand state
        residual_ba: Target BA after thinning (ft²/ac)
        dbh_bounds: Optional custom DBH class boundaries
    
    Returns:
        Tuple of (new_state, post_thin_distribution)
    
    Approximation notes:
        - Weibull parameters are derived heuristically from BA/TPA using QMD
        - Location parameter (a) = 0.3 * QMD
        - Scale parameter (b) = 0.6 * QMD  
        - Shape parameter (c) = 3.0 (fixed, typical for pine)
        - BA per class uses midpoint DBH: BA_tree = 0.005454154 * DBH²
    """
    if residual_ba >= state.ba:
        # No thinning needed
        if dbh_bounds is None:
            dbh_bounds = DEFAULT_DBH_BOUNDS
        dist = size_class_distribution_from_state(state, dbh_bounds)
        return state, dist
    
    if dbh_bounds is None:
        dbh_bounds = DEFAULT_DBH_BOUNDS
    
    # Build size class distribution
    dist_pre = size_class_distribution_from_state(state, dbh_bounds)
    
    # Calculate BA to remove
    ba_to_remove = state.ba - residual_ba
    
    # Thin from smallest classes first
    dist_post = thin_smallest_first(dist_pre, ba_to_remove)
    
    # Compute new stand-level values
    new_tpa = float(np.sum(dist_post.tpa_per_class))
    new_ba = float(np.sum(dist_post.ba_per_class))
    
    # Create new state
    new_state = StandState(
        age=state.age,
        hd=state.hd,  # Height unchanged by thinning
        tpa=max(1.0, new_tpa),  # Ensure at least 1 TPA
        ba=max(0.0, new_ba),
        si25=state.si25,
        region=state.region,
        phwd=state.phwd,
    )
    
    return new_state, dist_post


class StochasticPMRC:
    """Stochastic wrapper for PMRCModel with disturbances and recruitment."""

    def __init__(
        self,
        pmrc: PMRCModel,
        *,
        sigma_log_ba: float = 0.14,
        sigma_tpa: float = 30.0,
        sigma_log_hd: Optional[float] = None,
        use_binomial_tpa: bool = True,
        p_mild: float = 0.02,
        severe_mean_interval: float = 25.0,
        mild_tpa_multiplier: float = 0.8,
        severe_tpa_multiplier: float = 0.4,
        mild_hd_multiplier: float = 0.95,
        severe_hd_multiplier: float = 0.8,
        severe_reset_age: float = 0.5,
        severe_reset_tpa: float = 700.0,
        recruitment_alpha: Tuple[float, float, float] = (1.0, -0.005, 0.02),
    ) -> None:
        self.pmrc = pmrc
        self.sigma_log_ba = sigma_log_ba
        self.sigma_tpa = sigma_tpa
        self.sigma_log_hd = sigma_log_hd
        self.use_binomial_tpa = use_binomial_tpa
        self.p_mild = p_mild
        self.severe_mean_interval = severe_mean_interval
        self.mild_tpa_multiplier = mild_tpa_multiplier
        self.severe_tpa_multiplier = severe_tpa_multiplier
        self.mild_hd_multiplier = mild_hd_multiplier
        self.severe_hd_multiplier = severe_hd_multiplier
        self.recruitment_alpha = recruitment_alpha
        self.severe_reset_age = severe_reset_age
        self.severe_reset_tpa = severe_reset_tpa

    @classmethod
    def from_config(
        cls,
        pmrc: PMRCModel,
        noise: "NoiseParams",
        disturbance: "DisturbanceParams",
    ) -> "StochasticPMRC":
        """Create StochasticPMRC from config dataclasses.
        
        Args:
            pmrc: Deterministic PMRC model
            noise: NoiseParams from core.config
            disturbance: DisturbanceParams from core.config
        """
        return cls(
            pmrc=pmrc,
            sigma_log_ba=noise.sigma_log_ba,
            sigma_tpa=noise.sigma_tpa,
            sigma_log_hd=noise.sigma_log_hd,
            use_binomial_tpa=noise.use_binomial_tpa,
            p_mild=disturbance.p_mild,
            severe_mean_interval=disturbance.severe_mean_interval,
            mild_tpa_multiplier=disturbance.mild_tpa_multiplier,
            severe_tpa_multiplier=disturbance.severe_tpa_multiplier,
            mild_hd_multiplier=disturbance.mild_hd_multiplier,
            severe_hd_multiplier=disturbance.severe_hd_multiplier,
            severe_reset_age=disturbance.severe_reset_age,
            severe_reset_tpa=disturbance.severe_reset_tpa,
        )

    def _gaussian_sample(self, mean: float, sigma: float, rng: np.random.Generator, min_val: float = 0.0) -> float:
        """Sample from Gaussian centered on mean, clipped to min_val."""
        if mean <= 0.0:
            return min_val
        return max(min_val, mean + sigma * mean * rng.normal())

    def sample_recruitment(self, state: StandState, rng: np.random.Generator) -> float:
        """Sample new trees per acre for the smallest class."""
        a0, a1, a2 = self.recruitment_alpha
        lam = max(0.0, a0 + a1 * state.ba + a2 * state.si25)
        return float(rng.poisson(lam))

    def _apply_disturbance_with_label(
        self,
        tpa: float,
        ba: float,
        hd: float,
        rng: np.random.Generator,
        dt: float,
    ) -> Tuple[float, float, float, Optional[str]]:
        label: Optional[str]
        dt_years = max(dt, 0.0)
        severe_prob = 1.0 - np.exp(-dt_years / max(self.severe_mean_interval, 1e-6))
        rand = rng.random()
        if rand < severe_prob:
            factor_tpa = self.severe_tpa_multiplier
            factor_hd = self.severe_hd_multiplier
            label = "severe"
        elif rand < severe_prob + self.p_mild:
            factor_tpa = self.mild_tpa_multiplier
            factor_hd = self.mild_hd_multiplier
            label = "mild"
        else:
            factor_tpa = 1.0
            factor_hd = 1.0
            label = None
        tpa = max(self.pmrc.min_tpa_asymptote, tpa * factor_tpa)
        ba = max(0.0, ba * factor_tpa)
        hd = max(0.0, hd * factor_hd)
        return tpa, ba, hd, label

    def sample_next_state(
        self,
        state: StandState,
        dt: float,
        rng: np.random.Generator,
    ) -> StandState:
        """Sample the next stand state after dt years."""
        next_state, _, _ = self.sample_next_state_with_event(state, dt, rng)
        return next_state

    def sample_next_state_with_event(
        self,
        state: StandState,
        dt: float,
        rng: np.random.Generator,
    ) -> Tuple[StandState, Optional[str], Optional[float]]:
        """Return next state and disturbance severity flag if event occurs."""
        next_state, level, event_age, _ = self.sample_next_state_with_trace(state, dt, rng)
        return next_state, level, event_age

    def sample_next_state_with_trace(
        self,
        state: StandState,
        dt: float,
        rng: np.random.Generator,
    ) -> Tuple[StandState, Optional[str], Optional[float], TransitionTrace]:
        """Return next state, disturbance info, and a debug trace of the transition.
        
        The trace captures deterministic means, realized stochastic values,
        noise deltas, and disturbance impacts for explainability.
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        age2 = state.age + dt
        
        # Deterministic projections
        hd_mean = self.pmrc.hd_project(state.age, state.hd, age2)
        tpa_mean = self.pmrc.tpa_project(state.tpa, state.si25, state.age, age2)
        
        # Apply stochastic noise to HD
        if self.sigma_log_hd:
            hd2 = self._gaussian_sample(hd_mean, self.sigma_log_hd, rng, min_val=1.0)
        else:
            hd2 = hd_mean

        # Apply stochastic noise to TPA
        if self.use_binomial_tpa:
            n = max(0, int(round(state.tpa)))
            if n == 0:
                tpa2 = 0.0
            else:
                p_surv = max(0.0, min(1.0, tpa_mean / max(state.tpa, 1.0)))
                tpa2 = float(rng.binomial(n, p_surv))
        else:
            tpa2 = max(self.pmrc.min_tpa_asymptote, tpa_mean + self.sigma_tpa * rng.normal())

        # BA projection uses realized TPA and HD
        ba_mean = self.pmrc.ba_project(
            age1=state.age,
            tpa1=state.tpa,
            tpa2=tpa2,
            ba1=state.ba,
            hd1=state.hd,
            hd2=hd2,
            age2=age2,
            region=state.region,
        )
        ba2 = self._gaussian_sample(ba_mean, self.sigma_log_ba, rng, min_val=1.0)

        # Capture pre-disturbance values for trace
        hd_pre_dist = hd2
        tpa_pre_dist = tpa2
        ba_pre_dist = ba2

        # Apply disturbance
        tpa2, ba2, hd2, level = self._apply_disturbance_with_label(tpa2, ba2, hd2, rng, dt)
        
        # Recruitment
        recruits = 0.0 if level == "severe" else self.sample_recruitment(state, rng)
        tpa2 = max(self.pmrc.min_tpa_asymptote, tpa2 + recruits)
        
        # Cap BA
        if tpa2 > 0.0 and hd2 > 0.0:
            ba_cap = self.pmrc.ba_predict(age=age2, tpa=tpa2, hd=hd2, region=state.region)
            ba2 = min(ba2, ba_cap)

        # Build trace
        trace = TransitionTrace(
            hd_mean=hd_mean,
            tpa_mean=tpa_mean,
            ba_mean=ba_mean,
            hd_realized=hd_pre_dist,
            tpa_realized=tpa_pre_dist,
            ba_realized=ba_pre_dist,
            delta_hd=hd_pre_dist - hd_mean,
            delta_tpa=tpa_pre_dist - tpa_mean,
            delta_ba=ba_pre_dist - ba_mean,
            disturbance_label=level,
            disturbance_tpa_loss=tpa_pre_dist - tpa2 + recruits if level else 0.0,
            disturbance_hd_loss=hd_pre_dist - hd2 if level else 0.0,
            recruitment=recruits,
        )

        next_state = StandState(
            age=age2,
            hd=hd2,
            tpa=tpa2,
            ba=ba2,
            si25=state.si25,
            region=state.region,
            phwd=state.phwd,
        )
        event_age = None
        if level == "severe":
            event_age = age2
            next_state = self._reset_to_bare(state)
        return next_state, level, event_age, trace

    def _reset_to_bare(self, prior: StandState) -> StandState:
        """Return a nascent stand after a severe disturbance."""
        reset_age = max(0.5, self.severe_reset_age)
        hd_reset = self._hd_from_site(prior.si25, reset_age)
        tpa_reset = self.severe_reset_tpa
        ba_reset = self.pmrc.ba_predict(
            age=reset_age,
            tpa=tpa_reset,
            hd=max(hd_reset, 1e-3),
            region=prior.region,
        )
        return StandState(
            age=reset_age,
            hd=hd_reset,
            tpa=tpa_reset,
            ba=ba_reset,
            si25=prior.si25,
            region=prior.region,
            phwd=prior.phwd,
        )

    def _hd_from_site(self, si25: float, age: float) -> float:
        """Compute dominant height at a given age from site index."""
        age = max(0.5, age)
        num = 1.0 - np.exp(-self.pmrc.k * age)
        den = 1.0 - np.exp(-self.pmrc.k * 25.0)
        return max(1.0, si25 * (num / den) ** self.pmrc.m)


class StateDiscretizer:
    """Overlay a grid on continuous state space to assign discrete indices."""

    def __init__(
        self,
        age_bins: np.ndarray,
        tpa_bins: np.ndarray,
        ba_bins: np.ndarray,
    ) -> None:
        self.age_bins = age_bins
        self.tpa_bins = tpa_bins
        self.ba_bins = ba_bins

    @property
    def n_states(self) -> int:
        return (len(self.age_bins) - 1) * (len(self.tpa_bins) - 1) * (len(self.ba_bins) - 1)

    def encode(self, state: StandState) -> int:
        """Assign state to a bin index without modifying the state."""
        i_age = int(np.clip(np.digitize(state.age, self.age_bins) - 1, 0, len(self.age_bins) - 2))
        i_tpa = int(np.clip(np.digitize(state.tpa, self.tpa_bins) - 1, 0, len(self.tpa_bins) - 2))
        i_ba = int(np.clip(np.digitize(state.ba, self.ba_bins) - 1, 0, len(self.ba_bins) - 2))
        return i_age * ((len(self.tpa_bins) - 1) * (len(self.ba_bins) - 1)) + i_tpa * (len(self.ba_bins) - 1) + i_ba

    def decode(self, index: int) -> Tuple[int, int, int]:
        """Return bin coordinates for a discrete index."""
        n_tpa = len(self.tpa_bins) - 1
        n_ba = len(self.ba_bins) - 1
        n_plane = n_tpa * n_ba
        i_age = index // n_plane
        rem = index % n_plane
        i_tpa = rem // n_ba
        i_ba = rem % n_ba
        return i_age, i_tpa, i_ba


ThinMode = Literal["constant_qmd", "smallest_first"]


def apply_thin_to_state(
    state: StandState,
    residual_ba: float,
    pmrc: PMRCModel,
    *,
    mode: ThinMode = "constant_qmd",
) -> StandState:
    """Apply thinning to a StandState.
    
    Two thinning modes are supported:
    
    1. "constant_qmd" (default): Proportional removal across diameter classes.
       This matches Stand._apply_thin_event and preserves QMD.
       
    2. "smallest_first": Remove trees from smallest diameter classes first
       (low thinning / thinning from below). This increases mean tree size
       and matches the stated management rule.
    
    Args:
        state: Current stand state
        residual_ba: Target BA after thinning
        pmrc: PMRC model for QMD calculations
        mode: Thinning mode - "constant_qmd" or "smallest_first"
    
    Returns:
        New StandState with reduced BA and TPA
    """
    if residual_ba >= state.ba:
        return state  # No thinning needed
    
    if mode == "smallest_first":
        new_state, _ = thin_to_residual_ba_smallest_first(state, residual_ba)
        return new_state
    
    # Default: constant QMD
    qmd_pre = pmrc.qmd(tpa=state.tpa, ba=state.ba)
    post_tpa = pmrc.tpa_from_ba_qmd(ba=residual_ba, qmd_in=qmd_pre)
    post_tpa = max(pmrc.min_tpa_asymptote, post_tpa)
    
    return StandState(
        age=state.age,
        hd=state.hd,
        tpa=post_tpa,
        ba=residual_ba,
        si25=state.si25,
        region=state.region,
        phwd=state.phwd,
    )


def apply_harvest_replant(
    state: StandState,
    pmrc: PMRCModel,
    *,
    replant_tpa: float = 600.0,
    replant_age: float = 1.0,
) -> StandState:
    """Apply harvest and replant action.
    
    Args:
        state: Current stand state
        pmrc: PMRC model
        replant_tpa: TPA for new stand
        replant_age: Initial age of replanted stand
    
    Returns:
        New StandState representing freshly planted stand
    """
    hd_new = pmrc.hd_from_si(state.si25, form="projection")
    ba_new = pmrc.ba_predict(age=replant_age, tpa=replant_tpa, hd=hd_new, region=state.region)
    
    return StandState(
        age=replant_age,
        hd=hd_new,
        tpa=replant_tpa,
        ba=ba_new,
        si25=state.si25,
        region=state.region,
        phwd=state.phwd,
    )


def apply_action_to_state(
    state: StandState,
    action: Action,
    pmrc: PMRCModel,
    action_spec: Optional["ActionSpec"] = None,
) -> StandState:
    """Apply a discrete action to a StandState.
    
    Actions are defined by ActionSpec:
    - Action 0: No-op
    - Action 1..n-1: Thin to retention fraction
    - Action n (if harvest_replant): Harvest and replant
    
    Uses constant-QMD thinning logic consistent with Stand._apply_thin_event.
    """
    if action_spec is None:
        # Default: no-op=1.0, light=0.8, heavy=0.6
        thin_fractions = (1.0, 0.80, 0.60)
        harvest_replant = True
    else:
        thin_fractions = action_spec.thin_fractions
        harvest_replant = action_spec.harvest_replant
    
    n_thin = len(thin_fractions)
    
    # Check for harvest action
    if harvest_replant and action == n_thin:
        return apply_harvest_replant(state, pmrc)
    
    # Thin action
    if action < n_thin:
        retention = thin_fractions[action]
        if retention >= 1.0:
            return state  # No-op
        residual_ba = state.ba * retention
        return apply_thin_to_state(state, residual_ba, pmrc)
    
    # Invalid action - return unchanged
    return state


def estimate_transition_matrix(
    stochastic_pmrc: StochasticPMRC,
    discretizer: StateDiscretizer,
    actions: List[Action],
    dt: float,
    n_mc: int,
    rng: np.random.Generator,
    *,
    si25: float,
    region: Region,
    init_state: StandState,
    steps: int,
) -> Dict[Action, np.ndarray]:
    """Estimate transition probabilities via Monte Carlo simulation."""
    pmrc = stochastic_pmrc.pmrc
    matrices: Dict[Action, np.ndarray] = {
        a: np.zeros((discretizer.n_states, discretizer.n_states), dtype=float) for a in actions
    }

    for action in actions:
        counts = matrices[action]
        for _ in range(n_mc):
            state = StandState(
                age=init_state.age,
                hd=init_state.hd,
                tpa=init_state.tpa,
                ba=init_state.ba,
                si25=si25,
                region=region,
                phwd=init_state.phwd,
            )
            for _ in range(steps):
                s_idx = discretizer.encode(state)
                acted = apply_action_to_state(state, action, pmrc)
                state = stochastic_pmrc.sample_next_state(acted, dt, rng)
                s_next = discretizer.encode(state)
                counts[s_idx, s_next] += 1.0

        # normalize rows
        row_sums = counts.sum(axis=1, keepdims=True)
        nonzero = row_sums[:, 0] > 0
        counts[nonzero] /= row_sums[nonzero]

    return matrices


def generate_transition_matrices_for_profile(
    risk_level: str,
    discretizer: StateDiscretizer,
    n_samples_per_state: int = 100,
    dt: float = 1.0,
    si25: float = 60.0,
    region: Region = "ucp",
    seed: int = 42,
) -> dict[int, np.ndarray]:
    """Generate transition matrices for a risk profile.
    
    Samples transitions from each discrete state to estimate P(s'|s,a).
    
    Args:
        risk_level: "low", "medium", or "high"
        discretizer: State discretizer with bin definitions
        n_samples_per_state: MC samples per (state, action) pair
        dt: Time step (years)
        si25: Site index for height calculations
        region: PMRC region
        seed: Random seed for reproducibility
    
    Returns:
        Dict mapping action index to transition matrix (n_states x n_states)
    """
    from core.config import get_risk_profile
    
    profile = get_risk_profile(risk_level)
    pmrc = PMRCModel(region=region)
    stoch = StochasticPMRC.from_config(pmrc, profile.noise, profile.disturbance)
    
    n_states = discretizer.n_states
    n_actions = 4  # no-op, light thin, heavy thin, harvest
    
    matrices = {a: np.zeros((n_states, n_states), dtype=float) for a in range(n_actions)}
    rng = np.random.default_rng(seed)
    
    # For each discrete state, create a representative continuous state
    for s_idx in range(n_states):
        i_age, i_tpa, i_ba = discretizer.decode(s_idx)
        
        # Get bin midpoints
        age = 0.5 * (discretizer.age_bins[i_age] + discretizer.age_bins[i_age + 1])
        tpa = 0.5 * (discretizer.tpa_bins[i_tpa] + discretizer.tpa_bins[i_tpa + 1])
        ba = 0.5 * (discretizer.ba_bins[i_ba] + discretizer.ba_bins[i_ba + 1])
        
        # Compute HD from SI and age
        k, m = pmrc.k, pmrc.m
        hd = si25 * ((1 - np.exp(-k * max(1.0, age))) / (1 - np.exp(-k * 25.0))) ** m
        
        state = StandState(age=age, hd=hd, tpa=tpa, ba=ba, si25=si25, region=region)
        
        # Sample transitions for each action
        for action in range(n_actions):
            for _ in range(n_samples_per_state):
                # Apply action
                acted = apply_action_to_state(state, action, pmrc)
                # Sample next state
                next_state = stoch.sample_next_state(acted, dt, rng)
                s_next = discretizer.encode(next_state)
                matrices[action][s_idx, s_next] += 1.0
    
    # Normalize rows
    for action in range(n_actions):
        row_sums = matrices[action].sum(axis=1, keepdims=True)
        nonzero = row_sums[:, 0] > 0
        matrices[action][nonzero] /= row_sums[nonzero]
    
    return matrices


def validate_transition_matrices(
    matrices: dict[int, np.ndarray],
) -> dict[str, float]:
    """Validate transition matrices and return summary statistics.
    
    Returns:
        Dict with validation metrics
    """
    results = {}
    
    for action, P in matrices.items():
        row_sums = P.sum(axis=1)
        nonzero_rows = row_sums > 0
        
        if nonzero_rows.sum() == 0:
            results[f"action_{action}_valid_rows"] = 0
            continue
        
        # Check row sums are ~1
        valid_row_sums = row_sums[nonzero_rows]
        results[f"action_{action}_valid_rows"] = int(nonzero_rows.sum())
        results[f"action_{action}_row_sum_mean"] = float(valid_row_sums.mean())
        results[f"action_{action}_row_sum_std"] = float(valid_row_sums.std())
        
        # Sparsity
        results[f"action_{action}_sparsity"] = float((P == 0).sum() / P.size)
    
    return results


if __name__ == "__main__":
    pmrc = PMRCModel(region="ucp")
    stochastic = StochasticPMRC(pmrc)
    age_bins = np.linspace(0, 30, 6)
    tpa_bins = np.linspace(100, 600, 6)
    ba_bins = np.linspace(20, 200, 6)
    discretizer = StateDiscretizer(age_bins, tpa_bins, ba_bins)
    init_state = StandState(age=5.0, hd=40.0, tpa=500.0, ba=80.0, si25=60.0, region="ucp")
    rng = np.random.default_rng(42)
    for _ in range(3):
        next_state = stochastic.sample_next_state(init_state, dt=1.0, rng=rng)
        idx = discretizer.encode(next_state)
        print("Next state:", next_state)
        print("Discrete index:", idx)
        init_state = next_state
    matrices = estimate_transition_matrix(
        stochastic,
        discretizer,
        actions=[0, 1, 2],
        dt=1.0,
        n_mc=1000,
        rng=rng,
        si25=60.0,
        region="ucp",
        init_state=init_state,
        steps=30,
    )
    print("Transition probabilities for action 0 (first row):", matrices)
