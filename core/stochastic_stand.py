"""Stochastic PMRC wrapper with size classes and MDP utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Sequence, Tuple, List, Optional

import numpy as np

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
    """Construct a size class distribution using a heuristic Weibull."""
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
    dist = SizeClassDistribution(
        dbh_bounds=dbh_bounds.copy(),
        tpa_per_class=np.asarray(trees, dtype=float),
        ba_per_class=np.asarray(basals, dtype=float),
    )
    dist.validate()
    return dist


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

    def _lognormal_sample(self, mean: float, sigma: float, rng: np.random.Generator) -> float:
        if mean <= 0.0:
            return 0.0
        return float(np.exp(np.log(mean) + sigma * rng.normal()))

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
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        age2 = state.age + dt
        hd_mean = self.pmrc.hd_project(state.age, state.hd, age2)
        if self.sigma_log_hd:
            hd2 = self._lognormal_sample(hd_mean, self.sigma_log_hd, rng)
        else:
            hd2 = hd_mean

        tpa_mean = self.pmrc.tpa_project(state.tpa, state.si25, state.age, age2)
        if self.use_binomial_tpa:
            n = max(0, int(round(state.tpa)))
            if n == 0:
                tpa2 = 0.0
            else:
                p_surv = max(0.0, min(1.0, tpa_mean / max(state.tpa, 1.0)))
                tpa2 = float(rng.binomial(n, p_surv))
        else:
            tpa2 = max(self.pmrc.min_tpa_asymptote, tpa_mean + self.sigma_tpa * rng.normal())

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
        ba2 = self._lognormal_sample(max(ba_mean, 1e-6), self.sigma_log_ba, rng)

        tpa2, ba2, hd2, level = self._apply_disturbance_with_label(tpa2, ba2, hd2, rng, dt)
        recruits = 0.0 if level == "severe" else self.sample_recruitment(state, rng)
        tpa2 = max(self.pmrc.min_tpa_asymptote, tpa2 + recruits)
        if tpa2 > 0.0 and hd2 > 0.0:
            ba_cap = self.pmrc.ba_predict(age=age2, tpa=tpa2, hd=hd2, region=state.region)
            ba2 = min(ba2, ba_cap)

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
        return next_state, level, event_age

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


def apply_action_to_state(
    state: StandState,
    action: Action,
    pmrc: PMRCModel,
) -> StandState:
    """Apply an instantaneous thinning action."""
    fractions = {0: 1.0, 1: 0.8, 2: 0.6}
    factor = fractions[action]
    ba = state.ba * factor
    tpa = max(pmrc.min_tpa_asymptote, state.tpa * factor)
    return StandState(
        age=state.age,
        hd=state.hd,
        tpa=tpa,
        ba=ba,
        si25=state.si25,
        region=state.region,
        phwd=state.phwd,
    )


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
