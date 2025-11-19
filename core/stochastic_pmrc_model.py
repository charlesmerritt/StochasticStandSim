from typing import Optional
import math
import numpy as np
from .pmrc_model import PMRCModel
from .growth import StandState

    
class StochasticPMRC:
    """
    Stochastic wrapper for the deterministic PMRCModel.

    Adds process noise around the PMRC expectations for TPA and BA.
    """

    def __init__(
        self,
        pmrc: PMRCModel,
        sigma_log_ba: float = 0.14,
        use_binomial_tpa: bool = True,
        sigma_tpa: float = 30.0,
        sigma_log_hd: Optional[float] = None,
    ) -> None:
        """
        Parameters
        ----------
        pmrc :
            An instance of PMRCModel with the appropriate region set.
        sigma_log_ba :
            Standard deviation of log(BA) noise.
        use_binomial_tpa :
            If True, TPA is sampled with a Binomial around the PMRC survival expectation.
        sigma_tpa :
            Standard deviation for Gaussian TPA noise if use_binomial_tpa is False.
        sigma_log_hd :
            Optional standard deviation of log(HD) noise. If None, HD is deterministic.
        """
        self.pmrc = pmrc
        self.sigma_log_ba = sigma_log_ba
        self.use_binomial_tpa = use_binomial_tpa
        self.sigma_tpa = sigma_tpa
        self.sigma_log_hd = sigma_log_hd

    def sample_next_state(
        self,
        state: StandState,
        dt: float,
        rng: np.random.Generator,
    ) -> StandState:
        """
        Sample the next stand state after a time increment dt (in years).

        Uses PMRC equations for the mean transition and adds stochastic residuals.
        """
        age1 = state.age
        age2 = age1 + dt
        if age2 <= age1:
            raise ValueError("dt must be positive")

        # Site index is static, but you can recompute it from hd if you prefer
        si25 = state.si25

        # 1. Height: deterministic or lognormal noise
        hd_mean = self.pmrc.hd_project(age1=age1, hd1=state.hd, age2=age2)
        if self.sigma_log_hd is not None and hd_mean > 0.0:
            log_hd = math.log(hd_mean) + self.sigma_log_hd * rng.normal()
            hd2 = max(0.0, math.exp(log_hd))
        else:
            hd2 = hd_mean

        # 2. TPA: survival via PMRC, then stochastic
        tpa_mean = self.pmrc.tpa_project(
            tpa1=state.tpa,
            si25=si25,
            age1=age1,
            age2=age2,
        )

        if self.use_binomial_tpa:
            tpa1_int = max(0, int(round(state.tpa)))
            if tpa1_int == 0:
                tpa2 = 0.0
            else:
                p_surv = max(0.0, min(1.0, tpa_mean / max(1.0, state.tpa)))
                tpa2 = float(rng.binomial(n=tpa1_int, p=p_surv))
        else:
            tpa2 = tpa_mean + self.sigma_tpa * rng.normal()
            tpa2 = max(0.0, tpa2)

        # Enforce PMRC minimum asymptote logic if desired
        if tpa2 <= self.pmrc.min_tpa_asymptote:
            tpa2 = self.pmrc.min_tpa_asymptote

        # 3. BA: project deterministically, then lognormal noise
        ba_mean = self.pmrc.ba_project(
            age1=age1,
            tpa1=state.tpa,
            tpa2=tpa2,
            ba1=state.ba,
            hd1=state.hd,
            hd2=hd2,
            age2=age2,
            region=state.region,
        )

        if ba_mean > 0.0:
            log_ba = math.log(ba_mean) + self.sigma_log_ba * rng.normal()
            ba2 = max(0.0, math.exp(log_ba))
        else:
            ba2 = 0.0

        # 4. Return new state
        return StandState(
            age=age2,
            hd=hd2,
            tpa=tpa2,
            ba=ba2,
            si25=si25,
            region=state.region,
            phwd=state.phwd,
        )
