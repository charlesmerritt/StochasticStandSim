"""Process noise modeling for stochastic forest simulation.

This module handles aleatoric uncertainty from growth variability:
- Multiplicative lognormal noise on growth increments (BA, HD)
- Binomial/normal noise on TPA mortality
- Recruitment sampling (Poisson)

Per PLANNING.md Section 4.2, noise is applied to increments of atomic
state variables only, not to derived quantities like volume.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NoiseParams:
    """Parameters controlling process noise on growth increments.
    
    Aleatoric uncertainty parameters for year-to-year growth variability.
    
    Attributes:
        sigma_log_ba: Log-scale std dev for BA increment noise
        sigma_log_hd: Log-scale std dev for HD increment noise (None = no noise)
        sigma_tpa: Std dev for TPA noise (used if use_binomial_tpa=False)
        use_binomial_tpa: If True, use binomial mortality; else normal noise
        lambda_proc: Global noise scaling factor (0=off, 1=full)
    """
    sigma_log_ba: float = 0.14
    sigma_log_hd: float | None = None
    sigma_tpa: float = 30.0
    use_binomial_tpa: bool = True
    lambda_proc: float = 1.0


@dataclass
class NoiseRealization:
    """Record of noise applied to a single transition.
    
    Attributes:
        ba_multiplier: Multiplicative factor applied to BA increment
        hd_multiplier: Multiplicative factor applied to HD increment
        tpa_delta: Additive noise applied to TPA (negative = extra mortality)
        recruitment: New trees added to smallest class
    """
    ba_multiplier: float = 1.0
    hd_multiplier: float = 1.0
    tpa_delta: float = 0.0
    recruitment: float = 0.0


class ProcessNoiseModel:
    """Samples and applies process noise to growth increments.
    
    Implements the aleatoric uncertainty from growth variability as described
    in PLANNING.md Section 4.2:
    - Multiplicative lognormal noise on BA and HD increments
    - Mean-corrected lognormal to preserve expected increment
    - Binomial or normal noise on TPA mortality
    """

    def __init__(self, params: NoiseParams | None = None) -> None:
        self.params: NoiseParams = params or NoiseParams()

    def sample_ba_multiplier(self, rng: np.random.Generator) -> float:
        """Sample multiplicative noise for BA increment.
        
        Uses mean-corrected lognormal: exp(sigma*Z - 0.5*sigma^2)
        so E[multiplier] = 1.0.
        
        Args:
            rng: NumPy random generator
            
        Returns:
            Multiplicative factor for BA increment
        """
        if self.params.sigma_log_ba <= 0 or self.params.lambda_proc <= 0:
            return 1.0
        
        sigma = self.params.lambda_proc * self.params.sigma_log_ba
        z = rng.standard_normal()
        # Mean-corrected lognormal
        return float(np.exp(sigma * z - 0.5 * sigma**2))

    def sample_hd_multiplier(self, rng: np.random.Generator) -> float:
        """Sample multiplicative noise for HD increment.
        
        Args:
            rng: NumPy random generator
            
        Returns:
            Multiplicative factor for HD increment (1.0 if no HD noise)
        """
        if self.params.sigma_log_hd is None or self.params.sigma_log_hd <= 0:
            return 1.0
        if self.params.lambda_proc <= 0:
            return 1.0
        
        sigma = self.params.lambda_proc * self.params.sigma_log_hd
        z = rng.standard_normal()
        return float(np.exp(sigma * z - 0.5 * sigma**2))

    def sample_tpa_noise(
        self,
        tpa: float,
        expected_mortality: float,
        rng: np.random.Generator,
    ) -> float:
        """Sample noise for TPA change (mortality variability).
        
        Args:
            tpa: Current trees per acre
            expected_mortality: Expected TPA loss from deterministic projection
            rng: NumPy random generator
            
        Returns:
            Additional TPA change (negative = extra mortality beyond expected)
        """
        if self.params.lambda_proc <= 0:
            return 0.0
        
        if self.params.use_binomial_tpa and expected_mortality > 0:
            # Binomial mortality: each tree has p=expected_mortality/tpa chance of dying
            n_trees = int(round(tpa))
            if n_trees <= 0:
                return 0.0
            p_die = min(1.0, max(0.0, expected_mortality / tpa))
            actual_deaths = rng.binomial(n_trees, p_die)
            # Return deviation from expected
            return float(expected_mortality - actual_deaths)
        else:
            # Normal noise on TPA
            sigma = self.params.lambda_proc * self.params.sigma_tpa
            return float(rng.normal(0, sigma))

    def sample_recruitment(
        self,
        ba: float,
        si25: float,
        alpha: tuple[float, float, float] = (1.0, -0.005, 0.02),
        rng: np.random.Generator | None = None,
    ) -> float:
        """Sample new trees per acre (recruitment to smallest class).
        
        Recruitment rate: lambda = max(0, alpha0 + alpha1*BA + alpha2*SI25)
        
        Recruitment is disabled when lambda_proc=0 to ensure zero-noise
        recovery (stochastic model reduces to deterministic PMRC).
        
        Args:
            ba: Current basal area
            si25: Site index at base age 25
            alpha: Recruitment coefficients (intercept, BA effect, SI effect)
            rng: NumPy random generator
            
        Returns:
            Number of new trees to add
        """
        if rng is None:
            return 0.0
        
        # Disable recruitment when lambda_proc=0 for zero-noise recovery
        if self.params.lambda_proc <= 0:
            return 0.0
        
        a0, a1, a2 = alpha
        lam = max(0.0, a0 + a1 * ba + a2 * si25)
        return float(rng.poisson(lam))

    def apply_to_increments(
        self,
        delta_ba: float,
        delta_hd: float,
        tpa: float,
        expected_tpa_loss: float,
        rng: np.random.Generator,
    ) -> tuple[float, float, float, NoiseRealization]:
        """Apply noise to all growth increments.
        
        Args:
            delta_ba: Deterministic BA increment
            delta_hd: Deterministic HD increment
            tpa: Current TPA (for binomial mortality)
            expected_tpa_loss: Expected TPA loss from PMRC projection
            rng: NumPy random generator
            
        Returns:
            Tuple of (noisy_delta_ba, noisy_delta_hd, tpa_adjustment, realization)
        """
        ba_mult = self.sample_ba_multiplier(rng)
        hd_mult = self.sample_hd_multiplier(rng)
        tpa_delta = self.sample_tpa_noise(tpa, expected_tpa_loss, rng)
        
        noisy_delta_ba = delta_ba * ba_mult
        noisy_delta_hd = delta_hd * hd_mult
        
        realization = NoiseRealization(
            ba_multiplier=ba_mult,
            hd_multiplier=hd_mult,
            tpa_delta=tpa_delta,
        )
        
        return noisy_delta_ba, noisy_delta_hd, tpa_delta, realization
