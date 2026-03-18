"""Tests for stochastic simulation coherence with PMRC baseline.

Implements the 7 tests from PLANNING.md Section 11:
1. Zero-noise recovery
2. Mean-trend closeness under low process noise
3. Feasibility invariants
4. Disturbance frequency sanity
5. Severity distribution sanity
6. Monotonic risk intuition
7. Deterministic policy consistency
"""

from __future__ import annotations

import numpy as np
import pytest

from core.config import ScenarioConfig, ThinningParams
from core.disturbances import DisturbanceModel, DisturbanceParams
from core.pmrc_model import PMRCModel
from core.process_noise import NoiseParams
from core.metrics import compare_scenarios
from core.simulate import run_batch, run_scenario
from core.state import StandState
from core.stochastic_model import StochasticPMRC


# =============================================================================
# Test 1: Zero-noise recovery
# =============================================================================

class TestZeroNoiseRecovery:
    """If λ_proc=0 and p_dist=0, stochastic simulator must exactly reproduce deterministic PMRC."""

    def test_zero_noise_matches_deterministic(self):
        """Stochastic with zero noise should match deterministic exactly."""
        # Deterministic scenario
        det_config = ScenarioConfig(
            name="deterministic",
            scenario_type="deterministic",
        )
        det_result = run_scenario(det_config)

        # Stochastic with zero noise and zero disturbance
        stoch_config = ScenarioConfig(
            name="zero_noise",
            scenario_type="stochastic",
            noise_params=NoiseParams(lambda_proc=0.0),
            disturbance_params=DisturbanceParams(p_dist=0.0),
        )
        stoch_result = run_scenario(stoch_config, rng=np.random.default_rng(42))

        # Compare trajectories
        for det_rec, stoch_rec in zip(det_result.trajectory, stoch_result.trajectory):
            assert det_rec.age == pytest.approx(stoch_rec.age, rel=1e-6), \
                f"Age mismatch at year {det_rec.year}"
            assert det_rec.hd == pytest.approx(stoch_rec.hd, rel=1e-6), \
                f"HD mismatch at year {det_rec.year}"
            assert det_rec.tpa == pytest.approx(stoch_rec.tpa, rel=1e-6), \
                f"TPA mismatch at year {det_rec.year}"
            assert det_rec.ba == pytest.approx(stoch_rec.ba, rel=1e-6), \
                f"BA mismatch at year {det_rec.year}"

        # Compare terminal values
        assert det_result.npv == pytest.approx(stoch_result.npv, rel=1e-6)
        assert det_result.lev == pytest.approx(stoch_result.lev, rel=1e-6)


# =============================================================================
# Test 2: Mean-trend closeness under low process noise
# =============================================================================

class TestMeanTrendCloseness:
    """For small λ_proc, MC mean trajectory should remain close to deterministic."""

    def test_low_noise_mean_close_to_deterministic(self):
        """Monte Carlo mean with low noise should be close to deterministic."""
        # Deterministic baseline
        det_config = ScenarioConfig(
            name="deterministic",
            scenario_type="deterministic",
        )
        det_result = run_scenario(det_config)
        det_npv = det_result.npv

        # Low noise stochastic (λ=0.25, no disturbance)
        stoch_config = ScenarioConfig(
            name="low_noise",
            scenario_type="stochastic",
            noise_params=NoiseParams(lambda_proc=0.25),
            disturbance_params=DisturbanceParams(p_dist=0.0),
        )
        
        # Run batch
        batch = run_batch(stoch_config, n_trajectories=500, seed=42)
        mean_npv = float(np.mean(batch.npvs))

        # Mean should stay reasonably close to deterministic.
        # With PMRC merchantability equations now driving valuation, the
        # NPV map is more nonlinear than the previous approximation-based path.
        relative_diff = abs(mean_npv - det_npv) / det_npv
        assert relative_diff < 0.15, \
            f"Mean NPV ${mean_npv:.2f} differs from deterministic ${det_npv:.2f} by {relative_diff:.1%}"


# =============================================================================
# Test 3: Feasibility invariants
# =============================================================================

class TestFeasibilityInvariants:
    """After every step, atomic state variables must satisfy constraints."""

    def test_tpa_lower_bound(self):
        """TPA must be >= 100 (PMRC lower bound) after every step."""
        config = ScenarioConfig(
            name="high_disturbance",
            scenario_type="stochastic",
            noise_params=NoiseParams(lambda_proc=0.5),
            disturbance_params=DisturbanceParams(p_dist=0.10),  # High disturbance
        )
        
        rng = np.random.default_rng(42)
        for _ in range(100):  # Run 100 trajectories
            result = run_scenario(config, rng=rng)
            for rec in result.trajectory:
                assert rec.tpa >= 100.0, f"TPA {rec.tpa} < 100 at year {rec.year}"

    def test_ba_non_negative(self):
        """BA must be >= 0 after every step."""
        config = ScenarioConfig(
            name="high_disturbance",
            scenario_type="stochastic",
            noise_params=NoiseParams(lambda_proc=0.5),
            disturbance_params=DisturbanceParams(p_dist=0.10),
        )
        
        rng = np.random.default_rng(42)
        for _ in range(100):
            result = run_scenario(config, rng=rng)
            for rec in result.trajectory:
                assert rec.ba >= 0.0, f"BA {rec.ba} < 0 at year {rec.year}"

    def test_hd_non_decreasing(self):
        """HD should be non-decreasing (height doesn't shrink)."""
        # Use scenario without disturbance to HD (c_hd=0 by default)
        config = ScenarioConfig(
            name="noise_only",
            scenario_type="stochastic",
            noise_params=NoiseParams(lambda_proc=0.5),
            disturbance_params=DisturbanceParams(p_dist=0.0),
        )
        
        rng = np.random.default_rng(42)
        for _ in range(50):
            result = run_scenario(config, rng=rng)
            prev_hd = 0.0
            for rec in result.trajectory:
                assert rec.hd >= prev_hd, \
                    f"HD decreased from {prev_hd:.2f} to {rec.hd:.2f} at year {rec.year}"
                prev_hd = rec.hd

    def test_no_nans(self):
        """No NaN values in any atomic variable."""
        config = ScenarioConfig(
            name="combined",
            scenario_type="stochastic",
            noise_params=NoiseParams(lambda_proc=1.0),
            disturbance_params=DisturbanceParams(p_dist=0.10),
        )
        
        rng = np.random.default_rng(42)
        for _ in range(100):
            result = run_scenario(config, rng=rng)
            for rec in result.trajectory:
                assert not np.isnan(rec.age), f"NaN age at year {rec.year}"
                assert not np.isnan(rec.hd), f"NaN HD at year {rec.year}"
                assert not np.isnan(rec.tpa), f"NaN TPA at year {rec.year}"
                assert not np.isnan(rec.ba), f"NaN BA at year {rec.year}"

    def test_age_positive(self):
        """Age must be > 0."""
        config = ScenarioConfig(
            name="combined",
            scenario_type="stochastic",
            noise_params=NoiseParams(lambda_proc=0.5),
            disturbance_params=DisturbanceParams(p_dist=0.05),
        )
        
        rng = np.random.default_rng(42)
        for _ in range(50):
            result = run_scenario(config, rng=rng)
            for rec in result.trajectory:
                assert rec.age > 0, f"Age {rec.age} <= 0 at year {rec.year}"


# =============================================================================
# Test 4: Disturbance frequency sanity
# =============================================================================

class TestDisturbanceFrequency:
    """Empirical disturbance frequency should approximate p_dist."""

    def test_disturbance_rate_matches_probability(self):
        """Observed disturbance rate should be close to p_dist."""
        p_dist = 0.05  # 5% annual probability (20-year return)
        n_years = 30
        n_runs = 1000
        
        model = DisturbanceModel(DisturbanceParams(p_dist=p_dist))
        rng = np.random.default_rng(42)
        
        total_years = 0
        total_disturbances = 0
        
        for _ in range(n_runs):
            for _ in range(n_years):
                total_years += 1
                if model.sample_occurrence(rng):
                    total_disturbances += 1
        
        empirical_rate = total_disturbances / total_years
        
        # Should be within 20% of expected rate
        assert abs(empirical_rate - p_dist) / p_dist < 0.20, \
            f"Empirical rate {empirical_rate:.4f} differs from p_dist {p_dist:.4f}"

    def test_batch_outputs_capture_disturbance_paths(self):
        """Batch results should expose disturbance occurrence, timing, and severities."""
        config = ScenarioConfig(
            name="disturbance_tracking",
            scenario_type="stochastic",
            noise_params=NoiseParams(lambda_proc=0.0),
            disturbance_params=DisturbanceParams(
                p_dist=1.0,
                severity_mean=0.30,
                severity_kappa=12.0,
            ),
            rotation_length=5,
        )

        batch = run_batch(config, n_trajectories=10, seed=42)

        assert batch.disturbance_occurred.shape == (10,)
        assert batch.disturbance_counts.shape == (10,)
        assert len(batch.disturbance_years) == 10
        assert len(batch.disturbance_severity_paths) == 10
        assert np.all(batch.disturbance_occurred)
        assert np.all(batch.disturbance_counts == 5)
        assert all(len(years) == 5 for years in batch.disturbance_years)
        assert all(len(path) == 5 for path in batch.disturbance_severity_paths)


# =============================================================================
# Test 5: Severity distribution sanity
# =============================================================================

class TestSeverityDistribution:
    """Conditional on occurrence, severity should match Beta parameters."""

    def test_severity_mean_matches_parameter(self):
        """Mean severity should be close to severity_mean parameter."""
        severity_mean = 0.30
        severity_kappa = 12.0
        
        model = DisturbanceModel(DisturbanceParams(
            p_dist=1.0,  # Always occurs
            severity_mean=severity_mean,
            severity_kappa=severity_kappa,
        ))
        rng = np.random.default_rng(42)
        
        severities = [model.sample_severity(rng) for _ in range(10000)]
        empirical_mean = np.mean(severities)
        
        # Should be within 5% of expected mean
        assert abs(empirical_mean - severity_mean) / severity_mean < 0.05, \
            f"Empirical mean {empirical_mean:.4f} differs from expected {severity_mean:.4f}"

    def test_severity_variance_matches_beta(self):
        """Variance should match Beta distribution variance."""
        severity_mean = 0.30
        severity_kappa = 12.0
        
        # Beta variance = α*β / ((α+β)^2 * (α+β+1))
        alpha = severity_mean * severity_kappa
        beta = (1 - severity_mean) * severity_kappa
        expected_var = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
        
        model = DisturbanceModel(DisturbanceParams(
            p_dist=1.0,
            severity_mean=severity_mean,
            severity_kappa=severity_kappa,
        ))
        rng = np.random.default_rng(42)
        
        severities = [model.sample_severity(rng) for _ in range(10000)]
        empirical_var = np.var(severities)
        
        # Should be within 20% of expected variance
        assert abs(empirical_var - expected_var) / expected_var < 0.20, \
            f"Empirical variance {empirical_var:.6f} differs from expected {expected_var:.6f}"


# =============================================================================
# Test 6: Monotonic risk intuition
# =============================================================================

class TestMonotonicRiskIntuition:
    """Increasing noise/disturbance should increase dispersion and worsen downside."""

    def test_increasing_noise_increases_dispersion(self):
        """Higher λ_proc should increase standard deviation of terminal value."""
        results = {}
        
        for lam in [0.0, 0.25, 0.5, 1.0]:
            if lam == 0.0:
                config = ScenarioConfig(
                    name=f"lambda_{lam}",
                    scenario_type="deterministic",
                )
                result = run_scenario(config)
                results[lam] = {"std": 0.0, "mean": result.npv}
            else:
                config = ScenarioConfig(
                    name=f"lambda_{lam}",
                    scenario_type="stochastic",
                    noise_params=NoiseParams(lambda_proc=lam),
                    disturbance_params=DisturbanceParams(p_dist=0.0),
                )
                batch = run_batch(config, n_trajectories=500, seed=42)
                results[lam] = {"std": float(np.std(batch.npvs)), "mean": float(np.mean(batch.npvs))}
        
        # Check monotonicity of std
        stds = [results[lam]["std"] for lam in [0.0, 0.25, 0.5, 1.0]]
        for i in range(len(stds) - 1):
            assert stds[i] <= stds[i + 1], \
                f"Std not monotonic: {stds[i]:.2f} > {stds[i + 1]:.2f}"

    def test_increasing_disturbance_lowers_mean(self):
        """Higher p_dist should lower mean terminal value."""
        results = {}
        
        for p_dist in [0.0, 1/30, 1/20, 1/10]:
            if p_dist == 0.0:
                config = ScenarioConfig(
                    name=f"pdist_{p_dist}",
                    scenario_type="deterministic",
                )
                result = run_scenario(config)
                results[p_dist] = float(result.npv)
            else:
                config = ScenarioConfig(
                    name=f"pdist_{p_dist}",
                    scenario_type="stochastic",
                    noise_params=NoiseParams(lambda_proc=0.0),
                    disturbance_params=DisturbanceParams(p_dist=p_dist),
                )
                batch = run_batch(config, n_trajectories=500, seed=42)
                results[p_dist] = float(np.mean(batch.npvs))
        
        # Check monotonicity of mean (should decrease)
        means = [results[p] for p in [0.0, 1/30, 1/20, 1/10]]
        for i in range(len(means) - 1):
            assert means[i] >= means[i + 1], \
                f"Mean not decreasing: {means[i]:.2f} < {means[i + 1]:.2f}"


# =============================================================================
# Test 7: Deterministic policy consistency
# =============================================================================

class TestDeterministicPolicyConsistency:
    """Thinning policy should fire consistently when enabled."""

    def test_thinning_fires_in_deterministic(self):
        """Thinning should occur at trigger age in deterministic scenario with BAT."""
        config = ScenarioConfig(
            name="deterministic",
            scenario_type="deterministic",
            thin_params=ThinningParams(),  # Explicitly enable BAT thinning
        )
        result = run_scenario(config)
        
        # BAT thinning at age 15
        assert result.thin_occurred, "Thinning should occur in deterministic baseline with BAT"
        assert result.thin_year == 10, "Thinning should occur at year 10 (age 15)"

    def test_thinning_fires_in_stochastic_without_disturbance(self):
        """Thinning should still fire in stochastic scenario without disturbance."""
        config = ScenarioConfig(
            name="stochastic_no_dist",
            scenario_type="stochastic",
            noise_params=NoiseParams(lambda_proc=0.25),
            disturbance_params=DisturbanceParams(p_dist=0.0),
            thin_params=ThinningParams(),  # Explicitly enable BAT thinning
        )
        
        # Run multiple times - thinning should usually occur
        rng = np.random.default_rng(42)
        thin_count = 0
        n_runs = 50
        
        for _ in range(n_runs):
            result = run_scenario(config, rng=rng)
            if result.thin_occurred:
                thin_count += 1
        
        # Should thin in most runs (BA threshold should still be reached)
        thin_rate = thin_count / n_runs
        assert thin_rate > 0.8, \
            f"Thinning rate {thin_rate:.1%} too low - expected >80%"


class TestComparisonMetrics:
    """Scenario comparisons should expose downside probability versus deterministic."""

    def test_downside_probability_relative_to_deterministic_baseline(self):
        deterministic = run_scenario(
            ScenarioConfig(name="deterministic", scenario_type="deterministic")
        )
        stochastic = run_batch(
            ScenarioConfig(
                name="disturbed",
                scenario_type="stochastic",
                noise_params=NoiseParams(lambda_proc=0.0),
                disturbance_params=DisturbanceParams(p_dist=1 / 10),
            ),
            n_trajectories=200,
            seed=42,
        )

        comparison = compare_scenarios(
            {"deterministic": deterministic, "disturbed": stochastic},
            metric="npv",
        )

        assert comparison["deterministic"]["downside_prob_vs_deterministic"] == 0.0
        assert 0.0 <= comparison["disturbed"]["downside_prob_vs_deterministic"] <= 1.0
        assert comparison["disturbed"]["downside_prob_vs_deterministic"] > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
