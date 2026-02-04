"""Tests for configuration loading and risk profiles."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from core.config import (
    NoiseParams,
    DisturbanceParams,
    RiskProfile,
    RISK_PROFILES,
    get_risk_profile,
    MDPDiscretization,
    ActionSpec,
    SimulationConfig,
    load_config,
)


class TestNoiseParams:
    """Test noise parameter dataclass."""

    def test_default_values(self):
        noise = NoiseParams()
        assert noise.sigma_log_ba == 0.10
        assert noise.sigma_log_hd is None
        assert noise.sigma_tpa == 20.0
        assert noise.use_binomial_tpa is True

    def test_scale(self):
        noise = NoiseParams(sigma_log_ba=0.10, sigma_tpa=20.0)
        scaled = noise.scale(2.0)
        assert scaled.sigma_log_ba == 0.20
        assert scaled.sigma_tpa == 40.0

    def test_scale_with_none_hd(self):
        noise = NoiseParams(sigma_log_hd=None)
        scaled = noise.scale(2.0)
        assert scaled.sigma_log_hd is None


class TestDisturbanceParams:
    """Test disturbance parameter dataclass."""

    def test_default_values(self):
        dist = DisturbanceParams()
        assert dist.p_mild == 0.02
        assert dist.severe_mean_interval == 25.0
        assert dist.mild_tpa_multiplier == 0.85
        assert dist.severe_tpa_multiplier == 0.40

    def test_p_severe_annual(self):
        dist = DisturbanceParams(severe_mean_interval=25.0)
        # 1 - exp(-1/25) ≈ 0.0392
        assert math.isclose(dist.p_severe_annual, 0.0392, rel_tol=0.01)

    def test_p_severe_annual_zero_interval(self):
        dist = DisturbanceParams(severe_mean_interval=0.0)
        assert dist.p_severe_annual == 0.0


class TestRiskProfiles:
    """Test predefined risk profiles."""

    def test_all_profiles_exist(self):
        assert "low" in RISK_PROFILES
        assert "medium" in RISK_PROFILES
        assert "high" in RISK_PROFILES

    def test_get_risk_profile(self):
        profile = get_risk_profile("medium")
        assert profile.name == "Medium Risk"
        assert profile.noise.sigma_log_ba == 0.10

    def test_risk_ordering(self):
        """Higher risk should have higher noise and more frequent disturbances."""
        low = get_risk_profile("low")
        med = get_risk_profile("medium")
        high = get_risk_profile("high")

        # Noise increases with risk
        assert low.noise.sigma_log_ba < med.noise.sigma_log_ba < high.noise.sigma_log_ba

        # Severe disturbance interval decreases with risk (more frequent)
        assert low.disturbance.severe_mean_interval > med.disturbance.severe_mean_interval
        assert med.disturbance.severe_mean_interval > high.disturbance.severe_mean_interval


class TestMDPDiscretization:
    """Test MDP discretization configuration."""

    def test_default_bins(self):
        disc = MDPDiscretization()
        assert disc.n_age == 8  # 9 bin edges -> 8 bins
        assert disc.n_tpa == 5
        assert disc.n_ba == 5

    def test_n_states(self):
        disc = MDPDiscretization()
        assert disc.n_states == 8 * 5 * 5  # 200 states

    def test_to_numpy(self):
        disc = MDPDiscretization()
        age_bins, tpa_bins, ba_bins = disc.to_numpy()
        assert len(age_bins) == 9
        assert age_bins[0] == 0
        assert age_bins[-1] == 40


class TestActionSpec:
    """Test action specification."""

    def test_default_actions(self):
        actions = ActionSpec()
        assert actions.n_actions == 4  # no-op, thin-20%, thin-40%, harvest
        assert "no-op" in actions.action_names
        assert "harvest-replant" in actions.action_names

    def test_action_names(self):
        actions = ActionSpec(thin_fractions=(1.0, 0.75, 0.50), harvest_replant=True)
        names = actions.action_names
        assert names[0] == "no-op"
        assert "thin-25%" in names
        assert "thin-50%" in names

    def test_no_harvest(self):
        actions = ActionSpec(harvest_replant=False)
        assert actions.n_actions == 3
        assert "harvest-replant" not in actions.action_names


class TestLoadConfig:
    """Test config loading functions."""

    def test_load_default_config(self):
        config = load_config()
        assert config.econ is not None
        assert config.risk_profile.name == "Medium Risk"
        assert config.effective_discount_rate == 0.05

    def test_load_with_risk_level(self):
        config = load_config(risk_level="high")
        assert config.risk_profile.name == "High Risk"
        assert config.risk_profile.noise.sigma_log_ba == 0.15

    def test_discount_rate_override(self):
        config = load_config()
        assert config.effective_discount_rate == 0.05

        # Create config with override
        from core.economics import load_econ_params
        econ = load_econ_params(Path(__file__).parent.parent / "data" / "econ_params.yaml")
        config_override = SimulationConfig(
            econ=econ,
            risk_profile=get_risk_profile("medium"),
            discount_rate=0.08,
        )
        assert config_override.effective_discount_rate == 0.08


class TestSimulationConfig:
    """Test complete simulation config."""

    def test_config_has_all_components(self):
        config = load_config()
        assert config.econ is not None
        assert config.risk_profile is not None
        assert config.discretization is not None
        assert config.actions is not None
        assert config.dt == 1.0
        assert config.max_age == 40.0
