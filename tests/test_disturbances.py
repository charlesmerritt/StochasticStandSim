from __future__ import annotations

import inspect
import math
from dataclasses import is_dataclass, fields as dataclass_fields
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest
import yaml

from .utils import get_stand_record, load_reference_yields


DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "disturbances"
ADSR_PATH = DATA_ROOT / "adsr_envelopes.yaml"
JUMP_KERNELS_PATH = DATA_ROOT / "jump_kernels.yaml"


def _instantiate(cls: Any, data: Mapping[str, Any]) -> Any:
    if is_dataclass(cls):
        names = {field.name for field in dataclass_fields(cls)}
    else:
        params = inspect.signature(cls).parameters
        names = {
            name
            for name, param in params.items()
            if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
        }
    kwargs = {name: data[name] for name in names if name in data}
    return cls(**kwargs)


def _extract(obj: Any, *candidates: str) -> Any:
    for name in candidates:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AttributeError(f"Object does not expose any of {candidates}")


@pytest.fixture(scope="module")
def adsr_configs() -> Mapping[str, Any]:
    if not ADSR_PATH.exists():
        pytest.fail(f"Expected ADSR config at {ADSR_PATH}")
    with ADSR_PATH.open() as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def jump_kernels() -> Mapping[str, Any]:
    if not JUMP_KERNELS_PATH.exists():
        pytest.fail(f"Expected jump kernel config at {JUMP_KERNELS_PATH}")
    with JUMP_KERNELS_PATH.open() as fh:
        return yaml.safe_load(fh)


def test_adsr_wind_envelope_complete(adsr_configs):
    wind = adsr_configs.get("wind")
    assert wind is not None, "Wind ADSR envelope missing."
    for key in ("target", "attack_years", "decay_years", "sustain_years", "release_years", "sustain_level"):
        assert key in wind, f"Wind ADSR envelope missing {key}"
    assert isinstance(wind["target"], str)
    for key in ("attack_years", "decay_years", "sustain_years", "release_years"):
        assert isinstance(wind[key], (int, float))
        assert wind[key] >= 0
    assert 0 <= wind["sustain_level"] <= 1


@pytest.mark.xfail(reason="Additional ADSR envelopes not populated", strict=True)
def test_adsr_all_disturbances_specified(adsr_configs):
    for name, cfg in adsr_configs.items():
        assert cfg not in (None, "..."), f"ADSR envelope for {name} is incomplete."


def test_jump_kernel_wind_structure(jump_kernels):
    wind = jump_kernels.get("wind")
    assert wind is not None, "Wind jump kernel missing."
    assert "base_rate_per_year" in wind
    assert 0 <= wind["base_rate_per_year"] <= 1
    severity = wind.get("severity")
    assert severity is not None, "Wind severity kernel missing."
    assert severity.get("distribution") in {"beta"}
    alpha_map = severity.get("alpha_map")
    assert isinstance(alpha_map, list) and alpha_map, "Wind severity alpha_map must be a non-empty list."
    for entry in alpha_map:
        assert {"alpha", "beta"} <= entry.keys()
        assert entry["alpha"] > 0 and entry["beta"] > 0


@pytest.mark.xfail(reason="Fire jump kernel not populated", strict=True)
def test_jump_kernel_all_disturbances_specified(jump_kernels):
    for name, cfg in jump_kernels.items():
        assert cfg not in (None, "..."), f"Jump kernel for {name} is incomplete."


def _resolve_beta_params(age: float, alpha_map: list[Mapping[str, Any]]) -> tuple[float, float]:
    for entry in alpha_map:
        max_age = entry.get("age_le")
        min_age = entry.get("age_gt")
        if max_age is not None and age <= max_age:
            return entry["alpha"], entry["beta"]
        if min_age is not None and age > min_age:
            return entry["alpha"], entry["beta"]
    raise ValueError(f"No beta parameters found for age {age}")


@pytest.mark.xfail(reason="Disturbance sampling not implemented", strict=True)
def test_seeded_disturbance_severity_matches_rng(jump_kernels, adsr_configs):
    disturbances = pytest.importorskip("core.disturbances")
    types_module = pytest.importorskip("core.types")

    if not hasattr(types_module, "StandState"):
        pytest.fail("core.types.StandState must be defined for disturbance tests.")
    if not hasattr(disturbances, "sample_event"):
        pytest.fail("core.disturbances.sample_event must be implemented.")
    if not hasattr(disturbances, "apply_event"):
        pytest.fail("core.disturbances.apply_event must be implemented.")
    if not hasattr(disturbances, "update_adsr_effects"):
        pytest.fail("core.disturbances.update_adsr_effects must be implemented.")

    stand_record = get_stand_record("stand1")
    trajectory = load_reference_yields("stand1", "regime1")

    StandState = types_module.StandState
    state_data = {
        "age": trajectory[0]["age"],
        "N": trajectory[0]["tpa"],
        "BA": trajectory[0]["ba"],
        "H": trajectory[0]["hd"],
        "Dq": trajectory[0]["qmd"],
        "volume": {
            "total": trajectory[0]["volume"],
            "pulp": trajectory[0]["pulp"],
            "chip": trajectory[0]["chip"],
            "saw": trajectory[0]["saw"],
        },
        "site_region": stand_record["region"],
        "site_index": stand_record["si"],
        "carbon": 0.0,
        "last_disturbance": None,
        "rng_state_id": None,
    }

    state = _instantiate(StandState, state_data)

    wind_kernel = jump_kernels["wind"].copy()
    wind_kernel["base_rate_per_year"] = 1.0  # force event for deterministic validation
    kernels = {"wind": wind_kernel}

    rng_seed = 20241118
    rng = np.random.default_rng(rng_seed)
    event = disturbances.sample_event(state, rng, kernels)
    assert event is not None, "Expected disturbance event when base rate is forced to 1."
    event_type = _extract(event, "type", "kind")
    assert event_type == "wind"

    alpha_map = wind_kernel["severity"]["alpha_map"]
    age = _extract(state, "age", "Age")
    alpha, beta = _resolve_beta_params(age, alpha_map)

    rng_reference = np.random.default_rng(rng_seed)
    rng_reference.random()  # probability draw
    expected_severity = rng_reference.beta(alpha, beta)

    severity = _extract(event, "severity")
    assert 0 <= severity <= 1
    assert math.isclose(severity, expected_severity, rel_tol=1e-6)

    adsr = {event_type: adsr_configs[event_type]}
    disturbed_state = disturbances.apply_event(state, event, adsr)
    assert disturbed_state is not None

    for _ in range(int(adsr[event_type]["attack_years"] + adsr[event_type]["decay_years"] + adsr[event_type]["sustain_years"] + adsr[event_type]["release_years"] + 2)):
        disturbed_state = disturbances.update_adsr_effects(disturbed_state, adsr)
        assert disturbed_state is not None
