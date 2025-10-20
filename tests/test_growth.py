from __future__ import annotations

import inspect
import math
from dataclasses import is_dataclass, fields as dataclass_fields
from pathlib import Path
from typing import Any, Mapping

import pytest
import yaml

from .utils import (
    get_regime_record,
    get_stand_record,
    load_reference_yields,
)


DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
PMRC_PATH = DATA_ROOT / "pmrc_params.yaml"


def _instantiate(cls: Any, data: Mapping[str, Any]) -> Any:
    """Instantiate ``cls`` selecting only supported keyword arguments."""
    if is_dataclass(cls):
        names = {field.name for field in dataclass_fields(cls)}
    else:
        params = inspect.signature(cls).parameters
        names = {name for name, param in params.items() if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)}
    kwargs = {name: data[name] for name in names if name in data}
    return cls(**kwargs)


def _extract(state: Any, *candidates: str) -> Any:
    """Return the first matching attribute or mapping key."""
    for name in candidates:
        if isinstance(state, Mapping) and name in state:
            return state[name]
        if hasattr(state, name):
            return getattr(state, name)
    raise AttributeError(f"State object does not expose any of {candidates}")


@pytest.fixture(scope="module")
def stand_record() -> Mapping[str, Any]:
    return get_stand_record("stand1")


@pytest.fixture(scope="module")
def regime_record() -> Mapping[str, Any]:
    return get_regime_record("regime1")


@pytest.fixture(scope="module")
def reference_unthinned() -> list[Mapping[str, Any]]:
    return load_reference_yields("stand1", "regime1")


@pytest.fixture(scope="module")
def pmrc_params() -> Mapping[str, Any]:
    if not PMRC_PATH.exists():
        pytest.fail(f"Expected PMRC parameter file at {PMRC_PATH}")
    with PMRC_PATH.open() as fh:
        params = yaml.safe_load(fh)
    if not params:
        pytest.fail("PMRC parameter file is empty; populate coefficients before running growth tests.")
    return params


def _noop_action(types_module: Any) -> Any:
    action = getattr(types_module, "Action", None)
    if action is None:
        return "noop"
    # Enum with NOOP attribute
    if hasattr(action, "NOOP"):
        return getattr(action, "NOOP")
    # Enum lookup by value
    try:
        return action("noop")
    except Exception:
        pass
    return "noop"


def _volume_components(state: Any) -> Mapping[str, float] | None:
    try:
        volume = _extract(state, "volume", "Volume")
    except AttributeError:
        return None
    if isinstance(volume, Mapping):
        return volume
    return {"total": volume}


@pytest.mark.xfail(reason="Deterministic PMRC growth not implemented", strict=True)
def test_project_one_year_matches_reference(reference_unthinned, stand_record, pmrc_params):
    growth = pytest.importorskip("core.growth")
    types_module = pytest.importorskip("core.types")

    if not hasattr(types_module, "StandState"):
        pytest.fail("core.types.StandState must be defined for growth tests.")
    if not hasattr(types_module, "GrowthParams"):
        pytest.fail("core.types.GrowthParams must be defined for growth tests.")

    region = stand_record.get("region")
    if region not in pmrc_params:
        pytest.fail(f"PMRC parameters missing region {region!r}")

    stand_state_data = {
        "age": reference_unthinned[0]["age"],
        "N": reference_unthinned[0]["tpa"],
        "tpa": reference_unthinned[0]["tpa"],
        "BA": reference_unthinned[0]["ba"],
        "ba": reference_unthinned[0]["ba"],
        "H": reference_unthinned[0]["hd"],
        "Dq": reference_unthinned[0]["qmd"],
        "volume": {
            "total": reference_unthinned[0]["volume"],
            "pulp": reference_unthinned[0]["pulp"],
            "chip": reference_unthinned[0]["chip"],
            "saw": reference_unthinned[0]["saw"],
        },
        "carbon": 0.0,
        "site_region": region,
        "site_index": stand_record.get("si"),
        "region": region,
        "si": stand_record.get("si"),
        "last_disturbance": None,
        "rng_state_id": None,
    }
    growth_params_data = {
        "site_region": region,
        "site_index": stand_record.get("si"),
        "step_years": 1,
        "coefficients": pmrc_params[region],
        "pmrc": pmrc_params[region],
    }

    StandState = types_module.StandState
    GrowthParams = types_module.GrowthParams

    state = _instantiate(StandState, stand_state_data)
    params = _instantiate(GrowthParams, growth_params_data)
    action = _noop_action(types_module)

    for expected in reference_unthinned[1:11]:
        result = growth.project_one_year(state, params, action)
        if result is not None:
            state = result
        # Validate age progression
        age = _extract(state, "age", "Age")
        assert math.isclose(age, expected["age"], rel_tol=1e-6)

        ba = _extract(state, "BA", "ba")
        assert math.isclose(ba, expected["ba"], rel_tol=1e-3, abs_tol=1e-3)

        tpa = _extract(state, "N", "n", "tpa", "TPA")
        assert math.isclose(tpa, expected["tpa"], rel_tol=1e-3, abs_tol=1e-3)

        height = _extract(state, "H", "hd", "HD")
        assert math.isclose(height, expected["hd"], rel_tol=1e-3, abs_tol=1e-3)

        dq = _extract(state, "Dq", "qmd", "DQ")
        assert math.isclose(dq, expected["qmd"], rel_tol=1e-3, abs_tol=1e-3)

        volume_components = _volume_components(state)
        if volume_components:
            if "total" in volume_components:
                assert math.isclose(
                    volume_components["total"], expected["volume"], rel_tol=1e-3, abs_tol=1e-3
                )
            for key in ("pulp", "chip", "saw"):
                if key in volume_components:
                    assert math.isclose(
                        volume_components[key],
                        expected[key],
                        rel_tol=1e-2,
                        abs_tol=1e-3,
                    )
