"""Disturbance envelopes applied by :mod:`core.stand_env`."""

from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping

import numpy as np

__all__ = ["apply_deterministic_risk"]


_SEVERITY_PROFILES = {
    "none": (0.0, 0.0),
    "low": (0.05, 0.25),
    "high": (0.35, 0.85),
}


def _infer_profile(risk_metric: float) -> str:
    if risk_metric < 0.15:
        return "none"
    if risk_metric < 0.45:
        return "low"
    return "high"


def _estimate_height(state: Mapping[str, Any]) -> float:
    if "hd" in state:
        return float(state["hd"])
    site_index = float(state.get("site_index", 70.0))
    age = max(float(state.get("age", 0.0)), 1.0)
    # Simple Chapman–Richards inspired proxy for dominant height.
    alpha = 0.04
    m = 1.2
    g_age = 1.0 - np.exp(-alpha * age)
    g_base = 1.0 - np.exp(-alpha * 25.0)
    return site_index * (g_age / g_base) ** m


def apply_deterministic_risk(
    state: Mapping[str, Any],
    risk_cfg: MutableMapping[str, Any],
) -> Dict[str, Any]:
    """Apply a deterministic disturbance envelope to ``state``.

    The helper converts the stand's risk metric into low/high/none risk
    categories, derives an event severity from dominant height and stand
    density, and then adjusts basal area, trees-per-acre, biomass, and risk
    accordingly.  Severe events mark the stand as catastrophic and reset the
    age to reflect salvage and replanting operations.
    """

    risk_cfg = risk_cfg or {}
    next_state: Dict[str, Any] = dict(state)

    risk_metric = float(np.clip(state.get("risk", risk_cfg.get("baseline", 0.0)), 0.0, 1.0))
    profile = str(risk_cfg.get("profile", _infer_profile(risk_metric))).lower()
    if profile not in _SEVERITY_PROFILES:
        raise ValueError(f"Unknown risk profile '{profile}'")

    sev_low, sev_high = _SEVERITY_PROFILES[profile]
    height = _estimate_height(state)
    tpa = max(float(state.get("tpa", 0.0)), 0.0)
    height_ref = float(risk_cfg.get("height_reference", 90.0))
    density_ref = float(risk_cfg.get("density_reference", 600.0))

    height_component = min(height / max(height_ref, 1e-6), 1.0)
    density_component = min(tpa / max(density_ref, 1e-6), 1.0)
    exposure = 0.5 * height_component + 0.5 * density_component
    exposure = (exposure + risk_metric) / 2.0
    severity = sev_low + (sev_high - sev_low) * exposure
    severity = float(np.clip(severity, 0.0, 0.95))

    ba = max(float(state.get("basal_area", 0.0)), 0.0)
    biomass = max(float(state.get("biomass", 0.0)), 0.0)

    ba_loss = ba * severity
    tpa_loss = tpa * severity
    biomass_loss = biomass * severity

    next_state["basal_area"] = max(ba - ba_loss, 0.0)
    next_state["tpa"] = max(tpa - tpa_loss, 0.0)
    next_state["biomass"] = max(biomass - biomass_loss, 0.0)

    salvage_rate = float(np.clip(risk_cfg.get("salvage_recovery_rate", 0.55), 0.0, 1.0))
    salvage_biomass = biomass_loss * salvage_rate

    catastrophic_threshold = float(np.clip(risk_cfg.get("catastrophic_threshold", 0.4), 0.0, 1.0))
    catastrophic = severity >= catastrophic_threshold

    if catastrophic:
        next_state["catastrophic"] = True
        if bool(risk_cfg.get("reset_age_on_salvage", True)):
            next_state["age"] = float(risk_cfg.get("post_event_age", 0.0))
        next_state["risk"] = float(np.clip(risk_cfg.get("post_event_risk", 0.05), 0.0, 1.0))
    else:
        next_state["catastrophic"] = False
        increment = float(risk_cfg.get("annual_increment", 0.01))
        mitigation = float(risk_cfg.get("mitigation", 0.0))
        next_state["risk"] = float(np.clip(risk_metric + increment - mitigation, 0.0, 1.0))

    next_state["salvage_biomass"] = salvage_biomass
    next_state["disturbance_severity"] = severity

    info = {
        "profile": profile,
        "severity": severity,
        "salvage_biomass": salvage_biomass,
        "catastrophic": catastrophic,
    }

    return {"state": next_state, "info": info}

