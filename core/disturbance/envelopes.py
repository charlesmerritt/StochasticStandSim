"""Simple deterministic disturbance envelopes."""

from __future__ import annotations

from typing import Mapping, MutableMapping, Dict, Any

import numpy as np

__all__ = ["apply_deterministic_risk"]


def apply_deterministic_risk(
    state: Mapping[str, Any],
    risk_cfg: MutableMapping[str, Any],
) -> Dict[str, Any]:
    """Apply deterministic catastrophic risk rules.

    The helper examines the current risk metric and, if it exceeds a configured
    threshold, reduces growing-stock variables according to a fixed severity.
    """

    risk_cfg = risk_cfg or {}
    next_state = dict(state)

    threshold = float(risk_cfg.get("catastrophe_threshold", 1.0))
    severity = float(np.clip(risk_cfg.get("catastrophe_severity", 0.75), 0.0, 1.0))
    risk_decay = float(np.clip(risk_cfg.get("post_event_risk", 0.1), 0.0, 1.0))
    baseline_risk = float(np.clip(risk_cfg.get("baseline_risk", 0.0), 0.0, 1.0))

    triggered = float(state.get("risk", 0.0)) >= threshold
    info = {"catastrophe": bool(triggered), "severity": severity if triggered else 0.0}

    if triggered:
        for key in ("biomass", "tpa", "basal_area"):
            next_state[key] = max(0.0, float(state.get(key, 0.0)) * (1.0 - severity))
        next_state["catastrophic"] = True
        next_state["risk"] = risk_decay
    else:
        next_state["catastrophic"] = False
        next_state["risk"] = float(np.clip(state.get("risk", 0.0) + baseline_risk, 0.0, 1.0))

    return {"state": next_state, "info": info}
