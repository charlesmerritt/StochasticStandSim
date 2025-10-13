"""Stochastic disturbance helpers shared by the simulator and demo UI."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

import copy
import numpy as np
import yaml

__all__ = [
    "DisturbanceSetting",
    "load_disturbance_catalog",
    "resolve_disturbance_settings",
    "initialise_disturbance_status",
    "decay_disturbance_status",
    "apply_stochastic_disturbances",
]


@dataclass(frozen=True)
class DisturbanceSetting:
    """Configuration for a stochastic disturbance type."""

    label: str = "Disturbance"
    emoji: str = "❗"
    enabled: bool = True
    probability: float = 0.05
    severity_min: float = 0.1
    severity_max: float = 0.35
    envelope_boost: float = 0.1
    envelope_years: int = 5
    effects: Mapping[str, float] = field(default_factory=dict)
    catastrophic_threshold: float = 0.65
    risk_increment: float = 0.2
    salvage_recovery_rate: float = 0.55


_FALLBACK_DISTURBANCES: Dict[str, Dict[str, Any]] = {
    "fire": {
        "label": "Fire",
        "emoji": "🔥",
        "enabled": True,
        "probability": 0.04,
        "severity_min": 0.1,
        "severity_max": 0.45,
        "envelope_boost": 0.1,
        "envelope_years": 5,
        "effects": {"biomass": 0.5, "tpa": 0.4, "basal_area": 0.6, "risk": 0.3},
        "catastrophic_threshold": 0.55,
        "risk_increment": 0.2,
        "salvage_recovery_rate": 0.55,
    },
    "wind": {
        "label": "Wind",
        "emoji": "💨",
        "enabled": True,
        "probability": 0.03,
        "severity_min": 0.08,
        "severity_max": 0.3,
        "envelope_boost": 0.08,
        "envelope_years": 4,
        "effects": {"biomass": 0.4, "tpa": 0.45, "basal_area": 0.35, "risk": 0.25},
        "catastrophic_threshold": 0.6,
        "risk_increment": 0.18,
        "salvage_recovery_rate": 0.55,
    },
    "insect": {
        "label": "Insects",
        "emoji": "🪲",
        "enabled": True,
        "probability": 0.05,
        "severity_min": 0.05,
        "severity_max": 0.25,
        "envelope_boost": 0.12,
        "envelope_years": 6,
        "effects": {"biomass": 0.35, "tpa": 0.2, "basal_area": 0.3, "risk": 0.2},
        "catastrophic_threshold": 0.5,
        "risk_increment": 0.15,
        "salvage_recovery_rate": 0.5,
    },
}


def _coerce_setting(key: str, data: Mapping[str, Any]) -> DisturbanceSetting:
    base = _FALLBACK_DISTURBANCES.get(key, {})
    merged = dict(base)
    merged.update({k: v for k, v in data.items() if v is not None})
    effects = merged.get("effects", {})
    merged["effects"] = {str(k): float(v) for k, v in dict(effects).items()}
    return DisturbanceSetting(
        label=str(merged.get("label", key.title())),
        emoji=str(merged.get("emoji", "❗")),
        enabled=bool(merged.get("enabled", True)),
        probability=float(merged.get("probability", 0.05)),
        severity_min=float(merged.get("severity_min", 0.1)),
        severity_max=float(merged.get("severity_max", 0.35)),
        envelope_boost=float(merged.get("envelope_boost", 0.1)),
        envelope_years=int(merged.get("envelope_years", 5)),
        effects=merged["effects"],
        catastrophic_threshold=float(merged.get("catastrophic_threshold", 0.65)),
        risk_increment=float(merged.get("risk_increment", 0.2)),
        salvage_recovery_rate=float(merged.get("salvage_recovery_rate", 0.55)),
    )


def load_disturbance_catalog(path: Optional[str] = None) -> Dict[str, DisturbanceSetting]:
    """Load disturbance settings from ``path`` or the default YAML file."""

    search_order: Iterable[Path]
    if path is None:
        search_order = (
            Path("config/disturbances.yaml"),
            Path(__file__).resolve().parents[2] / "config" / "disturbances.yaml",
        )
    else:
        search_order = (Path(path),)

    data: Dict[str, Any] = {}
    for candidate in search_order:
        try:
            with candidate.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
        except FileNotFoundError:
            continue
        else:
            if isinstance(loaded, Mapping):
                data = dict(loaded)
            break

    if not data:
        data = dict(_FALLBACK_DISTURBANCES)

    settings: Dict[str, DisturbanceSetting] = {}
    for key, value in data.items():
        if isinstance(value, Mapping):
            settings[str(key)] = _coerce_setting(str(key), value)
    return settings


def resolve_disturbance_settings(
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, DisturbanceSetting]:
    """Resolve stochastic disturbance settings using ``config`` overrides."""

    cfg = config or {}
    if not isinstance(cfg, Mapping):
        cfg = {}

    enabled = bool(cfg.get("stochastic_enabled", True))
    if not enabled:
        return {}

    path = cfg.get("stochastic_path") or cfg.get("stochastic_config_path")
    settings = load_disturbance_catalog(path)

    overrides = cfg.get("stochastic_overrides") or cfg.get("stochastic")
    if isinstance(overrides, Mapping):
        for key, value in overrides.items():
            if not isinstance(value, Mapping):
                continue
            base = settings.get(key, _coerce_setting(key, {}))
            settings[key] = replace(
                base,
                label=str(value.get("label", base.label)),
                emoji=str(value.get("emoji", base.emoji)),
                enabled=bool(value.get("enabled", base.enabled)),
                probability=float(value.get("probability", base.probability)),
                severity_min=float(value.get("severity_min", base.severity_min)),
                severity_max=float(value.get("severity_max", base.severity_max)),
                envelope_boost=float(value.get("envelope_boost", base.envelope_boost)),
                envelope_years=int(value.get("envelope_years", base.envelope_years)),
                effects={
                    str(k): float(v)
                    for k, v in dict(value.get("effects", base.effects)).items()
                },
                catastrophic_threshold=float(
                    value.get("catastrophic_threshold", base.catastrophic_threshold)
                ),
                risk_increment=float(value.get("risk_increment", base.risk_increment)),
                salvage_recovery_rate=float(
                    value.get("salvage_recovery_rate", base.salvage_recovery_rate)
                ),
            )

    return settings


def initialise_disturbance_status(
    settings: Mapping[str, DisturbanceSetting],
) -> Dict[str, Dict[str, float]]:
    """Return mutable status tracking envelope boosts for each disturbance."""

    status: Dict[str, Dict[str, float]] = {}
    for key, cfg in settings.items():
        status[key] = {
            "base_prob": float(cfg.probability),
            "current_prob": float(cfg.probability if cfg.enabled else 0.0),
            "boost_years": 0.0,
        }
    return status


def decay_disturbance_status(
    status: MutableMapping[str, Dict[str, float]],
    settings: Mapping[str, DisturbanceSetting],
) -> None:
    """Advance the disturbance envelope timers by one step."""

    for key, entry in status.items():
        cfg = settings.get(key)
        if cfg is None:
            continue
        boost_years = float(entry.get("boost_years", 0.0))
        if boost_years > 0:
            boost_years -= 1
            entry["boost_years"] = float(max(boost_years, 0.0))
            if boost_years <= 0:
                entry["current_prob"] = float(cfg.probability if cfg.enabled else 0.0)


def apply_stochastic_disturbances(
    state: Mapping[str, Any],
    rng: Optional[np.random.Generator],
    status: MutableMapping[str, Dict[str, float]],
    settings: Mapping[str, DisturbanceSetting],
    global_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Stochastically modify ``state`` and update ``status``."""

    next_state: Dict[str, Any] = dict(state)
    if not settings:
        return {"state": next_state, "info": {"events": [], "triggered": False}}

    if rng is None:
        rng = np.random.default_rng()

    decay_disturbance_status(status, settings)

    cfg = global_config or {}
    events: List[Dict[str, Any]] = []

    for key, setting in settings.items():
        entry = status.setdefault(
            key,
            {
                "base_prob": float(setting.probability),
                "current_prob": float(setting.probability if setting.enabled else 0.0),
                "boost_years": 0.0,
            },
        )

        if not setting.enabled:
            entry["current_prob"] = 0.0
            continue

        prob = float(np.clip(entry.get("current_prob", setting.probability), 0.0, 1.0))
        if rng.random() >= prob:
            continue

        severity = float(
            np.clip(
                rng.uniform(setting.severity_min, setting.severity_max),
                0.0,
                1.0,
            )
        )

        effects = dict(setting.effects)
        biomass_before = float(next_state.get("biomass", 0.0))

        for attr, weight in effects.items():
            if attr not in next_state:
                continue
            current_val = max(float(next_state.get(attr, 0.0)), 0.0)
            next_state[attr] = max(current_val * (1.0 - severity * float(weight)), 0.0)

        biomass_after = float(next_state.get("biomass", 0.0))
        biomass_loss = max(biomass_before - biomass_after, 0.0)

        salvage_rate = float(
            cfg.get("stochastic_salvage_recovery", setting.salvage_recovery_rate)
        )
        salvage = biomass_loss * float(np.clip(salvage_rate, 0.0, 1.0))
        next_state["salvage_biomass"] = float(next_state.get("salvage_biomass", 0.0) + salvage)

        risk_increment = float(cfg.get("stochastic_risk_increment", setting.risk_increment))
        next_state["risk"] = float(
            np.clip(next_state.get("risk", 0.0) + severity * risk_increment, 0.0, 1.0)
        )

        max_severity = float(next_state.get("disturbance_severity", 0.0))
        next_state["disturbance_severity"] = max(max_severity, severity)

        catastrophic_threshold = float(
            cfg.get("stochastic_catastrophic_threshold", setting.catastrophic_threshold)
        )
        catastrophic = severity >= catastrophic_threshold
        if catastrophic:
            next_state["catastrophic"] = True

        entry["boost_years"] = float(max(setting.envelope_years, 0))
        entry["current_prob"] = float(
            np.clip(prob + float(setting.envelope_boost), 0.0, 1.0)
        )

        events.append(
            {
                "type": key,
                "label": setting.label,
                "emoji": setting.emoji,
                "severity": severity,
                "catastrophic": catastrophic,
                "biomass_loss": biomass_loss,
            }
        )

    info = {
        "events": events,
        "triggered": bool(events),
        "status": copy.deepcopy(status),
    }

    return {"state": next_state, "info": info}

