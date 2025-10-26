# core/disturbances.py
from __future__ import annotations
from typing import Dict, Tuple, Mapping, Literal, Any, Optional
from dataclasses import dataclass, field
import yaml, random, math

from .rng import rng

Stage = Literal["attack", "decay", "sustain", "release"]
Metric = Literal["basal_area", "volume", "height"]

# ------------------- Utils -------------------

def get_severity(seed: int | None = None) -> float:
    """Random severity in (0,1), never exactly 0 or 1."""
    val = rng(seed)
    if val <= 0.0: return 0.001
    if val >= 1.0: return 0.999
    return round(val, 3)

def _tup(x: Any) -> Tuple[float, float]:
    lo, hi = x
    return float(lo), float(hi)

# ------------------- Envelope -------------------

@dataclass(frozen=True)
class StageRanges:
    basal_area: Tuple[float, float]
    volume: Tuple[float, float]
    height: Tuple[float, float]
    def __str__(self) -> str:
        return f"BA{self.basal_area}  Vol{self.volume}  Ht{self.height}"

@dataclass(frozen=True)
class SeverityClass:
    desc: str
    attack: StageRanges
    decay: StageRanges
    sustain: StageRanges
    release: StageRanges
    duration: int
    def range_for(self, stage: Stage, metric: Metric) -> Tuple[float, float]:
        return getattr(getattr(self, stage), metric)

class Envelope:
    """
    YAML-backed base. Provides:
      - Attribute access by prefix: env.mild -> 'mild_0_10'
      - Mapping access: env['moderate_20_50']
      - Helpers: .range(), .keys(), .classes()
      - Meta: metadata/defaults/modulators available at .meta
    """
    def __init__(self, sev_classes: Mapping[str, SeverityClass], meta: Optional[Dict[str, Any]] = None):
        self._sev_classes: Dict[str, SeverityClass] = dict(sev_classes)
        self.meta: Dict[str, Any] = dict(meta or {})
        self._prefix_index: Dict[str, str] = {}
        for k in self._sev_classes:
            prefix = k.split("_", 1)[0]
            self._prefix_index.setdefault(prefix, k)

    def __getitem__(self, key: str) -> SeverityClass:
        return self._sev_classes[key]

    def keys(self):
        return self._sev_classes.keys()

    def classes(self):
        return self._sev_classes.values()

    def range(self, severity_key_or_prefix: str, stage: Stage, metric: Metric) -> Tuple[float, float]:
        key = self._prefix_index.get(severity_key_or_prefix, severity_key_or_prefix)
        return self._sev_classes[key].range_for(stage, metric)

    def sample(self, severity_key_or_prefix: str, stage: Stage, metric: Metric) -> float:
        lo, hi = self.range(severity_key_or_prefix, stage, metric)
        return round(random.uniform(lo, hi), 3)

    def __getattr__(self, name: str) -> Any:
        if name in self._prefix_index:
            return self._sev_classes[self._prefix_index[name]]
        raise AttributeError(name)

def _to_stage_ranges(d: Dict[str, Any]) -> StageRanges:
    return StageRanges(
        basal_area=_tup(d["basal_area"]),
        volume=_tup(d["volume"]),
        height=_tup(d["height"]),
    )

def _load_envelope_from_path(path: str) -> tuple[Dict[str, SeverityClass], Dict[str, Any]]:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    sev: Dict[str, SeverityClass] = {}
    for name, sc in raw["sev_classes"].items():
        sev[name] = SeverityClass(
            desc=sc["desc"],
            attack=_to_stage_ranges(sc["attack"]),
            decay=_to_stage_ranges(sc["decay"]),
            sustain=_to_stage_ranges(sc["sustain"]),
            release=_to_stage_ranges(sc["release"]),
            duration=int(sc["duration"]),
        )
    meta = {
        "metadata": raw.get("metadata", {}),
        "defaults": raw.get("defaults", {}),
        "modulators": raw.get("modulators", {}),
    }
    return sev, meta

class FireEnvelope(Envelope):
    def __init__(self, path: str = "data/disturbances/envelopes/fire_envelope.yaml"):
        sev, meta = _load_envelope_from_path(path)
        super().__init__(sev_classes=sev, meta=meta)

class WindEnvelope(Envelope):
    def __init__(self, path: str = "data/disturbances/envelopes/wind_envelope.yaml"):
        sev, meta = _load_envelope_from_path(path)
        super().__init__(sev_classes=sev, meta=meta)

# ------------------- Kernel -------------------

class Kernel:
    """
    YAML-backed kernel that converts envelope proportions to multipliers.
    Structure expected at data/disturbances/kernels/<kind>_kernel.yaml:
      defaults:
        combine: "multiplicative" | "additive"   # how to combine per-stage effects
        floor: 0.60
        ceiling: 1.10
      # optional knobs per metric or stage if you want
    """
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        dfl = params.get("defaults", {})
        self.combine: str = dfl.get("combine", "multiplicative")
        self.floor: float = float(dfl.get("floor", 0.60))
        self.ceiling: float = float(dfl.get("ceiling", 1.10))

    @classmethod
    def from_yaml(cls, path: str) -> "Kernel":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls(raw or {})

    def proportion_to_multiplier(self, p: float) -> float:
        """
        Convert an envelope proportion p in [0,1] to a growth multiplier.
        Default: multiplier = clip(1 - p, floor, ceiling).
        """
        m = 1.0 - float(p)
        return max(self.floor, min(self.ceiling, m))

class FireKernel(Kernel):
    def __init__(self, path: str = "data/disturbances/kernels/fire_kernel.yaml"):
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        super().__init__(raw)

class WindKernel(Kernel):
    def __init__(self, path: str = "data/disturbances/kernels/wind_kernel.yaml"):
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        super().__init__(raw)

# ------------------- Disturbance event -------------------

@dataclass
class DisturbanceEvent:
    """A single disturbance occurrence anchored at start_age."""
    envelope: Envelope
    kernel: Kernel
    start_age: float
    severity: float            # scalar in (0,1)
    seed: Optional[int] = None # for sampling within ranges

    def _severity_class_key(self) -> str:
        """
        Map severity ∈ (0,1) to a sev_class key by parsing the numeric suffix 'x_y' in keys.
        Picks the class whose [x, y] percent range contains severity*100.
        Fallback: first key.
        """
        pct = 100.0 * max(0.0, min(1.0, self.severity))
        for k in self.envelope.keys():
            parts = k.split("_")
            try:
                lo = float(parts[-2])
                hi = float(parts[-1])
                if lo <= pct <= hi:
                    return k
            except Exception:
                continue
        return next(iter(self.envelope.keys()))

    def _stage_at(self, age: float, sev_cls: SeverityClass) -> Stage:
        """
        Split duration into 4 phases: attack, decay, sustain, release.
        Heuristic partition: 1, 1, max(duration-2,0), 1 years respectively,
        truncating if duration < 4.
        """
        d = max(1, int(sev_cls.duration))
        # Build edges
        segments: list[Tuple[Stage, int]] = []
        if d == 1:
            segments = [("attack", 1)]
        elif d == 2:
            segments = [("attack", 1), ("decay", 1)]
        elif d == 3:
            segments = [("attack", 1), ("decay", 1), ("release", 1)]
        else:
            segments = [("attack", 1), ("decay", 1), ("sustain", d - 2), ("release", 1)]
        t = age - self.start_age
        if t < 0:
            return "attack"  # not started; harmless default
        acc = 0
        for name, years in segments:
            acc_next = acc + years
            if t < acc_next:
                return name  # type: ignore
            acc = acc_next
        return "release"

    def _apply_modulators(self, metric: Metric, m: float, stand_state: Any) -> float:
        """
        Example modulator: young_stand from envelope.meta["modulators"].
        If stand dominant height < threshold, scale toward multiplier_range[1].
        """
        mods = self.envelope.meta.get("modulators", {}) or {}
        ys = mods.get("young_stand")
        if ys and hasattr(stand_state, "hd"):
            threshold = float(ys.get("threshold_height_ft", 0.0))
            lo, hi = ys.get("multiplier_range", [1.0, 1.0])
            lo, hi = float(lo), float(hi)
            if stand_state.hd < threshold:
                # linear ramp 0→threshold
                frac = 1.0 - max(0.0, min(1.0, stand_state.hd / threshold))
                m = m * (1.0 + frac * (hi - 1.0))
        return m

    def multipliers(self, age: float, stand_state: Any) -> Dict[Metric, float]:
        """
        Compute per-metric multipliers to apply at 'age'.
        Steps:
          1) pick sev class from severity
          2) pick stage from age - start_age and class.duration
          3) sample envelope proportion for each metric
          4) convert to multiplier via kernel
          5) apply envelope defaults floor/ceiling and modulators
        Returns dict of multipliers for metrics present in envelope.
        """
        random.seed(self.seed)
        key = self._severity_class_key()
        sev_cls = self.envelope[key]
        stage = self._stage_at(age, sev_cls)

        # Sample proportions
        props: Dict[Metric, float] = {}
        for metric in ("basal_area", "volume", "height"):
            lo, hi = sev_cls.range_for(stage, metric)  # type: ignore
            props[metric] = random.uniform(lo, hi)

        # Convert to multipliers using kernel and clamp by envelope defaults
        m: Dict[Metric, float] = {}
        floor = float(self.envelope.meta.get("defaults", {}).get("floor", 0.60))
        ceiling = float(self.envelope.meta.get("defaults", {}).get("ceiling", 1.10))
        for metric, p in props.items():
            mi = self.kernel.proportion_to_multiplier(p)
            mi = max(floor, min(ceiling, mi))
            mi = self._apply_modulators(metric, mi, stand_state)
            m[metric] = mi
        return m

# ------------------- Convenience factories -------------------

def make_fire_event(start_age: float, severity: float | None = None, seed: int | None = None) -> DisturbanceEvent:
    env = FireEnvelope()
    ker = FireKernel()
    sev = severity if severity is not None else get_severity(seed)
    return DisturbanceEvent(envelope=env, kernel=ker, start_age=start_age, severity=sev, seed=seed)

def make_wind_event(start_age: float, severity: float | None = None, seed: int | None = None) -> DisturbanceEvent:
    env = WindEnvelope()
    ker = WindKernel()
    sev = severity if severity is not None else get_severity(seed)
    return DisturbanceEvent(envelope=env, kernel=ker, start_age=start_age, severity=sev, seed=seed)
