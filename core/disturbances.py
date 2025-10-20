"""Dataclasses representing deterministic disturbances."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from .rng import rng


def random_severity(seed: int | None = None) -> float:
    """Generate random severity in (0, 1) exclusive range.
    
    Returns a value that cannot be exactly 0 or 1, suitable for
    discretization into severity classes.
    """
    val = rng(seed)
    # Ensure we never return exactly 0.0 or 1.0
    if val == 0.0:
        val = 0.001
    elif val == 1.0:
        val = 0.999
    return val


# ==================== Kernel and Envelope Structures ====================

@dataclass(frozen=True)
class ADSREnvelope:
    """ADSR (Attack-Decay-Sustain-Release) envelope for post-disturbance BA growth.
    
    Defines how BA GROWTH INCREMENT multipliers evolve over time after a disturbance.
    These multipliers affect the per-year CHANGE in BA, not the total BA.
    
    Example: If normal BA growth is +2 ft²/ac/year and multiplier is 0.5,
    then actual growth that year is +1 ft²/ac/year.
    
    Values represent multipliers applied to BA growth increments (1.0 = normal growth).
    """
    attack_drop: float  # Initial drop in growth rate (0-1)
    attack_duration_years: int  # Years at reduced growth rate
    decay_years: int  # Years to recover from attack to sustain
    sustain_level: float  # Sustained growth rate multiplier (can be >1.0 for compensatory growth)
    sustain_years: int = 0  # Years at sustain level (0 = indefinite)
    release_years: int = 0  # Years to return to 1.0x baseline growth
    
    def __post_init__(self) -> None:
        """Validate envelope parameters."""
        if not (0.0 <= self.attack_drop <= 1.0):
            raise ValueError(f"attack_drop must be in [0, 1], got {self.attack_drop}")
        if self.sustain_level < 0.0:
            raise ValueError(f"sustain_level must be >= 0, got {self.sustain_level}")
        if self.attack_duration_years < 0:
            raise ValueError(f"attack_duration_years must be >= 0, got {self.attack_duration_years}")


@dataclass(frozen=True)
class DisturbanceKernel:
    """Kernel defining immediate losses per severity class across multiple metrics.
    
    Each severity class has a 5-number distribution (min, q1, median, q3, max)
    for immediate loss percentage in stand metrics:
    - basal_area: BA loss percentage
    - volume: Volume loss percentage  
    - height: Height loss percentage
    - density: TPA (trees per acre) loss percentage
    """
    sev_classes: Mapping[str, Mapping[str, tuple[float, float, float, float, float]]]  # class_name -> {metric -> 5-number}
    
    def get_loss_distribution(self, severity_class: str, metric: str) -> tuple[float, float, float, float, float]:
        """Get the 5-number loss distribution for a severity class and metric.
        
        Args:
            severity_class: Name of severity class (e.g., 'moderate_25_50')
            metric: Metric name ('basal_area', 'volume', 'height', 'density')
            
        Returns:
            (min, q1, median, q3, max) tuple representing loss percentage distribution
        """
        if severity_class not in self.sev_classes:
            raise ValueError(f"Severity class '{severity_class}' not found in kernel")
        class_data = self.sev_classes[severity_class]
        if metric not in class_data:
            raise ValueError(f"Metric '{metric}' not found for class '{severity_class}'")
        return class_data[metric]
    
    def get_all_losses(self, severity_class: str) -> dict[str, tuple[float, float, float, float, float]]:
        """Get all loss distributions for a severity class.
        
        Returns:
            Dict mapping metric names to their 5-number loss distributions
        """
        if severity_class not in self.sev_classes:
            raise ValueError(f"Severity class '{severity_class}' not found in kernel")
        return dict(self.sev_classes[severity_class])
    
    def sample_ba_loss(self, severity_class: str, metric: str = "basal_area") -> tuple[float, float, float, float, float]:
        """Backward compatibility method. Use get_loss_distribution instead."""
        return self.get_loss_distribution(severity_class, metric)
    
    def sample_losses(self, severity_class: str, ba: float, vol: float, hd: float, tpa: float, seed: int | None = None) -> dict[str, float]:
        """Sample random losses from kernel distributions and apply to stand metrics.
        
        Samples from a triangular distribution using the 5-number summary,
        approximating the underlying distribution.
        
        Args:
            severity_class: Severity class name
            ba: Current basal area
            vol: Current volume
            hd: Current height
            tpa: Current trees per acre
            seed: Random seed for reproducibility (optional)
            
        Returns:
            Dict with post-disturbance values: {'ba': ..., 'vol': ..., 'hd': ..., 'tpa': ...}
        """
        import random
        if seed is not None:
            random.seed(seed)
        
        all_losses = self.get_all_losses(severity_class)
        
        result = {}
        if 'basal_area' in all_losses:
            min_loss, q1, median, q3, max_loss = all_losses['basal_area']
            # Sample from triangular distribution (min, mode=median, max)
            sampled_loss = random.triangular(min_loss, max_loss, median)
            result['ba'] = ba * (1 - sampled_loss)
        else:
            result['ba'] = ba
            
        if 'volume' in all_losses:
            min_loss, q1, median, q3, max_loss = all_losses['volume']
            sampled_loss = random.triangular(min_loss, max_loss, median)
            result['vol'] = vol * (1 - sampled_loss)
        else:
            result['vol'] = vol
            
        if 'height' in all_losses:
            min_loss, q1, median, q3, max_loss = all_losses['height']
            sampled_loss = random.triangular(min_loss, max_loss, median)
            result['hd'] = hd * (1 - sampled_loss)
        else:
            result['hd'] = hd
            
        if 'density' in all_losses:
            min_loss, q1, median, q3, max_loss = all_losses['density']
            sampled_loss = random.triangular(min_loss, max_loss, median)
            result['tpa'] = tpa * (1 - sampled_loss)
        else:
            result['tpa'] = tpa
            
        return result
    
    def apply_median_losses(self, severity_class: str, ba: float, vol: float, hd: float, tpa: float) -> dict[str, float]:
        """Apply median losses from kernel to stand metrics. (Deprecated: use sample_losses)
        
        Args:
            severity_class: Severity class name
            ba: Current basal area
            vol: Current volume
            hd: Current height
            tpa: Current trees per acre
            
        Returns:
            Dict with post-disturbance values: {'ba': ..., 'vol': ..., 'hd': ..., 'tpa': ...}
        """
        all_losses = self.get_all_losses(severity_class)
        
        result = {}
        if 'basal_area' in all_losses:
            result['ba'] = ba * (1 - all_losses['basal_area'][2])
        else:
            result['ba'] = ba
            
        if 'volume' in all_losses:
            result['vol'] = vol * (1 - all_losses['volume'][2])
        else:
            result['vol'] = vol
            
        if 'height' in all_losses:
            result['hd'] = hd * (1 - all_losses['height'][2])
        else:
            result['hd'] = hd
            
        if 'density' in all_losses:
            result['tpa'] = tpa * (1 - all_losses['density'][2])
        else:
            result['tpa'] = tpa
            
        return result


@dataclass(frozen=True)
class EnvelopeSet:
    """Set of ADSR envelopes for different severity classes.
    
    Maps severity class names to their corresponding ADSR envelopes.
    Envelopes define multipliers that affect the INCREMENTAL BA GROWTH per year,
    NOT the total BA. They impede or enhance the per-year delta in basal area.
    """
    envelopes: Mapping[str, ADSREnvelope]
    metric: str = "basal_area"  # These envelopes apply to BA growth increments only
    
    def get_envelope(self, severity_class: str) -> ADSREnvelope:
        """Get the ADSR envelope for a severity class."""
        if severity_class not in self.envelopes:
            raise ValueError(f"Severity class '{severity_class}' not found in envelope set")
        return self.envelopes[severity_class]


@dataclass(frozen=True)
class BaseDisturbance:
    """Base class for disturbances with age."""
    age: float


@dataclass(frozen=True)
class ThinningDisturbance(BaseDisturbance):
    """Thinning disturbance with explicit removal fraction."""
    removal_fraction: float


@dataclass(frozen=True)
class FireDisturbance(BaseDisturbance):
    """Fire disturbance with severity in (0, 1).
    
    Severity is discretized into classes. Load kernel/envelope separately from YAML.
    """
    severity: float
    
    def __post_init__(self) -> None:
        """Validate severity is in valid range."""
        if not (0.0 < self.severity < 1.0):
            raise ValueError(f"Severity must be in (0, 1), got {self.severity}")
    
    def get_severity_class(self) -> str:
        """Discretize continuous severity into a severity class.
        
        5 classes: mild (0-10%), low (10-20%), moderate (20-50%), severe (50-80%), catastrophic (80-100%)
        """
        if self.severity < 0.10:
            return "mild_0_10"
        elif self.severity < 0.20:
            return "low_10_20"
        elif self.severity < 0.50:
            return "moderate_20_50"
        elif self.severity < 0.80:
            return "severe_50_80"
        else:
            return "catastrophic_80_100"


@dataclass(frozen=True)
class WindDisturbance(BaseDisturbance):
    """Wind disturbance with severity in (0, 1).
    
    Severity is discretized into classes. Load kernel/envelope separately from YAML.
    """
    severity: float
    
    def __post_init__(self) -> None:
        """Validate severity is in valid range."""
        if not (0.0 < self.severity < 1.0):
            raise ValueError(f"Severity must be in (0, 1), got {self.severity}")
    
    def get_severity_class(self) -> str:
        """Discretize continuous severity into a severity class.
        
        5 classes: mild (0-10%), low (10-20%), moderate (20-50%), severe (50-80%), catastrophic (80-100%)
        """
        if self.severity < 0.10:
            return "mild_0_10"
        elif self.severity < 0.20:
            return "low_10_20"
        elif self.severity < 0.50:
            return "moderate_20_50"
        elif self.severity < 0.80:
            return "severe_50_80"
        else:
            return "catastrophic_80_100"


Disturbance = BaseDisturbance


# ==================== Loaders ====================

def load_kernel(path: str | Path) -> DisturbanceKernel:
    """Load a disturbance kernel from a YAML file.
    
    Args:
        path: Path to kernel YAML file
        
    Returns:
        DisturbanceKernel with severity classes and loss distributions
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    if not isinstance(data, Mapping):
        raise ValueError(f"Kernel file {path} is empty or invalid")
    
    sev_classes_raw = data.get("sev_classes", {})
    if not sev_classes_raw:
        raise ValueError(f"No severity classes found in {path}")
    
    # Parse severity classes into proper structure
    sev_classes = {}
    for class_name, class_data in sev_classes_raw.items():
        loss_ranges = class_data.get("immediate_loss_range", {})
        parsed_ranges = {}
        for metric, values in loss_ranges.items():
            if isinstance(values, (list, tuple)) and len(values) == 5:
                parsed_ranges[metric] = tuple(float(v) for v in values)
            else:
                raise ValueError(
                    f"Expected 5-number distribution for '{metric}' in class '{class_name}', got {values}"
                )
        sev_classes[class_name] = parsed_ranges
    
    return DisturbanceKernel(sev_classes=sev_classes)


def load_envelope_set(path: str | Path, metric: str = "basal_area") -> EnvelopeSet:
    """Load a set of ADSR envelopes from a YAML file.
    
    Args:
        path: Path to envelope YAML file
        metric: Metric to extract (default: basal_area)
        
    Returns:
        EnvelopeSet with severity class envelopes
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    if not isinstance(data, Mapping):
        raise ValueError(f"Envelope file {path} is empty or invalid")
    
    # Try different possible structures
    metric_block = data.get("metrics") or data.get("envelopes_by_metric")
    if metric_block and metric in metric_block:
        envelopes_raw = metric_block[metric]
    else:
        envelopes_raw = (
            data.get("envelopes_by_class")
            or data.get("envelopes_by_scorch_class")
            or data.get("sev_classes")  # Also check for sev_classes
            or data.get("envelopes")
            or {}
        )
    
    if not envelopes_raw:
        raise ValueError(f"No envelopes found in {path}")
    
    # Parse envelopes
    envelopes = {}
    for class_name, class_data in envelopes_raw.items():
        adsr = class_data.get("ADSR", {})
        
        # Extract ADSR parameters with defaults
        # Handle dict, list, or scalar values
        def _extract_mid(val, default):
            if val is None:
                return default
            if isinstance(val, Mapping):
                return val.get("mid", default)
            if isinstance(val, (list, tuple)):
                # For ranges like [min, mid, max], take middle value
                return val[len(val) // 2] if val else default
            return val
        
        attack_drop = float(_extract_mid(adsr.get("attack_drop"), 0.0))
        attack_duration = int(_extract_mid(adsr.get("attack_duration_years"), 1))
        decay_years = int(_extract_mid(adsr.get("decay_years"), 0))
        sustain_level = float(_extract_mid(adsr.get("sustain_level"), 1.0))
        sustain_years = int(_extract_mid(adsr.get("sustain_years"), 0))
        release_years = int(_extract_mid(adsr.get("release_years"), 0))
        
        envelopes[class_name] = ADSREnvelope(
            attack_drop=attack_drop,
            attack_duration_years=attack_duration,
            decay_years=decay_years,
            sustain_level=sustain_level,
            sustain_years=sustain_years,
            release_years=release_years,
        )
    
    return EnvelopeSet(envelopes=envelopes, metric=metric)

