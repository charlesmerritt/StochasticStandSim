""" PMRC deterministic growth baseline for loblolly pine plantations

State variables this module updates per step:
- age (years)
- trees per acre (TPA)
- dominant height (HD, ft) OR site index at base age 25 (SI25, ft)
- region flag ("LCP" for Lower Coastal Plain; "UCP" for Upper Coastal Plain + Piedmont).

Only unthinned, unfertilized growth. Merchantability and thinning omitted by design.

Equations and coefficients follow Harrison & Borders (PMRC Technical Report 1996-1):
- Dominant height site-index curves for soil group A (Pienaar & Shiver form) and a Chapman–Richards
  height projection equation with parameters k=0.014452 and m=0.8216.
- Survival with asymptote 100 TPA.
- Basal area prediction equation parameters by region (Table 15). BA is derived each step when needed.
"""

from __future__ import annotations

import os
import sys

# When executed as a script (`python core/growth.py`), Python prepends the package directory
# (`core/`) to sys.path. That causes stdlib imports (e.g., `enum -> import types`) to resolve to
# our local `core/types.py`, breaking the runtime. This drops the package directory from sys.path
# and prepends the repo root instead.
_PKG_DIR = os.path.dirname(__file__)
_ROOT_DIR = os.path.dirname(_PKG_DIR)
if sys.path and sys.path[0] == _PKG_DIR:
    sys.path[0] = _ROOT_DIR
else:
    if _PKG_DIR in sys.path:
        sys.path.remove(_PKG_DIR)
    if _ROOT_DIR not in sys.path:
        sys.path.insert(0, _ROOT_DIR)

from dataclasses import dataclass
from enum import Enum
from math import exp, log
from typing import Any, Optional, Tuple

from .disturbances import BaseDisturbance, ThinningDisturbance, FireDisturbance, WindDisturbance


def _debug_enabled() -> bool:
    flag = os.getenv("PMRC_GROWTH_DEBUG", "0")
    return flag.lower() not in ("", "0", "false")


def _debug_print(label: str, value: Any) -> None:
    if _debug_enabled():
        print(f"{label}: {value}")


class Region(str, Enum):
    """Physiographic regions supported by PMRC whole-stand models."""
    LCP = "LCP"  # Lower Coastal Plain
    UCP = "UCP"  # Upper Coastal Plain + Piedmont


@dataclass(frozen=True)
class GrowthConfig:
    """Static configuration for the PMRC growth model.

    Attributes:
        base_age: Base age for site index in years. PMRC uses 25.
        min_tpa_asymptote: Asymptotic lower bound in survival equation. PMRC uses 100.
    """
    base_age: int = 25
    min_tpa_asymptote: float = 100.0
# -------------------- Subproblem 1: Dominant height and site index --------------------
# We need: (a) project HD from age1 to age2, given HD1; (b) convert between SI25 and HD at any age.
# Use soil group A curves and PMRC Chapman–Richards parameters.

# Height projection parameters (soil group A, all regions)
_K = 0.014452  # growth rate
_M = 0.8216    # shape exponent

# Site-index conversion parameters (Pienaar & Shiver for soil group A)
# Two alternative published forms exist in the report:
#   Form A (older PS80) with constants 0.7476 and 0.05507
#   Form B (PMRC projection-consistent) with constants 0.30323 and 0.014452 and exponent 0.8216
# We provide both for completeness and default to the projection-consistent set so HD↔SI is compatible
# with the Chapman–Richards projector.

class SIForm(str, Enum):
    PS80 = "PS80"              # 0.7476, 0.05507
    PROJECTION = "PROJECTION"  # 0.30323, 0.014452, m=0.8216


def hd_project(age1: float, hd1: float, age2: float) -> float:
    """Eq. 12
    Project dominant height from age1 to age2 using PMRC Chapman–Richards.

    Args:
        age1: current stand age in years.
        hd1: dominant height at age1 in feet.
        age2: target age in years (> age1).
    Returns:
        Projected dominant height at age2 in feet.
    """
    if age2 <= 0 or age1 <= 0:
        raise ValueError("Ages must be positive.")
    if age2 < age1:
        raise ValueError("age2 must be >= age1 for projection.")
    num = 1.0 - exp(-_K * age2)
    den = 1.0 - exp(-_K * age1)
    if den <= 0.0:
        raise ValueError("Invalid age1 for projection; denominator nonpositive.")
    result = hd1 * (num / den) ** _M
    _debug_print("hd_project result", result)
    return result


def si_from_hd(hd: float, form: SIForm = SIForm.PROJECTION) -> float:
    """
    Eq. 10 & 13
    Convert dominant height at age to site index at base age 25.
    'A' in the report is not age! It is just a notational symbol, perhaps referring to soil group.
    Two forms. Default uses the projection-consistent parameters to keep HD and SI internally compatible.
    """
    if form == SIForm.PS80: # Eq. 10
        c, k, e = 0.7476, 0.05507, 1.435
        result = hd * (c / (1.0 - exp(-k))) ** e
    else: # Eq. 13
        # hd = si25 * \left(\frac{0.30323}{1-e^{-0.014452}}\right)^{-0.8216}
        c, k, m = 0.30323, _K, _M
        result = hd * (c / (1.0 - exp(-k)))** m
    _debug_print("si_from_hd result", result)
    return result


def hd_from_si(si25: float, form: SIForm) -> float:
    """
    Eq. 11 & 14
    Convert site index at base age 25 to dominant height at age.

    Inverse of si_from_hd under the same form selection.
    """
    if form == SIForm.PS80: # Eq. 11
        c, k, e = 0.7476, 0.05507, 1.435
        result = si25 * (c / (1.0 - exp(-k))) ** -e
    else: # Eq. 14
        # hd = si25 * \left(\frac{0.30323}{1-e^{-0.014452}}\right)^{0.8216}
        c, k, m = 0.30323, _K, _M
        result = si25 * (c / (1.0 - exp(-k))) ** -m
    _debug_print("hd_from_si result", result)
    return result



# -------------------- Subproblem 2: Survival (TPA) --------------------
# Asymptotic survival of 100 TPA, function of SI25 and ages.

def tpa_project(tpa1: float, si25: float, age1: float, age2: float, cfg: GrowthConfig = GrowthConfig()) -> float:
    """Project trees per acre using the PMRC survival equation with a 100 TPA asymptote.

    TPA2 = 100 + (TPA1 - 100)^{0.745339} * [1 + 0.00034252 * SI25 * (A2^{1.97472} - A1^{1.97472})]^{-1/0.745339}

    Constraints:
        - Only valid when TPA1 > asymptote (100) as per PMRC guidance.
        - If TPA1 <= 100, caller should hold TPA constant or apply a user-specified survival rate.
    """
    a = cfg.min_tpa_asymptote
    if tpa1 <= a:
        result = tpa1  # hold constant as a conservative default per report guidance
        _debug_print("tpa_project result", result)
        return result
    p = 0.745339
    g = 0.00034252
    r = 1.97472
    result = 100 + (((tpa1- 100)**-p) + (g**2) * si25 * ((age2**r) - (age1**r)))**-(1/p)
    _debug_print("tpa_project result", result)
    return result


# -------------------- Subproblem 3: Basal area (prediction form) --------------------
# We use the prediction form ln(BA) = b0 + b1/A + b2 ln(TPA) + b3 ln(HD) + b4 ln(TPA)/A + b5 ln(HD)/A.
# Coefficients differ by region (Table 15). BA in ft^2/acre. Optional Piedmont hardwood term is omitted here.

_BA_COEFFS = {
    Region.LCP:  (0.0,       -42.689283, 0.367244, 0.659985, 2.012724, 7.703502),
    Region.UCP:  (-0.855557, -36.050347, 0.299071, 0.980246, 3.309212, 3.787258),
}


def ba_predict(age: float, tpa: float, hd: float, region: Region) -> float:
    """Eq. 16
    Predict basal area per acre at given age, trees per acre, and dominant height.

    Returns BA in ft^2/acre. Strictly used for derived outputs. Not stored in state.
    """
    if tpa <= 0 or hd <= 0 or age <= 0:
        raise ValueError("age, tpa, and hd must be positive.")
    b0, b1, b2, b3, b4, b5 = _BA_COEFFS[region]
    # Eq. 16
    lnBA = (
        b0
        + (b1 / age)
        + (b2 * log(tpa))
        + (b3 * log(hd))
        + (b4 * (log(tpa) / age))
        + (b5 * (log(hd) / age))
    )
    result = max(0.0, float(exp(lnBA)))
    _debug_print("ba_predict result", result)
    return result

def ba_project(age1: float, tpa1: float, tpa2: float, ba1: float, hd1: float, hd2: float, age2: float, region: Region) -> float:
    """Eq. 17
    Project basal area per acre at given age, trees per acre, and dominant height.

    Returns BA in ft^2/acre. Strictly used for derived outputs. Not stored in state.
    """
    if tpa1 <= 0 or hd1 <= 0 or age1 <= 0 or age2 <= 0:
        raise ValueError("age1, tpa1, hd1, and age2 must be positive.")
    if age2 <= age1:
        raise ValueError("age2 must be > age1 for projection.")
    b0, b1, b2, b3, b4, b5 = _BA_COEFFS[region]
    # Eq. 17
    lnBA2 = (
        log(ba1) + (b1 * ((1/age2) - (1/age1))) + 
        (b2 * (log(tpa2) - log(tpa1))) + 
        (b3 * (log(hd2) - log(hd1))) + 
        (b4 * ((log(tpa2)/age2) - (log(tpa1)/age1))) + 
        (b5 * ((log(hd2)/age2) - (log(hd1)/age1)))
    )
    result = max(0.0, float(exp(lnBA2)))
    _debug_print("ba_project result", result)
    return result

def competition_index(ba_after: float, ba_unthinned: float) -> float:
    """
    Eq. (26) CI = 1 - BA_after / BA_unthinned.
    Caller must supply BA of the residual stand immediately after thinning (same age)
    and BA of the unthinned counterpart at that age. Returns CI in [0, 1].
    """
    if ba_unthinned <= 0:
        raise ValueError("ba_unthinned must be positive.")
    ci = 1.0 - (ba_after / ba_unthinned)
    return max(0.0, min(1.0, ci))  # clip numerically

def ci_project(ci1: float, age1: float, age2: float, region: Region=Region.UCP) -> float:
    """
    Eq. (27) recovery envelope: CI2 = CI1 * exp(-beta * (A2 - A1)).
    beta > 0 governs recovery speed toward the unthinned condition.
    """
    if region == Region.UCP:
        beta = 0.076472
    elif region == Region.LCP:
        beta = 0.110521  
    else:
        raise ValueError("region must be Region.UCP or Region.LCP.")
    if age2 <= age1:
        raise ValueError("age2 must be > age1.")
    if ci1 < 0.0:
        raise ValueError("ci1 must be >= 0.")
    dt = age2 - age1
    return max(0.0, ci1 * exp(-beta * dt))

def ba_thinned(ba_unthinned2: float, ci2: float) -> float:
    """
    Eq. (28) reconstruction at A2: BA_thinned,2 = BA_unthinned,2 * (1 - CI2).
    """
    if ba_unthinned2 < 0:
        raise ValueError("ba_unthinned2 must be >= 0.")
    ci2 = max(0.0, min(1.0, ci2))
    return ba_unthinned2 * (1.0 - ci2)
    
def tvob(age: float, tpa: float, hd: float, ba: float, region: Region = Region.UCP) -> float:
    """
    PMRC Eq. 19 – Total volume outside bark (ft^3/ac) for UCP.
    ln(Y) = b0 + b1*ln(HD) + b2*ln(BA) + b3*(ln(TPA)/A)
            + b4*(ln(HD)/A) + b5*(ln(BA)/A)
    """

    if age <= 0 or tpa <= 0 or hd <= 0 or ba <= 0:
        return 0.0

    if region == Region.UCP:
        b0 = 0.0
        b1 = 0.268552
        b2 = 1.368844
        b3 = -7.466863
        b4 = 8.934524
        b5 = 3.553411
    else:
        raise ValueError("TVOB coefficients currently implemented for Region.UCP only.")

    lnY = (
        b0
        + b1 * log(hd)
        + b2 * log(ba)
        + b3 * (log(tpa) / age)
        + b4 * (log(hd) / age)
        + b5 * (log(ba) / age)
    )
    result = float(exp(lnY))
    _debug_print("tvob result", result)
    return result

# -------------------- Subproblem 4: One-step state update --------------------
@dataclass
class StandState:
    """Minimal MDP state for growth.

    If si25 is provided, hd is ignored and recomputed each step for internal consistency.
    """
    age: float
    tpa: float
    region: Region
    si25: Optional[float] = None
    hd: Optional[float] = None
    ba: Optional[float] = None
    ba_unthinned: Optional[float] = None
    ci: Optional[float] = None
    pending_disturbances: Tuple[BaseDisturbance, ...] = ()
    tpa_unthinned: Optional[float] = None
    si_form: SIForm = SIForm.PROJECTION
    vol_ob: Optional[float] = None
    vol_ob_unthinned: Optional[float] = None
    active_envelopes: Tuple[dict, ...] = ()  # Track active disturbance envelopes for BA growth modulation

    def resolved_hd(self) -> float:
        if self.hd is not None:
            return self.hd
        if self.si25 is not None:
            return hd_from_si(self.si25, form=self.si_form)
        raise ValueError("hd or si25 must be set.")


@dataclass(frozen=True)
class StandParams:
    """Configuration bundle for initializing example stands."""

    name: str
    age: float
    tpa: float
    region: Region
    si25: Optional[float] = None
    hd: Optional[float] = None
    ba: Optional[float] = None
    ba_unthinned: Optional[float] = None
    ci: Optional[float] = None
    disturbances: Tuple[BaseDisturbance, ...] = ()
    fert: Optional[Tuple[float, bool]] = None
    si_form: SIForm = SIForm.PROJECTION
    vol_ob: Optional[float] = None
    vol_ob_unthinned: Optional[float] = None

    def to_state(self) -> StandState:
        hd_value = self.hd
        if hd_value is None:
            if self.si25 is None:
                raise ValueError(f"StandParams '{self.name}' requires either hd or si25.")
            hd_value = hd_from_si(self.si25, form=self.si_form)

        si_value = self.si25 if self.si25 is not None else si_from_hd(hd_value, form=self.si_form)

        ba_unthinned_value = self.ba_unthinned
        if ba_unthinned_value is None:
            ba_unthinned_value = ba_predict(self.age, self.tpa, hd_value, self.region)
        ba_value = self.ba if self.ba is not None else ba_unthinned_value
        ci_value = self.ci
        if ci_value is None and ba_value is not None and ba_unthinned_value > 0:
            ci_value = competition_index(ba_value, ba_unthinned_value)
        if ci_value is None:
            ci_value = 0.0

        for ev in self.disturbances:
            if isinstance(ev, ThinningDisturbance) and not (0.0 <= ev.removal_fraction < 1.0):
                raise ValueError(f"Thinning removal fraction must be in [0,1); got {ev.removal_fraction}")
        sorted_events = tuple(sorted(self.disturbances, key=lambda ev: ev.age))

        vol_unthinned_value = self.vol_ob_unthinned
        if vol_unthinned_value is None:
            vol_unthinned_value = tvob(self.age, self.tpa, hd_value, ba_unthinned_value, self.region)

        vol_value = self.vol_ob
        if vol_value is None:
            vol_value = tvob(self.age, self.tpa, hd_value, ba_value, self.region)

        return StandState(
            age=self.age,
            tpa=self.tpa,
            region=self.region,
            si25=si_value,
            hd=hd_value,
            ba=ba_value,
            ba_unthinned=ba_unthinned_value,
            ci=ci_value,
            pending_disturbances=sorted_events,
            tpa_unthinned=self.tpa,
            si_form=self.si_form,
            vol_ob=vol_value,
            vol_ob_unthinned=vol_unthinned_value,
        )


def _advance_internal(
    state: StandState,
    dt: float,
    cfg: GrowthConfig,
) -> Tuple[StandState, float, Tuple[dict, ...], float, float]:
    if dt <= 0:
        raise ValueError("dt must be positive.")

    a1 = state.age
    a2 = a1 + dt

    hd1 = state.resolved_hd()
    si_value = state.si25 if state.si25 is not None else si_from_hd(hd1, form=state.si_form)

    tpa_unthinned_current = state.tpa_unthinned if state.tpa_unthinned is not None else state.tpa
    ba_unthinned_current = (
        state.ba_unthinned
        if state.ba_unthinned is not None
        else ba_predict(a1, tpa_unthinned_current, hd1, state.region)
    )

    tpa_actual_current = state.tpa
    ba_actual_current = state.ba if state.ba is not None else ba_unthinned_current
    ci_current = state.ci
    if ci_current is None and ba_unthinned_current > 0:
        ci_current = competition_index(ba_actual_current, ba_unthinned_current)

    event_logs: list[dict] = []
    remaining_disturbances: list[BaseDisturbance] = []
    new_envelopes: list[dict] = []  # Track new envelopes to add
    vol_actual_current = state.vol_ob
    if vol_actual_current is None:
        try:
            vol_actual_current = tvob(a1, tpa_actual_current, hd1, ba_actual_current or 0.0, state.region)
        except ValueError:
            vol_actual_current = 0.0
    vol_unthinned_current = state.vol_ob_unthinned
    if vol_unthinned_current is None:
        try:
            vol_unthinned_current = tvob(a1, tpa_unthinned_current, hd1, ba_unthinned_current or 0.0, state.region)
        except ValueError:
            vol_unthinned_current = 0.0

    for disturbance in state.pending_disturbances:
        # Check if disturbance occurs during this step [a1, a2]
        if a1 < disturbance.age <= a2 or abs(disturbance.age - a1) <= 1e-9:
            # Apply disturbance at current age
            if isinstance(disturbance, ThinningDisturbance):
                # Thinning: proportional removal
                before_tpa = tpa_actual_current
                before_ba = ba_actual_current
                before_vol = vol_actual_current
                before_hd = hd1

                residual = 1.0 - disturbance.removal_fraction
                tpa_actual_current *= residual
                ba_actual_current *= residual
                vol_actual_current *= residual

                if ba_unthinned_current > 0:
                    ci_current = competition_index(ba_actual_current, ba_unthinned_current)

                event_logs.append(
                    {
                        "type": "thinning",
                        "age": disturbance.age,
                        "tpa_before": before_tpa,
                        "tpa_after": tpa_actual_current,
                        "ba_before": before_ba,
                        "ba_after": ba_actual_current,
                        "vol_before": before_vol,
                        "vol_after": vol_actual_current,
                        "hd_before": before_hd,
                        "hd_after": hd1,
                    }
                )
            
            elif isinstance(disturbance, (FireDisturbance, WindDisturbance)):
                # Fire/Wind: Load kernel and apply losses
                before_tpa = tpa_actual_current
                before_ba = ba_actual_current
                before_vol = vol_actual_current
                before_hd = hd1
                
                # Load appropriate kernel from YAML
                from .disturbances import load_kernel
                from pathlib import Path
                
                dist_type = "fire" if isinstance(disturbance, FireDisturbance) else "wind"
                kernel_path = Path(__file__).parent.parent / "data" / "disturbances" / "kernels" / f"{dist_type}_kernel.yaml"
                
                try:
                    kernel = load_kernel(kernel_path)
                    sev_class = disturbance.get_severity_class()
                    
                    # Sample random losses from kernel distributions
                    post_dist = kernel.sample_losses(
                        sev_class,
                        ba=ba_actual_current,
                        vol=vol_actual_current,
                        hd=hd1,
                        tpa=tpa_actual_current
                    )
                    
                    tpa_actual_current = post_dist['tpa']
                    ba_actual_current = post_dist['ba']
                    vol_actual_current = post_dist['vol']
                    hd1 = post_dist['hd']
                    
                    # Load envelope for future BA growth modulation
                    from .disturbances import load_envelope_set
                    envelope_path = Path(__file__).parent.parent / "data" / "disturbances" / "envelopes" / f"{dist_type}_envelope.yaml"
                    
                    try:
                        envelope_set = load_envelope_set(envelope_path)
                        envelope = envelope_set.get_envelope(sev_class)
                        
                        # Track this envelope for future BA growth modulation
                        # Note: envelope duration will come from YAML in future
                        active_envelope = {
                            'type': dist_type,
                            'age_occurred': a1,
                            'envelope': envelope,
                            'severity_class': sev_class,
                        }
                        new_envelopes.append(active_envelope)
                    except (FileNotFoundError, ValueError, KeyError) as e_env:
                        # Envelope loading failed - continue without it
                        import warnings
                        warnings.warn(f"Could not load {dist_type} envelope: {e_env}. Continuing without envelope effects.")
                    
                except (FileNotFoundError, ValueError, KeyError) as e:
                    # If kernel loading fails, log warning and skip disturbance
                    import warnings
                    warnings.warn(f"Could not load {dist_type} kernel: {e}. Disturbance not applied.")
                
                # Update competition index
                if ba_unthinned_current > 0:
                    ci_current = competition_index(ba_actual_current, ba_unthinned_current)
                
                event_logs.append(
                    {
                        "type": dist_type,
                        "age": disturbance.age,
                        "severity": disturbance.severity,
                        "severity_class": disturbance.get_severity_class(),
                        "tpa_before": before_tpa,
                        "tpa_after": tpa_actual_current,
                        "ba_before": before_ba,
                        "ba_after": ba_actual_current,
                        "vol_before": before_vol,
                        "vol_after": vol_actual_current,
                        "hd_before": before_hd,
                        "hd_after": hd1,
                    }
                )
        else:
            remaining_disturbances.append(disturbance)

    hd2 = hd_project(a1, hd1, a2)
    tpa2_actual = tpa_project(tpa_actual_current, si_value, a1, a2, cfg=cfg)
    tpa2_unthinned = tpa_project(tpa_unthinned_current, si_value, a1, a2, cfg=cfg)

    ba2_unthinned = ba_project(
        age1=a1,
        tpa1=tpa_unthinned_current,
        tpa2=tpa2_unthinned,
        ba1=ba_unthinned_current,
        hd1=hd1,
        hd2=hd2,
        age2=a2,
        region=state.region,
    )

    ci_next: Optional[float] = None
    if ci_current is not None:
        ci_next = ci_project(ci_current, a1, a2, state.region)
    elif ba_unthinned_current > 0:
        ci_est = competition_index(ba_actual_current, ba_unthinned_current)
        ci_next = ci_project(ci_est, a1, a2, state.region)

    if ci_next is not None:
        ba2_actual = ba_thinned(ba2_unthinned, ci_next)
    else:
        ba2_actual = ba2_unthinned
    
    # Apply active envelope multipliers to BA growth increment
    if state.active_envelopes:
        ba_increment = ba2_actual - ba_actual_current
        combined_multiplier = 1.0
        
        for env_info in state.active_envelopes:
            years_since = a1 - env_info['age_occurred']
            envelope = env_info['envelope']
            
            # Calculate ADSR multiplier for this year
            if years_since < envelope.attack_duration_years:
                mult = 1.0 - envelope.attack_drop
            elif years_since < envelope.attack_duration_years + envelope.decay_years:
                # Decay phase - linear interpolation
                t = (years_since - envelope.attack_duration_years) / envelope.decay_years
                attack_val = 1.0 - envelope.attack_drop
                mult = attack_val + (envelope.sustain_level - attack_val) * t
            else:
                # Sustain phase (duration check will be added when YAML has it)
                mult = envelope.sustain_level
            
            # Compound multiple envelope effects
            combined_multiplier *= mult
        
        # Apply combined multiplier to BA increment
        ba_increment_modulated = ba_increment * combined_multiplier
        ba2_actual = ba_actual_current + ba_increment_modulated

    try:
        vol2_unthinned = tvob(a2, tpa2_unthinned, hd2, ba2_unthinned, state.region)
    except ValueError:
        vol2_unthinned = 0.0

    try:
        vol2_actual = tvob(a2, tpa2_actual, hd2, ba2_actual, state.region)
    except ValueError:
        vol2_actual = 0.0

    # Combine existing and new envelopes
    all_active_envelopes = list(state.active_envelopes) + new_envelopes
    
    next_state = StandState(
        age=a2,
        tpa=tpa2_actual,
        region=state.region,
        si25=si_value,
        hd=hd2,
        ba=ba2_actual,
        ba_unthinned=ba2_unthinned,
        ci=ci_next,
        pending_disturbances=tuple(remaining_disturbances),
        tpa_unthinned=tpa2_unthinned,
        si_form=state.si_form,
        vol_ob=vol2_actual,
        vol_ob_unthinned=vol2_unthinned,
        active_envelopes=tuple(all_active_envelopes),
    )

    _debug_print("step next_state", next_state)
    _debug_print("step ba_unthinned", ba2_unthinned)
    _debug_print("step ba_result", ba2_actual)

    return next_state, ba2_actual, tuple(event_logs), tpa2_unthinned, ba2_unthinned, vol2_unthinned


def step(state: StandState, dt: float = 1.0, cfg: GrowthConfig = GrowthConfig()) -> Tuple[StandState, float]:
    next_state, ba_value, _, _, _, _ = _advance_internal(state, dt, cfg)
    return next_state, ba_value


def step_with_log(
    state: StandState, dt: float = 1.0, cfg: GrowthConfig = GrowthConfig()
) -> Tuple[StandState, float, Tuple[dict, ...], float, float, float]:
    return _advance_internal(state, dt, cfg)


# -------------------- Subproblem 5: Convenience multi-step runner --------------------

def run_horizon(state: StandState, years: float, dt: float = 1.0) -> list[Tuple[StandState, float]]:
    """Project a stand forward for a given horizon.

    Returns a list of (state_t, BA_t) for each completed step. Includes the terminal state only.
    """
    if years <= 0:
        raise ValueError("years must be positive.")
    if abs(dt - 1.0) > 1e-9:
        raise ValueError("run_horizon currently assumes dt == 1.0 year.")
    steps = int(round(years / dt))
    out: list[Tuple[StandState, float]] = []
    s = state
    for _ in range(steps):
        s, ba = step(s, dt=dt)
        out.append((s, ba))
        _debug_print("run_horizon step_state", s)
        _debug_print("run_horizon step_ba", ba)
    return out


# -------------------- Defaults and simple test harness --------------------
EXAMPLE_STANDS: dict[str, StandParams] = {
    "ucp_baseline": StandParams(
        name="ucp_baseline",
        age=1.0,
        tpa=600.0,
        region=Region.UCP,
        si25=60.0,
        disturbances=(ThinningDisturbance(age=10.0, removal_fraction=0.15),),
    ),
    "case_1_low_si": StandParams(
        name="case_1_low_si",
        age=1.0,
        tpa=505.0,
        region=Region.UCP,
        si25=60.0,
    ),
    "case_1_high_si": StandParams(
        name="case_1_high_si",
        age=1.0,
        tpa=505.0,
        region=Region.UCP,
        si25=80.0,
    ),
    "fire_moderate": StandParams(
        name="fire_moderate",
        age=1.0,
        tpa=600.0,
        region=Region.UCP,
        si25=60.0,
        disturbances=(
            FireDisturbance(age=15.0, severity=0.35),  # Moderate fire at year 15
        ),
    ),
    "fire_severe": StandParams(
        name="fire_severe",
        age=1.0,
        tpa=600.0,
        region=Region.UCP,
        si25=60.0,
        disturbances=(
            FireDisturbance(age=12.0, severity=0.65),  # Severe fire at year 12
        ),
    ),
    "wind_event": StandParams(
        name="wind_event",
        age=1.0,
        tpa=600.0,
        region=Region.UCP,
        si25=60.0,
        disturbances=(
            WindDisturbance(age=18.0, severity=0.45),  # Wind event at year 18
        ),
    ),
    "multi_disturbance": StandParams(
        name="multi_disturbance",
        age=1.0,
        tpa=600.0,
        region=Region.UCP,
        si25=60.0,
        disturbances=(
            ThinningDisturbance(age=8.0, removal_fraction=0.20),
            FireDisturbance(age=15.0, severity=0.30),
            ThinningDisturbance(age=22.0, removal_fraction=0.15),
        ),
    ),
}


if __name__ == "__main__":
    params = EXAMPLE_STANDS["ucp_baseline"]
    s0 = params.to_state()
    traj = run_horizon(s0, years=10, dt=1.0)
    # Print last state for quick verification
    sT, baT = traj[-1]
    print({"age": sT.age, "tpa": round(sT.tpa, 1), "hd": round(sT.hd or 0.0, 2), "ba": round(baT, 2)})
