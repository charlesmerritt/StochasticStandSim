"""
Demonstration of how disturbances work with kernels and envelopes.

This shows:
1. Creating disturbances with random severity
2. Loading kernels and envelopes from YAML files
3. Discretizing severity into classes
4. Sampling BA loss from kernels
5. Applying ADSR envelopes over time
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.disturbances import (
    FireDisturbance,
    WindDisturbance,
    ThinningDisturbance,
    random_severity,
    load_kernel,
    load_envelope_set,
)


def demo_basic_disturbances():
    """Show basic disturbance creation."""
    print("=" * 60)
    print("BASIC DISTURBANCES")
    print("=" * 60)
    
    # Thinning - explicit removal fraction
    thin = ThinningDisturbance(age=10.0, removal_fraction=0.4)
    print(f"\nThinning at age {thin.age}: {thin.removal_fraction*100}% removal")
    
    # Fire - random severity
    fire_severity = random_severity(seed=42)
    fire = FireDisturbance(age=15.0, severity=fire_severity)
    print(f"\nFire at age {fire.age}: severity={fire.severity:.3f}")
    print(f"  Severity class: {fire.get_severity_class()}")
    
    # Wind - random severity
    wind_severity = random_severity(seed=123)
    wind = WindDisturbance(age=20.0, severity=wind_severity)
    print(f"\nWind at age {wind.age}: severity={wind.severity:.3f}")
    print(f"  Severity class: {wind.get_severity_class()}")


def demo_with_kernel_and_envelope():
    """Show how to use kernels and envelopes with disturbances."""
    print("\n" + "=" * 60)
    print("DISTURBANCES WITH KERNELS AND ENVELOPES")
    print("=" * 60)
    
    data_dir = Path("data/disturbances")
    
    # Load fire kernel and envelope
    fire_kernel_path = data_dir / "kernels" / "fire_kernel.yaml"
    fire_envelope_path = data_dir / "envelopes" / "fire_envelope.yaml"
    
    try:
        # Load kernel (defines immediate BA loss)
        fire_kernel = load_kernel(fire_kernel_path)
        print(f"\n✓ Loaded fire kernel from {fire_kernel_path.name}")
        print(f"  Severity classes: {list(fire_kernel.sev_classes.keys())}")
        
        # Load envelope set (defines post-disturbance trajectory)
        fire_envelopes = load_envelope_set(fire_envelope_path)
        print(f"\n✓ Loaded fire envelopes from {fire_envelope_path.name}")
        print(f"  Envelope classes: {list(fire_envelopes.envelopes.keys())}")
        
        # Create fire disturbance (kernel/envelope loaded separately)
        fire_severity = random_severity(seed=42)
        fire = FireDisturbance(
            age=15.0,
            severity=fire_severity
        )
        
        print(f"\n🔥 Fire Disturbance:")
        print(f"  Age: {fire.age}")
        print(f"  Severity: {fire.severity:.3f}")
        print(f"  Class: {fire.get_severity_class()}")
        
        # Sample BA loss from kernel
        sev_class = fire.get_severity_class()
        
        # Try to find matching or approximate class in kernel
        kernel_class = None
        if sev_class in fire_kernel.sev_classes:
            kernel_class = sev_class
        else:
            # Try to find approximate match based on severity percentage
            print(f"\n  Note: Exact class '{sev_class}' not in kernel, using approximate match")
            available = list(fire_kernel.sev_classes.keys())
            # Map to closest available class
            if fire.severity < 0.20:
                kernel_class = next((k for k in available if 'low' in k or '10_20' in k), available[0])
            elif fire.severity < 0.50:
                kernel_class = next((k for k in available if 'moderate' in k or '20_50' in k), available[1] if len(available) > 1 else available[0])
            elif fire.severity < 0.80:
                kernel_class = next((k for k in available if 'severe' in k or '50_80' in k), available[-2] if len(available) > 1 else available[-1])
            else:
                kernel_class = next((k for k in available if 'catastrophic' in k or '80_100' in k), available[-1])
        
        if kernel_class:
            all_losses = fire_kernel.get_all_losses(kernel_class)
            print(f"\n  Immediate Loss Distributions (using '{kernel_class}'):")
            
            for metric, dist in all_losses.items():
                print(f"\n    {metric.replace('_', ' ').title()}:")
                print(f"      Min: {dist[0]:.1%}, Q1: {dist[1]:.1%}, Median: {dist[2]:.1%}, Q3: {dist[3]:.1%}, Max: {dist[4]:.1%}")
        
        # Get ADSR envelope
        envelope_class = None
        if sev_class in fire_envelopes.envelopes:
            envelope_class = sev_class
        else:
            # Try approximate match
            available = list(fire_envelopes.envelopes.keys())
            if fire.severity < 0.25:
                envelope_class = next((k for k in available if 'low' in k), available[0])
            elif fire.severity < 0.50:
                envelope_class = next((k for k in available if 'moderate' in k), available[1] if len(available) > 1 else available[0])
            elif fire.severity < 0.80:
                envelope_class = next((k for k in available if 'high' in k), available[-2] if len(available) > 1 else available[-1])
            else:
                envelope_class = next((k for k in available if 'extreme' in k), available[-1])
        
        if envelope_class:
            envelope = fire_envelopes.get_envelope(envelope_class)
            print(f"\n  ADSR Envelope for BA Growth Increments (using '{envelope_class}'):")
            print(f"    Attack drop:     {envelope.attack_drop:.1%}")
            print(f"    Attack duration: {envelope.attack_duration_years} years")
            print(f"    Decay period:    {envelope.decay_years} years")
            print(f"    Sustain level:   {envelope.sustain_level:.2f}x")
            print(f"    Release period:  {envelope.release_years} years")
            
            # Show multiplier trajectory
            print(f"\n  BA Growth Increment Multipliers Over Time:")
            print(f"    (Applied to per-year delta BA, not total BA)")
            total_years = (
                envelope.attack_duration_years +
                envelope.decay_years +
                min(envelope.sustain_years or 5, 5) +
                envelope.release_years
            )
            for year in range(min(total_years + 1, 10)):
                if year < envelope.attack_duration_years:
                    mult = 1.0 - envelope.attack_drop
                elif year < envelope.attack_duration_years + envelope.decay_years:
                    t = (year - envelope.attack_duration_years) / envelope.decay_years
                    attack_val = 1.0 - envelope.attack_drop
                    mult = attack_val + (envelope.sustain_level - attack_val) * t
                elif year < envelope.attack_duration_years + envelope.decay_years + (envelope.sustain_years or 5):
                    mult = envelope.sustain_level
                else:
                    years_in_release = year - envelope.attack_duration_years - envelope.decay_years - (envelope.sustain_years or 5)
                    if envelope.release_years > 0:
                        t = min(years_in_release / envelope.release_years, 1.0)
                        mult = envelope.sustain_level + (1.0 - envelope.sustain_level) * t
                    else:
                        mult = 1.0
                
                print(f"    Year {year}: {mult:.2f}x growth rate")
        
    except FileNotFoundError as e:
        print(f"\n⚠ Could not load files (data gitignored): {e}")
        print("  To run this demo, populate data/disturbances/ with YAML files")


def demo_severity_discretization():
    """Show how continuous severity maps to discrete classes."""
    print("\n" + "=" * 60)
    print("SEVERITY DISCRETIZATION")
    print("=" * 60)
    
    print("\nContinuous Severity → Discrete Classes:")
    print("-" * 40)
    
    test_severities = [0.10, 0.25, 0.35, 0.50, 0.65, 0.75, 0.90]
    
    for sev in test_severities:
        fire = FireDisturbance(age=10.0, severity=sev)
        wind = WindDisturbance(age=10.0, severity=sev)
        print(f"  {sev:.2f} → {fire.get_severity_class()}")


if __name__ == "__main__":
    demo_basic_disturbances()
    demo_with_kernel_and_envelope()
    demo_severity_discretization()
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
