"""Matplotlib plots returned as figures. No Streamlit calls.

growth curves, ADSR overlays, severity histograms, value traces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
import yaml

from dataclasses import replace

from core import growth


def _iterable_values(value: Any) -> list[float]:
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return [float(x) for x in value]
    return []


def _midpoint(value: Any, fallback: float = 0.0) -> float:
    items = _iterable_values(value)
    if items:
        return sum(items) / len(items)
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _select(value: Any, mode: str, fallback: float = 0.0) -> float:
    items = _iterable_values(value)
    if items:
        if mode == "min":
            return min(items)
        if mode == "max":
            return max(items)
        return sum(items) / len(items)
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _adsr_envelope(
    envelope_cfg: Mapping[str, Any],
    total_years: int,
    floor: float | None,
    ceiling: float | None,
    mode: str = "mid",
    metric: str = "basal_area",
    is_additive: bool = False,
) -> list[float]:
    """Extract ADSR envelope from [min, max] format."""
    
    # Try old ADSR format first for backward compatibility
    adsr = envelope_cfg.get("ADSR", {})
    if adsr:
        attack_drop = _select(adsr.get("attack_drop"), mode, 0.0)
        attack_years = int(max(_select(adsr.get("attack_duration_years", 1), mode, 1), 0))
        decay_years = int(max(_select(adsr.get("decay_years", 0), mode, 0), 0))
        sustain_level = _select(adsr.get("sustain_level"), mode, 1.0)
        release_years = int(max(_select(adsr.get("release_years", 0), mode, 0), 0))
        sustain_years_param = adsr.get("sustain_years")
        if sustain_years_param is not None:
            sustain_years = int(max(_select(sustain_years_param, mode, 0), 0))
        else:
            sustain_years = max(total_years - (attack_years + decay_years + release_years), 0)
        
        attack_value = 1.0 - attack_drop
        sustain_value = sustain_level
    else:
        # New format: attack/decay/sustain/release phases with [min, max] per metric
        duration = envelope_cfg.get("duration", total_years)
        
        # Extract [min, max] for each phase
        attack_range = envelope_cfg.get("attack", {}).get(metric, [0.0, 0.0])
        decay_range = envelope_cfg.get("decay", {}).get(metric, [0.0, 0.0])
        sustain_range = envelope_cfg.get("sustain", {}).get(metric, [0.0, 0.0])
        release_range = envelope_cfg.get("release", {}).get(metric, [0.0, 0.0])
        
        # Take midpoint of ranges for plotting
        if isinstance(attack_range, (list, tuple)) and len(attack_range) >= 2:
            attack_drop = (attack_range[0] + attack_range[1]) / 2
        else:
            attack_drop = 0.0
            
        if isinstance(decay_range, (list, tuple)) and len(decay_range) >= 2:
            decay_drop = (decay_range[0] + decay_range[1]) / 2
        else:
            decay_drop = attack_drop
            
        if isinstance(sustain_range, (list, tuple)) and len(sustain_range) >= 2:
            sustain_drop = (sustain_range[0] + sustain_range[1]) / 2
        else:
            sustain_drop = 0.0
            
        if isinstance(release_range, (list, tuple)) and len(release_range) >= 2:
            release_drop = (release_range[0] + release_range[1]) / 2
        else:
            release_drop = 0.0
        
        # Convert to multiplier values
        # For positive effect_direction (thinning): add (e.g., 0.10 -> 1.10 = 10% boost)
        # For negative effect_direction (fire/wind): subtract (e.g., 0.10 -> 0.90 = 10% reduction)
        if is_additive:
            attack_value = 1.0 + attack_drop
            decay_value = 1.0 + decay_drop
            sustain_value = 1.0 + sustain_drop
            release_value = 1.0 + release_drop
        else:
            attack_value = 1.0 - attack_drop
            decay_value = 1.0 - decay_drop
            sustain_value = 1.0 - sustain_drop
            release_value = 1.0 - release_drop
        
        # Estimate phase durations based on total duration
        attack_years = max(1, duration // 6)
        decay_years = max(2, duration // 4)
        release_years = max(1, duration // 6)
        sustain_years = max(0, duration - attack_years - decay_years - release_years)

    series: list[float] = []

    # Attack phase
    for _ in range(attack_years):
        series.append(attack_value)

    # Decay phase
    if decay_years:
        for step in range(1, decay_years + 1):
            t = step / decay_years
            if 'decay_value' in locals():
                series.append(attack_value + (decay_value - attack_value) * t)
            else:
                series.append(attack_value + (sustain_value - attack_value) * t)

    # Sustain phase
    for _ in range(sustain_years):
        series.append(sustain_value)

    # Release phase - return to baseline (1.0 = no effect)
    if release_years:
        for step in range(1, release_years + 1):
            t = step / release_years
            series.append(sustain_value + (1.0 - sustain_value) * t)

    if len(series) < total_years:
        series.extend([series[-1] if series else 1.0] * (total_years - len(series)))
    elif len(series) > total_years:
        series = series[:total_years]

    if floor is not None:
        series = [max(floor, v) for v in series]
    if ceiling is not None:
        series = [min(ceiling, v) for v in series]

    return series


def plot_disturbance_envelope(
    envelope_path: str | Path,
    *,
    envelope_key: str | Sequence[str] | None = None,
    metric: str = "basal_area",
) -> plt.Figure:
    """
    Plot the ADSR-style disturbance envelope defined in the provided YAML file.

    Args:
        envelope_path: Path to the envelope YAML definition.
        envelope_key: Optional specific envelope key (e.g., scorch class) to draw.

    Returns:
        Matplotlib Figure containing the envelope curves.
    """

    path = Path(envelope_path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, Mapping):
        raise ValueError(f"Envelope definition {path} is empty or invalid")

    defaults = data.get("defaults", {})
    floor = defaults.get("floor")
    floor = float(floor) if floor is not None else None
    ceiling = defaults.get("ceiling")
    ceiling = float(ceiling) if ceiling is not None else None

    metric_block = data.get("metrics") or data.get("envelopes_by_metric")
    if metric_block:
        if metric not in metric_block:
            available = ", ".join(sorted(metric_block))
            raise ValueError(f"Metric '{metric}' not found. Available: {available}")
        envelopes: Mapping[str, Mapping[str, Any]] = metric_block[metric]
    else:
        envelopes = (
            data.get("envelopes_by_class")
            or data.get("envelopes_by_scorch_class")
            or data.get("sev_classes")  # Also check for sev_classes
            or data.get("envelopes")
            or {}
        )
    if not envelopes:
        raise ValueError(f"No envelopes found in {path}")

    selected_keys: list[str]
    if envelope_key:
        if isinstance(envelope_key, str):
            selected_keys = [
                part.strip()
                for chunk in envelope_key.split(":")
                for part in chunk.split(",")
                if part.strip()
            ]
        else:
            selected_keys = [str(k).strip() for k in envelope_key if str(k).strip()]
        filtered = {k: envelopes[k] for k in selected_keys if k in envelopes}
        if not filtered:
            raise ValueError(f"Requested classes {selected_keys} not found in envelope definition")
        envelopes = filtered
    else:
        # Plot ALL severity classes
        selected_keys = list(envelopes.keys())
        envelopes = {k: envelopes[k] for k in selected_keys}

    fig, ax = plt.subplots()
    metadata = data.get("metadata", {})
    
    # Use same color palette as kernel plots
    base_palette = [
        "#2ecc71",  # green
        "#3498db",  # blue
        "#9b59b6",  # purple
        "#e67e22",  # orange
        "#e74c3c",  # red
    ]
    colors = [base_palette[idx % len(base_palette)] for idx in range(len(envelopes))]
    
    # Check if this is an additive envelope (thinning)
    is_additive = metadata.get("effect_direction") == "positive"

    for idx, (key, cfg) in enumerate(envelopes.items()):
        cap_year = cfg.get("cap_after_year") or cfg.get("duration") or defaults.get("cap_after_year")
        if cap_year is None:
            # Fallback calculation for old format
            adsr = cfg.get("ADSR", {})
            cap_year = (
                (adsr.get("attack_duration_years") or 1)
                + (adsr.get("decay_years") or 0)
                + (adsr.get("release_years") or 0)
                + 2
            )
        cap_year = int(cap_year)

        series_mid = _adsr_envelope(cfg, cap_year, floor, ceiling, mode="mid", metric=metric, is_additive=is_additive)
        years = list(range(len(series_mid) + 1))
        
        # For additive envelopes, show as positive growth increase
        # For subtractive envelopes, show as reduction (1 - multiplier)
        if is_additive:
            effects = [0.0] + [max(0.0, value - 1.0) for value in series_mid]  # Show increase
        else:
            effects = [0.0] + [max(0.0, 1.0 - value) for value in series_mid]  # Show reduction

        # Format label with percentage range
        label = key.replace("_", " ").title()
        # Extract percentage range from key (e.g., "moderate_20_50" -> "Moderate 20-50%")
        import re
        match = re.search(r'(\d+)_(\d+)$', key)
        if match:
            low, high = match.groups()
            # For thinning (additive), just show percentage range
            # For fire/wind (subtractive), show severity class + percentage
            if is_additive:
                label = f"{low}-{high}%"
            else:
                # Get the base name without numbers
                base_name = re.sub(r'_\d+_\d+$', '', key).replace("_", " ").title()
                label = f"{base_name} {low}-{high}%"
        
        color = colors[idx]
        line, = ax.plot(years, effects, marker="o", label=label, color=color)
        ax.fill_between(years, effects, alpha=0.2, color=color)

    ax.set_xlabel("Years After Disturbance")
    
    # Set appropriate ylabel and title based on envelope type
    if is_additive:
        ax.set_ylabel("Growth Enhancement (multiplier - 1)")
        title = f"{metadata.get('disturbance', 'Disturbance').title()} Envelope (Additive)"
    else:
        ax.set_ylabel("Growth Reduction (1 - multiplier)")
        title = f"{metadata.get('disturbance', 'Disturbance').title()} Envelope (Subtractive)"
    
    if metric:
        title += f" – {metric.replace('_', ' ').title()}"
    ax.set_title(title)
    ax.set_ylim(0, 1)  # Always maintain 0 to 1 scale
    ax.legend()
    fig.tight_layout()

    return fig


def _collect_metrics(classes: Mapping[str, Mapping[str, Any]]) -> list[str]:
    metrics: set[str] = set()
    for cfg in classes.values():
        range_map = cfg.get("immediate_loss_range", {})
        metrics.update(range_map.keys())
    return sorted(metrics)


def _format_severity_label(name: str) -> str:
    parts = name.split("_")
    if len(parts) >= 3 and all(part.replace(".", "", 1).isdigit() for part in parts[1:]):
        head = parts[0].replace("-", " ").title()
        range_part = "-".join(parts[1:])
        return f"{head} {range_part}%"
    return name.replace("_", " ").title()


def plot_disturbance_kernel(kernel_path: str | Path) -> plt.Figure:
    """
    Visualise five-number loss summaries per severity class from a disturbance kernel.

    Each metric is rendered with whiskers (min/max), an interquartile box (q1–q3),
    and a median bar so classes remain visually distinct. Modulators defined in
    the YAML are ignored because they depend on scenario-specific context.
    """

    path = Path(kernel_path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, Mapping):
        raise ValueError(f"Kernel definition {path} is empty or invalid")

    sev_classes: Mapping[str, Mapping[str, Any]] = data.get("sev_classes", {})
    if not sev_classes:
        raise ValueError(f"No severity classes found in {path}")

    class_items = list(sev_classes.items())
    metrics = _collect_metrics(sev_classes)
    if not metrics:
        raise ValueError(f"No immediate_loss_range metrics in {path}")

    fig, axes = plt.subplots(1, len(metrics), sharey=True, figsize=(4 * len(metrics), 3.5))
    if isinstance(axes, Axes):
        axes = [axes]
    elif isinstance(axes, Iterable):
        axes = list(axes)
    else:
        axes = [axes]

    class_labels = [_format_severity_label(name) for name, _ in class_items]
    y_positions = list(range(len(class_items)))

    base_palette = [
        "#2ecc71",  # green
        "#3498db",  # blue
        "#9b59b6",  # purple
        "#e67e22",  # orange
        "#e74c3c",  # red
    ]
    palette = [base_palette[idx % len(base_palette)] for idx in range(len(class_items))]
    box_height = 0.6
    half_height = box_height / 2

    for ax, metric in zip(axes, metrics):
        for idx, (name, cfg) in enumerate(class_items):
            range_map = cfg.get("immediate_loss_range", {})
            value = range_map.get(metric)
            if value is None:
                continue
            if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
                values = [float(x) for x in value]
            else:
                values = [float(value)]

            if len(values) != 5:
                raise ValueError(
                    f"Expected five-number loss distribution for '{metric}' in severity '{name}'"
                )

            low, q1, median, q3, high = values
            low = max(0.0, low)
            q1 = max(low, q1)
            median = max(q1, median)
            q3 = max(median, q3)
            high = max(q3, high)

            y = y_positions[idx]
            color = palette[idx]

            ax.hlines(y, low, high, color=color, linewidth=1.2)
            ax.add_patch(
                Rectangle(
                    (q1, y - half_height),
                    max(q3 - q1, 0.0),
                    box_height,
                    facecolor=color,
                    alpha=0.35,
                    edgecolor="black",
                    linewidth=1.2,
                )
            )
            ax.vlines(median, y - half_height, y + half_height, color=color, linewidth=1.4)
            ax.scatter([low, high], [y, y], color=color, s=15, zorder=3)

        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("Fractional Loss")
        ax.set_xlim(0, 1)
        ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.5)

    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(class_labels)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Severity Class")

    metadata = data.get("metadata", {})
    title = f"{metadata.get('disturbance', 'Disturbance').title()} Kernel Loss Ranges"
    fig.suptitle(title)
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)

    return fig


def plot_growth_trajectory(
    stand_name: str = "ucp_baseline",
    *,
    years: float = 10.0,
    dt: float = 1.0,
    compare_unthinned: bool = False,
) -> plt.Figure:
    """
    Run the PMRC growth model trajectory and plot key stand variables over time.
    """

    params = growth.EXAMPLE_STANDS.get(stand_name)
    if params is None:
        available = ", ".join(sorted(growth.EXAMPLE_STANDS))
        raise ValueError(f"Unknown stand '{stand_name}'. Available stands: {available}")

    steps = int(round(years / dt))
    initial_state = params.to_state()

    ages = [initial_state.age]
    tpas = [initial_state.tpa]
    hds = [initial_state.hd if initial_state.hd is not None else initial_state.resolved_hd()]
    basals = [
        initial_state.ba
        if initial_state.ba is not None
        else growth.ba_predict(initial_state.age, initial_state.tpa, hds[-1], initial_state.region)
    ]
    vols = [initial_state.vol_ob if initial_state.vol_ob is not None else 0.0]
    unth_tpas = [initial_state.tpa_unthinned if initial_state.tpa_unthinned is not None else initial_state.tpa]
    unth_basals = [initial_state.ba_unthinned if initial_state.ba_unthinned is not None else basals[-1]]
    unth_vols = [initial_state.vol_ob_unthinned if initial_state.vol_ob_unthinned is not None else vols[-1]]
    event_records: list[dict] = []
    state = initial_state
    for _ in range(steps):
        next_state, ba_val, events, tpa_unth, ba_unth, vol_unth = growth.step_with_log(state, dt=dt)
        ages.append(next_state.age)
        tpas.append(next_state.tpa)
        hds.append(next_state.hd if next_state.hd is not None else next_state.resolved_hd())
        basals.append(next_state.ba if next_state.ba is not None else ba_val)
        vols.append(next_state.vol_ob if next_state.vol_ob is not None else vols[-1])
        unth_tpas.append(tpa_unth if tpa_unth is not None else unth_tpas[-1])
        unth_basals.append(ba_unth if ba_unth is not None else unth_basals[-1])
        unth_vols.append(vol_unth if vol_unth is not None else unth_vols[-1])
        event_records.extend(events)
        state = next_state

    # Always compare with undisturbed trajectory
    compare_state = None
    compare_meta = None
    if True:  # Always enabled
        compare_params = replace(params, disturbances=())
        compare_state = compare_params.to_state()
        comp_state = compare_state
        comp_ages = [comp_state.age]
        comp_tpas = [comp_state.tpa]
        comp_hds = [comp_state.hd if comp_state.hd is not None else comp_state.resolved_hd()]
        comp_basals = [
            comp_state.ba
            if comp_state.ba is not None
            else growth.ba_predict(comp_state.age, comp_state.tpa, comp_hds[-1], comp_state.region)
        ]
        comp_vols = [comp_state.vol_ob if comp_state.vol_ob is not None else 0.0]
        for _ in range(steps):
            comp_state, comp_ba = growth.step(comp_state, dt=dt)
            comp_ages.append(comp_state.age)
            comp_tpas.append(comp_state.tpa)
            comp_hds.append(comp_state.hd if comp_state.hd is not None else comp_state.resolved_hd())
            if comp_ba is not None:
                comp_basals.append(comp_ba)
            elif comp_state.ba is not None:
                comp_basals.append(comp_state.ba)
            else:
                comp_basals.append(
                    growth.ba_predict(comp_state.age, comp_state.tpa, comp_hds[-1], comp_state.region)
                )
            comp_vols.append(comp_state.vol_ob if comp_state.vol_ob is not None else comp_vols[-1])
        compare_meta = (comp_ages, comp_tpas, comp_hds, comp_basals, comp_vols)

    # Extract comparison data
    comp_ages, comp_tpas, comp_hds, comp_basals, comp_vols = compare_meta

    # Simple linear interpolation - no complex stair-stepping
    tpa_plot_ages, tpa_plot_values = ages, tpas
    ba_plot_ages, ba_plot_values = ages, basals
    vol_plot_ages, vol_plot_values = ages, vols

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(8, 12))
    
    # Plot TPA
    axes[0].plot(tpa_plot_ages, tpa_plot_values, marker="o", markersize=3, label="Managed")
    axes[0].plot(comp_ages, comp_tpas, marker="o", markersize=3, linestyle="--", alpha=0.6, label="Undisturbed")
    axes[0].set_ylabel("Trees per Acre")
    axes[0].set_title(f"Growth Trajectory: {stand_name}")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()
    
    # Plot HD
    axes[1].plot(ages, hds, marker="o", markersize=3, color="#1f77b4", label="Managed")
    axes[1].plot(comp_ages, comp_hds, marker="o", markersize=3, linestyle="--", color="#1f77b4", alpha=0.6, label="Undisturbed")
    axes[1].set_ylabel("Dominant Height (ft)")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()
    
    # Plot BA
    axes[2].plot(ba_plot_ages, ba_plot_values, marker="o", markersize=3, color="#e67e22", label="Managed")
    axes[2].plot(comp_ages, comp_basals, marker="o", markersize=3, linestyle="--", color="#e67e22", alpha=0.6, label="Undisturbed")
    axes[2].set_ylabel("Basal Area (ft²/ac)")
    axes[2].grid(True, linestyle="--", alpha=0.4)
    axes[2].legend()
    
    # Plot Volume
    axes[3].plot(vol_plot_ages, vol_plot_values, marker="s", markersize=3, color="#2c3e50", label="Managed")
    axes[3].plot(comp_ages, comp_vols, marker="s", markersize=3, linestyle="--", color="#2c3e50", alpha=0.6, label="Undisturbed")
    axes[3].set_ylabel("Volume OB (ft³/ac)")
    axes[3].set_xlabel("Age (years)")
    axes[3].grid(True, linestyle="--", alpha=0.4)
    axes[3].legend()
    
    # Add disturbance markers and collect legend info
    disturbance_colors = {"thinning": "green", "fire": "red", "wind": "purple"}
    disturbance_markers = {"thinning": "^", "fire": "*", "wind": "D"}
    disturbance_labels = {"thinning": "Thinning", "fire": "Fire", "wind": "Wind"}
    
    # Track which disturbance types we've seen for the legend
    seen_disturbances = set()
    
    for ev in event_records:
        dist_type = ev.get("type", "thinning")
        seen_disturbances.add(dist_type)
        color = disturbance_colors.get(dist_type, "gray")
        marker = disturbance_markers.get(dist_type, "o")
        age = ev["age"]
        
        for ax in axes:
            ymin, ymax = ax.get_ylim()
            ax.axvline(x=age, color=color, linestyle=":", alpha=0.3, linewidth=1)
            ax.plot(age, ymax * 0.95, marker=marker, color=color, markersize=8, 
                   markeredgecolor='black', markeredgewidth=0.5, zorder=10)
    
    # Add disturbance markers to legend on first subplot
    if seen_disturbances:
        from matplotlib.lines import Line2D
        handles, labels = axes[0].get_legend_handles_labels()
        
        # Add disturbance marker handles
        for dist_type in sorted(seen_disturbances):
            color = disturbance_colors.get(dist_type, "gray")
            marker = disturbance_markers.get(dist_type, "o")
            label = disturbance_labels.get(dist_type, dist_type.title())
            handles.append(Line2D([0], [0], marker=marker, color='w', 
                                markerfacecolor=color, markeredgecolor='black',
                                markersize=8, label=label))
            labels.append(label)
        
        axes[0].legend(handles, labels)

    fig.tight_layout()
    return fig

def plot_interface(**kwargs: Any) -> plt.Figure:
    plot_name = kwargs.pop("plot", None)

    if plot_name in {"disturbance_envelope", "envelope"}:
        return plot_disturbance_envelope(**kwargs)
    if plot_name in {"disturbance_kernel", "kernel"}:
        return plot_disturbance_kernel(**kwargs)
    if plot_name in {"growth", "growth_trajectory"}:
        return plot_growth_trajectory(**kwargs)

    raise ValueError(f"Unknown plot: {plot_name}")
