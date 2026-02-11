from __future__ import annotations

from dataclasses import replace
from typing import Dict, Callable, Any

import matplotlib.pyplot as plt

from core.growth import Stand, StandConfig, StandState


# Lightweight plotting utilities used by the main CLI. If you add new plot types,
# register them in _PLOT_FACTORY.


def _stand_templates() -> Dict[str, tuple[StandState, StandConfig]]:
    """Curated starting points for growth plots."""
    return {
        "ucp_baseline": (
            StandState(age=1.0, tpa=520.0, si25=60.0, hd=0.0, ba=0.0),
            StandConfig(region="ucp", tpa_geometric_decay=0.995),
        ),
        "dense_high_site": (
            StandState(age=1.0, tpa=750.0, si25=70.0, hd=0.0, ba=0.0),
            StandConfig(region="ucp", tpa_geometric_decay=0.99),
        ),
        "open_low_site": (
            StandState(age=1.0, tpa=320.0, si25=54.0, hd=0.0, ba=0.0),
            StandConfig(region="ucp", tpa_geometric_decay=0.998),
        ),
    }


def _clone_stand(init: StandState, cfg: StandConfig) -> Stand:
    """Return a fresh Stand so templates are not mutated."""
    return Stand(init=replace(init), cfg=replace(cfg))


def _simulate_growth(stand: Stand, *, years: float, dt: float) -> dict[str, list[float]]:
    start_age = stand.state.age
    target_age = start_age + years
    history = {
        "age": [start_age],
        "hd": [stand.state.hd],
        "ba": [stand.state.ba],
        "tpa": [stand.state.tpa],
    }
    while stand.state.age < target_age - 1e-9:
        step = min(dt, target_age - stand.state.age)
        stand.step(step)
        history["age"].append(stand.state.age)
        history["hd"].append(stand.state.hd)
        history["ba"].append(stand.state.ba)
        history["tpa"].append(stand.state.tpa)
    return history


def plot_growth(
    *,
    stand_name: str = "ucp_baseline",
    years: float = 10.0,
    dt: float = 1.0,
    compare_unthinned: bool = False,
) -> plt.Figure:
    templates = _stand_templates()
    if stand_name not in templates:
        available = ", ".join(sorted(templates))
        raise ValueError(f"Unknown stand '{stand_name}'. Available: {available}")

    init, cfg = templates[stand_name]
    base = _clone_stand(init, cfg)
    baseline_history = _simulate_growth(base, years=years, dt=dt)

    compare_history = None
    if compare_unthinned:
        cfg_unthinned = replace(cfg, tpa_geometric_decay=1.0, hold_tpa_below_asymptote=False)
        compare_history = _simulate_growth(_clone_stand(init, cfg_unthinned), years=years, dt=dt)

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, dpi=150)
    metrics = [("hd", "Dominant height (ft)", "#2c3e50"), ("ba", "Basal area (ft²/ac)", "#c0392b"), ("tpa", "Trees per acre", "#16a085")]
    ages0 = baseline_history["age"][0]
    for ax, (key, label, color) in zip(axes, metrics):
        ax.plot(
            [a - ages0 for a in baseline_history["age"]],
            baseline_history[key],
            color=color,
            linewidth=2.0,
            label="Managed",
        )
        if compare_history:
            ax.plot(
                [a - compare_history["age"][0] for a in compare_history["age"]],
                compare_history[key],
                color=color,
                linestyle="--",
                linewidth=1.5,
                alpha=0.8,
                label="Unthinned",
            )
        ax.set_ylabel(label)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best")

    axes[-1].set_xlabel("Age (years)")
    fig.suptitle(f"Stand growth: {stand_name}", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def plot_placeholder(title: str, *, subtitle: str | None = None) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(8, 3), dpi=150)
    ax.axis("off")
    ax.text(
        0.5,
        0.6,
        title,
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )
    if subtitle:
        ax.text(0.5, 0.35, subtitle, ha="center", va="center", fontsize=10)
    fig.tight_layout()
    return fig


def plot_envelope(**_: Any) -> plt.Figure:
    return plot_placeholder(
        "Envelope plots not implemented",
        subtitle="Provide a renderer for envelope YAMLs in plots/plots.py",
    )


def plot_kernel(**_: Any) -> plt.Figure:
    return plot_placeholder(
        "Kernel plots not implemented",
        subtitle="Provide a renderer for disturbance kernels in plots/plots.py",
    )


PLOT_FACTORY: Dict[str, Callable[..., plt.Figure]] = {
    "growth": plot_growth,
    "envelope": plot_envelope,
    "kernel": plot_kernel,
}


def plot_interface(*, plot: str, **kwargs: Any):
    """Dispatcher used by main.py to construct a matplotlib Figure."""
    if plot not in PLOT_FACTORY:
        available = ", ".join(sorted(PLOT_FACTORY))
        raise ValueError(f"Unsupported plot '{plot}'. Available: {available}")
    return PLOT_FACTORY[plot](**kwargs)
