"""Compare growth trajectories for multiple stands under different management regimes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from core.growth import Stand, StandConfig, StandState


YEARS = 30.0
DT = 1.0
THIN_AGE = 10.0
THIN_RESIDUAL_FRACTION = 0.6
FERT_AGE = 5.0
OUTPUT_PATH = Path("plots") / "stand_management_comparison.png"


@dataclass(frozen=True)
class StandTemplate:
    name: str
    init: StandState
    cfg: StandConfig


def make_stands() -> Sequence[StandTemplate]:
    """Three diverse stand templates to stress the growth model."""
    return [
        StandTemplate(
            name="Dense / high site",
            init=StandState(age=1.0, tpa=750.0, si25=70.0, hd=0.0, ba=0.0),
            cfg=StandConfig(region="ucp", tpa_geometric_decay=0.99),
        ),
        StandTemplate(
            name="Moderate / mid site",
            init=StandState(age=1.0, tpa=520.0, si25=60.0, hd=0.0, ba=0.0),
            cfg=StandConfig(region="pucp", tpa_geometric_decay=0.995),
        ),
        StandTemplate(
            name="Open / low site",
            init=StandState(age=1.0, tpa=320.0, si25=54.0, hd=0.0, ba=0.0),
            cfg=StandConfig(region="ucp", tpa_geometric_decay=0.998),
        ),
    ]


ScenarioHook = Callable[[Stand], None]


def baseline(_: Stand) -> None:
    """No action."""


def _clone_for_projection(stand: Stand) -> Stand:
    """Standalone copy used to estimate BA at the thin age."""
    init = StandState(
        age=stand.state.age,
        tpa=stand.state.tpa,
        si25=stand.state.si25,
        hd=stand.state.hd,
        ba=stand.state.ba,
        tvob=stand.state.tvob,
        ci=stand.state.ci,
    )
    return Stand(init=init, cfg=stand.cfg)


def schedule_absolute_thin(stand: Stand, *, age: float, residual_fraction: float) -> None:
    preview = _clone_for_projection(stand)
    preview.run_to(age, dt=0.5)
    target_ba = max(preview.state.ba, 1e-3)
    residual_ba = max(residual_fraction * target_ba, 0.1)
    stand.add_thin_to_residual_ba(
        age=age,
        residual_ba=residual_ba,
        residual_fraction=residual_fraction,
    )


def add_thinning(stand: Stand) -> None:
    """Thin once to a residual fraction of projected BA."""
    schedule_absolute_thin(
        stand,
        age=THIN_AGE,
        residual_fraction=THIN_RESIDUAL_FRACTION,
    )


def add_fertilization(stand: Stand) -> None:
    stand.add_fertilization(age=FERT_AGE, N=200.0, P=1.0)


def add_thin_plus_fert(stand: Stand) -> None:
    add_thinning(stand)
    add_fertilization(stand)


SCENARIOS: Dict[str, ScenarioHook] = {
    "baseline": baseline,
    "thin": add_thinning,
    "fert": add_fertilization,
    "thin+fert": add_thin_plus_fert,
}


def simulate(stand: Stand, *, years: float, dt: float) -> Dict[str, List[float]]:
    """Run the stand forward, collecting key metrics."""
    age_offset = stand.state.age
    target_age = stand.state.age + years
    history: Dict[str, List[float]] = {
        "age": [stand.state.age - age_offset],
        "tpa": [stand.state.tpa],
        "hd": [stand.state.hd],
        "ba": [stand.state.ba],
        "tvob": [stand.state.tvob],
    }
    start_int = int(stand.state.age)
    end_int = int(target_age)
    for age in range(start_int + 1, end_int + 1):
        stand.run_to(float(age), dt=dt)
        state = stand.state
        history["age"].append(state.age - age_offset)
        history["tpa"].append(state.tpa)
        history["hd"].append(state.hd)
        history["ba"].append(state.ba)
        history["tvob"].append(state.tvob)
    return history


def main() -> None:
    templates = make_stands()
    stand_labels = {tpl.name: f"{tpl.name} (SI25={tpl.init.si25:.0f})" for tpl in templates}
    stand_palette = ["#1abc9c", "#9b59b6", "#e67e22", "#2c3e50"]
    stand_colors = {tpl.name: stand_palette[i % len(stand_palette)] for i, tpl in enumerate(templates)}
    metric_labels = {
        "tpa": "TPA",
        "hd": "Dominant height (ft)",
        "ba": "Basal area (ft²/ac)",
        "tvob": "TVOB (ft³/ac)",
    }
    metrics = list(metric_labels.keys())

    # Run every scenario for every stand independently.
    results: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for template in templates:
        stand_results: Dict[str, Dict[str, List[float]]] = {}
        for scenario_name, hook in SCENARIOS.items():
            stand = Stand(init=template.init, cfg=template.cfg)
            hook(stand)
            offset = stand.state.age
            thin_events = [ev.age - offset for ev in getattr(stand, "_thin", [])]
            fert_events = [ev.age - offset for ev in getattr(stand, "_fert", [])]
            history = simulate(stand, years=YEARS, dt=DT)
            history["thin_events"] = thin_events
            history["fert_events"] = fert_events
            stand_results[scenario_name] = history
        results[template.name] = stand_results

    n_rows = len(templates) + 1  # extra row for baseline
    n_cols = len(metrics)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 2.8 * n_rows),
        sharex="col",
        dpi=200,
    )
    if n_rows == 1:
        axes = [axes]  # type: ignore[assignment]

    colors = {
        "baseline": "#7f8c8d",
        "thin": "#2980b9",
        "fert": "#8d6e63",
        "thin+fert": "#c0392b",
    }

    for row in range(n_rows):
        if row == 0:
            row_label = "Baseline comparison"
            scenarios = ["baseline"]
            plotted_templates = templates
        else:
            template = templates[row - 1]
            row_label = stand_labels[template.name]
            scenarios = list(SCENARIOS.keys())
            plotted_templates = [template]
        for col, metric in enumerate(metrics):
            ax = axes[row][col]
            for template in plotted_templates:
                for scenario_name in scenarios:
                    history = results[template.name][scenario_name]
                    if row == 0:
                        label = stand_labels[template.name]
                        color = stand_colors[template.name]
                    else:
                        label = scenario_name
                        color = colors.get(scenario_name, "#555555")
                    ax.plot(
                        history["age"],
                        history[metric],
                        label=label,
                        color=color,
                        linewidth=1.8,
                    )
                    for age in history.get("thin_events", []):
                        ax.axvline(age, color=colors.get(scenario_name), linestyle=":", linewidth=1.0, alpha=0.4)
                    for age in history.get("fert_events", []):
                        ax.axvline(age, color=colors.get(scenario_name), linestyle="--", linewidth=1.0, alpha=0.4)
            if row == 0:
                ax.set_title(metric_labels[metric])
            if col == 0:
                ax.set_ylabel(row_label)
            ax.grid(True, linestyle="--", alpha=0.3)
    for col in range(n_cols):
        axes[-1][col].set_xlabel("Age (years)")
    stand_handles = [
        Line2D([0], [0], color=stand_colors[tpl.name], linewidth=2.0, label=stand_labels[tpl.name])
        for tpl in templates
    ]
    scenario_handles = [
        Line2D([0], [0], color=colors[name], linewidth=2.0, label=name)
        for name in SCENARIOS.keys()
    ]
    event_handles = [
        Line2D([0], [0], color="black", linestyle=":", linewidth=1.0, label="Thin timing"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.0, label="Fert timing"),
    ]
    legend_handles = stand_handles + scenario_handles + event_handles
    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle("Stand growth comparison across management regimes", y=0.99)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    print(f"Saved plot to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
