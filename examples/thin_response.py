"""Visualize basal-area response to thinning, inspired by PMRC Figure 36."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple
from dataclasses import replace

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from core.growth import Stand, StandConfig, StandState


def _build_stand(*, tpa: float, si25: float, age: float) -> Stand:
    init = StandState(age=age, tpa=tpa, si25=si25, hd=0.0, ba=0.0)
    cfg = StandConfig(region="ucp", tpa_geometric_decay=0.995)
    return Stand(init=init, cfg=cfg)


def _simulate_baseline(stand: Stand, end_age: float, dt: float = 0.25) -> Tuple[List[float], List[float]]:
    ages = [stand.state.age]
    bas = [stand.state.ba]
    while stand.state.age < end_age:
        step = min(dt, end_age - stand.state.age)
        s = stand.step(step)
        ages.append(s.age)
        bas.append(s.ba)
    return ages, bas


def main() -> None:
    start_age = 5.0
    end_age = 35.0
    si25 = 60.0

    stand_unthin_mid = _build_stand(tpa=380.0, si25=si25, age=start_age)
    mid_ref = Stand(replace(stand_unthin_mid.state), stand_unthin_mid.cfg)
    mid_ref.run_to(15.0)
    unthin_mid_ages, unthin_mid_ba = _simulate_baseline(stand_unthin_mid, end_age)

    stand_unthin_low = _build_stand(tpa=300.0, si25=si25, age=start_age)
    preview_low = Stand(replace(stand_unthin_low.state), stand_unthin_low.cfg)
    preview_low.run_to(15.0)
    low_ages, low_ba = _simulate_baseline(stand_unthin_low, end_age)

    stand_thin = _build_stand(tpa=700.0, si25=si25, age=start_age)
    preview = Stand(replace(stand_thin.state), stand_thin.cfg)
    preview.run_to(15.0)
    target_ba = preview_low.state.ba
    residual_ba = min(target_ba, preview.state.ba)
    stand_thin.add_thin_to_residual_ba(age=15.0, residual_ba=residual_ba)
    thin_ages, thin_ba = _simulate_baseline(stand_thin, end_age)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)
    ax.plot(unthin_mid_ages, unthin_mid_ba, label="Unthinned 380 TPA @ age 5", color="#2c3e50", linewidth=2.2)
    ax.plot(thin_ages, thin_ba, label="Thinned: high-density -> match 300-TPA BA", color="#c0392b", linewidth=2.2)
    ax.plot(low_ages, low_ba, label="Unthinned 300 TPA @ age 5", color="#27ae60", linestyle="--", linewidth=2.0)

    # highlight BA gap between thinned stand and low-density unthinned baseline
    ages_after, thin_after, low_after = [], [], []
    for a, ba_thin, ba_low in zip(thin_ages, thin_ba, low_ba):
        if a >= 15.0 and len(ages_after) < len(low_ba):
            ages_after.append(a)
            thin_after.append(ba_thin)
            low_after.append(ba_low)
    if ages_after:
        ax.fill_between(ages_after, thin_after, low_after, color="#bdc3c7", alpha=0.5)

    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Basal area (ft²/ac)")
    ax.set_xlim(start_age, end_age)
    ax.set_ylim(0, max(unthin_mid_ba) * 1.05)
    ax.set_title("Basal area response to thinning (300-TPA reference baseline)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    output = Path("plots") / "thin_response.png"
    output.parent.mkdir(exist_ok=True, parents=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    print(f"Saved plot to {output.resolve()}")


if __name__ == "__main__":
    main()
