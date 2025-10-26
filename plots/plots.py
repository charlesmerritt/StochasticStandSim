"""Matplotlib plots for kernels, envelopes, and growth trajectories."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple, List, Literal

import yaml
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

# Growth model imports from your codebase
from core.growth import Stand, StandConfig, StandState, ThinEvent
from core.pmrc_model import PMRCModel
from core.disturbances import make_fire_event, make_wind_event, get_severity
from core.env import StandMgmtEnv, EnvConfig

ACTION_LABELS = [
    "noop",
    "thin_40",
    "thin_60",
    "fert_n200_p1",
    "harvest",
    "plant_600_si60",
    "salvage_0p3",
    "rxfire",
]

ACTION_LABELS = [
    "noop",
    "thin_40",
    "thin_60",
    "fert_n200_p1",
    "harvest",
    "plant_600_si60",
    "salvage_0p3",
    "rxfire",
]

# -------------------------- small helpers --------------------------

def _is_iter(x: Any) -> bool:
    return isinstance(x, Iterable) and not isinstance(x, (str, bytes))

def _five(vals: Sequence[float]) -> Tuple[float, float, float, float, float]:
    if len(vals) != 5:
        raise ValueError("Expected five-number summary [min, q1, median, q3, max].")
    lo, q1, med, q3, hi = map(float, vals)
    # monotone sanitize
    q1 = max(lo, q1)
    med = max(q1, med)
    q3 = max(med, q3)
    hi = max(q3, hi)
    return lo, q1, med, q3, hi

def _palette(n: int) -> List[str]:
    base = ["#2ecc71", "#3498db", "#9b59b6", "#e67e22", "#e74c3c", "#16a085", "#8e44ad", "#d35400", "#c0392b"]
    return [base[i % len(base)] for i in range(n)]

def _sev_label(k: str) -> str:
    # "moderate_20_50" -> "Moderate 20-50%"
    parts = k.split("_")
    if len(parts) >= 3 and parts[-2].isdigit() and parts[-1].isdigit():
        head = " ".join(parts[:-2]).title()
        return f"{head} {parts[-2]}-{parts[-1]}%"
    return k.replace("_", " ").title()

def adsr_envelope(cfg: Mapping[str, Any], total_years: int, metric: str, additive: bool) -> List[float]:
    # New envelopes format with attack/decay/sustain/release blocks of [min, max]
    def mid(key: str) -> float:
        rng = cfg.get(key, {}).get(metric, [0.0, 0.0])
        if _is_iter(rng) and len(rng) >= 2:
            return float(rng[0] + rng[1]) / 2.0
        return float(rng)
    attack_drop = mid("attack")
    decay_drop = mid("decay")
    sustain_drop = mid("sustain")
    release_drop = mid("release")
    if additive:
        a_val = 1.0 + attack_drop
        d_val = 1.0 + decay_drop
        s_val = 1.0 + sustain_drop
        r_val = 1.0 + release_drop
    else:
        a_val = 1.0 - attack_drop
        d_val = 1.0 - decay_drop
        s_val = 1.0 - sustain_drop
        r_val = 1.0 - release_drop
    dur = int(cfg.get("duration", total_years))
    atk = max(1, dur // 6)
    dec = max(2, dur // 4)
    rel = max(1, dur // 6)
    sus = max(0, dur - atk - dec - rel)
    y: List[float] = []
    y += [a_val] * atk
    if dec:
        for i in range(1, dec + 1):
            t = i / dec
            y.append(a_val + (d_val - a_val) * t)
    y += [s_val] * sus
    if rel:
        for i in range(1, rel + 1):
            t = i / rel
            y.append(s_val + (1.0 - s_val) * t)
    if len(y) < total_years:
        y += [y[-1] if y else 1.0] * (total_years - len(y))
    return y[:total_years]


# -------------------------- 1) kernel boxplots --------------------------

def plot_kernel_boxplots(*, kernel_paths: Sequence[str | Path]) -> plt.Figure:
    """
    Box and whisker plots per severity per metric for each kernel file.
    Assumes YAML format:
      sev_classes:
        <class>:
          immediate_loss_range:
            basal_area: [min,q1,med,q3,max]
            height:     [min,q1,med,q3,max]
            volume:     [min,q1,med,q3,max]
    Values are fractional losses in [0,1].
    """
    kernel_paths = [Path(p) for p in kernel_paths]
    if not kernel_paths:
        raise ValueError("Provide at least one kernel_path.")

    # Load one kernel to define metrics
    with kernel_paths[0].open("r", encoding="utf-8") as h:
        first = yaml.safe_load(h)
    sev0 = first["sev_classes"]
    metrics = sorted(next(iter(sev0.values()))["immediate_loss_range"].keys())

    fig, axes = plt.subplots(len(kernel_paths), len(metrics), figsize=(4 * len(metrics), 3.5 * len(kernel_paths)), squeeze=False)
    for row, path in enumerate(kernel_paths):
        with path.open("r", encoding="utf-8") as h:
            data = yaml.safe_load(h)
        sev = data["sev_classes"]
        classes = list(sev.keys())
        colors = _palette(len(classes))
        y_positions = list(range(len(classes)))
        for col, metric in enumerate(metrics):
            ax = axes[row][col]
            for i, name in enumerate(classes):
                values = sev[name]["immediate_loss_range"][metric]
                lo, q1, med, q3, hi = _five(values)
                y = y_positions[i]
                color = colors[i]
                ax.hlines(y, lo, hi, color=color, linewidth=1.2)
                ax.add_patch(Rectangle((q1, y - 0.3), max(q3 - q1, 0.0), 0.6, facecolor=color, alpha=0.35, edgecolor="black", linewidth=1.0))
                ax.vlines(med, y - 0.3, y + 0.3, color=color, linewidth=1.4)
                ax.scatter([lo, hi], [y, y], color=color, s=15, zorder=3)
            ax.set_title(f"{metric.replace('_', ' ').title()} loss")
            ax.set_xlabel("Fraction")
            ax.set_xlim(0, 1)
            ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.5)
            if col == 0:
                ax.set_yticks(y_positions)
                ax.set_yticklabels([_sev_label(c) for c in classes])
            else:
                ax.set_yticks([])
        axes[row][0].set_ylabel(path.stem.replace("_", " ").title())
    fig.suptitle("Disturbance Kernels")
    fig.tight_layout()
    return fig


# -------------------------- 2) envelope shaded lines --------------------------

def plot_disturbance_envelope(
    envelope_path: str | Path,
    *,
    envelope_key: str | Sequence[str] | None = None,
    dpi: int = 300,
    invert: bool = False,  # show 1−p when True for subtractive envelopes
) -> plt.Figure:
    path = Path(envelope_path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, Mapping):
        raise ValueError(f"Envelope definition {path} is empty or invalid")

    metadata = data.get("metadata", {})
    is_additive = metadata.get("effect_direction") == "positive"

    # Gather envelopes (severity classes)
    envelopes: Mapping[str, Mapping[str, Any]] = (
        data.get("sev_classes")
        or data.get("envelopes")
        or data.get("envelopes_by_class")
        or {}
    )
    if not envelopes:
        raise ValueError(f"No envelopes found in {path}")

    # Optional filtering by key(s)
    if envelope_key:
        keys = [envelope_key] if isinstance(envelope_key, str) else list(envelope_key)
        envelopes = {k: envelopes[k] for k in keys if k in envelopes}
        if not envelopes:
            raise ValueError("Requested envelope_key(s) not found in file")

    # Plot each metric in its own row
    metrics = ["basal_area", "height", "volume"]
    n_rows = len(metrics)
    fig, axes = plt.subplots(n_rows, 1, figsize=(7, 2.8 * n_rows), dpi=dpi, sharex=True)
    if n_rows == 1:
        axes = [axes]

    colors = _palette(len(envelopes))

    for m_idx, metric in enumerate(metrics):
        ax = axes[m_idx]
        for idx, (key, cfg) in enumerate(envelopes.items()):
            duration = int(cfg.get("duration", 10))
            series = adsr_envelope(cfg, duration, metric, is_additive)  # midpoints timeline
            # optional inversion for subtractive envelopes only
            plot_vals = series if (is_additive or not invert) else [1.0 - v for v in series]
            years = list(range(len(plot_vals)))  # left-aligned at t=0
            color = colors[idx]
            ax.plot(years, plot_vals, color=color, linewidth=1.8, label=key.replace("_", " ").title())
            ax.fill_between(years, plot_vals, alpha=0.15, color=color)

        ax.set_title(metric.replace("_", " ").title(), loc="left", fontsize=11, fontweight="bold")
        ax.set_ylabel("Multiplier")
        # Tight to y-axis: remove left padding
        ax.set_xlim(left=0)
        ax.margins(x=0)  # no x padding
        ax.grid(True, linestyle="--", alpha=0.4)
        if m_idx == n_rows - 1:
            ax.set_xlabel("Years After Disturbance")

    axes[0].legend(fontsize=8, loc="upper right", frameon=False)
    fig.suptitle(f"{metadata.get('disturbance', 'Disturbance').title()} Envelope", x=0.01, ha="left", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig





# -------------------------- 3) growth trajectory with PCT --------------------------

def plot_growth_example(*, years: float = 40.0, dt: float = 1.0) -> plt.Figure:
    """
    Ordinary growth with si25=60, initial_tpa=600, and a pre-commercial thinning at age 10.
    PCT approximated as a residual BA target at age 10. Adjust as needed.
    """
    model = PMRCModel(region="ucp")
    # Resolve initial HD and BA from SI25 and TPA
    si25 = 60.0
    age0 = 1.0
    tpa0 = 600.0
    hd0 = model.hd_from_si(si25, form="projection")
    ba0 = model.ba_predict(age=age0, tpa=tpa0, hd=hd0, region="ucp")

    stand = Stand(
        init=StandState(age=age0, tpa=tpa0, si25=si25, hd=hd0, ba=ba0),
        cfg=StandConfig(region="ucp", tpa_geometric_decay=0.99),
    )

    #stand.add_thin_to_residual_ba(age=10.0, residual_ba=max(10.0, 0.8 * ba0))

    target_age = age0 + years
    ages, tpas, hds, bas, vols = [stand.state.age], [stand.state.tpa], [stand.state.hd], [stand.state.ba], [0.0]
    
    # Step until we reach target age
    while stand.state.age < target_age:
        remaining = target_age - stand.state.age
        step_size = min(dt, remaining)
        if step_size <= 0:
            break
        s = stand.step(step_size)
        ages.append(s.age)
        tpas.append(s.tpa)
        hds.append(s.hd)
        bas.append(s.ba)
        vols.append(s.tvob)

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(8, 10))
    axes[0].plot(ages, tpas, marker="o", markersize=3, label="TPA")
    axes[0].set_ylabel("TPA")
    axes[1].plot(ages, hds, marker="o", markersize=3, color="#1f77b4", label="HD")
    axes[1].set_ylabel("HD (ft)")
    axes[2].plot(ages, bas, marker="o", markersize=3, color="#e67e22", label="BA")
    axes[2].set_ylabel("BA (ft²/ac)")
    axes[3].plot(ages, vols, marker="o", markersize=3, color="#2c3e50", label="TVOB")
    axes[3].set_ylabel("TVOB (ft³/ac)")
    axes[3].set_xlabel("Age (years)")
    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.4)
    # Mark PCT
    # for ax in axes:
    #     ax.axvline(10.0, color="green", linestyle=":", alpha=0.4)
    fig.suptitle("PMRC growth")
    fig.tight_layout()
    return fig

def plot_disturbance_comparison(
    *,
    disturbance: Literal["fire", "wind"] = "fire",
    start_age: float = 15.0,
    severity: float | None = None,   # if None, draw via seeded RNG
    seed: int = 123,
    years: float = 25.0,
    dt: float = 1.0,
    si25: float = 60.0,
    tpa0: float = 600.0,
    age0: float = 1.0,
    region: str = "ucp",
) -> plt.Figure:
    """Same stand twice: with and without a disturbance. Shade the difference."""
    model = PMRCModel(region=region)
    hd0 = model.hd_from_si(si25, form="projection")
    ba0 = model.ba_predict(age=age0, tpa=tpa0, hd=hd0, region=region)

    base_cfg = StandConfig(region=region, tpa_geometric_decay=0.99)
    s_base = Stand(StandState(age=age0, tpa=tpa0, si25=si25, hd=hd0, ba=ba0), base_cfg)
    s_dist = Stand(StandState(age=age0, tpa=tpa0, si25=si25, hd=hd0, ba=ba0), base_cfg)

    sev = float(severity) if severity is not None else get_severity(seed=seed)
    event = make_fire_event(start_age=start_age, severity=sev, seed=seed) if disturbance == "fire" \
            else make_wind_event(start_age=start_age, severity=sev, seed=seed)
    s_dist.add_disturbance(event)

    steps = int(round(years / dt))
    def simulate(st: Stand):
        ages, tpa, hd, ba, vol = [st.state.age], [st.state.tpa], [st.state.hd], [st.state.ba], [st.state.tvob]
        for _ in range(steps):
            ns = st.step(min(dt, years - (ages[-1] - age0)))
            ages.append(ns.age); tpa.append(ns.tpa); hd.append(ns.hd); ba.append(ns.ba); vol.append(ns.tvob)
        return ages, tpa, hd, ba, vol

    a0, t0, h0, b0, v0 = simulate(s_base)
    a1, t1, h1, b1, v1 = simulate(s_dist)

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(8, 10), dpi=300)
    series = [
        ("TPA", t0, t1, None),
        ("Dominant height (ft)", h0, h1, "#1f77b4"),
        ("Basal area (ft²/ac)", b0, b1, "#e67e22"),
        ("TVOB (ft³/ac)", v0, v1, "#2c3e50"),
    ]
    for ax, (ylab, y_base, y_dist, color) in zip(axes, series):
        ax.plot(a0, y_base, label="Undisturbed", linestyle="--", alpha=0.7)
        ax.plot(a1, y_dist, label=f"{disturbance.title()} (seed={seed}, sev={sev:.2f})", color=color)
        # shaded difference (absolute band)
        y_lo = [min(u, d) for u, d in zip(y_base, y_dist)]
        y_hi = [max(u, d) for u, d in zip(y_base, y_dist)]
        ax.fill_between(a0, y_lo, y_hi, color=color if color else "#555555", alpha=0.15, linewidth=0)
        ax.set_ylabel(ylab)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].legend()
    axes[-1].set_xlabel("Age (years)")
    fig.suptitle(f"Disturbance comparison – {disturbance.title()} at age {start_age}", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_policy_rollout(
    *,
    model_path: str | Path,
    steps: int = 200,
    deterministic: bool = True,
    env_overrides: Mapping[str, Any] | None = None,
) -> plt.Figure:
    """Visualise a learned policy's rollout on :class:`StandMgmtEnv`.

    Args:
        model_path: Path to a Stable Baselines3 policy (e.g. PPO ``.zip`` file).
        steps: Maximum number of environment steps to simulate.
        deterministic: Whether to request deterministic actions from the policy.
        env_overrides: Optional overrides for :class:`EnvConfig` parameters.

    Returns:
        Matplotlib figure summarising state trajectories, rewards, and actions.
    """

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Stable Baselines3 is required for policy rollout plots.") from exc

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    overrides = dict(env_overrides or {})
    env_cfg = EnvConfig(**overrides)
    env = StandMgmtEnv(env_cfg)

    try:
        model = PPO.load(str(model_path), env=env)
    except Exception as exc:  # pragma: no cover - SB3 loading errors
        env.close()
        raise RuntimeError(f"Failed to load model from {model_path}: {exc}") from exc

    obs, _ = env.reset()
    if env.stand is None:
        env.close()
        raise RuntimeError("Environment failed to initialise stand state.")

    segments: list[dict[str, list[float]]] = [
        {
            "ages": [float(env.stand.state.age)],
            "tpa": [float(env.stand.state.tpa)],
            "ba": [float(env.stand.state.ba)],
            "tvob": [float(env.stand.state.tvob)],
            "rewards": [],
            "cashflows": [],
            "actions": [],
        }
    ]

    for _ in range(max(1, int(steps))):
        action_raw, _ = model.predict(obs, deterministic=deterministic)
        action_id = int(np.asarray(action_raw).item())
        obs_next, reward, done, _, info = env.step(action_id)

        next_age = float(env.stand.state.age)
        next_tpa = float(env.stand.state.tpa)
        next_ba = float(env.stand.state.ba)
        next_vol = float(env.stand.state.tvob)

        if next_age + 1e-6 < segments[-1]["ages"][-1]:
            segments.append(
                {
                    "ages": [next_age],
                    "tpa": [next_tpa],
                    "ba": [next_ba],
                    "tvob": [next_vol],
                    "rewards": [],
                    "cashflows": [],
                    "actions": [],
                }
            )
        seg = segments[-1]
        seg["ages"].append(next_age)
        seg["tpa"].append(next_tpa)
        seg["ba"].append(next_ba)
        seg["tvob"].append(next_vol)
        seg["rewards"].append(float(reward))
        seg["cashflows"].append(float(info.get("cashflow_now") or 0.0))
        seg["actions"].append(action_id)

        obs = obs_next
        if done:
            break

    env.close()

    if not any(seg["actions"] for seg in segments):
        raise RuntimeError("Rollout produced no transitions; check the model and environment configuration.")

    return _render_rollout(segments, title_prefix=f"Policy rollout – {model_path.name}")


def _render_rollout(
    segments: Sequence[dict[str, list[float]]],
    *,
    title_prefix: str,
) -> plt.Figure:
    if not segments:
        raise ValueError("No rollout segments to plot.")
    segment = segments[0]

    ages_arr = np.asarray(segment["ages"], dtype=float)
    tpa_arr = np.asarray(segment["tpa"], dtype=float)
    ba_arr = np.asarray(segment["ba"], dtype=float)
    vol_arr = np.asarray(segment["tvob"], dtype=float)
    rewards_arr = np.asarray(segment["rewards"], dtype=float)
    cash_arr = np.asarray(segment["cashflows"], dtype=float)
    actions = segment["actions"]
    step_ages = ages_arr[1:] if ages_arr.size > 1 else np.array([], dtype=float)

    total_steps = len(actions)
    total_reward = float(rewards_arr.sum())
    total_cash = float(cash_arr.sum())

    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=False)
    action_colors = _palette(len(ACTION_LABELS))

    def _set_limits(ax, data):
        if data.size == 0:
            return
        pad = max(1e-6, 0.05 * (data.max() - data.min() if data.max() != data.min() else data.max() or 1.0))
        ax.set_ylim(data.min() - pad, data.max() + pad)

    axes[0].plot(ages_arr, tpa_arr, color="#2ecc71", linewidth=2.0)
    axes[0].set_ylabel("TPA")
    _set_limits(axes[0], tpa_arr)

    axes[1].plot(ages_arr, ba_arr, color="#e67e22", linewidth=2.0)
    axes[1].set_ylabel("Basal area\n(ft²/ac)")
    _set_limits(axes[1], ba_arr)

    axes[2].plot(ages_arr, vol_arr, color="#34495e", linewidth=2.0)
    axes[2].set_ylabel("TVOB\n(ft³/ac)")
    _set_limits(axes[2], vol_arr)

    axes[3].set_ylabel("Reward / Cashflow")
    if step_ages.size:
        axes[3].plot(step_ages, rewards_arr, color="#3498db", linewidth=1.3, label="Reward")
        cumulative_rewards = np.cumsum(rewards_arr)
        axes[3].plot(step_ages, cumulative_rewards, color="#1abc9c", linestyle="--", linewidth=1.2, label="Cumulative reward")
        if np.any(cash_arr):
            axes[3].plot(step_ages, cash_arr, color="#e74c3c", linestyle=":", linewidth=1.1, label="Cashflow")
        _set_limits(axes[3], rewards_arr if np.any(rewards_arr) else cumulative_rewards)
        axes[3].legend(loc="upper left", frameon=False)

    axes[4].set_ylabel("Action")
    axes[4].set_xlabel("Age (years)")
    axes[4].set_yticks(range(len(ACTION_LABELS)))
    axes[4].set_yticklabels(ACTION_LABELS)
    if step_ages.size:
        axes[4].plot(step_ages, actions, color="#8e44ad", linewidth=1.0)
        for age, act in zip(step_ages, actions):
            axes[4].scatter(age, act, color=action_colors[act % len(action_colors)], s=45, zorder=3)
        axes[4].set_ylim(-0.5, len(ACTION_LABELS) - 0.5)

    axes[4].set_ylabel("Action")
    axes[4].set_xlabel("Age (years)")
    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.4)

    summary = f"Steps: {total_steps} | Total reward: {total_reward:.1f} | Total cashflow: {total_cash:.1f}"
    fig.suptitle(f"{title_prefix}\n{summary}", fontsize=13)
    fig.tight_layout()
    return fig


def plot_fixed_policy(
    *,
    steps: int = 200,
    rotation_years: float = 25.0,
    thin_years: Sequence[float] = (10.0,),
    fert_years: Sequence[float] = (12.0, 20.0),
    env_overrides: Mapping[str, Any] | None = None,
) -> plt.Figure:
    overrides = dict(env_overrides or {})
    env_cfg = EnvConfig(**overrides)
    env = StandMgmtEnv(env_cfg)

    obs, _ = env.reset()
    if env.stand is None:
        env.close()
        raise RuntimeError("Environment failed to initialise stand state.")

    rotation_years = float(rotation_years)
    thin_schedule = sorted(float(v) for v in thin_years)
    fert_schedule = sorted(float(v) for v in fert_years)
    thin_done = {year: False for year in thin_schedule}
    fert_done = {year: False for year in fert_schedule}
    rotation_base = float(env.stand.state.age)

    segments: list[dict[str, list[float]]] = [
        {
            "ages": [float(env.stand.state.age)],
            "tpa": [float(env.stand.state.tpa)],
            "ba": [float(env.stand.state.ba)],
            "tvob": [float(env.stand.state.tvob)],
            "rewards": [],
            "cashflows": [],
            "actions": [],
        }
    ]

    for _ in range(max(1, int(steps))):
        age = float(env.stand.state.age)
        offset = age - rotation_base

        if env.manager.harvested:
            action_id = 5
        elif offset >= rotation_years:
            action_id = 4
        else:
            action_id = 0
            for idx, year in enumerate(thin_schedule):
                if not thin_done[year] and offset >= year:
                    action_id = 1 if idx == 0 else 2
                    thin_done[year] = True
                    break
            if action_id == 0:
                for year in fert_schedule:
                    if not fert_done[year] and offset >= year:
                        action_id = 3
                        fert_done[year] = True
                        break

        obs_next, reward, done, _, info = env.step(action_id)

        next_age = float(env.stand.state.age)
        next_tpa = float(env.stand.state.tpa)
        next_ba = float(env.stand.state.ba)
        next_vol = float(env.stand.state.tvob)
        if next_age + 1e-6 < segments[-1]["ages"][-1]:
            segments.append(
                {
                    "ages": [next_age],
                    "tpa": [next_tpa],
                    "ba": [next_ba],
                    "tvob": [next_vol],
                    "rewards": [],
                    "cashflows": [],
                    "actions": [],
                }
            )
        seg = segments[-1]
        seg["ages"].append(next_age)
        seg["tpa"].append(next_tpa)
        seg["ba"].append(next_ba)
        seg["tvob"].append(next_vol)
        seg["rewards"].append(float(reward))
        seg["cashflows"].append(float(info.get("cashflow_now") or 0.0))
        seg["actions"].append(action_id)

        if action_id == 5 and info.get("plant_ok"):
            rotation_base = float(env.stand.state.age)
            thin_done = {year: False for year in thin_schedule}
            fert_done = {year: False for year in fert_schedule}

        obs = obs_next
        if done:
            break

    env.close()

    if not any(seg["actions"] for seg in segments):
        raise RuntimeError("Fixed policy produced no transitions; check configuration.")
    return _render_rollout(segments, title_prefix="Fixed policy rollout")

# -------------------------- unified interface --------------------------

def plot_interface(**kwargs: Any) -> plt.Figure:
    """
    Dispatcher:
      plot='kernel'         + kernel_paths=[...]
      plot='envelope'       + envelope_path=...
      plot='growth'         + years, dt
      plot='compare'        + disturbance comparison kwargs
      plot='policy_rollout' + model_path, steps, env_overrides
    """
    kind = kwargs.pop("plot", None)
    if kind == "kernel":
        paths = kwargs.get("kernel_paths") or ([kwargs["kernel_path"]] if "kernel_path" in kwargs else None)
        if not paths:
            raise ValueError("kernel_paths or kernel_path required")
        return plot_kernel_boxplots(kernel_paths=paths)
    if kind == "envelope":
        # pass envelope_path and invert directly
        return plot_disturbance_envelope(
            envelope_path=kwargs["envelope_path"],
            envelope_key=kwargs.get("envelope_key"),
            dpi=kwargs.get("dpi", 300),
            invert=kwargs.get("invert", False),
        )
    if kind == "growth":
        return plot_growth_example(years=kwargs.get("years", 40.0), dt=kwargs.get("dt", 1.0))
    if kind == "compare":
        return plot_disturbance_comparison(**kwargs)
    if kind == "policy_rollout":
        env_overrides = kwargs.pop("env_overrides", None)
        return plot_policy_rollout(env_overrides=env_overrides, **kwargs)
    if kind == "fixed_policy":
        env_overrides = kwargs.pop("env_overrides", None)
        return plot_fixed_policy(env_overrides=env_overrides, **kwargs)
    raise ValueError("Unknown plot type")
