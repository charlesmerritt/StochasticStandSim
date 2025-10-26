"""Interactive debugger. Only calls env, rollout, evaluation, and plots.

Controls: scenario selector, seed, risk toggles, disturbance rate sliders, alpha for CVaR, baseline pickers, RL checkpoint loader.

Views: per-step table, state panel, reward breakdown, plots."""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:  # Stable Baselines is optional; the UI gracefully degrades if missing.
    from stable_baselines3 import PPO
except Exception:  # pragma: no cover - optional dependency for the demo app.
    PPO = None

from core.disturbance.stochastic import (
    DisturbanceSetting,
    apply_stochastic_disturbances,
    initialise_disturbance_status,
    load_disturbance_catalog,
)
from core.stand_env import StandConfig, StandEnv


# --------------------------------------------------------------------------- UI

STATE_SERIES = [
    ("age", "Age (years)"),
    ("biomass", "Biomass (tons/ac)"),
    ("tpa", "Trees per acre"),
    ("basal_area", "Basal area (ft²/ac)"),
    ("risk", "Disturbance risk"),
    ("value", "Stand value ($/ac)"),
]


DISTURBANCE_TEMPLATES: Dict[str, DisturbanceSetting] = load_disturbance_catalog()


# ---------------------------------------------------------------------- Helpers


def _initial_product_mix() -> Dict[str, float]:
    return {"Pulpwood": 0.0, "Chip-n-saw": 0.0, "Sawtimber": 0.0}


def _derive_product_mix(state: Dict[str, float], acreage: float) -> Dict[str, float]:
    """Heuristic breakdown of products based on stand structure."""

    mix = _initial_product_mix()
    biomass = max(state.get("biomass", 0.0), 0.0)
    basal_area = max(state.get("basal_area", 0.0), 0.0)
    tpa = max(state.get("tpa", 0.0), 1e-6)

    # Approximate quadratic mean diameter in inches.
    avg_dbh = np.sqrt(max(basal_area, 1e-6) / (0.005454 * tpa)) * 12.0  # ft -> inches

    # Use smooth transitions to partition biomass among products.
    pulp_share = float(1.0 / (1.0 + np.exp((avg_dbh - 6.5) / 0.7)))
    saw_share = float(1.0 / (1.0 + np.exp(-(avg_dbh - 11.5) / 0.8)))
    chip_share = float(max(0.0, 1.0 - pulp_share - saw_share))

    total_biomass = biomass * max(acreage, 1.0)
    mix["Pulpwood"] = total_biomass * pulp_share
    mix["Sawtimber"] = total_biomass * saw_share
    mix["Chip-n-saw"] = total_biomass * chip_share
    return mix


def _state_dict_from_env(env: StandEnv) -> Dict[str, float]:
    state = env.state
    return {key: float(state.get(key, 0.0)) for key, _ in STATE_SERIES}


def _encode_state(env: StandEnv, state: Dict[str, float]) -> np.ndarray:
    env._state = dict(state)
    return env._encode_state(env._state)


def _format_state_table(state: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for key, label in STATE_SERIES:
        value = state.get(key, np.nan)
        rows.append((label, value))
    df = pd.DataFrame(rows, columns=["Metric", "Value"])
    return df


def _probability_slider(label: str, default: float, *, key: Optional[str] = None) -> float:
    return st.slider(
        label,
        min_value=0.0,
        max_value=1.0,
        value=float(default),
        step=0.01,
        key=key,
    )


def _severity_slider(
    label: str, default_min: float, default_max: float, *, key: Optional[str] = None
) -> Tuple[float, float]:
    return st.slider(
        label,
        min_value=0.0,
        max_value=1.0,
        value=(float(default_min), float(default_max)),
        step=0.01,
        key=key,
    )


def _build_disturbance_settings() -> Dict[str, DisturbanceSetting]:
    settings: Dict[str, DisturbanceSetting] = {}
    st.sidebar.header("Disturbance Configuration")
    for key, template in DISTURBANCE_TEMPLATES.items():
        with st.sidebar.expander(f"{template.label} settings", expanded=False):
            enabled = st.checkbox(
                f"Enable {template.label}", value=template.enabled, key=f"toggle_{key}"
            )
            prob = _probability_slider(
                "Annual probability", template.probability, key=f"{key}_annual_probability"
            )
            sev_min, sev_max = _severity_slider(
                "Severity range",
                template.severity_min,
                template.severity_max,
                key=f"{key}_severity_range",
            )
            boost = st.slider(
                "Post-event probability boost",
                min_value=0.0,
                max_value=1.0,
                value=float(template.envelope_boost),
                step=0.01,
                key=f"{key}_probability_boost",
            )
            years = st.slider(
                "Duration of elevated risk (years)",
                min_value=0,
                max_value=25,
                value=int(template.envelope_years),
                key=f"{key}_envelope_years",
            )

        settings[key] = DisturbanceSetting(
            enabled=enabled,
            probability=prob,
            severity_min=sev_min,
            severity_max=sev_max,
            envelope_boost=boost,
            envelope_years=years,
            emoji=template.emoji,
            label=template.label,
            effects=dict(template.effects),
            catastrophic_threshold=template.catastrophic_threshold,
            risk_increment=template.risk_increment,
            salvage_recovery_rate=template.salvage_recovery_rate,
        )
    return settings


def _initialise_disturbance_status(settings: Dict[str, DisturbanceSetting]) -> Dict[str, Dict[str, float]]:
    return initialise_disturbance_status(settings)


def _apply_envelope_decay(status: Dict[str, Dict[str, float]], settings: Dict[str, DisturbanceSetting]) -> None:
    for key, state in status.items():
        cfg = settings[key]
        if state["boost_years"] > 0:
            state["boost_years"] -= 1
            if state["boost_years"] == 0:
                state["current_prob"] = cfg.probability if cfg.enabled else 0.0


def _apply_disturbances(
    env: StandEnv,
    settings: Dict[str, DisturbanceSetting],
    status: Dict[str, Dict[str, float]],
    rng: np.random.Generator,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    result = apply_stochastic_disturbances(env.state, rng, status, settings)
    current_state = result.get("state", env.state)
    info = result.get("info", {})
    events = info.get("events", [])

    _encode_state(env, current_state)
    return current_state, events


def _apply_treatments(
    env: StandEnv,
    treatments: Dict[str, bool],
    disturbance_status: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    state = dict(env.state)

    if treatments.get("prescribed_fire"):
        state["risk"] = max(0.0, state.get("risk", 0.0) - 0.1)
        fire_status = disturbance_status.get("fire")
        if fire_status is not None:
            fire_status["current_prob"] = max(0.0, fire_status["current_prob"] - 0.1)
        state["biomass"] = max(0.0, state.get("biomass", 0.0) * 0.98)

    if treatments.get("pesticide"):
        insect = disturbance_status.get("insect")
        if insect:
            insect["current_prob"] = max(0.0, insect["current_prob"] - 0.15)
        state["risk"] = max(0.0, state.get("risk", 0.0) - 0.05)

    if treatments.get("fertiliser"):
        # Fertiliser effect is primarily handled via the action vector, but we
        # allow a small immediate bump to reflect operational assumptions.
        state["biomass"] = state.get("biomass", 0.0) * 1.01
        state["basal_area"] = state.get("basal_area", 0.0) * 1.005

    _encode_state(env, state)
    return state


def _plot_state_history(history: List[Dict[str, float]], events: List[Dict[str, float]]) -> None:
    if not history:
        st.info("Run at least one step to view the trajectory.")
        return

    df = pd.DataFrame(history)
    df.index.name = "Step"
    fig = go.Figure()
    for key, label in STATE_SERIES:
        if key not in df.columns:
            continue
        fig.add_trace(
            go.Scatter(x=df.index, y=df[key], mode="lines", name=label)
        )

    if events:
        xs, ys, texts = [], [], []
        for event in events:
            step = event.get("step")
            if step is None or step >= len(df):
                continue
            xs.append(step)
            ys.append(df.loc[step, "basal_area"] if "basal_area" in df.columns else df.loc[step].max())
            texts.append(event.get("emoji", "❗"))
        if xs:
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="text",
                    text=texts,
                    textposition="top center",
                    showlegend=False,
                )
            )

    fig.update_layout(
        height=450,
        margin=dict(l=40, r=20, t=30, b=40),
        template="plotly_white",
        xaxis_title="Step",
        yaxis_title="Value",
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_product_history(product_history: List[Dict[str, float]]) -> None:
    if not product_history:
        return

    df = pd.DataFrame(product_history)
    df.index.name = "Step"
    fig = go.Figure()
    for column in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[column],
                mode="lines",
                stackgroup="one",
                name=column,
            )
        )
    fig.update_layout(
        height=350,
        title="Product distribution (tons across acreage)",
        xaxis_title="Step",
        yaxis_title="Estimated volume",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------------- UI Helpers


def _render_initialisation_controls() -> Tuple[Dict[str, float], float, int, Dict[str, DisturbanceSetting]]:
    st.sidebar.header("Stand initialisation")
    init_mode = st.sidebar.radio(
        "Starting state",
        options=("Plant new stand", "Load stand table"),
    )

    acreage = float(st.sidebar.number_input("Managed acreage", min_value=1.0, value=50.0))
    horizon = int(
        st.sidebar.number_input("Time horizon (years)", min_value=1, max_value=500, value=120)
    )

    if init_mode == "Plant new stand":
        planting_age = float(st.sidebar.number_input("Initial age", min_value=0.0, value=1.0))
        tpa = float(
            st.sidebar.number_input("Planting density (trees/ac)", min_value=50.0, value=600.0)
        )
        basal_area = float(
            st.sidebar.number_input("Initial basal area (ft²/ac)", min_value=1.0, value=20.0)
        )
        biomass = float(
            st.sidebar.number_input("Initial biomass (tons/ac)", min_value=0.1, value=12.0)
        )
        site_index = float(
            st.sidebar.number_input("Site index", min_value=50.0, max_value=200.0, value=120.0)
        )
        state = {
            "age": planting_age,
            "biomass": biomass,
            "tpa": tpa,
            "basal_area": basal_area,
            "risk": 0.01,
            "value": 0.0,
            "site_index": site_index,
        }
    else:
        uploaded = st.sidebar.file_uploader("Stand table CSV", type=["csv"])
        if uploaded is None:
            st.sidebar.warning("Upload a stand table to continue.")
            st.stop()

        with io.BytesIO(uploaded.read()) as buffer:
            df = pd.read_csv(buffer)
        st.sidebar.dataframe(df.head())
        if df.empty:
            st.sidebar.error("Stand table is empty")
            st.stop()

        row = int(
            st.sidebar.number_input(
                "Row to initialise from", min_value=0, max_value=len(df) - 1, value=0
            )
        )
        record = df.iloc[row].to_dict()
        state = {
            "age": float(record.get("age", 0.0)),
            "biomass": float(record.get("biomass", record.get("Biomass", 0.0))),
            "tpa": float(record.get("tpa", record.get("TPA", 0.0))),
            "basal_area": float(record.get("basal_area", record.get("BA", 0.0))),
            "risk": float(record.get("risk", record.get("Risk", 0.0))),
            "value": float(record.get("value", record.get("Value", 0.0))),
            "site_index": float(record.get("site_index", record.get("SiteIndex", 120.0))),
        }

    disturbance_settings = _build_disturbance_settings()
    return state, acreage, horizon, disturbance_settings


def _create_environment(
    state: Dict[str, float],
    horizon: int,
    disturbance_settings: Dict[str, DisturbanceSetting],
) -> StandEnv:
    total_prob = sum(cfg.probability for cfg in disturbance_settings.values() if cfg.enabled)
    growth_cfg = {
        "age_increment": 1.0,
        "biomass_growth": 2.5,
        "basal_area_growth": 1.8,
        "risk_increment": min(total_prob * 0.1, 0.05),
    }
    disturbance_cfg = {
        "catastrophe_threshold": 1.5,  # Disable deterministic catastrophic loss.
        "baseline_risk": 0.0,
        "catastrophe_severity": 0.0,
    }
    config = StandConfig(
        initial_age=state.get("age", 1.0),
        initial_tpa=state.get("tpa", 500.0),
        initial_basal_area=state.get("basal_area", 20.0),
        initial_biomass=state.get("biomass", 10.0),
        site_index=state.get("site_index", 120.0),
        risk_level=state.get("risk", 0.01),
        horizon=horizon,
        growth_config=growth_cfg,
        disturbance_config=disturbance_cfg,
    )
    env = StandEnv(config=config)
    obs, _ = env.reset()
    _encode_state(env, state)
    return env


def _ensure_session_defaults() -> None:
    defaults = {
        "env": None,
        "history": [],
        "product_history": [],
        "events": [],
        "step": 0,
        "done": False,
        "cumulative_reward": 0.0,
        "actions": [],
        "disturbance_status": {},
        "disturbance_settings": {},
        "acreage": 1.0,
        "rng": np.random.default_rng(),
        "ppo_model": None,
        "ppo_choice": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _reset_environment(
    initial_state: Dict[str, float],
    acreage: float,
    horizon: int,
    disturbance_settings: Dict[str, DisturbanceSetting],
) -> None:
    env = _create_environment(initial_state, horizon, disturbance_settings)
    st.session_state.env = env
    st.session_state.history = [_state_dict_from_env(env)]
    st.session_state.product_history = [_derive_product_mix(env.state, acreage)]
    st.session_state.events = []
    st.session_state.step = 0
    st.session_state.done = False
    st.session_state.cumulative_reward = 0.0
    st.session_state.actions = []
    st.session_state.acreage = acreage
    st.session_state.disturbance_settings = disturbance_settings
    st.session_state.disturbance_status = _initialise_disturbance_status(disturbance_settings)
    st.session_state.rng = np.random.default_rng()


def _load_available_models() -> List[str]:
    if PPO is None:
        return []
    models = st.sidebar.file_uploader(
        "Upload PPO policy (zip)",
        type=["zip"],
        accept_multiple_files=True,
        key="model_uploader",
    )
    names: List[str] = []
    if not models:
        return names

    st.session_state.setdefault("uploaded_models", {})
    for file in models:
        name = file.name
        if name in st.session_state.uploaded_models:
            names.append(name)
            continue
        bytes_data = file.read()
        st.session_state.uploaded_models[name] = io.BytesIO(bytes_data)
        names.append(name)
    return names


def _select_model(names: Iterable[str]) -> None:
    if PPO is None or not names:
        return

    choice = st.sidebar.selectbox("Select uploaded policy", options=list(names))
    if not choice:
        return

    if st.session_state.ppo_choice == choice:
        return

    buffer = st.session_state.uploaded_models.get(choice)
    if buffer is None:
        return

    buffer.seek(0)
    try:
        st.session_state.ppo_model = PPO.load(buffer, device="cpu")
        st.session_state.ppo_choice = choice
        st.sidebar.success(f"Loaded policy: {choice}")
    except Exception as exc:  # pragma: no cover - user feedback path.
        st.sidebar.error(f"Failed to load PPO model: {exc}")
        st.session_state.ppo_model = None
        st.session_state.ppo_choice = None


def _draw_action_controls() -> Tuple[np.ndarray, Dict[str, bool]]:
    st.sidebar.header("Management actions")
    thin_pct = st.sidebar.slider("Thin proportion", 0.0, 0.6, 0.0, step=0.01)
    fert_toggle = st.sidebar.checkbox("Apply fertiliser", value=False)
    fert_strength = st.sidebar.slider("Fertiliser intensity", 0.0, 1.0, 0.3, step=0.05)
    pesticide = st.sidebar.checkbox("Apply pesticide treatment", value=False)
    rx_fire = st.sidebar.checkbox("Apply prescribed fire", value=False)

    action = np.array(
        [
            thin_pct,
            fert_strength if fert_toggle else 0.0,
            fert_strength * 0.6 if fert_toggle else 0.0,
        ],
        dtype=np.float32,
    )

    treatments = {
        "fertiliser": fert_toggle,
        "pesticide": pesticide,
        "prescribed_fire": rx_fire,
    }
    return action, treatments


def _run_step(action: np.ndarray, treatments: Dict[str, bool], auto: bool = False) -> None:
    env: StandEnv = st.session_state.env
    obs, reward, terminated, truncated, info = env.step(action)
    st.session_state.cumulative_reward += reward

    # Update state history from environment.
    state = dict(info.get("state", env.state))
    _encode_state(env, state)

    # Apply user treatments and stochastic disturbances.
    state = _apply_treatments(env, treatments, st.session_state.disturbance_status)
    _apply_envelope_decay(st.session_state.disturbance_status, st.session_state.disturbance_settings)
    state, events = _apply_disturbances(
        env,
        st.session_state.disturbance_settings,
        st.session_state.disturbance_status,
        st.session_state.rng,
    )

    st.session_state.history.append(_state_dict_from_env(env))
    st.session_state.product_history.append(
        _derive_product_mix(state, st.session_state.acreage)
    )

    for event in events:
        event.update({"step": st.session_state.step + 1})
        st.session_state.events.append(event)

    st.session_state.actions.append(
        {
            "step": st.session_state.step + 1,
            "auto": auto,
            "action": action.tolist(),
            "reward": reward,
        }
    )

    st.session_state.step += 1
    st.session_state.done = terminated or truncated


# --------------------------------------------------------------------------- App


def main() -> None:
    st.set_page_config(layout="wide", page_title="Stochastic Stand Simulator")
    st.title("Stochastic Stand Simulator")
    st.caption("Manual control and policy playback for the StandEnv gymnasium environment.")

    _ensure_session_defaults()
    initial_state, acreage, horizon, disturbance_settings = _render_initialisation_controls()

    if st.sidebar.button("Initialise stand", type="primary"):
        _reset_environment(initial_state, acreage, horizon, disturbance_settings)
        st.experimental_rerun()

    if st.session_state.env is None:
        st.info("Configure the stand on the left and click *Initialise stand* to begin.")
        return

    # Allow reconfiguration mid-run.
    if st.sidebar.button("Reset simulation"):
        _reset_environment(initial_state, acreage, horizon, disturbance_settings)
        st.experimental_rerun()

    _select_model(_load_available_models())

    action, treatments = _draw_action_controls()

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.subheader("Simulation controls")
        c1, c2, c3 = st.columns(3)
        if c1.button("Step", disabled=st.session_state.done):
            _run_step(action, treatments, auto=False)
            st.experimental_rerun()

        auto_disabled = st.session_state.done or st.session_state.ppo_model is None
        if c2.button("Auto step", disabled=auto_disabled):
            model = st.session_state.ppo_model
            if model is None:
                st.warning("Load a PPO policy to use auto-stepping.")
            else:
                predicted_action, _ = model.predict(_encode_state(st.session_state.env, st.session_state.env.state))
                _run_step(predicted_action, treatments, auto=True)
                st.experimental_rerun()

        if c3.button("Run 10 steps", disabled=st.session_state.done):
            for _ in range(10):
                if st.session_state.done:
                    break
                _run_step(action, treatments, auto=False)
            st.experimental_rerun()

        st.markdown("---")
        st.metric("Steps completed", st.session_state.step)
        st.metric("Cumulative reward", f"{st.session_state.cumulative_reward:,.2f}")

        st.markdown("### Stand state trajectory")
        _plot_state_history(st.session_state.history, st.session_state.events)

        st.markdown("### Product distribution")
        _plot_product_history(st.session_state.product_history)

    with col2:
        st.subheader("Current state")
        state_df = _format_state_table(_state_dict_from_env(st.session_state.env))
        st.table(state_df)

        st.markdown("### Disturbance log")
        if st.session_state.events:
            log_df = pd.DataFrame(st.session_state.events)
            log_df = log_df.sort_values("step", ascending=False)
            st.dataframe(log_df, height=260)
        else:
            st.info("No disturbances recorded yet.")

        st.markdown("### Actions history")
        if st.session_state.actions:
            actions_df = pd.DataFrame(st.session_state.actions)
            st.dataframe(actions_df, height=220)
        else:
            st.caption("Actions will appear here after you step the environment.")

    if st.session_state.done:
        st.warning("Episode finished – reset to start a new run.")


if __name__ == "__main__":  # pragma: no cover - entry-point for `streamlit run`.
    main()
