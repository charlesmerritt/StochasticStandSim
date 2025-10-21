"""Streamlit app for manual control of the thinning RL environment."""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.env import ActionType, StandEnv
from core.growth import Region, StandParams


STATE_FIELDS = [
    ("age", "Age (years)"),
    ("tpa", "Trees per acre"),
    ("ba", "Basal area (ft²/ac)"),
    ("hd", "Dominant height (ft)"),
    ("vol_ob", "Total volume (ft³/ac)"),
    ("ci", "Competition index"),
]


def _init_session() -> None:
    if "env" not in st.session_state:
        st.session_state["env"] = None
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "rewards" not in st.session_state:
        st.session_state["rewards"] = []
    if "terminated" not in st.session_state:
        st.session_state["terminated"] = False
    if "thin_max_fraction" not in st.session_state:
        st.session_state["thin_max_fraction"] = 0.4


def _region_options() -> Dict[str, Region]:
    return {region.value: region for region in Region}


def _stand_snapshot(env: StandEnv) -> Dict[str, float]:
    state = env.state
    data = asdict(state)
    # Ensure derived metrics exist even if None in dataclass.
    for key, _ in STATE_FIELDS:
        data[key] = float(data.get(key) or 0.0)
    return data


def _append_history(snapshot: Dict[str, float], reward: float, breakdown: Dict[str, float]) -> None:
    entry = {label: snapshot[key] for key, label in STATE_FIELDS}
    entry["Reward"] = reward
    st.session_state["history"].append(entry)
    st.session_state["rewards"].append({"Reward": reward, **breakdown})


def _reset_environment(
    params: StandParams,
    thin_max: float,
    disturbance_fire: float,
    disturbance_wind: float,
    max_age: float,
) -> None:
    env = StandEnv(
        stand=params,
        thin_max_fraction=thin_max,
        disturbance_probs={"fire": disturbance_fire, "wind": disturbance_wind},
        max_age=max_age,
    )
    env.reset()
    st.session_state["env"] = env
    st.session_state["history"] = []
    st.session_state["rewards"] = []
    st.session_state["terminated"] = False
    st.session_state["thin_max_fraction"] = thin_max

    snapshot = _stand_snapshot(env)
    _append_history(
        snapshot,
        0.0,
        {
            "cash_flow": 0.0,
            "cumulative_cash": 0.0,
            "npv": 0.0,
            "lev": 0.0,
        },
    )


def _build_setup_section() -> Tuple[bool, Dict[str, float]]:
    st.sidebar.header("Setup")
    options: Dict[str, float] = {}
    with st.sidebar.form("setup_form"):
        region_map = _region_options()
        region_choice = st.selectbox("Region", list(region_map.keys()), index=0)
        options["region"] = region_choice
        options["age"] = st.number_input("Initial age (years)", min_value=1.0, max_value=20.0, value=1.0, step=1.0)
        options["tpa"] = st.number_input("Planting density (trees/ac)", min_value=100.0, max_value=1200.0, value=600.0, step=25.0)
        options["si25"] = st.number_input("Site index (ft @ 25 yrs)", min_value=40.0, max_value=90.0, value=60.0, step=1.0)
        options["max_age"] = st.number_input("Rotation horizon (years)", min_value=10.0, max_value=60.0, value=35.0, step=1.0)
        thin_max = st.slider("Max thinning removal fraction", min_value=0.0, max_value=0.9, value=0.4, step=0.05)
        fire_prob = st.slider("Fire probability", min_value=0.0, max_value=0.5, value=0.05, step=0.01)
        wind_prob = st.slider("Wind probability", min_value=0.0, max_value=0.5, value=0.03, step=0.01)
        submitted = st.form_submit_button("Start rotation")
    options["thin_max"] = thin_max
    options["fire_prob"] = fire_prob
    options["wind_prob"] = wind_prob
    return submitted, options


def _render_state_panel(snapshot: Dict[str, float]) -> None:
    st.subheader("Stand State")
    for key, label in STATE_FIELDS:
        st.metric(label, f"{snapshot[key]:.2f}")


def _render_history() -> None:
    if not st.session_state["history"]:
        return
    df = pd.DataFrame(st.session_state["history"])
    st.subheader("Trajectory")
    chart_df = df.set_index("Age (years)") if "Age (years)" in df.columns else df
    st.line_chart(chart_df, height=260)
    st.dataframe(df.style.format("{:.2f}"), width="stretch")


def _render_events(events: Tuple[dict, ...]) -> None:
    if not events:
        st.info("No disturbances or thinnings applied this step.")
        return
    st.subheader("Events")
    cleaned: List[Dict[str, object]] = []
    for event in events:
        entry: Dict[str, object] = {}
        for key, value in event.items():
            if hasattr(value, "__dataclass_fields__"):
                entry[key] = asdict(value)
            elif isinstance(value, tuple):
                entry[key] = list(value)
            else:
                entry[key] = value
        cleaned.append(entry)
    st.json(cleaned)


def _render_reward_summary() -> None:
    if not st.session_state["rewards"]:
        return
    st.subheader("Latest reward breakdown")
    st.json(st.session_state["rewards"][-1])


def main() -> None:
    st.set_page_config(page_title="Stand RL Controller", layout="wide")
    _init_session()

    started, setup_opts = _build_setup_section()
    if started:
        params = StandParams(
            name="custom",
            age=setup_opts["age"],
            tpa=setup_opts["tpa"],
            region=_region_options()[setup_opts["region"]],
            si25=setup_opts["si25"],
        )
        _reset_environment(
            params,
            setup_opts["thin_max"],
            setup_opts["fire_prob"],
            setup_opts["wind_prob"],
            setup_opts["max_age"],
        )
        st.success("Rotation started. Use the controls below to advance time.")

    env: StandEnv | None = st.session_state.get("env")
    if env is None:
        st.title("Manual Thinning Simulator")
        st.write("Configure the stand in the setup sidebar to begin a rotation.")
        return

    st.title("Manual Thinning Simulator")
    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.subheader("Controls")
        thin_max = st.session_state["thin_max_fraction"]
        action_map = {
            "No action": ActionType.NOOP,
            "Thinning": ActionType.THIN,
            "Harvest (optional replant)": ActionType.HARVEST,
            "Sell (exit rotation)": ActionType.SELL,
            "Salvage": ActionType.SALVAGE,
            "Replant": ActionType.REPLANT,
            "Fertilize": ActionType.FERTILIZE,
            "Prescribed fire": ActionType.RX_FIRE,
        }
        action_label = st.selectbox("Select management action", list(action_map.keys()))
        selected_action = action_map[action_label]

        action_params = [0.0, 0.0]
        if selected_action == ActionType.THIN:
            if thin_max <= 0.0:
                st.info("Thinning disabled: maximum removal fraction is 0.")
            removal = st.slider(
                "Removal fraction (absolute)",
                min_value=0.0,
                max_value=float(thin_max),
                value=min(0.25, float(thin_max)),
                step=0.01,
                key="thin_amount",
            )
            if thin_max > 0:
                action_params[0] = removal / thin_max
        elif selected_action == ActionType.HARVEST:
            replant_now = st.checkbox("Replant immediately after harvest", value=True)
            action_params[0] = 1.0 if replant_now else 0.0

        advance = st.button("Step environment")
        reset = st.button("Reset rotation")

        if reset:
            st.session_state["env"] = None
            st.rerun()

        if advance and not st.session_state["terminated"]:
            payload = {"type": int(selected_action.value), "value": action_params}
            obs, reward, terminated, _, info = env.step(payload)
            snapshot = _stand_snapshot(env)
            breakdown = info.get("finance", {})
            _append_history(snapshot, reward, breakdown if isinstance(breakdown, dict) else {})
            st.session_state["terminated"] = terminated
            if "action_warn" in info:
                st.warning(info["action_warn"])
            elif info.get("action_blocked"):
                st.info("Action skipped due to disturbance occurring this year.")
            elif terminated:
                st.warning("Rotation terminated: stand reached max age or sell action.")
            else:
                st.rerun()

    with right_col:
        current_snapshot = _stand_snapshot(env)
        _render_state_panel(current_snapshot)

    st.divider()
    _render_history()
    st.divider()
    _render_events(env.events)
    st.divider()
    _render_reward_summary()


if __name__ == "__main__":
    main()
