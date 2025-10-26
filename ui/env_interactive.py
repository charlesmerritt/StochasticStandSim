"""Streamlit app for manual interaction with StandMgmtEnv."""

from __future__ import annotations

import sys
from dataclasses import asdict
from enum import IntEnum
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.env import EnvConfig, StandMgmtEnv


class ActionType(IntEnum):
    NOOP = 0
    THIN_40 = 1
    THIN_60 = 2
    FERTILIZE = 3
    HARVEST = 4
    PLANT = 5
    SALVAGE = 6
    RX_FIRE = 7


STATE_FIELDS = [
    ("age", "Age (years)"),
    ("tpa", "Trees per acre"),
    ("ba", "Basal area (ft²/ac)"),
    ("hd", "Dominant height (ft)"),
    ("tvob", "Volume (ft³/ac)"),
    ("ci", "Competition index"),
]


def _init_session() -> None:
    defaults = {
        "env": None,
        "history": [],
        "rewards": [],
        "terminated": False,
        "cum_cash": 0.0,
        "cum_npv": 0.0,
        "last_info": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _stand_snapshot(env: StandMgmtEnv) -> Dict[str, float]:
    state = env.stand.state
    data = asdict(state)
    for key, _ in STATE_FIELDS:
        data[key] = float(data.get(key) or 0.0)
    return data


def _append_history(snapshot: Dict[str, float], reward: float, cash_flow: float, npv: float) -> None:
    st.session_state["cum_cash"] += cash_flow
    st.session_state["cum_npv"] += npv
    entry = {label: snapshot[key] for key, label in STATE_FIELDS}
    entry["Reward"] = reward
    st.session_state["history"].append(entry)
    st.session_state["rewards"].append(
        {
            "Reward": reward,
            "cash_flow": cash_flow,
            "cumulative_cash": st.session_state["cum_cash"],
            "npv": st.session_state["cum_npv"],
            "lev": 0.0,
        }
    )


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


def _render_events(events) -> None:
    st.subheader("Events")
    if not events:
        st.info("No disturbances or thinnings recorded this step.")
        return
    cleaned = []
    for event in events:
        cleaned.append({k: (asdict(v) if hasattr(v, "__dataclass_fields__") else v) for k, v in event.items()})
    st.json(cleaned)


def _render_reward_summary() -> None:
    if not st.session_state["rewards"]:
        return
    st.subheader("Latest reward breakdown")
    st.json(st.session_state["rewards"][-1])


def _build_setup_section() -> Tuple[bool, Dict[str, float]]:
    st.sidebar.header("Setup")
    options: Dict[str, float] = {}
    with st.sidebar.form("setup_form"):
        region_choice = st.selectbox("Region", ["ucp", "pucp", "lcp"], index=0)
        options["region"] = region_choice
        options["age"] = st.number_input("Initial age (years)", min_value=1.0, max_value=20.0, value=1.0, step=1.0)
        options["tpa"] = st.number_input("Planting density (trees/ac)", min_value=100.0, max_value=2000.0, value=600.0, step=25.0)
        options["si25"] = st.number_input("Site index (ft @ 25 yrs)", min_value=40.0, max_value=100.0, value=60.0, step=1.0)
        options["max_age"] = st.number_input("Rotation horizon (years)", min_value=10.0, max_value=80.0, value=35.0, step=1.0)
        options["fire_prob"] = st.slider("Fire probability", min_value=0.0, max_value=0.5, value=0.0, step=0.01)
        options["wind_prob"] = st.slider("Wind probability", min_value=0.0, max_value=0.5, value=0.0, step=0.01)
        submitted = st.form_submit_button("Start rotation")
    return submitted, options


def _create_env(options: Dict[str, float]) -> StandMgmtEnv:
    cfg = EnvConfig(
        age0=options["age"],
        tpa0=options["tpa"],
        si25=options["si25"],
        region=str(options["region"]).lower(),
        horizon_years=options["max_age"],
        disturbance_enabled=(options["fire_prob"] > 0.0 or options["wind_prob"] > 0.0),
        disturbance_probs={"fire": options["fire_prob"], "wind": options["wind_prob"]},
        growth_reward_weight=0.0,
    )
    env = StandMgmtEnv(cfg)
    env.reset()
    return env


def main() -> None:
    st.set_page_config(page_title="Stand RL Controller", layout="wide")
    _init_session()

    started, setup_opts = _build_setup_section()
    if started:
        st.session_state["env"] = _create_env(setup_opts)
        st.session_state["history"] = []
        st.session_state["rewards"] = []
        st.session_state["terminated"] = False
        st.session_state["cum_cash"] = 0.0
        st.session_state["cum_npv"] = 0.0
        st.session_state["last_info"] = {}
        snapshot = _stand_snapshot(st.session_state["env"])
        _append_history(snapshot, 0.0, 0.0, 0.0)
        st.success("Rotation started. Use the controls below to advance time.")

    env: StandMgmtEnv | None = st.session_state.get("env")
    if env is None:
        st.title("Manual Thinning Simulator")
        st.write("Configure the stand in the setup sidebar to begin a rotation.")
        return

    st.title("Manual Thinning Simulator")
    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.subheader("Controls")
        action_map = {
            "No action": ActionType.NOOP,
            "Thin (40% residual)": ActionType.THIN_40,
            "Thin (60% residual)": ActionType.THIN_60,
            "Fertilize": ActionType.FERTILIZE,
            "Harvest": ActionType.HARVEST,
            "Plant": ActionType.PLANT,
            "Salvage": ActionType.SALVAGE,
            "Prescribed fire": ActionType.RX_FIRE,
        }
        action_label = st.selectbox("Select management action", list(action_map.keys()))
        selected_action = action_map[action_label]

        replant_toggle = False
        salvage_frac = 0.3
        if selected_action == ActionType.HARVEST:
            replant_toggle = st.checkbox("Replant immediately after harvest", value=True)
        if selected_action == ActionType.SALVAGE:
            salvage_frac = st.slider("Salvage fraction", min_value=0.0, max_value=1.0, value=0.3, step=0.05)

        advance = st.button("Step environment")
        reset = st.button("Reset rotation")

        if reset:
            st.session_state["env"] = None
            st.rerun()

        if advance and not st.session_state["terminated"]:
            payload = {"type": int(selected_action.value), "value": [0.0, 0.0]}
            if selected_action == ActionType.HARVEST:
                payload["value"][0] = 1.0 if replant_toggle else 0.0
            if selected_action == ActionType.SALVAGE:
                payload["value"][0] = float(salvage_frac)

            obs, reward, terminated, truncated, info = env.step(payload)
            snapshot = _stand_snapshot(env)
            cash = info.get("cashflow_now", 0.0)
            npv = info.get("npv_at0", 0.0)
            _append_history(snapshot, reward, cash, npv)
            st.session_state["terminated"] = bool(terminated or truncated)
            st.session_state["last_info"] = info
            if info.get("action_reason"):
                st.warning(info["action_reason"])

    with right_col:
        if st.session_state["history"]:
            _render_state_panel(_stand_snapshot(env))
        _render_reward_summary()

    st.divider()
    _render_history()
    _render_events(st.session_state.get("last_info", {}).get("events", ()))


if __name__ == "__main__":
    main()
