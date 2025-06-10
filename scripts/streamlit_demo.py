import streamlit as st
import numpy as np
import sys
import os
from glob import glob
from stable_baselines3 import PPO
import plotly.graph_objects as go
import pandas as pd

# ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from env.ForestEnv import ForestStandEnv

STATE_LABELS = [
    "Age", "Biomass", "Density",
    "Fire Risk", "Wind Risk", "Value"
]

class PlotManager:
    def __init__(self, labels):
        self.labels = labels
        self.history = []

    def add(self, obs):
        row = obs.flatten().tolist()
        self.history.append(row)

    def plot(self, fire_steps=None):
        T = len(self.history)
        if T < 2:
            st.info(f"Waiting for more data… ({T}/2 timesteps)")
            return

        sequences = list(zip(*self.history))
        fig = go.Figure()

        for i, seq in enumerate(sequences):
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, T + 1)),
                    y=seq,
                    mode="lines",
                    name=self.labels[i]
                )
            )

        if fire_steps:
            max_y = max(max(seq) for seq in sequences)
            fig.add_trace(
                go.Scatter(
                    x=fire_steps,
                    y=[max_y] * len(fire_steps),
                    mode="markers",
                    marker=dict(color="red", size=8),
                    name="Fire Event"
                )
            )

        fig.update_layout(
            title="Forest Stand State Trajectory",
            xaxis_title="Time Step",
            yaxis_title="Value",
            legend_title="State Variables",
            margin=dict(l=40, r=20, t=30, b=40),
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)


st.set_page_config(layout="wide")
st.title("Forest Stand Dynamics Visualizer")

# session state init
if "env" not in st.session_state:
    st.session_state.env = ForestStandEnv()
    st.session_state.state, _ = st.session_state.env.reset()
    st.session_state.step = 0
    st.session_state.done = False
    st.session_state.first_action = False
    st.session_state.forest_state_history = [st.session_state.state]
    st.session_state.ppo_actions = []
    st.session_state.messages = []
    st.session_state.cumulative_revenue = 0.0
    st.session_state.max_horizon = 200

# model selection
model_files = sorted(glob(os.path.join("models", "ppo_forest_*.zip")))
model_names = [os.path.basename(f) for f in model_files]

st.sidebar.markdown("### Load PPO Model")
if model_names:
    choice = st.sidebar.selectbox("Select PPO model", model_names)
    if st.session_state.get("selected_model") != choice:
        try:
            st.session_state.ppo_model = PPO.load(os.path.join("models", choice), device="cpu")
            st.session_state.selected_model = choice
            st.sidebar.success(f"Loaded {choice}")
        except Exception as e:
            st.session_state.ppo_model = None
            st.sidebar.error(f"Load failed: {e}")
else:
    st.sidebar.warning("No PPO models found")
    st.session_state.ppo_model = None

# controls
if not st.session_state.first_action:
    st.session_state.max_horizon = st.sidebar.number_input(
        "Simulation Horizon (Years)", min_value=1, max_value=500, value=200
    )
else:
    st.sidebar.markdown(f"**Horizon:** {st.session_state.max_horizon}")

thin_pct = st.sidebar.slider("Thinning %", 0.0, 1.0, 0.0, 0.01)
fert_N = st.sidebar.slider("Nitrogen Fert %", 0.0, 1.0, 0.0, 0.01)
fert_P = st.sidebar.slider("Phosphorus Fert %", 0.0, 1.0, 0.0, 0.01)

if st.sidebar.button("Reset"):
    st.session_state.env = ForestStandEnv()
    st.session_state.state, _ = st.session_state.env.reset()
    st.session_state.step = 0
    st.session_state.done = False
    st.session_state.first_action = False
    st.session_state.forest_state_history = [st.session_state.state]
    st.session_state.ppo_actions = []
    st.session_state.messages = []
    st.session_state.cumulative_revenue = 0.0
    st.rerun()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Manual Step")
    if st.button("Step") and not st.session_state.done:
        action = np.array([thin_pct, fert_N, fert_P])
        new_state, reward, done, _, info = st.session_state.env.step(action)
        st.session_state.state = new_state
        st.session_state.done = done
        st.session_state.step += 1
        st.session_state.first_action = True
        st.session_state.forest_state_history.append(new_state)
        rev = info.get("revenue", 0.0)
        st.session_state.cumulative_revenue += rev

        st.session_state.messages.append(
            f"success: Step {st.session_state.step}, reward {reward:.2f}, revenue {rev:.2f}"
        )
        if info.get("fire_event"):
            sev = info.get("fire_severity", 0.0)
            st.session_state.messages.append(
                f"warning: 🔥 Fire at step {st.session_state.step}, severity {sev:.2f}"
            )
        st.rerun()

    st.subheader("PPO Step")
    if st.button("Auto Step") and not st.session_state.done:
        model = st.session_state.ppo_model
        if model:
            action, _ = model.predict(st.session_state.state, deterministic=True)
            new_state, reward, done, _, info = st.session_state.env.step(action)
            st.session_state.state = new_state
            st.session_state.done = done
            st.session_state.step += 1
            st.session_state.first_action = True
            st.session_state.forest_state_history.append(new_state)
            st.session_state.ppo_actions.append(action.tolist())
            rev = info.get("revenue", 0.0)
            st.session_state.cumulative_revenue += rev

            st.session_state.messages.append(
                f"success: PPO step {st.session_state.step}, reward {reward:.2f}, revenue {rev:.2f}"
            )
            if info.get("fire_event"):
                sev = info.get("fire_severity", 0.0)
                st.session_state.messages.append(
                    f"warning: 🔥 Fire at step {st.session_state.step}, severity {sev:.2f}"
                )
            st.rerun()
        else:
            st.error("PPO model not loaded")

    st.subheader("Metrics & Logs")
    st.metric("Cumulative Revenue", f"${st.session_state.cumulative_revenue:,.2f}")

    if st.session_state.ppo_actions:
        df = pd.DataFrame(
            st.session_state.ppo_actions,
            columns=["Thinning", "Nitrogen", "Phosphorus"]
        )
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download PPO Actions", csv, "ppo_actions.csv")

    for msg in reversed(st.session_state.messages[-30:]):
        if msg.startswith("success:"):
            st.success(msg[len("success:"):])
        elif msg.startswith("warning:"):
            st.warning(msg[len("warning:"):])
        elif msg.startswith("error:"):
            st.error(msg[len("error:"):])
        else:
            st.info(msg)

with col2:
    st.subheader("Forest Stand State Over Time")

    if "plot_mgr" not in st.session_state:
        st.session_state.plot_mgr = PlotManager(STATE_LABELS)

    mgr = st.session_state.plot_mgr
    mgr.add(np.array(st.session_state.state).reshape(1, -1))

    fire_steps = [
        i + 1
        for i, m in enumerate(st.session_state.messages)
        if "🔥 Fire" in m
    ]

    mgr.plot(fire_steps=fire_steps)

    if st.session_state.done:
        st.warning("Simulation complete")
        st.info(f"Final revenue: ${st.session_state.cumulative_revenue:,.2f}")
