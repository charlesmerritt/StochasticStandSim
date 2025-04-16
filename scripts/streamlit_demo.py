import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
from glob import glob
from stable_baselines3 import PPO

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.ForestEnv import ForestStandEnv

# --- Constants ---
STATE_LABELS = [
    "Age", "Biomass", "Density", "Carbon",
    "Fire Risk", "Pest Risk", "Wind Risk", "Value"
]
STATE_MIN = np.array([0, 0, 0, 0, 0, 0, 0, 0])
STATE_MAX = np.array([200, 500, 500, 300, 1, 3, 1, 50000])

st.set_page_config(layout="wide")
st.title("🌲 Forest Stand Dynamics Visualizer")

# --- Session state initialization ---
if "env" not in st.session_state:
    st.session_state.env = ForestStandEnv()
    st.session_state.state, _ = st.session_state.env.reset()
    st.session_state.step = 0
    st.session_state.done = False
    st.session_state.first_action_taken = False

# Find available models
model_files = sorted(glob(os.path.join("models", "ppo_forest_*.zip")))
model_names = [os.path.basename(f) for f in model_files]

# Select model
st.sidebar.markdown("### Load PPO Model")
if model_names:
    selected_model = st.sidebar.selectbox("Select trained PPO model:", model_names)
    if "selected_model" not in st.session_state or st.session_state.selected_model != selected_model:
        try:
            st.session_state.ppo_model = PPO.load(os.path.join("models", selected_model), device="cpu")
            st.session_state.selected_model = selected_model
            st.sidebar.success(f"Loaded model: {selected_model}")
        except Exception as e:
            st.sidebar.error(f"Failed to load model: {e}")
            st.session_state.ppo_model = None
else:
    st.sidebar.warning("No PPO models found in /models.")
    st.session_state.ppo_model = None

if "ppo_actions" not in st.session_state:
    st.session_state.ppo_actions = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "cumulative_revenue" not in st.session_state:
    st.session_state.cumulative_revenue = 0.0

if "max_horizon" not in st.session_state:
    st.session_state.max_horizon = 200  # default time horizon

env = st.session_state.env
state = st.session_state.state
step = st.session_state.step
done = st.session_state.done

# --- Sidebar controls ---
st.sidebar.header("Control Actions")

if not st.session_state.first_action_taken:
    st.session_state.max_horizon = st.sidebar.number_input(
        "Simulation Horizon (Years)", min_value=1, max_value=500, value=200, step=1
    )
else:
    st.sidebar.markdown(f"**Simulation Horizon:** {st.session_state.max_horizon} (locked)")

thin_pct = st.sidebar.slider("Thinning %", 0.0, 1.0, 0.0, 0.01)
fert_N = st.sidebar.slider("Nitrogen Fertilizer %", 0.0, 1.0, 0.0, 0.01)
fert_P = st.sidebar.slider("Phosphorus Fertilizer %", 0.0, 1.0, 0.0, 0.01)

if st.sidebar.button("Reset Environment"):
    st.session_state.env = ForestStandEnv()
    st.session_state.state, _ = st.session_state.env.reset()
    st.session_state.step = 0
    st.session_state.done = False
    st.session_state.ppo_actions = []
    st.session_state.messages = []
    st.session_state.first_action_taken = False
    st.session_state.cumulative_revenue = 0.0
    st.rerun()

# --- Action buttons and charts ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Manual Actions")
    if st.button("Take Action Step") and not done:
        action = np.array([thin_pct, fert_N, fert_P])
        new_state, reward, done, _, info = env.step(action)
        st.session_state.state = new_state
        st.session_state.step += 1
        st.session_state.done = done
        st.session_state.first_action_taken = True
        revenue = info.get("revenue", 0.0)
        st.session_state.cumulative_revenue += revenue
        fire_event = info.get("fire_event", False)
        severity = info.get("fire_severity", 0.0)

        messages = [
            f"info: Manual Action: [Thinning: {action[0]:.2f}, N-Fert: {action[1]:.2f}, P-Fert: {action[2]:.2f}]",
            f"success: Reward: {reward:.2f} at Step {st.session_state.step}, Revenue: {revenue:.2f}"
        ]

        if fire_event:
            messages.append(f"warning: 🔥 Fire occurred at Step {st.session_state.step} — Severity: {severity:.2f}")

        st.session_state.messages.extend(messages)

        st.rerun()

    st.subheader("Agent Actions")
    if st.button("Auto Step with PPO") and not done:
        model = st.session_state.ppo_model
        if model is not None:
            action, _ = model.predict(state, deterministic=True)
            new_state, reward, done, _, info = env.step(action)
            st.session_state.state = new_state
            st.session_state.step += 1
            st.session_state.done = done
            st.session_state.first_action_taken = True
            st.session_state.ppo_actions.append(action.tolist())
            revenue = info.get("revenue", 0.0)
            st.session_state.cumulative_revenue += revenue
            fire_event = info.get("fire_event", False)
            severity = info.get("fire_severity", 0.0)

            messages = [
                f"info: Manual Action: [Thinning: {action[0]:.2f}, N-Fert: {action[1]:.2f}, P-Fert: {action[2]:.2f}]",
                f"success: Reward: {reward:.2f} at Step {st.session_state.step}, Revenue: {revenue:.2f}"
            ]

            if fire_event:
                messages.append(f"warning: 🔥 Fire occurred at Step {st.session_state.step} — Severity: {severity:.2f}")

            st.session_state.messages.extend(messages)

            st.rerun()
        else:
            st.session_state.messages.append("error: PPO model not loaded.")

    if st.button("Run PPO to Completion") and not done:
        model = st.session_state.ppo_model
        if model is not None:
            while not st.session_state.done:
                action, _ = model.predict(st.session_state.state, deterministic=True)
                new_state, reward, done, _, info = env.step(action)
                st.session_state.state = new_state
                st.session_state.step += 1
                st.session_state.done = done
                st.session_state.first_action_taken = True
                st.session_state.ppo_actions.append(action.tolist())
                st.session_state.cumulative_revenue += info.get("revenue", 0.0)
                if st.session_state.step >= st.session_state.max_horizon:
                    st.session_state.done = True
            st.session_state.messages.append(
                f"warning: Completed PPO rollout in {st.session_state.step} steps."
            )
            st.session_state.messages.append(
                f"success: Total Banked Revenue: {st.session_state.cumulative_revenue:.2f}"
            )
            st.rerun()
        else:
            st.session_state.messages.append("error: PPO model not loaded.")

    if st.session_state.ppo_actions:
        df = pd.DataFrame(st.session_state.ppo_actions, columns=["Thinning", "Nitrogen", "Phosphorus"])
        csv = df.to_csv(index=True).encode("utf-8")
        st.download_button("📥 Download PPO Actions (CSV)", csv, "ppo_actions.csv", "text/csv")

    st.metric("💰 Cumulative Harvest Revenue", f"${st.session_state.cumulative_revenue:,.2f}")
    st.subheader("Message Log")
    for msg in reversed(st.session_state.messages[-30:]):
        if msg.startswith("info:"):
            st.info(msg[5:])
        elif msg.startswith("success:"):
            st.success(msg[8:])
        elif msg.startswith("warning:"):
            st.warning(msg[8:])
        elif msg.startswith("error:"):
            st.error(msg[6:])
        else:
            st.text(msg)

# --- Visualization ---
with col2:
    st.subheader("Normalized Forest State")
    state_norm = (state - STATE_MIN) / (STATE_MAX - STATE_MIN + 1e-8)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(STATE_LABELS, state_norm)
    ax.set_ylim([0, 1])
    ax.set_ylabel("Normalized Value [0–1]")
    ax.set_title(f"Stand State at Year {step}")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

    if st.session_state.ppo_actions:
        st.subheader("PPO Actions History")
        actions_arr = np.array(st.session_state.ppo_actions).T
        fig2, ax2 = plt.subplots()
        for i, label in enumerate(["Thinning", "Nitrogen", "Phosphorus"]):
            ax2.plot(actions_arr[i], label=label)
        ax2.set_ylabel("Action Value")
        ax2.set_xlabel("Step")
        ax2.set_title("PPO Action Trajectory")
        ax2.set_ylim([0, 1])
        ax2.legend()
        st.pyplot(fig2)

# --- Completion condition ---
if done and not any("terminal age" in msg for msg in st.session_state.messages):
    st.session_state.messages.append("warning: Simulation complete. Forest has reached terminal age.")
    st.session_state.messages.append(f"success: Final Banked Revenue: {st.session_state.cumulative_revenue:.2f}")
