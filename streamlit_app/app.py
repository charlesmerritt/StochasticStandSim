import streamlit as st
import numpy as np
from StochasticStandSim.envs import StandEnv, EnvConfig

st.set_page_config(page_title="Forest Stand Simulator", layout="wide")
st.title("Forest Stand Dynamics Visualizer")

seed = st.sidebar.number_input("Seed", 0, 10_000, 7)
horizon = st.sidebar.slider("Horizon (years)", 10, 80, 40, step=5)

cfg = EnvConfig(seed=seed, horizon_years=horizon)
env = StandEnv(cfg)
obs, _ = env.reset()

cols = st.columns(3)
charts = {"age": [], "volume": [], "cash": []}

for _ in range(int(horizon / cfg.dt_years)):
    # very simple policy: harvest when volume > threshold else noop
    a = 5 if obs[1] > 300 else 0
    obs, r, done, _, info = env.step(a)
    charts["age"].append(obs[0])
    charts["volume"].append(obs[1])
    charts["cash"].append(env.s.cash_account if env.s else 0.0)
if done:
    print("done")

with cols[0]:
    st.line_chart(np.array(charts["age"]))
with cols[1]:
    st.line_chart(np.array(charts["volume"]))
with cols[2]:
    st.line_chart(np.array(charts["cash"]))

st.caption("Baseline demo. Replace policy with RL agent and wire real growth/economics.")