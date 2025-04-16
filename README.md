Build Instructions:
This repository contains both a ```requirements.txt``` and a ```environment.yml``` for creating the proper environment
with Conda/pip.

All commands should be run from the root directory and in an appropriate conda or venv python environment, which can be
set up with pip by running
```
pip install -r requirements.txt
```
inside your activated conda environment or venv. All commands should be run from the root directory 'rl_yield_modelling'. 
Alternatively, the environment can be created directly in conda using the ```environment.yml``` file. Note that both of 
these methods assume you have CUDA v12.6 installed on the system. See this link for instructions https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html
```
conda env create -f environment.yml
```
Usage:
1) Train or retrain RL agents
```
python train_[agent].py 
```
or 
```
python -m scripts.train_[agent].py
```
Where [agent] is replaced with one of PPO, DDPG, etc. Currently only the PPO agent is implemented, but future choices 
will include more options such as DDPG to compare different agents/reinforcement learning algorithms.

This will train a RL agent on the stochastic ForestStandEnv with its current parameters, in this case an agent refers 
to a hypothetical stand manager with the following actions available at each one year time step: thin %, 
fertilization_phos %, and fertilization_nit %. The choice of learning algorithm or agent will affect the final outcome
of the automatic management decisions, with different agents performing different actions as they each learn a distinct 
policy.

Models are saved to: models/[agent]_forest_<timestamp>.zip

Logs are saved to: logs/[agent]_forest_<timestamp>/

2) Run the Streamlit visualizer and policy debugger
```
streamlit run streamlit_demo.py
```
3) View training logs with tensorboard
```
tensorboard --logdir logs/
```

4) You can also run a RL agent directly by running
```
python run_[agent].py
```

## 🌲 Forest Simulation Design

### 🧠 State & Environment Modeling
- [ ] Expand state to include long-term dependencies (via recurrent policy or state history)
- [ ] Flesh out state dynamics and reward functions
- [ ] Model forest compartments as interacting agents for multi-agent or spatial dynamics
- [ ] Allow granular control over stand initial conditions and environmental parameters
- [ ] Clarify and define environmental variables like fertilization %

### 🔁 Dynamic Systems & Interventions
- [ ] Add fire suppression strategies or pest control interventions
- [ ] Introduce sparse or delayed rewards to better model long-term sustainability goals
- [ ] Model CO₂ price as an external economic variable (e.g., time series or stochastic process)
- [ ] Integrate real-world economic data (e.g., timber-mart south) to drive pricing dynamics

---

## 🧠 Reinforcement Learning Strategy

### 🏗 Agent Training Approaches
- [ ] Train agents using PPO, DDPG, and SAC
- [ ] Explore model-based RL (e.g., Dreamer, PlaNet) for expensive simulation environments
- [ ] Investigate offline RL methods using batch or previously collected forest management data
- [ ] Use curriculum learning to ease policy learning: begin with simpler dynamics, then add disturbances

### 🎯 Reward Engineering
- [ ] Add a reward function that incorporates:
  - [ ] Timber economic value
  - [ ] Carbon credit value
  - [ ] Penalties/incentives for ecological sustainability

---

## 📊 Visualization, Logging, and Diagnostics

### 📈 Visualizations & Interpretability
- [ ] Visualize state evolution over time using Matplotlib or Plotly
- [ ] Provide flexible control over visualizations (e.g., raw vs normalized views)

### 📝 Logging & Debugging Tools
- [ ] Add logging mechanisms to track agent performance over episodes/time
- [ ] Integrate with TensorBoard or Weights & Biases for enhanced diagnostic logging

---

## 📐 Validation & Realism
- [ ] Validate simulation outputs and learned policies against real-world data or known benchmarks
