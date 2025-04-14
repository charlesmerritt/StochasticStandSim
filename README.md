Usage:
All commands should be run from the root directory and in an appropriate conda or venv python environment, which can be
set up easily by running
```
pip install -r requirements.txt
```
inside your activated conda environment or venv. All commands should be run from the root directory 'rl_yield_modelling'

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

Main TODO:
- Train agents with PPO, DDPG, and SAC.
- Flesh out state dynamics and reward functions.
- Validate the model with real-world data.
- Integrate real price values from timber-mart south.
- Allow granular control over stand initial conditions and environmental parameters.
- More control over the types of visualizations/avoid normalization.
- How should fertilization % be interpreted?

Not working:
- Agent not learning with PPO.
- Stand dynamics not quite right.
- Pest risk not quite right.
- Windstorm risk not quite right.

TODO:
- Add long-term state dependencies using a recurrent policy (e.g., RecurrentPPO from sb3-contrib) or by expanding state to include history.
- Model forest compartments as interacting agents to simulate larger-scale dynamics.
- Add fire suppression strategies or pest control interventions.
- Model CO₂ price as an external economic variable.
- Add a reward function that considers the economic value of timber and carbon credits.
- Introduce sparse or delayed rewards to simulate long-term goals like sustainability.
- Use curriculum learning: start with easy dynamics, then introduce disturbances gradually.
- Use Matplotlib or Plotly to visualize state evolution.
- Add a logging mechanism to track agent performance over time.
- Integrate with TensorBoard/WandB for richer diagnostics beyond just reward.
- Explore model-based RL like Dreamer or PlaNet for environments with costly simulations.
- Investigate offline RL methods using previously collected forest management data/Batch RL.