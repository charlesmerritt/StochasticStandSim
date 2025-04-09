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