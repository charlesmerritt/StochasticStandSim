# StochasticStandSim

StochasticStandSim provides a light-weight simulation environment for exploring
forest stand management strategies with deterministic and stochastic
disturbance processes.

## Installation

1. Ensure you are using Python 3.12 or newer.
2. (Optional) Create and activate a virtual environment.
3. Install the project dependencies:

   ```bash
   pip install -e .
   ```

   The project depends on `pandas` and `pyyaml` for configuration loading in
   addition to the reinforcement learning, plotting, and UI stacks defined in
   `pyproject.toml`.

## Quick start

* Launch the interactive Streamlit dashboard:

  ```bash
  streamlit run app/streamlit_demo.py
  ```

  The dashboard visualises stand trajectories, allows you to toggle stochastic
  disturbance envelopes, and demonstrates how the environment responds to
  management actions.

* Interact with the Gymnasium environment directly:

  ```python
  from core.stand_env import StandEnv

  env = StandEnv()
  obs, info = env.reset()
  obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
  ```

## Configuration files

All configuration lives in the `config/` directory and is loaded at runtime:

* `config/prices.csv` – regional biomass prices, salvage discounts, and thinning
  costs used by `core.economics.reward_functions` when computing rewards.
* `config/costs.csv` – reference operating costs that downstream workflows can
  consume when constructing economic scenarios.
* `config/disturbances.yaml` – stochastic disturbance templates shared between
  the Gymnasium environment and the Streamlit demo.

Example stand tables are provided in `data/example_stands.csv` for quickly
bootstrapping simulations from empirical data.
