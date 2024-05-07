import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from gymnasium.utils.env_checker import check_env
from gymnasium.envs.registration import register
import random

register(
    id='ForestGrowthEnv',
    entry_point='env:ForestGrowthEnv',
)


# Forest growth is nonlinear, and often shows diminishing returns as stands become mature or overcrowded.
# Growth rate changes based on innumerable factors, only some of which can be modelled here.
# These factors can include things like weather, species, soil, competition, and more.
# We will focus on competition and weather as a starting point for this project.

class ForestGrowthEnv(gym.Env):
    metadata = {'render_modes': ['console']}

    # Constants for growth calculations
    r = 0.03  # intrinsic growth rate
    K0 = 1200  # baseline carrying capacity
    alpha = 0.004  # sensitivity to density
    step_size = 5  # Years between thinning decisions

    def __init__(self, render_mode=None):
        super(ForestGrowthEnv, self).__init__()
        # Actions: percentage of area to thin (0% to 100%)
        self.action_space = spaces.Box(low=np.array([0.0]), high=np.array([1.0]), dtype=np.float32)

        # Observation space: [Timber Volume (V), Density (D), Forest Stand Area (A)]
        self.observation_space = spaces.Box(low=np.array([0, 0, 0]), high=np.array([np.inf, np.inf, np.inf]),
                                            dtype=np.float32)

        # Initial state
        self.state = None
        self.np_random, _ = gym.utils.seeding.np_random(None)


        # Render
        self.render_mode = render_mode

    def reset(self, seed=None):
        super().reset(seed=seed)
        print("RESET")

        # Reset the state of the environment to an initial state
        V = self.np_random.uniform(50, 150)  # initial volume randomized
        D = self.np_random.uniform(25, 75)  # initial density randomized
        A = 10  # forest area in hectares constant for simplicity
        self.state = np.array([V, D, A], dtype=np.float32)

        # Auxiliary info, SB3 requirement
        info = {}
        self.current_step = 0
        self.total_reward = 0
        return self.state, info

    def step(self, action):
        V, D, A = self.state
        print(f"Initial Volume V: {V}, Density D: {D}, Area A: {A}")

        # Calculate growth & random weather impact
        for _ in range(self.step_size):
            K = self.K0 * np.exp(-self.alpha * D) * A
            growth = self.r * V * (1 - V / K)
            if random.random() < 0.10:
                # 10% chance that weather completely stops growth
                print("Weather impact: No growth due to drought.")
                growth = 0
            elif random.random() < 0.10:
                # 20% chance to reduce the total volume by 1% to 5%
                reduction_percentage = random.uniform(0.01, 0.05)  # Random reduction between 1% and 5%
                weather_reduction = V * reduction_percentage
                V -= weather_reduction
                print(f"Weather impact: Volume reduced by {weather_reduction:.3f} due to weather conditions.")

            V += growth
            print(f"Updated Volume V after growth and weather impact: {V:.3f}")

        # Thinning action
        thinning = action[0]
        removed_volume = V * thinning
        V -= removed_volume
        D *= (1 - thinning)  # Density reduces proportionally
        print(f"Action (thinning): {thinning}, Removed Volume: {removed_volume}")
        print(f"Volume V after thinning: {V}")

        self.state = np.array([V, D, A], dtype=np.float32)
        reward = removed_volume
        self.total_reward += reward
        print(f"Total Reward: {self.total_reward:.3f}")
        terminated = V <= 0 or D <= 0
        if not isinstance(terminated, bool):
            terminated = bool(terminated)

        self.last_action = action
        self.current_step += 1

        # SB3 Requirements
        truncated = False
        info = {}

        return self.state, reward, terminated, truncated, info

    def render(self, mode='console'):
        if self.render_mode == 'console':
            # Simple text-based rendering
            print(f"Step: {self.current_step}, Action taken: {self.last_action}, Current State: Volume: "
                  f"{self.state[0]:.3f}, Density: {self.state[1]:.3f}")
            print(f"Total Reward: {self.total_reward:.3f}")
        elif self.render_mode == 'human':
            # Potentially more complex graphical rendering
            pass
