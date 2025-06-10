import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ForestStandEnv(gym.Env):
    def __init__(self):
        super(ForestStandEnv, self).__init__()

        # State: [age, biomass, density, fire_risk, windthrow, value]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0]),
            high=np.array([200, 500, 500, 1, 1, 50000]),
            dtype=np.float32
        )

        # Action: [thinning %, N fert %, P fert %]
        self.action_space = spaces.Box(low=np.array([0, 0, 0]), high=np.array([1, 1, 1]), dtype=np.float32)

        # Cost model (reduced for incentive)
        self.k1, self.k2, self.k3 = 500, 100, 80

        # Fertilizer effectiveness
        self.mu_N, self.sigma_N = 5, 2
        self.mu_P, self.sigma_P = 3, 1

        self.reset()

    def reset(self, seed=None, options=None):
        self.state = np.array([
            np.random.randint(1, 10),  # Age
            np.random.uniform(50, 100),  # Biomass
            np.random.uniform(100, 300),  # Density
            np.random.uniform(0, 0.2),  # Fire risk
            np.random.uniform(0, 0.2),  # Windthrow risk
            np.random.uniform(500, 5000),  # Value
        ], dtype=np.float32)

        return self.state, {}

    def step(self, action):
        age, biomass, density, fire_risk, windthrow, value = self.state
        thin_pct, fert_N, fert_P = action

        # --- Biomass before thinning ---
        biomass_before_thinning = biomass

        # --- Apply thinning ---
        biomass *= (1 - thin_pct)
        density *= (1 - thin_pct)
        density = min(500, density + np.random.uniform(1, 3))  # passive regen

        # --- Timber revenue from harvested biomass ---
        harvested_biomass = biomass_before_thinning * thin_pct
        harvest_value = harvested_biomass * 10  # unit price per biomass
        revenue = harvest_value  # can accumulate in agent if modeled

        # --- Growth ---
        competition_factor = max(0.1, 1.0 - (density / 500))
        fert_boost = 1 + 0.3 * fert_N + 0.2 * fert_P
        base_growth = biomass * 0.05 * competition_factor * fert_boost
        noise = np.random.normal(0, 2.0)
        growth = max(0, base_growth + noise)
        biomass = min(500, biomass + growth)

        # --- Fire risk accumulation ---
        prev_fire_risk = fire_risk
        fire_risk += 0.002 * (biomass / 500)  # normalized slope
        fire_risk = min(fire_risk, 1.0)
        fire_risk = min(fire_risk, prev_fire_risk + 0.02)  # slow year-over-year increase

        # --- Fire event ---
        fire_event = np.random.rand() < fire_risk
        if fire_event:
            severity = np.clip(np.random.normal(loc=0.5, scale=0.15), 0.1, 0.9)
            biomass *= (1 - severity)
            fire_risk = 0

        # --- Windthrow dynamics ---
        if np.random.rand() < 0.1:
            windthrow = min(1.0, windthrow + 0.2 * (1 - thin_pct))
        windthrow = max(0.0, windthrow - 0.05)

        # --- Stand value (future potential) ---
        value = 500 + 5 * biomass + 2 * age - 100 * fire_risk - 50
        value = np.clip(value, 0, 50000)

        # --- Cost and reward ---
        cost = self.k1 * thin_pct + self.k2 * fert_N + self.k3 * fert_P
        reward = revenue + value * 0.01 - (fire_risk * 100) - cost

        # --- Advance time ---
        new_state = np.array([
            age + 1,
            biomass,
            density,
            fire_risk,
            windthrow,
            value
        ], dtype=np.float32)

        self.state = new_state
        done = age + 1 >= 200

        info = {
            "revenue": revenue,
            "fire_event": fire_event,
            "fire_severity": severity if fire_event else 0.0
        }
        return self.state, reward, done, False, info

    def render(self):
        print(f"Age: {self.state[0]}, Biomass: {self.state[1]:.2f}, Fire Risk: {self.state[6]:.2f}")
