import gymnasium as gym
from gymnasium import spaces
import numpy as np


price_table = {
    "costs": {
        "admin_cost": 50,
        "thinning_cost": 150,
        "harvesting_cost": 75,
        "seedling_cost": 150,
        "planting_cost": 70,
        "other_cost": 150
    },
    "revenues": {
        "interest_rate": 0.04,
        "hunting_income": 20,
        "pulp_price": 100,
        "cns_price": 100,
        "saw_price": 100,
    },
}

product_specs_table = {
    "pulp_diameter_max": 5,
    "cns_diameter_max": 7,
    "saw_diameter_max": 12,
}


class PriceTable:
    """
    Holds fixed unit costs and revenues associated with stand-level forest operations and products.
    """
    def __init__(self, price_data: dict):
        self.costs = price_data.get("costs", {})
        self.revenues = price_data.get("revenues", {})

    def cost(self, key: str) -> float:
        return self.costs.get(key, 0.0)

    def price(self, product: str) -> float:
        return self.revenues.get(f"{product}_price", 0.0)

    def interest_rate(self) -> float:
        return self.revenues.get("interest_rate", 0.04)

    def hunting_income(self) -> float:
        return self.revenues.get("hunting_income", 0.0)


class CashFlowLedger:
    """
    Records all forest management-related cash flows for a given rotation.
    """
    def __init__(self):
        self.cash_flows: Dict[int, float] = {}

    def add_cost(self, year: int, amount: float):
        self.cash_flows[year] = self.cash_flows.get(year, 0.0) - abs(amount)

    def add_revenue(self, year: int, amount: float):
        self.cash_flows[year] = self.cash_flows.get(year, 0.0) + amount

    def add_annual_payment(self, start_year: int, end_year: int, amount: float):
        for year in range(start_year, end_year + 1):
            self.add_revenue(year, amount) if amount > 0 else self.add_cost(year, -amount)


class LEVModel:
    """
    Computes the Land Expectation Value (LEV) for a rotation of timber production.
    """
    def __init__(self, price_table: PriceTable, ledger: CashFlowLedger):
        self.prices = price_table
        self.ledger = ledger

    def compute_nfv(self, rotation_length: int) -> float:
        """
        Computes the Net Future Value (NFV) of a single rotation by compounding each cash flow to the end of rotation.
        """
        i = self.prices.interest_rate()
        return sum(
            amount * ((1 + i) ** (rotation_length - year))
            for year, amount in self.ledger.cash_flows.items()
        )

    def compute_lev(self, rotation_length: int) -> float:
        """
        Calculates the LEV using Faustmann’s formula for infinite identical rotations:
        LEV = NFV / ((1 + i)^t - 1)
        """
        nfv = self.compute_nfv(rotation_length)
        i = self.prices.interest_rate()
        denominator = (1 + i) ** rotation_length - 1
        return nfv / denominator if denominator != 0 else float("-inf")


def optimal_rotation() -> int:
    """
    Placeholder function for determining the biologically or economically optimal rotation length.
    """
    return 30  # To be replaced with real analysis later


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
