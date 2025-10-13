from __future__ import annotations
from StochasticStandSim.core.types import ObsDict
from StochasticStandSim.core.state import initial_state, observe
from StochasticStandSim.core.rng import RNG
from StochasticStandSim.core.config import EnvConfig
from StochasticStandSim.disturbances.regimes import DisturbanceEngine
from StochasticStandSim.economics.finance import npv_update_cash
from StochasticStandSim.economics.pricing import thinning_revenue_cost, revenue_for_harvest
from StochasticStandSim.economics.salvage import salvage_value
from StochasticStandSim.growth import step_growth
from StochasticStandSim.actions import ACTIONS
from StochasticStandSim.types import Action


self.s: StandState | None = None
self.step_idx: int = 0
self.max_steps: int = 1000
self.dt: float = 1.0
self.rng: RNG | None = None
self.dist_engine: DisturbanceEngine | None = None


def reset(self, *, seed: int | None = None, options: dict | None = None):
    if seed is not None:
        self.rng = RNG(seed)
    self.dist_engine = DisturbanceEngine(self.cfg.disturbance, self.rng)
    self.s = initial_state(self.cfg)
    self.step_idx = 0
    obs = np.array(list(observe(self.s).values()), dtype=np.float32)
    return obs, {}


def step(self, action_idx: int):
    assert self.s is not None
    done = False
    info: dict = {}
    action: Action = ACTIONS[action_idx]
    reward = 0.0
    # management actions
    if action.name == "thin":
        removed = self.s.volume_m3 * action.thin_frac
        self.s = replace(self.s, volume_m3=self.s.volume_m3 - removed)
        reward += thinning_revenue_cost(removed, self.cfg.economics)
    elif action.name == "fertilize":
        reward -= self.cfg.economics.fert_cost_per_ha
    elif action.name == "pesticide":
        reward -= self.cfg.economics.pesticide_cost_per_ha
    elif action.name == "rx_fire":
        reward -= self.cfg.economics.rxfire_cost_per_ha
    elif action.name == "harvest_replant":
        reward += revenue_for_harvest(self.s.volume_m3, self.cfg.economics)
        # reset stand after clearcut and replant
        self.s = replace(self.s, volume_m3=0.0, basal_area_m2=1.5, trees_per_ha=1600.0, age=0.0)
        reward -= self.cfg.economics.replant_cost_per_ha

    # disturbances
    self.s, lost, ev = self.dist_engine.step(self.s)
    if ev is not None and lost > 0.0:
        reward += salvage_value(lost, self.cfg.economics)
    info["disturbance"] = {"type": ev, "lost_m3": lost}

    # growth
    self.s = step_growth(self.s, self.dt, self.cfg.growth)

    # time and discount
    self.s = advance_time(self.s, self.dt)
    t_years = self.step_idx * self.dt
    self.s = replace(self.s, cash_account=npv_update_cash(self.s.cash_account, reward, t_years, self.cfg.economics))

    self.step_idx += 1
    if self.step_idx >= self.max_steps:
        done = True


    obs = np.array(list(observe(self.s).values()), dtype=np.float32)
    # episodic return in undiscounted space for RL; economic discounting tracked in cash_account
    return obs, reward, done, False, info


def render(self):
    if self.s is None:
        return
    print(f"t={self.s.t} age={self.s.age:.1f} V={self.s.volume_m3:.1f} BA={self.s.basal_area_m2:.1f} cash={self.s.cash_account:.1f}")

# expose EnvConfig for import from package root
EnvConfig = EnvConfig