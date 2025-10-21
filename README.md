# StochasticStandSim

StochasticStandSim provides a simulation environment for exploring
forest stand management strategies with deterministic and stochastic
disturbance processes.

## Installation

1. Ensure you are using Python 3.12 or newer.
2. (Optional) Create and activate a virtual environment.
3. Install the project dependencies:

   ```bash
   pip install -e .
   ```


## TODO
What to implement next (concrete)

Reward adapter that maps your economics (revenues, costs, salvage, sell) to scalar rt with discount γ.

Action space as a hybrid head (categorical for choice; optional Gaussian head for continuous params like TPA).

PPO with GAE and observation normalization.

```python
env = ForestEnv(...)
policy = ActorCritic(...)
opt_pi, opt_v = Adam(policy.actor), Adam(policy.critic)

for iter in range(K):
    traj = collect_rollouts(env, policy, T, gamma, lam=0.95)  # states, actions, log_probs, rewards, values
    adv, ret = compute_gae(traj.rewards, traj.values, gamma, lam=0.95)
    for epoch in range(E):
        for batch in minibatches(traj, adv, ret):
            loss_pi = ppo_clipped_loss(batch)
            loss_v  = mse(policy.V(batch.s), batch.returns)
            entropy  = policy.entropy(batch.s).mean()
            total = loss_pi + c_v*loss_v - c_ent*entropy
            opt_pi.zero_grad(); opt_v.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            opt_pi.step(); opt_v.step()
```

Eval harness with fixed scenario seeds + CSV logs of actions/returns/harvest ages.

## What does each step do?
At a high level, step() only does nine things, in this order:

1. Read inputs: Grab s0 = self.state and unpack the action vector.
2. Sanitize action: Clip to valid ranges and coerce booleans (no fancy adapters).
3. Growth transition (no money here): Call the growth model with the sanitized action (e.g., thinning %, fert%, planting).
4. Disturbance overlay (still no money): Ask the disturbance manager to sample/apply events (fire/wind) and update the state.
5. Economics for this step: Pass (s0, action, s1, t) to the economics model to compute cash flow: revenues (harvest/salvage), fert cost, treatment costs, etc. (No discounting here; RL/planning handles γ.)
6. Termination check: Decide terminated (e.g., SELL/EXIT action, max age, catastrophic rule) and truncated (time-horizon cutoff).
7. Advance env: self.t += 1, self.state = s1, set self.done = terminated or truncated.
8. Assemble info (without overwriting): Merge {"growth": …, "disturbance": …, "economics": …, "action": …, "time": …}.
9. Return: Produce obs (or raw state if that’s your contract) and return (obs, reward, terminated, truncated, info).