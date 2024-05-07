import numpy as np
import gymnasium as gym

def policy_iteration(env, gamma=0.99, theta=1e-5):
    def one_step_lookahead(state, V, gamma):
        action_values = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            # Reset the environment to the current state
            env.reset()
            next_state, reward, done, truncated, info = env.step(action)
            action_values[action] = reward + gamma * V[next_state] * (not done and not truncated)
        return action_values

    V = np.zeros(env.observation_space.n)
    policy = np.zeros(env.observation_space.n, dtype=int)

    while True:
        # Policy Evaluation
        delta = 0
        for state in range(env.observation_space.n):
            v = V[state]
            V[state] = np.max(one_step_lookahead(state, V, gamma))
            delta = max(delta, np.abs(v - V[state]))
            if delta < theta:
                break

        # Policy Improvement
        policy_stable = True
        for state in range(env.observation_space.n):
            old_action = policy[state]
            action_values = one_step_lookahead(state, V, gamma)
            new_action = np.argmax(action_values)
            policy[state] = new_action
            if old_action != new_action:
                policy_stable = False

        if policy_stable:
            break

    return policy, V

if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=False)
    policy, value = policy_iteration(env)
    print("Optimal Policy:", policy)
    print("Value Function:", value)
