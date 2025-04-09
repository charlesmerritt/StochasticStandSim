import numpy as np
import matplotlib.pyplot as plt
from env.ForestEnv import ForestStandEnv

def plot_state(state, step):
    labels = [
        "Age", "Biomass", "Density", "Carbon", "Species",
        "Moisture", "Fire Risk", "Pest", "Windthrow", "Value"
    ]
    plt.figure(figsize=(12, 4))
    plt.bar(labels, state)
    plt.title(f"Forest State at Step {step}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def get_action_input():
    try:
        print("\nEnter thinning %, nitrogen %, phosphorus % (e.g., 0.1 0.5 0.3):")
        user_input = input("> ").strip()
        action = np.array([float(x) for x in user_input.split()])
        if len(action) != 3 or not np.all((0 <= action) & (action <= 1)):
            raise ValueError
        return action
    except ValueError:
        print("Invalid input. Please enter three numbers between 0 and 1.")
        return get_action_input()

def main():
    env = ForestStandEnv()
    state, _ = env.reset()
    done = False
    step = 0

    while not done:
        print(f"\nStep {step} - Current Forest State:")
        env.render()
        plot_state(state, step)

        action = get_action_input()
        state, reward, done, _, _ = env.step(action)

        print(f"Reward received: {reward:.2f}")
        step += 1

        if done or step >= 100:
            print("Simulation complete.")
            break

if __name__ == "__main__":
    main()
