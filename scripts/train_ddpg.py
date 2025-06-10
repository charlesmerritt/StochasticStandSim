import os
import sys
import datetime
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from env.ForestEnv import ForestStandEnv

# Timestamp for logging and saving
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
model_name = f"ddpg_forest_{timestamp}"
log_dir = os.path.join("logs", model_name)
os.makedirs("models", exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Create and vectorize the environment
env = ForestStandEnv()
env = DummyVecEnv([lambda: env])

# Initialize DDPG model
model = DDPG(
    "MlpPolicy",
    env,
    learning_rate=1e-3,
    buffer_size=1_000_000,
    batch_size=64,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    policy_kwargs=dict(net_arch=[128, 128]),
    verbose=1,
    tensorboard_log=log_dir,
    device="cpu"
)

# Train the model
print("Training DDPG agent on the forest stand environment...")
model.learn(total_timesteps=100_000)

# Save the model
model_path = os.path.join("models", model_name)
model.save(model_path)
print(f"Model saved to: {model_path}.zip")
print(f"TensorBoard logs saved to: {log_dir}")

# Evaluate the model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f}, Std reward: {std_reward:.2f}")

env.close()

# To view training curves:
# tensorboard --logdir logs/
