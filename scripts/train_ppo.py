import os
import sys
import torch
import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from env.ForestEnv import ForestStandEnv

# Timestamp for logging and saving
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
model_name = f"ppo_forest_{timestamp}"
log_dir = os.path.join("logs", model_name)
os.makedirs("models", exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Create and vectorize the environment
env = ForestStandEnv()
env = DummyVecEnv([lambda: env])  # SB3 requires vectorized envs for PPO

# Initialize PPO model
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    policy_kwargs=dict(net_arch=[128, 128]),
    verbose=1,
    tensorboard_log=log_dir
)

# Train the model
print("Training PPO agent on the forest stand environment...")
model.learn(total_timesteps=100_000)  # You can increase this

# Save the model
model_path = os.path.join("models", model_name)
model.save(model_path)
print(f"Model saved to: {model_path}.zip")
print(f"TensorBoard logs saved to: {log_dir}")

# Evaluate the model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f}, Std reward: {std_reward:.2f}")

env.close()

# 'tensorboard --logdir logs/' to view logs