from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
import wandb
from wandb.integration.sb3 import WandbCallback
from env import ForestGrowthEnv
import gymnasium as gym
import os

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

if __name__ == '__main__':
    # Weights & Biases setup
    wandb.init(project="Optimal-Thinning-RL-PPO", entity="cem96047", name="PPO")
    config = {
        "learning_rate": 0.0003,
        "architecture": "MlpPolicy",
        "env_name": "CartPole-v1"
    }

    # env = gym.make("CartPole-v1", render_mode="human")
    env = gym.make("ForestGrowthEnv", render_mode="console")

    # check_env(env.unwrapped)

    model = PPO("MlpPolicy", env, verbose=1, device="cuda", tensorboard_log=log_dir)
    # model.learn(total_timesteps=5000, progress_bar=True, callback=WandbCallback(
    #     gradient_save_freq=1000,
    #     model_save_path=f"{wandb.run.dir}/model",
    #     verbose=2,
    # ))
    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(100):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render()

    env.close()
    wandb.finish()
