import gymnasium as gym
import multiprocessing
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import torch
import os
import time
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import math


class AdjustHyperparamsPerEnvCallback(BaseCallback):
    def __init__(self, env, percentage=1, window_size=100, verbose=0):
        super().__init__(verbose)
        self.n_envs = env.num_envs  # Number of parallel environments
        self.window_size = window_size  # Moving average window

        # Get max_episode_steps **per environment**
        self.max_episode_steps_per_env = [e.spec.max_episode_steps for e in env.envs]
        self.threshold_per_env = [int(steps * percentage) for steps in self.max_episode_steps_per_env]

        # Track per-environment hyperparameters
        self.modified_length = [False] * self.n_envs  # Track if length condition was modified
        self.modified_reward = [False] * self.n_envs  # Track if reward condition was modified
        self.ent_coef_per_env = [0.01] * self.n_envs  # Default ent_coef
        self.gamma_per_env = [0.999] * self.n_envs  # Default gamma
        self.vf_coef_per_env = [0.5] * self.n_envs  # Default vf_coef
        self.n_steps_per_env = [512] * self.n_envs  # Default n_steps
        self.n_epochs_per_env = [8] * self.n_envs  # Default n_epochs

    def _on_step(self) -> bool:
        if self.model.ep_info_buffer:
            for i in range(self.n_envs):  # Loop over each environment
                recent_eps = list(self.model.ep_info_buffer)[-self.window_size:]
                recent_ep_lengths = [ep["l"] for ep in recent_eps]
                recent_ep_rewards = [ep["r"] for ep in recent_eps]

                smoothed_ep_len_mean = np.mean(recent_ep_lengths)
                smoothed_ep_rew_mean = np.mean(recent_ep_rewards)

                ## 🔹 1. Adjust for **high episode length** → Increase exploration
                if smoothed_ep_len_mean >= self.threshold_per_env[i] and not self.modified_length[i]:
                    self.ent_coef_per_env[i] = 0.02  # More exploration
                    self.gamma_per_env[i] = 0.95  # Shorter-term focus
                    self.n_steps_per_env[i] = 512  # Faster updates
                    self.n_epochs_per_env[i] = 8  # Less training per batch
                    self.modified_length[i] = True  # Mark as modified
                    if self.verbose > 0:
                        print(f"🔄 Env {i}: Increased ent_coef to {self.ent_coef_per_env[i]}, gamma to {self.gamma_per_env[i]}, n_steps={self.n_steps_per_env[i]}, n_epochs={self.n_epochs_per_env[i]} at ep_len_mean = {smoothed_ep_len_mean}")

                elif smoothed_ep_len_mean < self.threshold_per_env[i] and self.modified_length[i]:
                    ## 🔹 2. Adjust for **high rewards** → Less exploration, more weight on rewards

                    if smoothed_ep_rew_mean > 275:
                        self.ent_coef_per_env[i] = 0.002  # Very low exploration
                        self.gamma_per_env[i] = 0.9995  # Almost deterministic long-term planning
                        self.vf_coef_per_env[i] = 0.9  # Maximize value function weight
                        self.n_steps_per_env[i] = 2048  # Larger batch for more stable updates
                        self.n_epochs_per_env[i] = 32  # More training per batch
                        self.threshold_per_env[i] = max(self.threshold_per_env[i]-3, 150)
                        if self.verbose > 0:
                            print(f"🔥 Env {i}: High reward! Updated n_steps={self.n_steps_per_env[i]}, n_epochs={self.n_epochs_per_env[i]}.")

                    elif smoothed_ep_rew_mean > 250:
                        self.ent_coef_per_env[i] = 0.003  # Even lower exploration
                        self.gamma_per_env[i] = 0.9993  # Stronger focus on long-term rewards
                        self.vf_coef_per_env[i] = 0.85  # Increase importance of value function
                        self.n_steps_per_env[i] = 1536  # Medium batch size
                        self.n_epochs_per_env[i] = 18
                        self.threshold_per_env[i] = max(self.threshold_per_env[i]-2, 175)
                        if self.verbose > 0:
                            print(f"🔴 Env {i}: Reward > 250, Updated n_steps={self.n_steps_per_env[i]}, n_epochs={self.n_epochs_per_env[i]}.")

                    elif smoothed_ep_rew_mean > 225:
                        self.ent_coef_per_env[i] = 0.004  # Reduce exploration
                        self.gamma_per_env[i] = 0.9992
                        self.vf_coef_per_env[i] = 0.8
                        self.n_steps_per_env[i] = 1024  # Smaller batch for faster updates
                        self.n_epochs_per_env[i] = 16
                        self.threshold_per_env[i] = max(self.threshold_per_env[i]-2, 200)
                        if self.verbose > 0:
                            print(f"🟠 Env {i}: Reward > 225, Updated n_steps={self.n_steps_per_env[i]}, n_epochs={self.n_epochs_per_env[i]}.")

                    elif smoothed_ep_rew_mean > 200:
                        self.ent_coef_per_env[i] = 0.005  # Moderate exploration reduction
                        self.gamma_per_env[i] = 0.9991  # Keep long-term rewards
                        self.vf_coef_per_env[i] = 0.75  # Shift more weight to value function
                        self.n_steps_per_env[i] = 768  # Smaller batch, frequent updates
                        self.n_epochs_per_env[i] = 12
                        self.threshold_per_env[i] = max(self.threshold_per_env[i]-1, 215)
                        if self.verbose > 0:
                            print(f"🟡 Env {i}: Reward > 200, Updated n_steps={self.n_steps_per_env[i]}, n_epochs={self.n_epochs_per_env[i]}.")

                    elif smoothed_ep_rew_mean > 175:
                        self.ent_coef_per_env[i] = 0.005  # Moderate exploration reduction
                        self.gamma_per_env[i] = 0.999  # Keep long-term rewards
                        self.vf_coef_per_env[i] = 0.7  # Shift more weight to value function
                        self.n_steps_per_env[i] = 512  # Smaller batch, quick updates
                        self.n_epochs_per_env[i] = 10
                        self.threshold_per_env[i] = max(self.threshold_per_env[i]-1, 230)
                        if self.verbose > 0:
                            print(f"🟡 Env {i}: Reward > 175, Updated n_steps={self.n_steps_per_env[i]}, n_epochs={self.n_epochs_per_env[i]}.")

                    elif smoothed_ep_rew_mean <= 175 and self.modified_reward[i]:
                        # Reset everything if reward drops
                        self.ent_coef_per_env[i] = 0.01
                        self.gamma_per_env[i] = 0.999
                        self.vf_coef_per_env[i] = 0.5
                        self.n_steps_per_env[i] = 512
                        self.n_epochs_per_env[i] = 8
                        self.modified_reward[i] = False
                        if self.verbose > 0:
                            print(f"🔄 Env {i}: Reward dropped, resetting hyperparameters.")

        return True


def train_with_ppo(total_timesteps=2_000_000):
    """
    Train a PPO (Proximal Policy Optimization) agent on LunarLander-v3.
    """
    n_envs = 64
    env = make_vec_env("LunarLander-v3", n_envs=n_envs, env_kwargs={"max_episode_steps": 225})
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load or create model
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}...")
        model = PPO.load(MODEL_PATH, env=env, device=device)  # Load model
        current_timesteps = model.num_timesteps  # Keep track of total steps trained
    else:
        print("No existing model found. Training from scratch...")
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=lambda _: 0.01,
            n_steps=512,
            batch_size=4096,
            n_epochs=8,
            gamma=0.999,
            gae_lambda=0.98,
            clip_range=lambda _: 0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_sde=False,
            normalize_advantage=True,
            policy_kwargs=dict(net_arch=dict(pi=[128, 64], vf=[128, 64])),
            verbose=1,
            device=device,
            tensorboard_log="./final_tensorboard"
            # type the following into the command prompt to visualize the learning progress
            # tensorboard --logdir=final_tensorboard --reload_interval=5
        )
        current_timesteps = 0  # Start from zero
    
    callback = AdjustHyperparamsPerEnvCallback(env, percentage=0.99, window_size=100, verbose=1)
  
    # Track progress in increments of 
    eval_interval = total_timesteps // 10
    for _ in range(total_timesteps // eval_interval):
        progress_remaining = 1 - (current_timesteps / total_timesteps)

        # Manually update learning rate & clip range dynamically
        model.learning_rate = lambda progress_remaining: max(0.01 * (0.5 * (1 + math.cos(math.pi * progress_remaining))), 0.0003)

        model.clip_range = lambda progress_remaining: max(0.3 * progress_remaining, 0.2)


        train_step = min(eval_interval, total_timesteps - current_timesteps)
        model.learn(total_timesteps=eval_interval, reset_num_timesteps=False, callback=callback)
        current_timesteps += train_step

        
        # Save after each interval
        model.save(MODEL_PATH)
        
        # Evaluate the model in parallel
        # eval_process = multiprocessing.Process(target=evaluate, args=(MODEL_PATH,))
        # eval_process.start()

    env.close()


def evaluate(model_path, episodes=4):
    # Create environment in "human" render mode if you want to see the lander
    env = gym.make("LunarLander-v3", render_mode="human")

    model = PPO.load(model_path, env=env, device="cuda" if torch.cuda.is_available() else "cpu")

    for ep in range(episodes):
        obs, info = env.reset()
        done = True # False  # Change according to whether you want the animation
        truncated = False
        ep_reward = 0
        steps = 0 
        
        while not (done or truncated):
            # Render
            env.render()

            # Predict action
            action, _ = model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1
            
            if steps == 300:  # We don't want to visualize too long
                done = True

            # Slow down the steps so you can visually follow
            time.sleep(0.01)

        print(f"Episode {ep+1} reward: {ep_reward:.4f}, Steps= {steps}")

    env.close()

if __name__ == "__main__":
    os.environ["TF_TRT_ALLOW_GPU_FALLBACK"] = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.set_float32_matmul_precision("high")
    multiprocessing.set_start_method("spawn")  # 🔹 Required for CUDA

    torch.cuda.empty_cache()  # Clears unused memory
    torch.cuda.set_per_process_memory_fraction(0.95)  # Use 90% of the GPU memory
    # Define model path, if one already exist, we continue training
    MODEL_PATH = "final.zip"
    print("Training with PPO...")
    train_with_ppo(total_timesteps=5_000_000)
