# just hiding some warnings
import os
import logging
import warnings
warnings.filterwarnings("ignore", message="You are trying to run PPO on the GPU, but it is primarily intended to run on the CPU")
os.environ["TF_TRT_ALLOW_GPU_FALLBACK"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# imports
import time
import json
import torch
import random
import numpy as np
import multiprocessing
import gymnasium as gym
from torch.optim import AdamW
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback


MODEL_PATH = "final"  # Global model save path

class AdjustHyperparamsCallback(BaseCallback):
    """
    A custom callback for dynamically adjusting reinforcement learning hyperparameters based on recent training performance.

    This callback monitors episode lengths and rewards, and modifies hyperparameters accordingly to optimize learning:
    - If episode lengths are too high but rewards are good, it increases exploration (`ent_coef`) and adjusts learning rate.
    - If episode lengths stabilize, it fine-tunes hyperparameters based on rewards.
    - If rewards drop significantly, it resets hyperparameters to default values.

    Hyperparameters managed:
    - `ent_coef` (Entropy coefficient): Controls exploration.
    - `vf_coef` (Value function coefficient): Balances policy and value loss.
    - `n_epochs` (Number of training epochs): Defines how often each batch is trained.
    - `lr` (Learning rate): Adjusted dynamically for better convergence.
    - `threshold` (Episode length threshold): Used to decide when to modify hyperparameters.

    The updated hyperparameters are saved to a JSON file (`hyperparams.json`) to maintain consistency across training runs.

    Attributes:
        env (VecEnv): The training environment (vectorized).
        percentage (float): Fraction of max episode steps used as threshold for modification.
        window_size (int): Number of recent episodes to consider for statistics.
        save_path (str): Path to save hyperparameters.
        modified_length (bool): Tracks if episode length-based modifications were applied.
        modified_reward (bool): Tracks if reward-based modifications were applied.
        ent_coef (float): Entropy coefficient for exploration.
        vf_coef (float): Value function coefficient.
        n_epochs (int): Number of training epochs.
        lr (float): Learning rate.

    Methods:
        save_hyperparams():
            Saves the updated hyperparameters to a JSON file.

        load_hyperparams():
            Loads hyperparameters from a JSON file if it exists.

        _on_step():
            Evaluates recent training performance and adjusts hyperparameters dynamically.
    """
    def __init__(self, env, percentage=1, window_size=100, verbose=0, save_path="hyperparams.json"):
        super().__init__(verbose)
        self.window_size = window_size  # Moving average window
        self.save_path = save_path

        # Compute a global threshold based on the average max_episode_steps across envs
        max_steps = np.mean([e.spec.max_episode_steps for e in env.envs])
        self.threshold = int(max_steps * percentage)

        # Global flags to track if modifications have been applied
        self.modified_length = False
        self.modified_reward = False

        # Default global hyperparameters
        self.ent_coef = 0.01
        # self.gamma = 0.999
        self.vf_coef = 0.5
        self.n_epochs = 8
        self.lr = 0.01

        # Load previous hyperparameters if available
        self.load_hyperparams()

    def save_hyperparams(self):
        """Save modified hyperparameters to a JSON file."""
        hyperparams = {
            "ent_coef": self.ent_coef,
            # "gamma": self.gamma,
            "vf_coef": self.vf_coef,
            "n_epochs": self.n_epochs,
            "modified_length": self.modified_length,
            "modified_reward": self.modified_reward,
            "threshold": self.threshold,
            "lr": self.lr
        }
        with open(self.save_path, "w") as f:
            json.dump(hyperparams, f)
    
    def load_hyperparams(self):
        """Load modified hyperparameters from a JSON file if it exists."""
        if os.path.exists(self.save_path):
            with open(self.save_path, "r") as f:
                hyperparams = json.load(f)
                self.ent_coef = hyperparams.get("ent_coef", self.ent_coef)
                # self.gamma = hyperparams.get("gamma", self.gamma)
                self.vf_coef = hyperparams.get("vf_coef", self.vf_coef)
                self.n_epochs = hyperparams.get("n_epochs", self.n_epochs)
                self.modified_length = hyperparams.get("modified_length", self.modified_length)
                self.modified_reward = hyperparams.get("modified_reward", self.modified_reward)
                self.threshold = hyperparams.get("threshold", self.threshold)
                self.lr = hyperparams.get("lr", self.lr)
                print("✅ Loaded previous hyperparameters from file.")

    def _on_step(self) -> bool:
        """
        Dynamically adjusts hyperparameters based on recent training performance.

        This method is executed at each training step and analyzes recent episode statistics (length and reward)
        to modify key hyperparameters adaptively. The goal is to:
        1. Increase exploration if episodes are too long but rewards are high.
        2. Fine-tune learning rate and optimization parameters based on reward trends.
        3. Reset hyperparameters if performance drops.

        Hyperparameters adjusted:
        - `ent_coef`: Entropy coefficient (controls exploration).
        - `vf_coef`: Value function coefficient.
        - `n_epochs`: Number of training epochs.
        - `lr`: Learning rate.
        - `threshold`: Adjusted based on training performance.

        Returns:
            bool: `True` to continue training, `False` to stop the current `.learn()` call if major adjustments occur.
        """

        lr_min, lr_max = 0.0001, 0.01  # Define learning rate range

        # Ensure there is at least one episode in the buffer.
        if self.model.ep_info_buffer:

            # Compute global statistics over the most recent episodes.
            recent_eps = list(self.model.ep_info_buffer)[-self.window_size:]
            recent_ep_lengths = [ep["l"] for ep in recent_eps]
            recent_ep_rewards = [ep["r"] for ep in recent_eps]

            smoothed_ep_len_mean = np.mean(recent_ep_lengths)
            smoothed_ep_rew_mean = np.mean(recent_ep_rewards)

            # ------------------------------------------------------------
            # 1. If the episode length is too high and rewards are high, 
            #    increase exploration and adjust learning rate.
            # ------------------------------------------------------------
            if smoothed_ep_len_mean >= self.threshold and smoothed_ep_rew_mean > 120 and not self.modified_length:
                self.ent_coef = 0.015  # Increase entropy to encourage exploration
                # self.gamma = 0.99  # Focus on short-term rewards
                
                # Adjust learning rate based on reward scaling
                self.lr = lr_min + (lr_max - lr_min) * np.log1p((315 - smoothed_ep_rew_mean) / 315) / np.log1p(1)

                self.modified_length = True  # Mark that modifications were applied

                if self.verbose > 0:
                    print(f"🔄 High episode length (mean: {smoothed_ep_len_mean:.2f}) detected.")
                    print(f"    Updating hyperparams: ent_coef={self.ent_coef}, gamma={self.gamma}")

                self.save_hyperparams()
                return False  # Stop training for adjustments to take effect

            # ------------------------------------------------------------
            # 2. If episode length is below the threshold and adjustments 
            #    were previously made, fine-tune based on reward trends.
            # ------------------------------------------------------------
            elif smoothed_ep_len_mean < self.threshold and self.modified_length:
                self.lr = lr_min + (lr_max - lr_min) * np.log1p((315 - smoothed_ep_rew_mean) / 315) / np.log1p(1)

                # If rewards are high, refine hyperparameters to improve performance
                if smoothed_ep_rew_mean >= 200:
                    min_ent_coef, max_ent_coef = 0.001, 0.01
                    # min_gamma, max_gamma = 0.999, 0.9999
                    min_vf_coef, max_vf_coef = 0.5, 1.0
                    min_n_epochs, max_n_epochs = 8, 28

                    # Normalize reward between 0 and 1
                    normalized_reward = max(0, min((smoothed_ep_rew_mean - 200) / (315 - 200), 1))
                    self.ent_coef = max_ent_coef - (max_ent_coef - min_ent_coef) * normalized_reward
                    # self.gamma = min_gamma + (max_gamma - min_gamma) * normalized_reward
                    self.vf_coef = min_vf_coef + (max_vf_coef - min_vf_coef) * normalized_reward
                    self.n_epochs = int(min_n_epochs + (max_n_epochs - min_n_epochs) * normalized_reward)

                    # Gradually decrease the threshold
                    self.threshold = max(self.threshold - int(3 * normalized_reward), 130)

                    self.modified_reward = True  # Mark that reward-based modifications were applied

                    if self.verbose > 0:
                        print(f"🔄 Adjusted hyperparams based on high reward (mean: {smoothed_ep_rew_mean:.2f})")
                        # print(f"    ent_coef: {self.ent_coef:.5f}, gamma: {self.gamma:.6f}, vf_coef: {self.vf_coef:.2f}")

                # ------------------------------------------------------------
                # 3. If rewards drop significantly, reset hyperparameters.
                # ------------------------------------------------------------
                elif smoothed_ep_rew_mean < 175 and self.modified_reward:
                    self.ent_coef = 0.01  # Reset entropy coefficient
                    # self.gamma = 0.999  # Reset gamma
                    self.vf_coef = 0.5  # Reset value function coefficient
                    self.n_epochs = 8  # Reset epochs to default
                    self.modified_reward = False  # Reset flag

                    if self.verbose > 0:
                        print("🔄 Reset hyperparameters due to reward drop.")

            # ------------------------------------------------------------
            # Apply the final hyperparameter changes to the model.
            # ------------------------------------------------------------
            self.model.ent_coef = self.ent_coef
            # self.model.gamma = self.gamma
            self.model.vf_coef = self.vf_coef
            self.model.n_epochs = self.n_epochs

            # Save updated hyperparameters to JSON
            self.save_hyperparams()

        return True  # Continue training


def train_with_ppo(total_timesteps=3_000_000):
    """
    Train a PPO (Proximal Policy Optimization) agent on the LunarLander-v3 environment.

    This function initializes or loads a PPO model and continuously trains it while adjusting 
    hyperparameters dynamically. The model is periodically restarted when the `AdjustHyperparamsCallback` 
    signals that adjustments are necessary.

    Key Features:
    - Uses **Stable-Baselines3 PPO** with a custom neural network architecture.
    - Supports **dynamic hyperparameter tuning** via the `AdjustHyperparamsCallback`.
    - Implements **periodic model saving** and evaluation in a separate process.
    - Trains on **multiple environments in parallel (64 environments)** for faster learning.
    - Uses **adaptive exploration strategies** to optimize agent performance.

    Args:
        total_timesteps (int): The total number of training timesteps before stopping.
    """

    # Number of parallel environments to use for training
    n_envs = 64  

    # Flag to track if reward-based modifications have been applied
    rew_mod_anytime = False  

    # Define the policy architecture and optimizer settings
    policy_kwargs = dict(
        optimizer_class=AdamW,  # Use AdamW optimizer for better weight decay handling
        optimizer_kwargs=dict(weight_decay=1e-2),  # Weight decay for regularization
        net_arch=dict(pi=[128, 64], vf=[128, 64])  # Define network structure for policy and value functions
    )

    # Create a vectorized environment with a fixed episode length of 250 steps
    env = make_vec_env("LunarLander-v3", n_envs=n_envs, env_kwargs={"max_episode_steps": 250})

    # Set a fixed random seed for reproducibility
    env.seed(SEED)

    # Select the computation device: Use GPU if available, otherwise fall back to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------------------------
    # Load an existing model if available; otherwise, train from scratch
    # -----------------------------------------------
    if os.path.exists(MODEL_PATH + ".zip"):
        print(f"Loading existing model from {MODEL_PATH}...")

        # Load hyperparameters from a JSON file (if previously saved)
        with open("hyperparams.json", "r") as f:
            data = json.load(f)

        lr_value = data.get("lr")  # Extract the stored learning rate

        # Load the pretrained PPO model with the environment and existing hyperparameters
        model = PPO.load(
            MODEL_PATH,
            env=env,
            device=device,
            learning_rate=lr_value,  # Attempt to apply the saved learning rate
            custom_objects=data,  # Load other hyperparameters dynamically
            policy_kwargs=policy_kwargs
        )

        # Manually update the optimizer's learning rate (alternative approach)
        for param_group in model.policy.optimizer.param_groups:
            param_group["lr"] = lr_value

        # Get the current number of timesteps already trained
        current_timesteps = model.num_timesteps

    else:
        print("No existing model found. Training from scratch...")

        # Initialize a new PPO model with predefined hyperparameters
        model = PPO(
            policy="MlpPolicy",  # Use a Multi-Layer Perceptron (MLP) policy
            env=env,
            learning_rate=lambda _: 0.01,  # Initial learning rate (adjustable later)
            n_steps=512,  # Number of steps per environment before updating
            batch_size=4096,  # Batch size for training updates
            n_epochs=8,  # Number of passes over each batch during training
            gamma=0.999,  # Discount factor for future rewards
            gae_lambda=0.98,  # Generalized Advantage Estimation lambda
            clip_range=lambda _: 0.2,  # Clipping range for PPO objective
            ent_coef=0.01,  # Entropy coefficient (controls exploration)
            vf_coef=0.5,  # Value function coefficient (importance of critic loss)
            max_grad_norm=0.5,  # Gradient clipping for stability
            use_sde=False,  # Whether to use State-Dependent Exploration (SDE)
            normalize_advantage=True,  # Normalize advantages for stability
            verbose=1,  # Print training logs
            device=device,  # Set computation device (GPU/CPU)
            tensorboard_log="./final_tensorboard",  # Path for TensorBoard logs
            policy_kwargs=policy_kwargs  # Apply custom network architecture
        )

        # Since training starts fresh, initialize current timestep count to 0
        current_timesteps = 0

    # -----------------------------------------------
    # Initialize the hyperparameter tuning callback
    # -----------------------------------------------
    callback = AdjustHyperparamsCallback(
        env, percentage=0.999, window_size=n_envs * 512 * 2, verbose=0
    )

    # -----------------------------------------------
    # Main training loop with periodic evaluation
    # -----------------------------------------------
    eval_interval = 500_000  # Evaluate the model every 500,000 steps

    while current_timesteps < total_timesteps:
        # Determine how many steps to train in this cycle
        train_step = min(eval_interval, total_timesteps - current_timesteps)

        # Retrieve updated hyperparameters from the callback
        new_hyperparams = {
            "ent_coef": callback.ent_coef,
            # "gamma": callback.gamma,
            "vf_coef": callback.vf_coef,
            "n_epochs": callback.n_epochs,
            "threshold": callback.threshold,
            "modified_length": callback.modified_length,
            "modified_reward": callback.modified_reward,
            "lr": callback.lr,      
        }

        # Restart training with updated hyperparameters if needed
        if os.path.exists(MODEL_PATH + ".zip"):
            # Track whether rewards have been modified anytime during training
            rew_mod_anytime = rew_mod_anytime or callback.modified_reward

            print(f"Restarting PPO with new hyperparameters: {new_hyperparams}")

            if rew_mod_anytime:
                # Reload the model with the new hyperparameters if reward-based modifications were applied
                model = PPO.load(
                    MODEL_PATH,
                    env=env,
                    device=device,
                    learning_rate=callback.lr,  # Apply the new learning rate
                    custom_objects=new_hyperparams,  # Load updated hyperparams
                    policy_kwargs=policy_kwargs
                )
                for param_group in model.policy.optimizer.param_groups:
                    param_group["lr"] = callback.lr
            else:
                # Reload the model with non-reward-modified hyperparameters
                model = PPO.load(
                    MODEL_PATH,
                    env=env,
                    device=device,
                    custom_objects=new_hyperparams,
                    policy_kwargs=policy_kwargs
                )

        # -----------------------------------------------
        # Train the model for the current step interval
        # -----------------------------------------------
        model.learn(total_timesteps=train_step, reset_num_timesteps=False, callback=callback)
        current_timesteps += train_step  # Update the total timesteps count

        # -----------------------------------------------
        # Save the model after training iteration
        # -----------------------------------------------
        model._logger = None  # Remove logger to avoid pickling issues
        model.save(MODEL_PATH)  # Save the trained model

        # -----------------------------------------------
        # Launch evaluation in a separate process
        # -----------------------------------------------
        eval_process = multiprocessing.Process(target=evaluate, args=(MODEL_PATH,))
        eval_process.start()  # Run evaluation asynchronously

    # Close the environment after training is complete
    env.close()


def evaluate(model_path, episodes=3):
    """
    Evaluates a trained PPO model on the LunarLander-v3 environment.

    This function loads a pretrained model, runs it in the environment for a specified 
    number of episodes, and renders the gameplay while tracking the total reward and steps.

    Features:
    - Uses `render_mode="human"` to visualize the agent's behavior.
    - Loads the model from the specified `model_path` and selects GPU (`cuda`) if available.
    - Runs multiple episodes, tracking rewards and number of steps per episode.
    - Enforces a maximum step limit of 250 per episode for stability.
    - Introduces a small time delay (`0.01s`) between steps to make visualization smoother.

    Args:
        model_path (str): Path to the saved PPO model file.
        episodes (int): Number of episodes to evaluate. Default is 3.
    """

    # Create the environment with rendering enabled for visualization.
    env = gym.make("LunarLander-v3", render_mode="human")

    # Load the trained PPO model with the environment attached.
    model = PPO.load(model_path, env=env, device="cuda" if torch.cuda.is_available() else "cpu")

    for ep in range(episodes):
        # Reset the environment for a new episode.
        obs, info = env.reset()
        done = False
        truncated = False
        ep_reward = 0  # Track total episode reward
        steps = 0  # Track number of steps in the episode

        while not (done or truncated):
            env.render()  # Render the environment for visualization
            
            # Select action from the trained model (deterministic mode for evaluation)
            action, _ = model.predict(obs, deterministic=True)
            
            # Execute the action in the environment and receive the next state and reward.
            obs, reward, done, truncated, info = env.step(action)
            
            ep_reward += reward  # Accumulate total reward
            steps += 1  # Count steps

            # Enforce a maximum step limit per episode to prevent infinite loops
            if steps == 250:
                done = True

            time.sleep(0.01)  # Small delay for smoother rendering

        print(f"Episode {ep+1} reward: {ep_reward:.4f}, Steps= {steps}")

    env.close()  # Close the environment to free resources


if __name__ == "__main__":
    # Trying to add some reproducibility
    SEED = 1337
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    
    # Trying to force it to use GPU as much as it can
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.set_float32_matmul_precision("high")
    torch.cuda.empty_cache()  # Clear unused memory
    torch.cuda.set_per_process_memory_fraction(0.95)  # Adjust GPU memory fraction as needed

    # Required for CUDA when we show the eval video while continuing to train in the background
    multiprocessing.set_start_method("spawn")

    # Actual training
    print("Training with PPO...")
    train_with_ppo(total_timesteps=20_000_000)
