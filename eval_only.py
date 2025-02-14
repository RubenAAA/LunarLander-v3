import gymnasium as gym
import time
# imports
import time
import torch
import gymnasium as gym
from stable_baselines3 import PPO


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
    evaluate("final.zip")