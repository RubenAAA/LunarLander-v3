# 🚀 Reinforcement Learning with PPO on LunarLander-v3

## 📝 Project Overview
This project implements **Proximal Policy Optimization (PPO)** to train an agent in the **LunarLander-v3** environment using **Stable-Baselines3**. The training process leverages **dynamic hyperparameter tuning**, parallel environments, and periodic model evaluations to optimize performance.

## 🎯 Features
- **Custom Adaptive Hyperparameter Tuning**  
  - Adjusts **entropy coefficient**, **learning rate**, and **training epochs** in response to training conditions.
  - Uses a **custom callback** (`AdjustHyperparamsCallback`) to modify hyperparameters on the fly.

- **Efficient Parallel Training**  
  - Runs **64 environments in parallel** to speed up learning.
  - Optimized for **CUDA** acceleration with `torch`.

- **Periodic Model Checkpointing & Evaluation**  
  - Saves the trained model at intervals.
  - Evaluates performance in a separate **multiprocessing** process.

- **Multiprocessing Support in Jupyter Notebooks**  
  - Works with `"spawn"` to avoid CUDA multiprocessing issues.
  - Ensures compatibility with notebooks and Python scripts.

## 🛠️ Installation
First, clone the repository:
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```
## 📌 Install Dependencies
To install all required dependencies, run:
```bash
pip install -r requirements.txt
```

## 🚀 Running the Training
### 📌 Training from Scratch:
```bash
python
from train import train_with_ppo

train_with_ppo(total_timesteps=20_000_000)
```

### 📌 Running Evaluation:
```bash
python
from evaluate import evaluate

evaluate("trained_model.zip")
```

## 🖥️ Running in Jupyter Notebook
If using Jupyter Notebook, ensure **multiprocessing compatibility** by setting:
```bash
python
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)
```
Then, restart the kernel before running the training.

## 📂 Project Structure
```bash
📦 lunar_lander/
 ┣ 📜 __pycache__/         # Compiled Python files
 ┣ 📜 final_tensorboard/   # TensorBoard logs
 ┣ 📜 .gitattributes       # Git configuration file
 ┣ 📜 eval_only.py         # Script for evaluation-only purposes
 ┣ 📜 final_10m_timesteps_292.zip  # Model checkpoint file
 ┣ 📜 final_285.zip        # Model checkpoint file
 ┣ 📜 final_285.py         # Script associated with final_285.zip
 ┣ 📜 final_291.zip        # Model checkpoint file
 ┣ 📜 final_293.zip        # Model checkpoint file
 ┣ 📜 final_296.zip        # Model checkpoint file
 ┣ 📜 final.py             # Main training script
 ┣ 📜 final.zip            # Final model checkpoint
 ┣ 📜 hyperparams.json     # JSON file for hyperparameters
 ┣ 📜 pipeline.ipynb       # Jupyter Notebook for pipeline
 ┣ 📜 rapids-23.10.yml     # Environment configuration for RAPIDS
 ┣ 📜 README.md            # Project documentation
 ┣ 📜 requirements.txt     # Required dependencies for the project
```

## 📌 Hyperparameters Used
| Parameter     | Value    |
|--------------|---------|
| `policy`     | MlpPolicy |
| `n_steps`    | 512     |
| `batch_size` | 4096    |
| `n_epochs`   | 8       |
| `gamma`      | 0.999   |
| `gae_lambda` | 0.98    |
| `clip_range` | 0.2     |
| `ent_coef`   | 0.01    |
| `vf_coef`    | 0.5     |

## 📜 License
This project is **open-source** and available under the MIT License.

---

🔥 **Feel free to contribute, report issues, or suggest improvements!** 🔥
