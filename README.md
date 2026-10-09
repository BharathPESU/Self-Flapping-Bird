# 🐦 Self-Flapping Bird: Autonomous DQN Reinforcement Learning Agent

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-Environment-green)](https://gymnasium.farama.org/)
[![Managed with uv](https://img.shields.io/badge/Managed%20with-uv-DE5FE9?logo=astral&logoColor=white)](https://github.com/astral-sh/uv)
[![Hugging Face Hub](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Hub-yellow)](https://huggingface.co)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

An autonomous Reinforcement Learning agent capable of playing Flappy Bird with superhuman precision, trained via **Double Deep Q-Networks (DDQN)** on an 8-dimensional continuous kinematic observation space. Managed using **Astral `uv`**, equipped with automated **Hugging Face Hub** model publishing, and accompanied by a real-time interactive browser simulator.

---

## 👨‍💻 Developer Information

* **Developer:** **Bharath CD**
* **Academic Profile:** **3rd Year Undergraduate, PES University**
* **Domain:** Artificial Intelligence, Reinforcement Learning & Machine Learning Engineering

---

## 📖 Table of Contents
- [Project Overview](#-project-overview)
- [How It Works: Reinforcement Learning Architecture](#-how-it-works-reinforcement-learning-architecture)
  - [1. Kinematic State Space (8D Vector)](#1-kinematic-state-space-8d-vector)
  - [2. Action Space](#2-action-space)
  - [3. Reward Shaping Signal](#3-reward-shaping-signal)
  - [4. Double DQN & Bellman Optimization](#4-double-dqn--bellman-optimization)
- [Directory Structure](#-directory-structure)
- [Prerequisites & Dependency Installation](#-prerequisites--dependency-installation)
  - [Option A: Fast Setup with Astral `uv` (Recommended)](#option-a-fast-setup-with-astral-uv-recommended)
  - [Option B: Setup with standard `pip` / `venv`](#option-b-setup-with-standard-pip--venv)
  - [Option C: Running the React Web Simulator](#option-c-running-the-react-web-simulator)
- [How to Run the Project](#-how-to-run-the-project)
  - [1. Configure Environment Secrets](#1-configure-environment-secrets)
  - [2. Train the Agent](#2-train-the-agent)
  - [3. Watch the Agent Fly Live (GUI Evaluation)](#3-watch-the-agent-fly-live-gui-evaluation)
  - [4. Record Gameplay GIFs & MP4 Videos](#4-record-gameplay-gifs--mp4-videos)
  - [5. Publish Model & Artifacts to Hugging Face Hub](#5-publish-model--artifacts-to-hugging-face-hub)
- [Interactive Browser Simulator](#-interactive-browser-simulator)
- [License](#-license)

---

## 🚀 Project Overview

Flappy Bird presents a classic continuous kinematic challenge: navigating a dynamic corridor under constant gravity with high-penalty obstacle boundaries. 

Instead of processing raw screen pixels (which requires large Convolutional Neural Networks and days of GPU compute), this project extracts a **normalized continuous kinematic state representation**. This reduces sample complexity by over **99%**, allowing the Double DQN agent to achieve consistent navigation and solve the environment in **under 10–12 minutes** on a standard laptop CPU.

---

## 🧠 How It Works: Reinforcement Learning Architecture

The agent models the environment as a **Markov Decision Process (MDP)** denoted by $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$.

### 1. Kinematic State Space (8D Vector)
At every time-step $t$, the agent receives a normalized vector $\mathbf{s}_t \in \mathbb{R}^8$:

| Index | Feature | Description | Range |
| :---: | :--- | :--- | :---: |
| `s[0]` | `bird_y` | Normalized vertical altitude of the bird | $[0.0, 1.0]$ |
| `s[1]` | `bird_vel` | Normalized vertical velocity ($v_y$) | $[-1.0, 1.0]$ |
| `s[2]` | `dx_next_pipe` | Horizontal distance to upcoming pipe lip | $[0.0, 1.0]$ |
| `s[3]` | `top_gap_y` | Vertical position of the upper pipe boundary | $[0.0, 1.0]$ |
| `s[4]` | `bottom_gap_y` | Vertical position of the lower pipe boundary | $[0.0, 1.0]$ |
| `s[5]` | `gap_diff_y` | Offset from the bird to the target gap center | $[-1.0, 1.0]$ |
| `s[6]` | `dx_second_pipe` | Horizontal distance to the secondary pipe pair | $[0.0, 1.0]$ |
| `s[7]` | `second_gap_diff_y` | Offset to secondary pipe gap center (look-ahead) | $[-1.0, 1.0]$ |

### 2. Action Space
Discrete action space $\mathcal{A} = \{0, 1\}$:
- **`0` (Coast / Idle):** Applies standard downward acceleration ($g = 0.60$), allowing the bird to glide.
- **`1` (Flap):** Applies an instantaneous vertical impulse ($v_{\text{flap}} = -7.8$), launching the bird upward along a parabolic arc.

### 3. Reward Shaping Signal
To prevent the inefficiencies of sparse rewards ($+1$ only upon passing a pipe), the reward function $\mathcal{R}(s, a, s')$ is calibrated as:

$$\mathcal{R} = R_{\text{alive}} + R_{\text{pass}} + R_{\text{proximity}} + R_{\text{ceiling\_penalty}} + R_{\text{collision}}$$

* **Alive Baseline ($+0.15$):** Encourages survival.
* **Pipe Cleared ($+5.0$):** High milestone reward for successfully navigating an obstacle.
* **Gap Center Proximity ($+0.25 \times (1 - \frac{|y - y_{\text{center}}|}{\text{gap}/2}))$:** Continuous gradient steering the bird to fly through the safe middle corridor.
* **Anti-Ceiling Penalty ($-0.35$ if $y < 40$):** Penalizes hovering dangerously near the ceiling.
* **Collision Penalty ($-10.0$):** Severe penalty upon crashing into pipes, ground, or ceiling.

### 4. Double DQN & Bellman Optimization
* **Experience Replay Buffer:** Stores transitions $(s_t, a_t, r_t, s_{t+1}, d_t)$ in a circular memory buffer of 100,000 transitions. Uniform random mini-batch sampling decorrelates temporal samples.
* **Decoupled Double Q-Learning:** Eliminates the positive maximization bias of standard Q-learning by using the online network $\theta$ for action selection and the target network $\theta^-$ for value evaluation:

$$Y^{\text{Double-DQN}} = r_t + \gamma \, Q\Big(s_{t+1}, \arg\max_{a'} Q(s_{t+1}, a'; \theta); \, \theta^-\Big) (1 - d_t)$$

* **Polyak Soft Target Updates:** Updates the target network smoothly each step with $\tau = 0.005$:
$$\theta^- \leftarrow \tau \theta + (1 - \tau) \theta^-$$
* **Huber Loss (Smooth L1):** Prevents exploding gradients when dealing with high TD errors.

---

## 📁 Directory Structure

```text
flappy-bird-dqn/
├── .env                         # Hugging Face write token & configuration (secret)
├── .env.example                 # Example template for environment secrets
├── .gitignore                   # Git ignore file for venvs, models, and cache
├── pyproject.toml               # Modern Astral uv project specification
├── uv.lock                      # Deterministic locked dependency manifest
├── README.md                    # Project documentation
├── reinforcement_guide.md       # Comprehensive deep-dive tutorial and theoretical guide
├── app.py                       # Hugging Face Space interactive Gradio web application
└── src/
    ├── __init__.py
    ├── env/
    │   ├── __init__.py
    │   └── flappy_env.py        # Custom Gymnasium Flappy Bird environment
    ├── agent/
    │   ├── __init__.py
    │   ├── dqn_model.py         # PyTorch Multi-Layer Perceptron Q-Network
    │   ├── replay_buffer.py     # Circular experience replay memory buffer
    │   └── dqn_agent.py         # Double DQN agent with Polyak soft updates
    ├── train.py                 # Training pipeline with rolling metrics & plotting
    ├── evaluate.py              # Real-time visual Pygame playback script
    ├── record_video.py          # Headless GIF and MP4 recorder for portfolio demos
    └── push_to_hub.py           # Automated Hugging Face Hub uploader & Model Card generator
```

---

## 🛠️ Prerequisites & Dependency Installation

### Option A: Fast Setup with Astral `uv` (Recommended)

[Astral `uv`](https://github.com/astral-sh/uv) is a blazing-fast Python package and project manager written in Rust.

#### 1. Install `uv`
```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or using Homebrew (macOS)
brew install uv
```

#### 2. Install Project Dependencies
In the project root directory, run:
```bash
uv sync
```
This automatically sets up a virtual environment (`.venv`), resolves all dependencies from `pyproject.toml`, and locks them into `uv.lock`.

---

### Option B: Setup with standard `pip` / `venv`

If you prefer traditional Python tooling:
```bash
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install torch torchvision gymnasium pygame numpy huggingface-hub python-dotenv imageio imageio-ffmpeg matplotlib tqdm gradio
```

---

### Option C: Running the React Web Simulator

This repository also contains an interactive React + Vite web visualizer:
```bash
# Install frontend dependencies
npm install

# Start development server
npm run dev
```
Open your browser at `http://localhost:3000`.

---

## 🎮 How to Run the Project

### 1. Configure Environment Secrets
Create a `.env` file in the root directory by copying the example template:
```bash
cp .env.example .env
```
Open `.env` and fill in your Hugging Face credentials:
```ini
# Obtain your token with WRITE permissions at: https://huggingface.co/settings/tokens
HF_TOKEN=hf_your_huggingface_write_token
HF_USERNAME=your_hf_username
HF_REPO_NAME=flappy-bird-dqn-agent
```

---

### 2. Train the Agent
Start the Double DQN training loop:
```bash
uv run python src/train.py
```
* **Output:** Logs training progress via `tqdm`, tracks rolling average scores over 50-episode windows, automatically saves the best model checkpoint to `checkpoints/best_model.pth`, and produces a learning curve graph at `checkpoints/training_curve.png`.
* **Convergence Time:** Typically 8 to 15 minutes (~500 episodes) on CPU.

---

### 3. Watch the Agent Fly Live (GUI Evaluation)
To watch your trained agent fly live in a Pygame window:
```bash
uv run python src/evaluate.py
```
The agent executes greedy actions ($\arg\max_a Q(s, a)$) without random exploration, navigating pipe gaps with continuous precision.

---

### 4. Record Gameplay GIFs & MP4 Videos
To generate visual portfolio assets and demonstration GIFs:
```bash
uv run python src/record_video.py
```
This runs the environment in headless `rgb_array` mode and exports:
* `recordings/replay.gif`
* `recordings/replay.mp4`

---

### 5. Publish Model & Artifacts to Hugging Face Hub
To publish your trained weights, metadata, learning curves, and model cards directly to Hugging Face:
```bash
uv run python src/push_to_hub.py
```
This automated script:
1. Validates your `HF_TOKEN`.
2. Creates the repository `https://huggingface.co/{HF_USERNAME}/{HF_REPO_NAME}` if it doesn't already exist.
3. Generates a rich, Markdown-formatted **Model Card** (`README.md`) with tags and benchmark scores.
4. Uploads `best_model.pth`, `hyperparameters.json`, `training_curve.png`, and `replay.gif`.

---

## 🌐 Interactive Browser Simulator

The included Web Application provides:
1. **Live DQN Simulation:** Real-time canvas simulation showing the agent flying with forward parabolic trajectory arcs.
2. **Sensory Raycasts:** Interactive lasers tracking gap centers and upper/lower pipe lips.
3. **Q-Value Telemetry:** Real-time bar indicators displaying $Q(s, \text{coast})$ vs $Q(s, \text{flap})$.
4. **Agent Skill Switcher:** Switch between **Master DQN**, **Learner ($\epsilon = 0.18$)**, **Untrained ($\epsilon = 1.0$)**, and **Manual Mode (Spacebar)**.
5. **Code Explorer:** Interactive file-by-file viewer with one-click copy buttons for all Python source code.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

**Developed with ❤️ by Bharath CD (3rd year PES University)**
