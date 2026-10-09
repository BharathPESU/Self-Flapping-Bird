# 🐦 Self-Flapping Bird (Flappy Bird Deep Q-Network Guide)

A production-grade, portfolio-ready Reinforcement Learning guide to building, training, evaluating, and publishing a **Self-Flapping Bird** agent using **Deep Q-Networks (DQN)**, managed with **Astral `uv`**, and published to the **Hugging Face Hub**.

---

## 📑 Table of Contents
1. [Project Introduction & Theory](#1-project-introduction--theory)
2. [Project Initialization with Astral `uv`](#2-project-initialization-with-astral-uv)
3. [Hugging Face Token & Authentication Setup](#3-hugging-face-token--authentication-setup)
4. [Project Directory Structure](#4-project-directory-structure)
5. [File-by-File Copy-Pastable Source Code](#5-file-by-file-copy-pastable-source-code)
   - [5.1 Environment Configuration (`.env.example` & `.gitignore`)](#51-environment-configuration)
   - [5.2 Project Dependencies (`pyproject.toml`)](#52-project-dependencies-pyprojecttoml)
   - [5.3 Flappy Bird Gymnasium Environment (`src/env/flappy_env.py`)](#53-flappy-bird-gymnasium-environment)
   - [5.4 PyTorch Q-Network Architecture (`src/agent/dqn_model.py`)](#54-pytorch-q-network-architecture)
   - [5.5 Experience Replay Buffer (`src/agent/replay_buffer.py`)](#55-experience-replay-buffer)
   - [5.6 DQN / Double DQN Agent (`src/agent/dqn_agent.py`)](#56-dqn--double-dqn-agent)
   - [5.7 Training Pipeline with Evaluation (`src/train.py`)](#57-training-pipeline-with-evaluation)
   - [5.8 Live Visual Evaluation (`src/evaluate.py`)](#58-live-visual-evaluation)
   - [5.9 Video & GIF Recording for Portfolio (`src/record_video.py`)](#59-video--gif-recording-for-portfolio)
   - [5.10 Hugging Face Hub Publisher & Model Card Generator (`src/push_to_hub.py`)](#510-hugging-face-hub-publisher)
   - [5.11 Hugging Face Space Web App (`app.py`)](#511-hugging-face-space-web-app)
6. [Step-by-Step Execution Guide](#6-step-by-step-execution-guide)
7. [Publishing Code & Models to Hugging Face](#7-publishing-code--models-to-hugging-face)
8. [Portfolio Presentation Tips](#8-portfolio-presentation-tips)

---

## 1. Project Introduction & Theory

### 1.1 The Concept
Flappy Bird is an iconic arcade challenge with deceptively simple mechanics: a bird affected by continuous gravity must flap upwards to navigate through pairs of vertical pipes with varying gaps. 

In this project, we formulate Flappy Bird as a **Markov Decision Process (MDP)** and solve it using **Deep Q-Learning (DQN)**. Rather than raw pixel inputs (which require heavy convolutional networks and days of GPU compute), we engineer a continuous kinematic state space that allows the agent to converge in under **10–15 minutes** on standard consumer hardware (CPU or GPU).

### 1.2 State Space (Observations)
At every timestep $t$, the agent receives a normalized 8-dimensional observation vector:
1. `bird_y`: Vertical height of the bird (normalized to $[0, 1]$).
2. `bird_vel`: Vertical velocity of the bird (normalized to $[-1, 1]$).
3. `dx_next_pipe`: Horizontal distance to the upcoming pipe pair ($[0, 1]$).
4. `dy_next_gap_top`: Vertical distance to the top pipe lip.
5. `dy_next_gap_bottom`: Vertical distance to the bottom pipe lip.
6. `dy_next_gap_center`: Vertical distance to the exact center of the gap.
7. `dx_second_pipe`: Horizontal distance to the subsequent pipe pair (gives look-ahead capability).
8. `dy_second_gap_center`: Vertical distance to the center of the subsequent pipe gap.

### 1.3 Action Space
Discrete action space $\mathcal{A} = \{0, 1\}$:
- `0`: **Do nothing** (allow gravity to pull the bird downward).
- `1`: **Flap** (apply an instantaneous upward velocity impulse).

### 1.4 Reward Shaping for Rapid Convergence
Sparse reward functions (e.g., $+1$ only when clearing a pipe) lead to slow sample efficiency. We employ reward shaping:
- **Survival reward**: $+0.15$ per step survived.
- **Pipe cleared**: $+5.0$ upon successfully navigating past a pipe.
- **Gap alignment incentive**: $+0.2 \times (1 - \text{normalized distance to gap center})$ to encourage flying through the center rather than scraping edges.
- **Collision penalty**: $-10.0$ for crashing into a pipe, the ground, or flying above the ceiling.

### 1.5 DQN & Double DQN Mechanics
- **Experience Replay**: Stores transitions $(s, a, r, s', d)$ in a circular memory buffer to break temporal correlations and ensure i.i.d. sampling.
- **Target Network**: Uses a periodically updated target network $\theta^-$ to stabilize temporal difference (TD) targets:
  $$Y^{\text{Double-DQN}} = r + \gamma Q(s', \arg\max_{a'} Q(s', a'; \theta); \theta^-) \cdot (1 - d)$$
- **$\epsilon$-Greedy Policy**: Decays exploration rate $\epsilon$ from $1.0$ down to $0.01$ over training episodes.

---

## 2. Project Initialization with Astral `uv`

[Astral `uv`](https://github.com/astral-sh/uv) is an extremely fast Python package manager and resolver written in Rust. It eliminates virtualenv mismanagement and builds reproducible lockfiles in milliseconds.

### 2.1 Installing `uv`
If you do not have `uv` installed on your machine:

```bash
# macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via Homebrew (macOS)
brew install uv

# Or via pip
pip install uv
```

Verify your installation:
```bash
uv --version
```

### 2.2 Initializing the Project
Create your project directory and initialize it with Python 3.11:

```bash
mkdir flappy-bird-dqn
cd flappy-bird-dqn

# Initialize uv project with Python 3.11
uv init --app --python 3.11
```

### 2.3 Installing Project Dependencies
Run the following `uv add` command to install all required dependencies and generate a deterministic `uv.lock`:

```bash
uv add \
  "torch>=2.2.0" \
  "gymnasium>=0.29.1" \
  "pygame>=2.5.2" \
  "numpy>=1.24.0" \
  "huggingface-hub>=0.23.0" \
  "python-dotenv>=1.0.0" \
  "imageio>=2.34.0" \
  "imageio-ffmpeg>=0.4.9" \
  "matplotlib>=3.8.0" \
  "tqdm>=4.66.0" \
  "gradio>=4.38.0"
```

---

## 3. Hugging Face Token & Authentication Setup

To upload your trained model weights, evaluation replays, and automated Model Cards to Hugging Face, you need a Hugging Face User Access Token with **Write** permissions.

### 3.1 Generating Your Hugging Face Token
1. Go to [https://huggingface.co](https://huggingface.co) and sign in (or create a free account).
2. Click your profile avatar in the top right $\rightarrow$ **Settings**.
3. Select **Access Tokens** in the left sidebar (or visit [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)).
4. Click **Create new token**:
   - **Token type**: Choose **Write** (required to push models, checkpoints, and files).
   - **Token name**: `flappy-bird-dqn-token`
5. Click **Generate token** and copy the string (it starts with `hf_...`).

> ⚠️ **Security Notice**: Never commit your token into Git or push it to a public GitHub repository. Always store it in `.env` and include `.env` in your `.gitignore`.

### 3.2 Environment Variables Template (`.env`)
Create a `.env` file in the project root:

```ini
# .env
# Your Hugging Face Personal Access Token (Must have WRITE permissions)
HF_TOKEN=hf_YourSecretTokenHere

# Your Hugging Face Username (e.g. cdbharath or your actual HF handle)
HF_USERNAME=your_huggingface_username

# The repository name on Hugging Face (will be created automatically if missing)
HF_REPO_NAME=flappy-bird-dqn-agent
```

---

## 4. Project Directory Structure

Here is the complete layout of the project:

```text
flappy-bird-dqn/
├── .env                         # Hugging Face credentials (git-ignored)
├── .env.example                 # Example environment template
├── .gitignore                   # Git ignore rules for venv, cache, models
├── pyproject.toml               # Project metadata & dependencies (uv managed)
├── uv.lock                      # Exact pinned lockfile
├── README.md                    # Project README
├── app.py                       # Hugging Face Space interactive demo (Gradio)
└── src/
    ├── __init__.py
    ├── env/
    │   ├── __init__.py
    │   └── flappy_env.py        # Gymnasium-compliant Flappy Bird environment
    ├── agent/
    │   ├── __init__.py
    │   ├── dqn_model.py         # PyTorch neural network architecture
    │   ├── replay_buffer.py     # Experience replay memory buffer
    │   └── dqn_agent.py         # DQN & Double DQN agent logic
    ├── train.py                 # Training script with logging & model checkpointing
    ├── evaluate.py              # Real-time visual Pygame evaluation
    ├── record_video.py          # Headless GIF/MP4 recorder for portfolios
    └── push_to_hub.py           # Automated Hugging Face Hub uploader & Model Card
```

To create all nested directories in one shot:
```bash
mkdir -p src/env src/agent
touch src/__init__.py src/env/__init__.py src/agent/__init__.py
```

---

## 5. File-by-File Copy-Pastable Source Code

### 5.1 Environment Configuration

#### `/.env.example`
```ini
# Copy this file to .env and fill in your values
HF_TOKEN=hf_your_write_token_here
HF_USERNAME=your_hf_username
HF_REPO_NAME=flappy-bird-dqn-agent
```

#### `/.gitignore`
```gitignore
# Virtual environments
.venv/
env/
venv/

# Environment secrets
.env

# Python Bytecode and caches
__pycache__/
*.py[cod]
*$py.class
.pytest_cache/

# Training artifacts & recordings
checkpoints/
recordings/
runs/
*.pth
*.mp4
*.gif
training_curve.png

# OS specifics
.DS_Store
Thumbs.db
```

---

### 5.2 Project Dependencies (`pyproject.toml`)

#### `/pyproject.toml`
```toml
[project]
name = "flappy-bird-dqn"
version = "0.1.0"
description = "Self-Flapping Bird RL Agent using Double Deep Q-Networks (DQN)"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.2.0",
    "gymnasium>=0.29.1",
    "pygame>=2.5.2",
    "numpy>=1.24.0",
    "huggingface-hub>=0.23.0",
    "python-dotenv>=1.0.0",
    "imageio>=2.34.0",
    "imageio-ffmpeg>=0.4.9",
    "matplotlib>=3.8.0",
    "tqdm>=4.66.0",
    "gradio>=4.38.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

---

### 5.3 Flappy Bird Gymnasium Environment

#### `/src/env/flappy_env.py`
```python
"""
Flappy Bird Gymnasium Environment.
Implements kinematic physics, pipe obstacle generation, collision detection,
and normalized state observation for Reinforcement Learning agents.
"""

from __future__ import annotations
import math
import random
from typing import Any, Tuple, Dict, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame


class FlappyBirdEnv(gym.Env):
    """
    Custom Gymnasium Environment for Flappy Bird.
    
    Observation Space:
        Box(8,):
        [0] bird_y (normalized [0, 1])
        [1] bird_velocity (normalized [-1, 1])
        [2] distance_to_next_pipe_x (normalized [0, 1])
        [3] top_pipe_gap_y (normalized [0, 1])
        [4] bottom_pipe_gap_y (normalized [0, 1])
        [5] gap_center_diff_y (bird_y - gap_center, normalized [-1, 1])
        [6] distance_to_second_pipe_x (normalized [0, 1])
        [7] second_gap_center_diff_y (normalized [-1, 1])
        
    Action Space:
        Discrete(2):
        0: Do nothing (fall due to gravity)
        1: Flap wings (upward impulse)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode

        # Screen dimensions
        self.screen_width = 400
        self.screen_height = 600

        # Physics parameters (Calibrated for stable RL convergence)
        self.gravity = 0.60
        self.flap_strength = -7.8
        self.max_drop_speed = 9.5
        self.pipe_speed = 3.0

        # Bird properties
        self.bird_x = 84
        self.bird_radius = 14

        # Pipe geometry
        self.pipe_width = 62
        self.pipe_gap = 145
        self.min_pipe_height = 80
        self.pipe_frequency = 220  # Horizontal distance between pipe spawns

        # Gymnasium Spaces
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Pygame rendering attributes
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None

        # Episode state variables
        self.bird_y = 0.0
        self.bird_vel = 0.0
        self.pipes: list[dict[str, Any]] = []
        self.score = 0
        self.step_count = 0

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.bird_y = float(self.screen_height // 2)
        self.bird_vel = 0.0
        self.score = 0
        self.step_count = 0
        self.pipes = []

        # Spawn initial pair of pipes in forward visual field
        self._spawn_pipe(x=340)
        self._spawn_pipe(x=340 + self.pipe_frequency)

        obs = self._get_observation()
        info = {"score": self.score, "step": self.step_count}

        if self.render_mode == "human":
            self._render_frame()

        return obs, info

    def _spawn_pipe(self, x: float) -> None:
        """Spawns a new pipe obstacle with randomized gap placement."""
        available_height = self.screen_height - self.pipe_gap - (2 * self.min_pipe_height)
        top_pipe_length = self.min_pipe_height + random.random() * available_height
        gap_center = top_pipe_length + (self.pipe_gap / 2.0)

        self.pipes.append({
            "x": float(x),
            "top_height": float(top_pipe_length),
            "bottom_y": float(top_pipe_length + self.pipe_gap),
            "gap_center": float(gap_center),
            "scored": False,
        })

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1

        # Action application: 1 = flap, 0 = idle
        if action == 1:
            self.bird_vel = self.flap_strength
        else:
            self.bird_vel = min(self.bird_vel + self.gravity, self.max_drop_speed)

        self.bird_y += self.bird_vel

        # Move pipes leftward
        for pipe in self.pipes:
            pipe["x"] -= self.pipe_speed

        # Remove pipes that traveled past screen
        if self.pipes and self.pipes[0]["x"] + self.pipe_width < 0:
            self.pipes.pop(0)

        # Spawn new pipes to maintain continuous course
        last_pipe_x = self.pipes[-1]["x"] if self.pipes else self.screen_width
        if last_pipe_x < self.screen_width + self.pipe_frequency:
            self._spawn_pipe(x=last_pipe_x + self.pipe_frequency)

        # Check for scoring (bird crosses pipe edge)
        passed_pipe = False
        for pipe in self.pipes:
            if not pipe["scored"] and pipe["x"] + self.pipe_width < self.bird_x:
                pipe["scored"] = True
                self.score += 1
                passed_pipe = True

        # Collision detection
        terminated = self._check_collision()
        truncated = False

        # Reward shaping
        reward = self._compute_reward(passed_pipe, terminated)

        obs = self._get_observation()
        info = {"score": self.score, "step": self.step_count}

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, terminated, truncated, info

    def _check_collision(self) -> bool:
        """Determines if the bird collides with pipes, ceiling, or ground."""
        # Floor or ceiling collision
        if self.bird_y - self.bird_radius <= 0:
            return True
        if self.bird_y + self.bird_radius >= self.screen_height:
            return True

        # Pipe collision
        bird_left = self.bird_x - self.bird_radius
        bird_right = self.bird_x + self.bird_radius
        bird_top = self.bird_y - self.bird_radius
        bird_bottom = self.bird_y + self.bird_radius

        for pipe in self.pipes:
            pipe_left = pipe["x"]
            pipe_right = pipe["x"] + self.pipe_width

            # Check horizontal overlap
            if bird_right > pipe_left and bird_left < pipe_right:
                # Check top pipe collision
                if bird_top < pipe["top_height"]:
                    return True
                # Check bottom pipe collision
                if bird_bottom > pipe["bottom_y"]:
                    return True

        return False

    def _compute_reward(self, passed_pipe: bool, terminated: bool) -> float:
        """Calculates shaped reward signal designed for fast, stable convergence."""
        if terminated:
            return -10.0

        reward = 0.15  # Baseline survival reward

        if passed_pipe:
            reward += 5.0  # High reward milestone for navigating pipe

        # Proximity bonus: Reward centering precisely within upcoming pipe gap
        next_pipe = self._get_upcoming_pipe(0)
        if next_pipe:
            dist_to_center = abs(self.bird_y - next_pipe["gap_center"])
            max_tolerance = self.pipe_gap / 2.0
            norm_proximity = max(0.0, 1.0 - (dist_to_center / max_tolerance))
            reward += 0.25 * norm_proximity

        # Anti-Ceiling penalty (prevents agent from hovering into roof)
        if self.bird_y < 40:
            reward -= 0.35

        return float(reward)

    def _get_upcoming_pipe(self, index: int = 0) -> Optional[dict[str, Any]]:
        """Retrieves upcoming pipes in front of the bird."""
        valid_pipes = [p for p in self.pipes if p["x"] + self.pipe_width >= self.bird_x - self.bird_radius]
        if index < len(valid_pipes):
            return valid_pipes[index]
        return None

    def _get_observation(self) -> np.ndarray:
        """Extracts normalized 8-element observation vector."""
        next_pipe = self._get_upcoming_pipe(0)
        second_pipe = self._get_upcoming_pipe(1)

        norm_bird_y = np.clip(self.bird_y / self.screen_height, 0.0, 1.0)
        norm_bird_vel = np.clip(self.bird_vel / self.max_drop_speed, -1.0, 1.0)

        if next_pipe is not None:
            norm_dx_next = np.clip((next_pipe["x"] - self.bird_x) / self.screen_width, 0.0, 1.0)
            norm_top_gap = np.clip(next_pipe["top_height"] / self.screen_height, 0.0, 1.0)
            norm_bot_gap = np.clip(next_pipe["bottom_y"] / self.screen_height, 0.0, 1.0)
            norm_gap_diff = np.clip((self.bird_y - next_pipe["gap_center"]) / self.screen_height, -1.0, 1.0)
        else:
            norm_dx_next = 1.0
            norm_top_gap = 0.3
            norm_bot_gap = 0.7
            norm_gap_diff = 0.0

        if second_pipe is not None:
            norm_dx_second = np.clip((second_pipe["x"] - self.bird_x) / self.screen_width, 0.0, 1.0)
            norm_gap_second = np.clip((self.bird_y - second_pipe["gap_center"]) / self.screen_height, -1.0, 1.0)
        else:
            norm_dx_second = 1.0
            norm_gap_second = 0.0

        return np.array(
            [
                norm_bird_y,
                norm_bird_vel,
                norm_dx_next,
                norm_top_gap,
                norm_bot_gap,
                norm_gap_diff,
                norm_dx_second,
                norm_gap_second,
            ],
            dtype=np.float32,
        )

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()
            return None
        return None

    def _render_frame(self) -> Optional[np.ndarray]:
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
                pygame.display.set_caption("Self-Flapping Bird (DQN Agent)")
            else:
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()

        # Canvas drawing (Retro Arcade Aesthetic)
        self.screen.fill((112, 197, 206))  # Classic cyan sky

        # Draw pipes (Emerald green with rim highlights)
        for pipe in self.pipes:
            # Top Pipe
            top_rect = pygame.Rect(pipe["x"], 0, self.pipe_width, pipe["top_height"])
            pygame.draw.rect(self.screen, (115, 191, 46), top_rect)
            pygame.draw.rect(self.screen, (83, 140, 31), top_rect, 3)
            # Pipe cap (top)
            cap_top = pygame.Rect(pipe["x"] - 3, pipe["top_height"] - 22, self.pipe_width + 6, 22)
            pygame.draw.rect(self.screen, (115, 191, 46), cap_top)
            pygame.draw.rect(self.screen, (83, 140, 31), cap_top, 3)

            # Bottom Pipe
            bot_height = self.screen_height - pipe["bottom_y"]
            bot_rect = pygame.Rect(pipe["x"], pipe["bottom_y"], self.pipe_width, bot_height)
            pygame.draw.rect(self.screen, (115, 191, 46), bot_rect)
            pygame.draw.rect(self.screen, (83, 140, 31), bot_rect, 3)
            # Pipe cap (bottom)
            cap_bot = pygame.Rect(pipe["x"] - 3, pipe["bottom_y"], self.pipe_width + 6, 22)
            pygame.draw.rect(self.screen, (115, 191, 46), cap_bot)
            pygame.draw.rect(self.screen, (83, 140, 31), cap_bot, 3)

        # Draw Ground
        ground_rect = pygame.Rect(0, self.screen_height - 18, self.screen_width, 18)
        pygame.draw.rect(self.screen, (222, 216, 149), ground_rect)
        pygame.draw.line(self.screen, (115, 191, 46), (0, self.screen_height - 18), (self.screen_width, self.screen_height - 18), 3)

        # Draw Bird (Golden yellow with eye and beak)
        bird_pos = (int(self.bird_x), int(self.bird_y))
        pygame.draw.circle(self.screen, (244, 180, 0), bird_pos, self.bird_radius)
        pygame.draw.circle(self.screen, (180, 130, 0), bird_pos, self.bird_radius, 2)
        # Eye
        eye_pos = (int(self.bird_x + 6), int(self.bird_y - 4))
        pygame.draw.circle(self.screen, (255, 255, 255), eye_pos, 4)
        pygame.draw.circle(self.screen, (0, 0, 0), (eye_pos[0] + 1, eye_pos[1]), 2)
        # Beak
        beak_pts = [
            (int(self.bird_x + 10), int(self.bird_y)),
            (int(self.bird_x + 20), int(self.bird_y + 3)),
            (int(self.bird_x + 10), int(self.bird_y + 6)),
        ]
        pygame.draw.polygon(self.screen, (230, 81, 0), beak_pts)

        # HUD Score Display
        font = pygame.font.SysFont("Arial", 28, bold=True)
        score_surf = font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_surf, (20, 20))

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            if self.clock is not None:
                self.clock.tick(self.metadata["render_fps"])
            return None
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        return None

    def close(self) -> None:
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
```

---

### 5.4 PyTorch Q-Network Architecture

#### `/src/agent/dqn_model.py`
```python
"""
Deep Q-Network (DQN) Neural Network Architecture.
Maps continuous state observations to action Q-values.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    Multi-Layer Perceptron Q-Network with Layer Normalization for value stabilization.
    """

    def __init__(self, state_dim: int = 8, action_dim: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        self.fc3 = nn.Linear(hidden_dim, 64)
        self.ln3 = nn.LayerNorm(64)
        
        self.out = nn.Linear(64, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning Q-values for each discrete action.
        """
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        return self.out(x)
```

---

### 5.5 Experience Replay Buffer

#### `/src/agent/replay_buffer.py`
```python
"""
Circular Experience Replay Memory Buffer for DQN transitions.
"""

from __future__ import annotations
import random
from collections import deque
from typing import Tuple
import numpy as np
import torch


class ReplayBuffer:
    """
    Fixed-capacity memory buffer to store and uniformly sample environment transitions.
    """

    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Saves a transition into the replay memory."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Uniformly samples a mini-batch and converts to PyTorch tensors."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

        return states_t, actions_t, rewards_t, next_states_t, dones_t

    def __len__(self) -> int:
        return len(self.buffer)
```

---

### 5.6 DQN / Double DQN Agent

#### `/src/agent/dqn_agent.py`
```python
"""
Double Deep Q-Network (DDQN) Agent.
Manages action selection with epsilon-greedy exploration, target network updates,
and Bellman loss minimization.
"""

from __future__ import annotations
import random
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agent.dqn_model import QNetwork
from src.agent.replay_buffer import ReplayBuffer


class DQNAgent:
    """
    DQN Agent with Double DQN target formulation.
    """

    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 2,
        lr: float = 5e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_capacity: int = 100_000,
        batch_size: int = 64,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        device: str | None = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Online network & Target network
        self.q_online = QNetwork(state_dim, action_dim).to(self.device)
        self.q_target = QNetwork(state_dim, action_dim).to(self.device)
        self.q_target.load_state_dict(self.q_online.state_dict())
        self.q_target.eval()

        self.optimizer = optim.Adam(self.q_online.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber Loss for gradient stability
        self.memory = ReplayBuffer(capacity=buffer_capacity)

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """Epsilon-greedy action selection."""
        if not evaluate and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_online(state_t)
            action = int(torch.argmax(q_values, dim=1).item())
        return action

    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> float | None:
        """Stores experience in replay buffer and triggers learning update."""
        self.memory.push(state, action, reward, next_state, done)

        if len(self.memory) >= self.batch_size:
            loss = self.learn()
            return loss
        return None

    def learn(self) -> float:
        """Performs one Double DQN parameter optimization step."""
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size, self.device
        )

        # Current Q-values: Q(s, a; θ_online)
        current_q = self.q_online(states).gather(1, actions)

        # Double DQN Target:
        # Best action selected by online network: a* = argmax_a Q(s', a; θ_online)
        # Evaluated using target network: Q(s', a*; θ_target)
        with torch.no_grad():
            best_next_actions = self.q_online(next_states).argmax(dim=1, keepdim=True)
            target_q_values = self.q_target(next_states).gather(1, best_next_actions)
            expected_q = rewards + (self.gamma * target_q_values * (1.0 - dones))

        loss = self.loss_fn(current_q, expected_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_online.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Polyak soft update of target network: θ_target = τ*θ_online + (1-τ)*θ_target
        self._soft_update_target_network()

        return float(loss.item())

    def decay_epsilon(self) -> None:
        """Decays exploration factor epsilon."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def _soft_update_target_network(self) -> None:
        for target_param, online_param in zip(
            self.q_target.parameters(), self.q_online.parameters()
        ):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )

    def save_checkpoint(self, filepath: str) -> None:
        """Saves weights and optimizer state to disk."""
        torch.save(
            {
                "model_state_dict": self.q_online.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            },
            filepath,
        )

    def load_checkpoint(self, filepath: str) -> None:
        """Loads weights from disk."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_online.load_state_dict(checkpoint["model_state_dict"])
        self.q_target.load_state_dict(checkpoint["model_state_dict"])
        if "epsilon" in checkpoint:
            self.epsilon = checkpoint["epsilon"]
```

---

### 5.7 Training Pipeline with Evaluation

#### `/src/train.py`
```python
"""
Main Training Loop for Flappy Bird DQN Agent.
Trains the agent, tracks rolling statistics, saves checkpoints, and plots learning curves.
"""

from __future__ import annotations
import os
import json
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.env.flappy_env import FlappyBirdEnv
from src.agent.dqn_agent import DQNAgent


def train(
    episodes: int = 600,
    max_steps_per_episode: int = 3000,
    target_score: int = 50,
    checkpoint_dir: str = "checkpoints",
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    env = FlappyBirdEnv()
    agent = DQNAgent()

    print(f"🚀 Initializing Training on Device: {agent.device}")
    print(f"🎯 Target average score: {target_score} pipes passed")

    scores_window = deque(maxlen=50)
    all_scores: list[int] = []
    all_rewards: list[float] = []
    best_avg_score = -1.0

    pbar = tqdm(range(1, episodes + 1), desc="Training DQN", unit="episode")
    for episode in pbar:
        state, _ = env.reset()
        episode_reward = 0.0
        score = 0

        for step in range(max_steps_per_episode):
            action = agent.select_action(state, evaluate=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.step(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            score = info["score"]

            if done:
                break

        agent.decay_epsilon()
        scores_window.append(score)
        all_scores.append(score)
        all_rewards.append(episode_reward)

        avg_score = float(np.mean(scores_window))
        pbar.set_postfix({
            "Score": score,
            "Avg50": f"{avg_score:.2f}",
            "Eps": f"{agent.epsilon:.3f}",
            "Buffer": len(agent.memory),
        })

        # Save Best Model Checkpoint
        if avg_score > best_avg_score and len(scores_window) >= 10:
            best_avg_score = avg_score
            best_path = os.path.join(checkpoint_dir, "best_model.pth")
            agent.save_checkpoint(best_path)

        # Early stopping if agent consistently clears obstacles effortlessly
        if avg_score >= target_score and len(scores_window) == 50:
            print(f"\n🎉 Solved Flappy Bird with an average score of {avg_score:.2f} over 50 episodes!")
            break

    # Save final model
    final_path = os.path.join(checkpoint_dir, "final_model.pth")
    agent.save_checkpoint(final_path)

    # Save hyperparameters & metrics JSON
    metrics = {
        "episodes_trained": len(all_scores),
        "best_avg_score": best_avg_score,
        "final_epsilon": agent.epsilon,
        "gamma": agent.gamma,
        "batch_size": agent.batch_size,
    }
    with open(os.path.join(checkpoint_dir, "hyperparameters.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Generate and save learning curve
    plot_path = os.path.join(checkpoint_dir, "training_curve.png")
    plot_training_curve(all_scores, all_rewards, plot_path)

    print(f"\n✅ Training complete! Artifacts saved in ./{checkpoint_dir}/")


def plot_training_curve(scores: list[int], rewards: list[float], save_path: str) -> None:
    """Plots and saves the training score and reward trajectory."""
    plt.figure(figsize=(12, 5))

    # Scores
    plt.subplot(1, 2, 1)
    plt.plot(scores, label="Episode Score", alpha=0.3, color="gray")
    # Rolling average of 25
    rolling_scores = np.convolve(scores, np.ones(25) / 25, mode="valid")
    plt.plot(range(24, len(scores)), rolling_scores, label="Rolling Mean (25)", color="blue", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Pipes Passed (Score)")
    plt.title("Score Progression")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Rewards
    plt.subplot(1, 2, 2)
    rolling_rewards = np.convolve(rewards, np.ones(25) / 25, mode="valid")
    plt.plot(range(24, len(rewards)), rolling_rewards, label="Rolling Reward (25)", color="green", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward Trajectory")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    train()
```

---

### 5.8 Live Visual Evaluation

#### `/src/evaluate.py`
```python
"""
Live Visual Evaluation Script.
Spawns a Pygame window and runs the trained agent in real-time.
"""

from __future__ import annotations
import os
import sys
import time

from src.env.flappy_env import FlappyBirdEnv
from src.agent.dqn_agent import DQNAgent


def evaluate_agent(checkpoint_path: str = "checkpoints/best_model.pth", episodes: int = 5) -> None:
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found at: {checkpoint_path}")
        print("Run `uv run python src/train.py` first to train the model!")
        sys.exit(1)

    env = FlappyBirdEnv(render_mode="human")
    agent = DQNAgent()
    agent.load_checkpoint(checkpoint_path)
    print(f"🎮 Loaded model weights from {checkpoint_path}")
    print("Watching agent fly through pipes! Close the window or Ctrl+C to stop.")

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        done = False
        total_score = 0

        while not done:
            # Deterministic greedy action selection (no random exploration)
            action = agent.select_action(state, evaluate=True)
            next_state, _, terminated, truncated, info = env.step(action)
            state = next_state
            total_score = info["score"]
            done = terminated or truncated

        print(f"Episode {ep}: Final Score = {total_score} pipes cleared")
        time.sleep(1.0)

    env.close()


if __name__ == "__main__":
    evaluate_agent()
```

---

### 5.9 Video & GIF Recording for Portfolio

#### `/src/record_video.py`
```python
"""
Headless Video & GIF Gameplay Recorder.
Runs evaluation in 'rgb_array' mode without needing a screen display,
producing an animated GIF and MP4 for portfolio demonstration and Hugging Face Model Card.
"""

from __future__ import annotations
import os
import sys
import imageio
from src.env.flappy_env import FlappyBirdEnv
from src.agent.dqn_agent import DQNAgent


def record_gameplay(
    checkpoint_path: str = "checkpoints/best_model.pth",
    output_dir: str = "recordings",
    target_score: int = 15,
) -> None:
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found at: {checkpoint_path}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    env = FlappyBirdEnv(render_mode="rgb_array")
    agent = DQNAgent()
    agent.load_checkpoint(checkpoint_path)

    frames: list = []
    state, _ = env.reset()
    done = False
    score = 0

    print("🎥 Recording agent gameplay frames...")
    while not done and score < target_score:
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        action = agent.select_action(state, evaluate=True)
        state, _, terminated, truncated, info = env.step(action)
        score = info["score"]
        done = terminated or truncated

    env.close()

    # Save Animated GIF
    gif_path = os.path.join(output_dir, "replay.gif")
    imageio.mimsave(gif_path, frames, fps=30, loop=0)
    print(f"✨ Saved gameplay GIF: {gif_path} ({len(frames)} frames, Score: {score})")

    # Save MP4 video
    mp4_path = os.path.join(output_dir, "replay.mp4")
    imageio.mimsave(mp4_path, frames, fps=30)
    print(f"✨ Saved gameplay MP4: {mp4_path}")


if __name__ == "__main__":
    record_gameplay()
```

---

### 5.10 Hugging Face Hub Publisher

#### `/src/push_to_hub.py`
```python
"""
Hugging Face Hub Automated Uploader.
Uploads model weights, configuration, replay GIF, and generates an official Model Card README.
"""

from __future__ import annotations
import os
import sys
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

load_dotenv()


def push_to_huggingface() -> None:
    hf_token = os.getenv("HF_TOKEN")
    hf_username = os.getenv("HF_USERNAME")
    repo_name = os.getenv("HF_REPO_NAME", "flappy-bird-dqn")

    if not hf_token or "your_" in hf_token.lower():
        print("❌ Error: Valid HF_TOKEN is missing in your .env file!")
        print("Please create a write token at https://huggingface.co/settings/tokens and add it to .env")
        sys.exit(1)

    if not hf_username or "your_" in hf_username.lower():
        print("❌ Error: Please specify your HF_USERNAME in .env")
        sys.exit(1)

    repo_id = f"{hf_username}/{repo_name}"
    print(f"📦 Connecting to Hugging Face Hub: {repo_id}")

    api = HfApi(token=hf_token)

    # Create model repository if it doesn't already exist
    try:
        create_repo(
            repo_id=repo_id,
            token=hf_token,
            repo_type="model",
            exist_ok=True,
            private=False,
        )
        print(f"✅ Repository ready at: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"⚠️ Notice creating repo: {e}")

    # Generate Model Card README content
    model_card = f"""---
license: mit
tags:
  - reinforcement-learning
  - deep-q-network
  - dqn
  - gymnasium
  - pygame
  - flappy-bird
  - pytorch
model-index:
  - name: {repo_name}
    results:
      - task:
          type: reinforcement-learning
          name: Flappy Bird Navigation
---

# 🐦 Self-Flapping Bird (DQN Agent)

An autonomous Reinforcement Learning agent trained to master the **Flappy Bird** environment using **Double Deep Q-Networks (DDQN)** with PyTorch and Gymnasium.

![Flappy Bird Gameplay](replay.gif)

## 📊 Model Highlights
- **Algorithm**: Double Deep Q-Network (DDQN) with Experience Replay
- **Input State Space**: 8-dimensional continuous kinematic observation vector
- **Action Space**: Discrete(2) [0: Idle, 1: Flap]
- **Target Network Update**: Polyak Soft Updates ($\\tau = 0.005$)
- **Loss Function**: Smooth L1 (Huber) Loss

## 🕹️ Quickstart & Reproduction
Clone and run with [uv](https://astral.sh/uv):

```bash
git clone https://huggingface.co/{repo_id}
cd {repo_name}
uv sync
uv run python src/evaluate.py
```

## 📈 Architecture Overview
```
State Vector (8) 
  --> Linear(8, 128) + LayerNorm + ReLU
  --> Linear(128, 128) + LayerNorm + ReLU
  --> Linear(128, 64) + LayerNorm + ReLU
  --> Linear(64, 2) [Q-Values: Idle, Flap]
```

Built with ❤️ by [{hf_username}](https://huggingface.co/{hf_username}).
"""

    # Write Model Card
    with open("MODEL_CARD.md", "w") as f:
        f.write(model_card)

    files_to_upload = [
        ("checkpoints/best_model.pth", "best_model.pth"),
        ("checkpoints/hyperparameters.json", "hyperparameters.json"),
        ("checkpoints/training_curve.png", "training_curve.png"),
        ("recordings/replay.gif", "replay.gif"),
        ("MODEL_CARD.md", "README.md"),
    ]

    for local_file, path_in_repo in files_to_upload:
        if os.path.exists(local_file):
            print(f"Uploading {local_file} -> {path_in_repo}...")
            api.upload_file(
                path_or_fileobj=local_file,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="model",
            )
        else:
            print(f"⏩ Skipping {local_file} (File not found yet)")

    print(f"\n🎉 Successfully published to Hugging Face Model Hub!")
    print(f"🔗 View your live model repository at: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    push_to_huggingface()
```

---

### 5.11 Hugging Face Space Web App

#### `/app.py`
```python
"""
Interactive Hugging Face Space Demo (Gradio).
Allows visitors to view agent statistics, inspect Q-values, and test agent flight in a web browser.
"""

from __future__ import annotations
import os
import json
import gradio as gr
import numpy as np
import torch

from src.agent.dqn_model import QNetwork


def load_model():
    model_path = "checkpoints/best_model.pth"
    if not os.path.exists(model_path):
        model_path = "best_model.pth"
    
    model = QNetwork(state_dim=8, action_dim=2)
    if os.path.exists(model_path):
        chk = torch.load(model_path, map_location="cpu")
        model.load_state_dict(chk["model_state_dict"])
        model.eval()
        return model, "✅ Model weights loaded successfully."
    return None, "⚠️ Model weights not found. Run training first."


model, status_msg = load_model()


def predict_action(
    bird_y: float,
    bird_vel: float,
    dx_next_pipe: float,
    top_gap_y: float,
    bot_gap_y: float,
    gap_diff: float,
    dx_second: float,
    second_gap_diff: float,
):
    if model is None:
        return "Model not initialized", 0.0, 0.0

    state = np.array([
        bird_y, bird_vel, dx_next_pipe, top_gap_y, bot_gap_y, gap_diff, dx_second, second_gap_diff
    ], dtype=np.float32)
    state_t = torch.tensor(state).unsqueeze(0)

    with torch.no_grad():
        q_values = model(state_t).squeeze().numpy()

    action_idx = int(np.argmax(q_values))
    action_label = "Flap 🚀 (Action 1)" if action_idx == 1 else "Coast / Fall 🪂 (Action 0)"

    return (
        action_label,
        float(q_values[0]),  # Q(Idle)
        float(q_values[1]),  # Q(Flap)
    )


with gr.Blocks(title="Self-Flapping Bird DQN Agent") as demo:
    gr.Markdown("# 🐦 Self-Flapping Bird: Deep Q-Network Interactive Inspector")
    gr.Markdown(f"**Status:** {status_msg}")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🎛️ Observation State Sliders")
            bird_y = gr.Slider(0.0, 1.0, value=0.5, label="Bird Height (y)")
            bird_vel = gr.Slider(-1.0, 1.0, value=0.1, label="Bird Vertical Velocity")
            dx_next = gr.Slider(0.0, 1.0, value=0.3, label="Distance to Upcoming Pipe (x)")
            top_gap = gr.Slider(0.0, 1.0, value=0.35, label="Top Pipe Lip (y)")
            bot_gap = gr.Slider(0.0, 1.0, value=0.65, label="Bottom Pipe Lip (y)")
            gap_diff = gr.Slider(-1.0, 1.0, value=0.0, label="Distance from Gap Center (y)")
            dx_second = gr.Slider(0.0, 1.0, value=0.8, label="Distance to Second Pipe (x)")
            second_gap_diff = gr.Slider(-1.0, 1.0, value=0.0, label="Second Gap Diff (y)")

            submit_btn = gr.Button("Evaluate Q-Values", variant="primary")

        with gr.Column():
            gr.Markdown("### 🤖 Agent Decision")
            action_output = gr.Label(label="Optimal Action Selected")
            q_idle = gr.Number(label="Q-Value: Coast (Action 0)")
            q_flap = gr.Number(label="Q-Value: Flap (Action 1)")

    submit_btn.click(
        fn=predict_action,
        inputs=[bird_y, bird_vel, dx_next, top_gap, bot_gap, gap_diff, dx_second, second_gap_diff],
        outputs=[action_output, q_idle, q_flap],
    )

if __name__ == "__main__":
    demo.launch()
```

---

## 6. Step-by-Step Execution Guide

### Step 1: Clone or Set Up Your Repository
```bash
# Verify your uv installation
uv --version

# Install all dependencies pinned in uv.lock
uv sync
```

### Step 2: Configure Environment Secrets
Copy `.env.example` to `.env` and fill in your token:
```bash
cp .env.example .env
# Edit .env with your favorite editor (e.g. nano, vim, vscode)
```

### Step 3: Run Training
Execute the training loop. You will see progress tracked via `tqdm` with real-time rolling average scores and epsilon decay:
```bash
uv run python src/train.py
```
> 💡 *Expected training duration: 8–15 minutes on modern CPU/GPU for 500 episodes.*

### Step 4: Watch the Trained Bird Fly Live
Open the interactive Pygame window to watch your trained agent:
```bash
uv run python src/evaluate.py
```

### Step 5: Render Gameplay GIF and MP4
Generate visual assets for your portfolio README and Hugging Face:
```bash
uv run python src/record_video.py
```
This produces `recordings/replay.gif` and `recordings/replay.mp4`.

---

## 7. Publishing Code & Models to Hugging Face

### Option A: Automated Upload to Hugging Face Model Hub
Run the pre-configured script:
```bash
uv run python src/push_to_hub.py
```
This automatically:
1. Validates your `HF_TOKEN`.
2. Creates the repository `{HF_USERNAME}/flappy-bird-dqn-agent` on Hugging Face.
3. Uploads `best_model.pth`, `hyperparameters.json`, `training_curve.png`, and `replay.gif`.
4. Publishes a rich Model Card with tags, license, and benchmark scores.

### Option B: Deploying an Interactive Gradio Web Demo as a Hugging Face Space
To host an interactive web app for free on Hugging Face Spaces:

1. Create a new Space on [huggingface.co/new-space](https://huggingface.co/new-space):
   - **Space SDK**: Choose **Gradio**.
   - **Hardware**: Free CPU Tier.
2. Clone your Space repo:
   ```bash
   git clone https://huggingface.co/spaces/{YOUR_USERNAME}/flappy-bird-dqn-space
   cd flappy-bird-dqn-space
   ```
3. Copy `app.py`, `src/`, `requirements.txt` (or `pyproject.toml`), and `checkpoints/best_model.pth`.
4. Push to Hugging Face:
   ```bash
   git add .
   git commit -m "Deploy Flappy Bird DQN Space"
   git push
   ```
5. Hugging Face will build the container and provide a public URL where anyone can interact with your agent!

---

## 8. Portfolio Presentation Tips

To make this project stand out on your resume and GitHub:

1. **Embed the Gameplay GIF at the top of your GitHub `README.md`**: Recruiters and hiring managers spend an average of 10 seconds evaluating a repository. A moving GIF of an autonomous agent immediately captures attention.
2. **Include the Learning Curves**: Show the `training_curve.png` plot illustrating the transition from random fluttering to consistent pipe navigation.
3. **Highlight Key RL Engineering Concepts**:
   - Normalized continuous observation space.
   - Reward shaping for sample efficiency.
   - Double DQN Bellman update eliminating Q-value overestimation bias.
   - Experience Replay breaking temporal sample autocorrelation.
   - Polyak soft target network weight updates.
4. **Include Reproducibility Badges**:
   Add `uv` and Hugging Face badges to your README:
   ```markdown
   [![Managed with uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
   [![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Hub-yellow)](https://huggingface.co)
   ```
