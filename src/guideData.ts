export interface CodeFile {
  filename: string;
  path: string;
  language: string;
  description: string;
  content: string;
}

export const GUIDE_FILES: CodeFile[] = [
  {
    filename: "README.md",
    path: "README.md",
    language: "markdown",
    description: "Official Project Documentation, Developer Credits (Bharath CD, PES University), Architecture, and Quickstart.",
    content: `# 🐦 Self-Flapping Bird: Autonomous DQN Reinforcement Learning Agent

Developer: Bharath CD, 3rd year PES University
Algorithms: Double Deep Q-Networks (DDQN), Experience Replay, Polyak Soft Target Updates
Tools: Astral uv, PyTorch 2.x, Gymnasium, Hugging Face Hub

## Quickstart
1. Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh
2. Install dependencies: uv sync
3. Train Agent: uv run python src/train.py
4. Watch live GUI: uv run python src/evaluate.py
5. Record Video: uv run python src/record_video.py
6. Push to Hugging Face: uv run python src/push_to_hub.py
`,
  },
  {
    filename: ".env.example",
    path: ".env.example",
    language: "ini",
    description: "Template for Hugging Face authentication credentials and repository identifiers.",
    content: `# .env.example
# Obtain your token with WRITE access at: https://huggingface.co/settings/tokens
HF_TOKEN=hf_your_write_token_here
HF_USERNAME=your_hf_username
HF_REPO_NAME=flappy-bird-dqn-agent
`,
  },
  {
    filename: ".gitignore",
    path: ".gitignore",
    language: "gitignore",
    description: "Prevents virtualenvs, tokens, binary weights, and cache files from leaking into git.",
    content: `# Virtual environments
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
`,
  },
  {
    filename: "pyproject.toml",
    path: "pyproject.toml",
    language: "toml",
    description: "Modern Astral uv project definition with pinned dependencies and Python 3.11.",
    content: `[project]
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
`,
  },
  {
    filename: "flappy_env.py",
    path: "src/env/flappy_env.py",
    language: "python",
    description: "Gymnasium-compliant Flappy Bird environment with normalized 8D state vector and shaped rewards.",
    content: `"""
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
        self.pipe_frequency = 220

        # Gymnasium Spaces
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None

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

        self._spawn_pipe(x=340)
        self._spawn_pipe(x=340 + self.pipe_frequency)

        obs = self._get_observation()
        info = {"score": self.score, "step": self.step_count}

        if self.render_mode == "human":
            self._render_frame()

        return obs, info

    def _spawn_pipe(self, x: float) -> None:
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

        if action == 1:
            self.bird_vel = self.flap_strength
        else:
            self.bird_vel = min(self.bird_vel + self.gravity, self.max_drop_speed)

        self.bird_y += self.bird_vel

        for pipe in self.pipes:
            pipe["x"] -= self.pipe_speed

        if self.pipes and self.pipes[0]["x"] + self.pipe_width < 0:
            self.pipes.pop(0)

        last_pipe_x = self.pipes[-1]["x"] if self.pipes else self.screen_width
        if last_pipe_x < self.screen_width + self.pipe_frequency:
            self._spawn_pipe(x=last_pipe_x + self.pipe_frequency)

        passed_pipe = False
        for pipe in self.pipes:
            if not pipe["scored"] and pipe["x"] + self.pipe_width < self.bird_x:
                pipe["scored"] = True
                self.score += 1
                passed_pipe = True

        terminated = self._check_collision()
        truncated = False
        reward = self._compute_reward(passed_pipe, terminated)

        obs = self._get_observation()
        info = {"score": self.score, "step": self.step_count}

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, terminated, truncated, info

    def _check_collision(self) -> bool:
        if self.bird_y - self.bird_radius <= 0 or self.bird_y + self.bird_radius >= self.screen_height:
            return True

        bird_left = self.bird_x - self.bird_radius
        bird_right = self.bird_x + self.bird_radius
        bird_top = self.bird_y - self.bird_radius
        bird_bottom = self.bird_y + self.bird_radius

        for pipe in self.pipes:
            pipe_left = pipe["x"]
            pipe_right = pipe["x"] + self.pipe_width
            if bird_right > pipe_left and bird_left < pipe_right:
                if bird_top < pipe["top_height"] or bird_bottom > pipe["bottom_y"]:
                    return True
        return False

    def _compute_reward(self, passed_pipe: bool, terminated: bool) -> float:
        if terminated:
            return -10.0
        reward = 0.15
        if passed_pipe:
            reward += 5.0
        next_pipe = self._get_upcoming_pipe(0)
        if next_pipe:
            dist_to_center = abs(self.bird_y - next_pipe["gap_center"])
            max_tolerance = self.pipe_gap / 2.0
            norm_proximity = max(0.0, 1.0 - (dist_to_center / max_tolerance))
            reward += 0.25 * norm_proximity
        if self.bird_y < 40:
            reward -= 0.35
        return float(reward)

    def _get_upcoming_pipe(self, index: int = 0) -> Optional[dict[str, Any]]:
        valid_pipes = [p for p in self.pipes if p["x"] + self.pipe_width >= self.bird_x - self.bird_radius]
        if index < len(valid_pipes):
            return valid_pipes[index]
        return None

    def _get_observation(self) -> np.ndarray:
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
            norm_dx_next, norm_top_gap, norm_bot_gap, norm_gap_diff = 1.0, 0.3, 0.7, 0.0

        if second_pipe is not None:
            norm_dx_second = np.clip((second_pipe["x"] - self.bird_x) / self.screen_width, 0.0, 1.0)
            norm_gap_second = np.clip((self.bird_y - second_pipe["gap_center"]) / self.screen_height, -1.0, 1.0)
        else:
            norm_dx_second, norm_gap_second = 1.0, 0.0

        return np.array([
            norm_bird_y, norm_bird_vel, norm_dx_next, norm_top_gap,
            norm_bot_gap, norm_gap_diff, norm_dx_second, norm_gap_second
        ], dtype=np.float32)

    def render(self) -> Optional[np.ndarray]:
        return self._render_frame()

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

        self.screen.fill((112, 197, 206))
        for pipe in self.pipes:
            top_rect = pygame.Rect(pipe["x"], 0, self.pipe_width, pipe["top_height"])
            pygame.draw.rect(self.screen, (115, 191, 46), top_rect)
            bot_height = self.screen_height - pipe["bottom_y"]
            bot_rect = pygame.Rect(pipe["x"], pipe["bottom_y"], self.pipe_width, bot_height)
            pygame.draw.rect(self.screen, (115, 191, 46), bot_rect)

        pygame.draw.circle(self.screen, (244, 180, 0), (int(self.bird_x), int(self.bird_y)), self.bird_radius)

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            if self.clock is not None:
                self.clock.tick(self.metadata["render_fps"])
            return None
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))
        return None

    def close(self) -> None:
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
`,
  },
  {
    filename: "dqn_model.py",
    path: "src/agent/dqn_model.py",
    language: "python",
    description: "Deep Q-Network MLP architecture with Layer Normalization for stable value predictions.",
    content: `"""
Deep Q-Network (DQN) Neural Network Architecture.
Maps continuous state observations to action Q-values.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
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
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        return self.out(x)
`,
  },
  {
    filename: "replay_buffer.py",
    path: "src/agent/replay_buffer.py",
    language: "python",
    description: "Circular experience replay buffer for decorrelated i.i.d. batch sampling.",
    content: `"""
Circular Experience Replay Memory Buffer for DQN transitions.
"""

from __future__ import annotations
import random
from collections import deque
from typing import Tuple
import numpy as np
import torch


class ReplayBuffer:
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
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
`,
  },
  {
    filename: "dqn_agent.py",
    path: "src/agent/dqn_agent.py",
    language: "python",
    description: "Double DQN agent with Polyak target updates and epsilon decay schedule.",
    content: `"""
Double Deep Q-Network (DDQN) Agent.
Manages action selection with epsilon-greedy exploration, target network updates,
and Bellman loss minimization.
"""

from __future__ import annotations
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agent.dqn_model import QNetwork
from src.agent.replay_buffer import ReplayBuffer


class DQNAgent:
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

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.q_online = QNetwork(state_dim, action_dim).to(self.device)
        self.q_target = QNetwork(state_dim, action_dim).to(self.device)
        self.q_target.load_state_dict(self.q_online.state_dict())
        self.q_target.eval()

        self.optimizer = optim.Adam(self.q_online.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = ReplayBuffer(capacity=buffer_capacity)

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        if not evaluate and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_online(state_t)
            action = int(torch.argmax(q_values, dim=1).item())
        return action

    def step(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool
    ) -> float | None:
        self.memory.push(state, action, reward, next_state, done)
        if len(self.memory) >= self.batch_size:
            return self.learn()
        return None

    def learn(self) -> float:
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size, self.device
        )
        current_q = self.q_online(states).gather(1, actions)

        with torch.no_grad():
            best_next_actions = self.q_online(next_states).argmax(dim=1, keepdim=True)
            target_q_values = self.q_target(next_states).gather(1, best_next_actions)
            expected_q = rewards + (self.gamma * target_q_values * (1.0 - dones))

        loss = self.loss_fn(current_q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_online.parameters(), max_norm=10.0)
        self.optimizer.step()

        self._soft_update_target_network()
        return float(loss.item())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def _soft_update_target_network(self) -> None:
        for target_param, online_param in zip(self.q_target.parameters(), self.q_online.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1.0 - self.tau) * target_param.data)

    def save_checkpoint(self, filepath: str) -> None:
        torch.save({
            "model_state_dict": self.q_online.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
        }, filepath)

    def load_checkpoint(self, filepath: str) -> None:
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_online.load_state_dict(checkpoint["model_state_dict"])
        self.q_target.load_state_dict(checkpoint["model_state_dict"])
        if "epsilon" in checkpoint:
            self.epsilon = checkpoint["epsilon"]
`,
  },
  {
    filename: "train.py",
    path: "src/train.py",
    language: "python",
    description: "Production training loop with rolling window metrics, checkpointing, and matplotlib curves.",
    content: `"""
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
        pbar.set_postfix({"Score": score, "Avg50": f"{avg_score:.2f}", "Eps": f"{agent.epsilon:.3f}"})

        if avg_score > best_avg_score and len(scores_window) >= 10:
            best_avg_score = avg_score
            agent.save_checkpoint(os.path.join(checkpoint_dir, "best_model.pth"))

        if avg_score >= target_score and len(scores_window) == 50:
            print(f"\\n🎉 Solved Flappy Bird with an average score of {avg_score:.2f}!")
            break

    agent.save_checkpoint(os.path.join(checkpoint_dir, "final_model.pth"))
    with open(os.path.join(checkpoint_dir, "hyperparameters.json"), "w") as f:
        json.dump({
            "episodes_trained": len(all_scores),
            "best_avg_score": best_avg_score,
            "final_epsilon": agent.epsilon,
        }, f, indent=2)

    plot_training_curve(all_scores, all_rewards, os.path.join(checkpoint_dir, "training_curve.png"))
    print(f"✅ Training complete! Artifacts saved in ./{checkpoint_dir}/")


def plot_training_curve(scores: list[int], rewards: list[float], save_path: str) -> None:
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(scores, label="Episode Score", alpha=0.3, color="gray")
    rolling_scores = np.convolve(scores, np.ones(25) / 25, mode="valid")
    plt.plot(range(24, len(scores)), rolling_scores, label="Rolling Mean (25)", color="blue", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Pipes Passed (Score)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    rolling_rewards = np.convolve(rewards, np.ones(25) / 25, mode="valid")
    plt.plot(range(24, len(rewards)), rolling_rewards, label="Rolling Reward (25)", color="green", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    train()
`,
  },
  {
    filename: "evaluate.py",
    path: "src/evaluate.py",
    language: "python",
    description: "Interactive visual evaluation script rendering live Pygame gameplay for human viewing.",
    content: `"""
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
        print("Run \`uv run python src/train.py\` first!")
        sys.exit(1)

    env = FlappyBirdEnv(render_mode="human")
    agent = DQNAgent()
    agent.load_checkpoint(checkpoint_path)
    print(f"🎮 Loaded weights from {checkpoint_path}")

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        done = False
        total_score = 0
        while not done:
            action = agent.select_action(state, evaluate=True)
            next_state, _, terminated, truncated, info = env.step(action)
            state = next_state
            total_score = info["score"]
            done = terminated or truncated

        print(f"Episode {ep}: Final Score = {total_score}")
        time.sleep(1.0)
    env.close()


if __name__ == "__main__":
    evaluate_agent()
`,
  },
  {
    filename: "record_video.py",
    path: "src/record_video.py",
    language: "python",
    description: "Headless gameplay video and GIF recorder for portfolio demonstration and Hugging Face card.",
    content: `"""
Headless Video & GIF Gameplay Recorder.
Runs evaluation in 'rgb_array' mode to output an animated GIF and MP4.
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

    gif_path = os.path.join(output_dir, "replay.gif")
    imageio.mimsave(gif_path, frames, fps=30, loop=0)
    print(f"✨ Saved gameplay GIF: {gif_path}")

    mp4_path = os.path.join(output_dir, "replay.mp4")
    imageio.mimsave(mp4_path, frames, fps=30)
    print(f"✨ Saved gameplay MP4: {mp4_path}")


if __name__ == "__main__":
    record_gameplay()
`,
  },
  {
    filename: "push_to_hub.py",
    path: "src/push_to_hub.py",
    language: "python",
    description: "Automated Hugging Face Hub publisher with token verification, artifact upload, and Model Card generation.",
    content: `"""
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
        print("Create a write token at https://huggingface.co/settings/tokens and add to .env")
        sys.exit(1)

    if not hf_username or "your_" in hf_username.lower():
        print("❌ Error: Please specify your HF_USERNAME in .env")
        sys.exit(1)

    repo_id = f"{hf_username}/{repo_name}"
    print(f"📦 Connecting to Hugging Face Hub: {repo_id}")
    api = HfApi(token=hf_token)

    try:
        create_repo(repo_id=repo_id, token=hf_token, repo_type="model", exist_ok=True, private=False)
        print(f"✅ Repository ready at: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Notice: {e}")

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
---

# 🐦 Self-Flapping Bird (DQN Agent)
Trained with Double Deep Q-Networks on kinematic observation space.

![Flappy Bird Gameplay](replay.gif)
"""
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
            api.upload_file(path_or_fileobj=local_file, path_in_repo=path_in_repo, repo_id=repo_id, repo_type="model")

    print(f"\\n🎉 Successfully published to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    push_to_huggingface()
`,
  },
  {
    filename: "app.py",
    path: "app.py",
    language: "python",
    description: "Interactive Gradio web demo suitable for deployment as a free Hugging Face Space.",
    content: `"""
Interactive Hugging Face Space Demo (Gradio).
Allows visitors to test observation inputs and inspect real-time Q-values.
"""

from __future__ import annotations
import os
import gradio as gr
import numpy as np
import torch
from src.agent.dqn_model import QNetwork


def load_model():
    model_path = "checkpoints/best_model.pth" if os.path.exists("checkpoints/best_model.pth") else "best_model.pth"
    model = QNetwork(state_dim=8, action_dim=2)
    if os.path.exists(model_path):
        chk = torch.load(model_path, map_location="cpu")
        model.load_state_dict(chk["model_state_dict"])
        model.eval()
        return model, "✅ Model weights loaded."
    return None, "⚠️ Model weights not found."


model, status = load_model()


def predict(bird_y, bird_vel, dx_next, top_gap, bot_gap, gap_diff, dx_second, second_gap_diff):
    if model is None:
        return "Model not initialized", 0.0, 0.0

    state = np.array([bird_y, bird_vel, dx_next, top_gap, bot_gap, gap_diff, dx_second, second_gap_diff], dtype=np.float32)
    state_t = torch.tensor(state).unsqueeze(0)
    with torch.no_grad():
        q_vals = model(state_t).squeeze().numpy()

    action = "Flap 🚀 (Action 1)" if np.argmax(q_vals) == 1 else "Coast 🪂 (Action 0)"
    return action, float(q_vals[0]), float(q_vals[1])


with gr.Blocks(title="Self-Flapping Bird DQN") as demo:
    gr.Markdown("# 🐦 Self-Flapping Bird: Deep Q-Network Inspector")
    with gr.Row():
        with gr.Column():
            by = gr.Slider(0.0, 1.0, value=0.5, label="Bird Y")
            bv = gr.Slider(-1.0, 1.0, value=0.1, label="Bird Velocity")
            dx = gr.Slider(0.0, 1.0, value=0.3, label="Distance to Pipe X")
            tg = gr.Slider(0.0, 1.0, value=0.35, label="Top Gap Y")
            bg = gr.Slider(0.0, 1.0, value=0.65, label="Bottom Gap Y")
            gd = gr.Slider(-1.0, 1.0, value=0.0, label="Gap Center Diff Y")
            dx2 = gr.Slider(0.0, 1.0, value=0.8, label="Second Pipe X")
            gd2 = gr.Slider(-1.0, 1.0, value=0.0, label="Second Gap Diff Y")
            btn = gr.Button("Evaluate Q-Values", variant="primary")
        with gr.Column():
            act = gr.Label(label="Selected Action")
            q0 = gr.Number(label="Q(Coast)")
            q1 = gr.Number(label="Q(Flap)")
    btn.click(fn=predict, inputs=[by, bv, dx, tg, bg, gd, dx2, gd2], outputs=[act, q0, q1])

if __name__ == "__main__":
    demo.launch()
`,
  },
];
