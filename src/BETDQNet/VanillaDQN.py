"""
vanilla_dqn.py

Implementation of a vanilla Deep Q-Network (DQN) for discrete-action Gym environments.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import gymnasium as gym


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[64, 64]):
        super(QNetwork, self).__init__()
        layers = []
        last_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, h_dim))
            layers.append(nn.ReLU())
            last_dim = h_dim
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.uint8),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        obs_dim,
        action_dim,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_capacity=10000,
        batch_size=64,
        target_update_freq=1000,
        device=None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = QNetwork(obs_dim, action_dim).to(self.device)
        self.target_net = QNetwork(obs_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learn_step_counter = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.policy_net.net[-1].out_features)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return q_values.argmax().item()

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Current Q-values
        q_values = self.policy_net(states).gather(1, actions)
        # Next Q-values from target network
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
        # Compute target
        target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Update target network
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


def train(env_name="CartPole-v1", episodes=250, max_steps=200):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(obs_dim, action_dim)

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        done = False
        step = 0

        while not done and step < max_steps:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.replay_buffer.add(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            total_reward += reward
            step += 1

        print(f"Episode {ep}	Reward: {total_reward:.2f}	Epsilon: {agent.epsilon:.3f}")

    env.close()


def run_VanillaDQN(env, hyper_params_cfg: dict, env_cfg: dict):
    """
    Run the VanillaDQN.

    Params:
        env: a gymnasium env from env_factory
        hyper_params_cfg: the dict of the hyperparameters (default in main or can be changed from config.yaml).
        env_cfg: the per‐environment config from config.yaml.

    Returns:
        A list of total reward per episode.
    """
    episodes = hyper_params_cfg["episodes"]
    max_steps = hyper_params_cfg.get("max_steps", env_cfg.get("max_steps", 200))

    # pull in all the common hyperparams
    lr = hyper_params_cfg["learning_rate"]
    gamma = hyper_params_cfg["discount_factor"]
    eps_start = hyper_params_cfg["epsilon_start"]
    eps_end = hyper_params_cfg["epsilon_end"]
    eps_decay = (
        (eps_start - eps_end)
        / hyper_params_cfg["epsilon_decay_period"]
        if "epsilon_decay_period" in hyper_params_cfg
        else 0.995
    )

    # build agent
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(
        obs_dim,
        action_dim,
        lr=lr,
        gamma=gamma,
        epsilon_start=eps_start,
        epsilon_end=eps_end,
        epsilon_decay=eps_decay,
        buffer_capacity=hyper_params_cfg["memory_size"],
        batch_size=hyper_params_cfg["batch_size"],
        target_update_freq=hyper_params_cfg.get("target_update_freq", 1000),
    )

    try:
        episode_rewards = []
        for ep in range(1, episodes + 1):
            state, _ = env.reset()
            total_r = 0.0
            done = False
            step = 0

            while not done and step < max_steps:
                action, step = agent.select_action(state), step + 1
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                agent.replay_buffer.add(state, action, reward, next_state, done)
                agent.learn()

                state = next_state
                total_r += reward

            episode_rewards.append(total_r)
            print(f"[Vanilla] Ep {ep}\tReward {total_r:.2f}\tEps {agent.epsilon:.3f}")

    finally:
        print("Closing Environment for VanillaDQN.")
        env.close()
    return episode_rewards
