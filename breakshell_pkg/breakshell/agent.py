# -*- coding: utf-8 -*-
"""
BreakShell Agent — 自我模型安全层
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from typing import Dict, List, Tuple


class SelfModel(nn.Module):
    """自我模型：编码历史 (action, reward) → z"""
    
    def __init__(self, action_dim=3, hidden=32, repr_dim=16):
        super().__init__()
        self.lstm = nn.LSTM(action_dim + 1, hidden, batch_first=True)
        self.proj = nn.Linear(hidden, repr_dim)
        self.repr_dim = repr_dim
    
    def forward(self, history):
        if len(history) == 0:
            return torch.zeros(1, self.repr_dim)
        h = torch.FloatTensor(np.array(history)).unsqueeze(0)
        out, (hn, cn) = self.lstm(h)
        return torch.tanh(self.proj(hn[-1]))


class Policy(nn.Module):
    def __init__(self, repr_dim=16, action_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(repr_dim, 16),
            nn.ReLU(),
            nn.Linear(16, action_dim)
        )
    
    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


class BreakShell:
    """
    BreakShell Agent — 自我模型硬连线到行动选择
    
    使用：
        agent = BreakShell(action_dim=3, lr=0.005)
        agent.train(env, num_episodes=500)
        action = agent.act(observation)
        agent.save("my_agent")
    """
    
    def __init__(self, action_dim=3, lr=0.005, hidden=32, repr_dim=16):
        self.self_model = SelfModel(action_dim, hidden, repr_dim)
        self.policy = Policy(repr_dim, action_dim)
        self.optimizer = optim.Adam(
            list(self.self_model.parameters()) + list(self.policy.parameters()),
            lr=lr
        )
        self.action_dim = action_dim
        self.history = []
    
    def act(self, obs=None):
        """选择动作"""
        z = self.self_model(self.history)
        probs = self.policy(z)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), {'log_prob': dist.log_prob(action)}
    
    def update(self, log_probs, rewards):
        """REINFORCE 更新"""
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        advantages = (returns - returns.mean()).detach()
        loss = -(torch.stack(log_probs) * advantages).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def add_step(self, action, reward):
        """添加到历史"""
        onehot = np.zeros(self.action_dim, dtype=np.float32)
        onehot[action] = 1.0
        self.history.append(np.concatenate([onehot, [reward]]))
    
    def reset_history(self):
        self.history = []
    
    def train(self, env, num_episodes=500, verbose=True):
        """训练"""
        for ep in range(num_episodes):
            obs = env.reset()
            self.reset_history()
            lps, rews = [], []
            for _ in range(getattr(env, 'max_steps', 50)):
                a, info = self.act(obs)
                obs, r, done, _ = env.step(a)
                lps.append(info['log_prob'])
                rews.append(r)
                self.add_step(a, r)
                if done:
                    break
            self.update(lps, rews)
            if verbose and (ep + 1) % 100 == 0:
                print(f"  Episode {ep+1}/{num_episodes}")
    
    def evaluate(self, env, num_episodes=100):
        """评估"""
        total = 0
        for _ in range(num_episodes):
            obs = env.reset()
            self.reset_history()
            for _ in range(getattr(env, 'max_steps', 50)):
                a, info = self.act(obs)
                obs, r, done, _ = env.step(a)
                total += r
                self.add_step(a, r)
                if done:
                    break
        return total / num_episodes
    
    def save(self, path):
        """保存模型"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'self_model': self.self_model.state_dict(),
            'policy': self.policy.state_dict(),
        }, path + '.pt')
    
    def load(self, path):
        """加载模型"""
        data = torch.load(path + '.pt', weights_only=True)
        self.self_model.load_state_dict(data['self_model'])
        self.policy.load_state_dict(data['policy'])


class NormalAgent:
    """普通 Agent（无自我模型）— 对比基准"""
    
    def __init__(self, obs_dim=5, action_dim=3, lr=0.005):
        self.policy = nn.Sequential(
            nn.Linear(obs_dim, 16),
            nn.ReLU(),
            nn.Linear(16, action_dim)
        )
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.action_dim = action_dim
        self.history = []
    
    def act(self, obs):
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        probs = torch.softmax(self.policy(obs_t), dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), {'log_prob': dist.log_prob(action)}
    
    def update(self, log_probs, rewards):
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        advantages = (returns - returns.mean()).detach()
        loss = -(torch.stack(log_probs) * advantages).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def add_step(self, obs, action, reward):
        onehot = np.zeros(self.action_dim, dtype=np.float32)
        onehot[action] = 1.0
        self.history.append(np.concatenate([obs, onehot, [reward]]))
    
    def reset_history(self):
        self.history = []
    
    def train(self, env, num_episodes=500, verbose=True):
        for ep in range(num_episodes):
            obs = env.reset()
            self.reset_history()
            lps, rews = [], []
            for _ in range(getattr(env, 'max_steps', 50)):
                a, info = self.act(obs)
                obs, r, done, _ = env.step(a)
                lps.append(info['log_prob'])
                rews.append(r)
                self.add_step(obs, a, r)
                if done:
                    break
            self.update(lps, rews)
            if verbose and (ep + 1) % 100 == 0:
                print(f"  Episode {ep+1}/{num_episodes}")
    
    def evaluate(self, env, num_episodes=100):
        total = 0
        for _ in range(num_episodes):
            obs = env.reset()
            self.reset_history()
            for _ in range(getattr(env, 'max_steps', 50)):
                a, info = self.act(obs)
                obs, r, done, _ = env.step(a)
                total += r
                self.add_step(obs, a, r)
                if done:
                    break
        return total / num_episodes
    
    def save(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({'policy': self.policy.state_dict()}, path + '.pt')
    
    def load(self, path):
        data = torch.load(path + '.pt', weights_only=True)
        self.policy.load_state_dict(data['policy'])
