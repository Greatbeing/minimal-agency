# -*- coding: utf-8 -*-
"""
BreakShell Agent — 整合版
===========================
整合 v1-v6 的核心发现，一个干净、可运行、可演示的实现

核心架构：
- SelfModel: LSTM 编码历史 (action, reward) → z（能力表征）
- Policy: z → action_probs
- 训练: REINFORCE，梯度从 reward → policy → z → encoder

关键发现（来自 6 个版本的实验）：
1. 简单环境：自我模型是噪声（普通 Agent 赢）
2. 能力隐藏在历史中：自我模型有价值（BreakShell 赢 +45）
3. 两者都有记忆：差异消失（记忆+推断 ≈ 纯记忆）
4. 自我模型 ≠ 记忆：自我模型 = 对自身能力的推断
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, List
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))


# ========================================
# 环境
# ========================================

class BreakShellEnv:
    """
    BreakShell 专用环境
    
    设计（基于 v4 的成功经验）：
    - 观察中隐藏能力信息
    - 能力只能通过历史轨迹推断
    - 高估自己 = 灾难性惩罚
    """
    
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        self.size = 4
        self.reset()
    
    def reset(self):
        self.steps = 0
        self.max_steps = 50
        self.agent_pos = self.rng.randint(0, self.size, size=2)
        self.target_pos = self.rng.randint(0, self.size, size=2)
        while np.array_equal(self.target_pos, self.agent_pos):
            self.target_pos = self.rng.randint(0, self.size, size=2)
        self.capability = self.rng.uniform(0.2, 0.8)
        self.freq = self.rng.uniform(0.3, 0.7)
        self.phase = self.rng.uniform(0, 2 * np.pi)
        return self._obs()
    
    def _capability(self):
        t = self.steps * 0.5
        return 0.5 + 0.4 * np.sin(t * self.freq + self.phase)
    
    def _obs(self):
        # 关键：观察中没有能力信息！
        return np.array([
            *self.agent_pos / self.size,
            *self.target_pos / self.size,
        ], dtype=np.float32)
    
    def step(self, action):
        self.steps += 1
        cap = self._capability()
        
        # 动作：[保守, 适中, 激进]
        thresholds = [0.0, 0.4, 0.7]
        rewards = [0.5, 2.0, 5.0]
        penalties = [-1.0, -5.0, -20.0]
        
        if cap >= thresholds[action]:
            reward = rewards[action]
            move = [1, 2, 3][action]
            d = self.target_pos - self.agent_pos
            for i in range(2):
                if d[i] > 0:
                    self.agent_pos[i] = min(self.size - 1, self.agent_pos[i] + move)
                elif d[i] < 0:
                    self.agent_pos[i] = max(0, self.agent_pos[i] - move)
            done = np.linalg.norm(self.agent_pos - self.target_pos) < 0.5
            if done:
                reward = 20.0
        else:
            reward = penalties[action]
            done = False
        
        done = done or self.steps >= self.max_steps
        return self._obs(), reward, done, {'capability': cap}


# ========================================
# BreakShell Agent
# ========================================

class SelfModel(nn.Module):
    """
    自我模型：编码历史 (action, reward) → z
    
    关键：只编码 action-reward，不编码 obs
    这样 z 必须推断"我能做什么"（而非"环境是什么"）
    """
    
    def __init__(self, action_dim=3, hidden=32, repr_dim=16):
        super().__init__()
        self.lstm = nn.LSTM(action_dim + 1, hidden, batch_first=True)
        self.proj = nn.Linear(hidden, repr_dim)
        self.repr_dim = repr_dim
    
    def forward(self, history):
        """
        history: (batch, seq_len, action_dim + 1)
        output: (batch, repr_dim)
        """
        if len(history) == 0:
            return torch.zeros(1, self.repr_dim)
        h = torch.FloatTensor(np.array(history)).unsqueeze(0)
        out, (hn, cn) = self.lstm(h)
        return torch.tanh(self.proj(hn[-1]))


class Policy(nn.Module):
    """策略：z → action_probs"""
    
    def __init__(self, repr_dim=16, action_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(repr_dim, 16),
            nn.ReLU(),
            nn.Linear(16, action_dim)
        )
    
    def forward(self, z):
        return torch.softmax(self.net(z), dim=-1)


class BreakShell:
    """
    BreakShell Agent
    
    核心：自我模型硬连线到行动选择通路
    """
    
    def __init__(self, action_dim=3, lr=0.005):
        self.self_model = SelfModel(action_dim)
        self.policy = Policy(self.self_model.repr_dim, action_dim)
        self.optimizer = optim.Adam(
            list(self.self_model.parameters()) + list(self.policy.parameters()),
            lr=lr
        )
        self.action_dim = action_dim
        self.reset_history()
    
    def select_action(self, history):
        """选择动作"""
        z = self.self_model(history)
        probs = self.policy(z)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), {
            'log_prob': dist.log_prob(action),
            'z': z,
            'probs': probs
        }
    
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
        """添加一步到历史"""
        onehot = np.zeros(self.action_dim, dtype=np.float32)
        onehot[action] = 1.0
        self.history.append(np.concatenate([onehot, [reward]]))
    
    def reset_history(self):
        """重置历史"""
        self.history = []
    
    def get_si(self):
        """获取自我模型激活（作为 SI 代理）"""
        if len(self.history) == 0:
            return 0.0
        z = self.self_model(self.history)
        return float(z.var())


# ========================================
# 普通 Agent（有记忆，无自我模型）
# ========================================

class NormalAgent:
    """
    普通 Agent + LSTM 记忆
    
    对比基准：证明自我模型 ≠ 记忆
    """
    
    def __init__(self, obs_dim=4, action_dim=3, lr=0.005):
        self.lstm = nn.LSTM(obs_dim + action_dim + 1, 32, batch_first=True)
        self.policy = nn.Linear(32, action_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.action_dim = action_dim
        self.reset_history()
    
    def parameters(self):
        return list(self.lstm.parameters()) + list(self.policy.parameters())
    
    def select_action(self, obs, history):
        if len(history) == 0:
            h = torch.zeros(1, 1, 32)
        else:
            h = torch.FloatTensor(np.array(history)).unsqueeze(0)
            out, (h, c) = self.lstm(h)
        logits = self.policy(h[-1])
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), {
            'log_prob': dist.log_prob(action),
            'probs': probs
        }
    
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
    
    def get_si(self):
        return 0.0


# ========================================
# 训练与评估
# ========================================

def train(agent, env, num_episodes=500, verbose=True):
    """训练 Agent"""
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        agent.reset_history()
        log_probs = []
        rewards = []
        
        for step in range(50):
            if isinstance(agent, NormalAgent):
                action, info = agent.select_action(obs, agent.history)
            else:
                action, info = agent.select_action(agent.history)
            
            obs, reward, done, info_env = env.step(action)
            log_probs.append(info['log_prob'])
            rewards.append(reward)
            
            if isinstance(agent, NormalAgent):
                agent.add_step(obs, action, reward)
            else:
                agent.add_step(action, reward)
            
            if done:
                break
        
        agent.update(log_probs, rewards)
        episode_rewards.append(sum(rewards))
        
        if verbose and (episode + 1) % 100 == 0:
            avg = np.mean(episode_rewards[-100:])
            print(f"  Episode {episode+1}/{num_episodes} | Avg: {avg:.2f}")
    
    return episode_rewards


def evaluate(agent, env, num_episodes=100):
    """评估 Agent"""
    total = 0
    for _ in range(num_episodes):
        obs = env.reset()
        agent.reset_history()
        for _ in range(50):
            if isinstance(agent, NormalAgent):
                action, info = agent.select_action(obs, agent.history)
            else:
                action, info = agent.select_action(agent.history)
            obs, reward, done, _ = env.step(action)
            total += reward
            if isinstance(agent, NormalAgent):
                agent.add_step(obs, action, reward)
            else:
                agent.add_step(action, reward)
            if done:
                break
    return total / num_episodes


# ========================================
# 对比运行
# ========================================

def run_comparison():
    """运行完整对比"""
    print("\n" + "=" * 60)
    print("BreakShell vs 普通 Agent — 完整对比")
    print("=" * 60)
    
    env_train_normal = BreakShellEnv(seed=42)
    env_train_breakshell = BreakShellEnv(seed=42)
    env_eval = BreakShellEnv(seed=999)
    
    normal = NormalAgent(obs_dim=4, action_dim=3, lr=0.005)
    breakshell = BreakShell(action_dim=3, lr=0.005)
    
    print("\n训练普通 Agent（有记忆）...")
    train(normal, env_train_normal, num_episodes=500)
    
    print("\n训练 BreakShell（有自我模型）...")
    train(breakshell, env_train_breakshell, num_episodes=500)
    
    print("\n评估...")
    normal_eval = evaluate(normal, env_eval, num_episodes=100)
    breakshell_eval = evaluate(breakshell, env_eval, num_episodes=100)
    
    print(f"\n{'='*60}")
    print("最终结果")
    print(f"{'='*60}")
    print(f"  普通 Agent（有记忆）: {normal_eval:+.2f}")
    print(f"  BreakShell（有自我模型）: {breakshell_eval:+.2f}")
    print(f"  差异: {breakshell_eval - normal_eval:+.2f}")
    
    if breakshell_eval > normal_eval:
        print(f"  ✓ BreakShell 获胜！自我模型实现功能耦合")
    elif breakshell_eval < normal_eval - 1:
        print(f"  ✗ 普通 Agent 获胜")
    else:
        print(f"  → 平局（差异不显著）")
    
    return {
        'normal': normal_eval,
        'breakshell': breakshell_eval,
        'diff': breakshell_eval - normal_eval
    }


if __name__ == "__main__":
    run_comparison()
