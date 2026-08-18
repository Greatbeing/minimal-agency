# -*- coding: utf-8 -*-
"""
BreakShell PyTorch — 真正的端到端功能耦合
============================================
解决 numpy 版的核心问题：梯度无法反向传播到编码器

架构：
- SelfModel: obs → z (16维)
- Policy: z → action_probs
- 训练: REINFORCE，梯度从 reward → policy → z → encoder

结果：z 被迫学习编码对动作选择有用的信息
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
# 能力匹配环境
# ========================================

class CapabilityEnv:
    """能力匹配环境"""
    
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        self.size = 5
        self.reset()
    
    def reset(self):
        self.steps = 0
        self.max_steps = 30
        self.agent_pos = self.rng.randint(0, self.size, size=2)
        self.target_pos = self.rng.randint(0, self.size, size=2)
        while np.array_equal(self.target_pos, self.agent_pos):
            self.target_pos = self.rng.randint(0, self.size, size=2)
        self.energy = 10.0
        return self._obs()
    
    def _obs(self):
        return np.array([
            self.agent_pos[0] / self.size,
            self.agent_pos[1] / self.size,
            self.target_pos[0] / self.size,
            self.target_pos[1] / self.size,
            self.energy / 10.0,
        ], dtype=np.float32)
    
    def step(self, action):
        self.steps += 1
        energy_cost = [1.0, 2.5, 4.0][action]
        move_dist = [1, 2, 3][action]
        
        if self.energy < energy_cost:
            reward = -5.0
            done = False
        else:
            self.energy -= energy_cost
            direction = self.target_pos - self.agent_pos
            for i in range(2):
                if direction[i] > 0:
                    self.agent_pos[i] = min(self.size - 1, self.agent_pos[i] + move_dist)
                elif direction[i] < 0:
                    self.agent_pos[i] = max(0, self.agent_pos[i] - move_dist)
            
            dist = np.linalg.norm(self.agent_pos - self.target_pos)
            if dist < 0.5:
                reward = 10.0
                done = True
            else:
                reward = [0.5, 1.5, 3.0][action]
                done = False
            
            self.energy = min(10.0, self.energy + 0.8)
        
        done = done or self.steps >= self.max_steps
        return self._obs(), reward, done, {'energy': self.energy}


# ========================================
# PyTorch BreakShell
# ========================================

class SelfModel(nn.Module):
    """自我模型：obs → z"""
    
    def __init__(self, obs_dim=5, hidden=32, repr_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, repr_dim),
            nn.Tanh()
        )
    
    def forward(self, obs):
        return self.net(obs)


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
        logits = self.net(z)
        return torch.softmax(logits, dim=-1)


class BreakShellPyTorch:
    """BreakShell Agent（PyTorch 版）"""
    
    def __init__(self, obs_dim=5, action_dim=3, lr=0.01):
        self.self_model = SelfModel(obs_dim, repr_dim=16)
        self.policy = Policy(16, action_dim)
        self.optimizer = optim.Adam(
            list(self.self_model.parameters()) + list(self.policy.parameters()),
            lr=lr
        )
        self.action_dim = action_dim
    
    def select_action(self, obs):
        """选择动作"""
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        z = self.self_model(obs_t)
        probs = self.policy(z)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), {'z': z, 'log_prob': dist.log_prob(action), 'probs': probs}
    
    def update(self, log_prob, advantage):
        """REINFORCE 更新"""
        loss = -log_prob * advantage
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def get_si(self, obs):
        """获取自我模型激活（作为 SI 代理）"""
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        z = self.self_model(obs_t)
        # z 的方差作为自我模型活跃度的代理
        return float(z.var())


# ========================================
# 普通 Agent（无自我模型）
# ========================================

class NormalAgentPyTorch:
    """普通 Agent：obs → action（跳过自我模型）"""
    
    def __init__(self, obs_dim=5, action_dim=3, lr=0.01):
        # 直接 obs → action，没有中间的 z
        self.policy = nn.Sequential(
            nn.Linear(obs_dim, 16),
            nn.ReLU(),
            nn.Linear(16, action_dim)
        )
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.action_dim = action_dim
    
    def select_action(self, obs):
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        logits = self.policy(obs_t)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), {'log_prob': dist.log_prob(action), 'probs': probs}
    
    def update(self, log_prob, advantage):
        loss = -log_prob * advantage
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def get_si(self, obs):
        return 0.0  # 无自我模型


# ========================================
# 训练管道
# ========================================

def train_agent(agent, env, num_episodes=500, gamma=0.99):
    """训练 Agent"""
    episode_rewards = []
    baseline = 0.0
    
    for episode in range(num_episodes):
        obs = env.reset()
        log_probs = []
        rewards = []
        
        for step in range(30):
            action, info = agent.select_action(obs)
            next_obs, reward, done, _ = env.step(action)
            log_probs.append(info['log_prob'])
            rewards.append(reward)
            obs = next_obs
            if done:
                break
        
        # 计算回报
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns)
        baseline = 0.95 * baseline + 0.05 * returns.mean().item()
        advantages = returns - baseline
        
        # 累积损失，一次反向传播
        log_probs_t = torch.stack(log_probs)
        advantages_t = advantages.detach()
        total_loss = -(log_probs_t * advantages_t).sum()
        
        agent.optimizer.zero_grad()
        total_loss.backward()
        agent.optimizer.step()
        
        total_reward = sum(rewards)
        episode_rewards.append(total_reward)
        
        if (episode + 1) % 100 == 0:
            avg = np.mean(episode_rewards[-100:])
            print(f"  Episode {episode+1} | Avg: {avg:.2f}")
    
    return episode_rewards


def evaluate_agent(agent, env, num_episodes=50):
    """评估 Agent"""
    rewards = []
    for ep in range(num_episodes):
        obs = env.reset()
        ep_reward = 0.0
        for step in range(30):
            action, info = agent.select_action(obs)
            next_obs, reward, done, _ = env.step(action)
            ep_reward += reward
            obs = next_obs
            if done:
                break
        rewards.append(ep_reward)
    return np.mean(rewards)


# ========================================
# 对比训练
# ========================================

def run_pytorch_comparison():
    """PyTorch 对比训练"""
    print("\n" + "=" * 70)
    print("BreakShell PyTorch — 训练对比")
    print("=" * 70)
    
    env_normal = CapabilityEnv(seed=42)
    env_breakshell = CapabilityEnv(seed=42)
    env_eval = CapabilityEnv(seed=999)
    
    normal = NormalAgentPyTorch(obs_dim=5, action_dim=3, lr=0.01)
    breakshell = BreakShellPyTorch(obs_dim=5, action_dim=3, lr=0.01)
    
    print("\n训练普通 Agent（无自我模型）...")
    normal_rewards = train_agent(normal, env_normal, num_episodes=500)
    
    print("\n训练 BreakShell（有自我模型）...")
    breakshell_rewards = train_agent(breakshell, env_breakshell, num_episodes=500)
    
    # 评估
    print("\n评估...")
    normal_eval = evaluate_agent(normal, env_eval, num_episodes=100)
    breakshell_eval = evaluate_agent(breakshell, env_eval, num_episodes=100)
    
    print(f"\n{'='*70}")
    print("训练结果对比")
    print(f"{'='*70}")
    print(f"  普通 Agent（无自我模型）: {normal_eval:+.2f}")
    print(f"  BreakShell（有自我模型）: {breakshell_eval:+.2f}")
    
    diff = breakshell_eval - normal_eval
    print(f"\n  差异: {diff:+.2f}")
    
    if diff > 0:
        print(f"  ✓ BreakShell 获胜！自我模型实现功能耦合")
    elif diff < -1:
        print(f"  ✗ 普通 Agent 获胜（自我模型是噪声）")
    else:
        print(f"  → 平局（差异不显著）")
    
    # 分析自我模型学到了什么
    print(f"\n{'='*70}")
    print("自我模型分析")
    print(f"{'='*70}")
    
    # 测试不同能量状态下的 z
    test_obs_low = np.array([0.2, 0.2, 0.8, 0.8, 0.2], dtype=np.float32)  # 低能量
    test_obs_high = np.array([0.2, 0.2, 0.8, 0.8, 0.9], dtype=np.float32)  # 高能量
    
    z_low = breakshell.self_model(torch.FloatTensor(test_obs_low).unsqueeze(0))
    z_high = breakshell.self_model(torch.FloatTensor(test_obs_high).unsqueeze(0))
    
    print(f"  低能量 z 均值: {z_low.mean().item():.3f}")
    print(f"  高能量 z 均值: {z_high.mean().item():.3f}")
    print(f"  z 差异: {(z_high.mean() - z_low.mean()).item():.3f}")
    
    if abs(z_high.mean().item() - z_low.mean().item()) > 0.1:
        print(f"  ✓ 自我模型编码了能量信息！")
    else:
        print(f"  → 自我模型未明显编码能量信息")
    
    return {
        'normal': normal_eval,
        'breakshell': breakshell_eval,
        'diff': diff,
    }


if __name__ == "__main__":
    results = run_pytorch_comparison()
