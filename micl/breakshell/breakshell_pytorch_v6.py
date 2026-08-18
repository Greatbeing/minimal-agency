# -*- coding: utf-8 -*-
"""
BreakShell v6 — 真正的差异
============================
核心洞察：自我模型 ≠ 记忆，自我模型 = 对自身能力的推断

场景：
- Agent 有 2 种能力：力量 + 速度（独立变化）
- 3 种任务：需要 (高力量) / (高速度) / (高力量+高速度)
- 观察中不包含能力信息
- 只能通过"尝试→结果"推断能力
- 选超出能力的任务 → 灾难性失败

普通 Agent + LSTM：记住"上次选任务1失败了"
BreakShell：推断"我力量不足，但速度够 → 选任务2"
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))


class MultiCapabilityEnv:
    """多能力环境：力量 + 速度独立变化"""
    
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        self.reset()
    
    def reset(self):
        self.steps = 0
        self.max_steps = 60
        # 两种能力（独立随机游走）
        self.strength = self.rng.uniform(0.2, 0.8)
        self.speed = self.rng.uniform(0.2, 0.8)
        return self._obs()
    
    def _obs(self):
        # 观察中不包含能力信息！
        return np.array([0.0, 0.0], dtype=np.float32)  # 固定观察
    
    def step(self, action):
        self.steps += 1
        
        # 任务需求：[高力量, 高速度, 高力量+高速度]
        req_strength = [0.7, 0.3, 0.8][action]
        req_speed = [0.3, 0.7, 0.8][action]
        
        # 检查能力是否满足
        if self.strength >= req_strength and self.speed >= req_speed:
            reward = (action + 1) * 5.0  # 任务 3 奖励最高
        else:
            reward = -20.0  # 高估自己 = 灾难
        
        # 能力随机游走
        self.strength += self.rng.normal(0, 0.05)
        self.speed += self.rng.normal(0, 0.05)
        self.strength = np.clip(self.strength, 0.1, 0.9)
        self.speed = np.clip(self.speed, 0.1, 0.9)
        
        done = self.steps >= self.max_steps
        return self._obs(), reward, done, {
            'strength': self.strength,
            'speed': self.speed,
        }


# ========================================
# 普通 Agent + LSTM 记忆
# ========================================

class NormalAgent(nn.Module):
    """普通 Agent：LSTM 编码 (obs, action, reward)"""
    
    def __init__(self, lr=0.005):
        super().__init__()
        self.lstm = nn.LSTM(2 + 3 + 1, 32, batch_first=True)
        self.policy = nn.Linear(32, 3)
        self.opt = optim.Adam(self.parameters(), lr=lr)
        self.reset_history()
    
    def forward(self, history):
        if len(history) == 0:
            h = torch.zeros(1, 1, 32)
        else:
            h = torch.FloatTensor(np.array(history)).unsqueeze(0)
            out, (h, c) = self.lstm(h)
        logits = self.policy(h[-1])
        return torch.softmax(logits, dim=-1)
    
    def act(self, history):
        p = self.forward(history)
        d = torch.distributions.Categorical(p)
        a = d.sample()
        return a.item(), {'lp': d.log_prob(a)}
    
    def add_step(self, obs, action, reward):
        onehot = np.zeros(3, dtype=np.float32)
        onehot[action] = 1.0
        self.history.append(np.concatenate([obs, onehot, [reward]]))
    
    def reset_history(self):
        self.history = []


# ========================================
# BreakShell：自我模型
# ========================================

class BreakShell(nn.Module):
    """
    BreakShell：自我模型编码 (action, reward) → 推断能力
    
    关键区别：
    - 普通 Agent 记忆完整的 (obs, action, reward)
    - BreakShell 只记录 (action, reward) → 推断"我能做什么"
    """
    
    def __init__(self, lr=0.005):
        super().__init__()
        # 自我模型：编码 (action, reward) → 能力表征
        self.self_model = nn.LSTM(3 + 1, 32, batch_first=True)
        # 策略：基于能力表征
        self.policy = nn.Linear(32, 3)
        self.opt = optim.Adam(self.parameters(), lr=lr)
        self.reset_history()
    
    def forward(self, history):
        if len(history) == 0:
            h = torch.zeros(1, 1, 32)
        else:
            h = torch.FloatTensor(np.array(history)).unsqueeze(0)
            out, (h, c) = self.self_model(h)
        logits = self.policy(h[-1])
        return torch.softmax(logits, dim=-1)
    
    def act(self, history):
        p = self.forward(history)
        d = torch.distributions.Categorical(p)
        a = d.sample()
        return a.item(), {'lp': d.log_prob(a)}
    
    def add_step(self, action, reward):
        """只记录 action-reward（不记录 obs）"""
        onehot = np.zeros(3, dtype=np.float32)
        onehot[action] = 1.0
        self.history.append(np.concatenate([onehot, [reward]]))
    
    def reset_history(self):
        self.history = []


# ========================================
# 训练
# ========================================

def train(agent, env, n=500):
    for ep in range(n):
        obs = env.reset()
        agent.reset_history()
        lps, rews = [], []
        for _ in range(60):
            a, info = agent.act(agent.history)
            obs, r, done, _ = env.step(a)
            lps.append(info['lp'])
            rews.append(r)
            if isinstance(agent, NormalAgent):
                agent.add_step(obs, a, r)
            else:
                agent.add_step(a, r)
            if done: break
        
        R, G = 0, []
        for r in reversed(rews):
            R = r + 0.99 * R
            G.insert(0, R)
        G = torch.FloatTensor(G)
        A = (G - G.mean()).detach()
        loss = -(torch.stack(lps) * A).sum()
        agent.opt.zero_grad()
        loss.backward()
        agent.opt.step()
        
        if (ep+1) % 100 == 0:
            print(f"  {ep+1}/500")

def evaluate(agent, env, n=100):
    total = 0
    for _ in range(n):
        obs = env.reset()
        agent.reset_history()
        for _ in range(60):
            a, _ = agent.act(agent.history)
            obs, r, done, _ = env.step(a)
            total += r
            if isinstance(agent, NormalAgent):
                agent.add_step(obs, a, r)
            else:
                agent.add_step(a, r)
            if done: break
    return total / n


def run():
    print("="*60)
    print("BreakShell vs 普通 Agent（都有记忆）")
    print("="*60)
    
    env_n = MultiCapabilityEnv(42)
    env_b = MultiCapabilityEnv(42)
    env_e = MultiCapabilityEnv(999)
    
    normal = NormalAgent()
    bs = BreakShell()
    
    print("\n训练普通 Agent...")
    train(normal, env_n)
    print("训练 BreakShell...")
    train(bs, env_b)
    
    ne = evaluate(normal, env_e)
    be = evaluate(bs, env_e)
    
    print(f"\n{'='*60}")
    print(f"普通 Agent（LSTM 记忆）: {ne:+.2f}")
    print(f"BreakShell（自我模型）: {be:+.2f}")
    print(f"差异: {be-ne:+.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run()
