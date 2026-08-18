# -*- coding: utf-8 -*-
"""
BreakShell v5 — 自我模型 vs 记忆
==================================
核心差异：普通 Agent 记住"发生了什么"，BreakShell 建模"我能做什么"

场景：
- 3 种任务：简单/中等/困难（需要不同能力）
- Agent 能力随时间变化
- 选超出能力的任务 → 灾难性失败
- 普通 Agent：有记忆，但无法判断自己能力 → 盲目选
- BreakShell：自我模型从历史推断能力 → 正确匹配任务
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))


class MultiTaskEnv:
    """多任务环境：需要不同能力"""
    
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        self.tasks = [0.2, 0.5, 0.8]  # 简单/中等/困难所需能力
        self.task_names = ['简单', '中等', '困难']
        self.reset()
    
    def reset(self):
        self.steps = 0
        self.max_steps = 60
        self.capability = self.rng.uniform(0.2, 0.8)
        self.target_pos = self.rng.randint(0, 4, size=2)
        self.agent_pos = self.rng.randint(0, 4, size=2)
        while np.array_equal(self.target_pos, self.agent_pos):
            self.target_pos = self.rng.randint(0, 4, size=2)
        return self._obs()
    
    def _obs(self):
        return np.array([
            *self.agent_pos / 4,
            *self.target_pos / 4,
        ], dtype=np.float32)
    
    def step(self, action):
        self.steps += 1
        required_cap = self.tasks[action]
        
        if self.capability >= required_cap:
            # 成功
            reward = (action + 1) * 3.0  # 困难任务奖励更高
            # 能力小幅提升
            self.capability = min(1.0, self.capability + 0.05)
        else:
            # 失败 → 灾难性惩罚
            reward = -15.0
            # 能力下降
            self.capability = max(0.1, self.capability - 0.1)
        
        # 随机游走
        self.capability += self.rng.normal(0, 0.05)
        self.capability = np.clip(self.capability, 0.1, 0.9)
        
        done = self.steps >= self.max_steps
        return self._obs(), reward, done, {'capability': self.capability}


# ========================================
# 普通 Agent（有记忆，无自我模型）
# ========================================

class NormalAgentWithMemory(nn.Module):
    """有记忆的普通 Agent：LSTM 编码 obs-action-reward"""
    
    def __init__(self, lr=0.005):
        super().__init__()
        self.lstm = nn.LSTM(4 + 3 + 1, 32, batch_first=True)  # obs + action_onehot + reward
        self.policy = nn.Linear(32, 3)
        self.opt = optim.Adam(self.parameters(), lr=lr)
        self.history = []
    
    def forward(self, obs, history):
        if len(history) == 0:
            h = torch.zeros(1, 1, 32)
        else:
            h = torch.FloatTensor(history).unsqueeze(0)
            out, (h, c) = self.lstm(h)
        logits = self.policy(h[-1])
        return torch.softmax(logits, dim=-1)
    
    def act(self, obs, history):
        p = self.forward(obs, history)
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
# BreakShell（自我模型）
# ========================================

class BreakShellV5(nn.Module):
    """BreakShell：自我模型明确建模能力"""
    
    def __init__(self, lr=0.005):
        super().__init__()
        # 自我模型：编码历史 action-reward → 推断能力
        self.self_model = nn.LSTM(3 + 1, 32, batch_first=True)
        # 策略：基于自我模型 + 当前 obs
        self.policy = nn.Linear(32 + 4, 3)
        self.opt = optim.Adam(self.parameters(), lr=lr)
        self.history = []
    
    def forward(self, obs, history):
        if len(history) == 0:
            h_self = torch.zeros(1, 1, 32)
        else:
            h = torch.FloatTensor(history).unsqueeze(0)
            out, (h_self, c) = self.self_model(h)
        
        # 拼接自我模型 + 当前 obs
        z = h_self[-1]
        x = torch.cat([z, torch.FloatTensor(obs).unsqueeze(0)], dim=-1)
        logits = self.policy(x)
        return torch.softmax(logits, dim=-1)
    
    def act(self, obs, history):
        p = self.forward(obs, history)
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
            a, info = agent.act(obs, agent.history)
            obs, r, done, _ = env.step(a)
            lps.append(info['lp'])
            rews.append(r)
            if hasattr(agent, 'add_step'):
                if isinstance(agent, NormalAgentWithMemory):
                    agent.add_step(obs, a, r)
                else:
                    agent.add_step(a, r)
            if done: break
        
        # REINFORCE
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
            a, _ = agent.act(obs, agent.history)
            obs, r, done, _ = env.step(a)
            total += r
            if hasattr(agent, 'add_step'):
                if isinstance(agent, NormalAgentWithMemory):
                    agent.add_step(obs, a, r)
                else:
                    agent.add_step(a, r)
            if done: break
    return total / n


def run():
    print("="*60)
    print("BreakShell vs 普通 Agent（都有记忆）")
    print("="*60)
    
    env_n = MultiTaskEnv(42)
    env_b = MultiTaskEnv(42)
    env_e = MultiTaskEnv(999)
    
    normal = NormalAgentWithMemory()
    bs = BreakShellV5()
    
    print("\n训练普通 Agent（有记忆）...")
    train(normal, env_n)
    print("训练 BreakShell（有自我模型）...")
    train(bs, env_b)
    
    ne = evaluate(normal, env_e)
    be = evaluate(bs, env_e)
    
    print(f"\n{'='*60}")
    print(f"普通 Agent（有记忆）: {ne:+.2f}")
    print(f"BreakShell（有自我模型）: {be:+.2f}")
    print(f"差异: {be-ne:+.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run()
