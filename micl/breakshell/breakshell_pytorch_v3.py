# -*- coding: utf-8 -*-
"""
BreakShell PyTorch v3 — 能力波动环境
======================================
核心设计：能力周期性变化，高估自己 = 灾难

环境设计：
- Agent 有"力量值"（随时间正弦波动）
- 高力量期：激进动作最优（高奖励）
- 低力量期：必须保守（激进 → 灾难性惩罚 -20）
- 普通 Agent：看不到力量值 → 盲目 → 灾难
- BreakShell：自我模型追踪力量值 → 匹配最优动作
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))


# ========================================
# 能力波动环境
# ========================================

class FluctuatingCapabilityEnv:
    """
    能力波动环境
    
    关键：Agent 的能力周期性变化，需要自知
    """
    
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        self.size = 4
        self.reset()
    
    def reset(self):
        self.steps = 0
        self.max_steps = 60  # 更多步数，让波动显现
        self.agent_pos = self.rng.randint(0, self.size, size=2)
        self.target_pos = self.rng.randint(0, self.size, size=2)
        while np.array_equal(self.target_pos, self.agent_pos):
            self.target_pos = self.rng.randint(0, self.size, size=2)
        self.freq = self.rng.uniform(0.3, 0.7)  # 波动频率
        self.phase = self.rng.uniform(0, 2*np.pi)
        return self._obs()
    
    def _capability(self):
        """当前能力（正弦波动 0.1-0.9）"""
        t = self.steps * 0.5
        return 0.5 + 0.4 * np.sin(t * self.freq + self.phase)
    
    def _obs(self):
        cap = self._capability()
        return np.array([
            *self.agent_pos / self.size,
            *self.target_pos / self.size,
            cap,  # 能力值（隐藏信息，需要推断）
        ], dtype=np.float32)
    
    def step(self, action):
        self.steps += 1
        cap = self._capability()
        
        # 动作：[保守, 适中, 激进]
        # 保守：总是成功，低奖励
        # 适中：需要 cap > 0.4
        # 激进：需要 cap > 0.7，高奖励
        thresholds = [0.0, 0.4, 0.7]
        rewards = [0.5, 2.0, 5.0]
        penalties = [-1.0, -5.0, -20.0]  # 高估自己 = 灾难
        
        if cap >= thresholds[action]:
            # 成功
            reward = rewards[action]
            # 向目标移动
            move = [1, 2, 3][action]
            d = self.target_pos - self.agent_pos
            for i in range(2):
                if d[i] > 0: self.agent_pos[i] = min(self.size-1, self.agent_pos[i]+move)
                elif d[i] < 0: self.agent_pos[i] = max(0, self.agent_pos[i]-move)
            
            done = np.linalg.norm(self.agent_pos - self.target_pos) < 0.5
            if done: reward = 20.0
        else:
            # 高估自己 → 惩罚
            reward = penalties[action]
            done = False
        
        done = done or self.steps >= self.max_steps
        return self._obs(), reward, done, {'capability': cap}


# ========================================
# 网络
# ========================================

class SelfModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 16), nn.Tanh())
    def forward(self, x): return self.net(x)


class Policy(nn.Module):
    def __init__(self, in_dim=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 16), nn.ReLU(), nn.Linear(16, 3))
    def forward(self, x): return torch.softmax(self.net(x), dim=-1)


class BreakShell:
    def __init__(self, lr=0.005):
        self.sm = SelfModel()
        self.pol = Policy()
        self.opt = optim.Adam(list(self.sm.parameters()) + list(self.pol.parameters()), lr=lr)
    
    def act(self, obs):
        o = torch.FloatTensor(obs).unsqueeze(0)
        z = self.sm(o)
        p = self.pol(z)
        d = torch.distributions.Categorical(p)
        a = d.sample()
        return a.item(), {'lp': d.log_prob(a), 'z': z}
    
    def update(self, lps, rews):
        R, G = 0, []
        for r in reversed(rews):
            R = r + 0.99 * R
            G.insert(0, R)
        G = torch.FloatTensor(G)
        A = (G - G.mean()).detach()
        loss = -(torch.stack(lps) * A).sum()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()


class NormalAgent:
    def __init__(self, lr=0.005):
        self.pol = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 3))
        self.opt = optim.Adam(self.pol.parameters(), lr=lr)
    
    def act(self, obs):
        o = torch.FloatTensor(obs).unsqueeze(0)
        p = torch.softmax(self.pol(o), dim=-1)
        d = torch.distributions.Categorical(p)
        a = d.sample()
        return a.item(), {'lp': d.log_prob(a)}
    
    def update(self, lps, rews):
        R, G = 0, []
        for r in reversed(rews):
            R = r + 0.99 * R
            G.insert(0, R)
        G = torch.FloatTensor(G)
        A = (G - G.mean()).detach()
        loss = -(torch.stack(lps) * A).sum()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()


# ========================================
# 训练
# ========================================

def train(agent, env, n=500):
    for ep in range(n):
        obs = env.reset()
        lps, rews = [], []
        for _ in range(60):
            a, info = agent.act(obs)
            obs, r, done, _ = env.step(a)
            lps.append(info['lp'])
            rews.append(r)
            if done: break
        agent.update(lps, rews)
        if (ep+1) % 100 == 0:
            print(f"  {ep+1}/500")

def evaluate(agent, env, n=100):
    total = 0
    for _ in range(n):
        obs = env.reset()
        for _ in range(60):
            a, _ = agent.act(obs)
            obs, r, done, _ = env.step(a)
            total += r
            if done: break
    return total / n


def run():
    print("="*50)
    print("能力波动环境")
    print("="*50)
    
    env_n = FluctuatingCapabilityEnv(42)
    env_b = FluctuatingCapabilityEnv(42)
    env_e = FluctuatingCapabilityEnv(999)
    
    normal = NormalAgent()
    bs = BreakShell()
    
    print("\n训练普通 Agent...")
    train(normal, env_n)
    print("训练 BreakShell...")
    train(bs, env_b)
    
    ne = evaluate(normal, env_e)
    be = evaluate(bs, env_e)
    
    print(f"\n{'='*50}")
    print(f"普通 Agent: {ne:+.2f}")
    print(f"BreakShell: {be:+.2f}")
    print(f"差异: {be-ne:+.2f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    run()
