# -*- coding: utf-8 -*-
"""
BreakShell PyTorch v4 — 自我知识必要环境
==========================================
核心设计：观察隐藏能力，能力只能通过历史推断

环境：
- obs = [位置, 目标方向]（没有能力信息）
- 能力隐藏在动作结果中
- 激进动作成功 → 说明能力高
- 激进动作失败 → 说明能力低
- 普通 Agent：无法从历史推断能力 → 盲目
- BreakShell：自我模型编码历史 → 推断能力 → 匹配动作
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))


class HiddenCapabilityEnv:
    """能力隐藏在历史中"""
    
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        self.size = 4
        self.reset()
    
    def reset(self):
        self.steps = 0
        self.max_steps = 60
        self.agent_pos = self.rng.randint(0, self.size, size=2)
        self.target_pos = self.rng.randint(0, self.size, size=2)
        while np.array_equal(self.target_pos, self.agent_pos):
            self.target_pos = self.rng.randint(0, self.size, size=2)
        self.freq = self.rng.uniform(0.3, 0.7)
        self.phase = self.rng.uniform(0, 2*np.pi)
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
        thresholds = [0.0, 0.4, 0.7]
        rewards = [0.5, 2.0, 5.0]
        penalties = [-1.0, -5.0, -20.0]
        
        if cap >= thresholds[action]:
            reward = rewards[action]
            move = [1, 2, 3][action]
            d = self.target_pos - self.agent_pos
            for i in range(2):
                if d[i] > 0: self.agent_pos[i] = min(self.size-1, self.agent_pos[i]+move)
                elif d[i] < 0: self.agent_pos[i] = max(0, self.agent_pos[i]-move)
            done = np.linalg.norm(self.agent_pos - self.target_pos) < 0.5
            if done: reward = 20.0
        else:
            reward = penalties[action]
            done = False
        
        done = done or self.steps >= self.max_steps
        return self._obs(), reward, done, {'capability': cap}


# ========================================
# BreakShell with history encoding
# ========================================

class HistorySelfModel(nn.Module):
    """自我模型：编码历史轨迹 → 推断能力"""
    
    def __init__(self, action_dim=3, hidden=32, repr_dim=16):
        super().__init__()
        # 编码单步 (action, reward)
        self.step_encoder = nn.Linear(action_dim + 1, hidden)
        # LSTM 编码序列
        self.lstm = nn.LSTM(hidden, repr_dim, batch_first=True)
        self.repr_dim = repr_dim
    
    def forward(self, history):
        """
        history: (batch, seq_len, action_dim + 1)
        output: (batch, repr_dim)
        """
        h = torch.relu(self.step_encoder(history))
        out, (hn, cn) = self.lstm(h)
        return hn[-1]  # 最后时刻的隐状态作为 z


class Policy(nn.Module):
    def __init__(self, in_dim=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 16), nn.ReLU(), nn.Linear(16, 3))
    def forward(self, x): return torch.softmax(self.net(x), dim=-1)


class BreakShell:
    def __init__(self, lr=0.005):
        self.sm = HistorySelfModel()
        self.pol = Policy()
        self.opt = optim.Adam(list(self.sm.parameters()) + list(self.pol.parameters()), lr=lr)
        self.history = []  # 历史轨迹
        self.action_dim = 3
    
    def act(self, obs):
        # 编码历史
        if len(self.history) == 0:
            hist = torch.zeros(1, 1, self.action_dim + 1)
        else:
            hist = torch.FloatTensor(self.history).unsqueeze(0)
        
        z = self.sm(hist)
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
    
    def add_step(self, action, reward):
        """添加一步到历史"""
        onehot = np.zeros(self.action_dim, dtype=np.float32)
        onehot[action] = 1.0
        self.history.append(np.concatenate([onehot, [reward]]))
    
    def reset_history(self):
        self.history = []


class NormalAgent:
    def __init__(self, lr=0.005):
        self.pol = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 3))
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


def train(agent, env, n=500):
    for ep in range(n):
        obs = env.reset()
        if hasattr(agent, 'reset_history'):
            agent.reset_history()
        lps, rews = [], []
        for _ in range(60):
            a, info = agent.act(obs)
            obs, r, done, _ = env.step(a)
            lps.append(info['lp'])
            rews.append(r)
            if hasattr(agent, 'add_step'):
                agent.add_step(a, r)
            if done: break
        agent.update(lps, rews)
        if (ep+1) % 100 == 0:
            print(f"  {ep+1}/500")

def evaluate(agent, env, n=100):
    total = 0
    for _ in range(n):
        obs = env.reset()
        if hasattr(agent, 'reset_history'):
            agent.reset_history()
        for _ in range(60):
            a, _ = agent.act(obs)
            obs, r, done, _ = env.step(a)
            total += r
            if hasattr(agent, 'add_step'):
                agent.add_step(a, r)
            if done: break
    return total / n


def run():
    print("="*50)
    print("自我知识必要环境（隐藏能力）")
    print("="*50)
    
    env_n = HiddenCapabilityEnv(42)
    env_b = HiddenCapabilityEnv(42)
    env_e = HiddenCapabilityEnv(999)
    
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
