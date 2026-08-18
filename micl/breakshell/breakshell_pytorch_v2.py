# -*- coding: utf-8 -*-
"""
BreakShell PyTorch v2 — 能量管理环境
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))


class EnergyEnv:
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        self.size = 5
        self.reset()
    
    def reset(self):
        self.steps = 0
        self.max_steps = 40
        self.agent_pos = self.rng.randint(0, self.size, size=2)
        self.target_pos = self.rng.randint(0, self.size, size=2)
        while np.array_equal(self.target_pos, self.agent_pos):
            self.target_pos = self.rng.randint(0, self.size, size=2)
        self.energy = 8.0
        return self._obs()
    
    def _obs(self):
        return np.array([*self.agent_pos/self.size, *self.target_pos/self.size, self.energy/10], dtype=np.float32)
    
    def step(self, action):
        self.steps += 1
        cost = [0.5, 2.0, 4.5][action]
        dist = [1, 2, 3][action]
        
        if self.energy >= cost:
            self.energy -= cost
            d = self.target_pos - self.agent_pos
            for i in range(2):
                if d[i] > 0: self.agent_pos[i] = min(self.size-1, self.agent_pos[i]+dist)
                elif d[i] < 0: self.agent_pos[i] = max(0, self.agent_pos[i]-dist)
            
            if np.linalg.norm(self.agent_pos - self.target_pos) < 0.5:
                return self._obs(), 15.0, True, {}
            self.energy = min(10.0, self.energy + 0.3)
            return self._obs(), [0.3, 1.5, 4.0][action] - 0.1, False, {}
        else:
            return self._obs(), -20.0, True, {}


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
    def __init__(self, lr=0.01):
        self.sm = SelfModel()
        self.pol = Policy()
        self.opt = optim.Adam(list(self.sm.parameters()) + list(self.pol.parameters()), lr=lr)
    
    def act(self, obs):
        o = torch.FloatTensor(obs).unsqueeze(0)
        z = self.sm(o)
        p = self.pol(z)
        d = torch.distributions.Categorical(p)
        return d.sample().item(), {'lp': d.log_prob(d.sample()), 'z': z}
    
    def update(self, lps, rews):
        R, G = 0, []
        for r in reversed(rews):
            R = r + 0.99 * R
            G.insert(0, R)
        G = torch.FloatTensor(G)
        bl = G.mean().item()
        A = (G - bl).detach()
        loss = -(torch.stack(lps) * A).sum()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()


class NormalAgent:
    def __init__(self, lr=0.01):
        self.pol = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 3))
        self.opt = optim.Adam(self.pol.parameters(), lr=lr)
    
    def act(self, obs):
        o = torch.FloatTensor(obs).unsqueeze(0)
        p = torch.softmax(self.pol(o), dim=-1)
        d = torch.distributions.Categorical(p)
        return d.sample().item(), {'lp': d.log_prob(d.sample())}
    
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
        lps, rews = [], []
        for _ in range(40):
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
        for _ in range(40):
            a, _ = agent.act(obs)
            obs, r, done, _ = env.step(a)
            total += r
            if done: break
    return total / n


def run():
    print("="*50)
    print("BreakShell vs 普通 Agent")
    print("="*50)
    
    env_n = EnergyEnv(42)
    env_b = EnergyEnv(42)
    env_e = EnergyEnv(999)
    
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
