# -*- coding: utf-8 -*-
"""
金融 Agent 对比实验
====================
传统交易 Agent vs BreakShell 交易 Agent
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from financial_env import FinancialEnv


# ========================================
# 传统交易 Agent（无自我模型）
# ========================================

class TradingAgent(nn.Module):
    """
    传统交易 Agent
    架构：obs → LSTM → 仓位决策
    """
    
    def __init__(self, obs_dim=5, hidden=32, lr=0.005):
        super().__init__()
        self.lstm = nn.LSTM(obs_dim + 3 + 1, hidden, batch_first=True)  # obs + action + reward
        self.policy = nn.Linear(hidden, 3)
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
        return a.item(), {'lp': d.log_prob(a), 'probs': p}
    
    def add_step(self, obs, action, reward):
        onehot = np.zeros(3, dtype=np.float32)
        onehot[action] = 1.0
        self.history.append(np.concatenate([obs, onehot, [reward]]))
    
    def reset_history(self):
        self.history = []
    
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
        return loss.item()


# ========================================
# BreakShell 交易 Agent（有自我模型）
# ========================================

class TradingSelfModel(nn.Module):
    """
    交易自我模型
    编码历史 → 推断市场状态和自身能力
    """
    
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


class BreakShellTradingAgent(nn.Module):
    """
    BreakShell 交易 Agent
    架构：history → SelfModel → z → Policy → action
    
    关键区别：
    - 传统 Agent 记忆完整的 (obs, action, reward)
    - BreakShell 只记录 (action, reward) → 推断能力边界
    """
    
    def __init__(self, lr=0.005):
        super().__init__()
        self.self_model = TradingSelfModel()
        self.policy = nn.Sequential(
            nn.Linear(self.self_model.repr_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )
        self.opt = optim.Adam(self.parameters(), lr=lr)
        self.reset_history()
    
    def forward(self, history):
        z = self.self_model(history)
        return torch.softmax(self.policy(z), dim=-1)
    
    def act(self, history):
        p = self.forward(history)
        d = torch.distributions.Categorical(p)
        a = d.sample()
        return a.item(), {'lp': d.log_prob(a), 'probs': p}
    
    def add_step(self, action, reward):
        onehot = np.zeros(3, dtype=np.float32)
        onehot[action] = 1.0
        self.history.append(np.concatenate([onehot, [reward]]))
    
    def reset_history(self):
        self.history = []
    
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
        return loss.item()


# ========================================
# 训练与回测
# ========================================

def train_agent(agent, env, n=300):
    """训练"""
    for ep in range(n):
        obs = env.reset()
        agent.reset_history()
        lps, rews = [], []
        
        for step in range(200):
            if isinstance(agent, BreakShellTradingAgent):
                a, info = agent.act(agent.history)
            else:
                a, info = agent.act(agent.history)
            
            obs, r, done, _ = env.step(a)
            lps.append(info['lp'])
            rews.append(r)
            
            if isinstance(agent, BreakShellTradingAgent):
                agent.add_step(a, r)
            else:
                agent.add_step(obs, a, r)
            
            if done:
                break
        
        agent.update(lps, rews)
        if (ep + 1) % 50 == 0:
            print(f"  {ep+1}/300")


def backtest(agent, env, n=50):
    """回测"""
    returns = []
    for _ in range(n):
        obs = env.reset()
        agent.reset_history()
        ep_ret = 0
        for step in range(200):
            if isinstance(agent, BreakShellTradingAgent):
                a, info = agent.act(agent.history)
            else:
                a, info = agent.act(agent.history)
            obs, r, done, _ = env.step(a)
            ep_ret += r
            if isinstance(agent, BreakShellTradingAgent):
                agent.add_step(a, r)
            else:
                agent.add_step(obs, a, r)
            if done:
                break
        returns.append(ep_ret)
    return np.mean(returns)


def run():
    print("="*60)
    print("金融 Agent 对比实验")
    print("="*60)
    
    env_train_normal = FinancialEnv(seed=42)
    env_train_breakshell = FinancialEnv(seed=42)
    env_eval = FinancialEnv(seed=999)
    
    normal = TradingAgent()
    breakshell = BreakShellTradingAgent()
    
    print("\n训练传统 Agent...")
    train_agent(normal, env_train_normal)
    print("训练 BreakShell...")
    train_agent(breakshell, env_train_breakshell)
    
    print("\n回测...")
    normal_ret = backtest(normal, env_eval)
    breakshell_ret = backtest(breakshell, env_eval)
    
    print(f"\n{'='*60}")
    print("回测结果")
    print(f"{'='*60}")
    print(f"  传统 Agent: {normal_ret:+.4f}")
    print(f"  BreakShell: {breakshell_ret:+.4f}")
    print(f"  差异: {breakshell_ret - normal_ret:+.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run()
