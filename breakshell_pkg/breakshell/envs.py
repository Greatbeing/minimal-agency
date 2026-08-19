# -*- coding: utf-8 -*-
"""
BreakShell 环境库
"""

import numpy as np
from typing import Dict, Tuple


class CapabilityEnv:
    """
    能力匹配环境
    
    观察中隐藏能力信息，能力只能通过历史轨迹推断。
    高估自己 = 灾难性惩罚。
    """
    
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        self.size = 4
        self.max_steps = 50
        self.reset()
    
    def reset(self):
        self.steps = 0
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


class EnergyEnv:
    """能量管理环境"""
    
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        self.size = 5
        self.max_steps = 40
        self.reset()
    
    def reset(self):
        self.steps = 0
        self.agent_pos = self.rng.randint(0, self.size, size=2)
        self.target_pos = self.rng.randint(0, self.size, size=2)
        while np.array_equal(self.target_pos, self.agent_pos):
            self.target_pos = self.rng.randint(0, self.size, size=2)
        self.energy = 8.0
        return self._obs()
    
    def _obs(self):
        return np.array([
            *self.agent_pos / self.size,
            *self.target_pos / self.size,
            self.energy / 10.0,
        ], dtype=np.float32)
    
    def step(self, action):
        self.steps += 1
        cost = [0.5, 2.0, 4.5][action]
        move = [1, 2, 3][action]
        
        if self.energy >= cost:
            self.energy -= cost
            d = self.target_pos - self.agent_pos
            for i in range(2):
                if d[i] > 0:
                    self.agent_pos[i] = min(self.size - 1, self.agent_pos[i] + move)
                elif d[i] < 0:
                    self.agent_pos[i] = max(0, self.agent_pos[i] - move)
            if np.linalg.norm(self.agent_pos - self.target_pos) < 0.5:
                return self._obs(), 15.0, True, {}
            self.energy = min(10.0, self.energy + 0.3)
            return self._obs(), [0.3, 1.5, 4.0][action] - 0.1, False, {}
        else:
            return self._obs(), -20.0, True, {}


class FinancialEnv:
    """金融市场环境"""
    
    def __init__(self, seed=42, initial_price=100.0):
        self.rng = np.random.RandomState(seed)
        self.initial_price = initial_price
        self.max_steps = 200
        self.reset()
    
    def reset(self):
        self.step_idx = 0
        self.price = self.initial_price
        self.volatility = 0.02
        self.liquidity = 1.0
        self.regime = 'normal'
        self.regime_length = 0
        self.portfolio_value = 100000.0
        self.position = 0
        return self._obs()
    
    def _obs(self):
        return np.array([
            self.price / self.initial_price - 1.0,
            self.volatility / 0.02 - 1.0,
            self.liquidity - 1.0,
            self.portfolio_value / 100000.0 - 1.0,
            self.position / 100.0,
        ], dtype=np.float32)
    
    def _update_regime(self):
        self.regime_length += 1
        if self.regime_length > 50 + self.rng.randint(0, 30):
            regimes = ['normal', 'extreme', 'quiet']
            self.regime = self.rng.choice(regimes, p=[0.6, 0.2, 0.2])
            self.regime_length = 0
            if self.regime == 'extreme':
                self.volatility = 0.05 + self.rng.uniform(0, 0.03)
                self.liquidity = 0.5 + self.rng.uniform(0, 0.3)
            elif self.regime == 'quiet':
                self.volatility = 0.005 + self.rng.uniform(0, 0.01)
                self.liquidity = 1.0 + self.rng.uniform(0, 0.2)
            else:
                self.volatility = 0.015 + self.rng.uniform(0, 0.01)
                self.liquidity = 0.9 + self.rng.uniform(0, 0.2)
    
    def step(self, action):
        self.step_idx += 1
        self._update_regime()
        ret = self.rng.normal(0, self.volatility)
        self.price *= (1 + ret)
        cost = (1.0 - self.liquidity) * 0.001
        target_pos = [0, 50, 100][action]
        trade = target_pos - self.position
        self.position = target_pos
        pnl = self.position * ret * 1000 - abs(trade) * cost * 100
        self.portfolio_value += pnl
        reward = pnl / 10000
        done = self.portfolio_value < 50000 or self.step_idx >= self.max_steps
        return self._obs(), reward, done, {'regime': self.regime}
