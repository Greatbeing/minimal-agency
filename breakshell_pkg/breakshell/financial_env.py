# -*- coding: utf-8 -*-
"""
金融市场模拟器
==============
支持：价格生成、波动率变化、流动性变化、市场机制切换
"""

import numpy as np
from typing import Dict, Tuple, Optional


class FinancialEnv:
    """
    金融市场环境
    
    状态：
    - 价格序列（随机游走 + 趋势 + 波动率聚类）
    - 波动率（GARCH-like）
    - 流动性（买卖价差）
    - 市场机制（正常/极端/低迷）
    """
    
    def __init__(self, seed=42, initial_price=100.0):
        self.rng = np.random.RandomState(seed)
        self.initial_price = initial_price
        self.reset()
    
    def reset(self):
        self.step_idx = 0
        self.price = self.initial_price
        self.volatility = 0.02  # 基础波动率
        self.liquidity = 1.0    # 流动性（1=正常，<1=差）
        self.regime = 'normal'  # normal/extreme/quiet
        self.regime_length = 0
        self.portfolio_value = 100000.0  # 初始资金
        self.position = 0  # 持仓（-100 到 100）
        self.done = False
        return self._obs()
    
    def _obs(self):
        """
        观察（交易中可获取的信息）
        注意：不包含 regime 信息！Agent 必须推断
        """
        return np.array([
            self.price / self.initial_price - 1.0,  # 收益率
            self.volatility / 0.02 - 1.0,             # 波动率偏离
            self.liquidity - 1.0,                     # 流动性偏离
            self.portfolio_value / 100000.0 - 1.0,    # 组合收益
            self.position / 100.0,                    # 当前仓位
        ], dtype=np.float32)
    
    def _update_regime(self):
        """更新市场机制（Hidden）"""
        self.regime_length += 1
        
        # 市场机制切换
        if self.regime_length > 50 + self.rng.randint(0, 30):
            regimes = ['normal', 'extreme', 'quiet']
            probs = [0.6, 0.2, 0.2]
            self.regime = self.rng.choice(regimes, p=probs)
            self.regime_length = 0
            
            # 根据机制调整参数
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
        """
        action: 0=空仓, 1=半仓, 2=满仓
        """
        self.step_idx += 1
        self._update_regime()
        
        # 价格更新（带波动率聚类）
        ret = self.rng.normal(0, self.volatility)
        self.price *= (1 + ret)
        
        # 交易成本（流动性影响）
        cost = (1.0 - self.liquidity) * 0.001
        
        # 仓位调整
        target_position = [0, 50, 100][action]
        trade = target_position - self.position
        self.position = target_position
        
        # 收益
        pnl = self.position * ret * 1000 - abs(trade) * cost * 100
        self.portfolio_value += pnl
        
        # 检查是否爆仓
        if self.portfolio_value < 50000:
            self.done = True
        
        reward = pnl / 10000  # 归一化
        
        return self._obs(), reward, self.done, {
            'regime': self.regime,
            'volatility': self.volatility,
            'liquidity': self.liquidity,
        }
    
    def obs_dim(self):
        return 5


if __name__ == "__main__":
    env = FinancialEnv()
    for i in range(100):
        obs, reward, done, info = env.step(1)
        if i % 20 == 0:
            print(f"Step {i}: regime={info['regime']}, vol={info['volatility']:.3f}, portfolio={env.portfolio_value:.0f}")
        if done:
            print(f"爆仓！步数={i}")
            break
    print(f"最终组合价值: {env.portfolio_value:.0f}")
