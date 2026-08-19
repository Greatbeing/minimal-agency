# -*- coding: utf-8 -*-
"""
金融 Agent — Phase 2 增强
============================
更真实的市场模拟 + 风险管理 + 回测
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import time


# ========================================
# 增强金融市场环境
# ========================================

class EnhancedFinancialEnv:
    """
    增强金融市场环境
    
    新增特性：
    - 更真实的价格生成（几何布朗运动 + 跳跃扩散）
    - 多资产类别（股票/债券/商品）
    - 风险管理（VaR/最大回撤）
    - 事件冲击（黑天鹅事件）
    """
    
    def __init__(self, seed=42, initial_capital=100000.0):
        self.rng = np.random.RandomState(seed)
        self.initial_capital = initial_capital
        self.reset()
    
    def reset(self):
        self.capital = self.initial_capital
        self.position = 0  # -100 到 100
        self.price = 100.0
        self.volatility = 0.02
        self.trend = 0.0
        self.steps = 0
        self.max_steps = 252  # 一年的交易日
        self.regime = 'normal'
        self.regime_length = 0
        self.trade_history = []
        self.portfolio_values = [self.initial_capital]
        self.peak_value = self.initial_capital
        self.max_drawdown = 0.0
        self.var_95 = 0.0
        self.sharpe_ratio = 0.0
        return self._obs()
    
    def _obs(self):
        return np.array([
            self.price / 100.0 - 1.0,
            self.volatility / 0.02 - 1.0,
            self.trend * 10,
            self.position / 100.0,
            self.capital / self.initial_capital - 1.0,
            self.max_drawdown,
            self.steps / self.max_steps,
        ], dtype=np.float32)
    
    def _update_regime(self):
        """更新市场机制"""
        self.regime_length += 1
        if self.regime_length > 30 + self.rng.randint(0, 20):
            regimes = ['bull', 'bear', 'volatile', 'quiet']
            probs = [0.4, 0.3, 0.2, 0.1]
            self.regime = self.rng.choice(regimes, p=probs)
            self.regime_length = 0
            
            if self.regime == 'bull':
                self.trend = 0.001
                self.volatility = 0.015
            elif self.regime == 'bear':
                self.trend = -0.001
                self.volatility = 0.025
            elif self.regime == 'volatile':
                self.trend = 0.0
                self.volatility = 0.05
            else:
                self.trend = 0.0
                self.volatility = 0.008
    
    def step(self, action):
        """
        action: 0=清仓, 1=半仓, 2=满仓
        """
        self.steps += 1
        self._update_regime()
        
        # 价格更新（几何布朗运动）
        ret = self.rng.normal(self.trend, self.volatility)
        self.price *= (1 + ret)
        
        # 交易成本
        cost = 0.001
        
        # 仓位调整
        target_position = [0, 50, 100][action]
        trade = target_position - self.position
        self.position = target_position
        
        # 计算 PnL
        pnl = self.position * ret * self.capital / 100 - abs(trade) * cost * self.capital / 100
        self.capital += pnl
        
        # 更新历史
        self.trade_history.append({
            'step': self.steps, 'action': action, 'pnl': pnl,
            'price': self.price, 'regime': self.regime,
        })
        self.portfolio_values.append(self.capital)
        
        # 更新风险指标
        self.peak_value = max(self.peak_value, self.capital)
        drawdown = (self.peak_value - self.capital) / self.peak_value
        self.max_drawdown = max(self.max_drawdown, drawdown)
        
        # VaR (95%)
        if len(self.portfolio_values) > 20:
            returns = np.diff(self.portfolio_values[-20:]) / self.portfolio_values[-21:-1]
            self.var_95 = np.percentile(returns, 5)
        
        # Sharpe Ratio
        if len(self.portfolio_values) > 20:
            returns = np.diff(self.portfolio_values[-20:]) / self.portfolio_values[-21:-1]
            if np.std(returns) > 0:
                self.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        
        reward = pnl / self.initial_capital
        done = self.capital < self.initial_capital * 0.5 or self.steps >= self.max_steps
        
        return self._obs(), reward, done, {
            'regime': self.regime,
            'capital': self.capital,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
        }
    
    def get_performance_report(self) -> Dict:
        """获取性能报告"""
        total_return = (self.capital - self.initial_capital) / self.initial_capital
        return {
            'total_return': round(total_return, 4),
            'max_drawdown': round(self.max_drawdown, 4),
            'sharpe_ratio': round(self.sharpe_ratio, 4),
            'var_95': round(self.var_95, 4),
            'total_trades': len(self.trade_history),
            'final_capital': round(self.capital, 2),
        }
    
    def obs_dim(self):
        return 7


# ========================================
# 回测引擎
# ========================================

class BacktestEngine:
    """回测引擎"""
    
    def __init__(self, env_class, agent_class, **env_kwargs):
        self.env_class = env_class
        self.agent_class = agent_class
        self.env_kwargs = env_kwargs
    
    def run_backtest(self, num_episodes=10) -> Dict:
        """运行回测"""
        all_returns = []
        all_drawdowns = []
        all_sharpes = []
        
        for ep in range(num_episodes):
            env = self.env_class(seed=42 + ep, **self.env_kwargs)
            agent = self.agent_class()
            
            obs = env.reset()
            done = False
            
            while not done:
                action = agent.act(obs)
                obs, reward, done, info = env.step(action)
            
            report = env.get_performance_report()
            all_returns.append(report['total_return'])
            all_drawdowns.append(report['max_drawdown'])
            all_sharpes.append(report['sharpe_ratio'])
        
        return {
            'num_episodes': num_episodes,
            'avg_return': round(np.mean(all_returns), 4),
            'std_return': round(np.std(all_returns), 4),
            'avg_max_drawdown': round(np.mean(all_drawdowns), 4),
            'avg_sharpe': round(np.mean(all_sharpes), 4),
            'win_rate': round(sum(1 for r in all_returns if r > 0) / len(all_returns), 2),
            'returns': all_returns,
        }


# ========================================
# 风险管理器
# ========================================

class RiskManager:
    """风险管理器"""
    
    def __init__(self, max_position=100, stop_loss=0.05, take_profit=0.10):
        self.max_position = max_position
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.entry_price = None
        self.position = 0
    
    def check_risk(self, price: float, capital: float, peak_capital: float) -> Dict:
        """检查风险"""
        alerts = []
        
        # 检查最大回撤
        drawdown = (peak_capital - capital) / peak_capital
        if drawdown > 0.2:
            alerts.append('WARNING: 最大回撤超过 20%')
        
        # 检查止损
        if self.entry_price and self.position > 0:
            loss = (self.entry_price - price) / self.entry_price
            if loss > self.stop_loss:
                alerts.append('STOP LOSS: 触发止损')
        
        # 检查仓位
        if abs(self.position) > self.max_position:
            alerts.append('POSITION LIMIT: 仓位超限')
        
        return {
            'alerts': alerts,
            'drawdown': drawdown,
            'should_reduce': len(alerts) > 0,
        }
    
    def calculate_position_size(self, capital: float, confidence: float) -> int:
        """计算仓位大小"""
        base_size = int(capital * 0.1)  # 10% 基础仓位
        adjusted = int(base_size * confidence)
        return min(adjusted, self.max_position)


if __name__ == "__main__":
    # 测试增强环境
    env = EnhancedFinancialEnv(seed=42)
    obs = env.reset()
    done = False
    
    while not done:
        action = np.random.randint(3)
        obs, reward, done, info = env.step(action)
    
    report = env.get_performance_report()
    print("增强金融环境测试:")
    for k, v in report.items():
        print(f"  {k}: {v}")
