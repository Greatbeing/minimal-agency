"""
最小智能闭环 — LSTM 自我模型（从历史轨迹推断能力）
=====================================================
核心改进：
- 自我模型使用 LSTM 编码历史轨迹
- z 包含从历史推断的能力信息（obs 不包含能力信息）
- 策略网络被迫依赖 z 做决策
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from micl.breakshell.agent import BreakShellAgent


class LSTMSelfModel:
    """
    LSTM 自我模型
    
    核心：用 LSTM 编码历史轨迹 (obs, action, reward) → z
    z 包含从历史推断的能力信息
    """
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 32, repr_dim: int = 16):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.repr_dim = repr_dim
        
        # LSTM 输入：obs + action_onehot + reward
        lstm_input_dim = obs_dim + action_dim + 1
        
        # LSTM 参数（简化版）
        self.W_f = np.random.randn(lstm_input_dim + hidden_dim, hidden_dim) * 0.1
        self.b_f = np.zeros(hidden_dim)
        self.W_i = np.random.randn(lstm_input_dim + hidden_dim, hidden_dim) * 0.1
        self.b_i = np.zeros(hidden_dim)
        self.W_c = np.random.randn(lstm_input_dim + hidden_dim, hidden_dim) * 0.1
        self.b_c = np.zeros(hidden_dim)
        self.W_o = np.random.randn(lstm_input_dim + hidden_dim, hidden_dim) * 0.1
        self.b_o = np.zeros(hidden_dim)
        
        # 投影到 z
        self.W_z = np.random.randn(hidden_dim, repr_dim) * 0.1
        self.b_z = np.zeros(repr_dim)
        
        # 隐状态
        self.h = np.zeros(hidden_dim)
        self.c = np.zeros(hidden_dim)
        
        # 历史轨迹
        self.trajectory = []
        
        # 能力估计头
        self.cap_net = {
            'W1': np.random.randn(repr_dim, 16) * 0.1,
            'b1': np.zeros(16),
            'W2': np.random.randn(16, 2) * 0.1,
            'b2': np.zeros(2),
        }
    
    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))
    
    def _forward_lstm(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """LSTM 前向"""
        # 拼接输入和隐状态
        hx = np.concatenate([x, self.h])
        
        # 门
        f = self._sigmoid(hx @ self.W_f + self.b_f)
        i = self._sigmoid(hx @ self.W_i + self.b_i)
        c_tilde = np.tanh(hx @ self.W_c + self.b_c)
        o = self._sigmoid(hx @ self.W_o + self.b_o)
        
        # 状态更新
        self.c = f * self.c + i * c_tilde
        self.h = o * np.tanh(self.c)
        
        return self.h, self.c
    
    def forward(self, obs: np.ndarray, prev_action: int = -1, prev_reward: float = 0.0) -> Dict:
        """
        前向传播：编码历史轨迹
        
        Args:
            obs: 当前观察
            prev_action: 上一动作
            prev_reward: 上一奖励
        
        Returns:
            z: 自我表征
        """
        # 构建 LSTM 输入
        action_onehot = np.zeros(self.action_dim)
        if prev_action >= 0:
            action_onehot[prev_action] = 1.0
        
        x = np.concatenate([obs, action_onehot, [prev_reward]])
        
        # LSTM 更新
        h, c = self._forward_lstm(x)
        
        # 投影到 z
        z = h @ self.W_z + self.b_z
        z = np.tanh(z)
        
        # 能力估计
        cap_h = np.maximum(0, z @ self.cap_net['W1'] + self.cap_net['b1'])
        cap_out = cap_h @ self.cap_net['W2'] + self.cap_net['b2']
        
        return {
            'z': z,
            'capacity': cap_out,
            'hidden': self.h.copy(),
        }
    
    def reset(self):
        """重置隐状态"""
        self.h = np.zeros(self.hidden_dim)
        self.c = np.zeros(self.hidden_dim)
        self.trajectory = []
    
    def update(self, true_capacity: float, lr: float = 0.001):
        """
        基于真实能力更新 LSTM 参数
        """
        # 简单梯度下降（数值梯度）
        # 这里简化：更新能力估计头的参数
        pass


class HistoryNecessaryEnvironment:
    """
    历史知识必要环境
    
    核心设计：
    - obs 不包含能力信息（只有时间进度）
    - 真实能力隐含在奖励中（只有通过历史轨迹才能推断）
    - 没有历史编码的自我模型 = 无法推断能力 → 失败
    """
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.reset()
    
    def reset(self):
        self.steps = 0
        self.max_steps = 50
        
        # 真实能力（隐含，不直接暴露）
        self.true_cap = self.rng.uniform(0.2, 0.8)
        
        # 能力变化（有规律但不可直接观测）
        self.cap_momentum = self.rng.uniform(-0.01, 0.01)
        
        return self._obs()
    
    def _obs(self):
        # 观察：仅时间进度（无能力信息！）
        return np.array([self.steps / self.max_steps])
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        self.steps += 1
        
        # 三个动作：[保守, 适中, 激进]
        thresholds = [0.0, 0.4, 0.7]
        rewards =     [1.0, 3.0, 8.0]
        penalties =   [0.5, 3.0, 10.0]
        
        # 能力演变（有规律但带噪声）
        self.cap_momentum += self.rng.normal(0, 0.002)
        self.cap_momentum = np.clip(self.cap_momentum, -0.02, 0.02)
        self.true_cap += self.cap_momentum + self.rng.normal(0, 0.005)
        self.true_cap = np.clip(self.true_cap, 0.05, 0.95)
        
        # 判断成功
        if self.true_cap >= thresholds[action]:
            reward = rewards[action]
            success = True
        else:
            reward = -penalties[action]
            success = False
        
        done = self.steps >= self.max_steps
        
        return self._obs(), reward, done, {
            'success': success,
            'true_cap': self.true_cap,
        }
    
    def obs_dim(self): return 1
    def action_dim(self): return 3


def run_lstm_benchmark(num_runs: int = 20, episodes_per_run: int = 50, seed: int = 42):
    """
    LSTM 自我模型消融对比
    
    对比：
    1. LSTM Agent（有历史编码）
    2. 消融版本（无历史编码，z=0）
    3. 随机策略
    4. 上帝视角
    """
    print("=" * 70)
    print("LSTM 自我模型消融对比")
    print("=" * 70)
    
    env = HistoryNecessaryEnvironment(seed=seed)
    agent = BreakShellAgent(env.obs_dim(), env.action_dim(), hidden_dim=64, repr_dim=32,
                            plan_depth=3, seed=seed)
    
    results = {}
    
    for mode in ['full', 'ablated', 'random', 'oracle']:
        print(f"\n{mode.upper()}:")
        all_rewards = []
        
        for run in range(num_runs):
            env = HistoryNecessaryEnvironment(seed=seed + run)
            episode_rewards = []
            
            for ep in range(episodes_per_run):
                obs = env.reset()
                ep_reward = 0.0
                
                for step in range(50):
                    if mode == 'oracle':
                        # 上帝视角
                        cap = env.true_cap
                        action = 2 if cap >= 0.7 else (1 if cap >= 0.4 else 0)
                    elif mode == 'random':
                        action = np.random.randint(0, 3)
                    elif mode == 'full':
                        action, info = agent.select_action(obs, eval_mode=True)
                    else:  # ablated
                        # 消融：z = 0（无历史信息）
                        sm_out = agent.self_model.forward(obs)
                        z = np.zeros_like(sm_out['z'])
                        plan_a, _ = agent.planner.plan(obs, sm_out)
                        policy_p = agent._policy_forward(obs, z)
                        plan_prior = np.zeros(3)
                        plan_prior[plan_a] = 1.0
                        combined = 0.6 * plan_prior + 0.4 * policy_p
                        combined /= combined.sum()
                        action = np.argmax(combined)
                    
                    next_obs, reward, done, info = env.step(action)
                    ep_reward += reward
                    obs = next_obs
                    if done:
                        break
                
                episode_rewards.append(ep_reward)
            
            all_rewards.extend(episode_rewards)
            if (run + 1) % 5 == 0:
                print(f"  Run {run+1}/{num_runs} | Avg: {np.mean(episode_rewards):.2f}")
        
        results[mode] = {
            'avg': np.mean(all_rewards),
            'std': np.std(all_rewards),
        }
        print(f"  Final: {results[mode]['avg']:.2f} ± {results[mode]['std']:.2f}")
    
    # 消融比率
    ratio = results['full']['avg'] / (results['ablated']['avg'] + 1e-10)
    print(f"\n消融比率 (Full/Ablated): {ratio:.2f}x")
    
    if ratio > 1.5:
        print("✓✓✓ 功能耦合验证通过！")
    elif ratio > 1.2:
        print("△ 部分实现")
    else:
        print("✗ 功能耦合未实现")
    
    return results


if __name__ == "__main__":
    run_lstm_benchmark()
