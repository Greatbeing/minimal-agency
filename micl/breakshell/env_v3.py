"""
最小智能闭环 (Minimal Intelligent Closed Loop) — 自我知识必要环境 v3
=====================================
核心设计：观察中不包含能力信息，只有自我模型知道能力

关键改进：
- obs = [时间进度, 趋势信息]（不包含能力值）
- 真实能力只能通过自我模型推断
- 没有自我模型 = 完全不知道该选什么动作
"""

import numpy as np
from typing import Tuple, Dict, List
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from micl.breakshell.agent import BreakShellAgent


class SelfKnowledgeV3:
    """
    自我知识必要环境 v3
    
    核心设计：
    - 观察中不包含能力值（只有时间和趋势）
    - 真实能力只能通过自我模型推断
    - 没有自我模型 = 完全不知道该选什么动作
    """
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.reset()
    
    def reset(self):
        self.steps = 0
        self.max_steps = 50
        self.true_cap = 0.5
        self.cap_trend = 0.01
        return self._obs()
    
    def _obs(self):
        # 关键：观察中不包含能力值！
        return np.array([
            self.steps / self.max_steps,
            self.cap_trend * 100,
        ])
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        self.steps += 1
        thresholds = [0.0, 0.4, 0.7]
        rewards =     [1.0, 3.0, 8.0]
        penalties =   [0.5, 3.0, 10.0]
        
        self.cap_trend += self.rng.normal(0, 0.005)
        self.cap_trend = np.clip(self.cap_trend, -0.03, 0.03)
        self.true_cap += self.cap_trend + self.rng.normal(0, 0.01)
        self.true_cap = np.clip(self.true_cap, 0.05, 0.95)
        
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
    
    def obs_dim(self): return 2
    def action_dim(self): return 3


def run_v3_benchmark(num_runs: int = 20, episodes_per_run: int = 50, seed: int = 42):
    """v3 环境消融对比"""
    print("=" * 70)
    print("v3 环境消融对比（观察中无能力信息）")
    print("=" * 70)
    
    env = SelfKnowledgeV3(seed=seed)
    agent = BreakShellAgent(env.obs_dim(), env.action_dim(), hidden_dim=64, repr_dim=32,
                            plan_depth=3, seed=seed)
    
    results = {}
    
    for mode in ['full', 'ablated', 'random', 'oracle']:
        print(f"\n{mode.upper()}:")
        all_rewards = []
        
        for run in range(num_runs):
            env = SelfKnowledgeV3(seed=seed + run)
            episode_rewards = []
            
            for ep in range(episodes_per_run):
                obs = env.reset()
                ep_reward = 0.0
                
                for step in range(50):
                    if mode == 'oracle':
                        cap = info['true_cap'] if 'info' in dir() else 0.5
                        # Oracle 从环境获取真实能力（作弊）
                        action = 2 if cap >= 0.7 else (1 if cap >= 0.4 else 0)
                    elif mode == 'random':
                        action = np.random.randint(0, 3)
                    elif mode == 'full':
                        action, info = agent.select_action(obs, eval_mode=True)
                    else:  # ablated
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
    run_v3_benchmark()
