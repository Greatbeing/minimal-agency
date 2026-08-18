# -*- coding: utf-8 -*-
"""
BreakShell vs 普通 Agent — 对比演示
=====================================
核心差异：自我模型让 Agent 知道自己"能做什么"和"不能做什么"

场景设计：
- 普通 Agent：只看环境，不看自己 → 盲目选高成本动作 → 能量耗尽失败
- BreakShell：看环境 + 看自己（能量状态）→ 选择匹配能力的动作 → 成功
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from micl.breakshell.agent import BreakShellAgent
from micl.breakshell.environment import GridWorld


# ========================================
# 场景：能力匹配环境
# ========================================

class CapabilityMatchingEnv:
    """
    能力匹配环境
    
    关键设计：
    - 智能体有"能量"，不同动作消耗不同能量
    - 高奖励动作需要高能量，低奖励动作需要低能量
    - 能量随时间恢复但有限
    - 没有自我模型的 Agent 会盲目选高奖励动作 → 能量耗尽
    - 有自我模型的 Agent 会匹配能力与动作 → 持续成功
    """
    
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
        
        self.energy = 10.0  # 初始能量
        self.total_reward = 0
        return self._obs()
    
    def _obs(self):
        return np.concatenate([
            self.agent_pos / self.size,  # 位置
            self.target_pos / self.size,  # 目标方向
            [self.energy / 10.0],  # 能量状态
        ])
    
    def step(self, action):
        self.steps += 1
        
        # 动作设计：[保守=低能耗低奖励, 适中=中能耗中奖励, 激进=高能耗高奖励]
        energy_cost = [1.0, 2.5, 4.0][action]
        move_distance = [1, 2, 3][action]
        
        # 能量检查
        if self.energy < energy_cost:
            # 能量不足 → 失败
            reward = -5.0
            done = False
        else:
            self.energy -= energy_cost
            
            # 移动（向目标靠近）
            direction = self.target_pos - self.agent_pos
            for i in range(2):
                if direction[i] > 0:
                    self.agent_pos[i] = min(self.size - 1, self.agent_pos[i] + move_distance)
                elif direction[i] < 0:
                    self.agent_pos[i] = max(0, self.agent_pos[i] - move_distance)
            
            # 奖励
            dist = np.linalg.norm(self.agent_pos - self.target_pos)
            if dist < 0.5:
                reward = 10.0
                done = True
            else:
                reward = [0.5, 1.5, 3.0][action]
                done = False
            
            # 能量恢复（少量）
            self.energy = min(10.0, self.energy + 0.8)
        
        self.total_reward += reward
        done = done or self.steps >= self.max_steps
        
        return self._obs(), reward, done, {'energy': self.energy}
    
    def obs_dim(self):
        return 5


# ========================================
# 普通 Agent（无自我模型）
# ========================================

class NormalAgent:
    """
    普通 Agent：只看环境，不看自己
    
    策略：总是选奖励最高的动作（激进），不考虑能量状态
    """
    
    def __init__(self, obs_dim, action_dim):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        # 简单的线性策略：只看位置和目标，不看能量
        self.W = np.random.randn(obs_dim - 1, action_dim) * 0.1  # 不看能量维度
        self.b = np.zeros(action_dim)
    
    def select_action(self, obs):
        # 忽略最后一个维度（能量）
        x = obs[:-1]
        logits = x @ self.W + self.b
        # 倾向选高奖励动作（激进）
        logits[2] += 0.5  # 偏向激进
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        return np.random.choice(self.action_dim, p=probs), {}
    
    def update(self, obs, action, next_obs, reward, done):
        pass


# ========================================
# 对比演示
# ========================================

def run_comparison():
    """运行对比演示"""
    print("\n" + "=" * 70)
    print("BreakShell vs 普通 Agent — 对比演示")
    print("=" * 70)
    print("\n场景：智能体有能量限制，需要匹配能力与动作")
    print("- 普通 Agent：只看目标，不看自己 → 盲目激进 → 能量耗尽")
    print("- BreakShell：看目标 + 看自己（能量）→ 合理分配 → 成功\n")
    
    env_normal = CapabilityMatchingEnv(seed=42)
    env_breakshell = CapabilityMatchingEnv(seed=42)
    
    normal = NormalAgent(env_normal.obs_dim(), 3)
    breakshell = BreakShellAgent(env_breakshell.obs_dim(), 3, hidden_dim=32, repr_dim=16, seed=42)
    
    results = {'普通 Agent': [], 'BreakShell': []}
    energy_normal = []
    energy_breakshell = []
    si_history = []
    
    for mode, env, agent in [('普通 Agent', env_normal, normal), ('BreakShell', env_breakshell, breakshell)]:
        print(f"\n{'='*60}")
        print(f"运行: {mode}")
        print(f"{'='*60}")
        
        obs = env.reset()
        episode_reward = 0.0
        
        for step in range(40):
            action, info = agent.select_action(obs)
            next_obs, reward, done, info_env = env.step(action)
            agent.update(obs, action, next_obs, reward, done)
            
            episode_reward += reward
            obs = next_obs
            
            si, _ = agent.get_si() if hasattr(agent, 'get_si') else (0, {})
            
            if step % 5 == 0 or done:
                action_name = ['保守', '适中', '激进'][action]
                print(f"  Step {step:2d}: action={action_name} | reward={reward:+.1f} | 能量={info_env['energy']:.1f} | SI={si:.3f}")
            
            if done:
                if reward > 0:
                    print(f"  ✓ 成功! 步数={step + 1}")
                else:
                    print(f"  ✗ 失败（能量耗尽或超时）")
                break
        
        results[mode].append(episode_reward)
        si_history.append(si)
        print(f"  总奖励: {episode_reward:.1f}")
    
    # 总结
    print(f"\n{'='*70}")
    print("对比总结")
    print(f"{'='*70}")
    
    normal_reward = results['普通 Agent'][0]
    breakshell_reward = results['BreakShell'][0]
    diff = breakshell_reward - normal_reward
    
    print(f"\n  普通 Agent 奖励:  {normal_reward:+.1f}")
    print(f"  BreakShell 奖励:  {breakshell_reward:+.1f}")
    print(f"\n  差异: {diff:+.1f}")
    print(f"  BreakShell 提升: {(diff / (abs(normal_reward) + 1)) * 100:.0f}%")
    
    if diff > 0:
        print(f"\n  ✓ BreakShell 获胜！自我模型帮助匹配能力与动作")
    elif diff < 0:
        print(f"\n  ✗ 普通 Agent 获胜（环境太简单，差异未体现）")
    else:
        print(f"\n  → 平局（需要更复杂环境）")
    
    print(f"\n{'='*70}")
    print("核心差异说明")
    print(f"{'='*70}")
    print("""
  普通 Agent:
    - 只看环境（位置、目标）
    - 不看自己（能量状态）
    - 盲目选高奖励动作（激进）
    - 能量耗尽 → 失败
  
  BreakShell Agent:
    - 看环境 + 看自己（自我模型 z_t）
    - 自我模型编码能量状态
    - 选择匹配当前能力的动作
    - 能量管理 → 持续成功
    """)
    
    return results


if __name__ == "__main__":
    run_comparison()
