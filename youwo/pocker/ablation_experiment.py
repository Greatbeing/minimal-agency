"""
Pocker Agent — 消融实验（Ablation Study）
=======================================
核心验证：移除自我模型后 SI 是否降至 0？

实验设计：
1. Pocker Agent (完整) — 自我模型参与行动选择
2. Pocker Agent-Ablated — 自我模型输出被替换为全零向量
3. 预期：Ablated 版本的 SI ≈ 0，验证 SEC-4
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import PockerAgent
from environment import GridWorld, NonStationaryGridWorld, ProceduralLabyrinth


class PockerAgentAblated(PockerAgent):
    """
    消融版本：自我模型完全不参与行动选择
    
    与完整版本的区别：
    1. 策略网络的 self_repr 替换为零向量
    2. 规划器中的 capability_mean 替换为固定值 0.5（不使用自我模型的能力估计）
    
    这样确保自我模型从行动选择通路中被完全移除。
    """
    
    def select_action(self, obs: np.ndarray, eval_mode: bool = False) -> tuple:
        """选择动作 — 消融版本：自我模型完全不参与"""
        # 获取自我表征但丢弃
        self_model_output = self.self_model.forward(obs)
        self_repr = np.zeros_like(self_model_output['z'])  # 消融1: 策略网络输入为零
        
        # 规划器使用固定能力估计（不使用自我模型的能力）
        # 关键：替换 capacity 为固定值
        fake_self_model_output = {
            'z': self_model_output['z'],
            'capacity': np.array([0.5, 0.5]),  # 消融2: 固定能力估计
            'state': self_model_output['state'],
            'goal': self_model_output['goal'],
            'hidden': self_model_output['hidden'],
        }
        
        planned_action, plan_info = self.planner.plan(
            obs, fake_self_model_output, value_fn=None
        )
        
        # 策略网络 (输入包含零自我表征)
        action_probs = self._policy_forward(obs, self_repr)
        
        # 结合规划和策略网络
        plan_prior = np.zeros(self.action_dim)
        plan_prior[planned_action] = 1.0
        combined_probs = 0.6 * plan_prior + 0.4 * action_probs
        combined_probs /= combined_probs.sum()
        
        if eval_mode:
            action = np.argmax(combined_probs)
        else:
            action = self.rng.choice(self.action_dim, p=combined_probs)
        
        info = {
            'self_model_output': self_model_output,
            'plan_info': plan_info,
            'action_probs': action_probs,
            'combined_probs': combined_probs,
            'planned_action': planned_action,
            'self_repr': self_repr,
        }
        
        return action, info


def run_ablation_experiment(env_name: str = "gridworld", 
                            num_episodes: int = 100,
                            max_steps: int = 100,
                            seed: int = 42) -> dict:
    """
    运行消融实验
    
    Returns:
        results: 包含完整版本和消融版本的结果
    """
    # 创建环境
    if env_name == "gridworld":
        env = GridWorld(size=8, seed=seed)
    elif env_name == "nonstationary":
        env = NonStationaryGridWorld(size=8, seed=seed, change_freq=50)
    elif env_name == "labyrinth":
        env = ProceduralLabyrinth(size=10, seed=seed, complexity=0.3)
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    
    obs_dim = env.get_obs_dim()
    action_dim = 4
    
    # 创建两个版本的 Agent
    agent_full = PockerAgent(obs_dim, action_dim, hidden_dim=32, repr_dim=16,
                         plan_depth=5, seed=seed)
    agent_ablated = PockerAgentAblated(obs_dim, action_dim, hidden_dim=32, repr_dim=16,
                                   plan_depth=5, seed=seed)
    
    results = {
        'env_name': env_name,
        'full': {'rewards': [], 'si_history': [], 'final_si': 0, 'avg_reward': 0},
        'ablated': {'rewards': [], 'si_history': [], 'final_si': 0, 'avg_reward': 0},
    }
    
    for agent_name, agent in [('full', agent_full), ('ablated', agent_ablated)]:
        print(f"\n--- {'完整 Pocker Agent' if agent_name == 'full' else '消融版本 (无自我模型)'} ---")
        
        for episode in range(num_episodes):
            obs = env.reset()
            episode_reward = 0.0
            
            for step in range(max_steps):
                action, info = agent.select_action(obs, eval_mode=False)
                next_obs, reward, done, env_info = env.step(action)
                update_info = agent.update(obs, action, next_obs, reward, done)
                
                episode_reward += reward
                obs = next_obs
                
                if done:
                    break
            
            results[agent_name]['rewards'].append(episode_reward)
            si, _ = agent.get_si()
            results[agent_name]['si_history'].append(si)
            
            if (episode + 1) % 20 == 0:
                recent_reward = np.mean(results[agent_name]['rewards'][-20:])
                recent_si = np.mean(results[agent_name]['si_history'][-20:])
                print(f"  Episode {episode+1}/{num_episodes} | "
                      f"Avg Reward: {recent_reward:.2f} | SI: {recent_si:.4f}")
        
        results[agent_name]['final_si'] = results[agent_name]['si_history'][-1]
        results[agent_name]['avg_reward'] = np.mean(results[agent_name]['rewards'][-20:])
    
    return results


def main():
    print("=" * 70)
    print("消融实验：验证 SEC-4（自我模型参与行动选择）")
    print("=" * 70)
    print()
    
    # 假设：
    # - 完整 Pocker Agent: SI > 0 (自我模型参与)
    # - 消融版本: SI ≈ 0 (自我模型不参与)
    
    all_results = {}
    
    for env_name in ['gridworld', 'nonstationary', 'labyrinth']:
        print(f"\n{'='*70}")
        print(f"环境: {env_name}")
        print(f"{'='*70}")
        
        results = run_ablation_experiment(env_name, num_episodes=100, seed=42)
        all_results[env_name] = results
        
        print(f"\n--- {env_name} 结果汇总 ---")
        print(f"  完整 Pocker Agent:")
        print(f"    Final SI: {results['full']['final_si']:.4f}")
        print(f"    Avg Reward (last 20): {results['full']['avg_reward']:.2f}")
        print(f"  消融版本:")
        print(f"    Final SI: {results['ablated']['final_si']:.4f}")
        print(f"    Avg Reward (last 20): {results['ablated']['avg_reward']:.2f}")
        print(f"  SI 差异 (Full - Ablated): "
              f"{results['full']['final_si'] - results['ablated']['final_si']:.4f}")
    
    print(f"\n{'='*70}")
    print("结论")
    print(f"{'='*70}")
    print()
    print("消融实验验证 SEC-4 假设:")
    
    for env_name, results in all_results.items():
        diff = results['full']['final_si'] - results['ablated']['final_si']
        status = "✓ 验证通过" if diff > 0.01 else "✗ 验证失败"
        print(f"  {env_name}: SI 差异 = {diff:.4f} {status}")
    
    return all_results


if __name__ == "__main__":
    results = main()
