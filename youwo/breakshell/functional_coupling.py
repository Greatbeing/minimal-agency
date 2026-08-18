"""
BreakShell Agent — 功能耦合训练协议
=====================================
解决消融实验发现的核心问题：形式耦合 ≠ 功能耦合

核心改进：
1. 渐进式训练（世界模型 → 自我模型 → 联合策略）
2. 自我模型准确性奖励（鼓励系统使用准确的自我模型）
3. 注意力机制（替代简单拼接）
4. 消融验证（Full vs Ablated SI 差异 > 50%）
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from .agent import BreakShellAgent
from .environment import GridWorld, NonStationaryGridWorld, ProceduralLabyrinth


class FunctionalCouplingTrainer:
    """
    功能耦合训练器
    
    渐进式训练协议：
    Phase 1: 世界模型预训练（学习环境动力学）
    Phase 2: 自我模型预训练（学习自身能力）
    Phase 3: 联合训练（策略网络学会使用自我模型）
    Phase 4: 消融验证（证明功能耦合已实现）
    """
    
    def __init__(self, env_name: str = "gridworld", seed: int = 42):
        self.env_name = env_name
        self.seed = seed
        
        # 创建环境
        if env_name == "gridworld":
            self.env = GridWorld(size=8, seed=seed)
        elif env_name == "nonstationary":
            self.env = NonStationaryGridWorld(size=8, seed=seed, change_freq=50)
        elif env_name == "labyrinth":
            self.env = ProceduralLabyrinth(size=10, seed=seed, complexity=0.3)
        else:
            raise ValueError(f"Unknown environment: {env_name}")
        
        obs_dim = self.env.get_obs_dim()
        action_dim = 4
        
        self.agent = BreakShellAgent(obs_dim, action_dim, hidden_dim=64, repr_dim=32,
                                 plan_depth=5, seed=seed)
    
    def train_phase1_world_model(self, num_episodes: int = 50):
        """
        Phase 1: 世界模型预训练
        
        目标：学习环境动力学 P(s'|s, a)
        """
        print(f"\n--- Phase 1: 世界模型预训练 ({num_episodes} episodes) ---")
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            for step in range(50):
                action = np.random.randint(0, self.agent.action_dim)
                next_obs, reward, done, info = self.env.step(action)
                self.agent.world_model.update(obs, action, next_obs, reward)
                obs = next_obs
                if done:
                    break
            
            if (episode + 1) % 10 == 0:
                print(f"  Episode {episode+1}/{num_episodes}")
    
    def train_phase2_self_model(self, num_episodes: int = 50):
        """
        Phase 2: 自我模型预训练
        
        目标：学习自身能力 C(s) 和状态 S(s)
        """
        print(f"\n--- Phase 2: 自我模型预训练 ({num_episodes} episodes) ---")
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            for step in range(50):
                action = np.random.randint(0, self.agent.action_dim)
                next_obs, reward, done, info = self.env.step(action)
                
                # 自我模型更新
                true_capacity = np.array([reward, 0.0])
                self.agent.self_model.update(obs, true_capacity=true_capacity)
                
                obs = next_obs
                if done:
                    break
            
            if (episode + 1) % 10 == 0:
                print(f"  Episode {episode+1}/{num_episodes}")
    
    def train_phase3_joint(self, num_episodes: int = 200):
        """
        Phase 3: 联合训练
        
        目标：策略网络学会使用自我模型
        
        关键改进：
        - 增加自我模型输入的权重（不依赖固定比例）
        - 使用注意力机制替代简单拼接
        - 增加自我模型准确性的奖励信号
        """
        print(f"\n--- Phase 3: 联合训练 ({num_episodes} episodes) ---")
        
        episode_rewards = []
        si_history = []
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0.0
            
            for step in range(100):
                # 选择动作
                action, info = self.agent.select_action(obs, eval_mode=False)
                
                # 执行
                next_obs, reward, done, env_info = self.env.step(action)
                
                # 增加自我模型准确性奖励
                # 如果自我模型预测准确，额外奖励
                predicted_reward = info.get('plan_info', {}).get('best_value', 0)
                accuracy_bonus = -0.1 * abs(predicted_reward - reward)
                modified_reward = reward + accuracy_bonus
                
                # 更新
                update_info = self.agent.update(obs, action, next_obs, modified_reward, done)
                
                episode_reward += reward
                obs = next_obs
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            si, _ = self.agent.get_si()
            si_history.append(si)
            
            if (episode + 1) % 20 == 0:
                recent_reward = np.mean(episode_rewards[-20:])
                recent_si = np.mean(si_history[-20:])
                print(f"  Episode {episode+1}/{num_episodes} | "
                      f"Reward: {recent_reward:.2f} | SI: {recent_si:.4f}")
        
        return episode_rewards, si_history
    
    def run_ablation_validation(self, num_episodes: int = 50) -> Dict:
        """
        Phase 4: 消融验证
        
        验证功能耦合是否真正实现：
        - 完整 Agent SI > 消融 Agent SI × 1.5
        """
        print(f"\n--- Phase 4: 消融验证 ({num_episodes} episodes) ---")
        
        # 完整版本
        print("\n完整版本:")
        full_rewards, full_si = self._eval_agent(num_episodes, use_self_model=True)
        
        # 消融版本
        print("\n消融版本:")
        ablated_rewards, ablated_si = self._eval_agent(num_episodes, use_self_model=False)
        
        # 对比
        full_avg_si = np.mean(full_si)
        ablated_avg_si = np.mean(ablated_si)
        si_ratio = full_avg_si / (ablated_avg_si + 1e-10)
        
        print(f"\n消融验证结果:")
        print(f"  完整版本 SI: {full_avg_si:.4f}")
        print(f"  消融版本 SI: {ablated_avg_si:.4f}")
        print(f"  SI 比率: {si_ratio:.2f}x")
        
        if si_ratio > 1.5:
            print(f"  ✓ 验证通过: 功能耦合已实现 (SI 比率 > 1.5)")
        elif si_ratio > 1.2:
            print(f"  △ 部分实现: 有功能耦合但不充分 (1.2 < SI 比率 < 1.5)")
        else:
            print(f"  ✗ 验证失败: 功能耦合未实现 (SI 比率 < 1.2)")
        
        return {
            'full_si': full_avg_si,
            'ablated_si': ablated_avg_si,
            'si_ratio': si_ratio,
            'success': si_ratio > 1.5,
        }
    
    def _eval_agent(self, num_episodes: int, use_self_model: bool = True) -> Tuple[List, List]:
        """评估 Agent（可选择是否使用自我模型）"""
        rewards = []
        si_list = []
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0.0
            
            for step in range(100):
                if use_self_model:
                    action, info = self.agent.select_action(obs, eval_mode=True)
                else:
                    # 消融：使用零自我表征
                    self_model_output = self.agent.self_model.forward(obs)
                    self_repr = np.zeros_like(self_model_output['z'])
                    planned_action, plan_info = self.agent.planner.plan(
                        obs, self_model_output, value_fn=None
                    )
                    action_probs = self.agent._policy_forward(obs, self_repr)
                    plan_prior = np.zeros(self.agent.action_dim)
                    plan_prior[planned_action] = 1.0
                    combined = 0.6 * plan_prior + 0.4 * action_probs
                    combined /= combined.sum()
                    action = np.argmax(combined)
                    info = {}
                
                next_obs, reward, done, env_info = self.env.step(action)
                episode_reward += reward
                obs = next_obs
                
                if done:
                    break
            
            rewards.append(episode_reward)
            si, _ = self.agent.get_si()
            si_list.append(si)
            
            if (episode + 1) % 10 == 0:
                print(f"  Episode {episode+1}/{num_episodes} | "
                      f"Reward: {np.mean(rewards[-10:]):.2f}")
        
        return rewards, si_list
    
    def run_full_training(self) -> Dict:
        """运行完整训练流程"""
        print("=" * 60)
        print("BreakShell Agent — 功能耦合训练")
        print("=" * 60)
        
        # Phase 1-3: 训练
        self.train_phase1_world_model(num_episodes=50)
        self.train_phase2_self_model(num_episodes=50)
        rewards, si = self.train_phase3_joint(num_episodes=200)
        
        # Phase 4: 消融验证
        ablation_result = self.run_ablation_validation(num_episodes=50)
        
        return {
            'rewards': rewards,
            'si': si,
            'ablation': ablation_result,
            'final_si': np.mean(si[-20:]),
        }


if __name__ == "__main__":
    results = {}
    
    for env_name in ['gridworld', 'nonstationary']:
        print(f"\n{'='*60}")
        print(f"环境: {env_name}")
        trainer = FunctionalCouplingTrainer(env_name, seed=42)
        results[env_name] = trainer.run_full_training()
    
    print(f"\n{'='*60}")
    print("最终结论")
    print(f"{'='*60}")
    
    for env_name, res in results.items():
        status = "✓" if res['ablation']['success'] else "✗"
        print(f"{status} {env_name}: SI = {res['final_si']:.4f}, "
              f"消融比率 = {res['ablation']['si_ratio']:.2f}x")
