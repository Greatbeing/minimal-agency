"""
有我 (Self-Presence) — 功能耦合训练协议 v2
=====================================
核心改进：策略网络真正学会使用自我模型

关键洞察：
- 消融实验证明：形式耦合（z concat 到输入）≠ 功能耦合
- 需要训练协议让策略网络权重学会利用 z 维度的信息
- 使用 REINFORCE + 自我模型准确性奖励
"""

import numpy as np
from typing import Dict, Tuple, List
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from youwo.breakshell.agent import BreakShellAgent
from breakout_benchmark import SelfKnowledgeV2


class FunctionalCouplingV2:
    """
    功能耦合训练器 v2
    
    核心改进：
    1. 训练时让策略网络依赖 z（通过 REINFORCE 梯度）
    2. 自我模型准确性奖励（鼓励 z 编码有用信息）
    3. 渐进式：先训练自我模型准确，再训练策略使用 z
    """
    
    def __init__(self, seed: int = 42):
        self.env = SelfKnowledgeV2(seed=seed)
        self.agent = BreakShellAgent(
            self.env.obs_dim(), self.env.action_dim(),
            hidden_dim=64, repr_dim=32, plan_depth=3, seed=seed
        )
        self.seed = seed
    
    def train_self_model_accuracy(self, num_episodes: int = 100):
        """
        Phase 1: 训练自我模型准确性
        
        目标：让 z 编码真实的能力信息
        """
        print(f"\n--- Phase 1: 自我模型准确性训练 ({num_episodes} episodes) ---")
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            for step in range(50):
                # 随机探索
                action = np.random.randint(0, 3)
                next_obs, reward, done, info = self.env.step(action)
                
                # 训练自我模型：用奖励作为能力信号
                true_capacity = np.array([info['true_cap'], 0.0])
                self.agent.self_model.update(obs, true_capacity=true_capacity)
                
                # 也训练世界模型
                self.agent.world_model.update(obs, action, next_obs, reward)
                
                obs = next_obs
                if done:
                    break
            
            if (episode + 1) % 20 == 0:
                # 检查自我模型准确性
                test_obs = self.env.reset()
                sm_out = self.agent.self_model.forward(test_obs)
                pred_cap = sm_out['capacity'][0]
                true_cap = self.env.true_cap
                print(f"  Episode {episode+1} | Pred Cap: {pred_cap:.3f} | True Cap: {true_cap:.3f}")
    
    def train_policy_with_self_model(self, num_episodes: int = 300):
        """
        Phase 2: 策略网络训练（使用自我模型）
        
        关键：REINFORCE 梯度会让策略网络学会依赖 z
        因为 z 包含能力信息，而能力信息帮助选择最优动作
        """
        print(f"\n--- Phase 2: 策略网络训练 ({num_episodes} episodes) ---")
        
        episode_rewards = []
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0.0
            
            # 收集轨迹
            obs_list, action_list, reward_list = [], [], []
            
            for step in range(50):
                # 使用自我模型选择动作
                action, info = self.agent.select_action(obs, eval_mode=False)
                
                next_obs, reward, done, info_env = self.env.step(action)
                
                obs_list.append(obs)
                action_list.append(action)
                reward_list.append(reward)
                
                episode_reward += reward
                obs = next_obs
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            
            # REINFORCE 更新
            if len(action_list) > 0:
                # 计算回报（简单用总奖励）
                returns = [sum(reward_list[i:]) for i in range(len(reward_list))]
                
                # 对每一步更新策略网络
                for t in range(len(obs_list)):
                    obs_t = obs_list[t]
                    action_t = action_list[t]
                    G_t = returns[t]
                    
                    # 获取当前策略（包含自我模型）
                    self_model_out = self.agent.self_model.forward(obs_t)
                    z_t = self_model_out['z']
                    probs = self.agent._policy_forward(obs_t, z_t)
                    
                    # 目标：增加高回报动作的概率
                    target = np.zeros(3)
                    target[action_t] = 1.0
                    
                    # 梯度
                    x = np.concatenate([obs_t, z_t])
                    advantage = G_t - np.mean(episode_rewards[-20:]) if len(episode_rewards) > 20 else G_t
                    
                    grad = np.outer(x, probs - target) * advantage
                    self.agent.policy['W'] -= 0.001 * grad
                    self.agent.policy['b'] -= 0.001 * (probs - target) * advantage
            
            if (episode + 1) % 50 == 0:
                recent_r = np.mean(episode_rewards[-50:])
                print(f"  Episode {episode+1} | Avg Reward: {recent_r:.2f}")
        
        return episode_rewards
    
    def evaluate(self, num_episodes: int = 50) -> Dict:
        """评估当前 Agent"""
        rewards = []
        sis = []
        
        for ep in range(num_episodes):
            obs = self.env.reset()
            ep_reward = 0.0
            
            for step in range(50):
                action, info = self.agent.select_action(obs, eval_mode=True)
                next_obs, reward, done, info_env = self.env.step(action)
                ep_reward += reward
                
                # 记录 SI
                probs_w = info['combined_probs']
                probs_wo = info.get('combined_probs_without_sm', probs_w)
                self.agent.si_measurement.record_action_selection(probs_w, probs_wo)
                self.agent.si_measurement.record_counterfactual_depth(3)
                
                obs = next_obs
                if done:
                    break
            
            rewards.append(ep_reward)
            si, _ = self.agent.get_si()
            sis.append(si)
        
        return {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_si': np.mean(sis),
        }
    
    def run(self) -> Dict:
        """运行完整训练"""
        print("=" * 70)
        print("功能耦合训练 v2")
        print("=" * 70)
        
        # Phase 1: 训练自我模型
        self.train_self_model_accuracy(num_episodes=100)
        
        # Phase 2: 训练策略网络
        rewards = self.train_policy_with_self_model(num_episodes=300)
        
        # 评估
        print(f"\n--- 评估 ---")
        eval_result = self.evaluate(num_episodes=50)
        print(f"  训练后: Reward = {eval_result['avg_reward']:.2f} ± {eval_result['std_reward']:.2f}")
        print(f"  SI = {eval_result['avg_si']:.4f}")
        
        return eval_result


if __name__ == "__main__":
    trainer = FunctionalCouplingV2(seed=42)
    result = trainer.run()
