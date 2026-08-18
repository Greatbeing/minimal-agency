"""
最小智能闭环 (Minimal Intelligent Closed Loop) — 最终验证：训练后消融比率
=====================================
对比训练后的 BreakShell Agent vs 消融版本
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from micl.breakshell.agent import BreakShellAgent
from breakout_benchmark import SelfKnowledgeV2
from functional_coupling_v2 import FunctionalCouplingV2


def train_and_validate():
    """训练 + 消融验证"""
    print("=" * 70)
    print("最终验证：训练 → 消融对比")
    print("=" * 70)
    
    # Step 1: 训练
    trainer = FunctionalCouplingV2(seed=42)
    trainer.train_self_model_accuracy(num_episodes=100)
    trainer.train_policy_with_self_model(num_episodes=300)
    
    # Step 2: 消融对比
    print(f"\n{'='*70}")
    print("消融对比")
    print(f"{'='*70}")
    
    env = SelfKnowledgeV2(seed=999)
    agent = trainer.agent
    
    for mode in ['full', 'ablated', 'random', 'oracle']:
        print(f"\n{mode.upper()}:")
        rewards = []
        
        for ep in range(100):
            obs = env.reset()
            ep_reward = 0.0
            
            for step in range(50):
                if mode == 'oracle':
                    cap = obs[0]
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
            
            rewards.append(ep_reward)
        
        avg = np.mean(rewards)
        std = np.std(rewards)
        print(f"  Reward: {avg:.2f} ± {std:.2f}")
        
        if mode == 'full':
            full_avg = avg
        elif mode == 'ablated':
            ablated_avg = avg
    
    # 消融比率
    ratio = full_avg / (ablated_avg + 1e-10)
    print(f"\n{'='*70}")
    print(f"消融比率 (Full/Ablated): {ratio:.2f}x")
    
    if ratio > 1.5:
        print("✓✓✓ 功能耦合验证通过！消融比率 > 1.5x")
    elif ratio > 1.2:
        print("△ 部分实现")
    else:
        print("✗ 功能耦合未实现")
    
    return ratio


if __name__ == "__main__":
    ratio = train_and_validate()
