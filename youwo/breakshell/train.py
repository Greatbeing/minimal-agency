"""
BreakShell Agent — 训练循环
====================
训练 BreakShell Agent 并追踪 SI 涌现
"""

import numpy as np
from typing import Dict, List, Tuple
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from .agent import BreakShellAgent
from .environment import GridWorld, NonStationaryGridWorld, ProceduralLabyrinth


def train_breakshell(env_name: str = "gridworld", num_episodes: int = 200,
                   max_steps_per_episode: int = 100, seed: int = 42,
                   verbose: bool = True) -> Dict:
    """
    训练 BreakShell Agent
    
    Args:
        env_name: 环境名称 ("gridworld", "nonstationary", "labyrinth")
        num_episodes: 训练轮数
        max_steps_per_episode: 每轮最大步数
        seed: 随机种子
        verbose: 是否打印详细信息
    
    Returns:
        results: 训练结果
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
    action_dim = 4  # 上/下/左/右
    
    # 创建 Agent
    agent = BreakShellAgent(obs_dim, action_dim, hidden_dim=32, repr_dim=16,
                    plan_depth=5, seed=seed)
    
    # 训练记录
    episode_rewards = []
    episode_lengths = []
    si_history = []
    wm_loss_history = []
    sm_loss_history = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0.0
        episode_step = 0
        
        for step in range(max_steps_per_episode):
            # 选择动作
            action, info = agent.select_action(obs, eval_mode=False)
            
            # 执行
            next_obs, reward, done, env_info = env.step(action)
            
            # 更新
            update_info = agent.update(obs, action, next_obs, reward, done)
            
            episode_reward += reward
            episode_step += 1
            obs = next_obs
            
            if done:
                break
        
        # 记录
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_step)
        
        si, _ = agent.get_si()
        si_history.append(si)
        wm_loss_history.append(update_info['wm_loss'])
        sm_loss_history.append(update_info['sm_loss'])
        
        # 打印
        if verbose and (episode + 1) % 20 == 0:
            recent_reward = np.mean(episode_rewards[-20:])
            recent_si = np.mean(si_history[-20:])
            print(f"Episode {episode+1}/{num_episodes} | "
                  f"Avg Reward: {recent_reward:.2f} | "
                  f"SI: {recent_si:.4f} | "
                  f"Steps: {episode_step}")
    
    results = {
        'env_name': env_name,
        'num_episodes': num_episodes,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'si_history': si_history,
        'wm_loss_history': wm_loss_history,
        'sm_loss_history': sm_loss_history,
        'final_si': si_history[-1] if si_history else 0,
        'avg_reward': np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 20 else np.mean(episode_rewards),
        'agent': agent,
    }
    
    return results


def compare_with_baseline() -> Dict:
    """
    对比 BreakShell Agent 与 baseline (无自我模型的 Agent)
    """
    print("=" * 60)
    print("BreakShell Agent 对比实验")
    print("=" * 60)
    
    results = {}
    
    # 1. BreakShell Agent (完整)
    print("\n--- BreakShell Agent (完整) ---")
    results['l6'] = train_breakshell("gridworld", num_episodes=100, verbose=True)
    
    # 2. 非平稳环境
    print("\n--- BreakShell Agent (非平稳环境) ---")
    results['l6_nonstationary'] = train_breakshell("nonstationary", num_episodes=100, verbose=True)
    
    # 3. 复杂迷宫
    print("\n--- BreakShell Agent (复杂迷宫) ---")
    results['l6_labyrinth'] = train_breakshell("labyrinth", num_episodes=100, verbose=True)
    
    return results


if __name__ == "__main__":
    results = compare_with_baseline()
    
    print("\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)
    
    for name, res in results.items():
        print(f"\n{name}:")
        print(f"  Final SI: {res['final_si']:.4f}")
        print(f"  Avg Reward (last 20): {res['avg_reward']:.2f}")
        print(f"  Avg Episode Length: {np.mean(res['episode_lengths'][-20:]):.1f}")
