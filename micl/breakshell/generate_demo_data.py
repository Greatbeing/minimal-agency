# -*- coding: utf-8 -*-
"""
生成 BreakShell 演示数据（训练曲线 + 对比数据）
用于 HTML 可视化
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from breakshell import BreakShellEnv, BreakShell, NormalAgent, train, evaluate


def generate_demo_data():
    """生成演示数据"""
    print("生成演示数据...")
    
    # 训练
    env_train_normal = BreakShellEnv(seed=42)
    env_train_breakshell = BreakShellEnv(seed=42)
    
    normal = NormalAgent(obs_dim=4, action_dim=3, lr=0.005)
    breakshell = BreakShell(action_dim=3, lr=0.005)
    
    print("训练普通 Agent...")
    normal_rewards = train(normal, env_train_normal, num_episodes=500, verbose=False)
    
    print("训练 BreakShell...")
    breakshell_rewards = train(breakshell, env_train_breakshell, num_episodes=500, verbose=False)
    
    # 评估
    env_eval = BreakShellEnv(seed=999)
    normal_eval = evaluate(normal, env_eval, num_episodes=100)
    breakshell_eval = evaluate(breakshell, env_eval, num_episodes=100)
    
    # 保存数据
    data = {
        'normal_rewards': [float(r) for r in normal_rewards],
        'breakshell_rewards': [float(r) for r in breakshell_rewards],
        'normal_eval': float(normal_eval),
        'breakshell_eval': float(breakshell_eval),
        'normal_avg_100': [float(np.mean(normal_rewards[max(0,i-99):i+1])) for i in range(len(normal_rewards))],
        'breakshell_avg_100': [float(np.mean(breakshell_rewards[max(0,i-99):i+1])) for i in range(len(breakshell_rewards))],
    }
    
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'docs', 'demo_data.json')
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"数据已保存: {data_path}")
    print(f"普通 Agent 最终评估: {normal_eval:.2f}")
    print(f"BreakShell 最终评估: {breakshell_eval:.2f}")
    print(f"差异: {breakshell_eval - normal_eval:+.2f}")
    
    return data


if __name__ == "__main__":
    generate_demo_data()
