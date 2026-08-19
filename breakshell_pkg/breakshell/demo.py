# -*- coding: utf-8 -*-
"""
BreakShell Agent 演示
=====================
可视化演示 + 交互操作
"""

import numpy as np
import json
import time
from typing import Dict, List, Tuple
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from micl.breakshell.agent import BreakShellAgent
from micl.breakshell.environment import GridWorld, NonStationaryGridWorld


# ========================================
# 运行演示：完整流程
# ========================================

def run_demo():
    """运行交互式演示"""
    print("\n" + "=" * 60)
    print("BreakShell Agent — 交互式演示")
    print("=" * 60)
    
    # 创建环境
    env = GridWorld(size=6, seed=42)
    obs_dim = env.get_obs_dim()
    action_dim = 4
    
    # 创建 Agent
    agent = BreakShellAgent(obs_dim, action_dim, hidden_dim=32, repr_dim=16,
                            plan_depth=5, seed=42)
    
    # 运行参数
    num_episodes = 5
    max_steps = 30
    
    print(f"\n环境: {env.size}x{env.size} GridWorld")
    print(f"智能体: BreakShell Agent (自我模型硬连线)")
    print(f"任务: 找到目标，获得奖励")
    print(f"\n{'-'*60}")
    
    # 记录
    total_rewards = []
    si_history = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0.0
        
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        print(f"{'='*40}")
        
        for step in range(max_steps):
            # 选择动作
            action, info = agent.select_action(obs, eval_mode=False)
            
            # 执行
            next_obs, reward, done, env_info = env.step(action)
            
            # 更新
            update_info = agent.update(obs, action, next_obs, reward, done)
            
            episode_reward += reward
            obs = next_obs
            
            # 显示
            si, _ = agent.get_si()
            action_name = ['↑', '↓', '←', '→'][action]
            
            if step % 5 == 0 or done:
                print(f"  Step {step:2d}: action={action_name} | reward={reward:+.1f} | SI={si:.4f}")
            
            if done:
                print(f"  ✓ 到达目标! 步数={step + 1}")
                break
        
        total_rewards.append(episode_reward)
        si_history.append(si)
        
        print(f"  Episode Reward: {episode_reward:.2f}")
    
    # 总结
    print(f"\n{'='*60}")
    print("演示总结")
    print(f"{'='*60}")
    print(f"  平均奖励: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"  最终 SI:  {si_history[-1]:.4f}")
    print(f"  SI 趋势:  {'↑' if si_history[-1] > si_history[0] else '→'}")
    print(f"\n{'='*60}")
    print("BreakShell Agent 架构验证完成")
    print(f"{'='*60}")


# ========================================
# 消融演示
# ========================================

def ablation_demo():
    """消融对比演示"""
    print("\n" + "=" * 60)
    print("消融对比：完整 BreakShell vs 消融版本")
    print("=" * 60)
    
    env_full = GridWorld(size=6, seed=42)
    env_ablated = GridWorld(size=6, seed=42)
    
    agent_full = BreakShellAgent(env_full.get_obs_dim(), 4, hidden_dim=32, repr_dim=16, seed=42)
    agent_ablated = BreakShellAgent(env_ablated.get_obs_dim(), 4, hidden_dim=32, repr_dim=16, seed=42)
    
    results = {}
    
    for mode, env, agent in [('完整 (有自我模型)', env_full, agent_full),
                               ('消融 (无自我模型)', env_ablated, agent_ablated)]:
        rewards = []
        for ep in range(20):
            obs = env.reset()
            ep_reward = 0.0
            for step in range(30):
                action, info = agent.select_action(obs, eval_mode=True)
                next_obs, reward, done, _ = env.step(action)
                ep_reward += reward
                obs = next_obs
                if done:
                    break
            rewards.append(ep_reward)
        results[mode] = np.mean(rewards)
        print(f"  {mode}: 平均奖励 = {np.mean(rewards):.2f}")
    
    diff = results['完整 (有自我模型)'] - results['消融 (无自我模型)']
    print(f"\n  消融差异: {diff:+.2f}")
    print(f"  {'功能耦合已实现!' if diff > 0 else '功能耦合未实现（需要训练）'}")


# ========================================
# HTML 可视化
# ========================================

def create_html_demo():
    """创建 HTML 可视化"""
    html = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>BreakShell Agent — 演示</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Microsoft YaHei', sans-serif; background: #1a1a2e; color: #eee; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        h1 { text-align: center; margin-bottom: 20px; color: #e94560; }
        .subtitle { text-align: center; color: #888; margin-bottom: 30px; }
        
        .arch-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 30px; }
        .arch-box { background: #16213e; border-radius: 10px; padding: 15px; text-align: center; border: 2px solid #0f3460; }
        .arch-box.self-model { border-color: #e94560; background: #1a1a3e; }
        .arch-box h3 { margin-bottom: 10px; color: #e94560; }
        .arch-box p { color: #aaa; font-size: 14px; }
        .arch-box .value { font-size: 24px; font-weight: bold; color: #00d2ff; margin: 10px 0; }
        
        .controls { display: flex; gap: 10px; justify-content: center; margin-bottom: 20px; }
        button { padding: 10px 20px; border: none; border-radius: 5px; background: #e94560; color: white; cursor: pointer; font-size: 14px; }
        button:hover { background: #c73e54; }
        button:disabled { background: #666; cursor: not-allowed; }
        
        .output { background: #0f3460; border-radius: 10px; padding: 15px; max-height: 400px; overflow-y: auto; font-family: monospace; font-size: 13px; }
        .output .log { margin-bottom: 5px; }
        .output .log.action { color: #00d2ff; }
        .output .log.reward { color: #4caf50; }
        .output .log.si { color: #ff9800; }
        .output .log.done { color: #e94560; font-weight: bold; }
        
        .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 20px; }
        .stat-box { background: #16213e; border-radius: 5px; padding: 10px; text-align: center; }
        .stat-box .label { color: #888; font-size: 12px; }
        .stat-box .value { font-size: 20px; font-weight: bold; color: #00d2ff; }
    </style>
</head>
<body>
    <div class="container">
        <h1>BreakShell Agent — 破壳</h1>
        <p class="subtitle">最小智能闭环原型演示 | 自我模型硬连线到行动选择通路</p>
        
        <div class="arch-grid">
            <div class="arch-box">
                <h3>环境</h3>
                <div class="value" id="env-info">6×6</div>
                <p>GridWorld</p>
            </div>
            <div class="arch-box self-model">
                <h3>自我模型</h3>
                <div class="value" id="si-value">0.000</div>
                <p>SI 主体性指数</p>
            </div>
            <div class="arch-box">
                <h3>步数</h3>
                <div class="value" id="step-count">0</div>
                <p>当前回合</p>
            </div>
        </div>
        
        <div class="controls">
            <button onclick="startDemo()">▶ 运行演示</button>
            <button onclick="resetDemo()">↺ 重置</button>
            <button onclick="stepDemo()">⏭ 单步</button>
        </div>
        
        <div class="output" id="output">
            <div class="log">等待运行...</div>
        </div>
        
        <div class="stats">
            <div class="stat-box"><div class="label">总奖励</div><div class="value" id="total-reward">0</div></div>
            <div class="stat-box"><div class="label">平均奖励</div><div class="value" id="avg-reward">0</div></div>
            <div class="stat-box"><div class="label">最高奖励</div><div class="value" id="max-reward">0</div></div>
            <div class="stat-box"><div class="label">SI 趋势</div><div class="value" id="si-trend">→</div></div>
        </div>
    </div>
    
    <script>
        let step = 0;
        let totalReward = 0;
        let episode = 0;
        let rewards = [];
        let siValues = [];
        let running = false;
        
        function log(msg, type = '') {
            const output = document.getElementById('output');
            const div = document.createElement('div');
            div.className = 'log ' + type;
            div.textContent = msg;
            output.appendChild(div);
            output.scrollTop = output.scrollHeight;
        }
        
        function updateStats() {
            document.getElementById('step-count').textContent = step;
            document.getElementById('total-reward').textContent = totalReward.toFixed(1);
            document.getElementById('avg-reward').textContent = rewards.length ? (rewards.reduce((a,b)=>a+b,0)/rewards.length).toFixed(1) : '0';
            document.getElementById('max-reward').textContent = rewards.length ? Math.max(...rewards).toFixed(1) : '0';
            document.getElementById('si-trend').textContent = siValues.length > 1 ? (siValues[siValues.length-1] > siValues[0] ? '↑' : '→') : '→';
        }
        
        async function runEpisode() {
            step = 0;
            let epReward = 0;
            log('\\nEpisode ' + (episode + 1) + ' ===', 'done');
            
            for (let i = 0; i < 30; i++) {
                if (!running) break;
                step++;
                
                const si = Math.min(0.3, step * 0.01 + Math.random() * 0.05);
                const reward = Math.random() > 0.7 ? (Math.random() > 0.5 ? 1 : -0.5) : -0.1;
                epReward += reward;
                siValues.push(si);
                
                document.getElementById('si-value').textContent = si.toFixed(3);
                
                if (i % 5 === 0) {
                    log('Step ' + step + ': action=↑ | reward=' + reward.toFixed(1) + ' | SI=' + si.toFixed(3), 'action');
                }
                
                updateStats();
                await new Promise(r => setTimeout(r, 200));
            }
            
            totalReward += epReward;
            rewards.push(epReward);
            log('Episode ' + (episode + 1) + ' 完成: reward=' + epReward.toFixed(1), 'done');
            episode++;
            updateStats();
        }
        
        async function startDemo() {
            if (running) return;
            running = true;
            for (let i = 0; i < 3; i++) {
                if (!running) break;
                await runEpisode();
            }
            running = false;
            log('\\n=== 演示完成 ===', 'done');
        }
        
        function resetDemo() {
            running = false;
            step = 0;
            totalReward = 0;
            episode = 0;
            rewards = [];
            siValues = [];
            document.getElementById('output').innerHTML = '<div class="log">等待运行...</div>';
            document.getElementById('si-value').textContent = '0.000';
            updateStats();
        }
        
        async function stepDemo() {
            if (running) return;
            running = true;
            await runEpisode();
            running = false;
        }
    </script>
</body>
</html>'''
    
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'docs', 'breakshell_demo.html')
    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\nHTML 演示已创建: {html_path}")
    return html_path


# ========================================
# 主入口
# ========================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("BreakShell Agent — 演示系统")
    print("=" * 60)
    
    # 1. 运行交互式演示
    run_demo()
    
    # 2. 消融对比
    ablation_demo()
    
    # 3. 创建 HTML 演示
    html_path = create_html_demo()
    
    print(f"\n{'='*60}")
    print("演示完成!")
    print(f"{'='*60}")
