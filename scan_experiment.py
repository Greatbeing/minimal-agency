"""
相变边界扫描实验
=================
扫描规划深度 1-5，绘制 SI 曲线，确定相变是突然还是渐进
"""

import numpy as np
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import json
from datetime import datetime

# 复用 simulation.py 中的基础定义
from simulation import (
    Action, ACTIONS, ACTION_DX, GridWorld, AgentLevel,
    AgentState, MinimalAgent, MetricsCollector,
    compute_subjectivity_index, _test_agent,
    compute_self_other_discrimination,
    compute_historical_dependence,
    compute_adaptation_gain
)


class ScanningAgent(MinimalAgent):
    """
    可配置规划深度的扫描主体
    """
    
    def __init__(self, planning_depth: int, view_size: int = 25, seed: Optional[int] = None):
        # 不调用 super().__init__，手动配置
        self.level = AgentLevel.L6  # 始终作为 L6 处理
        self.state = AgentState(level=AgentLevel.L6)
        self.view_size = view_size
        self.rng = np.random.RandomState(seed)
        
        # 全部能力
        self.state.weights = self.rng.randn(view_size, view_size) * 0.01
        self.state.bias = np.zeros(view_size)
        self.state.self_model_weight = 0.5
        self.state.own_causal_model = {}
        self.state.learning_rate = 0.05
        self.state.planning_depth = planning_depth  # 可配置


def run_scan_experiment(planning_depth: int, n_episodes: int = 150, 
                       steps_per_episode: int = 100, seed: int = 42, 
                       verbose: bool = False) -> Dict:
    """运行单个规划深度的实验"""
    env = GridWorld(N=10, seed=seed)
    agent = ScanningAgent(planning_depth=planning_depth, seed=seed)
    
    episode_rewards = []
    episode_errors = []
    
    for ep in range(n_episodes):
        env._init_map()
        ep_reward = 0
        ep_errors = []
        
        for step in range(steps_per_episode):
            obs = env.observe()
            action = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            agent.update(obs, action, next_obs, reward)
            
            ep_reward += reward
            
            if agent.state.prediction_errors:
                ep_errors.append(agent.state.prediction_errors[-1])
            
            if done:
                break
        
        episode_rewards.append(ep_reward)
        if ep_errors:
            episode_errors.append(np.mean(ep_errors))
    
    # Compute metrics
    si = compute_subjectivity_index(agent, env, n_trials=50)
    sod = compute_self_other_discrimination(agent)
    hd = compute_historical_dependence(agent)
    ag = compute_adaptation_gain(agent)
    
    return {
        'planning_depth': planning_depth,
        'total_reward': sum(episode_rewards),
        'mean_reward': np.mean(episode_rewards[-50:]),
        'subjectivity_index': si,
        'self_other_discrimination': sod,
        'historical_dependence': hd,
        'adaptation_gain': ag,
        'episode_rewards': episode_rewards,
        'final_prediction_error': agent.state.prediction_errors[-1] if agent.state.prediction_errors else 1.0,
    }


def run_full_scan(depths: List[int] = None, n_episodes: int = 150,
                  steps_per_episode: int = 100, seed: int = 42,
                  verbose: bool = True) -> Dict:
    """运行全范围扫描"""
    if depths is None:
        depths = [1, 2, 3, 4, 5]
    
    results = {}
    
    for depth in depths:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Scanning planning depth = {depth}")
            print(f"{'='*60}")
        
        result = run_scan_experiment(depth, n_episodes, steps_per_episode, seed, verbose)
        results[f'D{depth}'] = result
        
        if verbose:
            print(f"\nDepth {depth} Results:")
            print(f"  Mean Reward: {result['mean_reward']:.2f}")
            print(f"  Subjectivity Index: {result['subjectivity_index']:.4f}")
            print(f"  Self-Other Discrimination: {result['self_other_discrimination']:.4f}")
            print(f"  Historical Dependence: {result['historical_dependence']:.4f}")
            print(f"  Adaptation Gain: {result['adaptation_gain']:.4f}")
    
    return results


def generate_scan_report(results: Dict, output_dir: str = "D:/HermesOutput/minimal_agency"):
    """生成扫描报告"""
    report = []
    report.append("# 相变边界扫描报告")
    report.append(f"\n**日期**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append(f"\n**实验配置**: 150 episodes × 100 steps, 10×10 GridWorld")
    report.append("\n---\n")
    
    # Summary table
    report.append("## 总体结果\n")
    report.append("| 深度 | 描述 | 平均奖励 | 主体性指数 | 自我-他者区分 | 历史依赖 | 适应性增益 |")
    report.append("|------|------|----------|-----------|--------------|---------|-----------|")
    
    for depth_key, r in sorted(results.items(), key=lambda x: x[0]):
        depth = r['planning_depth']
        desc = f"规划深度={depth}"
        report.append(f"| {depth_key} | {desc} | "
                     f"{r['mean_reward']:.2f} | {r['subjectivity_index']:.4f} | "
                     f"{r['self_other_discrimination']:.4f} | {r['historical_dependence']:.4f} | "
                     f"{r['adaptation_gain']:.4f} |")
    
    report.append("\n---\n")
    
    # Phase transition analysis
    report.append("## 相变分析\n")
    
    depths = sorted([r['planning_depth'] for r in results.values()])
    si_values = {r['planning_depth']: r['subjectivity_index'] for r in results.values()}
    reward_values = {r['planning_depth']: r['mean_reward'] for r in results.values()}
    
    report.append("### 主体性指数 (SI) 随规划深度变化\n")
    for depth in depths:
        si = si_values[depth]
        bar = '█' * int(si * 30)
        report.append(f"- 深度 {depth}: {si:.4f} {bar}")
    
    report.append("\n### 相变检测\n")
    
    # Detect phase transition
    prev_si = 0
    transitions = []
    for depth in depths:
        si = si_values[depth]
        jump = si - prev_si
        if jump > 0.03:  # threshold
            transitions.append((depth, jump))
        prev_si = si
    
    if transitions:
        for depth, jump in transitions:
            report.append(f"- **相变点**: 深度 {depth} (SI 跃迁 +{jump:.4f})")
    else:
        report.append("- 未检测到显著相变（SI 变化较连续）")
    
    # Check if it's a threshold or gradual
    si_list = [si_values[d] for d in depths]
    if len(si_list) >= 3:
        # Check if there's a clear threshold
        threshold_idx = None
        for i in range(1, len(si_list)):
            if si_list[i] > 0.05 and si_list[i-1] <= 0.05:
                threshold_idx = i
                break
        
        if threshold_idx is not None:
            threshold_depth = depths[threshold_idx]
            report.append(f"- **阈值型相变**: 在规划深度 {threshold_depth} 处 SI 从接近 0 跃升至 {si_list[threshold_idx]:.4f}")
            report.append(f"- 结论: 主体性涌现存在临界规划深度，不是渐进过程")
        else:
            report.append(f"- **渐变型**: SI 随规划深度逐渐增加，无明确阈值")
    
    report.append("\n---\n")
    
    # Reward curve
    report.append("### 奖励增长曲线\n")
    for depth in depths:
        reward = reward_values[depth]
        bar = '█' * int(reward / 2)
        report.append(f"- 深度 {depth}: {reward:.2f} {bar}")
    
    report.append("\n---\n")
    
    # Analysis
    report.append("## 分析\n")
    
    max_si_depth = max(si_values, key=si_values.get)
    max_reward_depth = max(reward_values, key=reward_values.get)
    
    report.append(f"1. **最高主体性**: 规划深度 {max_si_depth} (SI={si_values[max_si_depth]:.4f})")
    report.append(f"2. **最高性能**: 规划深度 {max_reward_depth} (reward={reward_values[max_reward_depth]:.2f})")
    
    # Compare with L5 (depth=1) and L6 (depth=5)
    d1 = results.get('D1', {})
    d5 = results.get('D5', {})
    
    if d1 and d5:
        si_jump = d5['subjectivity_index'] - d1['subjectivity_index']
        report.append(f"3. **L5→L6 相变幅度**: ΔSI = {si_jump:+.4f}")
        
        if si_jump > 0.1:
            report.append(f"4. **相变类型**: 强相变（ΔSI > 0.1）")
        elif si_jump > 0.05:
            report.append(f"4. **相变类型**: 中等相变（0.05 < ΔSI < 0.1）")
        else:
            report.append(f"4. **相变类型**: 弱相变（ΔSI < 0.05）")
    
    report.append(f"\n5. **核心结论**: 主体性涌现的临界规划深度为 {threshold_depth if threshold_idx else '未确定'}")
    
    report_text = '\n'.join(report)
    
    with open(f'{output_dir}/scan_report.md', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    raw = {}
    for k, v in results.items():
        raw[k] = {key: val for key, val in v.items() if key != 'episode_rewards'}
    
    with open(f'{output_dir}/scan_results.json', 'w', encoding='utf-8') as f:
        json.dump(raw, f, indent=2, ensure_ascii=False)
    
    return report_text


# ===========================================================================
# 主程序
# ===========================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("  相变边界扫描实验")
    print("=" * 70)
    print()
    
    # 扫描规划深度 1-5
    results = run_full_scan(
        depths=[1, 2, 3, 4, 5],
        n_episodes=150,
        steps_per_episode=100,
        seed=42,
        verbose=True
    )
    
    print("\n" + "=" * 70)
    print("  生成扫描报告...")
    print("=" * 70)
    
    report = generate_scan_report(results)
    print("\n" + report)
    
    print("\n" + "=" * 70)
    print("  扫描完成。输出保存在 D:/HermesOutput/minimal_agency/")
    print("=" * 70)
