"""
非平稳环境实验
==============
验证假设：元认知（L7 校准）在非平稳/高复杂度环境中超越 L6
"""
import numpy as np
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import json
from datetime import datetime

# 复用
from simulation import (
    Action, ACTIONS, ACTION_DX, GridWorld,
    compute_subjectivity_index, _test_agent, AgentLevel, MinimalAgent
)
from l7_calibrated import (
    CalibratedMetaState, CalibratedAgent, compute_si_from_calibrated
)


# ============================================================
# 非平稳环境：资源分布随时间剧烈变化
# ============================================================

class NonstationaryGridWorld(GridWorld):
    """
    非平稳 GridWorld：
    - 资源区域周期性移动（模拟季节性/环境变化）
    - 障碍物偶尔重组
    - 要求主体持续适应，固定策略会失效
    """
    def __init__(self, N: int = 15, view_radius: int = 3,
                 resource_density: float = 0.08, obstacle_density: float = 0.12,
                 regen_rate: float = 0.005, env_change_rate: float = 0.02,
                 shift_period: int = 200, shift_magnitude: int = 3,
                 seed: Optional[int] = None):
        super().__init__(N, view_radius, resource_density, obstacle_density,
                         regen_rate, env_change_rate, seed)
        self.shift_period = shift_period        # 资源区域移动周期
        self.shift_magnitude = shift_magnitude  # 移动幅度
        self._init_resource_zones()

    def _init_map(self):
        """初始化：创建明显的资源区域"""
        self.grid = np.zeros((self.N, self.N), dtype=int)
        # 障碍物
        for i in range(self.N):
            for j in range(self.N):
                if random.random() < self.obstacle_density:
                    self.grid[i, j] = 2
        # 资源集中在 2-3 个区域
        self._init_resource_zones()
        self.grid[0, 0] = 0
        self.agent_pos = (0, 0)

    def _init_resource_zones(self):
        """创建集中的资源区域"""
        self.resource_zones = []
        n_zones = random.randint(2, 3)
        for _ in range(n_zones):
            # 随机中心，但离边界远一点
            cx = random.randint(3, self.N - 4)
            cy = random.randint(3, self.N - 4)
            radius = random.randint(2, 4)
            self.resource_zones.append((cx, cy, radius))
            # 填充资源
            for i in range(max(0, cx - radius), min(self.N, cx + radius + 1)):
                for j in range(max(0, cy - radius), min(self.N, cy + radius + 1)):
                    if self.grid[i, j] != 2 and random.random() < 0.6:
                        self.grid[i, j] = 1

    def _shift_resources(self):
        """周期性移动资源区域"""
        for idx, (cx, cy, radius) in enumerate(self.resource_zones):
            # 清除旧区域
            for i in range(max(0, cx - radius), min(self.N, cx + radius + 1)):
                for j in range(max(0, cy - radius), min(self.N, cy + radius + 1)):
                    if self.grid[i, j] == 1:
                        self.grid[i, j] = 0
            # 新位置（随机漂移）
            new_cx = np.clip(cx + random.randint(-self.shift_magnitude, self.shift_magnitude), 3, self.N - 4)
            new_cy = np.clip(cy + random.randint(-self.shift_magnitude, self.shift_magnitude), 3, self.N - 4)
            self.resource_zones[idx] = (new_cx, new_cy, radius)
            # 填充新区域
            for i in range(max(0, new_cx - radius), min(self.N, new_cx + radius + 1)):
                for j in range(max(0, new_cy - radius), min(self.N, new_cy + radius + 1)):
                    if self.grid[i, j] != 2 and random.random() < 0.6:
                        self.grid[i, j] = 1

    def step(self, action: Action):
        obs, reward, done, info = super().step(action)
        # 周期性移动资源
        if self.step_count % self.shift_period == 0:
            self._shift_resources()
            # 可选：稍微重组障碍物
            if random.random() < 0.1:
                self._reorganize_obstacles()
        return obs, reward, done, info

    def _reorganize_obstacles(self):
        """偶尔重组障碍物"""
        for i in range(self.N):
            for j in range(self.N):
                if self.grid[i, j] == 2 and random.random() < 0.3:
                    self.grid[i, j] = 0
                elif self.grid[i, j] == 0 and random.random() < 0.05:
                    self.grid[i, j] = 2
        # 确保智能体位置不是障碍物
        self.grid[self.agent_pos] = 0


def run_nonstationary_experiment(agent_type: str, n_episodes: int = 150,
                                 steps_per_episode: int = 150,
                                 seed: int = 42, verbose: bool = True) -> Dict:
    """运行非平稳环境实验"""
    env = NonstationaryGridWorld(N=15, shift_period=200, shift_magnitude=3, seed=seed)

    if agent_type == 'L6':
        agent = MinimalAgent(AgentLevel.L6, seed=seed)
    elif agent_type == 'L7_calibrated':
        agent = CalibratedAgent(view_size=49, seed=seed, calibrated=True)
    else:
        raise ValueError(f"Unknown agent_type: {agent_type}")

    episode_rewards = []
    episode_confidences = []
    episode_errors = []
    phase_rewards = []  # 每个资源移动周期的奖励

    for ep in range(n_episodes):
        env._init_map()
        ep_reward = 0
        ep_errors = []
        phase_reward = 0
        phase_step = 0

        for step in range(steps_per_episode):
            obs = env.observe()
            action = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            agent.update(obs, action, next_obs, reward)

            ep_reward += reward
            phase_reward += reward
            phase_step += 1

            if agent_type.startswith('L7') and agent.state.prediction_errors:
                ep_errors.append(agent.state.prediction_errors[-1])

            # 记录每个周期的奖励
            if env.step_count % env.shift_period == 0 and phase_step > 0:
                phase_rewards.append(phase_reward / phase_step)
                phase_reward = 0
                phase_step = 0

            if done:
                break

        episode_rewards.append(ep_reward)
        if episode_errors:
            episode_errors.append(np.mean(ep_errors))
        if agent_type.startswith('L7'):
            episode_confidences.append(agent.state.confidence_smoothed)

        if verbose and ep % 30 == 0:
            conf = agent.state.confidence_smoothed if agent_type.startswith('L7') else 'N/A'
            lr = agent.state.learning_rate if agent_type.startswith('L7') else 'N/A'
            print(f"  {agent_type} | Ep {ep} | Reward: {ep_reward:.1f} | Conf: {conf} | LR: {lr}")

    # 计算 SI
    if agent_type == 'L6':
        si = compute_subjectivity_index(agent, env, n_trials=30)
    else:
        si = compute_si_from_calibrated(agent, env, n_trials=30)

    return {
        'agent_type': agent_type,
        'total_reward': sum(episode_rewards),
        'mean_reward': np.mean(episode_rewards[-30:]),
        'subjectivity_index': si,
        'final_confidence': agent.state.confidence_smoothed if agent_type.startswith('L7') else 'N/A',
        'final_learning_rate': agent.state.learning_rate if agent_type.startswith('L7') else 'N/A',
        'phase_rewards': phase_rewards,
        'adaptation_slope': np.polyfit(range(len(phase_rewards)), phase_rewards, 1)[0] if len(phase_rewards) > 1 else 0,
    }


# ============================================================
# 主程序：对比实验
# ============================================================

if __name__ == '__main__':
    print("=" * 70)
    print("  非平稳环境：L6 vs L7 校准元认知")
    print("=" * 70)
    print()

    # L6 基线
    print("=" * 70)
    print("Experiment 1: L6 in Nonstationary Environment")
    print("=" * 70)
    l6 = run_nonstationary_experiment('L6', n_episodes=150, steps_per_episode=150, seed=42, verbose=True)
    print(f"\nL6: Mean Reward={l6['mean_reward']:.2f}, SI={l6['subjectivity_index']:.4f}")
    if l6['phase_rewards']:
        print(f"  Phase rewards: {np.mean(l6['phase_rewards']):.2f} ± {np.std(l6['phase_rewards']):.2f}")
        print(f"  Adaptation slope: {l6['adaptation_slope']:.4f}")

    # L7 校准
    print("\n" + "=" * 70)
    print("Experiment 2: L7 Calibrated in Nonstationary Environment")
    print("=" * 70)
    l7 = run_nonstationary_experiment('L7_calibrated', n_episodes=150, steps_per_episode=150, seed=42, verbose=True)
    print(f"\nL7-Cal: Mean Reward={l7['mean_reward']:.2f}, SI={l7['subjectivity_index']:.4f}")
    print(f"  Final Conf: {l7['final_confidence']:.4f}, Final LR: {l7['final_learning_rate']:.4f}")
    if l7['phase_rewards']:
        print(f"  Phase rewards: {np.mean(l7['phase_rewards']):.2f} ± {np.std(l7['phase_rewards']):.2f}")
        print(f"  Adaptation slope: {l7['adaptation_slope']:.4f}")

    # 对比
    print("\n" + "=" * 70)
    print("  非平稳环境对比总结")
    print("=" * 70)
    print(f"| 版本      | 平均奖励 | SI      | 阶段奖励均值 | 阶段奖励稳定性 | 适应斜率 |")
    print(f"|-----------|----------|---------|-------------|---------------|----------|")
    print(f"| L6        | {l6['mean_reward']:>6.1f} | {l6['subjectivity_index']:.4f} | "
          f"{np.mean(l6['phase_rewards']) if l6['phase_rewards'] else 0:>11.2f} | "
          f"{np.std(l6['phase_rewards']) if l6['phase_rewards'] else 0:>12.2f} | "
          f"{l6['adaptation_slope']:.4f} |")
    print(f"| L7 校准   | {l7['mean_reward']:>6.1f} | {l7['subjectivity_index']:.4f} | "
          f"{np.mean(l7['phase_rewards']) if l7['phase_rewards'] else 0:>11.2f} | "
          f"{np.std(l7['phase_rewards']) if l7['phase_rewards'] else 0:>12.2f} | "
          f"{l7['adaptation_slope']:.4f} |")

    reward_diff = l7['mean_reward'] - l6['mean_reward']
    print(f"\n奖励差异: L7 - L6 = {reward_diff:+.1f}")
    if reward_diff > 0:
        print("→ L7 校准元认知在非平稳环境中**超越** L6")
    else:
        print("→ L6 仍优于 L7（可能环境仍不够复杂，或元认知收益未覆盖开销）")

    # 保存
    output = {
        'L6': {k: v for k, v in l6.items() if k not in ['phase_rewards']},
        'L7_calibrated': {k: v for k, v in l7.items() if k not in ['phase_rewards']},
    }
    with open('D:/HermesOutput/minimal_agency/nonstationary_comparison.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print("\n" + "=" * 70)
    print("  实验完成。输出保存在 D:/HermesOutput/minimal_agency/")
    print("=" * 70)