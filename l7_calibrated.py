"""
校准元认知 L7 实验
==================
验证假设：元认知需要二阶校准（calibrated metacognition）才有效
三路对比：L6（无元认知） vs L7-未校准 vs L7-校准
"""
import numpy as np
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import json
from datetime import datetime

# 复用基础环境定义
from simulation import (
    Action, ACTIONS, ACTION_DX, GridWorld,
    compute_subjectivity_index, _test_agent
)


# ============================================================
# 校准版 L7 主体的核心：二阶校准机制
# ============================================================

@dataclass
class CalibratedMetaState:
    """校准版元认知状态"""
    # 基础认知参数
    weights: Optional[np.ndarray] = None
    bias: Optional[np.ndarray] = None
    self_model_weight: float = 0.5
    learning_rate: float = 0.05
    planning_depth: int = 5

    # 元认知参数（校准版）
    confidence_estimate: float = 0.5
    confidence_smoothed: float = 0.5      # 动量平滑的信心
    confidence_momentum: float = 0.95     # 平滑系数（关键修复1）
    lr_deadzone: float = 0.005            # 学习率调整死区（关键修复2）
    lr_adjust_count: int = 0
    meta_learning_rate: float = 0.01

    # 追踪
    prediction_errors: List[float] = field(default_factory=list)
    confidence_history: List[float] = field(default_factory=list)
    meta_updates: int = 0


class CalibratedAgent:
    """校准版 L7 主体：置信度动量平滑 + 学习率死区"""

    def __init__(self, view_size: int = 25, seed: Optional[int] = None,
                 calibrated: bool = True):
        self.rng = np.random.RandomState(seed)
        self.view_size = view_size
        self.state = CalibratedMetaState()
        self.calibrated = calibrated
        self.state.weights = self.rng.randn(view_size, view_size) * 0.01
        self.state.bias = np.zeros(view_size)
        self.own_causal_model = {}
        self.history = []
        self.total_reward = 0.0

    def predict(self, obs: Dict, action: Optional[Action] = None) -> np.ndarray:
        local = obs['local_view'].flatten()[:self.view_size]
        if len(local) < self.view_size:
            local = np.pad(local, (0, self.view_size - len(local)))
        pred = self.state.weights @ local + self.state.bias
        pred = np.tanh(pred)
        if action is not None and action in self.own_causal_model:
            effect = self.own_causal_model[action].get('effect', np.zeros(self.view_size))
            pred += self.state.self_model_weight * np.array(effect)
        return pred

    def select_action(self, obs: Dict) -> Action:
        # 置信度影响探索/利用权衡
        conf = self.state.confidence_smoothed if self.calibrated else self.state.confidence_estimate
        explore_threshold = max(0.05, 1.0 - conf)
        if self.rng.random() < explore_threshold:
            return random.choice(ACTIONS)
        return self._plan(obs, depth=self.state.planning_depth)

    def _plan(self, obs: Dict, depth: int = 5) -> Action:
        best_action = Action.STAY
        best_reward = -float('inf')
        for first in ACTIONS:
            reward = self._sim(obs, first)
            sim_obs = self._fake_step(obs, first)
            for d in range(depth - 1):
                na = self._greedy(sim_obs)
                reward += self._sim(sim_obs, na) * (0.9 ** (d + 1))
                sim_obs = self._fake_step(sim_obs, na)
            if reward > best_reward:
                best_reward = reward
                best_action = first
        return best_action

    def _sim(self, obs: Dict, action: Action) -> float:
        local = obs['local_view']
        dr, dc = ACTION_DX[action]
        r, c = len(local)//2 + dr, len(local)//2 + dc
        if 0 <= r < local.shape[0] and 0 <= c < local.shape[1]:
            return 1.0 if local[r, c] == 1 else 0.0
        return 0.0

    def _fake_step(self, obs: Dict, action: Action) -> Dict:
        local = obs['local_view'].copy()
        dr, dc = ACTION_DX[action]
        r, c = len(local)//2 + dr, len(local)//2 + dc
        if 0 <= r < local.shape[0] and 0 <= c < local.shape[1]:
            if local[r, c] == 1:
                local[r, c] = 0
            new_local = np.full_like(local, -1)
            for i in range(local.shape[0]):
                for j in range(local.shape[1]):
                    ni, nj = i + dr, j + dc
                    if 0 <= ni < local.shape[0] and 0 <= nj < local.shape[1]:
                        new_local[i, j] = local[ni, nj]
            local = new_local
        return {'local_view': local, 'pos': obs.get('pos', (0,0)), 'step': obs.get('step', 0) + 1}

    def _greedy(self, obs: Dict) -> Action:
        local = obs['local_view']
        center = len(local) // 2
        best_action = Action.STAY
        best_score = -1
        for action in ACTIONS:
            dr, dc = ACTION_DX[action]
            r, c = center + dr, center + dc
            if 0 <= r < local.shape[0] and 0 <= c < local.shape[1]:
                val = local[r, c]
                score = 1 if val == 1 else (0 if val == 0 else -1)
                if score > best_score:
                    best_score = score
                    best_action = action
        return best_action

    def update(self, obs: Dict, action: Action, next_obs: Dict, reward: float):
        """基础认知更新 + 元认知更新"""
        self.history.append({'obs': obs, 'action': action, 'next_obs': next_obs, 'reward': reward})

        pred = self.predict(obs, action)
        actual = next_obs['local_view'].flatten()[:self.view_size]
        if len(actual) < self.view_size:
            actual = np.pad(actual, (0, self.view_size - len(actual)))
        error = np.mean((pred - actual) ** 2)
        self.state.prediction_errors.append(error)

        # 权重更新
        local = obs['local_view'].flatten()[:self.view_size]
        if len(local) < self.view_size:
            local = np.pad(local, (0, self.view_size - len(local)))
        grad_w = 2 * np.outer((pred - actual), local) / self.view_size
        grad_b = 2 * (pred - actual) / self.view_size
        self.state.weights -= self.state.learning_rate * grad_w
        self.state.bias -= self.state.learning_rate * grad_b

        # 因果模型更新
        if action not in self.own_causal_model:
            self.own_causal_model[action] = {'effect': np.zeros(self.view_size), 'count': 0}
        self.own_causal_model[action]['effect'] = (actual - pred).tolist()
        self.own_causal_model[action]['count'] += 1

        self.total_reward += reward
        self._meta_update(error, reward)

    def _meta_update(self, error: float, reward: float):
        """元认知更新（含校准机制）"""
        # 基础信心更新（瞬时值）
        if len(self.state.prediction_errors) >= 10:
            recent = np.mean(self.state.prediction_errors[-10:])
            self.state.confidence_estimate = max(0.0, min(1.0, 1.0 - recent))

        # 校准版：动量平滑（关键修复1）
        if self.calibrated:
            self.state.confidence_smoothed = (
                self.state.confidence_momentum * self.state.confidence_smoothed
                + (1 - self.state.confidence_momentum) * self.state.confidence_estimate
            )
        else:
            self.state.confidence_smoothed = self.state.confidence_estimate

        # 学习率调整（校准版带死区，关键修复2）
        if len(self.state.prediction_errors) >= 20:
            recent = np.mean(self.state.prediction_errors[-10:])
            older = np.mean(self.state.prediction_errors[-20:-10])
            trend = recent - older  # 负 = 改善

            if self.calibrated:
                # 只在趋势超过死区时调整，且步长更小
                if trend > self.state.lr_deadzone:
                    self.state.learning_rate = min(0.15, self.state.learning_rate * 1.03)
                    self.state.lr_adjust_count += 1
                elif trend < -self.state.lr_deadzone:
                    self.state.learning_rate = max(0.001, self.state.learning_rate * 0.97)
                    self.state.lr_adjust_count += 1
            else:
                # 未校准：大幅快速调整（产生振荡）
                if trend > 0.01:
                    self.state.learning_rate = min(0.2, self.state.learning_rate * 1.1)
                elif trend < -0.01:
                    self.state.learning_rate = max(0.001, self.state.learning_rate * 0.95)

        self.state.confidence_history.append(self.state.confidence_smoothed)
        self.state.meta_updates += 1


def compute_si_from_calibrated(agent: CalibratedAgent, env: GridWorld, n_trials: int = 50) -> float:
    """计算主体性指数"""
    perf_complete = _test_calibrated_agent(agent, env, n_trials)

    # Lesioned: remove metacognition modulation
    agent_lesioned = CalibratedAgent(view_size=agent.view_size, seed=999, calibrated=agent.calibrated)
    if agent_lesioned.state.weights is not None and agent.state.weights is not None:
        agent_lesioned.state.weights = agent.state.weights.copy()
        agent_lesioned.state.bias = agent.state.bias.copy()
        agent_lesioned.own_causal_model = {k: {'effect': np.array(v['effect']), 'count': v['count']}
                                          for k, v in agent.own_causal_model.items()}
        agent_lesioned.state.confidence_estimate = 0.5
        agent_lesioned.state.confidence_smoothed = 0.5
        agent_lesioned.state.meta_learning_rate = 0.0

    perf_lesion = _test_calibrated_agent(agent_lesioned, env, n_trials)

    if perf_complete + perf_lesion == 0:
        return 0.0
    si = 1.0 - (perf_lesion / (perf_complete + 1e-8))
    return max(0.0, min(1.0, si))


def _test_calibrated_agent(agent: CalibratedAgent, env: GridWorld, n_trials: int) -> float:
    total_reward = 0
    for trial in range(n_trials):
        env._init_map()
        for step in range(50):
            obs = env.observe()
            action = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
    return total_reward / n_trials


def run_calibrated_experiment(calibrated: bool, n_episodes: int = 200, steps_per_episode: int = 100,
                              seed: int = 42, verbose: bool = True) -> Dict:
    """运行校准/未校准实验"""
    env = GridWorld(N=10, seed=seed)
    agent = CalibratedAgent(view_size=25, seed=seed, calibrated=calibrated)

    episode_rewards = []
    episode_confidences = []
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
        episode_confidences.append(agent.state.confidence_smoothed)
        if ep_errors:
            episode_errors.append(np.mean(ep_errors))

        if verbose and ep % 50 == 0:
            conf = agent.state.confidence_smoothed
            lr = agent.state.learning_rate
            print(f"  {'Calibrated' if calibrated else 'Uncalibrated'} | Ep {ep} | Reward: {ep_reward:.1f} | Conf: {conf:.3f} | LR: {lr:.4f}")

    si = compute_si_from_calibrated(agent, env, n_trials=50)

    return {
        'calibrated': calibrated,
        'total_reward': sum(episode_rewards),
        'mean_reward': np.mean(episode_rewards[-50:]),
        'subjectivity_index': si,
        'final_confidence': agent.state.confidence_smoothed,
        'final_learning_rate': agent.state.learning_rate,
        'lr_adjust_count': agent.state.lr_adjust_count,
        'confidence_history': agent.state.confidence_history,
        'prediction_errors': agent.state.prediction_errors,
        'episode_rewards': episode_rewards,
        'episode_confidences': episode_confidences,
    }


def run_l6_baseline(n_episodes: int = 200, steps_per_episode: int = 100,
                    seed: int = 42, verbose: bool = True) -> Dict:
    """运行 L6 基线"""
    from simulation import AgentLevel, MinimalAgent
    env = GridWorld(N=10, seed=seed)
    agent = MinimalAgent(AgentLevel.L6, seed=seed)

    episode_rewards = []

    for ep in range(n_episodes):
        env._init_map()
        ep_reward = 0
        for step in range(steps_per_episode):
            obs = env.observe()
            action = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            agent.update(obs, action, next_obs, reward)
            ep_reward += reward
            if done:
                break
        episode_rewards.append(ep_reward)

    si = compute_subjectivity_index(agent, env, n_trials=50)

    return {
        'level': 'L6',
        'total_reward': sum(episode_rewards),
        'mean_reward': np.mean(episode_rewards[-50:]),
        'subjectivity_index': si,
        'episode_rewards': episode_rewards,
    }


# ============================================================
# 主程序：三路对比
# ============================================================

if __name__ == '__main__':
    print("=" * 70)
    print("  校准元认知 L7 三路对比实验")
    print("=" * 70)
    print()

    # 1. L6 基线
    print("=" * 70)
    print("Experiment 1: L6 Baseline (no metacognition)")
    print("=" * 70)
    l6 = run_l6_baseline(n_episodes=200, steps_per_episode=100, seed=42, verbose=True)
    print(f"\nL6: Reward={l6['mean_reward']:.2f}, SI={l6['subjectivity_index']:.4f}")

    # 2. L7 未校准
    print("\n" + "=" * 70)
    print("Experiment 2: L7 Uncalibrated (pathological metacognition)")
    print("=" * 70)
    l7_uncal = run_calibrated_experiment(False, n_episodes=200, steps_per_episode=100, seed=42, verbose=True)
    print(f"\nL7-Uncal: Reward={l7_uncal['mean_reward']:.2f}, SI={l7_uncal['subjectivity_index']:.4f}")
    print(f"  Final Conf: {l7_uncal['final_confidence']:.4f}, Final LR: {l7_uncal['final_learning_rate']:.4f}")
    print(f"  Conf history std: {np.std(l7_uncal['confidence_history']):.4f}")

    # 3. L7 校准
    print("\n" + "=" * 70)
    print("Experiment 3: L7 Calibrated (calibrated metacognition)")
    print("=" * 70)
    l7_cal = run_calibrated_experiment(True, n_episodes=200, steps_per_episode=100, seed=42, verbose=True)
    print(f"\nL7-Cal: Reward={l7_cal['mean_reward']:.2f}, SI={l7_cal['subjectivity_index']:.4f}")
    print(f"  Final Conf: {l7_cal['final_confidence']:.4f}, Final LR: {l7_cal['final_learning_rate']:.4f}")
    print(f"  Conf history std: {np.std(l7_cal['confidence_history']):.4f}")
    print(f"  LR adjust count: {l7_cal['lr_adjust_count']}")

    # 三路对比总结
    print("\n" + "=" * 70)
    print("  三路对比总结")
    print("=" * 70)
    print(f"| 版本          | 平均奖励 | SI      | 置信度稳定性(std) | 学习率调整次数 |")
    print(f"|---------------|----------|---------|-------------------|---------------|")
    print(f"| L6 (无元认知) | {l6['mean_reward']:>6.1f} | {l6['subjectivity_index']:.4f} | {'N/A':>16} | {'N/A':>12} |")
    print(f"| L7 未校准     | {l7_uncal['mean_reward']:>6.1f} | {l7_uncal['subjectivity_index']:.4f} | {np.std(l7_uncal['confidence_history']):>16.4f} | {'N/A':>12} |")
    print(f"| L7 校准       | {l7_cal['mean_reward']:>6.1f} | {l7_cal['subjectivity_index']:.4f} | {np.std(l7_cal['confidence_history']):>16.4f} | {l7_cal['lr_adjust_count']:>12} |")

    # 保存结果
    output = {
        'L6': {k: v for k, v in l6.items() if k != 'episode_rewards'},
        'L7_uncalibrated': {k: v for k, v in l7_uncal.items() if k not in ['episode_rewards', 'episode_confidences', 'confidence_history', 'prediction_errors']},
        'L7_calibrated': {k: v for k, v in l7_cal.items() if k not in ['episode_rewards', 'episode_confidences', 'confidence_history', 'prediction_errors']},
    }

    with open('D:/HermesOutput/minimal_agency/l7_three_way_comparison.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print("\n" + "=" * 70)
    print("  实验完成。输出保存在 D:/HermesOutput/minimal_agency/")
    print("=" * 70)