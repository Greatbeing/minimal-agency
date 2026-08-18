"""
最小智能闭环：计算实现
========================
GridWorld 环境中 L0-L6 主体的涌现相变实验
"""

import numpy as np
import random
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, List, Tuple
from enum import Enum
from collections import defaultdict
import json
from datetime import datetime


# ===========================================================================
# 第一部分：环境
# ===========================================================================

class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4

ACTIONS = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT, Action.STAY]
ACTION_DX = {Action.UP: (-1, 0), Action.DOWN: (1, 0), Action.LEFT: (0, -1), Action.RIGHT: (0, 1), Action.STAY: (0, 0)}

@dataclass
class GridWorld:
    N: int = 10
    view_radius: int = 2
    resource_density: float = 0.15
    obstacle_density: float = 0.08
    regen_rate: float = 0.02
    env_change_rate: float = 0.001
    seed: Optional[int] = None
    
    def __post_init__(self):
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)
        self.step_count = 0
        self._init_map()
    
    def _init_map(self):
        self.grid = np.zeros((self.N, self.N), dtype=int)
        # 0=empty, 1=resource, 2=obstacle
        for i in range(self.N):
            for j in range(self.N):
                r = random.random()
                if r < self.obstacle_density:
                    self.grid[i, j] = 2
                elif r < self.obstacle_density + self.resource_density:
                    self.grid[i, j] = 1
        # Ensure agent start is clear
        self.grid[0, 0] = 0
        self.agent_pos = (0, 0)
    
    def get_local_view(self, pos=None, radius=None):
        if pos is None:
            pos = self.agent_pos
        if radius is None:
            radius = self.view_radius
        r, c = x = pos
        view = []
        for dr in range(-radius, radius + 1):
            row = []
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.N and 0 <= nc < self.N:
                    row.append(self.grid[nr, nc])
                else:
                    row.append(-1)  # out of bounds
            view.append(row)
        return np.array(view)
    
    def observe(self):
        """观测 = 局部视野 + 自身位置"""
        local = self.get_local_view()
        return {
            'local_view': local,
            'pos': self.agent_pos,
            'step': self.step_count
        }
    
    def step(self, action: Action):
        """执行动作，返回 (obs, reward, done, info)"""
        dr, dc = ACTION_DX[action]
        r, c = self.agent_pos
        nr, nc = r + dr, c + dc
        
        reward = 0
        valid = True
        if 0 <= nr < self.N and 0 <= nc < self.N and self.grid[nr, nc] != 2:
            self.agent_pos = (nr, nc)
        else:
            valid = False
        
        # Collect resource
        if self.grid[self.agent_pos] == 1:
            reward = 1
            self.grid[self.agent_pos] = 0
        
        # Regeneration
        for i in range(self.N):
            for j in range(self.N):
                if self.grid[i, j] == 0 and random.random() < self.regen_rate:
                    self.grid[i, j] = 1
        
        # Environmental drift
        if random.random() < self.env_change_rate:
            i, j = random.randint(0, self.N-1), random.randint(0, self.N-1)
            if self.grid[i, j] != 2:
                self.grid[i, j] = 1 if self.grid[i, j] == 0 else 0
        
        self.step_count += 1
        
        obs = self.observe()
        return obs, reward, False, {'valid': valid}
    
    def get_state_vector(self):
        """Flattened grid state"""
        return self.grid.flatten()


# ===========================================================================
# 第二部分：主体能力等级
# ===========================================================================

class AgentLevel(Enum):
    L0 = 0  # Random
    L1 = 1  # Prediction only (observer)
    L2 = 2  # Prediction + Action
    L3 = 3  # Prediction + Action + Feedback
    L4 = 4  # + Self-model
    L5 = 5  # + Learning/Update
    L6 = 6  # + Counterfactual planning


@dataclass
class AgentState:
    """Agent's internal state"""
    level: AgentLevel = AgentLevel.L0
    # Generative model (conditions 1, 4)
    weights: Optional[np.ndarray] = None
    bias: Optional[np.ndarray] = None
    self_model_weight: float = 0.0  # How much agent uses self-model
    # History
    history: List = field(default_factory=list)
    # Update mechanism (condition 5)
    learning_rate: float = 0.01
    prediction_errors: List[float] = field(default_factory=list)
    # Counterfactual planning (condition 6)
    planning_depth: int = 1
    # Metrics
    total_reward: float = 0.0
    predictions_correct: int = 0
    predictions_total: int = 0
    self_other_correct: int = 0
    self_other_total: int = 0
    
    # For self-model: track own influence on environment
    last_action_effect: Optional[float] = None
    own_causal_model: Dict = field(default_factory=dict)
    
    # Meta-cognitive state
    confidence: float = 0.5
    adaptation_rate: float = 0.0


class MinimalAgent:
    """
    最小智能主体：根据等级配置不同能力
    """
    
    def __init__(self, level: AgentLevel, view_size: int = 25, seed: Optional[int] = None):
        self.level = level
        self.state = AgentState(level=level)
        self.view_size = view_size
        self.rng = np.random.RandomState(seed)
        
        if level.value >= AgentLevel.L1.value:
            # Generative model: predict next local view
            self.state.weights = self.rng.randn(view_size, view_size) * 0.01
            self.state.bias = np.zeros(view_size)
        
        if level.value >= AgentLevel.L4.value:
            # Self-model: additional weight for self-referential prediction
            self.state.self_model_weight = 0.5
            self.state.own_causal_model = defaultdict(lambda: defaultdict(float))
        
        if level.value >= AgentLevel.L5.value:
            self.state.learning_rate = 0.05
        
        if level.value >= AgentLevel.L6.value:
            self.state.planning_depth = 5
    
    def predict(self, obs: Dict, action: Optional[Action] = None) -> np.ndarray:
        """生成模型：预测下一观测"""
        local = obs['local_view'].flatten()[:self.view_size]
        if len(local) < self.view_size:
            local = np.pad(local, (0, self.view_size - len(local)))
        
        if self.level.value < AgentLevel.L1.value:
            return self.rng.randint(0, 3, self.view_size).astype(float)
        
        # Base prediction from generative model
        pred = self.state.weights @ local + self.state.bias
        pred = np.tanh(pred)  # normalize
        
        # Self-model contribution (condition 4)
        if self.level.value >= AgentLevel.L4.value and action is not None:
            action_effect = self._estimate_action_effect(action)
            pred += self.state.self_model_weight * action_effect
        
        return pred
    
    def _estimate_action_effect(self, action: Action) -> np.ndarray:
        """估计动作对环境的因果效应"""
        action_data = self.state.own_causal_model.get(action, {})
        if not action_data or 'effect' not in action_data:
            return np.zeros(self.view_size)
        return np.array(action_data['effect'])
    
    def select_action(self, obs: Dict) -> Action:
        """动作选择"""
        if self.level.value < AgentLevel.L2.value:
            return random.choice(ACTIONS)
        
        if self.level.value < AgentLevel.L6.value:
            # Greedy: move toward nearest visible resource
            local = obs['local_view']
            center = len(local) // 2
            
            best_action = Action.STAY
            best_score = -1
            
            for action in ACTIONS:
                dr, dc = ACTION_DX[action]
                # Check what we'd see if we moved
                r, c = center + dr, center + dc
                if 0 <= r < local.shape[0] and 0 <= c < local.shape[1]:
                    val = local[r, c]
                    score = 1 if val == 1 else (0 if val == 0 else -1)
                    if score > best_score:
                        best_score = score
                        best_action = action
            
            return best_action
        else:
            # Counterfactual planning: simulate multiple steps
            best_action = self._plan(obs)
            return best_action
    
    def _plan(self, obs: Dict, depth: int = 5) -> Action:
        """反事实规划：模拟动作序列，选择最优"""
        best_action = Action.STAY
        best_expected_reward = -float('inf')
        
        for first_action in ACTIONS:
            # Simulate this action + greedy rollout
            reward = self._simulate_action(obs, first_action)
            
            # Simple rollout (could be more sophisticated)
            simulated_obs = self._fake_step(obs, first_action)
            for d in range(depth - 1):
                next_action = self._greedy_action(simulated_obs)
                reward += self._simulate_action(simulated_obs, next_action) * (0.9 ** (d+1))
                simulated_obs = self._fake_step(simulated_obs, next_action)
            
            if reward > best_expected_reward:
                best_expected_reward = reward
                best_action = first_action
        
        return best_action
    
    def _simulate_action(self, obs: Dict, action: Action) -> float:
        """模拟单个动作的奖励"""
        local = obs['local_view']
        dr, dc = ACTION_DX[action]
        r, c = len(local)//2 + dr, len(local)//2 + dc
        if 0 <= r < local.shape[0] and 0 <= c < local.shape[1]:
            if local[r, c] == 1:
                return 1.0
        return 0.0
    
    def _fake_step(self, obs: Dict, action: Action) -> Dict:
        """生成假想下一步观测（简化）"""
        local = obs['local_view'].copy()
        dr, dc = ACTION_DX[action]
        r, c = len(local)//2 + dr, len(local)//2 + dc
        if 0 <= r < local.shape[0] and 0 <= c < local.shape[1]:
            if local[r, c] == 1:
                local[r, c] = 0  # collected
            # Shift view
            new_local = np.full_like(local, -1)
            for i in range(local.shape[0]):
                for j in range(local.shape[1]):
                    ni, nj = i + dr, j + dc
                    if 0 <= ni < local.shape[0] and 0 <= nj < local.shape[1]:
                        new_local[i, j] = local[ni, nj]
            local = new_local
        return {'local_view': local, 'pos': obs.get('pos', (0,0)), 'step': obs.get('step', 0) + 1}
    
    def _greedy_action(self, obs: Dict) -> Action:
        """贪婪动作选择"""
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
        """基于预测误差更新（条件 5）"""
        if self.level.value < AgentLevel.L3.value:
            return
        
        # Record history
        self.state.history.append({
            'obs': obs, 'action': action, 'next_obs': next_obs, 'reward': reward
        })
        
        if self.level.value < AgentLevel.L5.value:
            return
        
        # Compute prediction error
        pred = self.predict(obs, action)
        actual = next_obs['local_view'].flatten()[:self.view_size]
        if len(actual) < self.view_size:
            actual = np.pad(actual, (0, self.view_size - len(actual)))
        
        error = np.mean((pred - actual) ** 2)
        self.state.prediction_errors.append(error)
        
        # Update weights (gradient descent on prediction error)
        local = obs['local_view'].flatten()[:self.view_size]
        if len(local) < self.view_size:
            local = np.pad(local, (0, self.view_size - len(local)))
        
        grad_w = 2 * np.outer((pred - actual), local) / self.view_size
        grad_b = 2 * (pred - actual) / self.view_size
        
        self.state.weights -= self.state.learning_rate * grad_w
        self.state.bias -= self.state.learning_rate * grad_b
        
        # Update self-model (causal attribution)
        if self.level.value >= AgentLevel.L4.value:
            self._update_self_model(obs, action, next_obs, reward)
        
        # Adaptation tracking
        self.state.total_reward += reward
        self.state.adaptation_rate = reward
    
    def _update_self_model(self, obs: Dict, action: Action, next_obs: Dict, reward: float):
        """更新自我模型：区分自身行动 vs 环境变化"""
        # Track: did the environment change as predicted?
        pred = self.predict(obs, action)
        actual = next_obs['local_view'].flatten()[:self.view_size]
        if len(actual) < self.view_size:
            actual = np.pad(actual, (0, self.view_size - len(actual)))
        
        residual = actual - pred
        # Update causal model for this action
        if action not in self.state.own_causal_model:
            self.state.own_causal_model[action] = {'effect': np.zeros(self.view_size), 'count': 0}
        self.state.own_causal_model[action]['effect'] = residual.tolist()
        self.state.own_causal_model[action]['count'] += 1
        
        # Self-other discrimination metric
        if abs(reward) > 0:
            self.state.self_other_total += 1
            if self.state.own_causal_model[action]['count'] > 1:
                self.state.self_other_correct += 1


# ===========================================================================
# 第三部分：度量
# ===========================================================================

class MetricsCollector:
    """收集各项指标"""
    
    def __init__(self):
        self.data = defaultdict(list)
    
    def record(self, key: str, value: float, step: int = 0):
        self.data[key].append({'step': step, 'value': value})
    
    def get_summary(self) -> Dict:
        summary = {}
        for key, values in self.data.items():
            if values:
                vals = [v['value'] for v in values]
                summary[key] = {
                    'mean': np.mean(vals),
                    'std': np.std(vals),
                    'final': vals[-1] if vals else 0,
                    'max': np.max(vals) if vals else 0,
                    'min': np.min(vals) if vals else 0,
                }
        return summary
    
    def export(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(dict(self.data), f, indent=2, ensure_ascii=False)


def compute_subjectivity_index(agent: MinimalAgent, env: GridWorld, n_trials: int = 100) -> float:
    """
    主体性指数 SI = 1 - (perf_lesion / perf_complete)
    """
    # Test complete agent
    perf_complete = _test_agent(agent, env, n_trials)
    
    # Test lesioned agent (replace self-model with random)
    agent_lesioned = MinimalAgent(agent.level, seed=42)
    if agent_lesioned.state.weights is not None and agent.state.weights is not None:
        agent_lesioned.state.weights = agent.state.weights.copy()
        agent_lesioned.state.bias = agent.state.bias.copy()
        # Lesion: remove self-model contribution
        agent_lesioned.state.self_model_weight = 0.0
        # Add noise to simulate lesion
        agent_lesioned.state.weights += np.random.randn(*agent_lesioned.state.weights.shape) * 0.1
    
    perf_lesion = _test_agent(agent_lesioned, env, n_trials)
    
    if perf_complete + perf_lesion == 0:
        return 0.0
    
    si = 1.0 - (perf_lesion / (perf_complete + 1e-8))
    return max(0.0, min(1.0, si))


def _test_agent(agent: MinimalAgent, env: GridWorld, n_trials: int) -> float:
    """测试主体性能"""
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


def compute_self_other_discrimination(agent: MinimalAgent) -> float:
    """自我-他者区分率（SEC-2）"""
    if agent.state.self_other_total == 0:
        return 0.0
    return agent.state.self_other_correct / agent.state.self_other_total


def compute_historical_dependence(agent: MinimalAgent) -> float:
    """历史依赖度（SEC-3）"""
    # Approximation: variance in prediction errors indicates history dependence
    if len(agent.state.prediction_errors) < 10:
        return 0.0
    errors = agent.state.prediction_errors[-100:]
    return np.std(errors) / (np.mean(errors) + 1e-8)


def compute_adaptation_gain(agent: MinimalAgent, window: int = 100) -> float:
    """适应性增益"""
    if len(agent.state.prediction_errors) < 2 * window:
        return 0.0
    recent = agent.state.prediction_errors[-window:]
    older = agent.state.prediction_errors[-2*window:-window]
    return np.mean(older) - np.mean(recent)


# ===========================================================================
# 第四部分：实验运行器
# ===========================================================================

def run_experiment(level: AgentLevel, n_episodes: int = 200, steps_per_episode: int = 100, 
                  seed: int = 42, verbose: bool = False) -> Dict:
    """运行单个实验"""
    env = GridWorld(N=10, seed=seed)
    agent = MinimalAgent(level, seed=seed)
    metrics = MetricsCollector()
    
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
            
            if level.value >= AgentLevel.L5.value and agent.state.prediction_errors:
                ep_errors.append(agent.state.prediction_errors[-1])
            
            if done:
                break
        
        episode_rewards.append(ep_reward)
        if ep_errors:
            episode_errors.append(np.mean(ep_errors))
        
        if verbose and ep % 50 == 0:
            print(f"  L{level.value} | Episode {ep} | Reward: {ep_reward:.1f} | "
                  f"Avg Reward (last 10): {np.mean(episode_rewards[-10:]):.1f}")
    
    # Compute metrics
    si = compute_subjectivity_index(agent, env, n_trials=50) if level.value >= AgentLevel.L4.value else 0.0
    sod = compute_self_other_discrimination(agent)
    hd = compute_historical_dependence(agent)
    ag = compute_adaptation_gain(agent)
    
    result = {
        'level': level.value,
        'level_name': f'L{level.value}',
        'total_reward': sum(episode_rewards),
        'mean_reward': np.mean(episode_rewards[-50:]),
        'subjectivity_index': si,
        'self_other_discrimination': sod,
        'historical_dependence': hd,
        'adaptation_gain': ag,
        'episode_rewards': episode_rewards,
        'final_prediction_error': agent.state.prediction_errors[-1] if agent.state.prediction_errors else 1.0,
    }
    
    if verbose:
        print(f"  L{level.value} Complete | SI: {si:.3f} | SoD: {sod:.3f} | HD: {hd:.3f} | AG: {ag:.3f}")
    
    return result


def run_full_experiment(n_episodes: int = 200, steps_per_episode: int = 100, 
                       seed: int = 42, verbose: bool = True) -> Dict:
    """运行全部等级的完整实验"""
    results = {}
    
    levels = [
        AgentLevel.L0, AgentLevel.L1, AgentLevel.L2,
        AgentLevel.L3, AgentLevel.L4, AgentLevel.L5, AgentLevel.L6
    ]
    
    for level in levels:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running L{level.value} experiment...")
            print(f"{'='*60}")
        
        result = run_experiment(level, n_episodes, steps_per_episode, seed, verbose)
        results[f'L{level.value}'] = result
        
        if verbose:
            print(f"\nL{level.value} Results:")
            print(f"  Mean Reward: {result['mean_reward']:.2f}")
            print(f"  Subjectivity Index: {result['subjectivity_index']:.4f}")
            print(f"  Self-Other Discrimination: {result['self_other_discrimination']:.4f}")
            print(f"  Historical Dependence: {result['historical_dependence']:.4f}")
            print(f"  Adaptation Gain: {result['adaptation_gain']:.4f}")
    
    return results


# ===========================================================================
# 第五部分：可视化与报告
# ===========================================================================

def generate_report(results: Dict, output_dir: str = "D:/HermesOutput/minimal_agency"):
    """生成实验报告"""
    
    # Build markdown report
    report = []
    report.append("# 最小智能闭环：实验报告")
    report.append(f"\n**日期**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append(f"\n**实验配置**: 200 episodes × 100 steps, 10×10 GridWorld")
    report.append("\n---\n")
    
    # Summary table
    report.append("## 总体结果\n")
    report.append("| 等级 | 描述 | 平均奖励 | 主体性指数 | 自我-他者区分 | 历史依赖 | 适应性增益 |")
    report.append("|------|------|----------|-----------|--------------|---------|-----------|")
    
    level_descs = {
        'L0': '随机',
        'L1': '仅预测(观察者)',
        'L2': '预测+行动',
        'L3': '预测+行动+反馈',
        'L4': '+自我模型',
        'L5': '+学习更新',
        'L6': '+反事实规划'
    }
    
    for level_name, r in sorted(results.items()):
        report.append(f"| {level_name} | {level_descs.get(level_name, '')} | "
                     f"{r['mean_reward']:.2f} | {r['subjectivity_index']:.4f} | "
                     f"{r['self_other_discrimination']:.4f} | {r['historical_dependence']:.4f} | "
                     f"{r['adaptation_gain']:.4f} |")
    
    report.append("\n---\n")
    
    # Phase transition analysis
    report.append("## 涌现相变分析\n")
    
    si_values = {k: v['subjectivity_index'] for k, v in results.items()}
    reward_values = {k: v['mean_reward'] for k, v in results.items()}
    
    report.append("### 主体性指数 (SI) 变化\n")
    for level_name, si in sorted(si_values.items()):
        bar = '█' * int(si * 20)
        report.append(f"- {level_name}: {si:.4f} {bar}")
    
    report.append("\n### 相变检测\n")
    prev_si = 0
    transitions = []
    for level_name in sorted(si_values.keys()):
        si = si_values[level_name]
        jump = si - prev_si
        if jump > 0.05:
            transitions.append((level_name, jump))
        prev_si = si
    
    if transitions:
        for level_name, jump in transitions:
            report.append(f"- **相变点**: {level_name} (SI 跃迁 +{jump:.4f})")
    else:
        report.append("- 未检测到显著相变（SI 变化较连续）")
    
    report.append("\n---\n")
    
    # Analysis
    report.append("## 分析\n")
    
    l3 = results.get('L3', {})
    l4 = results.get('L4', {})
    l5 = results.get('L5', {})
    l6 = results.get('L6', {})
    
    report.append(f"### L3→L4 跃迁 (自我模型引入)\n")
    si_jump = l4.get('subjectivity_index', 0) - l3.get('subjectivity_index', 0)
    report.append(f"- SI 变化: {si_jump:+.4f}")
    report.append(f"- 解释: 自我模型的引入 {'显著' if abs(si_jump) > 0.05 else '轻微'}改变了主体性结构\n")
    
    report.append(f"### L5→L6 跃迁 (反事实规划引入)\n")
    si_jump = l6.get('subjectivity_index', 0) - l5.get('subjectivity_index', 0)
    report.append(f"- SI 变化: {si_jump:+.4f}")
    report.append(f"- 解释: 反事实规划的引入 {'显著' if abs(si_jump) > 0.05 else '轻微'}改变了主体性结构\n")
    
    # Reward progression
    report.append("### 奖励增长曲线\n")
    for level_name, r in sorted(results.items()):
        reward = r['mean_reward']
        bar = '█' * int(reward / 2)
        report.append(f"- {level_name}: {reward:.2f} {bar}")
    
    report.append("\n---\n")
    
    # Conclusion
    report.append("## 结论\n")
    
    max_si_level = max(si_values, key=si_values.get) if si_values else 'N/A'
    max_reward_level = max(reward_values, key=reward_values.get) if reward_values else 'N/A'
    
    report.append(f"1. **主体性最高等级**: {max_si_level} (SI={si_values.get(max_si_level, 0):.4f})")
    report.append(f"2. **性能最高等级**: {max_reward_level} (reward={reward_values.get(max_reward_level, 0):.2f})")
    
    if l6.get('subjectivity_index', 0) > l3.get('subjectivity_index', 0) + 0.1:
        report.append("3. **相变检测**: 在 L3→L4 或 L5→L6 处检测到主体性相变")
    else:
        report.append("3. **相变检测**: 主体性变化较连续，无显著相变点（可能需要更大规模实验）")
    
    report.append(f"\n4. **核心假说验证**: {'部分支持' if transitions else '未支持'} — "
                 f"{'存在相变点' if transitions else '主体性随能力增加渐进增长'}")
    
    report_text = '\n'.join(report)
    
    # Save report
    with open(f'{output_dir}/report.md', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    # Save raw results
    raw = {}
    for k, v in results.items():
        raw[k] = {key: val for key, val in v.items() if key != 'episode_rewards'}
    
    with open(f'{output_dir}/results.json', 'w', encoding='utf-8') as f:
        json.dump(raw, f, indent=2, ensure_ascii=False)
    
    return report_text


# ===========================================================================
# 主程序
# ===========================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("  最小智能闭环：涌现相变实验")
    print("=" * 70)
    print()
    
    results = run_full_experiment(
        n_episodes=200,
        steps_per_episode=100,
        seed=42,
        verbose=True
    )
    
    print("\n" + "=" * 70)
    print("  生成报告...")
    print("=" * 70)
    
    report = generate_report(results)
    print("\n" + report)
    
    print("\n" + "=" * 70)
    print("  实验完成。输出保存在 D:/HermesOutput/minimal_agency/")
    print("=" * 70)
