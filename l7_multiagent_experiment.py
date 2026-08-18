"""
L7 元认知主体 + 多主体社会性实验
===================================
核心问题：
1. 元认知（自我模型关于自我模型的模型）如何增强主体性？
2. 多主体环境中，交互主体性如何涌现？
3. 主体间性（intersubjectivity）是否可测量？
"""

import numpy as np
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum
import json
from datetime import datetime


# ===========================================================================
# 第一部分：更复杂的环境（需要深度规划）
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
class ComplexGridWorld:
    """
    更复杂的环境：
    - 更大的网格 (20x20)
    - 资源稀疏且动态变化
    - 障碍物形成迷宫结构
    - 需要长期规划才能高效收集
    """
    N: int = 20
    view_radius: int = 3
    resource_density: float = 0.05  # 更稀疏
    obstacle_density: float = 0.15  # 更多障碍
    regen_rate: float = 0.005
    env_change_rate: float = 0.01
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
        # Create maze-like structure
        for i in range(self.N):
            for j in range(self.N):
                r = random.random()
                if r < self.obstacle_density:
                    self.grid[i, j] = 2
                elif r < self.obstacle_density + self.resource_density:
                    self.grid[i, j] = 1
        # Ensure start is clear
        self.grid[0, 0] = 0
        self.grid[0, 1] = 0
        self.grid[1, 0] = 0
        self.agent_pos = (0, 0)
    
    def get_local_view(self, pos=None, radius=None):
        if pos is None:
            pos = self.agent_pos
        if radius is None:
            radius = self.view_radius
        r, c = pos
        view = []
        for dr in range(-radius, radius + 1):
            row = []
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.N and 0 <= nc < self.N:
                    row.append(self.grid[nr, nc])
                else:
                    row.append(-1)
            view.append(row)
        return np.array(view)
    
    def observe(self):
        local = self.get_local_view()
        return {
            'local_view': local,
            'pos': self.agent_pos,
            'step': self.step_count
        }
    
    def step(self, action: Action):
        dr, dc = ACTION_DX[action]
        r, c = self.agent_pos
        nr, nc = r + dr, c + dc
        
        reward = -0.01  # small step penalty to encourage efficiency
        valid = True
        if 0 <= nr < self.N and 0 <= nc < self.N and self.grid[nr, nc] != 2:
            self.agent_pos = (nr, nc)
        else:
            valid = False
            reward = -0.05  # wall penalty
        
        # Collect resource
        if self.grid[self.agent_pos] == 1:
            reward = 1
            self.grid[self.agent_pos] = 0
        
        # Sparse regeneration
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


# ===========================================================================
# 第二部分：L7 元认知主体
# ===========================================================================

@dataclass
class MetacognitiveState:
    """元认知状态：关于自身认知能力的模型"""
    # 基础认知参数
    weights: Optional[np.ndarray] = None
    bias: Optional[np.ndarray] = None
    self_model_weight: float = 0.5
    learning_rate: float = 0.05
    planning_depth: int = 5
    
    # 元认知参数（L7 新增）
    confidence_estimate: float = 0.5  # 对当前自我模型准确性的估计
    learning_rate_estimate: float = 0.05  # 对当前学习速率是否充足的估计
    adaptability_estimate: float = 0.5  # 对自身适应能力的估计
    meta_learning_rate: float = 0.01  # 元学习速率（调整学习速率的速率）
    
    # 历史追踪
    prediction_errors: List[float] = field(default_factory=list)
    confidence_history: List[float] = field(default_factory=list)
    performance_history: List[float] = field(default_factory=list)
    
    # 元认知更新计数
    meta_updates: int = 0


class MetacognitiveAgent:
    """
    L7 元认知主体：
    - 拥有关于自身认知能力的模型
    - 能估计"我的自我模型有多准确"
    - 能根据估计调整学习策略
    - 能根据估计调整探索/利用权衡
    """
    
    def __init__(self, view_size: int = 49, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self.view_size = view_size
        self.state = MetacognitiveState()
        
        # 基础认知
        self.state.weights = self.rng.randn(view_size, view_size) * 0.01
        self.state.bias = np.zeros(view_size)
        
        # 因果模型
        self.own_causal_model = {}
        self.history = []
        self.total_reward = 0.0
    
    def predict(self, obs: Dict, action: Optional[Action] = None) -> np.ndarray:
        """生成模型预测"""
        local = obs['local_view'].flatten()[:self.view_size]
        if len(local) < self.view_size:
            local = np.pad(local, (0, self.view_size - len(local)))
        
        pred = self.state.weights @ local + self.state.bias
        pred = np.tanh(pred)
        
        if action is not None and action in self.own_causal_model:
            action_effect = self.own_causal_model[action].get('effect', np.zeros(self.view_size))
            pred += self.state.self_model_weight * np.array(action_effect)
        
        return pred
    
    def select_action(self, obs: Dict) -> Action:
        """基于元认知的动作选择"""
        # 元认知影响：信心低时更多探索
        confidence = self.state.confidence_estimate
        explore_threshold = max(0.1, 1.0 - confidence)
        
        if self.rng.random() < explore_threshold:
            # 探索：随机动作
            return random.choice(ACTIONS)
        
        # 利用：反事实规划
        return self._plan(obs, depth=self.state.planning_depth)
    
    def _plan(self, obs: Dict, depth: int = 5) -> Action:
        """反事实规划"""
        best_action = Action.STAY
        best_expected_reward = -float('inf')
        
        for first_action in ACTIONS:
            reward = self._simulate_action(obs, first_action)
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
        local = obs['local_view']
        dr, dc = ACTION_DX[action]
        r, c = len(local)//2 + dr, len(local)//2 + dc
        if 0 <= r < local.shape[0] and 0 <= c < local.shape[1]:
            if local[r, c] == 1:
                return 1.0
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
    
    def _greedy_action(self, obs: Dict) -> Action:
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
        """基础认知更新"""
        self.history.append({
            'obs': obs, 'action': action, 'next_obs': next_obs, 'reward': reward
        })
        
        pred = self.predict(obs, action)
        actual = next_obs['local_view'].flatten()[:self.view_size]
        if len(actual) < self.view_size:
            actual = np.pad(actual, (0, self.view_size - len(actual)))
        
        error = np.mean((pred - actual) ** 2)
        self.state.prediction_errors.append(error)
        
        # 更新权重
        local = obs['local_view'].flatten()[:self.view_size]
        if len(local) < self.view_size:
            local = np.pad(local, (0, self.view_size - len(local)))
        
        grad_w = 2 * np.outer((pred - actual), local) / self.view_size
        grad_b = 2 * (pred - actual) / self.view_size
        
        self.state.weights -= self.state.learning_rate * grad_w
        self.state.bias -= self.state.learning_rate * grad_b
        
        # 更新因果模型
        if action not in self.own_causal_model:
            self.own_causal_model[action] = {'effect': np.zeros(self.view_size), 'count': 0}
        self.own_causal_model[action]['effect'] = (actual - pred).tolist()
        self.own_causal_model[action]['count'] += 1
        
        self.total_reward += reward
        
        # 元认知更新
        self._metacognitive_update(error, reward)
    
    def _metacognitive_update(self, error: float, reward: float):
        """
        元认知更新：关于自身认知能力的模型如何变化
        """
        # 更新信心估计：基于近期预测误差
        if len(self.state.prediction_errors) >= 10:
            recent_errors = self.state.prediction_errors[-10:]
            mean_error = np.mean(recent_errors)
            # 信心 = 1 - 归一化误差
            self.state.confidence_estimate = max(0.0, min(1.0, 1.0 - mean_error))
        
        # 更新学习速率估计：基于误差变化趋势
        if len(self.state.prediction_errors) >= 20:
            recent = np.mean(self.state.prediction_errors[-10:])
            older = np.mean(self.state.prediction_errors[-20:-10])
            error_trend = recent - older  # 负值 = 改善
            
            # 如果误差在下降，当前学习速率足够；如果误差在上升，需要调整
            if error_trend > 0.01:  # 误差在上升
                # 增加学习速率
                self.state.learning_rate = min(0.2, self.state.learning_rate * 1.1)
            elif error_trend < -0.01:  # 误差在下降
                # 可以稍微降低学习速率以稳定
                self.state.learning_rate = max(0.001, self.state.learning_rate * 0.95)
        
        # 更新适应性估计：基于奖励趋势
        self.state.performance_history.append(reward)
        if len(self.state.performance_history) >= 50:
            recent_perf = np.mean(self.state.performance_history[-20:])
            older_perf = np.mean(self.state.performance_history[-50:-30])
            self.state.adaptability_estimate = max(0.0, min(1.0, 0.5 + (recent_perf - older_perf)))
        
        self.state.confidence_history.append(self.state.confidence_estimate)
        self.state.meta_updates += 1


# ===========================================================================
# 第三部分：多主体环境
# ===========================================================================

@dataclass
class MultiAgentWorld:
    """
    多主体 GridWorld：
    - 多个主体共享同一环境
    - 主体可以观测到其他主体的位置
    - 资源竞争/协作
    """
    N: int = 15
    n_agents: int = 3
    view_radius: int = 2
    resource_density: float = 0.08
    obstacle_density: float = 0.1
    regen_rate: float = 0.01
    seed: Optional[int] = None
    
    def __init__(self, N=15, n_agents=3, view_radius=2, resource_density=0.08, 
                 obstacle_density=0.1, regen_rate=0.01, seed=None):
        self.N = N
        self.n_agents = n_agents
        self.view_radius = view_radius
        self.resource_density = resource_density
        self.obstacle_density = obstacle_density
        self.regen_rate = regen_rate
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.step_count = 0
        self._init_map()
    
    def _init_map(self):
        self.grid = np.zeros((self.N, self.N), dtype=int)
        for i in range(self.N):
            for j in range(self.N):
                r = random.random()
                if r < self.obstacle_density:
                    self.grid[i, j] = 2
                elif r < self.obstacle_density + self.resource_density:
                    self.grid[i, j] = 1
        # Clear starting positions
        self.agent_positions = []
        for idx in range(self.n_agents):
            pos = (idx * (self.N // self.n_agents), 0)
            self.grid[pos] = 0
            self.agent_positions.append(pos)
    
    def get_local_view(self, pos, radius=None):
        if radius is None:
            radius = self.view_radius
        r, c = pos
        view = []
        for dr in range(-radius, radius + 1):
            row = []
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.N and 0 <= nc < self.N:
                    row.append(self.grid[nr, nc])
                else:
                    row.append(-1)
            view.append(row)
        return np.array(view)
    
    def observe_for(self, agent_idx: int):
        """特定主体的观测（包含其他主体位置）"""
        my_pos = self.agent_positions[agent_idx]
        local = self.get_local_view(my_pos)
        
        # 其他主体的相对位置
        others = []
        for i, pos in enumerate(self.agent_positions):
            if i != agent_idx:
                rel_r = pos[0] - my_pos[0]
                rel_c = pos[1] - my_pos[1]
                others.append({'rel_pos': (rel_r, rel_c), 'idx': i})
        
        return {
            'local_view': local,
            'pos': my_pos,
            'others': others,
            'step': self.step_count
        }
    
    def step_actions(self, actions: List[Action]):
        """执行所有主体的动作，返回各自的 (obs, reward, done, info)"""
        results = []
        
        for idx, action in enumerate(actions):
            dr, dc = ACTION_DX[action]
            r, c = self.agent_positions[idx]
            nr, nc = r + dr, c + dc
            
            reward = -0.01
            valid = True
            
            if 0 <= nr < self.N and 0 <= nc < self.N and self.grid[nr, nc] != 2:
                self.agent_positions[idx] = (nr, nc)
            else:
                valid = False
                reward = -0.05
            
            # Collect resource (competitive: first come)
            pos = self.agent_positions[idx]
            if self.grid[pos] == 1:
                reward = 1
                self.grid[pos] = 0
            
            results.append((self.observe_for(idx), reward, False, {'valid': valid}))
        
        # Regeneration
        for i in range(self.N):
            for j in range(self.N):
                if self.grid[i, j] == 0 and random.random() < self.regen_rate:
                    self.grid[i, j] = 1
        
        self.step_count += 1
        return results


# ===========================================================================
# 第四部分：社会性度量
# ===========================================================================

def compute_intersubjectivity(agents: List[MetacognitiveAgent], 
                             observations: List[Dict],
                             actions: List[Action]) -> Dict:
    """
    计算交互主体性指标
    
    核心问题：主体之间是否形成了"共享理解"？
    """
    n = len(agents)
    
    # 1. 预测相似度：不同主体对同一情境的预测是否趋同？
    predictions = []
    for i, agent in enumerate(agents):
        pred = agent.predict(observations[i], actions[i])
        predictions.append(pred)
    
    pred_similarity = 0
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            sim = 1 - np.mean((predictions[i] - predictions[j]) ** 2)
            pred_similarity += sim
            count += 1
    pred_similarity /= max(1, count)
    
    # 2. 信心相关性：不同主体的信心是否相关？
    confidences = [agents[i].state.confidence_estimate for i in range(n)]
    conf_correlation = np.std(confidences)  # low std = high correlation
    
    # 3. 行为协调度：动作是否互补/协作？
    action_counts = {}
    for a in actions:
        action_counts[a] = action_counts.get(a, 0) + 1
    # Entropy of action distribution (high = diverse = less coordination)
    probs = np.array(list(action_counts.values())) / n
    action_entropy = -np.sum(probs * np.log(probs + 1e-8))
    
    # 4. 元认知相似度：不同主体的元认知策略是否趋同？
    meta_similarities = []
    for i in range(n):
        for j in range(i+1, n):
            lr_sim = 1 - abs(agents[i].state.learning_rate - agents[j].state.learning_rate)
            conf_sim = 1 - abs(agents[i].state.confidence_estimate - agents[j].state.confidence_estimate)
            meta_similarities.append((lr_sim + conf_sim) / 2)
    meta_similarity = np.mean(meta_similarities) if meta_similarities else 0
    
    return {
        'prediction_similarity': pred_similarity,
        'confidence_correlation': conf_correlation,
        'action_entropy': action_entropy,
        'meta_similarity': meta_similarity,
        'intersubjectivity_index': (pred_similarity + (1 - conf_correlation) + meta_similarity) / 3
    }


# ===========================================================================
# 第五部分：实验运行器
# ===========================================================================

def run_metacognition_experiment(n_episodes: int = 200, steps_per_episode: int = 100,
                                seed: int = 42, verbose: bool = True) -> Dict:
    """运行 L7 元认知实验"""
    env = ComplexGridWorld(N=20, seed=seed)
    agent = MetacognitiveAgent(view_size=49, seed=seed)
    
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
        episode_confidences.append(agent.state.confidence_estimate)
        if ep_errors:
            episode_errors.append(np.mean(ep_errors))
        
        if verbose and ep % 50 == 0:
            print(f"  L7 | Episode {ep} | Reward: {ep_reward:.1f} | "
                  f"Confidence: {agent.state.confidence_estimate:.3f} | "
                  f"LR: {agent.state.learning_rate:.4f}")
    
    # Compute metrics
    si = compute_subjectivity_index_from_agent(agent, env, n_trials=50)
    
    return {
        'level': 7,
        'level_name': 'L7',
        'total_reward': sum(episode_rewards),
        'mean_reward': np.mean(episode_rewards[-50:]),
        'subjectivity_index': si,
        'final_confidence': agent.state.confidence_estimate,
        'final_learning_rate': agent.state.learning_rate,
        'final_adaptability': agent.state.adaptability_estimate,
        'meta_updates': agent.state.meta_updates,
        'episode_rewards': episode_rewards,
        'episode_confidences': episode_confidences,
    }


def run_multiagent_experiment(n_episodes: int = 150, steps_per_episode: int = 80,
                             n_agents: int = 3, seed: int = 42, 
                             verbose: bool = True) -> Dict:
    """运行多主体社会性实验"""
    env = MultiAgentWorld(N=15, n_agents=n_agents, seed=seed)
    agents = [MetacognitiveAgent(view_size=25, seed=seed+i) for i in range(n_agents)]
    
    episode_rewards = []
    episode_intersubjectivity = []
    episode_coordination = []
    
    for ep in range(n_episodes):
        env._init_map()
        ep_reward = 0
        
        for step in range(steps_per_episode):
            observations = [env.observe_for(i) for i in range(n_agents)]
            actions = [agents[i].select_action(observations[i]) for i in range(n_agents)]
            
            results = env.step_actions(actions)
            
            for i in range(n_agents):
                obs, reward, done, info = results[i]
                agents[i].update(observations[i], actions[i], obs, reward)
                ep_reward += reward
            
            # Measure intersubjectivity
            inter = compute_intersubjectivity(agents, observations, actions)
            episode_intersubjectivity.append(inter['intersubjectivity_index'])
            episode_coordination.append(1 - inter['action_entropy'] / np.log(len(ACTIONS)))
        
        episode_rewards.append(ep_reward / n_agents)
        
        if verbose and ep % 50 == 0:
            avg_inter = np.mean(episode_intersubjectivity[-steps_per_episode:])
            print(f"  Multi | Episode {ep} | Avg Reward: {ep_reward/n_agents:.1f} | "
                  f"Intersubjectivity: {avg_inter:.4f}")
    
    # Final intersubjectivity
    final_obs = [env.observe_for(i) for i in range(n_agents)]
    final_actions = [agents[i].select_action(final_obs[i]) for i in range(n_agents)]
    final_inter = compute_intersubjectivity(agents, final_obs, final_actions)
    
    return {
        'n_agents': n_agents,
        'total_reward': sum(episode_rewards),
        'mean_reward': np.mean(episode_rewards[-50:]),
        'final_intersubjectivity': final_inter['intersubjectivity_index'],
        'final_prediction_similarity': final_inter['prediction_similarity'],
        'final_meta_similarity': final_inter['meta_similarity'],
        'episode_rewards': episode_rewards,
        'episode_intersubjectivity': episode_intersubjectivity,
    }


def compute_subjectivity_index_from_agent(agent: MetacognitiveAgent, 
                                         env: ComplexGridWorld,
                                         n_trials: int = 50) -> float:
    """计算主体性指数"""
    perf_complete = _test_metacognitive_agent(agent, env, n_trials)
    
    # Lesioned version: remove metacognition
    agent_lesioned = MetacognitiveAgent(view_size=agent.view_size, seed=999)
    if agent_lesioned.state.weights is not None and agent.state.weights is not None:
        agent_lesioned.state.weights = agent.state.weights.copy()
        agent_lesioned.state.bias = agent.state.bias.copy()
        agent_lesioned.own_causal_model = {k: {'effect': np.array(v['effect']), 'count': v['count']} 
                                          for k, v in agent.own_causal_model.items()}
        # Lesion: remove metacognitive modulation
        agent_lesioned.state.confidence_estimate = 0.5  # fixed, no adaptation
        agent_lesioned.state.meta_learning_rate = 0.0  # no meta-learning
    
    perf_lesion = _test_metacognitive_agent(agent_lesioned, env, n_trials)
    
    if perf_complete + perf_lesion == 0:
        return 0.0
    
    si = 1.0 - (perf_lesion / (perf_complete + 1e-8))
    return max(0.0, min(1.0, si))


def _test_metacognitive_agent(agent: MetacognitiveAgent, env: ComplexGridWorld, 
                              n_trials: int) -> float:
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


# ===========================================================================
# 主程序
# ===========================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("  L7 元认知 + 多主体社会性实验")
    print("=" * 70)
    
    # Experiment 1: L7 Metacognition
    print("\n" + "=" * 70)
    print("Experiment 1: L7 Metacognitive Agent")
    print("=" * 70)
    
    result_l7 = run_metacognition_experiment(
        n_episodes=200,
        steps_per_episode=100,
        seed=42,
        verbose=True
    )
    
    print(f"\nL7 Results:")
    print(f"  Mean Reward: {result_l7['mean_reward']:.2f}")
    print(f"  Subjectivity Index: {result_l7['subjectivity_index']:.4f}")
    print(f"  Final Confidence: {result_l7['final_confidence']:.4f}")
    print(f"  Final Learning Rate: {result_l7['final_learning_rate']:.4f}")
    print(f"  Final Adaptability: {result_l7['final_adaptability']:.4f}")
    print(f"  Meta Updates: {result_l7['meta_updates']}")
    
    # Experiment 2: Multi-Agent
    print("\n" + "=" * 70)
    print("Experiment 2: Multi-Agent Sociality")
    print("=" * 70)
    
    result_multi = run_multiagent_experiment(
        n_episodes=150,
        steps_per_episode=80,
        n_agents=3,
        seed=42,
        verbose=True
    )
    
    print(f"\nMulti-Agent Results:")
    print(f"  Mean Reward (per agent): {result_multi['mean_reward']:.2f}")
    print(f"  Final Intersubjectivity: {result_multi['final_intersubjectivity']:.4f}")
    print(f"  Prediction Similarity: {result_multi['final_prediction_similarity']:.4f}")
    print(f"  Meta Similarity: {result_multi['final_meta_similarity']:.4f}")
    
    # Save results
    output = {
        'L7': {k: v for k, v in result_l7.items() if k not in ['episode_rewards', 'episode_confidences']},
        'multiagent': {k: v for k, v in result_multi.items() if k not in ['episode_rewards', 'episode_intersubjectivity', 'episode_coordination']},
    }
    
    with open('D:/HermesOutput/minimal_agency/l7_multiagent_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 70)
    print("  实验完成。输出保存在 D:/HermesOutput/minimal_agency/")
    print("=" * 70)
