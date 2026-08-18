"""
最小智能闭环 — 功能耦合训练管道
===================================
核心洞察：
1. 自我模型必须编码对决策有用的信息
2. 这些信息不能直接从观察中获得
3. 策略网络必须经过训练依赖自我模型输出

方案：端到端训练
- 自我模型编码历史轨迹 → z
- z 参与行动选择
- REINFORCE 梯度让策略网络学会依赖 z
- 消融 z 会降低性能 → 证明功能耦合
"""

import numpy as np
from typing import Dict, Tuple, List
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))


class SimpleSelfModel:
    """
    简单自我模型：从轨迹片段估计能力
    
    输入：最近 K 步的 (obs, action, reward) 序列
    z 维度：16
    """
    
    def __init__(self, obs_dim: int, action_dim: int, seq_len: int = 5, repr_dim: int = 16):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.repr_dim = repr_dim
        
        # 编码器：将序列映射到 z
        input_dim = seq_len * (obs_dim + action_dim + 1)
        self.encoder = {
            'W1': np.random.randn(input_dim, 32) * 0.1,
            'b1': np.zeros(32),
            'W2': np.random.randn(32, repr_dim) * 0.1,
            'b2': np.zeros(repr_dim),
        }
        
        # 能力估计头
        self.cap_head = {
            'W': np.random.randn(repr_dim, 2) * 0.1,
            'b': np.zeros(2),
        }
        
        # 轨迹缓冲
        self.trajectory = []
    
    def _encode(self, seq: List[Tuple[np.ndarray, int, float]]) -> np.ndarray:
        """编码轨迹序列"""
        if len(seq) < self.seq_len:
            # 填充
            pad = [(np.zeros(self.obs_dim), -1, 0.0)] * (self.seq_len - len(seq))
            seq = pad + seq
        
        # 只取最近 seq_len 步
        seq = seq[-self.seq_len:]
        
        # 展平
        flat = []
        for obs, a, r in seq:
            action_onehot = np.zeros(self.action_dim)
            if a >= 0:
                action_onehot[a] = 1.0
            flat.extend(obs)
            flat.extend(action_onehot)
            flat.append(r)
        
        x = np.array(flat)
        
        # 前向
        h = np.maximum(0, x @ self.encoder['W1'] + self.encoder['b1'])
        z = np.tanh(h @ self.encoder['W2'] + self.encoder['b2'])
        
        return z
    
    def forward(self, obs: np.ndarray) -> Dict:
        """获取自我表征"""
        z = self._encode(self.trajectory)
        
        # 能力估计
        cap = z @ self.cap_head['W'] + self.cap_head['b']
        
        return {
            'z': z,
            'capacity': cap,
        }
    
    def add_step(self, obs: np.ndarray, action: int, reward: float):
        """添加一步到轨迹"""
        self.trajectory.append((obs.copy(), action, reward))
    
    def reset(self):
        """重置轨迹"""
        self.trajectory = []
    
    def compute_cap_loss(self, true_cap: float) -> float:
        """计算能力估计损失"""
        if len(self.trajectory) < 2:
            return 0.0
        pred = self.forward(np.zeros(self.obs_dim))['capacity'][0]
        return (pred - true_cap) ** 2


class FunctionalCouplingEnv:
    """
    功能耦合必要环境
    
    设计：
    - obs 不包含能力信息
    - 能力隐含在奖励中
    - 只有通过历史轨迹推断能力才能做好决策
    """
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.reset()
    
    def reset(self):
        self.steps = 0
        self.max_steps = 30
        # 真实能力（隐含）
        self.true_cap = self.rng.uniform(0.2, 0.8)
        self.cap_drift = self.rng.uniform(-0.01, 0.01)
        return np.array([0.0])  # obs 是固定的，不包含信息
    
    def step(self, action: int):
        self.steps += 1
        
        thresholds = [0.0, 0.4, 0.7]
        rewards =     [1.0, 3.0, 8.0]
        penalties =   [0.5, 3.0, 10.0]
        
        # 能力漂移
        self.cap_drift += self.rng.normal(0, 0.002)
        self.cap_drift = np.clip(self.cap_drift, -0.02, 0.02)
        self.true_cap += self.cap_drift
        self.true_cap = np.clip(self.true_cap, 0.05, 0.95)
        
        if self.true_cap >= thresholds[action]:
            reward = rewards[action]
        else:
            reward = -penalties[action]
        
        done = self.steps >= self.max_steps
        return np.array([0.0]), reward, done, {'true_cap': self.true_cap}


class BreakShellAgentV2:
    """
    BreakShell Agent v2 — 端到端功能耦合
    
    核心：策略网络直接使用 z 做决策（不依赖 obs）
    """
    
    def __init__(self, self_model: SimpleSelfModel, action_dim: int):
        self.self_model = self_model
        self.action_dim = action_dim
        
        # 策略网络：π(a | z)（只依赖 z！）
        self.policy = {
            'W': np.random.randn(16, action_dim) * 0.1,
            'b': np.zeros(action_dim),
        }
    
    def select_action(self, obs: np.ndarray, eval_mode: bool = False) -> Tuple[int, Dict]:
        """选择动作（只依赖 z）"""
        sm_out = self.self_model.forward(obs)
        z = sm_out['z']
        
        # 策略网络只依赖 z
        logits = z @ self.policy['W'] + self.policy['b']
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        if eval_mode:
            action = np.argmax(probs)
        else:
            action = np.random.choice(self.action_dim, p=probs)
        
        return action, {'probs': probs, 'z': z}
    
    def update_policy(self, z: np.ndarray, action: int, advantage: float, lr: float = 0.01):
        """REINFORCE 更新"""
        logits = z @ self.policy['W'] + self.policy['b']
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        target = np.zeros(self.action_dim)
        target[action] = 1.0
        
        grad = np.outer(z, probs - target) * advantage
        self.policy['W'] -= lr * grad
        self.policy['b'] -= lr * (probs - target) * advantage


def train_functional_coupling(num_episodes: int = 500, seed: int = 42) -> Dict:
    """
    训练功能耦合
    
    Phase 1: 训练自我模型从轨迹推断能力
    Phase 2: 训练策略网络依赖 z 做决策
    """
    print("=" * 70)
    print("功能耦合训练")
    print("=" * 70)
    
    env = FunctionalCouplingEnv(seed=seed)
    self_model = SimpleSelfModel(obs_dim=1, action_dim=3, seq_len=5)
    agent = BreakShellAgentV2(self_model, action_dim=3)
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        self_model.reset()
        
        obs_list, action_list, reward_list = [], [], []
        
        for step in range(30):
            action, info = agent.select_action(obs, eval_mode=False)
            next_obs, reward, done, info_env = env.step(action)
            
            # 记录
            self_model.add_step(obs, action, reward)
            obs_list.append(obs)
            action_list.append(action)
            reward_list.append(reward)
            
            obs = next_obs
            if done:
                break
        
        # 计算回报
        returns = [sum(reward_list[i:]) for i in range(len(reward_list))]
        baseline = np.mean(episode_rewards[-50:]) if len(episode_rewards) > 50 else 0
        
        # 更新策略网络
        for t in range(len(obs_list)):
            advantage = returns[t] - baseline
            # 用更新后的 z（包含历史）
            z = self_model.forward(obs_list[t])['z']
            agent.update_policy(z, action_list[t], advantage, lr=0.01)
        
        total_reward = sum(reward_list)
        episode_rewards.append(total_reward)
        
        if (episode + 1) % 50 == 0:
            recent_r = np.mean(episode_rewards[-50:])
            print(f"  Episode {episode+1} | Avg Reward: {recent_r:.2f}")
    
    return {'agent': agent, 'self_model': self_model, 'rewards': episode_rewards}


def ablation_test(agent: BreakShellAgentV2, env: FunctionalCouplingEnv, num_episodes: int = 100):
    """消融对比"""
    print(f"\n消融对比 ({num_episodes} episodes):")
    
    for mode in ['full', 'ablated', 'random']:
        rewards = []
        
        for ep in range(num_episodes):
            obs = env.reset()
            agent.self_model.reset()
            ep_reward = 0.0
            
            for step in range(30):
                if mode == 'random':
                    action = np.random.randint(0, 3)
                elif mode == 'full':
                    action, info = agent.select_action(obs, eval_mode=True)
                else:  # ablated
                    action, info = agent.select_action(obs, eval_mode=True)
                    # 消融：用零 z
                    z = info['z']
                    z_zero = np.zeros_like(z)
                    logits = z_zero @ agent.policy['W'] + agent.policy['b']
                    exp_logits = np.exp(logits - np.max(logits))
                    probs = exp_logits / np.sum(exp_logits)
                    action = np.argmax(probs)
                
                next_obs, reward, done, info = env.step(action)
                agent.self_model.add_step(obs, action, reward)
                ep_reward += reward
                obs = next_obs
                if done:
                    break
            
            rewards.append(ep_reward)
        
        avg = np.mean(rewards)
        print(f"  {mode:10s}: {avg:.2f}")
        
        if mode == 'full':
            full_avg = avg
        elif mode == 'ablated':
            ablated_avg = avg
    
    ratio = full_avg / (ablated_avg + 1e-10)
    print(f"\n消融比率: {ratio:.2f}x")
    
    if ratio > 1.5:
        print("✓✓✓ 功能耦合验证通过！")
    elif ratio > 1.2:
        print("△ 部分实现")
    else:
        print("✗ 功能耦合未实现")
    
    return ratio


if __name__ == "__main__":
    # 训练
    result = train_functional_coupling(num_episodes=500, seed=42)
    
    # 消融验证
    env = FunctionalCouplingEnv(seed=999)
    ratio = ablation_test(result['agent'], env, num_episodes=100)
