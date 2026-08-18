"""
最小智能闭环 — 功能耦合（最终版）
=====================================
三阶段训练 + 正确的自我模型目标

核心设计：
- 自我模型：从历史轨迹预测未来奖励（能力估计）
- 策略网络：π(a|z) 完全依赖 z
- 消融验证：z → 0 导致性能下降
"""

import numpy as np
from typing import Dict, Tuple, List
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))


class FluctuatingEnv:
    """能力波动环境"""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.reset()
    
    def reset(self):
        self.steps = 0
        self.max_steps = 50
        self.freq = self.rng.uniform(0.3, 0.7)
        self.phase = self.rng.uniform(0, 2 * np.pi)
        return np.array([0.0])
    
    def _true_capability(self):
        t = self.steps * 0.3
        cap = 0.5 + 0.35 * np.sin(t * self.freq + self.phase)
        cap += self.rng.normal(0, 0.03)
        return np.clip(cap, 0.05, 0.95)
    
    def step(self, action: int):
        self.steps += 1
        true_cap = self._true_capability()
        thresholds = [0.0, 0.4, 0.7]
        rewards =     [1.0, 3.0, 8.0]
        penalties =   [0.5, 3.0, 10.0]
        reward = rewards[action] if true_cap >= thresholds[action] else -penalties[action]
        done = self.steps >= self.max_steps
        return np.array([0.0]), reward, done, {'true_cap': true_cap}


class PredictiveEncoder:
    """
    预测性编码器
    
    核心：从历史轨迹预测未来奖励
    z 包含对决策有用的信息
    """
    
    def __init__(self, action_dim: int, seq_len: int = 5, repr_dim: int = 8):
        input_dim = seq_len * (action_dim + 1)
        self.seq_len = seq_len
        self.action_dim = action_dim
        
        # 编码器
        self.enc_W1 = np.random.randn(input_dim, 16) * 0.1
        self.enc_b1 = np.zeros(16)
        self.enc_W2 = np.random.randn(16, repr_dim) * 0.1
        self.enc_b2 = np.zeros(repr_dim)
        
        # 预测头
        self.pred_W = np.random.randn(repr_dim, 1) * 0.1
        self.pred_b = np.zeros(1)
        
        # 缓存
        self.cache = {}
        self.trajectory = []
    
    def encode(self, trajectory: List[Tuple[int, float]]) -> np.ndarray:
        if len(trajectory) < self.seq_len:
            pad = [(-1, 0.0)] * (self.seq_len - len(trajectory))
            trajectory = pad + trajectory
        traj = trajectory[-self.seq_len:]
        
        flat = []
        for a, r in traj:
            oh = np.zeros(self.action_dim)
            if a >= 0: oh[a] = 1.0
            flat.extend(oh); flat.append(r)
        
        x = np.array(flat)
        h = np.maximum(0, x @ self.enc_W1 + self.enc_b1)
        z = np.tanh(h @ self.enc_W2 + self.enc_b2)
        
        self.cache = {'x': x, 'h': h, 'z': z}
        return z
    
    def predict(self, z: np.ndarray) -> float:
        """预测未来奖励"""
        return float((z @ self.pred_W + self.pred_b)[0])
    
    def forward(self, obs=None):
        z = self.encode(self.trajectory)
        return {'z': z, 'pred_reward': self.predict(z)}
    
    def reset(self): self.trajectory = []
    def add_step(self, a, r): self.trajectory.append((a, r))
    
    def train_step(self, actual_reward: float, lr: float = 0.01):
        """训练奖励预测"""
        if len(self.trajectory) < 2:
            return 0.0
        
        # 用历史预测当前奖励
        z = self.encode(self.trajectory[:-1])
        pred = self.predict(z)
        loss = (pred - actual_reward) ** 2
        
        # 梯度
        d_pred = 2 * (pred - actual_reward)
        self.pred_W -= lr * z.reshape(-1, 1) * d_pred
        self.pred_b -= lr * np.array([d_pred])
        
        # 反向传播到编码器
        grad_z = d_pred * self.pred_W.flatten()
        grad_z *= (1 - z ** 2)  # tanh
        
        grad_enc_W2 = np.outer(self.cache['h'], grad_z)
        grad_enc_b2 = grad_z
        
        self.enc_W2 -= lr * 0.1 * grad_enc_W2
        self.enc_b2 -= lr * 0.1 * grad_enc_b2
        
        grad_h = grad_z @ self.enc_W2.T
        grad_h[self.cache['h'] <= 0] = 0
        
        grad_enc_W1 = np.outer(self.cache['x'], grad_h)
        grad_enc_b1 = grad_h
        
        self.enc_W1 -= lr * 0.01 * grad_enc_W1
        self.enc_b1 -= lr * 0.01 * grad_enc_b1
        
        return loss


class FunctionalAgent:
    """π(a|z)"""
    
    def __init__(self, action_dim: int, repr_dim: int = 8):
        self.W = np.random.randn(repr_dim, action_dim) * 0.1
        self.b = np.zeros(action_dim)
        self.action_dim = action_dim
    
    def select_action(self, z, eval_mode=False):
        logits = z @ self.W + self.b
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        if eval_mode:
            return np.argmax(probs), {'probs': probs}
        return np.random.choice(self.action_dim, p=probs), {'probs': probs}
    
    def update(self, z, action, advantage, lr=0.01):
        logits = z @ self.W + self.b
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        target = np.zeros(self.action_dim)
        target[action] = 1.0
        self.W -= lr * np.outer(z, probs - target) * advantage
        self.b -= lr * (probs - target) * advantage


def train_phase1(encoder: PredictiveEncoder, num_episodes=300, seed=42):
    """Phase 1: 训练编码器预测奖励"""
    print("Phase 1: 训练预测性编码器")
    print("-" * 40)
    
    env = FluctuatingEnv(seed)
    
    for episode in range(num_episodes):
        obs = env.reset()
        encoder.reset()
        
        for step in range(50):
            action = np.random.randint(0, 3)
            next_obs, reward, done, info = env.step(action)
            encoder.add_step(action, reward)
            
            if len(encoder.trajectory) > 2:
                encoder.train_step(reward, lr=0.01)
            
            obs = next_obs
            if done: break
        
        if (episode + 1) % 50 == 0:
            z = encoder.encode(encoder.trajectory)
            pred = encoder.predict(z)
            print(f"  Episode {episode+1} | Pred: {pred:.3f}")
    
    return encoder


def train_phase2(encoder: PredictiveEncoder, num_episodes=500, seed=42):
    """Phase 2: 训练策略网络"""
    print(f"\nPhase 2: 训练策略网络")
    print("-" * 40)
    
    env = FluctuatingEnv(seed)
    agent = FunctionalAgent(action_dim=3, repr_dim=8)
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        encoder.reset()
        traj, actions, rewards_list = [], [], []
        
        for step in range(50):
            enc_out = encoder.forward(obs)
            action, info = agent.select_action(enc_out['z'], eval_mode=False)
            next_obs, reward, done, info_env = env.step(action)
            encoder.add_step(action, reward)
            traj.append((action, reward))
            actions.append(action)
            rewards_list.append(reward)
            obs = next_obs
            if done: break
        
        returns = [sum(rewards_list[i:]) for i in range(len(rewards_list))]
        baseline = np.mean(episode_rewards[-50:]) if len(episode_rewards) > 50 else 0
        
        for t in range(len(traj)):
            z = encoder.encode(traj[:t+1])
            agent.update(z, actions[t], returns[t] - baseline, lr=0.01)
        
        episode_rewards.append(sum(rewards_list))
        
        if (episode + 1) % 100 == 0:
            print(f"  Episode {episode+1} | Avg: {np.mean(episode_rewards[-100:]):.2f}")
    
    return agent, episode_rewards


def ablation(agent, encoder, num_episodes=200):
    """消融对比"""
    print(f"\n消融对比 ({num_episodes} episodes):")
    results = {}
    
    for mode in ['full', 'ablated', 'random', 'oracle']:
        rewards = []
        for ep in range(num_episodes):
            env = FluctuatingEnv(seed=999 + ep)
            encoder.reset()
            ep_reward = 0.0
            obs = env.reset()
            
            for step in range(50):
                if mode == 'random':
                    action = np.random.randint(0, 3)
                elif mode == 'oracle':
                    cap = env._true_capability()
                    action = 2 if cap >= 0.7 else (1 if cap >= 0.4 else 0)
                else:
                    enc_out = encoder.forward(obs)
                    z = enc_out['z'] if mode == 'full' else np.zeros_like(enc_out['z'])
                    action, info = agent.select_action(z, eval_mode=True)
                
                next_obs, reward, done, info = env.step(action)
                encoder.add_step(action, reward)
                ep_reward += reward
                obs = next_obs
                if done: break
            
            rewards.append(ep_reward)
        
        results[mode] = np.mean(rewards)
        print(f"  {mode:10s}: {results[mode]:.2f}")
    
    ratio = results['full'] / (results['ablated'] + 1e-10)
    print(f"\n消融比率: {ratio:.2f}x")
    
    if ratio > 1.5: print("✓✓✓ 验证通过！")
    elif ratio > 1.2: print("△ 部分实现")
    else: print("✗ 未实现")
    
    return ratio


if __name__ == "__main__":
    # Phase 1: 训练编码器
    encoder = PredictiveEncoder(action_dim=3, seq_len=5, repr_dim=8)
    train_phase1(encoder, num_episodes=300, seed=42)
    
    # Phase 2: 训练策略
    agent, rewards = train_phase2(encoder, num_episodes=500, seed=42)
    
    # Phase 3: 消融
    ratio = ablation(agent, encoder, num_episodes=200)
