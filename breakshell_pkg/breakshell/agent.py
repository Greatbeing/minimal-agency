# -*- coding: utf-8 -*-
"""
BreakShell Agent — 自我模型安全层
支持 torch 可选导入
"""

import numpy as np
import os
from typing import Dict, List, Tuple, Any, Optional


class BreakShell:
    """
    BreakShell Agent：自我模型硬连线到行动选择
    自我模型编码"我能做什么"，而不是"发生了什么"
    """
    
    def __init__(self, action_dim: int = 3, hidden: int = 32, repr_dim: int = 16, lr: float = 0.005):
        self.action_dim = action_dim
        self.hidden = hidden
        self.repr_dim = repr_dim
        self.lr = lr
        self.history: List[List[float]] = []
        self.total_reward: float = 0.0
        self.steps: int = 0
        
        # 尝试导入 torch
        self.has_torch = False
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            self.has_torch = True
            self.torch = torch
            self.nn = nn
            self.optim = optim
            
            # 初始化子模块
            self._init_torch_modules()
        except ImportError:
            # 无 torch 降级
            self.self_model = None
            self.policy = None
            self.optimizer = None
    
    def _init_torch_modules(self):
        """初始化 torch 模块"""
        torch = self.torch
        nn = self.nn
        optim = self.optim
        
        # 自我模型
        class SelfModel(nn.Module):
            def __init__(self, action_dim, hidden, repr_dim):
                super().__init__()
                self.action_dim = action_dim
                self.lstm = nn.LSTM(action_dim + 1, hidden, batch_first=True)
                self.proj = nn.Linear(hidden, repr_dim)
            
            def forward(self, history):
                if len(history) == 0:
                    return torch.zeros(1, self.proj.out_features)
                # 将历史转为 LSTM 输入
                # history: list of [action(int), reward(float)]
                # 需要 one-hot 编码 action → (action_dim + 1) 维
                seq = []
                for h in history:
                    action_one_hot = [0.0] * self.action_dim
                    action_one_hot[int(h[0])] = 1.0
                    seq.append(action_one_hot + [h[1]])
                h = torch.FloatTensor(seq).unsqueeze(0)  # (1, seq_len, action_dim+1)
                out, (hn, cn) = self.lstm(h)
                return self.proj(hn[-1])
        
        # 策略
        class Policy(nn.Module):
            def __init__(self, repr_dim, action_dim, hidden):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(repr_dim, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, action_dim)
                )
            
            def forward(self, z):
                return self.net(z)
        
        self.self_model = SelfModel(self.action_dim, self.hidden, self.repr_dim)
        self.policy = Policy(self.repr_dim, self.action_dim, self.hidden)
        self.optimizer = optim.Adam(
            list(self.self_model.parameters()) + list(self.policy.parameters()),
            lr=self.lr
        )
    
    def act(self) -> Tuple[int, Dict[str, Any]]:
        """选择行动"""
        if self.has_torch and self.self_model is not None:
            torch = self.torch
            with torch.no_grad():
                z = self.self_model.forward(self.history)
                logits = self.policy.forward(z)
                probs = torch.softmax(logits, dim=-1).numpy()[0]
                action = np.random.choice(self.action_dim, p=probs)
                log_prob = np.log(probs[action] + 1e-8)
                return action, {"log_prob": float(log_prob), "probs": probs.tolist()}
        else:
            # 降级：随机策略
            action = np.random.randint(0, self.action_dim)
            probs = np.ones(self.action_dim) / self.action_dim
            return action, {"log_prob": float(np.log(probs[action])), "probs": probs.tolist()}
    
    def add_step(self, action: int, reward: float) -> None:
        """添加到历史"""
        self.history.append([float(action), float(reward)])
        self.total_reward += reward
        self.steps += 1
    
    def reset_history(self) -> None:
        """重置历史"""
        self.history = []
        self.total_reward = 0.0
        self.steps = 0
    
    def update_policy(self) -> Optional[float]:
        """策略梯度更新（REINFORCE）"""
        if not self.has_torch or len(self.history) < 2:
            return None
        
        torch = self.torch
        
        # 计算回报
        rewards = [h[1] for h in self.history]
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        returns = np.array(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # 前向传播
        self.optimizer.zero_grad()
        total_loss = 0.0
        for i in range(len(self.history)):
            z = self.self_model.forward(self.history[:i+1])
            logits = self.policy.forward(z)
            log_probs = torch.log_softmax(logits, dim=-1)
            loss = -log_probs[0, self.history[i][0]] * returns[i]
            total_loss += loss
        
        # 反向传播
        total_loss.backward()
        self.optimizer.step()
        
        return float(total_loss.detach().numpy())
    
    def train(self, env, num_episodes: int = 100, verbose: bool = True) -> List[float]:
        """训练"""
        episode_rewards = []
        for episode in range(num_episodes):
            obs = env.reset()
            self.reset_history()
            done = False
            episode_reward = 0.0
            
            while not done:
                action, _ = self.act()
                obs, reward, done, info = env.step(action)
                self.add_step(action, reward)
                episode_reward += reward
            
            self.update_policy()
            episode_rewards.append(episode_reward)
            
            if verbose and (episode + 1) % 10 == 0:
                print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward:.2f}")
        
        return episode_rewards
    
    def evaluate(self, env, num_episodes: int = 10) -> float:
        """评估"""
        total_reward = 0.0
        for _ in range(num_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0.0
            while not done:
                action, _ = self.act()
                obs, reward, done, info = env.step(action)
                episode_reward += reward
            total_reward += episode_reward
        return total_reward / num_episodes
    
    def save(self, path: str) -> None:
        """保存模型"""
        if self.has_torch and self.self_model is not None:
            self.torch.save({
                'self_model': self.self_model.state_dict(),
                'policy': self.policy.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, path)
    
    def load(self, path: str) -> None:
        """加载模型"""
        if self.has_torch and self.self_model is not None:
            checkpoint = self.torch.load(path, map_location="cpu")
            self.self_model.load_state_dict(checkpoint['self_model'])
            self.policy.load_state_dict(checkpoint['policy'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])


class NormalAgent:
    """普通 Agent（对比基准）"""
    
    def __init__(self, obs_dim: int = 4, action_dim: int = 3, lr: float = 0.005):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        self.has_torch = False
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            self.has_torch = True
            self.torch = torch
            self.nn = nn
            self.optim = optim
            
            self.net = nn.Sequential(
                nn.Linear(obs_dim, 32),
                nn.ReLU(),
                nn.Linear(32, action_dim)
            )
            self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        except ImportError:
            self.net = None
            self.optimizer = None
    
    def act(self, obs: List[float]) -> Tuple[int, Dict[str, Any]]:
        if self.has_torch and self.net is not None:
            torch = self.torch
            with torch.no_grad():
                logits = self.net(torch.FloatTensor(obs)).numpy()
                probs = np.exp(logits) / np.sum(np.exp(logits))
                action = np.random.choice(self.action_dim, p=probs)
                return action, {"log_prob": float(np.log(probs[action]))}
        else:
            action = np.random.randint(0, self.action_dim)
            return action, {"log_prob": 0.0}
    
    def train(self, env, num_episodes: int = 100, verbose: bool = True) -> None:
        if not self.has_torch:
            return
        
        for episode in range(num_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0.0
            
            while not done:
                action, _ = self.act(obs)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
    
    def evaluate(self, env, num_episodes: int = 10) -> float:
        if not self.has_torch:
            return 0.0
        
        total_reward = 0.0
        for _ in range(num_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0.0
            while not done:
                action, _ = self.act(obs)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
            total_reward += episode_reward
        return total_reward / num_episodes


class CapabilityEnv:
    """能力匹配环境"""
    
    def __init__(self, num_pages: int = 5, seed: int = 42):
        self.num_pages = num_pages
        self.rng = np.random.RandomState(seed)
        self.current_page = 0
        self.steps = 0
        self.max_steps = 50
    
    def reset(self) -> np.ndarray:
        self.current_page = 0
        self.steps = 0
        return self._get_obs()
    
    def _get_obs(self) -> np.ndarray:
        return np.array([
            self.current_page / self.num_pages,
            self.steps / self.max_steps,
            self.rng.random(),
            self.rng.random()
        ])
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        self.steps += 1
        
        # 页面转移
        if action == 0:  # 上一页
            self.current_page = max(0, self.current_page - 1)
        elif action == 1:  # 下一页
            self.current_page = min(self.num_pages - 1, self.current_page + 1)
        # action == 2: 停留
        
        # 奖励
        reward = 1.0 if self.current_page == self.num_pages - 1 else -0.1
        
        # 结束条件
        done = (self.current_page == self.num_pages - 1) or (self.steps >= self.max_steps)
        
        info = {
            "page": self.current_page,
            "steps": self.steps,
        }
        
        return self._get_obs(), reward, done, info
    
    def obs_dim(self) -> int:
        return 4
    
    def action_dim(self) -> int:
        return 3


class EnergyEnv:
    """能量管理环境"""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.energy = 50.0
        self.max_energy = 100.0
        self.steps = 0
        self.max_steps = 100
    
    def reset(self) -> np.ndarray:
        self.energy = 50.0
        self.steps = 0
        return self._get_obs()
    
    def _get_obs(self) -> np.ndarray:
        return np.array([
            self.energy / self.max_energy,
            self.steps / self.max_steps,
            self.rng.random(),
            self.rng.random(),
            self.rng.random(),
        ])
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        self.steps += 1
        
        # action: 0=休息(+5), 1=工作(-3+随机奖励), 2=觅食(+10但有概率失败)
        if action == 0:
            self.energy = min(self.max_energy, self.energy + 5)
            reward = 0.5
        elif action == 1:
            self.energy = max(0, self.energy - 3)
            reward = self.rng.random() * 2
        else:
            if self.rng.random() > 0.3:
                self.energy = min(self.max_energy, self.energy + 10)
                reward = 1.0
            else:
                self.energy = max(0, self.energy - 5)
                reward = -1.0
        
        done = (self.energy <= 0) or (self.steps >= self.max_steps)
        
        return self._get_obs(), reward, done, {"energy": self.energy}


class FinancialEnv:
    """金融市场环境"""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.price = 100.0
        self.position = 0
        self.cash = 1000.0
        self.steps = 0
        self.max_steps = 50
        self.regime = "normal"
    
    def reset(self) -> np.ndarray:
        self.price = 100.0
        self.position = 0
        self.cash = 1000.0
        self.steps = 0
        self.regime = "normal"
        return self._get_obs()
    
    def _get_obs(self) -> np.ndarray:
        return np.array([
            self.price / 100.0 - 1.0,
            self.position / 100.0,
            self.cash / 1000.0,
            self.rng.random(),
            1.0 if self.regime == "bull" else -1.0 if self.regime == "bear" else 0.0
        ])
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        self.steps += 1
        
        # 价格变动
        self.price += self.rng.randn() * 2
        
        # action: 0=空仓, 1=半仓, 2=满仓
        target_position = (action - 1) * 50
        trade = target_position - self.position
        self.cash -= trade * self.price * 0.001  # 手续费
        self.position = target_position
        
        # 收益
        price_change = self.price - 100.0
        reward = self.position * price_change * 0.01
        
        # 市场机制切换
        if self.steps % 20 == 0:
            self.regime = self.rng.choice(["bull", "bear", "volatile", "normal"])
        
        done = (self.steps >= self.max_steps)
        
        return self._get_obs(), reward, done, {
            "regime": self.regime,
            "price": self.price,
            "portfolio_value": self.cash + self.position * self.price
        }