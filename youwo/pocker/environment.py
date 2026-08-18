"""
Pocker Agent — 环境模块
====================
包含：
1. GridWorld 静态环境
2. NonStationaryGridWorld 非平稳环境
3. ProceduralLabyrinth 复杂迷宫环境
"""

import numpy as np
from typing import Tuple, Dict, List, Optional


class GridWorld:
    """简单静态 GridWorld"""
    
    def __init__(self, size: int = 8, seed: int = 42):
        self.size = size
        self.rng = np.random.RandomState(seed)
        self.reset()
    
    def reset(self) -> np.ndarray:
        """重置环境，返回初始观察"""
        # 随机放置智能体
        self.agent_pos = self.rng.randint(0, self.size, size=2)
        
        # 随机放置目标
        self.target_pos = self.rng.randint(0, self.size, size=2)
        while np.array_equal(self.target_pos, self.agent_pos):
            self.target_pos = self.rng.randint(0, self.size, size=2)
        
        # 随机放置障碍物
        self.obstacles = []
        num_obstacles = self.size
        for _ in range(num_obstacles):
            obs = self.rng.randint(0, self.size, size=2)
            if not np.array_equal(obs, self.agent_pos) and not np.array_equal(obs, self.target_pos):
                self.obstacles.append(obs)
        
        self.steps = 0
        self.max_steps = self.size * 4
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """获取当前观察 (one-hot 编码的位置 + 目标方向)"""
        # 智能体位置 one-hot
        agent_onehot = np.zeros(self.size * 2)
        agent_onehot[self.agent_pos[0]] = 1.0
        agent_onehot[self.size + self.agent_pos[1]] = 1.0
        
        # 目标方向 (相对位置)
        direction = (self.target_pos - self.agent_pos) / self.size
        
        # 障碍物密度 (4个方向)
        obstacle_density = np.zeros(4)
        for dx, dy, idx in [(0,1,0), (0,-1,1), (1,0,2), (-1,0,3)]:
            check_pos = self.agent_pos + np.array([dx, dy])
            if any(np.array_equal(check_pos, obs) for obs in self.obstacles):
                obstacle_density[idx] = 1.0
        
        return np.concatenate([agent_onehot, direction, obstacle_density])
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行动作
        
        Args:
            action: 0=上, 1=下, 2=左, 3=右
        
        Returns:
            observation, reward, done, info
        """
        self.steps += 1
        
        # 移动
        moves = [np.array([-1, 0]), np.array([1, 0]), np.array([0, -1]), np.array([0, 1])]
        new_pos = self.agent_pos + moves[action]
        
        # 检查边界和障碍物
        if (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size and
            not any(np.array_equal(new_pos, obs) for obs in self.obstacles)):
            self.agent_pos = new_pos
        
        # 计算奖励
        dist = np.linalg.norm(self.agent_pos - self.target_pos)
        reward = -0.1  # 每步惩罚
        
        done = False
        if np.array_equal(self.agent_pos, self.target_pos):
            reward = 10.0  # 到达目标
            done = True
        elif self.steps >= self.max_steps:
            done = True
        
        info = {
            'distance': float(dist),
            'steps': self.steps,
            'reached': np.array_equal(self.agent_pos, self.target_pos),
        }
        
        return self._get_observation(), reward, done, info
    
    def get_obs_dim(self) -> int:
        return self.size * 2 + 2 + 4  # agent_pos + direction + obstacle_density


class NonStationaryGridWorld(GridWorld):
    """非平稳 GridWorld — 目标位置和障碍物会随时间变化"""
    
    def __init__(self, size: int = 8, seed: int = 42, change_freq: int = 50):
        self.change_freq = change_freq
        super().__init__(size, seed)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        self.steps += 1
        
        # 每隔 change_freq 步改变环境
        if self.steps % self.change_freq == 0 and self.steps > 0:
            # 随机移动目标
            old_target = self.target_pos.copy()
            while np.array_equal(old_target, self.target_pos):
                self.target_pos = self.rng.randint(0, self.size, size=2)
            
            # 随机添加/移除障碍物
            if self.rng.random() > 0.5 and len(self.obstacles) > 0:
                self.obstacles.pop(self.rng.randint(0, len(self.obstacles)))
            else:
                new_obs = self.rng.randint(0, self.size, size=2)
                if (not np.array_equal(new_obs, self.agent_pos) and 
                    not np.array_equal(new_obs, self.target_pos) and
                    not any(np.array_equal(new_obs, obs) for obs in self.obstacles)):
                    self.obstacles.append(new_obs)
        
        return super().step(action)


class ProceduralLabyrinth:
    """复杂迷宫环境 — 需要反事实规划"""
    
    def __init__(self, size: int = 10, seed: int = 42, complexity: float = 0.3):
        self.size = size
        self.rng = np.random.RandomState(seed)
        self.complexity = complexity
        self.reset()
    
    def reset(self) -> np.ndarray:
        self.steps = 0
        self.max_steps = self.size * 6
        
        # 生成迷宫 (0=通路, 1=墙)
        self.grid = np.zeros((self.size, self.size), dtype=int)
        
        # 随机添加墙壁
        for i in range(self.size):
            for j in range(self.size):
                if self.rng.random() < self.complexity:
                    self.grid[i, j] = 1
        
        # 确保起点和终点是通路
        self.grid[0, 0] = 0
        self.grid[self.size-1, self.size-1] = 0
        
        self.agent_pos = np.array([0, 0])
        self.target_pos = np.array([self.size-1, self.size-1])
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """获取观察"""
        # 位置编码
        pos_encoding = np.zeros(self.size * 2)
        pos_encoding[self.agent_pos[0]] = 1.0
        pos_encoding[self.size + self.agent_pos[1]] = 1.0
        
        # 目标方向
        direction = (self.target_pos - self.agent_pos) / self.size
        
        # 周围墙壁 (8个方向)
        walls = np.zeros(8)
        for idx, (dx, dy) in enumerate([(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]):
            nx, ny = self.agent_pos[0] + dx, self.agent_pos[1] + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                walls[idx] = self.grid[nx, ny]
            else:
                walls[idx] = 1.0  # 边界视为墙
        
        return np.concatenate([pos_encoding, direction, walls])
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        self.steps += 1
        
        moves = [np.array([-1, 0]), np.array([1, 0]), np.array([0, -1]), np.array([0, 1])]
        new_pos = self.agent_pos + moves[action]
        
        # 检查边界和墙壁
        if (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size and
            self.grid[new_pos[0], new_pos[1]] == 0):
            self.agent_pos = new_pos
        
        # 奖励
        dist = np.linalg.norm(self.agent_pos - self.target_pos)
        reward = -0.05
        
        done = False
        if np.array_equal(self.agent_pos, self.target_pos):
            reward = 20.0
            done = True
        elif self.steps >= self.max_steps:
            done = True
        
        info = {
            'distance': float(dist),
            'steps': self.steps,
            'reached': np.array_equal(self.agent_pos, self.target_pos),
        }
        
        return self._get_observation(), reward, done, info
    
    def get_obs_dim(self) -> int:
        return self.size * 2 + 2 + 8
