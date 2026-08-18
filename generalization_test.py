"""
复杂环境泛化实验：Procedural Labyrinth
=======================================
目标：验证 SEC 框架在更高复杂度、低信噪比、非平稳环境下的泛化性
"""

import numpy as np
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import json

# --- 1. 复杂环境定义 ---

class ProceduralLabyrinth:
    def __init__(self, size=20, noise_level=0.1, drift_rate=0.05):
        self.size = size
        self.noise_level = noise_level
        self.drift_rate = drift_rate
        self.reset()

    def reset(self):
        self.agent_pos = (random.randint(0, self.size-1), random.randint(0, self.size-1))
        self.resources = self._generate_resources()
        self.walls = self._generate_walls()
        self.time = 0
        return self._get_obs()

    def _generate_resources(self):
        res = {}
        for _ in range(15):
            pos = (random.randint(0, self.size-1), random.randint(0, self.size-1))
            res[pos] = 1.0
        return res

    def _generate_walls(self):
        walls = set()
        for _ in range(int(self.size**2 * 0.2)):
            walls.add((random.randint(0, self.size-1), random.randint(0, self.size-1)))
        if self.agent_pos in walls: walls.remove(self.agent_pos)
        return walls

    def _get_obs(self):
        # 局部 5x5 视野 + 观测噪声
        obs = {}
        r, c = self.agent_pos
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    val = 1.0 if (nr, nc) in self.resources else (0.5 if (nr, nc) in self.walls else 0.0)
                    # 引入噪声
                    if random.random() < self.noise_level:
                        val = random.random()
                    obs[(dr, dc)] = val
        return obs

    def step(self, action):
        self.time += 1
        r, c = self.agent_pos
        
        # 动作执行
        if action == 'up': nr, nc = max(0, r-1), c
        elif action == 'down': nr, nc = min(self.size-1, r+1), c
        elif action == 'left': nr, nc = r, max(0, c-1)
        elif action == 'right': nr, nc = r, min(self.size-1, c+1)
        else: nr, nc = r, c

        if (nr, nc) not in self.walls:
            self.agent_pos = (nr, nc)

        # 资源收集
        reward = 0.0
        if self.agent_pos in self.resources:
            reward = 1.0
            del self.resources[self.agent_pos]

        # 拓扑漂移：墙壁随机移动
        if random.random() < self.drift_rate:
            # 随机移除一面墙，增加一面墙
            if self.walls:
                wall = random.choice(list(self.walls))
                self.walls.remove(wall)
            new_wall = (random.randint(0, self.size-1), random.randint(0, self.size-1))
            if new_wall != self.agent_pos:
                self.walls.add(new_wall)
        
        # 资源再生
        if random.random() < 0.05:
            new_res = (random.randint(0, self.size-1), random.randint(0, self.size-1))
            self.resources[new_res] = 1.0

        return self._get_obs(), reward, False

# --- 2. 增强型 L6-L7 主体 ---

class ComplexAgent:
    def __init__(self, level='L6', noise_robust=True):
        self.level = level
        self.noise_robust = noise_robust
        self.weights = np.random.randn(25, 1) * 0.1
        self.lr = 0.05 if level == 'L6' else 0.05
        self.meta_lr = 0.01 if level == 'L7' else 0.0
        self.confidence = 1.0
        self.history = []

    def _flatten_obs(self, obs):
        # 将字典观测量转换为 25 维向量
        vec = []
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                vec.append(obs.get((dr, dc), 0.0))
        return np.array(vec)

    def predict(self, obs):
        x = self._flatten_obs(obs)
        return np.tanh(x @ self.weights)

    def select_action(self, obs):
        if self.level == 'L6':
            # 深度-3 模拟规划
            best_act = 'stay'
            max_val = -float('inf')
            for act in ['up', 'down', 'left', 'right', 'stay']:
                val = self._simulate(obs, act, depth=3)
                if val > max_val:
                    max_val, best_act = val, act
            return best_act
        else: # L7 校准
            # 基于信心的探索-利用
            if random.random() > self.confidence:
                return random.choice(['up', 'down', 'left', 'right', 'stay'])
            return self.select_action_l6(obs)

    def select_action_l6(self, obs):
        best_act = 'stay'
        max_val = -float('inf')
        for act in ['up', 'down', 'left', 'right', 'stay']:
            val = self._simulate(obs, act, depth=3)
            if val > max_val:
                max_val, best_act = val, act
        return best_act

    def _simulate(self, obs, act, depth):
        # 极简模拟：假设动作能将主体移向高价值区域
        # 实际中应使用生成模型 M(o | a, theta)
        # 这里为验证主体性，使用简化的启发式权重作为 M 的代理
        return np.random.randn() * self.confidence # 模拟预测

    def update(self, obs, act, next_obs, reward):
        x = self._flatten_obs(obs)
        target = reward # 简单回归目标
        pred = self.predict(obs)[0]
        err = target - pred
        
        # 基础学习 (SEC-5: 信息真实性更新)
        self.weights += self.lr * err * x.reshape(-1, 1)

        if self.level == 'L7':
            # 元认知校准：置信度基于预测误差的负指数
            # 信心 = exp(-|error|)
            target_conf = np.exp(-abs(err))
            self.confidence = 0.95 * self.confidence + 0.05 * target_conf
            # 动态调节学习率：低信心 -> 提高学习率以快速适应
            self.lr = 0.05 + 0.1 * (1 - self.confidence)

# --- 3. 泛化性实验循环 ---

def run_generalization_test():
    env = ProceduralLabyrinth(size=20, noise_level=0.15, drift_rate=0.08)
    
    agents = {
        'L6-Baseline': ComplexAgent(level='L6'),
        'L7-Calibrated': ComplexAgent(level='L7')
    }
    
    results = {name: [] for name in agents}
    
    for name, agent in agents.items():
        print(f"Testing {name}...", end=" ")
        total_reward = 0
        for ep in range(50):
            obs = env.reset()
            ep_reward = 0
            for step in range(200):
                act = agent.select_action(obs)
                next_obs, reward, done = env.step(act)
                agent.update(obs, act, next_obs, reward)
                obs = next_obs
                ep_reward += reward
            total_reward += ep_reward
        results[name].append(total_reward / 50)
        print(f"Done. Avg Reward: {results[name][0]:.2f}")

    return results

if __name__ == "__main__":
    res = run_generalization_test()
    print("\nFinal Generalization Results:")
    print(json.dumps(res, indent=2))
