"""
Pocker Agent — 反事实规划器 (Counterfactual Planner)
================================================
核心创新：规划时使用自我模型 — "基于我的能力，做这个动作的成功率是多少"
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class CounterfactualPlanner:
    """
    反事实规划器：使用世界模型 + 自我模型进行多步规划
    
    与标准 MPC 的关键区别：
    - 标准 MPC：评估 "做 a 后环境会变成什么样"
    - 本规划器：评估 "基于我的能力 ĉ，做 a 的成功率，以及环境会变成什么样"
    """
    
    def __init__(self, action_dim: int, plan_depth: int = 5, num_candidates: int = 16):
        self.action_dim = action_dim
        self.plan_depth = plan_depth
        self.num_candidates = num_candidates
        
        # 世界模型 (简单的前馈网络)
        self.world_model = None
        self.smoothing_factor = 0.9
    
    def set_world_model(self, world_model):
        """设置世界模型"""
        self.world_model = world_model
    
    def plan(self, obs: np.ndarray, self_model_output: Dict, 
             value_fn=None) -> Tuple[int, Dict]:
        """
        基于自我模型的反事实规划
        
        Args:
            obs: 当前观察
            self_model_output: 自我模型输出 (包含 capacity, state, goal)
            value_fn: 价值函数 (可选)
        
        Returns:
            best_action: 最佳动作
            info: 规划信息
        """
        # 从自我模型获取能力估计
        capacity = self_model_output.get('capacity', np.array([0.5, 0.5]))
        capability_mean = capacity[0] if len(capacity) > 0 else 0.5
        capability_uncertainty = capacity[1] if len(capacity) > 1 else 0.5
        
        # 采样候选动作
        candidate_actions = self._sample_actions()
        
        # 评估每个候选动作
        action_values = []
        for action in candidate_actions:
            value = self._evaluate_action(
                obs, action, capability_mean, capability_uncertainty, value_fn
            )
            action_values.append(value)
        
        # 选择最佳动作
        best_idx = np.argmax(action_values)
        best_action = candidate_actions[best_idx]
        
        info = {
            'action_values': action_values,
            'best_value': action_values[best_idx],
            'capability_mean': capability_mean,
            'capability_uncertainty': capability_uncertainty,
            'candidate_actions': candidate_actions,
        }
        
        return best_action, info
    
    def _sample_actions(self) -> List[int]:
        """采样候选动作"""
        return list(range(self.action_dim))
    
    def _evaluate_action(self, obs: np.ndarray, action: int, 
                         capability: float, uncertainty: float,
                         value_fn=None) -> float:
        """
        评估单个动作的价值
        
        关键公式：
        U(a) = 期望奖励 × 能力匹配度 + 折扣未来价值
        
        其中：
        - 能力匹配度 = P(成功 | 能力, 动作) = sigmoid(capability - difficulty)
        - 期望奖励来自世界模型预测
        """
        # 预测环境变化 (如果世界模型可用)
        if self.world_model is not None:
            next_obs, predicted_reward = self.world_model.predict(obs, action)
        else:
            next_obs = obs
            predicted_reward = 0.0
        
        # 能力匹配度：基于能力估计的成功概率
        # 难度随动作变化 (这里简化为固定难度)
        difficulty = 0.3
        capability_match = 1.0 / (1.0 + np.exp(-(capability - difficulty)))
        
        # 不确定性惩罚：不确定性高时降低行动意愿
        uncertainty_penalty = -0.1 * uncertainty
        
        # 总价值
        total_value = predicted_reward * capability_match + uncertainty_penalty
        
        # 加上未来价值 (如果提供)
        if value_fn is not None:
            total_value += 0.95 * value_fn(next_obs)
        
        return total_value
    
    def multi_step_plan(self, obs: np.ndarray, self_model_output: Dict,
                        value_fn=None) -> Tuple[int, Dict]:
        """
        多步反事实规划
        
        使用递归评估多步动作序列
        """
        capacity = self_model_output.get('capacity', np.array([0.5, 0.5]))
        capability_mean = capacity[0] if len(capacity) > 0 else 0.5
        
        best_action = 0
        best_value = -np.inf
        best_sequence = []
        
        # 对每个首步动作
        for first_action in range(self.action_dim):
            # 评估首步 + 后续
            value, sequence = self._recursive_plan(
                obs, first_action, capability_mean, depth=self.plan_depth, value_fn=value_fn
            )
            
            if value > best_value:
                best_value = value
                best_action = first_action
                best_sequence = sequence
        
        info = {
            'best_value': best_value,
            'plan_sequence': best_sequence,
            'depth': self.plan_depth,
        }
        
        return best_action, info
    
    def _recursive_plan(self, obs: np.ndarray, first_action: int, 
                        capability: float, depth: int, value_fn=None) -> Tuple[float, List[int]]:
        """递归规划"""
        if depth == 0:
            return 0.0, []
        
        # 评估首步
        first_value = self._evaluate_action(obs, first_action, capability, 0.5, value_fn)
        
        # 预测下一状态
        if self.world_model is not None:
            next_obs, _ = self.world_model.predict(obs, first_action)
        else:
            next_obs = obs
        
        # 递归规划后续
        if depth > 1:
            best_next_action = 0
            best_next_value = -np.inf
            best_next_sequence = []
            
            for next_action in range(self.action_dim):
                value, seq = self._recursive_plan(next_obs, next_action, capability, depth-1, value_fn)
                if value > best_next_value:
                    best_next_value = value
                    best_next_action = next_action
                    best_next_sequence = seq
            
            total_value = first_value + 0.95 * best_next_value
            sequence = [first_action] + best_next_sequence
        else:
            total_value = first_value
            sequence = [first_action]
        
        return total_value, sequence


class WorldModel:
    """
    世界模型：预测环境状态转移和奖励
    """
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 32):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # 状态转移网络
        input_dim = obs_dim + action_dim
        self.transition = {
            'W1': np.random.randn(input_dim, hidden_dim) * 0.1,
            'b1': np.zeros(hidden_dim),
            'W2': np.random.randn(hidden_dim, obs_dim) * 0.1,
            'b2': np.zeros(obs_dim),
        }
        
        # 奖励预测网络
        self.reward_net = {
            'W1': np.random.randn(input_dim, hidden_dim) * 0.1,
            'b1': np.zeros(hidden_dim),
            'W2': np.random.randn(hidden_dim, 1) * 0.1,
            'b2': np.zeros(1),
        }
        
        self.learning_rate = 0.001
    
    def _forward(self, x: np.ndarray, net: Dict) -> np.ndarray:
        """前向传播"""
        h = np.maximum(0, x @ net['W1'] + net['b1'])
        out = h @ net['W2'] + net['b2']
        return out
    
    def predict(self, obs: np.ndarray, action: int) -> Tuple[np.ndarray, float]:
        """预测下一状态和奖励"""
        # One-hot 编码动作
        action_onehot = np.zeros(self.action_dim)
        action_onehot[action] = 1.0
        
        # 拼接
        x = np.concatenate([obs, action_onehot])
        
        # 预测
        next_obs = self._forward(x, self.transition)
        reward = float(self._forward(x, self.reward_net)[0])
        
        return next_obs, reward
    
    def update(self, obs: np.ndarray, action: int, 
               true_next_obs: np.ndarray, true_reward: float) -> float:
        """基于真实反馈更新世界模型"""
        action_onehot = np.zeros(self.action_dim)
        action_onehot[action] = 1.0
        x = np.concatenate([obs, action_onehot])
        
        # 预测
        pred_next_obs = self._forward(x, self.transition)
        pred_reward = float(self._forward(x, self.reward_net)[0])
        
        # 计算损失
        obs_loss = np.mean((pred_next_obs - true_next_obs) ** 2)
        reward_loss = (pred_reward - true_reward) ** 2
        
        # 简单的梯度下降 (数值梯度)
        eps = 1e-5
        loss = obs_loss + reward_loss
        
        for net in [self.transition, self.reward_net]:
            for key in net:
                grad = np.zeros_like(net[key])
                # 随机采样几个维度计算梯度 (加速)
                num_samples = min(10, net[key].size)
                indices = np.random.choice(net[key].size, num_samples, replace=False)
                for idx in indices:
                    flat = net[key].flatten()
                    orig = flat[idx]
                    
                    flat[idx] = orig + eps
                    net[key] = flat.reshape(net[key].shape)
                    loss_plus = self._compute_loss(x, true_next_obs, true_reward, net)
                    
                    flat[idx] = orig - eps
                    net[key] = flat.reshape(net[key].shape)
                    loss_minus = self._compute_loss(x, true_next_obs, true_reward, net)
                    
                    grad.flat[idx] = (loss_plus - loss_minus) / (2 * eps)
                    flat[idx] = orig
                    net[key] = flat.reshape(net[key].shape)
                
                net[key] -= self.learning_rate * grad
        
        return loss
    
    def _compute_loss(self, x: np.ndarray, true_next_obs: np.ndarray, 
                      true_reward: float, target_net: Dict) -> float:
        """计算损失"""
        pred_obs = self._forward(x, self.transition)
        pred_reward = float(self._forward(x, self.reward_net)[0])
        return np.mean((pred_obs - true_next_obs) ** 2) + (pred_reward - true_reward) ** 2
