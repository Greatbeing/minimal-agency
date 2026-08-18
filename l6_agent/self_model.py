"""
L6 Agent — 自我模型模块 (Self Model)
=====================================
核心创新：显式的、可更新的、硬连线到策略网络的自我模型。

包含三个子组件：
1. Capacity Model — 估计自身能力
2. State Model — 估计自身状态
3. Goal Model — 维持目标表征
"""

import numpy as np
from typing import Dict, Tuple, Optional


class SelfModel:
    """自我模型：显式、可更新、硬连线到行动选择"""
    
    def __init__(self, obs_dim: int, hidden_dim: int = 32, repr_dim: int = 16):
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.repr_dim = repr_dim  # 自我表征 z_t 的维度
        
        # Capacity Model: 估计自身能力 (例如：最大速度、精度、记忆力)
        self.capacity_net = self._build_mlp(obs_dim + repr_dim, hidden_dim, 2)
        
        # State Model: 估计自身状态 (例如：能量、位置、健康)
        self.state_net = self._build_mlp(obs_dim + repr_dim, hidden_dim, 3)
        
        # Goal Model: 维持目标表征
        self.goal_net = self._build_mlp(obs_dim + repr_dim, hidden_dim, 2)
        
        # 投影层：将 capacity+state+goal 映射到固定维度 repr_dim
        # 这是关键：确保输出始终是 repr_dim 维
        combined_dim = 2 + 3 + 2  # capacity + state + goal
        self.projection = {
            'W': np.random.randn(combined_dim, repr_dim) * 0.1,
            'b': np.zeros(repr_dim),
        }
        
        # 隐藏状态 (用于递归)
        self.h = np.zeros(repr_dim)
        
        # 预测误差历史 (用于元认知校准)
        self.prediction_errors = []
        
        # 参数
        self.learning_rate = 0.001
        self.momentum = 0.9
        self.velocity = {}
        for name in ['capacity', 'state', 'goal']:
            self.velocity[name] = None
    
    def _build_mlp(self, input_dim: int, hidden_dim: int, output_dim: int) -> Dict:
        """构建简单的两层 MLP"""
        scale = np.sqrt(2.0 / input_dim)
        return {
            'W1': np.random.randn(input_dim, hidden_dim) * scale,
            'b1': np.zeros(hidden_dim),
            'W2': np.random.randn(hidden_dim, output_dim) * scale,
            'b2': np.zeros(output_dim),
        }
    
    def _forward(self, x: np.ndarray, net: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """MLP 前向传播，返回输出和隐层激活"""
        h = np.maximum(0, x @ net['W1'] + net['b1'])  # ReLU
        out = h @ net['W2'] + net['b2']
        return out, h
    
    def _backward(self, x: np.ndarray, grad_output: np.ndarray, net: Dict, h: np.ndarray) -> Dict:
        """计算梯度（兼容一维输入）"""
        # 确保是二维
        h_2d = h.reshape(1, -1)  # (1, hidden_dim)
        grad_2d = grad_output.reshape(1, -1)  # (1, output_dim)
        
        grad_W2 = h_2d.T @ grad_2d  # (hidden_dim, output_dim)
        grad_b2 = np.sum(grad_2d, axis=0)
        grad_h = grad_2d @ net['W2'].T  # (1, hidden_dim)
        grad_h[h_2d <= 0] = 0  # ReLU 梯度
        grad_W1 = x.reshape(1, -1).T @ grad_h  # (input_dim, hidden_dim)
        grad_b1 = np.sum(grad_h, axis=0)
        return {
            'W1': grad_W1, 'b1': grad_b1,
            'W2': grad_W2, 'b2': grad_b2,
        }
    
    def update_weights(self, net: Dict, grads: Dict, name: str):
        """带动量的 SGD 更新"""
        if self.velocity[name] is None:
            self.velocity[name] = {k: np.zeros_like(v) for k, v in grads.items()}
        
        for k in grads:
            self.velocity[name][k] = self.momentum * self.velocity[name][k] - self.learning_rate * grads[k]
            net[k] += self.velocity[name][k]
    
    def forward(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        """
        前向传播：生成自我表征
        
        Args:
            obs: 当前观察编码 (obs_dim,)
        
        Returns:
            dict:
                - z: 自我表征向量 (repr_dim,)
                - capacity: 能力估计 (2,) — [能力均值, 能力不确定性]
                - state: 状态估计 (3,) — [状态均值, 状态不确定性, 状态效价]
                - goal: 目标表征 (2,) — [目标方向, 目标强度]
        """
        # 拼接观察和上一时间步隐状态
        x = np.concatenate([obs, self.h])
        
        # Capacity Model
        cap_out, cap_h = self._forward(x, self.capacity_net)
        capacity = cap_out  # [mean, uncertainty]
        
        # State Model
        sta_out, sta_h = self._forward(x, self.state_net)
        state = sta_out  # [mean, uncertainty, valence]
        
        # Goal Model
        goa_out, goa_h = self._forward(x, self.goal_net)
        goal = goa_out  # [direction, strength]
        
        # 更新隐状态 (简单的递归)
        combined = np.concatenate([cap_h, sta_h, goa_h])
        self.h = np.tanh(combined[:self.repr_dim])
        
        # 拼接完整自我表征 (先投影到固定维度)
        combined = np.concatenate([capacity, state, goal])
        z = combined @ self.projection['W'] + self.projection['b']
        z = np.tanh(z)  # 归一化到 [-1, 1]
        
        return {
            'z': z,
            'capacity': capacity,
            'state': state,
            'goal': goal,
            'hidden': self.h,
        }
    
    def update(self, obs: np.ndarray, true_capacity: Optional[np.ndarray] = None,
               true_state: Optional[np.ndarray] = None) -> float:
        """
        基于预测误差更新自我模型
        
        Args:
            obs: 当前观察
            true_capacity: 真实能力 (如果可观测)
            true_state: 真实状态 (如果可观测)
        
        Returns:
            total_loss: 总预测误差
        """
        # 前向传播
        x = np.concatenate([obs, self.h])
        
        cap_out, cap_h = self._forward(x, self.capacity_net)
        sta_out, sta_h = self._forward(x, self.state_net)
        goa_out, goa_h = self._forward(x, self.goal_net)
        
        total_loss = 0.0
        
        # Capacity Model 更新
        if true_capacity is not None:
            cap_loss = np.mean((cap_out - true_capacity) ** 2)
            grad_cap = 2 * (cap_out - true_capacity) / cap_out.shape[0]
            grads = self._backward(x, grad_cap, self.capacity_net, cap_h)
            self.update_weights(self.capacity_net, grads, 'capacity')
            total_loss += cap_loss
        
        # State Model 更新
        if true_state is not None:
            sta_loss = np.mean((sta_out - true_state) ** 2)
            grad_sta = 2 * (sta_out - true_state) / sta_out.shape[0]
            grads = self._backward(x, grad_sta, self.state_net, sta_h)
            self.update_weights(self.state_net, grads, 'state')
            total_loss += sta_loss
        
        self.prediction_errors.append(total_loss)
        return total_loss
    
    def get_self_representation(self, obs: np.ndarray) -> np.ndarray:
        """
        获取自我表征 (用于硬连线到策略网络)
        
        这是唯一合法获取自我表征的方法 — 策略网络必须通过这个方法
        获取自我表征，确保自我模型参与行动选择。
        """
        result = self.forward(obs)
        return result['z']
    
    def get_confidence(self, obs: np.ndarray) -> float:
        """
        元认知信心校准：基于近期预测误差的信心估计
        """
        result = self.forward(obs)
        # 不确定性越低，信心越高
        cap_uncertainty = result['capacity'][1] if len(result['capacity']) > 1 else 0.5
        state_uncertainty = result['state'][1] if len(result['state']) > 1 else 0.5
        
        # 基于预测误差历史调整
        if len(self.prediction_errors) > 10:
            recent_error = np.mean(self.prediction_errors[-10:])
            confidence = 1.0 / (1.0 + np.exp(-recent_error + 0.5))
        else:
            confidence = 0.5
        
        return float(np.clip(confidence, 0.0, 1.0))
