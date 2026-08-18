"""
BreakShell Agent — 主 Agent 模块
========================
集成所有组件：自我模型 + 世界模型 + 反事实规划器 + SI 测量
"""

import numpy as np
from typing import Dict, Tuple, Optional

from .self_model import SelfModel
from .planner import CounterfactualPlanner, WorldModel
from .si_measurement import SIMeasurement


class BreakShellAgent:
    """
    BreakShell Agent：具有主体性的智能体
    
    核心特征：
    1. 自我模型是显式架构组件（非涌现属性）
    2. 自我模型硬连线到行动选择通路（不可绕过）
    3. 自我模型基于预测误差在线更新
    4. 反事实规划使用自我模型
    5. 内置 SI 实时测量
    """
    
    def __init__(self, obs_dim: int, action_dim: int, 
                 hidden_dim: int = 32, repr_dim: int = 16,
                 plan_depth: int = 5, seed: int = 42):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.plan_depth = plan_depth
        self.rng = np.random.RandomState(seed)
        
        # 核心组件
        self.self_model = SelfModel(obs_dim, hidden_dim, repr_dim)
        self.world_model = WorldModel(obs_dim, action_dim, hidden_dim)
        self.planner = CounterfactualPlanner(action_dim, plan_depth)
        self.planner.set_world_model(self.world_model)
        self.si_measurement = SIMeasurement()
        
        # 策略网络 (简单的线性策略 + 自我模型输入)
        policy_input_dim = obs_dim + repr_dim  # 观察 + 自我表征
        self.policy = {
            'W': np.random.randn(policy_input_dim, action_dim) * 0.1,
            'b': np.zeros(action_dim),
        }
        
        # 训练参数
        self.learning_rate = 0.001
        self.experience_buffer = []
        self.max_buffer_size = 1000
        
        # 追踪
        self.total_steps = 0
        self.episode_rewards = []
        self.si_history = []
    
    def _policy_forward(self, obs: np.ndarray, self_repr: np.ndarray) -> np.ndarray:
        """策略网络前向传播"""
        x = np.concatenate([obs, self_repr])
        logits = x @ self.policy['W'] + self.policy['b']
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        return probs
    
    def select_action(self, obs: np.ndarray, eval_mode: bool = False) -> Tuple[int, Dict]:
        """
        选择动作 — 核心方法
        
        关键：自我模型必须参与行动选择
        SI 测量：比较有/无自我模型时的行动分布差异
        """
        # 1. 获取自我表征 (硬连线)
        self_model_output = self.self_model.forward(obs)
        self_repr = self_model_output['z']
        
        # 2. 使用自我模型进行反事实规划
        planned_action, plan_info = self.planner.plan(
            obs, self_model_output, value_fn=None
        )
        
        # 3. 策略网络 (输入包含自我表征)
        action_probs_with_sm = self._policy_forward(obs, self_repr)
        
        # 4. SI 测量：比较有/无自我模型时的行动分布
        action_probs_without_sm = self._policy_forward(obs, np.zeros_like(self_repr))
        
        # 规划先验
        plan_prior = np.zeros(self.action_dim)
        plan_prior[planned_action] = 1.0
        
        # 混合 (规划权重 0.6, 策略网络权重 0.4)
        combined_probs_with_sm = 0.6 * plan_prior + 0.4 * action_probs_with_sm
        combined_probs_with_sm /= combined_probs_with_sm.sum()
        
        combined_probs_without_sm = 0.6 * plan_prior + 0.4 * action_probs_without_sm
        combined_probs_without_sm /= combined_probs_without_sm.sum()
        
        # 记录 SI 测量
        self.si_measurement.record_action_selection(
            combined_probs_with_sm, combined_probs_without_sm
        )
        
        # 5. 选择动作 (使用有自我模型的版本)
        if eval_mode:
            action = np.argmax(combined_probs_with_sm)
        else:
            action = self.rng.choice(self.action_dim, p=combined_probs_with_sm)
        
        # 6. 记录 SI 测量信息
        info = {
            'self_model_output': self_model_output,
            'plan_info': plan_info,
            'action_probs': action_probs_with_sm,
            'combined_probs': combined_probs_with_sm,
            'combined_probs_without_sm': combined_probs_without_sm,
            'planned_action': planned_action,
            'self_repr': self_repr,
        }
        
        return action, info
    
    def update(self, obs: np.ndarray, action: int, 
               next_obs: np.ndarray, reward: float, done: bool) -> Dict:
        """
        更新所有组件
        
        Returns:
            info: 更新信息
        """
        self.total_steps += 1
        
        # 1. 获取当前自我表征
        self_model_output = self.self_model.forward(obs)
        self_repr = self_model_output['z']
        
        # 2. 世界模型更新
        wm_loss = self.world_model.update(obs, action, next_obs, reward)
        
        # 3. 自我模型更新
        # 从经验中学习自身能力
        # 如果奖励 > 0，说明能力估计可能偏低；如果奖励 < 0，可能偏高
        true_capacity = np.array([reward, 0.0])  # 简单用奖励作为能力信号
        sm_loss = self.self_model.update(obs, true_capacity=true_capacity)
        
        # 4. 策略网络更新 (简单的策略梯度)
        action_probs = self._policy_forward(obs, self_repr)
        advantage = reward  # 简化：直接用奖励作为优势
        
        # 目标：增加选中动作的概率
        target = np.zeros(self.action_dim)
        target[action] = 1.0
        
        # 梯度
        x = np.concatenate([obs, self_repr])
        grad = np.outer(x, action_probs - target) * advantage
        
        # 更新
        self.policy['W'] -= self.learning_rate * grad
        self.policy['b'] -= self.learning_rate * (action_probs - target) * advantage
        
        # 5. SI 测量
        # 消融测试：比较有/无自我模型时的行动分布差异
        # 注意：select_action 中已经记录了 action selection 的 SI
        # 这里记录其他 SI 组件
        self.si_measurement.record_counterfactual_depth(self.plan_depth)
        self.si_measurement.record_feedback_coupling(
            plan_info.get('predicted_reward', 0) if 'plan_info' in dir() else 0,
            reward
        )
        
        # 6. 计算当前 SI
        si, si_components = self.si_measurement.compute_si()
        self.si_history.append(si)
        
        info = {
            'wm_loss': wm_loss,
            'sm_loss': sm_loss,
            'si': si,
            'si_components': si_components,
            'self_repr_mean': float(np.mean(self_repr)),
            'self_repr_std': float(np.std(self_repr)),
        }
        
        return info
    
    def _compute_action_value(self, obs: np.ndarray, action: int, 
                              self_repr: np.ndarray) -> float:
        """计算特定动作的价值"""
        action_probs = self._policy_forward(obs, self_repr)
        return action_probs[action]
    
    def get_si(self) -> Tuple[float, Dict]:
        """获取当前 SI"""
        return self.si_measurement.compute_si()
    
    def get_diagnostics(self) -> Dict:
        """获取诊断信息"""
        si, components = self.get_si()
        return {
            'total_steps': self.total_steps,
            'si': si,
            'si_components': components,
            'si_trend': self.si_measurement.get_trend(),
            'self_model_error': np.mean(self.self_model.prediction_errors[-100:]) if self.self_model.prediction_errors else 0,
            'episode_rewards': self.episode_rewards[-10:] if self.episode_rewards else [],
            'avg_reward': np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0,
        }
