"""
BreakShell Agent — SI 测量模块 v2
===========================
修复：在 select_action() 中直接比较有/无自我模型时的行动分布差异

核心改进：
- 不再在 update() 中事后计算贡献
- 而是在 select_action() 中直接运行两次：一次有自我模型，一次无自我模型
- 用 KL 散度衡量自我模型对行动选择的因果贡献
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque


class SIMeasurement:
    """
    主体性指数 (Subjectivity Index) 实时测量 v2
    
    核心改进：
    - 直接测量自我模型对行动选择的因果贡献
    - 使用 KL 散度比较 π(a|s,z) vs π(a|s,0)
    - 不再依赖事后的消融测试
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        
        # 历史记录
        self.self_model_contributions = deque(maxlen=window_size)
        self.counterfactual_depths = deque(maxlen=window_size)
        self.feedback_couplings = deque(maxlen=window_size)
        self.boundary_clarity = deque(maxlen=window_size)
        
        # 权重
        self.weights = {
            'sm': 0.35,   # 自我模型参与
            'cf': 0.25,   # 反事实规划
            'fb': 0.20,   # 反馈耦合
            'se': 0.20,   # 自我-环境边界
        }
    
    def record_action_selection(self, probs_with_sm: np.ndarray, probs_without_sm: np.ndarray):
        """
        记录自我模型对行动选择的因果贡献
        
        使用 KL 散度：KL(π(a|s,z) || π(a|s,0))
        如果 KL > 0，说明自我模型改变了行动选择 → SEC-4 满足
        如果 KL = 0，说明自我模型没有影响 → SEC-4 不满足
        """
        # 避免 log(0)
        p = probs_with_sm + 1e-10
        q = probs_without_sm + 1e-10
        
        # KL 散度
        kl = np.sum(p * np.log(p / q))
        self.self_model_contributions.append(kl)
    
    def record_counterfactual_depth(self, depth: int):
        """记录反事实规划深度"""
        self.counterfactual_depths.append(depth)
    
    def record_feedback_coupling(self, predicted_reward: float, actual_reward: float):
        """
        记录行为-反馈耦合
        
        预测奖励和实际奖励的相关性越高，耦合越强
        """
        coupling = 1.0 / (1.0 + abs(predicted_reward - actual_reward))
        self.feedback_couplings.append(coupling)
    
    def record_boundary_clarity(self, self_pred: np.ndarray, env_pred: np.ndarray):
        """
        记录自我-环境边界清晰度
        
        自我预测和环境预测的差异越大，边界越清晰
        """
        clarity = np.mean(np.abs(self_pred - env_pred))
        self.boundary_clarity.append(clarity)
    
    def compute_si(self) -> Tuple[float, Dict[str, float]]:
        """
        计算当前 SI
        
        Returns:
            si: 主体性指数
            components: 各组件得分
        """
        components = {}
        
        # 自我模型参与度 (KL 散度，需要归一化)
        if len(self.self_model_contributions) > 0:
            # KL 散度范围 [0, ∞)，用 tanh 归一化到 [0, 1)
            raw_kl = np.mean(list(self.self_model_contributions))
            components['sm'] = np.tanh(raw_kl * 10)  # 放大 10 倍后归一化
        else:
            components['sm'] = 0.0
        
        # 反事实规划深度 (归一化到 0-1)
        if len(self.counterfactual_depths) > 0:
            components['cf'] = np.mean(list(self.counterfactual_depths)) / 5.0
        else:
            components['cf'] = 0.0
        
        # 反馈耦合
        if len(self.feedback_couplings) > 0:
            components['fb'] = np.mean(list(self.feedback_couplings))
        else:
            components['fb'] = 0.0
        
        # 自我-环境边界
        if len(self.boundary_clarity) > 0:
            components['se'] = np.mean(list(self.boundary_clarity))
        else:
            components['se'] = 0.0
        
        # 加权求和
        si = sum(self.weights[k] * components[k] for k in self.weights)
        
        return si, components
    
    def get_trend(self, window: int = 20) -> str:
        """获取 SI 趋势"""
        if len(self.self_model_contributions) < window * 2:
            return "insufficient_data"
        
        recent = list(self.self_model_contributions)[-window:]
        previous = list(self.self_model_contributions)[-2*window:-window]
        
        recent_mean = np.mean(recent)
        previous_mean = np.mean(previous)
        
        if recent_mean > previous_mean * 1.1:
            return "increasing"
        elif recent_mean < previous_mean * 0.9:
            return "decreasing"
        else:
            return "stable"
    
    def summary(self) -> Dict:
        """获取测量摘要"""
        si, components = self.compute_si()
        return {
            'si': si,
            'components': components,
            'trend': self.get_trend(),
            'num_samples': len(self.self_model_contributions),
        }
