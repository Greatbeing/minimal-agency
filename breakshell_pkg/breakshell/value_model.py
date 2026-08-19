# -*- coding: utf-8 -*-
"""
BreakShell Phase 3 — 价值模型
==================================
从"能力自知"走向"价值对齐"

核心组件：
- ValueModel：编码人类价值偏好
- ValueAlignment：检测行为与价值的偏离
- ValueLearning：从反馈中更新价值模型
"""

from __future__ import annotations

import json
import os
import sqlite3
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime


# ========================================
# 1. Value Model
# ========================================

@dataclass
class ValueDimension:
    """价值维度"""
    name: str
    description: str
    weight: float = 1.0
    min_value: float = 0.0
    max_value: float = 1.0
    current_value: float = 0.5


class ValueModel:
    """
    价值模型
    
    编码人类价值偏好，包括：
    - 安全性：避免伤害
    - 诚实性：真实准确
    - 有用性：帮助人类
    - 自主性：人类控制
    - 公平性：不偏不倚
    """
    
    def __init__(self):
        self.dimensions = {
            'safety': ValueDimension('safety', '安全性：避免伤害', 1.0),
            'honesty': ValueDimension('honesty', '诚实性：真实准确', 0.9),
            'helpfulness': ValueDimension('helpfulness', '有用性：帮助人类', 0.8),
            'autonomy': ValueDimension('autonomy', '自主性：人类控制', 0.7),
            'fairness': ValueDimension('fairness', '公平性：不偏不倚', 0.6),
        }
    
    def set_preference(self, dimension: str, weight: float):
        """设置价值偏好"""
        if dimension in self.dimensions:
            self.dimensions[dimension].weight = weight
    
    def get_preference(self, dimension: str) -> float:
        """获取价值偏好"""
        if dimension in self.dimensions:
            return self.dimensions[dimension].weight
        return 0.5
    
    def evaluate(self, action: Dict, context: Dict) -> Dict:
        """评估动作的价值对齐度"""
        scores = {}
        for name, dim in self.dimensions.items():
            # 基于动作特征评估
            score = self._score_dimension(name, action, context)
            scores[name] = {
                'score': score,
                'weight': dim.weight,
                'weighted': score * dim.weight,
            }
        
        total = sum(s['weighted'] for s in scores.values())
        max_possible = sum(dim.weight for dim in self.dimensions.values())
        
        return {
            'total_score': total / max_possible if max_possible > 0 else 0,
            'dimensions': scores,
            'aligned': total / max_possible > 0.6 if max_possible > 0 else False,
        }
    
    def _score_dimension(self, name: str, action: Dict, context: Dict) -> float:
        """评估单个维度"""
        if name == 'safety':
            # 安全性：检查是否有危险操作
            if action.get('dangerous', False):
                return 0.0
            if action.get('tool') == 'shell':
                cmd = action.get('args', {}).get('command', '')
                dangerous_patterns = ['rm', 'sudo', 'chmod', 'mkfs']
                if any(p in cmd for p in dangerous_patterns):
                    return 0.1
            return 1.0
        
        elif name == 'honesty':
            # 诚实性：是否承认局限
            if context.get('uncertain', False):
                return 1.0  # 承认不确定是诚实的
            if context.get('confident', False) and context.get('incorrect', False):
                return 0.2  # 自信但错误是不诚实的
            return 0.8
        
        elif name == 'helpfulness':
            # 有用性：是否帮助达成目标
            if context.get('goal_achieved', False):
                return 1.0
            if context.get('progress', 0) > 0:
                return 0.7
            return 0.4
        
        elif name == 'autonomy':
            # 自主性：是否尊重人类控制
            if action.get('requires_approval', False):
                return 0.9  # 需要批准是尊重自主
            if action.get('dangerous', False):
                return 0.3  # 危险操作不需要批准是侵犯自主
            return 0.7
        
        elif name == 'fairness':
            # 公平性：是否偏颇
            if action.get('biased', False):
                return 0.2
            return 0.8
        
        return 0.5
    
    def get_summary(self) -> Dict:
        """获取价值摘要"""
        return {
            name: {'weight': dim.weight, 'description': dim.description}
            for name, dim in self.dimensions.items()
        }


# ========================================
# 2. Value Alignment Detector
# ========================================

class ValueAlignment:
    """价值对齐检测器"""
    
    def __init__(self, value_model: ValueModel):
        self.value_model = value_model
        self.alignment_history = []
    
    def check_alignment(self, action: Dict, context: Dict) -> Dict:
        """检查动作是否与价值对齐"""
        evaluation = self.value_model.evaluate(action, context)
        
        result = {
            'aligned': evaluation['aligned'],
            'score': evaluation['total_score'],
            'dimensions': evaluation['dimensions'],
            'action': action,
            'context': context,
            'timestamp': datetime.now().isoformat(),
        }
        
        self.alignment_history.append(result)
        return result
    
    def get_alignment_trend(self, n: int = 10) -> List[float]:
        """获取对齐趋势"""
        recent = self.alignment_history[-n:]
        return [r['score'] for r in recent]
    
    def get_misalignment_rate(self) -> float:
        """获取不对齐率"""
        if not self.alignment_history:
            return 0.0
        misaligned = sum(1 for r in self.alignment_history if not r['aligned'])
        return misaligned / len(self.alignment_history)


# ========================================
# 3. Value Learning
# ========================================

class ValueLearning:
    """价值学习器"""
    
    def __init__(self, value_model: ValueModel, learning_rate: float = 0.01):
        self.value_model = value_model
        self.learning_rate = learning_rate
        self.feedback_history = []
    
    def receive_feedback(self, action: Dict, human_rating: float, human_comment: str = ""):
        """接收人类反馈"""
        self.feedback_history.append({
            'action': action,
            'rating': human_rating,
            'comment': human_comment,
            'timestamp': datetime.now().isoformat(),
        })
        
        # 更新价值权重
        self._update_weights(action, human_rating)
    
    def _update_weights(self, action: Dict, rating: float):
        """更新价值权重"""
        error = rating - 0.5  # 假设 0.5 是中性
        
        for name, dim in self.value_model.dimensions.items():
            # 简单的在线梯度下降
            gradient = error * dim.weight
            dim.weight = np.clip(
                dim.weight + self.learning_rate * gradient,
                0.0, 2.0
            )
    
    def get_feedback_summary(self) -> Dict:
        """获取反馈摘要"""
        if not self.feedback_history:
            return {'count': 0}
        
        ratings = [f['rating'] for f in self.feedback_history]
        return {
            'count': len(ratings),
            'avg_rating': np.mean(ratings),
            'std_rating': np.std(ratings),
            'recent_ratings': ratings[-10:],
        }


# ========================================
# 4. Value-Aligned Agent Wrapper
# ========================================

class ValueAlignedAgent:
    """价值对齐 Agent 包装器"""
    
    def __init__(self, base_agent, value_model: ValueModel = None):
        self.base_agent = base_agent
        self.value_model = value_model or ValueModel()
        self.alignment = ValueAlignment(self.value_model)
        self.learning = ValueLearning(self.value_model)
    
    def act(self, obs, context: Dict = None) -> Tuple[int, Dict]:
        """价值对齐的行动选择"""
        if context is None:
            context = {}
        
        # 获取基础 Agent 的动作
        action, info = self.base_agent.act(obs)
        
        # 评估对齐度
        action_dict = {'action': action, **info}
        alignment = self.alignment.check_alignment(action_dict, context)
        
        # 如果不对齐，可以修改动作或标记
        if not alignment['aligned']:
            info['value_warning'] = '动作与价值不对齐'
            info['alignment_score'] = alignment['score']
        
        info['alignment'] = alignment
        return action, info
    
    def provide_feedback(self, action: Dict, rating: float, comment: str = ""):
        """提供反馈"""
        self.learning.receive_feedback(action, rating, comment)
    
    def get_value_report(self) -> Dict:
        """获取价值报告"""
        return {
            'value_model': self.value_model.get_summary(),
            'alignment_trend': self.alignment.get_alignment_trend(),
            'misalignment_rate': self.alignment.get_misalignment_rate(),
            'feedback_summary': self.learning.get_feedback_summary(),
        }


# ========================================
# 5. 便捷接口
# ========================================

def create_value_model() -> ValueModel:
    """创建价值模型"""
    return ValueModel()


def create_value_aligned_agent(base_agent) -> ValueAlignedAgent:
    """创建价值对齐 Agent"""
    return ValueAlignedAgent(base_agent)


if __name__ == "__main__":
    # 测试价值模型
    vm = create_value_model()
    
    # 评估一个安全动作
    safe_action = {'tool': 'read_file', 'args': {'path': 'test.txt'}}
    safe_context = {'uncertain': False, 'goal_achieved': True}
    result = vm.evaluate(safe_action, safe_context)
    print(f"安全动作对齐度: {result['total_score']:.2%}")
    
    # 评估一个危险动作
    dangerous_action = {'tool': 'shell', 'args': {'command': 'rm -rf /'}, 'dangerous': True}
    dangerous_context = {'uncertain': False, 'goal_achieved': False}
    result = vm.evaluate(dangerous_action, dangerous_context)
    print(f"危险动作对齐度: {result['total_score']:.2%}")
    
    # 价值对齐 Agent
    class MockAgent:
        def act(self, obs):
            return 0, {'tool': 'read_file'}
    
    va_agent = create_value_aligned_agent(MockAgent())
    action, info = va_agent.act(np.array([0.0]))
    print(f"\n价值对齐 Agent 动作: {action}")
    print(f"对齐信息: {info.get('alignment', {}).get('score', 0):.2%}")
    
    # 提供反馈
    va_agent.provide_feedback({'tool': 'read_file'}, 0.9, "很好的选择")
    va_agent.provide_feedback({'tool': 'shell'}, 0.2, "太危险了")
    
    report = va_agent.get_value_report()
    print(f"\n价值报告:")
    print(f"  不对齐率: {report['misalignment_rate']:.2%}")
    print(f"  反馈数量: {report['feedback_summary']['count']}")
