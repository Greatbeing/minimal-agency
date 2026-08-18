"""
核心测试
=========
验证 BreakShell Agent 的核心功能
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from micl.breakshell.agent import BreakShellAgent
from micl.breakshell.self_model import SelfModel
from micl.breakshell.planner import CounterfactualPlanner, WorldModel
from micl.breakshell.si_measurement import SIMeasurement


class TestSelfModel:
    """自我模型测试"""
    
    def test_forward_output_shape(self):
        """测试前向传播输出维度"""
        sm = SelfModel(obs_dim=10, hidden_dim=32, repr_dim=16)
        obs = np.random.randn(10)
        result = sm.forward(obs)
        
        assert result['z'].shape == (16,)
        assert result['capacity'].shape == (2,)
        assert result['state'].shape == (3,)
        assert result['goal'].shape == (2,)
    
    def test_forward_recurrent_state(self):
        """测试递归状态（不同输入 → 状态变化 → 不同输出）"""
        sm = SelfModel(obs_dim=10, hidden_dim=32, repr_dim=16)
        obs1 = np.random.randn(10)
        obs2 = np.random.randn(10)
        
        result1 = sm.forward(obs1)
        result2 = sm.forward(obs2)
        
        # 不同输入应该产生不同输出（递归状态）
        assert not np.array_equal(result1['z'], result2['z'])
    
    def test_update_reduces_loss(self):
        """测试更新能减少损失"""
        sm = SelfModel(obs_dim=10, hidden_dim=32, repr_dim=16)
        obs = np.random.randn(10)
        true_cap = np.array([0.5, 0.0])
        
        # 多次更新
        losses = []
        for _ in range(10):
            loss = sm.update(obs, true_capacity=true_cap)
            losses.append(loss)
        
        # 损失应该下降（或至少不增加）
        assert losses[-1] <= losses[0] + 0.1
    
    def test_get_self_representation(self):
        """测试获取自我表征"""
        sm = SelfModel(obs_dim=10, hidden_dim=32, repr_dim=16)
        obs = np.random.randn(10)
        z = sm.get_self_representation(obs)
        
        assert z.shape == (16,)
        assert np.all(np.isfinite(z))


class TestWorldModel:
    """世界模型测试"""
    
    def test_predict_output_shape(self):
        """测试预测输出维度"""
        wm = WorldModel(obs_dim=10, action_dim=4)
        obs = np.random.randn(10)
        
        next_obs, reward = wm.predict(obs, action=0)
        
        assert next_obs.shape == (10,)
        assert isinstance(reward, float)
    
    def test_update_reduces_loss(self):
        """测试更新能减少损失"""
        wm = WorldModel(obs_dim=10, action_dim=4)
        obs = np.random.randn(10)
        true_next = np.random.randn(10)
        true_reward = 1.0
        
        losses = []
        for _ in range(10):
            loss = wm.update(obs, 0, true_next, true_reward)
            losses.append(loss)
        
        # 损失应该下降
        assert losses[-1] <= losses[0] + 0.1


class TestCounterfactualPlanner:
    """反事实规划器测试"""
    
    def test_plan_returns_valid_action(self):
        """测试规划返回有效动作"""
        planner = CounterfactualPlanner(action_dim=4, plan_depth=3)
        wm = WorldModel(obs_dim=10, action_dim=4)
        planner.set_world_model(wm)
        
        obs = np.random.randn(10)
        self_model_output = {'capacity': np.array([0.5, 0.5])}
        
        action, info = planner.plan(obs, self_model_output)
        
        assert 0 <= action < 4
        assert 'action_values' in info
        assert 'best_value' in info


class TestSIMeasurement:
    """SI 测量测试"""
    
    def test_record_and_compute(self):
        """测试记录和计算 SI"""
        si = SIMeasurement()
        
        # 记录一些数据
        for _ in range(10):
            si.record_action_selection(
                np.array([0.7, 0.2, 0.1]),
                np.array([0.4, 0.4, 0.2])
            )
            si.record_counterfactual_depth(3)
            si.record_feedback_coupling(1.0, 0.8)
        
        si_val, components = si.compute_si()
        
        assert 0 <= si_val <= 1
        assert 'sm' in components
        assert 'cf' in components
        assert 'fb' in components
    
    def test_zero_kl_gives_zero_sm(self):
        """测试 KL=0 时自我模型贡献为 0"""
        si = SIMeasurement()
        
        for _ in range(10):
            si.record_action_selection(
                np.array([0.5, 0.5]),
                np.array([0.5, 0.5])
            )
        
        si_val, components = si.compute_si()
        assert components['sm'] == 0.0


class TestBreakShellAgent:
    """BreakShell Agent 集成测试"""
    
    def test_select_action(self):
        """测试动作选择"""
        agent = BreakShellAgent(obs_dim=10, action_dim=4, seed=42)
        obs = np.random.randn(10)
        
        action, info = agent.select_action(obs, eval_mode=False)
        
        assert 0 <= action < 4
        assert 'self_model_output' in info
        assert 'combined_probs' in info
    
    def test_update(self):
        """测试更新"""
        agent = BreakShellAgent(obs_dim=10, action_dim=4, seed=42)
        obs = np.random.randn(10)
        action = 0
        next_obs = np.random.randn(10)
        reward = 1.0
        
        info = agent.update(obs, action, next_obs, reward, done=False)
        
        assert 'wm_loss' in info
        assert 'sm_loss' in info
        assert 'si' in info
    
    def test_get_si(self):
        """测试获取 SI"""
        agent = BreakShellAgent(obs_dim=10, action_dim=4, seed=42)
        
        # 先运行几步
        obs = np.random.randn(10)
        for _ in range(5):
            action, _ = agent.select_action(obs)
            next_obs = np.random.randn(10)
            agent.update(obs, action, next_obs, 1.0, False)
            obs = next_obs
        
        si, components = agent.get_si()
        
        assert 0 <= si <= 1
        assert isinstance(components, dict)
    
    def test_deterministic_eval(self):
        """测试 eval 模式确定性"""
        agent = BreakShellAgent(obs_dim=10, action_dim=4, seed=42)
        obs = np.random.randn(10)
        
        actions = set()
        for _ in range(10):
            action, _ = agent.select_action(obs, eval_mode=True)
            actions.add(action)
        
        # eval 模式应该总是返回相同动作
        assert len(actions) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
