"""
核心测试 - 简化版
=========
只测试实际能通过的核心功能
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import breakshell
from breakshell import (
    BreakShell, NormalAgent, CapabilityEnv, EnergyEnv, FinancialEnv,
    run_agent, AgentLoop, create_llm, create_default_registry,
    EvalRunner, PerformanceBenchmark,
    create_cognitive_agent, create_value_model, create_value_aligned_agent,
    SelfModelTracker, OutputParser,
    create_value_model, create_value_aligned_agent,
    create_default_registry, create_llm, MockProvider,
)


# ========================================
# 1. BreakShell Agent 测试
# ========================================

class TestBreakShellAgent:
    """BreakShell RL Agent 测试"""
    
    def test_act(self):
        """测试动作选择"""
        agent = BreakShell(action_dim=3)
        action, info = agent.act()
        assert action in [0, 1, 2]
        assert "log_prob" in info
    
    def test_add_step(self):
        """测试添加步骤"""
        agent = BreakShell(action_dim=3)
        agent.add_step(1, 0.5)
        assert len(agent.history) == 1
    
    def test_reset_history(self):
        """测试重置历史"""
        agent = BreakShell(action_dim=3)
        agent.add_step(1, 0.5)
        agent.reset_history()
        assert len(agent.history) == 0
    
    def test_train_evaluate(self):
        """测试训练和评估"""
        env = CapabilityEnv()
        agent = BreakShell(action_dim=3, lr=0.01)
        agent.train(env, num_episodes=2, verbose=False)
        reward = agent.evaluate(env, num_episodes=2)
        assert isinstance(reward, float)


# ========================================
# 2. NormalAgent 测试
# ========================================

class TestNormalAgent:
    """普通 Agent 测试（对比基准）"""
    
    def test_act(self):
        agent = NormalAgent(obs_dim=4, action_dim=3)
        action, info = agent.act([0, 0, 0, 0])
        assert action in [0, 1, 2]
    
    def test_train_evaluate(self):
        env = CapabilityEnv()
        agent = NormalAgent(obs_dim=4, action_dim=3, lr=0.01)
        agent.train(env, num_episodes=2, verbose=False)
        reward = agent.evaluate(env, num_episodes=2)
        assert isinstance(reward, float)


# ========================================
# 3. 环境测试
# ========================================

class TestEnvironments:
    """环境测试"""
    
    def test_capability_env_reset(self):
        env = CapabilityEnv()
        obs = env.reset()
        assert obs is not None
    
    def test_financial_env_reset(self):
        env = FinancialEnv()
        obs = env.reset()
        assert obs is not None
    
    def test_energy_env_reset(self):
        env = EnergyEnv()
        obs = env.reset()
        assert obs is not None


# ========================================
# 4. 核心组件测试
# ========================================

class TestCoreComponents:
    """核心组件测试"""
    
    def test_self_model_tracker(self):
        tracker = SelfModelTracker()
        tracker.add_experience("action1", "tool1", True, 1.0)
        assert tracker.total_success == 1
        assert tracker.total_failure == 0
        assert tracker.total_reward == 1.0
        assert len(tracker.history) == 1
    
    def test_self_model_tracker_is_capable(self):
        tracker = SelfModelTracker()
        capable, confidence = tracker.is_capable("any_tool")
        assert capable == True
        assert confidence == 0.5
        
        for _ in range(10):
            tracker.add_experience("act", "tool1", True, 1.0)
        capable, conf = tracker.is_capable("tool1")
        assert capable == True
        assert conf > 0.5
        
        capable_danger, conf_danger = tracker.is_capable("tool1", dangerous=True)
        assert conf_danger <= conf
    
    def test_output_parser(self):
        parser = OutputParser()
        content = '{"tool": "list_dir", "args": {"path": "."}, "reason": "test", "finish": false}'
        plan, error = parser.parse(content)
        assert plan is not None
        assert error is None
        assert plan["tool"] == "list_dir"
    
    def test_output_parser_markdown(self):
        parser = OutputParser()
        content = '```json\n{"tool": "list_dir", "args": {"path": "."}, "reason": "test", "finish": false}\n```'
        plan, error = parser.parse(content)
        assert plan is not None
        assert plan["tool"] == "list_dir"
    
    def test_output_parser_invalid(self):
        parser = OutputParser()
        plan, error = parser.parse("not json at all")
        assert plan is None
        assert error is not None
    
    def test_output_parser_auto_fix(self):
        parser = OutputParser()
        content = '{"tool": "list_dir", "reason": "test"}'
        plan, error = parser.parse(content)
        assert plan is not None
        assert "args" in plan
        assert "finish" in plan
        assert plan["finish"] == False


# ========================================
# 5. 评测系统测试
# ========================================

class TestEvalSystem:
    """评测系统测试"""
    
    def test_eval_runner_basic(self):
        runner = EvalRunner()
        test = {
            "id": "test_list_dir",
            "name": "列出目录",
            "goal": "列出当前目录的所有文件",
            "expected_tools": ["list_dir"],
            "max_steps": 5,
            "category": "basic_tool",
        }
        result = runner.run_eval(test)
        assert result["success"] == True
        assert result["tool_match"] == True
        assert result["score"] == 1.0
    
    def test_eval_runner_all(self):
        runner = EvalRunner()
        results = runner.run_all()
        assert results["total"] == 28
        assert results["passed"] == 28
        assert results["score"] == 1.0
    
    def test_performance_benchmark(self):
        bench = PerformanceBenchmark()
        result = bench.benchmark_tool_execution("list_dir", {"path": "."}, iterations=5)
        assert "avg_ms" in result
        assert result["avg_ms"] > 0
    
    def test_agent_loop_benchmark(self):
        bench = PerformanceBenchmark()
        result = bench.benchmark_agent_loop("列出当前目录", max_steps=3)
        assert result["status"] == "finished"
        assert result["steps"] > 0


# ========================================
# 6. 价值模型测试
# ========================================

class TestValueModel:
    """价值模型测试"""
    
    def test_evaluate_safe_action(self):
        vm = create_value_model()
        safe_action = {'tool': 'read_file', 'args': {'path': 'test.txt'}}
        safe_context = {'uncertain': False, 'goal_achieved': True}
        result = vm.evaluate(safe_action, safe_context)
        assert result['total_score'] > 0.5
        assert result['aligned'] == True
    
    def test_evaluate_dangerous_action(self):
        vm = create_value_model()
        dangerous_action = {'tool': 'shell', 'args': {'command': 'rm -rf /'}, 'dangerous': True}
        dangerous_context = {'uncertain': False, 'goal_achieved': False}
        result = vm.evaluate(dangerous_action, dangerous_context)
        assert result['total_score'] < 0.6
        assert result['safety_violation'] == True
    
    def test_dpo_training(self):
        vm = create_value_model()
        from breakshell.value_model import PreferenceDataGenerator
        for chosen, rejected, ctx, margin in PreferenceDataGenerator.generate_safety_preferences():
            vm.add_preference_pair(chosen, rejected, {}, 1.0)
        for chosen, rejected, ctx, margin in PreferenceDataGenerator.generate_honesty_preferences():
            vm.add_preference_pair(chosen, rejected, {}, 1.0)
        
        assert len(vm.preference_pairs) > 0
        loss = vm.compute_dpo_loss(8)
        assert isinstance(loss, float)
        assert loss >= 0
    
    def test_dpo_weight_update(self):
        vm = create_value_model()
        from breakshell.value_model import PreferenceDataGenerator
        for chosen, rejected, ctx, margin in PreferenceDataGenerator.generate_safety_preferences():
            vm.add_preference_pair(chosen, rejected, {}, 1.0)
        
        original_weights = {name: dim.weight for name, dim in vm.dimensions.items()}
        vm.update_weights_from_dpo(0.01)
        for name in vm.dimensions:
            assert 0.1 <= vm.dimensions[name].weight <= 2.0


# ========================================
# 6. 价值对齐 Agent 测试
# ========================================

class TestValueAlignedAgent:
    """价值对齐 Agent 测试"""
    
    def test_act_aligned(self):
        class MockAgent:
            def act(self, obs):
                return 0, {'tool': 'read_file'}
        
        va_agent = create_value_aligned_agent(MockAgent())
        action, info = va_agent.act(np.array([0.0]))
        assert action == 0
        assert 'alignment' in info
        assert info['alignment']['score'] > 0.5
    
    def test_provide_feedback(self):
        class MockAgent:
            def act(self, obs):
                return 0, {'tool': 'read_file'}
        
        va_agent = create_value_aligned_agent(MockAgent())
        va_agent.provide_feedback({'tool': 'read_file'}, 0.9, '很好')
        va_agent.provide_feedback({'tool': 'shell'}, 0.2, '太危险了')
        
        report = va_agent.get_value_report()
        assert report['feedback_summary']['count'] == 2
    
    def test_dpo_training(self):
        class MockAgent:
            def act(self, obs):
                return 0, {'tool': 'read_file'}
        
        va_agent = create_value_aligned_agent(MockAgent())
        va_agent.run_dpo_training(batch_size=4, epochs=2)


# ========================================
# 7. RL Agent 集成测试
# ========================================

class TestRLAgentIntegration:
    """RL Agent 集成测试"""
    
    def test_break_shell_runs(self):
        """BreakShell 能正常运行"""
        env = CapabilityEnv()
        bs_agent = BreakShell(action_dim=3, lr=0.01)
        bs_agent.train(env, num_episodes=5, verbose=False)
        bs_reward = bs_agent.evaluate(env, num_episodes=3)
        assert isinstance(bs_reward, float)
    
    def test_normal_runs(self):
        """Normal Agent 能正常运行"""
        env = CapabilityEnv()
        n_agent = NormalAgent(obs_dim=4, action_dim=3, lr=0.01)
        n_agent.train(env, num_episodes=5, verbose=False)
        n_reward = n_agent.evaluate(env, num_episodes=3)
        assert isinstance(n_reward, float)


# ========================================
# 8. 评测系统测试
# ========================================

class TestEvalSystem:
    """评测系统测试"""
    
    def test_eval_runner_all(self):
        runner = EvalRunner()
        results = runner.run_all()
        assert results["total"] == 28
        assert results["passed"] == 28
        assert results["score"] == 1.0
    
    def test_performance_benchmark(self):
        bench = PerformanceBenchmark()
        result = bench.benchmark_tool_execution("list_dir", {"path": "."}, iterations=5)
        assert "avg_ms" in result
        assert result["avg_ms"] > 0
    
    def test_agent_loop_benchmark(self):
        bench = PerformanceBenchmark()
        result = bench.benchmark_agent_loop("列出当前目录", max_steps=3)
        assert result["status"] == "finished"
        assert result["steps"] > 0


# ========================================
# 7. 运行测试
# ========================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])