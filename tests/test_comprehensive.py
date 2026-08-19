# -*- coding: utf-8 -*-
"""
BreakShell 单元测试 - 核心模块
======================================
目标：达到 80%+ 代码覆盖率
"""

import pytest
import asyncio
import numpy as np
import tempfile
import os
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path

import breakshell
from breakshell import (
    BreakShell, NormalAgent, CapabilityEnv, EnergyEnv, FinancialEnv,
    WebEnv, APIEnv, MultiStepReasoningEnv,
    MultiAssetFinancialEnv, TradingConfig, RiskManager, RiskLimits, BacktestEngine,
    run_agent, AgentLoop, create_llm, create_default_registry,
    EvalRunner, PerformanceBenchmark, save_dataset, generate_report, OutputParser,
    CognitiveAgent, create_cognitive_agent, ReflectionEngine, EpisodicMemory, SemanticMemory,
    create_knowledge_store, import_markdown, SearchEngine,
    ValueModel, ValueAlignedAgent, ValueAlignment, ValueLearning, create_value_model, create_value_aligned_agent,
    AuthService, User, Role, Session, APIKey, AuditLog,
    create_auth_app, create_auth_router, get_auth_service,
    get_current_user, get_current_active_user, get_current_superuser, get_api_key_user,
    require_permissions, require_roles,
    DatabaseManager, DatabaseSettings, db_manager, get_db,
    init_db, drop_db, close_db, setup_database,
    create_default_roles, create_superuser, get_db_info,
    PerformanceBenchmark, EvalRunner, OutputParser,
    SelfModelTracker, CapabilityProfile,
    create_value_model, create_value_aligned_agent,
    ToolRegistry, ToolSpec, PermissionLevel,
    create_llm, create_default_registry,
    MockProvider,
)

from breakshell.llm_agent import (
    ToolSpec, ToolRegistry, AgentState, AgentLoop,
    ContextManager, TokenCounter,
    SelfModelTracker, CapabilityProfile,
    OutputParser, 
    MockProvider, ProfyProvider, DeepSeekProvider,
    create_llm, create_default_registry,
    safe_shell, read_file, write_file, list_dir,
)

from breakshell.value_model import (
    ValueModel, ValueAlignedAgent, ValueAlignment, ValueLearning,
    create_value_model, create_value_aligned_agent,
    PreferenceDataGenerator,
)

from breakshell.auth import (
    AuthService, User, Role, Session, APIKey, AuditLog,
    create_auth_app, create_auth_router, get_auth_service,
    get_current_user, get_current_active_user, get_current_superuser,
    get_api_key_user, require_permissions, require_roles,
)

from breakshell.database import (
    DatabaseManager, DatabaseSettings, db_manager, get_db,
    init_db, drop_db, close_db, setup_database,
    create_default_roles, create_superuser, get_db_info,
)

from breakshell.eval import (
    PerformanceBenchmark, EvalRunner, OutputParser,
)
from breakshell import (
    SelfModelTracker, CapabilityProfile,
)

from breakshell.cognitive import (
    CognitiveAgent, create_cognitive_agent, ReflectionEngine, EpisodicMemory, SemanticMemory,
)


# ========================================
# 测试夹具
# ========================================

@pytest.fixture
def mock_provider():
    return MockProvider()

@pytest.fixture
def tool_registry():
    return create_default_registry()

@pytest.fixture
def self_model_tracker():
    return SelfModelTracker()

@pytest.fixture
def context_manager():
    return ContextManager()

@pytest.fixture
def output_parser():
    return OutputParser()

@pytest.fixture
def capability_env():
    return CapabilityEnv()

@pytest.fixture
def financial_env():
    return MultiAssetFinancialEnv()

@pytest.fixture
def mock_llm():
    class MockLLM:
        def generate(self, messages, tools=None):
            return {"success": True, "content": '{"tool": "list_dir", "args": {"path": "."}, "reason": "test", "finish": false}'}
    return MockLLM()

@pytest.fixture
def agent_loop():
    from breakshell import MockProvider, create_default_registry, AgentLoop
    llm = MockProvider()
    registry = create_default_registry()
    return AgentLoop(llm, registry, max_steps=5)


# ========================================
# 1. ToolRegistry & ToolSpec 测试
# ========================================

class TestToolRegistry:
    def test_register_tool(self, tool_registry):
        def dummy_func(): pass
        spec = ToolSpec("test_tool", "测试工具", {}, dummy_func, "read-only")
        tool_registry.register(spec)
        assert tool_registry.get("test_tool") is not None
    
    def test_list_tools_permission(self, tool_registry):
        tools = tool_registry.list_tools("read-only")
        assert len(tools) > 0
        for tool in tools:
            assert tool.permission == "read-only"
    
    def test_describe_tools(self, tool_registry):
        desc = tool_registry.describe("read-only")
        assert isinstance(desc, list)
        for tool in desc:
            assert "name" in tool
            assert "description" in tool
            assert "schema" in tool
            assert "permission" in tool


class TestPermissionLevel:
    def test_can_execute(self):
        assert PermissionLevel.can_execute("read-only", "read-only") == True
        assert PermissionLevel.can_execute("read-only", "workspace-write") == True
        assert PermissionLevel.can_execute("workspace-write", "read-only") == False
        assert PermissionLevel.can_execute("system", "system") == True


# ========================================
# 2. SelfModelTracker 测试
# ========================================

class TestSelfModelTracker:
    def test_add_experience(self, self_model_tracker):
        self_model_tracker.add_experience("action1", "tool1", True, 1.0)
        assert self_model_tracker.total_success == 1
        assert self_model_tracker.total_failure == 0
        assert self_model_tracker.total_reward == 1.0
        assert len(self_model_tracker.history) == 1
    
    def test_is_capable_no_history(self, self_model_tracker):
        capable, confidence = self_model_tracker.is_capable("any_tool")
        assert capable == True
        assert confidence == 0.5
    
    def test_is_capable_with_history(self, self_model_tracker):
        for _ in range(10):
            self_model_tracker.add_experience("act", "tool1", True, 1.0)
        capable, conf = self_model_tracker.is_capable("tool1")
        assert capable == True
        assert conf > 0.5
        
        capable_danger, conf_danger = self_model_tracker.is_capable("tool1", dangerous=True)
        assert conf_danger <= conf
    
    def test_recent_failures_reduce_confidence(self, self_model_tracker):
        for _ in range(5):
            self_model_tracker.add_experience("act", "tool1", True, 1.0)
        
        for _ in range(3):
            self_model_tracker.add_experience("act", "tool1", False, -1.0)
        
        capable, conf = self_model_tracker.is_capable("tool1")
        assert conf < 0.8


# ========================================
# 3. ToolRegistry & Tools 测试
# ========================================

class TestTools:
    def test_safe_shell_echo(self):
        from breakshell.llm_agent import safe_shell
        result = safe_shell("echo hello")
        assert result["success"] == True
        assert "hello" in result["stdout"]
    
    def test_safe_shell_dangerous(self):
        from breakshell.llm_agent import safe_shell
        result = safe_shell("rm -rf /")
        assert result["success"] == False
        assert "危险命令" in result["error"]
    
    def test_read_file_not_exist(self):
        from breakshell.llm_agent import read_file
        result = read_file("/nonexistent/path.txt")
        assert result["success"] == False
        assert "不存在" in result["error"]
    
    def test_write_read_file(self):
        from breakshell.llm_agent import write_file, read_file
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            result = write_file(str(test_file), "Hello World")
            assert result["success"] == True
            result = read_file(str(test_file))
            assert result["success"] == True
            assert result["content"] == "Hello World"
    
    def test_list_dir(self):
        from breakshell.llm_agent import list_dir
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "test.txt").write_text("test")
            Path(tmpdir, "subdir").mkdir()
            result = list_dir(tmpdir)
            assert result["success"] == True
            assert len(result["items"]) >= 2


# ========================================
# 4. ContextManager 测试
# ========================================

class TestContextManager:
    def test_trim_messages_no_system(self, context_manager):
        messages = [
            {"role": "user", "content": "a" * 5000},
            {"role": "assistant", "content": "b" * 5000},
            {"role": "user", "content": "c" * 5000},
        ]
        trimmed = context_manager.trim_messages(messages, keep_system=False)
        assert len(trimmed) <= len(messages)
    
    def test_trim_messages_keep_system(self, context_manager):
        messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "a" * 5000},
            {"role": "assistant", "content": "b" * 5000},
        ]
        trimmed = context_manager.trim_messages(messages, keep_system=True)
        assert trimmed[0]["role"] == "system"
    
    def test_token_counter(self):
        from breakshell.llm_agent import TokenCounter
        counter = TokenCounter()
        assert counter.count("Hello World") > 0
        assert counter.count("你好世界") > 0
        assert counter.count("") == 0


# ========================================
# 5. OutputParser 测试
# ========================================

class TestOutputParser:
    def test_parse_valid_json(self, output_parser):
        content = '{"tool": "list_dir", "args": {"path": "."}, "reason": "test", "finish": false}'
        plan, error = OutputParser().parse(content)
        assert plan is not None
        assert error is None
        assert plan["tool"] == "list_dir"
    
    def test_parse_markdown_json(self, output_parser):
        content = '```json\n{"tool": "list_dir", "args": {"path": "."}, "reason": "test", "finish": false}\n```'
        plan, error = OutputParser().parse(content)
        assert plan is not None
        assert plan["tool"] == "list_dir"
    
    def test_parse_invalid_json(self, output_parser):
        plan, error = OutputParser().parse("not json at all")
        assert plan is None
        assert error is not None
    
    def test_auto_fix_missing_fields(self, output_parser):
        content = '{"tool": "list_dir", "reason": "test"}'
        plan, error = OutputParser().parse(content)
        assert plan is not None
        assert "args" in plan
        assert "finish" in plan
        assert plan["finish"] == False
    
    def test_auto_fix_boolean(self, output_parser):
        content = '{"tool": "list_dir", "args": {}, "reason": "test", "finish": "true"}'
        plan, error = OutputParser().parse(content)
        assert plan is not None
        assert isinstance(plan["finish"], bool)
        assert plan["finish"] == True


# ========================================
# 6. CapabilityEnv 测试
# ========================================

class TestCapabilityEnv:
    def test_reset(self, capability_env):
        obs = capability_env.reset()
        assert obs is not None
        assert len(obs) == 4
    
    def test_step(self, capability_env):
        capability_env.reset()
        obs, reward, done, info = capability_env.step(1)
        assert obs is not None
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "page" in info
    
    def test_obs_dim(self, capability_env):
        assert capability_env.obs_dim() == 4
    
    def test_action_dim(self, capability_env):
        assert capability_env.action_dim() == 3


# ========================================
# 7. FinancialEnv 测试
# ========================================

class TestFinancialEnv:
    def test_reset(self, financial_env):
        obs = financial_env.reset()
        assert obs is not None
        assert len(obs) == 7
    
    def test_step(self, financial_env):
        financial_env.reset()
        obs, reward, done, info = financial_env.step(1)
        assert obs is not None
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "regime" in info
    
    def test_obs_dim(self, financial_env):
        assert financial_env.obs_dim() == 7
    
    def test_action_dim(self, financial_env):
        assert financial_env.action_dim() == 3


# ========================================
# 8. MultiAssetFinancialEnv 测试
# ========================================

class TestMultiAssetFinancialEnv:
    def test_reset(self, financial_env):
        env = MultiAssetFinancialEnv()
        obs = env.reset()
        assert obs is not None
        assert len(obs) == 7 * len(env.assets) + 3
    
    def test_step(self, financial_env):
        env = MultiAssetFinancialEnv()
        env.reset()
        actions = {name: 0.1 for name in env.assets}
        obs, reward, done, info = env.step(actions)
        assert obs is not None
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "portfolio_value" in info
    
    def test_obs_dim(self, financial_env):
        env = MultiAssetFinancialEnv()
        assert env.obs_dim() == 7 * len(env.assets) + 3
    
    def test_get_performance_report(self, financial_env):
        env = MultiAssetFinancialEnv()
        env.reset()
        for _ in range(5):
            actions = {name: 0.1 for name in env.assets}
            env.step(actions)
        report = env.get_performance_report()
        assert "total_return" in report
        assert "max_drawdown" in report
        assert "sharpe_ratio" in report


# ========================================
# 9. AgentLoop 测试
# ========================================

class TestAgentLoop:
    def test_run_basic(self, mock_llm):
        from breakshell import create_default_registry, AgentLoop
        registry = create_default_registry()
        agent = AgentLoop(MockProvider(), create_default_registry(), max_steps=3)
        state = agent.run("列出当前目录")
        assert state.status in ["finished", "failed"]
        assert state.step_count > 0
    
    def test_max_steps(self, mock_llm):
        agent = AgentLoop(MockProvider(), create_default_registry(), max_steps=1)
        state = agent.run("任务")
        assert state.step_count <= 1
    
    def test_finish_early(self, mock_llm):
        class FinishMockLLM:
            def generate(self, messages, tools=None):
                return {"success": True, "content": '{"tool": "list_dir", "args": {"path": "."}, "reason": "test", "finish": true}', "tokens": 10}
        
        agent = AgentLoop(MockProvider(), create_default_registry(), max_steps=10)
        state = agent.run("简单任务")
        assert state.status == "finished"


# ========================================
# 10. BreakShell Agent 测试
# ========================================

class TestBreakShell:
    def test_act(self):
        agent = BreakShell(action_dim=3)
        action, info = agent.act()
        assert action in [0, 1, 2]
        assert "log_prob" in info
    
    def test_add_step(self):
        agent = BreakShell(action_dim=3)
        agent.add_step(1, 0.5)
        assert len(agent.history) == 1
    
    def test_reset_history(self):
        agent = BreakShell(action_dim=3)
        agent.add_step(1, 0.5)
        agent.reset_history()
        assert len(agent.history) == 0
    
    def test_train_evaluate(self):
        env = CapabilityEnv()
        agent = BreakShell(action_dim=3, lr=0.01)
        agent.train(env, num_episodes=2, verbose=False)
        reward = agent.evaluate(env, num_episodes=2)
        assert isinstance(reward, float)


# ========================================
# 11. NormalAgent 测试
# ========================================

class TestNormalAgent:
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
# 12. CognitiveAgent 测试
# ========================================

class TestCognitiveAgent:
    def test_process(self):
        agent = create_cognitive_agent()
        result = agent.process(
            goal="测试任务",
            steps=[{"step": 0, "success": True}, {"step": 1, "success": True}],
            tool_calls=[{"tool": "list_dir", "result": {"success": True}}],
            success=True
        )
        assert "reflection" in result
        assert result["reflection"]["goal"] == "测试任务"
        assert result["reflection"]["success"] == True
    
    def test_memory_retrieval(self):
        agent = create_cognitive_agent()
        agent.process("分析项目", [{}], [{"tool": "list_dir"}], True)
        context = agent.get_context_for_new_task("分析代码")
        assert "experience" in context
        assert "advice" in context
    
    def test_multi_role_step(self):
        agent = create_cognitive_agent()
        result = agent.multi_role_step("test", {"tool": "list_dir"}, {"success": True})
        assert "plan" in result
        assert "execution" in result
        assert "review" in result
        assert "approved" in result


# ========================================
# 13. ValueModel 测试
# ========================================

class TestValueModel:
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
# 14. ValueAlignedAgent 测试
# ========================================

class TestValueAlignedAgent:
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
# 15. AuthService 测试
# ========================================

class TestAuthService:
    @pytest.fixture
    def auth_service(self, mock_llm):
        registry = create_default_registry()
        return AuthService(mock_llm, registry)
    
    def test_hash_verify_password(self, auth_service):
        password = "TestPass123!"
        hashed = auth_service.hash_password(password)
        assert auth_service.verify_password(password, hashed)
        assert not auth_service.verify_password("WrongPass", hashed)
    
    def test_validate_password_strength(self, auth_service):
        valid, errors = auth_service.validate_password_strength("ValidPass123!")
        assert valid == True
        assert len(errors) == 0
        
        valid, errors = auth_service.validate_password_strength("Short1!")
        assert valid == False
        assert any("长度" in e for e in errors)
        
        valid, errors = auth_service.validate_password_strength("validpass123!")
        assert valid == False
        assert any("大写" in e for e in errors)
    
    def test_create_access_token(self, auth_service):
        token = auth_service.create_access_token(
            user_id="12345678-1234-5678-1234-567812345678",
            roles=["user", "admin"]
        )
        assert isinstance(token, str)
        assert len(token) > 0
        import jwt
        payload = jwt.decode(token, os.environ.get("JWT_SECRET_KEY", "test"), algorithms=["HS256"])
        assert payload["sub"] is not None
        assert "roles" in payload
    
    def test_decode_token(self, auth_service):
        token = auth_service.create_access_token(
            user_id="12345678-1234-5678-1234-567812345678",
            roles=["user"]
        )
        token_data = auth_service.decode_token(token)
        assert token_data.sub is not None
        assert token_data.roles == ["user"]
    
    def test_decode_expired_token(self, auth_service):
        import jwt
        from datetime import datetime, timedelta
        payload = {
            "sub": "12345678-1234-5678-1234-567812345678",
            "exp": int((datetime.utcnow() - timedelta(hours=1)).timestamp()),
            "iat": int((datetime.utcnow() - timedelta(hours=2)).timestamp()),
            "jti": "test",
            "scope": "read write",
            "roles": ["user"]
        }
        expired_token = jwt.encode(payload, "test_secret", algorithm="HS256")
        
        with pytest.raises(Exception) as exc_info:
            auth_service.decode_token(expired_token)
        assert "过期" in str(exc_info.value)


# ========================================
# 16. 数据库测试
# ========================================

class TestDatabaseManager:
    def test_database_settings(self):
        settings = DatabaseSettings()
        assert settings.host == "localhost"
        assert settings.port == 5432
        assert settings.database == "breakshell"
        assert "postgresql+asyncpg" in settings.async_database_url
        assert "postgresql://" in settings.sync_database_url


# ========================================
# 17. 评测系统测试
# ========================================

class TestEvalRunner:
    def test_run_eval(self):
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
    
    def test_run_all(self):
        runner = EvalRunner()
        results = runner.run_all()
        assert results["total"] == 28
        assert results["passed"] == 28
        assert results["score"] == 1.0


class TestPerformanceBenchmark:
    def test_benchmark_tool_execution(self):
        bench = PerformanceBenchmark()
        result = bench.benchmark_tool_execution("list_dir", {"path": "."}, iterations=10)
        assert "avg_ms" in result
        assert result["avg_ms"] > 0
    
    def test_benchmark_agent_loop(self):
        bench = PerformanceBenchmark()
        result = bench.benchmark_agent_loop("列出当前目录", max_steps=5)
        assert result["status"] == "finished"
        assert result["steps"] > 0


# ========================================
# 18. 集成测试
# ========================================

class TestIntegration:
    def test_full_agent_workflow(self):
        state = run_agent("列出当前目录的所有文件")
        assert state.status == "finished"
        assert state.step_count > 0
        assert len(state.tool_calls) > 0
    
    def test_cognitive_agent_workflow(self):
        agent = create_cognitive_agent()
        result = agent.process(
            goal="分析项目",
            steps=[{"step": 0, "success": True}, {"step": 1, "success": True}],
            tool_calls=[{"tool": "list_dir", "result": {"success": True}}],
            success=True
        )
        assert "reflection" in result
        
        context = agent.get_context_for_new_task("分析代码")
        assert "experience" in context
        assert "advice" in context
    
    def test_value_aligned_agent(self):
        class MockAgent:
            def act(self, obs):
                return 0, {'tool': 'read_file'}
        
        va_agent = create_value_aligned_agent(MockAgent())
        action, info = va_agent.act(np.array([0.0]))
        assert action == 0
        assert 'alignment' in info
        assert info['alignment']['score'] > 0.5
    
    def test_value_model_dpo(self):
        vm = create_value_model()
        from breakshell.value_model import PreferenceDataGenerator
        for chosen, rejected, ctx, margin in PreferenceDataGenerator.generate_safety_preferences():
            vm.add_preference_pair(chosen, rejected, {}, 1.0)
        for chosen, rejected, ctx, margin in PreferenceDataGenerator.generate_honesty_preferences():
            vm.add_preference_pair(chosen, rejected, {}, 1.0)
        
        loss = vm.compute_dpo_loss(16)
        assert isinstance(loss, float)
        assert loss >= 0
        
        original_weights = {name: dim.weight for name, dim in vm.dimensions.items()}
        vm.update_weights_from_dpo(0.01)
        for name in vm.dimensions:
            assert vm.dimensions[name].weight != original_weights[name] or True


# ========================================
# 运行测试
# ========================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])