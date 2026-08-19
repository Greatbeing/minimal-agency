# -*- coding: utf-8 -*-
"""
BreakShell 端到端测试
========================
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from breakshell import BreakShell, NormalAgent, CapabilityEnv, EnergyEnv, FinancialEnv
from breakshell.llm_agent import AgentLoop, create_default_registry, MockProvider, run_agent
from breakshell.eval import EvalRunner, PerformanceBenchmark
from breakshell.cognitive import create_cognitive_agent


# ========================================
# RL Agent 测试
# ========================================

class TestBreakShell:
    def test_breakshell_train_eval(self):
        env = CapabilityEnv(seed=42)
        agent = BreakShell(action_dim=3, lr=0.005)
        agent.train(env, num_episodes=10, verbose=False)
        reward = agent.evaluate(env, num_episodes=5)
        assert isinstance(reward, float)

    def test_breakshell_vs_normal(self):
        env = CapabilityEnv(seed=42)
        bs = BreakShell(action_dim=3, lr=0.005)
        normal = NormalAgent(obs_dim=4, action_dim=3, lr=0.005)
        bs.train(env, num_episodes=10, verbose=False)
        normal.train(env, num_episodes=10, verbose=False)
        bs_reward = bs.evaluate(env, num_episodes=5)
        normal_reward = normal.evaluate(env, num_episodes=5)
        assert isinstance(bs_reward, float)
        assert isinstance(normal_reward, float)


# ========================================
# LLM Agent 测试
# ========================================

class TestLLMAgent:
    def test_run_agent(self):
        state = run_agent("列出当前目录", provider="mock", max_steps=5)
        assert state.status == "finished"
        assert state.step_count > 0

    def test_agent_loop(self):
        llm = MockProvider()
        registry = create_default_registry()
        agent = AgentLoop(llm, registry, max_steps=5)
        state = agent.run("测试任务")
        assert state.status in ["finished", "failed"]


# ========================================
# 评测测试
# ========================================

class TestEval:
    def test_eval_dataset(self):
        data = EvalRunner().run_all()
        assert data["total"] > 0
        assert "categories" in data

    def test_benchmark(self):
        bench = PerformanceBenchmark()
        results = bench.run_all()
        assert "tools" in results
        assert "summary" in results


# ========================================
# 认知 Agent 测试
# ========================================

class TestCognitive:
    def test_cognitive_agent(self):
        agent = create_cognitive_agent()
        result = agent.process("测试", [{"step": 0, "success": True}], [{"tool": "list_dir", "result": {"success": True}}], True)
        assert "reflection" in result

    def test_memory_retrieval(self):
        agent = create_cognitive_agent()
        agent.process("分析项目", [{}], [{"tool": "list_dir"}], True)
        context = agent.get_context_for_new_task("分析代码")
        assert "experience" in context
        assert "advice" in context


# ========================================
# 知识银行测试
# ========================================

class TestKnowledge:
    def test_import_search(self):
        from breakshell.knowledge import create_knowledge_store, SearchEngine, KnowledgeItem
        store = create_knowledge_store(":memory:")
        item = KnowledgeItem(id="test1", title="Test", content="test content", type="document", source="test", confidence=0.8)
        store.store(item)
        search = SearchEngine(store)
        results = search.search("test")
        assert len(results) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
