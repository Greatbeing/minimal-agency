# -*- coding: utf-8 -*-
"""
BreakShell — AI Agent 自我模型安全层
=====================================

安装：
    pip install breakshell

使用：
    from breakshell import BreakShell, CapabilityEnv, run_agent, AgentLoop
    
    # 便捷函数
    state = run_agent("列出当前目录的所有文件")
    
    # 完整 Agent
    agent = AgentLoop(llm, registry)
    state = agent.run("你的任务")

命令行：
    breakshell run "列出当前目录的所有文件" --provider mock
    breakshell train --env capability --episodes 500 --output my_agent
    breakshell evaluate --model my_agent --episodes 100
    breakshell compare --env capability --episodes 500
    breakshell session list
"""

from .agent import BreakShell, NormalAgent
from .envs import CapabilityEnv, EnergyEnv, FinancialEnv
from .envs_ext import WebEnv, APIEnv, MultiStepReasoningEnv
from .llm_agent import run_agent, AgentLoop, create_llm, create_default_registry
from .eval import EvalRunner, PerformanceBenchmark, save_dataset, generate_report
from .cognitive import CognitiveAgent, create_cognitive_agent, ReflectionEngine, EpisodicMemory, SemanticMemory
from .knowledge import create_knowledge_store, import_markdown, SearchEngine

__version__ = "0.6.0"
__all__ = [
    "BreakShell", "NormalAgent",
    "CapabilityEnv", "EnergyEnv", "FinancialEnv",
    "WebEnv", "APIEnv", "MultiStepReasoningEnv",
    "run_agent", "AgentLoop", "create_llm", "create_default_registry",
    "EvalRunner", "PerformanceBenchmark", "save_dataset", "generate_report",
    "CognitiveAgent", "create_cognitive_agent", "ReflectionEngine",
    "EpisodicMemory", "SemanticMemory",
    "create_knowledge_store", "import_markdown", "SearchEngine",
]
