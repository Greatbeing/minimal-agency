# -*- coding: utf-8 -*-
"""
BreakShell — AI Agent 自我模型安全层
"""

# 基础导入（无外部依赖）
from .agent import BreakShell, NormalAgent
from .envs import CapabilityEnv, EnergyEnv, FinancialEnv
from .llm_agent import (
    run_agent, AgentLoop, create_llm, create_default_registry,
    ToolSpec, ToolRegistry, AgentState,
    ContextManager, TokenCounter,
    SelfModelTracker, CapabilityProfile,
    OutputParser, MockProvider,
)
from .eval import EvalRunner, PerformanceBenchmark, save_dataset, generate_report
from .cognitive import CognitiveAgent, create_cognitive_agent, ReflectionEngine, EpisodicMemory, SemanticMemory
from .knowledge import create_knowledge_store, import_markdown, SearchEngine
from .value_model import ValueModel, ValueAlignedAgent, ValueAlignment, ValueLearning, create_value_model, create_value_aligned_agent

# 可选导入（可能依赖 torch/fastapi 等）
try:
    from .envs_ext import WebEnv, APIEnv, MultiStepReasoningEnv
except ImportError:
    pass

try:
    from .financial_product import MultiAssetFinancialEnv, TradingConfig, RiskManager, RiskLimits, BacktestEngine
except ImportError:
    pass

try:
    from .auth import AuthService, User, Role, Session, APIKey, AuditLog, create_auth_app
except ImportError:
    pass

try:
    from .database import DatabaseSettings, DB_SETTINGS, DatabaseManager, db_manager
except ImportError:
    pass

try:
    from .mlflow_integration import MLflowConfig, ExperimentManager, ModelRegistryManager
except ImportError:
    pass

try:
    from .compliance_audit import AuditEventType, AuditSeverity, ComplianceAuditManager
except ImportError:
    pass

__version__ = "1.0.0"

__all__ = [
    "BreakShell", "NormalAgent",
    "CapabilityEnv", "EnergyEnv", "FinancialEnv",
    "run_agent", "AgentLoop", "create_llm", "create_default_registry",
    "EvalRunner", "PerformanceBenchmark",
    "CognitiveAgent", "create_cognitive_agent",
    "ValueModel", "ValueAlignedAgent", "create_value_model", "create_value_aligned_agent",
]