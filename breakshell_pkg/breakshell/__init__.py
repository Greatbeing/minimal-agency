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
from .financial_product import MultiAssetFinancialEnv, TradingConfig, RiskManager, RiskLimits, BacktestEngine, RiskCheckRequest, RiskCheckResponse
from .llm_agent import run_agent, AgentLoop, create_llm, create_default_registry, SelfModelTracker, CapabilityProfile, OutputParser, ToolRegistry, ToolSpec, PermissionLevel, MockProvider
from .eval import EvalRunner, PerformanceBenchmark, save_dataset, generate_report, OutputParser
from .cognitive import CognitiveAgent, create_cognitive_agent, ReflectionEngine, EpisodicMemory, SemanticMemory
from .knowledge import create_knowledge_store, import_markdown, SearchEngine
from .value_model import ValueModel, ValueAlignedAgent, ValueAlignment, ValueLearning, create_value_model, create_value_aligned_agent
from .auth import (
    AuthService,
    User,
    Role,
    Session,
    APIKey,
    AuditLog,
    OAuth2Client,
    AuthorizationCode,
    Token,
    TokenData,
    UserCreate,
    UserUpdate,
    UserResponse,
    LoginRequest,
    RefreshTokenRequest,
    APIKeyCreate,
    APIKeyResponse,
    RoleCreate,
    RoleResponse,
    AuditLogResponse,
    create_auth_app,
    create_auth_router,
    get_auth_service,
    get_current_user,
    get_current_active_user,
    get_current_superuser,
    get_api_key_user,
    require_permissions,
    require_roles,
    pwd_context,
    oauth2_scheme,
    security,
    limiter,
)
from .database import (
    DatabaseSettings,
    DB_SETTINGS,
    DatabaseManager,
    db_manager,
    get_db,
    init_db,
    drop_db,
    close_db,
    setup_database,
    create_default_roles,
    create_superuser,
    get_db_info,
    vacuum_analyze,
    get_alembic_config,
    get_alembic_ini_content,
    lifespan_db,
    setup_database,
    create_test_db,
    setup_test_db,
)
from .mlflow_integration import (
    MLflowConfig,
    MLFLOW_CONFIG,
    MLFLOW_AVAILABLE,
    BreakShellMLflowModel,
    ExperimentConfig,
    ExperimentManager,
    ModelRegistryManager,
    BreakShellTrainer,
    AutoExperimentRunner,
    setup_mlflow,
    train_with_mlflow,
    load_production_model,
    get_model_versions,
)
from .compliance_audit import (
    AuditEventType,
    AuditSeverity,
    AuditLogEntry,
    HashChain,
    EncryptedStorage,
    AuditStorageBackend,
    SQLiteAuditStorage,
    WORMAuditStorage,
    ComplianceAuditManager,
    AuditLifecycleManager,
    create_audit_manager,
    create_audit_entry,
    create_audit_middleware,
)

__version__ = "0.9.0"
__all__ = [
    "BreakShell", "NormalAgent",
    "CapabilityEnv", "EnergyEnv", "FinancialEnv",
    "WebEnv", "APIEnv", "MultiStepReasoningEnv",
    "MultiAssetFinancialEnv", "TradingConfig", "RiskManager", "RiskLimits", "BacktestEngine", "RiskCheckRequest", "RiskCheckResponse",
    "run_agent", "AgentLoop", "create_llm", "create_default_registry",
    "EvalRunner", "PerformanceBenchmark", "save_dataset", "generate_report", "OutputParser",
    "CognitiveAgent", "create_cognitive_agent", "ReflectionEngine",
    "EpisodicMemory", "SemanticMemory",
    "create_knowledge_store", "import_markdown", "SearchEngine",
    "ValueModel", "ValueAlignedAgent", "ValueAlignment", "ValueLearning",
    "create_value_model", "create_value_aligned_agent",
    "AuthService",
    "User",
    "Role",
    "Session",
    "APIKey",
    "AuditLog",
    "OAuth2Client",
    "AuthorizationCode",
    "Token",
    "TokenData",
    "UserCreate",
    "UserUpdate",
    "UserResponse",
    "LoginRequest",
    "RefreshTokenRequest",
    "APIKeyCreate",
    "APIKeyResponse",
    "RoleCreate",
    "RoleResponse",
    "AuditLogResponse",
    "Token",
    "create_auth_app",
    "create_auth_router",
    "get_auth_service",
    "get_current_user",
    "get_current_active_user",
    "get_current_superuser",
    "get_api_key_user",
    "require_permissions",
    "require_roles",
    "pwd_context",
    "oauth2_scheme",
    "security",
    "limiter",
    "DatabaseSettings",
    "DB_SETTINGS",
    "DatabaseManager",
    "db_manager",
    "get_db",
    "init_db",
    "drop_db",
    "close_db",
    "setup_database",
    "create_default_roles",
    "create_superuser",
    "get_db_info",
    "vacuum_analyze",
    "get_alembic_config",
    "get_alembic_ini_content",
    "lifespan_db",
    "setup_database",
    "create_test_db",
    "setup_test_db",
    "SelfModelTracker",
    "CapabilityProfile",
    "ToolRegistry",
    "ToolSpec",
    "MockProvider",
    "MLflowConfig",
    "MLFLOW_CONFIG",
    "MLFLOW_AVAILABLE",
    "BreakShellMLflowModel",
    "ExperimentConfig",
    "ExperimentManager",
    "ModelRegistryManager",
    "BreakShellTrainer",
    "AutoExperimentRunner",
    "setup_mlflow",
    "train_with_mlflow",
    "load_production_model",
    "get_model_versions",
    "AuditEventType",
    "AuditSeverity",
    "AuditLogEntry",
    "HashChain",
    "EncryptedStorage",
    "AuditStorageBackend",
    "SQLiteAuditStorage",
    "WORMAuditStorage",
    "ComplianceAuditManager",
    "AuditLifecycleManager",
    "create_audit_manager",
    "create_audit_entry",
    "create_audit_middleware",
]