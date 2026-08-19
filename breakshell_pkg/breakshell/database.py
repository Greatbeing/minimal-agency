# -*- coding: utf-8 -*-
"""
BreakShell 数据库配置
======================
PostgreSQL + SQLAlchemy + Alembic 完整配置
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker, AsyncEngine
from sqlalchemy.pool import NullPool
from sqlalchemy import event
from sqlalchemy.engine import Engine

from .auth import Base


# ========================================
# 1. 数据库配置
# ========================================

class DatabaseSettings:
    """数据库配置"""
    
    def __init__(self):
        self.host = os.environ.get("DB_HOST", "localhost")
        self.port = int(os.environ.get("DB_PORT", "5432"))
        self.user = os.environ.get("DB_USER", "breakshell")
        self.password = os.environ.get("DB_PASSWORD", "breakshell")
        self.database = os.environ.get("DB_NAME", "breakshell")
        self.schema = os.environ.get("DB_SCHEMA", "public")
        
        # 连接池配置
        self.pool_size = int(os.environ.get("DB_POOL_SIZE", "10"))
        self.max_overflow = int(os.environ.get("DB_MAX_OVERFLOW", "20"))
        self.pool_timeout = int(os.environ.get("DB_POOL_TIMEOUT", "30"))
        self.pool_recycle = int(os.environ.get("DB_POOL_RECYCLE", "3600"))
        
        # SSL 配置
        self.ssl_mode = os.environ.get("DB_SSL_MODE", "prefer")
        
    @property
    def async_database_url(self) -> str:
        """异步数据库 URL (asyncpg)"""
        return (
            f"postgresql+asyncpg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )
    
    @property
    def sync_database_url(self) -> str:
        """同步数据库 URL (psycopg2) - 用于 Alembic"""
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


DB_SETTINGS = DatabaseSettings()


# ========================================
# 2. 引擎与会话管理
# ========================================

class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, settings: DatabaseSettings = None):
        self.settings = settings or DatabaseSettings()
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker] = None
    
    @property
    def engine(self) -> AsyncEngine:
        if self._engine is None:
            self._engine = create_async_engine(
                DB_SETTINGS.async_database_url,
                pool_size=DB_SETTINGS.pool_size,
                max_overflow=DB_SETTINGS.max_overflow,
                pool_timeout=DB_SETTINGS.pool_timeout,
                pool_recycle=DB_SETTINGS.pool_recycle,
                pool_pre_ping=True,
                echo=os.environ.get("DB_ECHO", "false").lower() == "true",
                poolclass=NullPool if os.environ.get("DB_POOL_CLASS") == "null" else None,
            )
            
            # 设置 search_path
            @event.listens_for(self._engine.sync_engine, "connect")
            def set_search_path(dbapi_connection, connection_record):
                with dbapi_connection.cursor() as cursor:
                    cursor.execute(f"SET search_path TO {DB_SETTINGS.schema}")
        
        return self._engine
    
    @property
    def session_factory(self) -> async_sessionmaker:
        if self._session_factory is None:
            self._session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=True,
                autocommit=False,
            )
        return self._session_factory
    
    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """获取数据库会话"""
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def close(self):
        """关闭连接池"""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            async with self.session() as session:
                await session.execute("SELECT 1")
            return True
        except Exception:
            return False


# 全局数据库管理器
db_manager = DatabaseManager()


# ========================================
# 3. 依赖注入
# ========================================

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI 依赖注入：获取数据库会话"""
    async with db_manager.session() as session:
        yield session


# ========================================
# 4. 初始化与迁移
# ========================================

async def init_db(drop_all: bool = False) -> None:
    """初始化数据库表"""
    engine = db_manager.engine
    
    if drop_all:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_db() -> None:
    """删除所有表（危险操作，仅用于测试）"""
    engine = db_manager.engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def close_db() -> None:
    """关闭数据库连接"""
    await db_manager.close()


# ========================================
# 5. 数据库工具函数
# ========================================

async def get_db_info() -> dict:
    """获取数据库信息"""
    async with db_manager.session() as session:
        # 获取表信息
        result = await session.execute("""
            SELECT table_name, 
                   pg_size_pretty(pg_total_relation_size(quote_ident(table_name))) as size
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY pg_total_relation_size(quote_ident(table_name)) DESC
        """)
        tables = [{"name": row[0], "size": row[1]} for row in result.fetchall()]
        
        # 获取数据库大小
        result = await session.execute("SELECT pg_size_pretty(pg_database_size(current_database()))")
        db_size = result.scalar()
        
        return {
            "database_size": db_size,
            "tables": tables,
            "table_count": len(tables),
        }


async def vacuum_analyze() -> None:
    """执行 VACUUM ANALYZE（需要同步连接）"""
    from sqlalchemy import create_engine
    engine = create_engine(DB_SETTINGS.sync_database_url, isolation_level="AUTOCOMMIT")
    with engine.connect() as conn:
        conn.execute("VACUUM ANALYZE")
    engine.dispose()


# ========================================
# 6. Alembic 配置
# ========================================

def get_alembic_config() -> dict:
    """获取 Alembic 配置字典"""
    return {
        "script_location": "migrations",
        "sqlalchemy.url": DB_SETTINGS.sync_database_url,
        "version_locations": "migrations/versions",
        "file_template": "%%Y%%m%%d_%%H%%M%%S_%%m",
        "timezone": "UTC",
        "compare_type": True,
        "compare_server_default": True,
        "include_schemas": True,
        "version_table": "alembic_version",
        "version_table_schema": DB_SETTINGS.schema,
    }


def get_alembic_ini_content() -> str:
    """生成 alembic.ini 内容"""
    return f"""# Alembic 配置文件
# 自动生成，请勿手动修改关键配置

[alembic]
# 模板目录
script_location = migrations

# 数据库连接 (由 env.py 从环境变量读取)
sqlalchemy.url = {DB_SETTINGS.sync_database_url}

# 迁移文件模板
file_template = %%Y%%m%%d_%%H%%M%%S_%%m

# 时区
timezone = UTC

# 类型比较
compare_type = true
compare_server_default = true

# 版本表
version_table = alembic_version
version_table_schema = {DB_SETTINGS.schema}

[post_write_hooks]
# 格式化
hooks = black
black.type = console_scripts
black.entrypoint = black
black.options = -l 120 migrations/versions

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %%H:%%M%%S
"""


# ========================================
# 6. 初始化脚本
# ========================================

async def setup_database(drop_existing: bool = False) -> None:
    """初始化数据库（创建表 + 初始数据）"""
    print("初始化数据库...")
    
    if drop_existing:
        print("删除现有表...")
        await drop_db()
    
    print("创建表...")
    await init_db()
    
    print("创建默认角色...")
    await create_default_roles()
    
    print("创建超级用户...")
    await create_superuser()
    
    print("数据库初始化完成!")


async def create_default_roles() -> None:
    """创建默认角色"""
    from .auth import Role, AuthService
    from sqlalchemy import select
    
    async with db_manager.session() as session:
        # 检查是否已存在
        result = await session.execute(select(Role).where(Role.name == "user"))
        if result.scalar_one_or_none():
            print("默认角色已存在，跳过")
            return
        
        roles = [
            Role(name="user", description="普通用户", permissions=["read", "write"]),
            Role(name="admin", description="管理员", permissions=["read", "write", "admin", "delete"], is_system=True),
            Role(name="trader", description="交易员", permissions=["read", "write", "trade"]),
            Role(name="analyst", description="分析师", permissions=["read", "analyze"]),
        ]
        
        for role in roles:
            session.add(role)
        
        await session.commit()
        print(f"创建了 {len(roles)} 个默认角色")


async def create_superuser() -> None:
    """创建超级用户"""
    from .auth import User, AuthService, pwd_context
    from sqlalchemy import select
    
    email = os.environ.get("SUPERUSER_EMAIL", "admin@breakshell.local")
    username = os.environ.get("SUPERUSER_USERNAME", "admin")
    password = os.environ.get("SUPERUSER_PASSWORD", "changeme123")
    
    async with db_manager.session() as session:
        result = await session.execute(select(User).where(User.email == email))
        if result.scalar_one_or_none():
            print("超级用户已存在，跳过")
            return
        
        user = User(
            email=email,
            username=username,
            hashed_password=pwd_context.hash(password),
            full_name="Super Administrator",
            is_superuser=True,
            is_verified=True,
            is_active=True,
        )
        session.add(user)
        await session.flush()
        
        # 分配 admin 角色
        admin_role = await session.execute(select(Role).where(Role.name == "admin"))
        admin_role = admin_role.scalar_one()
        user.roles.append(admin_role)
        
        await session.commit()
        print(f"超级用户创建成功: {username} / {password}")


# ========================================
# 7. 生命周期管理
# ========================================

@asynccontextmanager
async def lifespan_db(app=None):
    """FastAPI 生命周期管理"""
    # 启动
    print("启动数据库连接...")
    await init_db()
    await setup_database()
    yield
    # 关闭
    print("关闭数据库连接...")
    await close_db()


# ========================================
# 7. 测试支持
# ========================================

async def create_test_db() -> DatabaseManager:
    """创建测试数据库管理器（使用内存 SQLite）"""
    test_settings = DatabaseSettings()
    test_settings.host = "localhost"
    test_settings.port = 5432
    test_settings.user = "test"
    test_settings.password = "test"
    test_settings.database = "test_breakshell"
    return DatabaseManager(test_settings)


async def setup_test_db() -> DatabaseManager:
    """设置测试数据库"""
    test_db = await create_test_db()
    await test_db.init_db()
    return test_db


# ========================================
# 8. 导出
# ========================================

__all__ = [
    "DatabaseSettings",
    "DB_SETTINGS",
    "DatabaseManager",
    "db_manager",
    "get_db",
    "init_db",
    "drop_db",
    "close_db",
    "init_db",
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
]

if __name__ == "__main__":
    import asyncio
    asyncio.run(setup_database(drop_existing=True))