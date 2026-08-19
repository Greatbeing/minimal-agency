# -*- coding: utf-8 -*-
"""
BreakShell 认证授权系统
=========================
JWT + OAuth2 + RBAC 完整实现
"""

from __future__ import annotations

import os
import time
import uuid
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps

from pydantic import BaseModel, Field, EmailStr
from passlib.context import CryptContext
from jose import jwt, JWTError
from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer
from fastapi.routing import APIRouter
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, DateTime, Boolean, Integer, ForeignKey, Text, Index, select
from sqlalchemy.dialects.postgresql import UUID
import sqlalchemy as sa
from contextlib import asynccontextmanager


# ========================================
# 1. 配置与常量
# ========================================

class AuthSettings(BaseModel):
    """认证配置"""
    # JWT
    secret_key: str = Field(default_factory=lambda: os.environ.get("JWT_SECRET_KEY", secrets.token_urlsafe(32)))
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    
    # OAuth2
    oauth2_providers: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    
    # 密码策略
    pwd_min_length: int = 8
    pwd_require_upper: bool = True
    pwd_require_lower: bool = True
    pwd_require_digit: bool = True
    pwd_require_special: bool = True
    
    # 速率限制
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    
    # 数据库
    database_url: str = "postgresql+asyncpg://user:pass@localhost/breakshell"
    
    class Config:
        env_prefix = "AUTH_"


AUTH_SETTINGS = AuthSettings()

# 密码哈希
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 密码流
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# Bearer token 认证
security = HTTPBearer(auto_error=False)

# Redis 客户端
redis_client: Optional[redis.Redis] = None

# 速率限制器
limiter = Limiter(key_func=get_remote_address)


# ========================================
# 2. 数据库模型
# ========================================

class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    username: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    failed_login_attempts: Mapped[int] = mapped_column(Integer, default=0)
    locked_until: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # 关系
    roles: Mapped[List["Role"]] = relationship("Role", secondary="user_roles", back_populates="users")
    sessions: Mapped[List["Session"]] = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    api_keys: Mapped[List["APIKey"]] = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    audit_logs: Mapped[List["AuditLog"]] = relationship("AuditLog", back_populates="user", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index("ix_users_email", "email"),
        Index("ix_users_username", "username"),
    )


class Role(Base):
    __tablename__ = "roles"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    permissions: Mapped[List[str]] = mapped_column(sa.ARRAY(String), default=[])
    is_system: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # 关系
    users: Mapped[List["User"]] = relationship("User", secondary="user_roles", back_populates="roles")


class UserRole(Base):
    __tablename__ = "user_roles"
    
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    role_id: Mapped[int] = mapped_column(Integer, ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True)
    assigned_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    assigned_by: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)


class Session(Base):
    __tablename__ = "sessions"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True)
    refresh_token_hash: Mapped[str] = mapped_column(String(255), index=True)
    user_agent: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ip_address: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # 关系
    user: Mapped["User"] = relationship("User", back_populates="sessions")
    
    __table_args__ = (
        Index("ix_sessions_user_id", "user_id"),
        Index("ix_sessions_expires_at", "expires_at"),
    )


class APIKey(Base):
    __tablename__ = "api_keys"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    key_prefix: Mapped[str] = mapped_column(String(20), nullable=False)  # bs_live_xxx
    key_hash: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    permissions: Mapped[List[str]] = mapped_column(sa.ARRAY(String), default=[])
    rate_limit: Mapped[int] = mapped_column(Integer, default=1000)  # requests per hour
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # 关系
    user: Mapped["User"] = relationship("User", back_populates="api_keys")


class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), index=True, nullable=True)
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    resource_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    resource_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    old_values: Mapped[Optional[Dict]] = mapped_column(sa.JSON, nullable=True)
    new_values: Mapped[Optional[Dict]] = mapped_column(sa.JSON, nullable=True)
    ip_address: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    
    # 关系
    user: Mapped[Optional["User"]] = relationship("User", back_populates="audit_logs")
    
    __table_args__ = (
        Index("ix_audit_logs_user_id", "user_id"),
        Index("ix_audit_logs_created_at", "created_at"),
        Index("ix_audit_logs_action", "action"),
    )


class OAuth2Client(Base):
    __tablename__ = "oauth2_clients"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    client_secret_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    redirect_uris: Mapped[List[str]] = mapped_column(sa.ARRAY(String), default=[])
    allowed_scopes: Mapped[List[str]] = mapped_column(sa.ARRAY(String), default=[])
    grant_types: Mapped[List[str]] = mapped_column(sa.ARRAY(String), default=["authorization_code", "refresh_token"])
    is_confidential: Mapped[bool] = mapped_column(Boolean, default=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AuthorizationCode(Base):
    __tablename__ = "authorization_codes"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    code: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    client_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("oauth2_clients.id"), index=True)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), index=True)
    redirect_uri: Mapped[str] = mapped_column(String(500), nullable=False)
    scopes: Mapped[List[str]] = mapped_column(sa.ARRAY(String), default=[])
    code_challenge: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    code_challenge_method: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    used_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


# ========================================
# 3. Pydantic 模型
# ========================================

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    scope: str = ""


class TokenData(BaseModel):
    sub: str  # user_id
    exp: int
    iat: int
    jti: str
    scope: str = ""
    roles: List[str] = []


class UserCreate(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=100, pattern="^[a-zA-Z0-9_-]+$")
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None


class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    password: Optional[str] = None
    is_active: Optional[bool] = None


class UserResponse(BaseModel):
    id: uuid.UUID
    email: str
    username: str
    full_name: Optional[str]
    is_active: bool
    is_superuser: bool
    is_verified: bool
    created_at: datetime
    roles: List[str] = []
    
    class Config:
        from_attributes = True


class LoginRequest(BaseModel):
    username: str
    password: str
    remember_me: bool = False


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class APIKeyCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    permissions: List[str] = []
    rate_limit: int = 1000
    expires_days: Optional[int] = None


class APIKeyResponse(BaseModel):
    id: uuid.UUID
    name: str
    key_prefix: str
    key: str  # 只在创建时返回完整 key
    permissions: List[str]
    rate_limit: int
    expires_at: Optional[datetime]
    created_at: datetime
    
    class Config:
        from_attributes = True


class RoleCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=50, pattern="^[a-z_]+$")
    description: Optional[str] = None
    permissions: List[str] = []


class RoleResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    permissions: List[str]
    is_system: bool
    
    class Config:
        from_attributes = True


class AuditLogResponse(BaseModel):
    id: uuid.UUID
    user_id: Optional[uuid.UUID]
    action: str
    resource_type: Optional[str]
    resource_id: Optional[str]
    success: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


# ========================================
# 4. 核心认证服务
# ========================================

class AuthService:
    """认证核心服务"""
    
    def __init__(self, session: AsyncSession, redis_client: redis.Redis = None):
        self.session = session
        self.redis = redis_client
    
    # ----- 密码处理 -----
    
    def hash_password(self, password: str) -> str:
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)
    
    def validate_password_strength(self, password: str) -> Tuple[bool, List[str]]:
        """验证密码强度"""
        errors = []
        settings = AUTH_SETTINGS
        
        if len(password) < settings.pwd_min_length:
            errors.append(f"密码长度至少 {settings.pwd_min_length} 位")
        if settings.pwd_require_upper and not any(c.isupper() for c in password):
            errors.append("必须包含大写字母")
        if settings.pwd_require_lower and not any(c.islower() for c in password):
            errors.append("必须包含小写字母")
        if settings.pwd_require_digit and not any(c.isdigit() for c in password):
            errors.append("必须包含数字")
        if settings.pwd_require_special and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            errors.append("必须包含特殊字符")
        
        return len(errors) == 0, errors
    
    # ----- JWT Token -----
    
    def create_access_token(
        self,
        user_id: uuid.UUID,
        roles: List[str],
        scopes: str = "",
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """创建访问令牌"""
        now = datetime.utcnow()
        expire = now + (expires_delta or timedelta(minutes=AUTH_SETTINGS.access_token_expire_minutes))
        
        jti = secrets.token_urlsafe(16)
        payload = {
            "sub": str(user_id),
            "exp": int(expire.timestamp()),
            "iat": int(datetime.utcnow().timestamp()),
            "jti": jti,
            "scope": " ".join(["read", "write"] + roles),
            "roles": roles,
        }
        
        return jwt.encode(payload, AUTH_SETTINGS.secret_key, algorithm=AUTH_SETTINGS.algorithm)
    
    def create_refresh_token(self, user_id: uuid.UUID) -> str:
        """创建刷新令牌"""
        now = datetime.utcnow()
        expire = now + timedelta(days=AUTH_SETTINGS.refresh_token_expire_days)
        
        jti = secrets.token_urlsafe(16)
        payload = {
            "sub": str(user_id),
            "exp": int(expire.timestamp()),
            "iat": int(datetime.utcnow().timestamp()),
            "jti": jti,
            "type": "refresh",
        }
        
        token = jwt.encode(payload, AUTH_SETTINGS.secret_key, algorithm=AUTH_SETTINGS.algorithm)
        
        # 存储到数据库
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        session = Session(
            user_id=user_id,
            refresh_token_hash=token_hash,
            expires_at=expire,
        )
        # 需要在外部添加到 session
        return token
    
    def decode_token(self, token: str) -> TokenData:
        """解码并验证 JWT"""
        try:
            payload = jwt.decode(
                token,
                AUTH_SETTINGS.secret_key,
                algorithms=[AUTH_SETTINGS.algorithm],
                options={"verify_exp": True}
            )
            return TokenData(**payload)
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token 已过期")
        except jwt.InvalidTokenError as e:
            raise HTTPException(status_code=401, detail=f"无效 Token: {str(e)}")
    
    def verify_refresh_token(self, token: str) -> Optional[uuid.UUID]:
        """验证刷新令牌"""
        try:
            payload = jwt.decode(
                token,
                AUTH_SETTINGS.secret_key,
                algorithms=[AUTH_SETTINGS.algorithm],
                options={"verify_exp": True}
            )
            if payload.get("type") != "refresh":
                return None
            return uuid.UUID(payload["sub"])
        except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
            return None
    
    # ----- 用户管理 -----
    
    async def create_user(self, user_data: UserCreate) -> User:
        """创建用户"""
        # 验证密码
        valid, errors = self.validate_password_strength(user_data.password)
        if not valid:
            raise HTTPException(status_code=400, detail="; ".join(errors))
        
        # 检查邮箱/用户名是否已存在
        existing = await self.session.execute(
            select(User).where((User.email == user_data.email) | (User.username == user_data.username))
        )
        if existing.scalar_one_or_none():
            raise HTTPException(status_code=400, detail="邮箱或用户名已存在")
        
        user = User(
            email=user_data.email,
            username=user_data.username,
            hashed_password=self.hash_password(user_data.password),
            full_name=user_data.full_name,
        )
        self.session.add(user)
        await self.session.flush()
        
        # 分配默认角色
        default_role = await self.session.execute(select(Role).where(Role.name == "user"))
        default_role = default_role.scalar_one_or_none()
        if default_role:
            user.roles.append(default_role)
        
        await self.session.commit()
        await self.session.refresh(user)
        return user
    
    async def authenticate(self, username: str, password: str) -> Optional[User]:
        """用户名/密码认证"""
        result = await self.session.execute(
            select(User).where(
                (User.username == username) | (User.email == username)
            )
        )
        user = result.scalar_one_or_none()
        
        if not user or not user.is_active:
            return None
        
        # 检查账户锁定
        if user.locked_until and user.locked_until > datetime.utcnow():
            raise HTTPException(status_code=403, detail="账户已锁定，请稍后重试")
        
        if not self.verify_password(password, user.hashed_password):
            # 记录失败尝试
            user.failed_login_attempts += 1
            if user.failed_login_attempts >= 5:
                user.locked_until = datetime.utcnow() + timedelta(minutes=15)
            await self.session.commit()
            return None
        
        # 登录成功，重置失败计数
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.utcnow()
        await self.session.commit()
        
        return user
    
    async def get_user(self, user_id: uuid.UUID) -> Optional[User]:
        result = await self.session.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()
    
    async def update_user(self, user_id: uuid.UUID, data: UserUpdate) -> Optional[User]:
        user = await self.get_user(user_id)
        if not user:
            return None
        
        update_data = data.model_dump(exclude_unset=True)
        if "password" in update_data:
            valid, errors = self.validate_password_strength(update_data["password"])
            if not valid:
                raise HTTPException(status_code=400, detail="; ".join(errors))
            update_data["hashed_password"] = self.hash_password(update_data.pop("password"))
        
        for key, value in update_data.items():
            setattr(user, key, value)
        
        user.updated_at = datetime.utcnow()
        await self.session.commit()
        await self.session.refresh(user)
        return user
    
    # ----- 角色管理 -----
    
    async def create_role(self, role_data: RoleCreate) -> Role:
        existing = await self.session.execute(select(Role).where(Role.name == role_data.name))
        if existing.scalar_one_or_none():
            raise HTTPException(status_code=400, detail="角色已存在")
        
        role = Role(
            name=role_data.name,
            description=role_data.description,
            permissions=role_data.permissions,
        )
        self.session.add(role)
        await self.session.commit()
        await self.session.refresh(role)
        return role
    
    async def assign_role(self, user_id: uuid.UUID, role_name: str, assigned_by: uuid.UUID) -> bool:
        user = await self.get_user(user_id)
        role = await self.session.execute(select(Role).where(Role.name == role_name))
        role = role.scalar_one_or_none()
        
        if not user or not role:
            return False
        
        if role in user.roles:
            return True
        
        user.roles.append(role)
        await self.session.commit()
        return True
    
    async def revoke_role(self, user_id: uuid.UUID, role_name: str) -> bool:
        user = await self.get_user(user_id)
        role = await self.session.execute(select(Role).where(Role.name == role_name))
        role = role.scalar_one_or_none()
        
        if not user or not role or role not in user.roles:
            return False
        
        user.roles.remove(role)
        await self.session.commit()
        return True
    
    # ----- 会话管理 -----
    
    async def create_session(
        self,
        user_id: uuid.UUID,
        user_agent: str = None,
        ip_address: str = None,
        remember_me: bool = False
    ) -> Session:
        expire_days = 30 if remember_me else 7
        expires_at = datetime.utcnow() + timedelta(days=expire_days)
        
        refresh_token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()
        
        session = Session(
            user_id=user_id,
            refresh_token_hash=token_hash,
            user_agent=None,  # 暂时简化
            ip_address=None,
            expires_at=datetime.utcnow() + timedelta(days=expire_days),
        )
        # 实际需要在外部添加到 session
        return session
    
    async def revoke_session(self, session_id: uuid.UUID) -> bool:
        result = await self.session.execute(select(Session).where(Session.id == session_id))
        session = result.scalar_one_or_none()
        if not session:
            return False
        session.revoked_at = datetime.utcnow()
        await self.session.commit()
        return True
    
    async def revoke_all_sessions(self, user_id: uuid.UUID) -> int:
        result = await self.session.execute(
            select(Session).where(Session.user_id == user_id, Session.revoked_at.is_(None))
        )
        sessions = result.scalars().all()
        for s in sessions:
            s.revoked_at = datetime.utcnow()
        await self.session.commit()
        return len(sessions)
    
    # ----- API Key 管理 -----
    
    async def create_api_key(self, user_id: uuid.UUID, data: APIKeyCreate) -> Tuple[APIKey, str]:
        """创建 API Key，返回 (key_obj, full_key)"""
        full_key = f"bs_{secrets.token_urlsafe(32)}"
        key_prefix = f"bs_{secrets.token_urlsafe(8)}"
        key_hash = hashlib.sha256(full_key.encode()).hexdigest()
        
        expires_at = None
        if data.expires_days:
            expires_at = datetime.utcnow() + timedelta(days=data.expires_days)
        
        api_key = APIKey(
            user_id=user_id,
            name=data.name,
            key_prefix=key_prefix,
            key_hash=key_hash,
            permissions=data.permissions,
            rate_limit=data.rate_limit,
            expires_at=expires_at,
        )
        # 需要在外部添加到 session
        return api_key, full_key
    
    async def verify_api_key(self, api_key: str) -> Optional[APIKey]:
        if not api_key.startswith("bs_"):
            return None
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        result = await self.session.execute(select(APIKey).where(APIKey.key_hash == key_hash, APIKey.is_active == True))
        api_key_obj = result.scalar_one_or_none()
        
        if not api_key_obj:
            return None
        
        if api_key_obj.expires_at and api_key_obj.expires_at < datetime.utcnow():
            return None
        
        # 更新最后使用时间
        api_key_obj.last_used_at = datetime.utcnow()
        return api_key_obj
    
    async def revoke_api_key(self, api_key_id: uuid.UUID, user_id: uuid.UUID) -> bool:
        result = await self.session.execute(
            select(APIKey).where(APIKey.id == api_key_id, APIKey.user_id == user_id)
        )
        api_key = result.scalar_one_or_none()
        if not api_key:
            return False
        api_key.is_active = False
        api_key.revoked_at = datetime.utcnow()
        await self.session.commit()
        return True
    
    # ----- 审计日志 -----
    
    async def log_audit(
        self,
        action: str,
        user_id: Optional[uuid.UUID] = None,
        resource_type: str = None,
        resource_id: str = None,
        old_values: Dict = None,
        new_values: Dict = None,
        ip_address: str = None,
        user_agent: str = None,
        success: bool = True,
        error_message: str = None
    ) -> AuditLog:
        log = AuditLog(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            old_values=old_values,
            new_values=new_values,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            error_message=error_message,
        )
        self.session.add(log)
        await self.session.commit()
        return log
    
    async def get_audit_logs(
        self,
        user_id: uuid.UUID = None,
        action: str = None,
        start_date: datetime = None,
        end_date: datetime = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[AuditLog]:
        query = select(AuditLog)
        
        if user_id:
            query = query.where(AuditLog.user_id == user_id)
        if action:
            query = query.where(AuditLog.action == action)
        if start_date:
            query = query.where(AuditLog.created_at >= start_date)
        if end_date:
            query = query.where(AuditLog.created_at <= end_date)
        
        query = query.order_by(AuditLog.created_at.desc()).limit(limit).offset(offset)
        result = await self.session.execute(query)
        return result.scalars().all()


# ========================================
# 5. 依赖注入
# ========================================

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    # 实际项目中从配置创建 engine
    engine = create_async_engine(AUTH_SETTINGS.database_url, echo=False)
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        yield session


async def get_redis() -> redis.Redis:
    global redis_client
    if redis_client is None:
        redis_client = redis.from_url(AUTH_SETTINGS.redis_url, decode_responses=True)
    return redis_client


async def get_auth_service(db: AsyncSession = Depends(get_db), redis_client: redis.Redis = Depends(get_redis)) -> AuthService:
    return AuthService(db, redis_client)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: AuthService = Depends(get_auth_service)
) -> User:
    """获取当前用户 (JWT Bearer Token)"""
    if not credentials:
        raise HTTPException(status_code=401, detail="需要认证")
    
    token_data = auth_service.decode_token(credentials.credentials)
    user = await auth_service.get_user(uuid.UUID(token_data.sub))
    
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="用户不存在或已禁用")
    
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="用户未激活")
    return current_user


async def get_current_superuser(current_user: User = Depends(get_current_active_user)) -> User:
    if not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="权限不足")
    return current_user


async def get_api_key_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: AuthService = Depends(get_auth_service)
) -> User:
    """API Key 认证"""
    if not credentials or not credentials.credentials.startswith("bs_"):
        raise HTTPException(status_code=401, detail="无效的 API Key")
    
    api_key = await auth_service.verify_api_key(credentials.credentials)
    if not api_key:
        raise HTTPException(status_code=401, detail="无效或过期的 API Key")
    
    user = await auth_service.get_user(api_key.user_id)
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="用户不存在或已禁用")
    
    return user


# ========================================
# 6. 权限装饰器
# ========================================

def require_permissions(*permissions: str):
    """要求特定权限"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, current_user: User = Depends(get_current_active_user), **kwargs):
            user_permissions = set()
            for role in current_user.roles:
                user_permissions.update(role.permissions)
            
            if not all(p in user_permissions for p in permissions):
                raise HTTPException(status_code=403, detail=f"缺少权限: {', '.join(permissions)}")
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def require_roles(*roles: str):
    """要求特定角色"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, current_user: User = Depends(get_current_active_user), **kwargs):
            user_roles = {role.name for role in current_user.roles}
            if not any(r in user_roles for r in roles):
                raise HTTPException(status_code=403, detail=f"需要角色: {', '.join(roles)}")
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def rate_limit(requests: int = 100, window: int = 60):
    """速率限制装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            # 这里简化处理，实际应用应使用 slowapi
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator


# ========================================
# 7. FastAPI 应用初始化
# ========================================

def create_auth_app() -> FastAPI:
    app = FastAPI(title="BreakShell Auth Service", version="0.9.0")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    if 'limiter' in globals():
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    
    return app


# ========================================
# 7. 路由定义
# ========================================

def create_auth_router() -> APIRouter:
    router = APIRouter(prefix="/auth", tags=["Authentication"])
    
    @router.post("/register", response_model=UserResponse, status_code=201)
    async def register(user_data: UserCreate, auth_service: AuthService = Depends(get_auth_service)):
        user = await auth_service.create_user(user_data)
        return UserResponse.model_validate(user)
    
    @router.post("/login", response_model=Token)
    async def login(
        request: Request,
        login_data: LoginRequest,
        auth_service: AuthService = Depends(get_auth_service)
    ):
        user = await auth_service.authenticate(login_data.username, login_data.password)
        if not user:
            # 记录审计日志
            await auth_service.log_audit(
                action="login_failed",
                resource_type="user",
                resource_id=login_data.username,
                ip_address=request.client.host,
                user_agent=request.headers.get("user-agent"),
                success=False,
                error_message="Invalid credentials"
            )
            raise HTTPException(status_code=401, detail="用户名或密码错误")
        
        access_token = auth_service.create_access_token(
            user_id=user.id,
            roles=[role.name for role in user.roles]
        )
        refresh_token = auth_service.create_refresh_token(user.id)
        
        # 记录审计日志
        await auth_service.log_audit(
            action="login_success",
            user_id=user.id,
            ip_address=request.client.host,
            user_agent=request.headers.get("user-agent"),
            success=True
        )
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=AUTH_SETTINGS.access_token_expire_minutes * 60,
        )
    
    @router.post("/refresh", response_model=Token)
    async def refresh_token(refresh_data: RefreshTokenRequest, auth_service: AuthService = Depends(get_auth_service)):
        user_id = auth_service.verify_refresh_token(refresh_data.refresh_token)
        if not user_id:
            raise HTTPException(status_code=401, detail="刷新令牌无效或已过期")
        
        user = await auth_service.get_user(user_id)
        if not user or not user.is_active:
            raise HTTPException(status_code=401, detail="用户不存在或已禁用")
        
        access_token = auth_service.create_access_token(
            user_id=user.id,
            roles=[role.name for role in user.roles]
        )
        new_refresh_token = auth_service.create_refresh_token(user.id)
        
        return Token(
            access_token=access_token,
            refresh_token=new_refresh_token,
            expires_in=AUTH_SETTINGS.access_token_expire_minutes * 60,
        )
    
    @router.post("/logout")
    async def logout(
        credentials: HTTPAuthorizationCredentials = Depends(security),
        auth_service: AuthService = Depends(get_auth_service),
        current_user: User = Depends(get_current_active_user)
    ):
        # 撤销当前会话（简化：撤销所有会话）
        await auth_service.revoke_all_sessions(current_user.id)
        return {"message": "登出成功"}
    
    @router.get("/me", response_model=UserResponse)
    async def get_me(current_user: User = Depends(get_current_active_user)):
        return UserResponse.model_validate(current_user)
    
    @router.patch("/me", response_model=UserResponse)
    async def update_me(
        user_data: UserUpdate,
        auth_service: AuthService = Depends(get_auth_service),
        current_user: User = Depends(get_current_active_user)
    ):
        user = await auth_service.update_user(current_user.id, user_data)
        return UserResponse.model_validate(user)
    
    @router.post("/api-keys", response_model=APIKeyResponse, status_code=201)
    async def create_api_key(
        key_data: APIKeyCreate,
        auth_service: AuthService = Depends(get_auth_service),
        current_user: User = Depends(get_current_active_user)
    ):
        api_key, full_key = await auth_service.create_api_key(current_user.id, key_data)
        # 这里需要处理返回完整 key
        return APIKeyResponse(
            id=api_key.id,
            name=api_key.name,
            key_prefix=api_key.key_prefix,
            key=f"{api_key.key_prefix}{secrets.token_urlsafe(32)}",  # 简化
            permissions=api_key.permissions,
            rate_limit=api_key.rate_limit,
            expires_at=api_key.expires_at,
            created_at=api_key.created_at,
        )
    
    @router.get("/api-keys", response_model=List[APIKeyResponse])
    async def list_api_keys(
        auth_service: AuthService = Depends(get_auth_service),
        current_user: User = Depends(get_current_active_user)
    ):
        result = await auth_service.session.execute(
            select(APIKey).where(APIKey.user_id == current_user.id, APIKey.is_active == True)
        )
        keys = result.scalars().all()
        return [APIKeyResponse.model_validate(k) for k in keys]
    
    @router.delete("/api-keys/{key_id}")
    async def revoke_api_key(
        key_id: uuid.UUID,
        auth_service: AuthService = Depends(get_auth_service),
        current_user: User = Depends(get_current_active_user)
    ):
        success = await auth_service.revoke_api_key(key_id, current_user.id)
        if not success:
            raise HTTPException(status_code=404, detail="API Key 不存在")
        return {"message": "API Key 已撤销"}
    
    @router.get("/audit-logs", response_model=List[AuditLogResponse])
    async def get_audit_logs(
        action: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
        auth_service: AuthService = Depends(get_auth_service),
        current_user: User = Depends(get_current_active_user)
    ):
        logs = await auth_service.get_audit_logs(
            user_id=current_user.id,
            action=action,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset
        )
        return [AuditLogResponse.model_validate(log) for log in logs]
    
    # 管理员路由
    admin_router = APIRouter(prefix="/admin", tags=["Admin"], dependencies=[Depends(get_current_superuser)])
    
    @admin_router.post("/roles", response_model=RoleResponse, status_code=201)
    async def create_role(role_data: RoleCreate, auth_service: AuthService = Depends(get_auth_service)):
        role = await auth_service.create_role(role_data)
        return RoleResponse.model_validate(role)
    
    @admin_router.get("/roles", response_model=List[RoleResponse])
    async def list_roles(auth_service: AuthService = Depends(get_auth_service)):
        result = await auth_service.session.execute(select(Role))
        return [RoleResponse.model_validate(r) for r in result.scalars().all()]
    
    @admin_router.post("/users/{user_id}/roles/{role_name}")
    async def assign_role(
        user_id: uuid.UUID,
        role_name: str,
        auth_service: AuthService = Depends(get_auth_service)
    ):
        success = await auth_service.assign_role(user_id, role_name, uuid.UUID("00000000-0000-0000-0000-000000000000"))
        if not success:
            raise HTTPException(status_code=404, detail="用户或角色不存在")
        return {"message": "角色分配成功"}
    
    @admin_router.delete("/users/{user_id}/roles/{role_name}")
    async def revoke_role(
        user_id: uuid.UUID,
        role_name: str,
        auth_service: AuthService = Depends(get_auth_service)
    ):
        success = await auth_service.revoke_role(user_id, role_name)
        if not success:
            raise HTTPException(status_code=404, detail="用户或角色不存在")
        return {"message": "角色撤销成功"}
    
    @admin_router.get("/users")
    async def list_users(
        limit: int = 50,
        offset: int = 0,
        auth_service: AuthService = Depends(get_auth_service)
    ):
        result = await auth_service.session.execute(
            select(User).offset(offset).limit(limit)
        )
        users = result.scalars().all()
        return [UserResponse.model_validate(u) for u in users]
    
    router.include_router(admin_router)
    return router


# ========================================
# 8. 主应用入口
# ========================================

def create_app() -> FastAPI:
    app = FastAPI(
        title="BreakShell Auth Service",
        description="认证授权微服务",
        version="0.9.0",
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    if 'limiter' in globals():
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    
    # 包含路由
    app.include_router(create_auth_router())
    
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "version": "0.9.0"}
    
    return app


# ========================================
# 9. 导出
# ========================================

__all__ = [
    "AuthSettings",
    "AUTH_SETTINGS",
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
    "rate_limit",
    "pwd_context",
    "oauth2_scheme",
    "security",
    "limiter",
]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("breakshell.auth:create_app", host="0.0.0.0", port=8001, reload=True)