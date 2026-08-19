# -*- coding: utf-8 -*-
"""
BreakShell 合规审计日志系统
=============================
不可篡改的审计日志存储，支持：
- 追加-only 写入
- 加密存储
- 哈希链完整性校验
- WORM (Write Once Read Many) 合规
- 自动归档和生命周期管理
"""

from __future__ import annotations

import os
import json
import hashlib
import hmac
import base64
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Generator
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import logging


# ========================================
# 1. 核心数据结构
# ========================================

class AuditEventType(Enum):
    """审计事件类型"""
    # 认证授权
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILED = "login_failed"
    LOGOUT = "logout"
    TOKEN_REFRESH = "token_refresh"
    PASSWORD_CHANGE = "password_change"
    
    # 权限变更
    ROLE_ASSIGNED = "role_assigned"
    ROLE_REVOKED = "role_revoked"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_REVOKED = "permission_revoked"
    
    # 数据访问
    DATA_READ = "data_read"
    DATA_WRITE = "data_write"
    DATA_DELETE = "data_delete"
    DATA_EXPORT = "data_export"
    
    # 配置变更
    CONFIG_CHANGED = "config_changed"
    SETTINGS_UPDATED = "settings_updated"
    
    # 模型操作
    MODEL_TRAINED = "model_trained"
    MODEL_DEPLOYED = "model_deployed"
    MODEL_ROLLED_BACK = "model_rolled_back"
    
    # 金融交易
    TRADE_EXECUTED = "trade_executed"
    ORDER_PLACED = "order_placed"
    ORDER_CANCELLED = "order_cancelled"
    RISK_LIMIT_BREACHED = "risk_limit_breached"
    
    # 系统事件
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    ERROR_OCCURRED = "error_occurred"
    
    # 安全事件
    SECURITY_VIOLATION = "security_violation"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"


class AuditSeverity(Enum):
    """审计严重级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditLogEntry:
    """审计日志条目"""
    # 基础字段
    id: str                    # 唯一标识 (UUID)
    timestamp: str             # ISO 8601 时间戳
    event_type: str            # 事件类型
    severity: str              # 严重级别
    
    # 主体信息
    user_id: Optional[str]     # 用户 ID
    session_id: Optional[str]  # 会话 ID
    api_key_id: Optional[str]  # API Key ID
    ip_address: Optional[str]  # 客户端 IP
    user_agent: Optional[str]  # User Agent
    
    # 资源信息
    resource_type: Optional[str]   # 资源类型
    resource_id: Optional[str]     # 资源 ID
    resource_name: Optional[str]   # 资源名称
    
    # 操作详情
    action: str                # 执行的动作
    description: str           # 人类可读描述
    old_values: Optional[Dict]     # 变更前值
    new_values: Optional[Dict]     # 变更后值
    
    # 结果
    success: bool              # 是否成功
    error_code: Optional[str]    # 错误码
    error_message: Optional[str] # 错误信息
    
    # 合规字段
    compliance_tags: List[str]   # 合规标签 (GDPR, SOX, PCI-DSS 等)
    retention_years: int         # 保留年限
    
    # 完整性字段
    previous_hash: str           # 前一条日志的哈希
    current_hash: str            # 当前日志哈希
    sequence_number: int         # 序列号
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AuditLogEntry':
        return cls(**data)


# ========================================
# 2. 哈希链完整性管理
# ========================================

class HashChain:
    """哈希链：保证日志不可篡改"""
    
    def __init__(self, secret_key: bytes = None):
        self.secret_key = secret_key or os.urandom(32)
        self._lock = threading.Lock()
        self._last_hash = "0" * 64  # 初始哈希
        self._sequence = 0
    
    def compute_hash(self, entry: AuditLogEntry) -> str:
        """计算日志条目哈希"""
        # 构造哈希输入
        hash_input = f"{entry.sequence_number}|{entry.previous_hash}|{entry.timestamp}|{entry.event_type}|{entry.user_id or ''}|{entry.action}|{entry.success}|{json.dumps(entry.old_values, sort_keys=True) if entry.old_values else ''}|{json.dumps(entry.new_values, sort_keys=True) if entry.new_values else ''}"
        
        # HMAC-SHA256
        h = hmac.new(self.secret_key, hash_input.encode(), hashlib.sha256)
        return h.hexdigest()
    
    def next_hash(self, entry: AuditLogEntry) -> str:
        """生成下一个哈希并更新状态"""
        with self._lock:
            entry.sequence_number = self._sequence + 1
            entry.previous_hash = self._last_hash
            entry.current_hash = self.compute_hash(entry)
            self._last_hash = entry.current_hash
            self._sequence = entry.sequence_number
            return entry.current_hash
    
    def verify_chain(self, entries: List[AuditLogEntry]) -> tuple[bool, Optional[int]]:
        """验证哈希链完整性"""
        expected_prev = "0" * 64
        expected_seq = 0
        
        for i, entry in enumerate(entries):
            # 检查序列号
            if entry.sequence_number != expected_seq + 1:
                return False, i
            
            # 检查前置哈希
            if entry.previous_hash != expected_prev:
                return False, i
            
            # 重新计算哈希
            computed = self.compute_hash(entry)
            if computed != entry.current_hash:
                return False, i
            
            expected_prev = entry.current_hash
            expected_seq = entry.sequence_number
        
        return True, None


# ========================================
# 3. 加密存储
# ========================================

class EncryptedStorage:
    """加密存储层"""
    
    def __init__(self, password: str = None, salt: bytes = None):
        self.salt = salt or os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive((password or os.environ.get("AUDIT_ENCRYPTION_KEY", "default-key-change-me")).encode()))
        self.cipher = Fernet(key)
    
    def encrypt(self, data: str) -> str:
        """加密数据"""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted: str) -> str:
        """解密数据"""
        return self.cipher.decrypt(encrypted.encode()).decode()
    
    def encrypt_dict(self, data: Dict) -> str:
        """加密字典"""
        return self.encrypt(json.dumps(data, sort_keys=True))
    
    def decrypt_dict(self, encrypted: str) -> Dict:
        """解密为字典"""
        return json.loads(self.decrypt(encrypted))


# ========================================
# 4. 审计日志存储后端
# ========================================

class AuditStorageBackend:
    """审计日志存储后端抽象"""
    
    def append(self, entry: AuditLogEntry) -> None:
        raise NotImplementedError
    
    def query(self, 
              start_time: datetime = None,
              end_time: datetime = None,
              event_type: str = None,
              user_id: str = None,
              severity: str = None,
              limit: int = 1000,
              offset: int = 0) -> List[AuditLogEntry]:
        raise NotImplementedError
    
    def verify_integrity(self) -> tuple[bool, Optional[int]]:
        raise NotImplementedError
    
    def export(self, 
               start_time: datetime,
               end_time: datetime,
               format: str = "json") -> str:
        raise NotImplementedError


class SQLiteAuditStorage(AuditStorageBackend):
    """SQLite 审计存储（带加密和哈希链）"""
    
    def __init__(self, db_path: str, encryption_key: str = None):
        self.db_path = db_path
        self.hash_chain = HashChain()
        self.encryption = EncryptedStorage(encryption_key)
        self._lock = threading.Lock()
        self._init_db()
    
    def _init_db(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id TEXT PRIMARY KEY,
                    sequence_number INTEGER UNIQUE NOT NULL,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    user_id TEXT,
                    session_id TEXT,
                    api_key_id TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    resource_type TEXT,
                    resource_id TEXT,
                    resource_name TEXT,
                    action TEXT NOT NULL,
                    description TEXT NOT NULL,
                    old_values TEXT,
                    new_values TEXT,
                    success INTEGER NOT NULL,
                    error_code TEXT,
                    error_message TEXT,
                    compliance_tags TEXT,
                    retention_years INTEGER DEFAULT 7,
                    previous_hash TEXT NOT NULL,
                    current_hash TEXT NOT NULL,
                    encrypted_data TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_logs(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON audit_logs(event_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON audit_logs(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sequence ON audit_logs(sequence_number)")
            conn.commit()
    
    def append(self, entry: AuditLogEntry) -> None:
        """追加日志（不可篡改）"""
        # 更新哈希链
        self.hash_chain.next_hash(entry)
        
        # 加密敏感数据
        encrypted_data = None
        if entry.old_values or entry.new_values:
            sensitive = {
                "old_values": entry.old_values,
                "new_values": entry.new_values,
            }
            encrypted_data = self.encryption.encrypt_dict(sensitive)
            entry.old_values = None
            entry.new_values = None
        
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO audit_logs (
                        id, sequence_number, timestamp, event_type, severity,
                        user_id, session_id, api_key_id, ip_address, user_agent,
                        resource_type, resource_id, resource_name,
                        action, description, old_values, new_values,
                        success, error_code, error_message,
                        compliance_tags, retention_years,
                        previous_hash, current_hash, encrypted_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.id, entry.sequence_number, entry.timestamp, entry.event_type, entry.severity,
                    entry.user_id, entry.session_id, entry.api_key_id, entry.ip_address, entry.user_agent,
                    entry.resource_type, entry.resource_id, entry.resource_name,
                    entry.action, entry.description, 
                    json.dumps(entry.old_values) if entry.old_values else None,
                    json.dumps(entry.new_values) if entry.new_values else None,
                    1 if entry.success else 0, entry.error_code, entry.error_message,
                    json.dumps(entry.compliance_tags), entry.retention_years,
                    entry.previous_hash, entry.current_hash, encrypted_data
                ))
                conn.commit()
    
    def query(self, 
              start_time: datetime = None,
              end_time: datetime = None,
              event_type: str = None,
              user_id: str = None,
              severity: str = None,
              limit: int = 1000,
              offset: int = 0) -> List[AuditLogEntry]:
        """查询日志"""
        conditions = []
        params = []
        
        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time.isoformat())
        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time.isoformat())
        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if severity:
            conditions.append("severity = ?")
            params.append(severity)
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(f"""
                SELECT * FROM audit_logs 
                {where_clause}
                ORDER BY sequence_number DESC
                LIMIT ? OFFSET ?
            """, params + [limit, offset])
            
            entries = []
            for row in cursor.fetchall():
                entry = self._row_to_entry(row)
                entries.append(entry)
            return entries
    
    def _row_to_entry(self, row: sqlite3.Row) -> AuditLogEntry:
        """行转实体"""
        # 解密数据
        old_values = None
        new_values = None
        if row["encrypted_data"]:
            try:
                decrypted = self.encryption.decrypt_dict(row["encrypted_data"])
                old_values = decrypted.get("old_values")
                new_values = decrypted.get("new_values")
            except:
                pass
        elif row["old_values"] or row["new_values"]:
            old_values = json.loads(row["old_values"]) if row["old_values"] else None
            new_values = json.loads(row["new_values"]) if row["new_values"] else None
        
        return AuditLogEntry(
            id=row["id"],
            timestamp=row["timestamp"],
            event_type=row["event_type"],
            severity=row["severity"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            api_key_id=row["api_key_id"],
            ip_address=row["ip_address"],
            user_agent=row["user_agent"],
            resource_type=row["resource_type"],
            resource_id=row["resource_id"],
            resource_name=row["resource_name"],
            action=row["action"],
            description=row["description"],
            old_values=old_values,
            new_values=new_values,
            success=bool(row["success"]),
            error_code=row["error_code"],
            error_message=row["error_message"],
            compliance_tags=json.loads(row["compliance_tags"]) if row["compliance_tags"] else [],
            retention_years=row["retention_years"],
            previous_hash=row["previous_hash"],
            current_hash=row["current_hash"],
            sequence_number=row["sequence_number"],
        )
    
    def verify_integrity(self) -> tuple[bool, Optional[int]]:
        """验证完整性"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM audit_logs ORDER BY sequence_number ASC")
            entries = [self._row_to_entry(row) for row in cursor.fetchall()]
            return self.hash_chain.verify_chain(entries)
    
    def export(self, 
               start_time: datetime,
               end_time: datetime,
               format: str = "json") -> str:
        """导出日志"""
        entries = self.query(start_time=start_time, end_time=end_time, limit=100000)
        
        if format == "json":
            return json.dumps([e.to_dict() for e in entries], indent=2, ensure_ascii=False)
        elif format == "csv":
            import csv
            import io
            output = io.StringIO()
            if entries:
                writer = csv.DictWriter(output, fieldnames=entries[0].to_dict().keys())
                writer.writeheader()
                for e in entries:
                    writer.writerow(e.to_dict())
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format}")


# ========================================
# 5. WORM 存储（不可篡改）
# ========================================

class WORMAuditStorage:
    """WORM (Write Once Read Many) 审计存储"""
    
    def __init__(self, base_path: str, encryption_key: str = None):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # 当前活跃文件
        self.current_file = self.base_path / f"audit_{datetime.now().strftime('%Y%m')}.log"
        self.current_file_handle = None
        
        # SQLite 索引
        self.index_db = self.base_path / "audit_index.db"
        self.sqlite_storage = SQLiteAuditStorage(str(self.index_db))
        
        # 文件锁
        self._file_lock = threading.Lock()
        self._rotation_lock = threading.Lock()
    
    def _get_current_file(self):
        """获取当前月份的文件句柄"""
        expected_file = self.base_path / f"audit_{datetime.now().strftime('%Y%m')}.log"
        
        if self.current_file != expected_file:
            with self._rotation_lock:
                if self.current_file_handle:
                    self.current_file_handle.close()
                self.current_file = expected_file
                self.current_file_handle = open(self.current_file, "a", encoding="utf-8")
        
        return self.current_file_handle
    
    def append(self, entry: AuditLogEntry) -> None:
        """追加到 WORM 存储"""
        # 同时写入 SQLite 索引
        self.sqlite_storage.append(entry)
        
        # 写入追加-only 文件
        line = json.dumps(entry.to_dict(), ensure_ascii=False)
        
        with self._file_lock:
            f = self._get_current_file()
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
    
    def query(self, **kwargs) -> List[AuditLogEntry]:
        """委托给 SQLite 索引"""
        return self.sqlite_storage.query(**kwargs)
    
    def verify_integrity(self) -> tuple[bool, Optional[int]]:
        """验证完整性"""
        return self.sqlite_storage.verify_integrity()
    
    def export(self, start_time: datetime, end_time: datetime, format: str = "json") -> str:
        return self.sqlite_storage.export(start_time, end_time, format)
    
    def close(self):
        """关闭文件句柄"""
        if self.current_file_handle:
            self.current_file_handle.close()


# ========================================
# 6. 合规审计管理器
# ========================================

class ComplianceAuditManager:
    """合规审计管理器"""
    
    def __init__(self, storage: AuditStorageBackend = None):
        self.storage = storage or SQLiteAuditStorage("audit.db")
        self._buffer = []
        self._buffer_size = 100
        self._flush_interval = 5  # 秒
        self._last_flush = time.time()
        self._lock = threading.Lock()
        self._start_background_flush()
    
    def _start_background_flush(self):
        """启动后台刷新线程"""
        def flush_worker():
            while True:
                time.sleep(self._flush_interval)
                self._flush_buffer()
        
        thread = threading.Thread(target=flush_worker, daemon=True)
        thread.start()
    
    def _flush_buffer(self):
        """刷新缓冲区"""
        with self._lock:
            if self._buffer:
                for entry in self._buffer:
                    self.storage.append(entry)
                self._buffer.clear()
                self._last_flush = time.time()
    
    def log(self, 
            event_type: AuditEventType,
            action: str,
            description: str,
            user_id: str = None,
            session_id: str = None,
            api_key_id: str = None,
            ip_address: str = None,
            user_agent: str = None,
            resource_type: str = None,
            resource_id: str = None,
            resource_name: str = None,
            old_values: Dict = None,
            new_values: Dict = None,
            success: bool = True,
            error_code: str = None,
            error_message: str = None,
            severity: AuditSeverity = AuditSeverity.INFO,
            compliance_tags: List[str] = None,
            retention_years: int = 7) -> AuditLogEntry:
        """记录审计日志"""
        
        entry = AuditLogEntry(
            id=hashlib.sha256(f"{time.time()}{os.urandom(8).hex()}".encode()).hexdigest()[:32],
            timestamp=datetime.utcnow().isoformat() + "Z",
            event_type=event_type.value,
            severity=severity.value,
            user_id=user_id,
            session_id=session_id,
            api_key_id=api_key_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource_type=resource_type,
            resource_id=resource_id,
            resource_name=resource_name,
            action=action,
            description=description,
            old_values=old_values,
            new_values=new_values,
            success=success,
            error_code=error_code,
            error_message=error_message,
            compliance_tags=compliance_tags or [],
            retention_years=retention_years,
            previous_hash="",  # 将由存储层填充
            current_hash="",   # 将由存储层填充
            sequence_number=0, # 将由存储层填充
        )
        
        # 添加到缓冲区
        with self._lock:
            self._buffer.append(entry)
            if len(self._buffer) >= self._buffer_size:
                self._flush_buffer()
        
        return entry
    
    def log_sync(self, **kwargs) -> AuditLogEntry:
        """同步记录（立即写入）"""
        entry = self.log(**kwargs)
        self._flush_buffer()
        return entry
    
    def query(self, **kwargs) -> List[AuditLogEntry]:
        """查询日志"""
        return self.storage.query(**kwargs)
    
    def verify_integrity(self) -> tuple[bool, Optional[int]]:
        """验证完整性"""
        return self.storage.verify_integrity()
    
    def export(self, start_time: datetime, end_time: datetime, format: str = "json") -> str:
        """导出日志"""
        return self.storage.export(start_time, end_time, format)
    
    def generate_compliance_report(self, 
                                   start_time: datetime,
                                   end_time: datetime,
                                   framework: str = "SOX") -> Dict:
        """生成合规报告"""
        entries = self.storage.query(
            start_time=start_time,
            end_time=end_time
        )
        
        report = {
            "framework": framework,
            "period_start": start_time.isoformat(),
            "period_end": end_time.isoformat(),
            "total_events": len(entries),
            "by_type": {},
            "by_severity": {},
            "by_user": {},
            "failed_events": 0,
            "security_incidents": 0,
            "data_access_events": 0,
            "config_changes": 0,
            "integrity_verified": True,
        }
        
        for entry in entries:
            # 按类型统计
            report["by_type"][entry.event_type] = report["by_type"].get(entry.event_type, 0) + 1
            
            # 按严重度统计
            report["by_severity"][entry.severity] = report["by_severity"].get(entry.severity, 0) + 1
            
            # 按用户统计
            if entry.user_id:
                report["by_user"][entry.user_id] = report["by_user"].get(entry.user_id, 0) + 1
            
            # 失败事件
            if not entry.success:
                report["failed_events"] += 1
            
            # 安全事件
            if entry.event_type in [e.value for e in [AuditEventType.SECURITY_VIOLATION, AuditEventType.SUSPICIOUS_ACTIVITY, AuditEventType.RATE_LIMIT_EXCEEDED]]:
                report["security_incidents"] += 1
            
            # 数据访问
            if entry.event_type in [e.value for e in [AuditEventType.DATA_READ, AuditEventType.DATA_WRITE, AuditEventType.DATA_DELETE, AuditEventType.DATA_EXPORT]]:
                report["data_access_events"] += 1
            
            # 配置变更
            if entry.event_type in [e.value for e in [AuditEventType.CONFIG_CHANGED, AuditEventType.SETTINGS_UPDATED]]:
                report["config_changes"] += 1
        
        # 验证完整性
        verified, error_idx = self.storage.verify_integrity()
        report["integrity_verified"] = verified
        if not verified:
            report["integrity_error_at"] = error_idx
        
        return report


# ========================================
# 7. 自动归档与生命周期
# ========================================

class AuditLifecycleManager:
    """审计日志生命周期管理"""
    
    def __init__(self, storage: WORMAuditStorage, archive_path: str):
        self.storage = storage
        self.archive_path = Path(archive_path)
        self.archive_path.mkdir(parents=True, exist_ok=True)
    
    def archive_old_logs(self, older_than_days: int = 90) -> int:
        """归档旧日志"""
        cutoff = datetime.utcnow() - timedelta(days=older_than_days)
        
        # 查询旧日志
        entries = self.storage.query(end_time=cutoff, limit=100000)
        
        if not entries:
            return 0
        
        # 按月分组归档
        by_month = {}
        for entry in entries:
            dt = datetime.fromisoformat(entry.timestamp.replace("Z", "+00:00"))
            month_key = dt.strftime("%Y%m")
            if month_key not in by_month:
                by_month[month_key] = []
            by_month[month_key].append(entry)
        
        archived_count = 0
        for month, month_entries in by_month.items():
            archive_file = self.archive_path / f"audit_archive_{month}.json.gz"
            
            # 压缩归档
            import gzip
            with gzip.open(archive_file, "wt", encoding="utf-8") as f:
                for entry in month_entries:
                    f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
            
            archived_count += len(month_entries)
        
        return archived_count
    
    def cleanup_expired(self, max_retention_years: int = 7) -> int:
        """清理过期日志（根据保留策略）"""
        cutoff = datetime.utcnow() - timedelta(days=max_retention_years * 365)
        
        # 注意：WORM 存储通常不允许删除
        # 这里仅标记，实际删除需人工审批
        expired_entries = self.storage.query(end_time=cutoff, limit=100000)
        return len(expired_entries)


# ========================================
# 8. 便捷函数
# ========================================

def create_audit_manager(storage_type: str = "sqlite", 
                         db_path: str = "audit.db",
                         worm_path: str = None,
                         encryption_key: str = None) -> ComplianceAuditManager:
    """创建审计管理器"""
    
    if storage_type == "worm" and worm_path:
        storage = WORMAuditStorage(worm_path, encryption_key)
    else:
        storage = SQLiteAuditStorage(db_path, encryption_key)
    
    return ComplianceAuditManager(storage)


def create_audit_entry(event_type: AuditEventType,
                       action: str,
                       description: str,
                       **kwargs) -> AuditLogEntry:
    """创建审计日志条目（供手动构建）"""
    return AuditLogEntry(
        id=hashlib.sha256(f"{time.time()}{os.urandom(8).hex()}".encode()).hexdigest()[:32],
        timestamp=datetime.utcnow().isoformat() + "Z",
        event_type=event_type.value,
        severity=kwargs.get("severity", AuditSeverity.INFO).value,
        user_id=kwargs.get("user_id"),
        session_id=kwargs.get("session_id"),
        api_key_id=kwargs.get("api_key_id"),
        ip_address=kwargs.get("ip_address"),
        user_agent=kwargs.get("user_agent"),
        resource_type=kwargs.get("resource_type"),
        resource_id=kwargs.get("resource_id"),
        resource_name=kwargs.get("resource_name"),
        action=action,
        description=description,
        old_values=kwargs.get("old_values"),
        new_values=kwargs.get("new_values"),
        success=kwargs.get("success", True),
        error_code=kwargs.get("error_code"),
        error_message=kwargs.get("error_message"),
        compliance_tags=kwargs.get("compliance_tags", []),
        retention_years=kwargs.get("retention_years", 7),
        previous_hash="",
        current_hash="",
        sequence_number=0,
    )


# ========================================
# 9. FastAPI 集成中间件
# ========================================

def create_audit_middleware(audit_manager: ComplianceAuditManager):
    """创建 FastAPI 审计中间件"""
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import Response
    
    class AuditMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            start_time = time.time()
            
            # 获取用户信息
            user_id = getattr(request.state, "user_id", None)
            session_id = request.headers.get("X-Session-ID")
            api_key_id = request.headers.get("X-API-Key-ID")
            ip_address = request.client.host if request.client else None
            user_agent = request.headers.get("User-Agent")
            
            # 执行请求
            response = await call_next(request)
            
            # 记录审计
            duration_ms = (time.time() - start_time) * 1000
            
            audit_manager.log_sync(
                event_type=AuditEventType.DATA_READ if request.method == "GET" else AuditEventType.DATA_WRITE,
                action=f"{request.method} {request.url.path}",
                description=f"API request to {request.url.path}",
                user_id=user_id,
                session_id=session_id,
                api_key_id=api_key_id,
                ip_address=ip_address,
                user_agent=user_agent,
                resource_type="api_endpoint",
                resource_id=request.url.path,
                success=response.status_code < 400,
                error_code=str(response.status_code) if response.status_code >= 400 else None,
                severity=AuditSeverity.INFO if response.status_code < 400 else AuditSeverity.WARNING,
                compliance_tags=["api_access"],
            )
            
            # 添加响应头
            response.headers["X-Audit-ID"] = hashlib.sha256(f"{time.time()}".encode()).hexdigest()[:16]
            
            return response
    
    return AuditMiddleware


# ========================================
# 10. 导出
# ========================================

__all__ = [
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

if __name__ == "__main__":
    # 测试
    print("测试合规审计系统...")
    
    # 创建管理器
    audit = create_audit_manager("sqlite", "test_audit.db")
    
    # 记录一些日志
    audit.log_sync(
        event_type=AuditEventType.LOGIN_SUCCESS,
        action="user_login",
        description="用户登录成功",
        user_id="user_123",
        ip_address="192.168.1.1",
        success=True,
        compliance_tags=["SOX", "GDPR"],
    )
    
    audit.log_sync(
        event_type=AuditEventType.TRADE_EXECUTED,
        action="execute_trade",
        description="执行买入订单",
        user_id="user_123",
        resource_type="trade",
        resource_id="trade_456",
        old_values={"position": 0},
        new_values={"position": 100, "symbol": "BTC"},
        success=True,
        compliance_tags=["SOX", "MiFID-II"],
        retention_years=7,
    )
    
    audit.log_sync(
        event_type=AuditEventType.RISK_LIMIT_BREACHED,
        action="risk_check",
        description="VaR 超限告警",
        severity=AuditSeverity.CRITICAL,
        user_id="system",
        resource_type="risk_limit",
        resource_id="var_95",
        success=False,
        error_code="VAR_EXCEEDED",
        error_message="VaR 95% 超过限制",
        compliance_tags=["Basel-III", "SOX"],
    )
    
    # 查询
    entries = audit.query(limit=10)
    print(f"查询到 {len(entries)} 条日志")
    
    # 验证完整性
    verified, error_idx = audit.verify_integrity()
    print(f"完整性验证: {'通过' if verified else f'失败 at {error_idx}'}")
    
    # 合规报告
    from datetime import datetime, timedelta
    report = audit.generate_compliance_report(
        start_time=datetime.utcnow() - timedelta(days=1),
        end_time=datetime.utcnow(),
        framework="SOX"
    )
    print(f"合规报告: {json.dumps(report, indent=2, ensure_ascii=False)}")
    
    # 导出
    exported = audit.export(
        start_time=datetime.utcnow() - timedelta(days=1),
        end_time=datetime.utcnow(),
        format="json"
    )
    print(f"导出大小: {len(exported)} 字符")
    
    print("测试完成!")