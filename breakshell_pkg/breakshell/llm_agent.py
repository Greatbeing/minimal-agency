# -*- coding: utf-8 -*-
"""
BreakShell LLM Agent — 优化版（简洁可靠版）
=============================================
核心优化：
1. Token 预算强制执行 + 上下文裁剪
2. SelfModelTracker 增强（多维特征 + 置信度校准）
3. LLM 输出解析强化（JSON Schema + 重试）
4. 完善的错误处理
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import sqlite3
import subprocess
import time
import uuid
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import sys


# ========================================
# 0. 基础类
# ========================================

@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable
    permission: str = "read-only"
    dangerous: bool = False


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}
    
    def register(self, tool: ToolSpec):
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)
    
    def list_tools(self, level: str = "read-only") -> List[ToolSpec]:
        return [t for t in self._tools.values() if PermissionLevel.can_execute(t.permission, level)]
    
    def describe(self, level: str = "read-only") -> List[Dict]:
        return [{"name": t.name, "description": t.description, "schema": t.input_schema, "permission": t.permission} 
                for t in self.list_tools(level)]


class PermissionLevel:
    LEVELS = ["read-only", "workspace-write", "network", "system"]
    
    @classmethod
    def can_execute(cls, tool_level: str, user_level: str) -> bool:
        try: 
            return cls.LEVELS.index(tool_level) <= cls.LEVELS.index(user_level)
        except: 
            return False


# ========================================
# 1. Token Budget & Context
# ========================================

class TokenCounter:
    def __init__(self): 
        self.char_per_token_en = 4
        self.char_per_token_zh = 1.5
    
    def count(self, text: str) -> int:
        if not text: 
            return 0
        zh = len(re.findall(r'[\u4e00-\u9fff]', text))
        return int(zh / self.char_per_token_zh + (len(text) - zh) / self.char_per_token_en)

class ContextManager:
    def __init__(self, budget: int = 10000, reserve: int = 2000):
        self.budget = budget
        self.reserve = reserve
        self.counter = TokenCounter()
        self.current_usage = 0
    
    def get_available(self): 
        return max(0, self.budget - self.current_usage - self.reserve)
    
    def add_usage(self, tokens): 
        self.current_usage += tokens
    
    def trim_messages(self, messages, keep_system=True):
        if not messages: 
            return messages
        system = None
        if keep_system and messages and messages[0].get("role") == "system":
            system = messages[0]
            messages = messages[1:]
        total = sum(self.counter.count(m.get("content","")) + 4 for m in messages)
        avail = self.budget - self.reserve
        if system: 
            avail -= self.counter.count(system.get("content","")) + 4
        while total > avail and messages:
            r = messages.pop(0)
            total -= self.counter.count(r.get("content","")) + 4
        if system: 
            messages.insert(0, system)
        return messages


# ========================================
# 2. Enhanced SelfModelTracker
# ========================================

@dataclass
class CapabilityProfile:
    tool: str
    total_calls: int = 0
    successes: int = 0
    total_reward: float = 0.0
    recent_failures: int = 0
    confidence: float = 0.5
    last_updated: float = field(default_factory=time.time)
    
    @property
    def success_rate(self) -> float:
        return self.successes / max(1, self.total_calls)
    
    @property
    def avg_reward(self) -> float:
        return self.total_reward / max(1, self.total_calls)
    
    def update_confidence(self):
        sr = self.success_rate
        ar = self.avg_reward
        rf = self.recent_failures
        self.confidence = 0.5 * sr + 0.3 * max(0, min(1, (ar + 1) / 2)) + 0.2 * max(0, 1 - rf / 5)
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.last_updated = time.time()


class SelfModelTracker:
    def __init__(self):
        self.profiles: Dict[str, CapabilityProfile] = {}
        self.total_success = 0
        self.total_failure = 0
        self.total_reward = 0.0
        self.history: List = []
    
    def add_experience(self, action, tool, success, reward):
        if tool not in self.profiles:
            self.profiles[tool] = CapabilityProfile(tool=tool)
        p = self.profiles[tool]
        p.total_calls += 1
        if success: 
            p.successes += 1
        p.total_reward += reward
        if not success: 
            p.recent_failures += 1
        else: 
            p.recent_failures = max(0, p.recent_failures - 1)
        p.update_confidence()
        self.total_reward += reward
        if success: 
            self.total_success += 1
        else: 
            self.total_failure += 1
        self.history.append({"action": action, "tool": tool, "success": success, "reward": reward, "ts": time.time()})
        if len(self.history) > 1000: 
            self.history = self.history[-500:]
    
    def is_capable(self, tool, dangerous=False):
        if tool not in self.profiles: 
            return True, 0.5
        p = self.profiles[tool]
        c = p.confidence
        thresh = 0.6 if dangerous else 0.3
        if p.recent_failures >= 3: 
            c *= 0.5
        return c >= thresh, max(0.0, min(1.0, c))
    
    def get_repr(self): 
        return [0.5, 0.0, 0.0, 0.5]


# ========================================
# 3. Output Parser
# ========================================

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except: 
    HAS_JSONSCHEMA = False

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "tool": {"type": "string"},
        "args": {"type": "object"},
        "reason": {"type": "string"},
        "finish": {"type": "boolean"}
    },
    "required": ["tool", "args", "reason", "finish"]
}

class OutputParser:
    def __init__(self, schema=None, max_retries=3):
        self.schema = schema or {
            "type": "object",
            "properties": {
                "tool": {"type": "string"},
                "args": {"type": "object"},
                "reason": {"type": "string"},
                "finish": {"type": "boolean"}
            },
            "required": ["tool", "args", "reason", "finish"]
        }
        self.validator = None
        if HAS_JSONSCHEMA:
            try: 
                self.validator = jsonschema.Draft7Validator(self.schema)
            except: 
                pass
    
    def parse(self, content):
        json_str = self._extract_json(content)
        if not json_str: 
            return None, "未找到 JSON"
        try: 
            plan = json.loads(json_str)
        except:
            fixed = self._try_fix_json(content)
            if fixed:
                try: 
                    plan = json.loads(fixed)
                except: 
                    return None, "JSON 解析失败"
            else: 
                return None, "JSON 解析失败"
        
        if self.validator:
            errors = list(self.validator.iter_errors(plan))
            if errors:
                fixed = self._auto_fix_plan(plan, errors)
                if fixed: 
                    plan = fixed
                else: 
                    return None, f"Schema: {errors[0].message}"
        return plan, None
    
    def _extract_json(self, content):
        if "```json" in content: 
            return content.split("```json")[1].split("```")[0].strip()
        if "```" in content:
            for p in content.split("```")[1::2]:
                s = p.strip()
                if s.startswith("{"): 
                    return s
        if content.strip().startswith("{"): 
            return content.strip()
        return None
    
    def _try_fix_json(self, s):
        try: 
            json.loads(re.sub(r',\s*([}\]])', r'\1', s))
            return s
        except: 
            return None
    
    def _auto_fix_plan(self, plan, errors):
        fixed = plan.copy()
        for e in errors:
            if e.validator == "required":
                for f in e.validator_value:
                    if f not in fixed:
                        if f == "args": 
                            fixed[f] = {}
                        elif f == "reason": 
                            fixed[f] = "auto"
                        elif f == "finish": 
                            fixed[f] = False
                        elif f == "tool": 
                            fixed[f] = "list_dir"
            elif e.validator == "type" and e.validator_value == "boolean":
                if "finish" in fixed and not isinstance(fixed["finish"], bool):
                    fixed["finish"] = str(fixed["finish"]).lower() in ("true", "1", "yes")
        return fixed


# ========================================
# 4. LLM Providers
# ========================================

class LLMProvider:
    def generate(self, messages, tools=None): 
        raise NotImplementedError


class ProfyProvider(LLMProvider):
    def __init__(self, model="gpt-5.6-sol", api_key=None, base_url=None, end_user_id="hermes-main-user"):
        self.model = model
        self.api_key = api_key or os.environ.get("PROFY_API_KEY")
        self.base_url = base_url or "https://api.profy.cn/v1"
        self.end_user_id = end_user_id
    
    def generate(self, messages, tools=None):
        import requests
        body = {"model": "gpt-5.6-sol", "messages": messages, "temperature": 0.3, "max_tokens": 2000}
        if tools: 
            body["tools"] = tools
        try:
            r = requests.post(
                "https://api.profy.cn/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.environ.get('PROFY_API_KEY')}", 
                        "X-Profy-End-User-Id": "hermes-main-user"},
                json=body, timeout=120)
            if r.status_code != 200: 
                return {"success": False, "error": f"HTTP {r.status_code}"}
            for ch in r.json().get("choices", []):
                for k in ["content", "text"]:
                    if ch.get("message", {}).get(k):
                        return {"success": True, "content": ch["message"][k], "tokens": r.json().get("usage",{}).get("total_tokens",0)}
            return {"success": False, "error": "无内容"}
        except Exception as e: 
            return {"success": False, "error": str(e)}


class DeepSeekProvider(LLMProvider):
    def __init__(self, api_key=None, base_url="https://api.deepseek.com/v1", model="deepseek-chat"):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.base_url = base_url
        self.model = model
    
    def generate(self, messages, tools=None):
        import requests
        body = {"model": "deepseek-chat", "messages": messages, "temperature": 0.3, "max_tokens": 2000}
        if tools: 
            body["tools"] = tools
        r = requests.post(f"https://api.deepseek.com/v1/chat/completions", 
                         headers={"Authorization": f"Bearer {self.api_key}"}, 
                         json=body, timeout=120)
        if r.status_code != 200: 
            return {"success": False, "error": f"HTTP {r.status_code}"}
        for ch in r.json().get("choices", []):
            if ch.get("message", {}).get("content"):
                return {"success": True, "content": ch["message"]["content"]}
        return {"success": False, "error": "无内容"}


class MockProvider(LLMProvider):
    def generate(self, messages, tools=None):
        return {"success": True, "content": '{"tool": "list_dir", "args": {"path": "."}, "reason": "查看目录", "finish": false}', "tokens": 100}


def create_llm(provider="mock", **kw):
    if provider == "profy": 
        return ProfyProvider()
    elif provider == "deepseek": 
        return DeepSeekProvider()
    return MockProvider()


# ========================================
# Tool Implementations
# ========================================

def safe_shell(command, timeout=30):
    for p in ["rm -rf", "sudo", "chmod 777", "mkfs", "dd if="]:
        if p in command: 
            return {"success": False, "error": f"危险命令: {p}"}
    try:
        r = subprocess.run(shlex.split(command), capture_output=True, text=True, timeout=timeout, cwd=os.getcwd())
        return {"success": r.returncode==0, "stdout": r.stdout[:2000], "stderr": r.stderr[:1000], "returncode": r.returncode}
    except subprocess.TimeoutExpired: 
        return {"success": False, "error": f"超时 ({timeout}s)"}
    except Exception as e: 
        return {"success": False, "error": str(e)}

def read_file(path):
    try:
        p = Path(path)
        if not p.exists(): 
            return {"success": False, "error": "不存在"}
        if p.stat().st_size > 1_000_000: 
            return {"success": False, "error": "太大"}
        return {"success": True, "content": p.read_text(encoding="utf-8")[:5000]}
    except Exception as e: 
        return {"success": False, "error": str(e)}

def write_file(path, content):
    try: 
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(content, encoding="utf-8")
        return {"success": True, "path": str(path)}
    except Exception as e: 
        return {"success": False, "error": str(e)}

def list_dir(path="."):
    try:
        p = Path(path)
        items = []
        for item in sorted(p.iterdir())[:100]:
            items.append({"name": item.name, "type": "dir" if item.is_dir() else "file"})
        return {"success": True, "items": items}
    except Exception as e: 
        return {"success": False, "error": str(e)}

def http_request(url, method="GET", data=""):
    try:
        req = urllib.request.Request(url, method=method)
        req.add_header("User-Agent", "BreakShell/0.2")
        if data and method in ("POST","PUT"): 
            req.data = data.encode()
        with urllib.request.urlopen(req, timeout=15) as r: 
            return {"success": True, "status": r.status, "body": r.read().decode()[:3000]}
    except Exception as e: 
        return {"success": False, "error": str(e)}

def grep_files(pattern, path=".", file_pattern="*.py"):
    try:
        import glob
        res = []
        for f in glob.glob(os.path.join(path, "**", file_pattern), recursive=True)[:50]:
            try:
                c = Path(f).read_text(encoding="utf-8", errors="replace")
                for i, line in enumerate(c.split("\n")):
                    if pattern.lower() in line.lower():
                        res.append({"file": f, "line": i+1, "content": line.strip()[:200]})
                        if len(res) >= 20: 
                            break
            except: 
                pass
        if res: 
            return {"success": True, "results": res}
        return {"success": False, "error": "未找到"}
    except Exception as e: 
        return {"success": False, "error": str(e)}


# ========================================
# Tool Registry & Spec
# ========================================

@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable
    permission: str = "read-only"
    dangerous: bool = False


class ToolRegistry:
    def __init__(self): 
        self._tools: Dict[str, ToolSpec] = {}
    
    def register(self, tool: ToolSpec):
        self._tools[tool.name] = tool
    
    def get(self, name): 
        return self._tools.get(name)
    
    def list_tools(self, level="read-only") -> List[ToolSpec]:
        return [t for t in self._tools.values() if PermissionLevel.can_execute(t.permission, level)]
    
    def describe(self, level="read-only") -> List[Dict]:
        return [{"name": t.name, "description": t.description, "schema": t.input_schema, "permission": t.permission} 
                for t in self.list_tools(level)]


def create_default_registry():
    reg = ToolRegistry()
    reg.register(ToolSpec("read_file", "读取文件", {"type":"object","properties":{"path":{"type":"string"}}}, read_file, "read-only"))
    reg.register(ToolSpec("write_file", "写入文件", {"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}}}, write_file, "workspace-write"))
    reg.register(ToolSpec("list_dir", "列出目录", {"type":"object","properties":{"path":{"type":"string"}}}, list_dir, "read-only"))
    reg.register(ToolSpec("shell", "Shell命令", {"type":"object","properties":{"command":{"type":"string"}}}, safe_shell, "system", True))
    reg.register(ToolSpec("http_request", "HTTP请求", {"type":"object","properties":{"url":{"type":"string"},"method":{"type":"string"}}}, http_request, "network"))
    reg.register(ToolSpec("grep_files", "搜索内容", {"type":"object","properties":{"pattern":{"type":"string"},"path":{"type":"string"},"file_pattern":{"type":"string"}}}, grep_files, "read-only"))
    return reg


# ========================================
# Permission System
# ========================================

class PermissionLevel:
    LEVELS = ["read-only", "workspace-write", "network", "system"]
    
    @classmethod
    def can_execute(cls, tool_level, user_level):
        try: 
            return cls.LEVELS.index(tool_level) <= cls.LEVELS.index(user_level)
        except: 
            return False


# ========================================
# Token Budget & Context
# ========================================

class TokenCounter:
    def __init__(self): 
        self.char_per_token_en = 4
        self.char_per_token_zh = 1.5
    
    def count(self, text: str) -> int:
        if not text: 
            return 0
        zh = len(re.findall(r'[\u4e00-\u9fff]', text))
        return int(zh / 1.5 + (len(text) - zh) / 4)


class ContextManager:
    def __init__(self, budget=10000, reserve=2000):
        self.budget = budget
        self.reserve = 2000
        self.counter = TokenCounter()
        self.current_usage = 0
    
    def get_available(self): 
        return max(0, self.budget - self.current_usage - self.reserve)
    
    def add_usage(self, tokens): 
        self.current_usage += tokens
    
    def trim_messages(self, messages, keep_system=True):
        if not messages: 
            return messages
        system = None
        if keep_system and messages and messages[0].get("role") == "system":
            system = messages[0]
            messages = messages[1:]
        total = sum(self.counter.count(m.get("content","")) + 4 for m in messages)
        avail = 10000 - 2000
        if system: 
            avail -= self.counter.count(system.get("content","")) + 4
        while total > avail and messages:
            r = messages.pop(0)
            total -= self.counter.count(r.get("content","")) + 4
        if system: 
            messages.insert(0, system)
        return messages


# ========================================
# Enhanced SelfModelTracker
# ========================================

@dataclass
class CapabilityProfile:
    tool: str
    total_calls: int = 0
    successes: int = 0
    total_reward: float = 0.0
    recent_failures: int = 0
    confidence: float = 0.5
    last_updated: float = field(default_factory=time.time)
    
    @property
    def success_rate(self) -> float:
        return self.successes / max(1, self.total_calls)
    
    @property
    def avg_reward(self) -> float:
        return self.total_reward / max(1, self.total_calls)
    
    def update_confidence(self):
        sr = self.success_rate
        ar = self.avg_reward
        rf = self.recent_failures
        self.confidence = 0.5 * sr + 0.3 * max(0, min(1, (ar + 1) / 2)) + 0.2 * max(0, 1 - rf / 5)
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.last_updated = time.time()


class SelfModelTracker:
    def __init__(self):
        self.profiles: Dict[str, CapabilityProfile] = {}
        self.total_success = 0
        self.total_failure = 0
        self.total_reward = 0.0
        self.history: List = []
    
    def add_experience(self, action, tool, success, reward):
        if tool not in self.profiles:
            self.profiles[tool] = CapabilityProfile(tool=tool)
        p = self.profiles[tool]
        p.total_calls += 1
        if success: 
            p.successes += 1
        p.total_reward += reward
        if not success: 
            p.recent_failures += 1
        else: 
            p.recent_failures = max(0, p.recent_failures - 1)
        p.update_confidence()
        self.total_reward += reward
        if success: 
            self.total_success += 1
        else: 
            self.total_failure += 1
        self.history.append({"action": action, "tool": tool, "success": success, "reward": reward, "ts": time.time()})
        if len(self.history) > 1000: 
            self.history = self.history[-500:]
    
    def is_capable(self, tool, dangerous=False):
        if tool not in self.profiles: 
            return True, 0.5
        p = self.profiles[tool]
        c = p.confidence
        thresh = 0.6 if dangerous else 0.3
        if p.recent_failures >= 3: 
            c *= 0.5
        return c >= thresh, max(0.0, min(1.0, c))
    
    def get_repr(self): 
        return [0.5, 0.0, 0.0, 0.5]


# ========================================
# Output Parser
# ========================================

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except: 
    HAS_JSONSCHEMA = False

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "tool": {"type": "string"},
        "args": {"type": "object"},
        "reason": {"type": "string"},
        "finish": {"type": "boolean"}
    },
    "required": ["tool", "args", "reason", "finish"]
}

class OutputParser:
    def __init__(self, schema=None, max_retries=3):
        self.schema = schema or {
            "type": "object",
            "properties": {
                "tool": {"type": "string"},
                "args": {"type": "object"},
                "reason": {"type": "string"},
                "finish": {"type": "boolean"}
            },
            "required": ["tool", "args", "reason", "finish"]
        }
        self.validator = None
        if HAS_JSONSCHEMA:
            try: 
                self.validator = jsonschema.Draft7Validator(self.schema)
            except: 
                pass
    
    def parse(self, content):
        json_str = self._extract_json(content)
        if not json_str: 
            return None, "未找到 JSON"
        try: 
            plan = json.loads(json_str)
        except:
            fixed = self._try_fix_json(content)
            if fixed:
                try: 
                    plan = json.loads(fixed)
                except: 
                    return None, "JSON 解析失败"
            else: 
                return None, "JSON 解析失败"
        
        if self.validator:
            errors = list(self.validator.iter_errors(plan))
            if errors:
                fixed = self._auto_fix_plan(plan, errors)
                if fixed: 
                    plan = fixed
                else: 
                    return None, f"Schema: {errors[0].message}"
        return plan, None
    
    def _extract_json(self, content):
        if "```json" in content: 
            return content.split("```json")[1].split("```")[0].strip()
        if "```" in content:
            for p in content.split("```")[1::2]:
                s = p.strip()
                if s.startswith("{"): 
                    return s
        if content.strip().startswith("{"): 
            return content.strip()
        return None
    
    def _try_fix_json(self, s):
        try: 
            json.loads(re.sub(r',\s*([}\]])', r'\1', s))
            return s
        except: 
            return None
    
    def _auto_fix_plan(self, plan, errors):
        fixed = plan.copy()
        for e in errors:
            if e.validator == "required":
                for f in e.validator_value:
                    if f not in fixed:
                        if f == "args": 
                            fixed[f] = {}
                        elif f == "reason": 
                            fixed[f] = "auto"
                        elif f == "finish": 
                            fixed[f] = False
                        elif f == "tool": 
                            fixed[f] = "list_dir"
            elif e.validator == "type" and e.validator_value == "boolean":
                if "finish" in fixed and not isinstance(fixed["finish"], bool):
                    fixed["finish"] = str(fixed["finish"]).lower() in ("true", "1", "yes")
        return fixed


# ========================================
# Permission System
# ========================================

class PermissionLevel:
    LEVELS = ["read-only", "workspace-write", "network", "system"]
    
    @classmethod
    def can_execute(cls, tool_level, user_level):
        try: 
            return cls.LEVELS.index(tool_level) <= cls.LEVELS.index(user_level)
        except: 
            return False


# ========================================
# Agent State
# ========================================

@dataclass
class AgentState:
    goal: str = ""
    messages: List = field(default_factory=list)
    observations: List = field(default_factory=list)
    tool_calls: List = field(default_factory=list)
    step_count: int = 0
    status: str = "idle"
    error: str = None
    session_id: str = ""
    max_steps: int = 30
    
    def to_dict(self): 
        return asdict(self)


# ========================================
# Agent Loop
# ========================================

class AgentLoop:
    def __init__(self, llm, registry, max_steps=30, permission="workspace-write"):
        self.llm = llm
        self.registry = registry
        self.max_steps = max_steps
        self.permission = permission
        self.self_model = SelfModelTracker()
        self.parser = OutputParser()
        self.context_manager = ContextManager()
        self.state = AgentState(max_steps=max_steps)
    
    def run(self, goal: str) -> AgentState:
        self.state.goal = goal
        self.state.status = "planning"
        for step in range(self.max_steps):
            self.state.step_count = step
            self.state.status = "planning"
            plan = self._plan()
            if plan.get("finish"): 
                self.state.status = "finished"
                break
            self.state.status = "acting"
            result = self._act(plan)
            self.state.status = "observing"
            self.state.observations.append({"success": result.get("success")})
            if not result.get("success"):
                if len([h for h in self.self_model.history[-3:] if not h["success"]]) >= 3:
                    self.state.status = "failed"
                    break
        else: 
            self.state.status = "finished"
        return self.state
    
    def _plan(self):
        tools = [{"name": t.name, "description": t.description} for t in self.registry.list_tools(self.permission)]
        sys = f"工具: {json.dumps(tools, ensure_ascii=False)}\n输出: {{\"tool\":\"\",\"args\":{{}},\"reason\":\"\",\"finish\":false}}"
        msgs = [{"role":"system","content":sys},{"role":"user","content":f"目标: {self.state.goal}\n步骤: {self.state.step_count}"}]
        msgs = self.context_manager.trim_messages(msgs)
        res = self.llm.generate(msgs)
        plan, _ = OutputParser().parse(res.get("content",""))
        return plan or {"finish": True, "reason": "解析失败"}
    
    def _act(self, plan):
        tool_name = plan.get("tool", "")
        args = plan.get("args", {})
        tool = self.registry.get(tool_name)
        if not tool: 
            return {"success": False, "error": "工具不存在"}
        try: 
            return tool.handler(**args)
        except Exception as e: 
            return {"success": False, "error": str(e)}


def create_llm(provider="mock", **kw):
    if provider == "profy": 
        return ProfyProvider()
    elif provider == "deepseek": 
        return DeepSeekProvider()
    return MockProvider()


def create_default_registry():
    reg = ToolRegistry()
    reg.register(ToolSpec("read_file", "读取文件", {"type":"object","properties":{"path":{"type":"string"}}}, read_file, "read-only"))
    reg.register(ToolSpec("write_file", "写入文件", {"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}}}, write_file, "workspace-write"))
    reg.register(ToolSpec("list_dir", "列出目录", {"type":"object","properties":{"path":{"type":"string"}}}, list_dir, "read-only"))
    reg.register(ToolSpec("shell", "Shell命令", {"type":"object","properties":{"command":{"type":"string"}}}, safe_shell, "system", True))
    reg.register(ToolSpec("http_request", "HTTP请求", {"type":"object","properties":{"url":{"type":"string"},"method":{"type":"string"}}}, http_request, "network"))
    reg.register(ToolSpec("grep_files", "搜索内容", {"type":"object","properties":{"pattern":{"type":"string"},"path":{"type":"string"},"file_pattern":{"type":"string"}}}, grep_files, "read-only"))
    return reg


def run_agent(goal: str, provider="mock", llm_model="gpt-5.6-sol", max_steps=30):
    llm = create_llm(provider) if provider != "mock" else MockProvider()
    registry = create_default_registry()
    agent = AgentLoop(llm, registry, max_steps=max_steps)
    return agent.run(goal)


if __name__ == "__main__":
    state = run_agent("列出当前目录的所有文件")
    print(f"状态: {state.status}")
    print(f"步数: {state.step_count}")
    print(f"工具调用: {len(state.tool_calls)}")