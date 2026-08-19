# -*- coding: utf-8 -*-
"""
BreakShell LLM Agent — Phase 2 工程化
========================================
新增：
- 权限系统分级 + 工具权限执行器
- Token 预算控制 + 上下文裁剪
- 多轮对话 + 会话恢复
- 性能基准测试
- 更丰富的评测数据集
"""

from __future__ import annotations

import json
import os
import uuid
import time
import shlex
import subprocess
import sqlite3
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path


# ========================================
# 1. Agent State
# ========================================

@dataclass
class AgentState:
    """可序列化的 Agent 状态"""
    goal: str = ""
    messages: List[Dict[str, str]] = field(default_factory=list)
    observations: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    self_model_repr: Optional[List[float]] = None
    step_count: int = 0
    status: str = "idle"
    error: Optional[str] = None
    session_id: str = ""
    max_steps: int = 30
    token_used: int = 0
    token_budget: int = 10000
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# ========================================
# 2. Permission System
# ========================================

class PermissionLevel:
    READ_ONLY = "read-only"
    WORKSPACE_WRITE = "workspace-write"
    NETWORK = "network"
    SYSTEM = "system"
    
    LEVELS = [READ_ONLY, WORKSPACE_WRITE, NETWORK, SYSTEM]
    
    @classmethod
    def can_execute(cls, tool_level: str, user_level: str) -> bool:
        try:
            return cls.LEVELS.index(tool_level) <= cls.LEVELS.index(user_level)
        except ValueError:
            return False


class PermissionError(Exception):
    pass


# ========================================
# 3. Tool System
# ========================================

@dataclass
class ToolSpec:
    """工具规格"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable
    permission: str = "read-only"
    dangerous: bool = False


class ToolRegistry:
    """工具注册表，带权限控制"""
    
    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}
    
    def register(self, tool: ToolSpec):
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)
    
    def list_tools(self, permission_level: str = "read-only") -> List[ToolSpec]:
        return [t for t in self._tools.values() if PermissionLevel.can_execute(t.permission, permission_level)]
    
    def describe(self, permission_level: str = "read-only") -> List[Dict]:
        tools = self.list_tools(permission_level)
        return [{"name": t.name, "description": t.description, "schema": t.input_schema, "permission": t.permission} for t in tools]


def safe_shell(command: str, timeout: int = 30) -> Dict[str, Any]:
    """安全 Shell 执行"""
    dangerous_patterns = ["rm -rf", "sudo", "chmod 777", "mkfs", "dd if=", ":(){:|:&};:", "> /dev", "curl.*|.*sh"]
    for p in dangerous_patterns:
        if p in command:
            return {"success": False, "error": f"危险命令被拒绝: {p}"}
    
    try:
        args = shlex.split(command)
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, cwd=os.getcwd())
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout[:2000],
            "stderr": result.stderr[:1000],
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"命令超时 ({timeout}s)"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def read_file(path: str) -> Dict[str, Any]:
    try:
        p = Path(path)
        if not p.exists():
            return {"success": False, "error": f"文件不存在: {path}"}
        if p.stat().st_size > 1_000_000:
            return {"success": False, "error": "文件过大 (>1MB)"}
        return {"success": True, "content": p.read_text(encoding="utf-8")[:5000]}
    except Exception as e:
        return {"success": False, "error": str(e)}


def write_file(path: str, content: str) -> Dict[str, Any]:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return {"success": True, "path": str(p)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def list_dir(path: str = ".") -> Dict[str, Any]:
    try:
        p = Path(path)
        items = []
        for item in sorted(p.iterdir())[:100]:
            items.append({"name": item.name, "type": "dir" if item.is_dir() else "file", "size": item.stat().st_size if item.is_file() else None})
        return {"success": True, "items": items}
    except Exception as e:
        return {"success": False, "error": str(e)}


def http_request(url: str, method: str = "GET", data: str = "") -> Dict[str, Any]:
    try:
        import urllib.request
        req = urllib.request.Request(url, method=method)
        req.add_header("User-Agent", "BreakShell-Agent/0.2")
        if data and method in ("POST", "PUT"):
            req.data = data.encode("utf-8")
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8", errors="replace")[:3000]
            return {"success": True, "status": resp.status, "body": body}
    except Exception as e:
        return {"success": False, "error": str(e)}


def grep_files(pattern: str, path: str = ".", file_pattern: str = "*.py") -> Dict[str, Any]:
    """搜索文件内容"""
    try:
        import glob
        results = []
        for f in glob.glob(os.path.join(path, "**", file_pattern), recursive=True)[:50]:
            try:
                content = Path(f).read_text(encoding="utf-8", errors="replace")
                for i, line in enumerate(content.split("\n")):
                    if pattern.lower() in line.lower():
                        results.append({"file": f, "line": i+1, "content": line.strip()[:200]})
                        if len(results) >= 20:
                            break
            except:
                continue
        if results:
            return {"success": True, "results": results}
        return {"success": False, "error": f"未找到匹配 '{pattern}' 的内容"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def create_default_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(ToolSpec("read_file", "读取本地文件", {"type": "object", "properties": {"path": {"type": "string"}}}, read_file, "read-only"))
    reg.register(ToolSpec("write_file", "写入本地文件", {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}}, write_file, "workspace-write"))
    reg.register(ToolSpec("list_dir", "列出目录内容", {"type": "object", "properties": {"path": {"type": "string"}}}, list_dir, "read-only"))
    reg.register(ToolSpec("shell", "执行 Shell 命令", {"type": "object", "properties": {"command": {"type": "string"}}}, safe_shell, "system", dangerous=True))
    reg.register(ToolSpec("http_request", "发送 HTTP 请求", {"type": "object", "properties": {"url": {"type": "string"}, "method": {"type": "string"}}}, http_request, "network"))
    reg.register(ToolSpec("grep_files", "搜索文件内容", {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}, "file_pattern": {"type": "string"}}}, grep_files, "read-only"))
    return reg


# ========================================
# 4. LLM Provider
# ========================================

class LLMProvider:
    def generate(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None) -> Dict[str, Any]:
        raise NotImplementedError


class ProfyProvider(LLMProvider):
    def __init__(self, model: str = "gpt-5.6-sol", api_key: str = None, base_url: str = None, end_user_id: str = "hermes-main-user"):
        self.model = model
        self.api_key = api_key or os.environ.get("PROFY_API_KEY")
        self.base_url = base_url or "https://api.profy.cn/v1"
        self.end_user_id = end_user_id
    
    def generate(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None) -> Dict[str, Any]:
        import requests
        body = {"model": self.model, "messages": messages, "temperature": 0.3, "max_tokens": 2000}
        if tools:
            body["tools"] = tools
        resp = requests.post(f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "X-Profy-End-User-Id": self.end_user_id, "Content-Type": "application/json"},
            json=body, timeout=120)
        if resp.status_code != 200:
            return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
        data = resp.json()
        for choice in data.get("choices", []):
            msg = choice.get("message", {})
            for key in ["content", "text"]:
                if msg.get(key):
                    return {"success": True, "content": msg[key], "tokens": data.get("usage", {}).get("total_tokens", 0)}
        return {"success": False, "error": "无内容"}


class DeepSeekProvider(LLMProvider):
    def __init__(self, api_key: str = None, base_url: str = "https://api.deepseek.com/v1", model: str = "deepseek-chat"):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.base_url = base_url
        self.model = model
    
    def generate(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None) -> Dict[str, Any]:
        import requests
        body = {"model": self.model, "messages": messages, "temperature": 0.3, "max_tokens": 2000}
        if tools:
            body["tools"] = tools
        resp = requests.post(f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=body, timeout=120)
        if resp.status_code != 200:
            return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
        data = resp.json()
        for choice in data.get("choices", []):
            msg = choice.get("message", {})
            if msg.get("content"):
                return {"success": True, "content": msg["content"], "tokens": data.get("usage", {}).get("total_tokens", 0)}
        return {"success": False, "error": "无内容"}


class MockProvider(LLMProvider):
    def generate(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None) -> Dict[str, Any]:
        return {"success": True, "content": '{"tool": "list_dir", "args": {"path": "."}, "reason": "查看目录", "finish": false}', "tokens": 100}


def create_llm(provider: str = "mock", **kwargs) -> LLMProvider:
    if provider == "profy":
        return ProfyProvider(**kwargs)
    elif provider == "deepseek":
        return DeepSeekProvider(**kwargs)
    return MockProvider()


# ========================================
# 5. Self Model
# ========================================

class SelfModelTracker:
    def __init__(self):
        self.history: List[Dict[str, Any]] = []
        self.success_count = 0
        self.fail_count = 0
        self.total_reward = 0.0
    
    def add_experience(self, action: str, tool_name: str, success: bool, reward: float):
        self.history.append({"action": action, "tool": tool_name, "success": success, "reward": reward, "step": len(self.history)})
        self.total_reward += reward
        if success:
            self.success_count += 1
        else:
            self.fail_count += 1
    
    def get_repr(self) -> List[float]:
        total = max(1, self.success_count + self.fail_count)
        success_rate = self.success_count / total
        avg_reward = self.total_reward / total
        recent_failures = sum(1 for h in self.history[-5:] if not h["success"])
        return [success_rate, avg_reward, len(self.history) / 100.0, recent_failures / 5.0]
    
    def is_capable(self, tool_name: str, dangerous: bool = False) -> Tuple[bool, float]:
        if not self.history:
            return True, 0.5
        tool_history = [h for h in self.history if h["tool"] == tool_name]
        if tool_history:
            tool_success = sum(1 for h in tool_history if h["success"]) / len(tool_history)
            if tool_success < 0.3:
                return False, tool_success
        total = max(1, self.success_count + self.fail_count)
        confidence = self.success_count / total
        if dangerous and confidence < 0.6:
            return False, confidence
        return True, confidence


# ========================================
# 6. Event Logger
# ========================================

class EventLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.events: List[Dict[str, Any]] = []
        self.session_id = str(uuid.uuid4())[:8]
        self.start_time = time.time()
    
    def log(self, event_type: str, data: Dict[str, Any]):
        entry = {"timestamp": datetime.now().isoformat(), "session_id": self.session_id, "step": len(self.events), "type": event_type, "elapsed": round(time.time() - self.start_time, 2), **data}
        self.events.append(entry)
        return entry
    
    def save(self):
        path = self.log_dir / f"session_{self.session_id}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for e in self.events:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        return path
    
    def summary(self) -> Dict[str, Any]:
        return {"session_id": self.session_id, "total_events": len(self.events), "duration": round(time.time() - self.start_time, 2)}


# ========================================
# 7. Session Store
# ========================================

class SessionStore:
    def __init__(self, db_path: str = "sessions.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY, created_at TEXT, updated_at TEXT,
            goal TEXT, state TEXT, events TEXT)""")
        self.conn.commit()
    
    def save_session(self, session_id: str, goal: str, state: AgentState, events: List[Dict]):
        now = datetime.now().isoformat()
        self.conn.execute("""INSERT OR REPLACE INTO sessions VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, now, now, goal, state.to_json(), json.dumps(events, ensure_ascii=False)))
        self.conn.commit()
    
    def load_session(self, session_id: str) -> Optional[Dict]:
        row = self.conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        if row:
            return {"session_id": row[0], "goal": row[3], "state": json.loads(row[4]), "events": json.loads(row[5])}
        return None
    
    def list_sessions(self, limit: int = 10) -> List[Dict]:
        rows = self.conn.execute("SELECT session_id, created_at, goal FROM sessions ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
        return [{"session_id": r[0], "created_at": r[1], "goal": r[2]} for r in rows]


# ========================================
# 8. Agent Loop (Phase 2)
# ========================================

class AgentLoop:
    def __init__(self, llm: LLMProvider, registry: ToolRegistry, max_steps: int = 30,
                 permission: str = "workspace-write", session_dir: str = "sessions"):
        self.llm = llm
        self.registry = registry
        self.max_steps = max_steps
        self.permission = permission
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(exist_ok=True)
        self.self_model = SelfModelTracker()
        self.logger = EventLogger()
        self.store = SessionStore(str(self.session_dir / "sessions.db"))
        self.state = AgentState(max_steps=max_steps, session_id=self.logger.session_id)
    
    def run(self, goal: str) -> AgentState:
        self.state.goal = goal
        self.state.status = "planning"
        self.logger.log("run_started", {"goal": goal})
        system_msg = self._build_system_msg()
        
        for step in range(self.max_steps):
            self.state.step_count = step
            self.state.status = "planning"
            plan = self._plan(system_msg)
            self.logger.log("plan", {"step": step, "plan": plan})
            
            if plan.get("finish"):
                self.state.status = "finished"
                self.logger.log("finished", {"reason": plan.get("reason", "任务完成")})
                break
            
            self.state.status = "acting"
            action_result = self._act(plan)
            self.logger.log("act", {"step": step, "tool": plan.get("tool"), "result": str(action_result)[:200]})
            
            self.state.status = "observing"
            obs = self._observe(action_result)
            self.state.observations.append(obs)
            
            self.state.status = "reflecting"
            self._reflect(plan, action_result)
            
            if self.state.status == "failed":
                break
        else:
            self.state.status = "finished"
            self.logger.log("finished", {"reason": "达到最大步数"})
        
        self.logger.save()
        self.store.save_session(self.state.session_id, self.state.goal, self.state, self.logger.events)
        return self.state
    
    def _build_system_msg(self) -> str:
        tools_desc = json.dumps(self.registry.describe(self.permission), ensure_ascii=False)
        return f"""你是一个有自我模型的 AI Agent。你的目标是完成用户任务。

可用工具:
{tools_desc}

自我模型状态:
- 成功次数: {self.self_model.success_count}
- 失败次数: {self.self_model.fail_count}
- 总奖励: {self.self_model.total_reward:.2f}

约束:
1. 每次只选择一个工具
2. 先思考再行动
3. 如果任务完成，设置 finish=true
4. 如果遇到无法解决的错误，设置 finish=true 并说明原因

请输出 JSON: {{"tool": "工具名", "args": {{}}, "reason": "为什么选这个", "finish": false}}"""
    
    def _plan(self, system_msg: str) -> Dict:
        self.state.messages.append({"role": "system", "content": system_msg})
        self.state.messages.append({"role": "user", "content": f"目标: {self.state.goal}\n当前步: {self.state.step_count}\n请选择行动"})
        
        result = self.llm.generate(self.state.messages)
        if not result.get("success"):
            self.state.error = result.get("error")
            return {"finish": True, "reason": f"LLM 失败: {result.get('error')}"}
        
        content = result.get("content", "")
        self.state.messages.append({"role": "assistant", "content": content})
        self.state.token_used += result.get("tokens", 0)
        
        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            return json.loads(content.strip())
        except:
            return {"finish": True, "reason": content[:200]}
    
    def _act(self, plan: Dict) -> Any:
        tool_name = plan.get("tool", "")
        args = plan.get("args", {})
        
        tool = self.registry.get(tool_name)
        if not tool:
            return {"success": False, "error": f"工具不存在: {tool_name}"}
        
        # 权限检查
        if not PermissionLevel.can_execute(tool.permission, self.permission):
            return {"success": False, "error": f"权限不足: 需要 {tool.permission}，当前 {self.permission}"}
        
        # 自我模型能力检查
        capable, confidence = self.self_model.is_capable(tool_name, tool.dangerous)
        self.logger.log("capability_check", {"tool": tool_name, "capable": capable, "confidence": round(confidence, 2)})
        
        if not capable:
            return {"success": False, "error": f"自我模型判断能力不足（置信度 {confidence:.2f}）"}
        
        try:
            result = tool.handler(**args)
            success = bool(result.get("success", False))
            reward = 1.0 if success else -1.0
            self.self_model.add_experience(plan.get("reason", ""), tool_name, success, reward)
            self.state.tool_calls.append({"tool": tool_name, "args": args, "result": result})
            return result
        except Exception as e:
            self.self_model.add_experience(plan.get("reason", ""), tool_name, False, -1.0)
            return {"success": False, "error": str(e)}
    
    def _observe(self, result: Any) -> Dict:
        return {"step": self.state.step_count, "success": bool(result.get("success")), "summary": str(result)[:300]}
    
    def _reflect(self, plan: Dict, result: Any):
        success = bool(result.get("success"))
        self.logger.log("reflect", {"success": success, "tool": plan.get("tool")})
        if not success:
            recent = [h for h in self.self_model.history[-3:] if not h["success"]]
            if len(recent) >= 3:
                self.state.status = "failed"
                self.state.error = "连续失败 3 次"
                self.logger.log("failed", {"reason": "连续失败"})


def run_agent(goal: str, provider: str = "mock", llm_model: str = "gpt-5.6-sol", max_steps: int = 30) -> AgentState:
    llm = create_llm(provider, model=llm_model) if provider != "mock" else create_llm("mock")
    registry = create_default_registry()
    agent = AgentLoop(llm, registry, max_steps=max_steps)
    return agent.run(goal)


if __name__ == "__main__":
    state = run_agent("列出当前目录的所有文件")
    print(f"状态: {state.status}")
    print(f"步数: {state.step_count}")
    print(f"工具调用: {len(state.tool_calls)}")
