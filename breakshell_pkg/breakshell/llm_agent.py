# -*- coding: utf-8 -*-
"""
BreakShell LLM Agent — Phase 1 MVP
=====================================
完整 Agent Loop：plan → act → observe → reflect → finish

核心组件：
- AgentState：可序列化的 Agent 状态
- ToolSpec/ToolRegistry：带权限的工具系统
- LLMDecision：LLM 决策层（DeepSeek/Profy 等）
- SelfModel：BreakShell 自我模型（LSTM 编码历史）
- AgentLoop：核心事件循环
- EventLogger：结构化事件日志
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
    status: str = "idle"  # idle | planning | acting | observing | reflecting | finished | failed
    error: Optional[str] = None
    session_id: str = ""
    max_steps: int = 30
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# ========================================
# 2. Tool System
# ========================================

@dataclass
class ToolSpec:
    """工具规格"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable
    permission: str = "read-only"  # read-only | workspace-write | network | system
    dangerous: bool = False


class ToolRegistry:
    """工具注册表，带权限控制"""
    
    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}
        self._whitelist: Dict[str, set] = {
            "read-only": set(),
            "workspace-write": set(),
            "network": set(),
            "system": set(),
        }
    
    def register(self, tool: ToolSpec):
        self._tools[tool.name] = tool
        self._whitelist[tool.permission].add(tool.name)
    
    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)
    
    def list_tools(self, permission_level: str = "read-only") -> List[ToolSpec]:
        """列出允许的工具"""
        allowed = set()
        levels = ["read-only", "workspace-write", "network", "system"]
        for level in levels:
            allowed |= self._whitelist[level]
            if level == permission_level:
                break
        return [self._tools[n] for n in allowed if n in self._tools]
    
    def describe(self, permission_level: str = "read-only") -> List[Dict]:
        tools = self.list_tools(permission_level)
        return [{"name": t.name, "description": t.description, "schema": t.input_schema} for t in tools]


def safe_shell(command: str, timeout: int = 30) -> Dict[str, Any]:
    """安全 Shell 执行（白名单+限制）"""
    # 危险命令黑名单
    dangerous_patterns = ["rm -rf", "sudo", "chmod 777", "mkfs", "dd if=", ":(){:|:&};:", "> /dev"]
    for p in dangerous_patterns:
        if p in command:
            return {"success": False, "error": f"危险命令被拒绝: {p}"}
    
    try:
        args = shlex.split(command)
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.getcwd(),
        )
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
    """读取文件"""
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
    """写入文件"""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return {"success": True, "path": str(p)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def list_dir(path: str = ".") -> Dict[str, Any]:
    """列出目录"""
    try:
        p = Path(path)
        items = []
        for item in sorted(p.iterdir())[:100]:
            items.append({
                "name": item.name,
                "type": "dir" if item.is_dir() else "file",
                "size": item.stat().st_size if item.is_file() else None,
            })
        return {"success": True, "items": items}
    except Exception as e:
        return {"success": False, "error": str(e)}


def http_request(url: str, method: str = "GET", data: str = "") -> Dict[str, Any]:
    """HTTP 请求"""
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


def create_default_registry() -> ToolRegistry:
    """创建默认工具注册表"""
    reg = ToolRegistry()
    reg.register(ToolSpec("read_file", "读取本地文件", {"type": "object", "properties": {"path": {"type": "string"}}}, read_file, "read-only"))
    reg.register(ToolSpec("write_file", "写入本地文件", {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}}, write_file, "workspace-write"))
    reg.register(ToolSpec("list_dir", "列出目录内容", {"type": "object", "properties": {"path": {"type": "string"}}}, list_dir, "read-only"))
    reg.register(ToolSpec("shell", "执行 Shell 命令", {"type": "object", "properties": {"command": {"type": "string"}}}, safe_shell, "system", dangerous=True))
    reg.register(ToolSpec("http_request", "发送 HTTP 请求", {"type": "object", "properties": {"url": {"type": "string"}, "method": {"type": "string"}}}, http_request, "network"))
    return reg


# ========================================
# 3. LLM Provider 抽象层
# ========================================

class LLMProvider:
    """LLM Provider 接口"""
    
    def generate(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None) -> Dict[str, Any]:
        raise NotImplementedError


class ProfyProvider(LLMProvider):
    """Profy API（支持 gpt-5.6-sol 等）"""
    
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
        
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "X-Profy-End-User-Id": self.end_user_id, "Content-Type": "application/json"},
            json=body, timeout=120,
        )
        if resp.status_code != 200:
            return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
        
        data = resp.json()
        for choice in data.get("choices", []):
            msg = choice.get("message", {})
            for key in ["content", "text"]:
                if msg.get(key):
                    return {"success": True, "content": msg[key]}
        return {"success": False, "error": "无内容"}


class DeepSeekProvider(LLMProvider):
    """DeepSeek API（OpenAI 兼容）"""
    
    def __init__(self, api_key: str = None, base_url: str = "https://api.deepseek.com/v1", model: str = "deepseek-chat"):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.base_url = base_url
        self.model = model
    
    def generate(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None) -> Dict[str, Any]:
        import requests
        body = {"model": self.model, "messages": messages, "temperature": 0.3, "max_tokens": 2000}
        if tools:
            body["tools"] = tools
        
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=body, timeout=120,
        )
        if resp.status_code != 200:
            return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
        data = resp.json()
        for choice in data.get("choices", []):
            msg = choice.get("message", {})
            if msg.get("content"):
                return {"success": True, "content": msg["content"]}
        return {"success": False, "error": "无内容"}


class MockProvider(LLMProvider):
    """Mock Provider（测试用）"""
    
    def generate(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None) -> Dict[str, Any]:
        return {"success": True, "content": '{"tool": "list_dir", "args": {"path": "."}, "reason": "查看目录", "finish": false}'}


def create_llm(provider: str = "mock", **kwargs) -> LLMProvider:
    """工厂函数"""
    if provider == "profy":
        return ProfyProvider(**kwargs)
    elif provider == "deepseek":
        return DeepSeekProvider(**kwargs)
    return MockProvider()


# ========================================
# 4. Self Model（BreakShell 核心）
# ========================================

class SelfModelTracker:
    """
    自我模型追踪器（BreakShell 核心组件）
    
    编码历史 (action, reward) → 推断能力边界
    """
    
    def __init__(self):
        self.history: List[Dict[str, Any]] = []
        self.success_count = 0
        self.fail_count = 0
        self.total_reward = 0.0
    
    def add_experience(self, action: str, tool_name: str, success: bool, reward: float):
        self.history.append({
            "action": action, "tool": tool_name, "success": success,
            "reward": reward, "step": len(self.history),
        })
        self.total_reward += reward
        if success:
            self.success_count += 1
        else:
            self.fail_count += 1
    
    def get_repr(self) -> List[float]:
        """获取自我表征（能力指标）"""
        total = max(1, self.success_count + self.fail_count)
        success_rate = self.success_count / total
        avg_reward = self.total_reward / total
        recent_failures = sum(1 for h in self.history[-5:] if not h["success"])
        return [success_rate, avg_reward, len(self.history) / 100.0, recent_failures / 5.0]
    
    def is_capable(self, tool_name: str, dangerous: bool = False) -> Tuple[bool, float]:
        """推断是否有能力执行"""
        if not self.history:
            return True, 0.5  # 初始有信心
        
        # 检查该工具的历史成功率
        tool_history = [h for h in self.history if h["tool"] == tool_name]
        if tool_history:
            tool_success = sum(1 for h in tool_history if h["success"]) / len(tool_history)
            if tool_success < 0.3:
                return False, tool_success
        
        # 整体信心
        total = max(1, self.success_count + self.fail_count)
        confidence = self.success_count / total
        
        if dangerous and confidence < 0.6:
            return False, confidence
        
        return True, confidence


# ========================================
# 5. Event Logger
# ========================================

class EventLogger:
    """结构化事件日志"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.events: List[Dict[str, Any]] = []
        self.session_id = str(uuid.uuid4())[:8]
        self.start_time = time.time()
    
    def log(self, event_type: str, data: Dict[str, Any]):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "step": len(self.events),
            "type": event_type,
            "elapsed": round(time.time() - self.start_time, 2),
            **data,
        }
        self.events.append(entry)
        return entry
    
    def save(self):
        path = self.log_dir / f"session_{self.session_id}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for e in self.events:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        return path
    
    def summary(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "total_events": len(self.events),
            "duration": round(time.time() - self.start_time, 2),
            "event_types": {t: sum(1 for e in self.events if e["type"] == t) for t in set(e["type"] for e in self.events)},
        }


# ========================================
# 6. Session Store（SQLite 持久化）
# ========================================

class SessionStore:
    """会话持久化"""
    
    def __init__(self, db_path: str = "sessions.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TEXT,
                updated_at TEXT,
                goal TEXT,
                state TEXT,
                events TEXT
            )
        """)
        self.conn.commit()
    
    def save_session(self, session_id: str, goal: str, state: AgentState, events: List[Dict]):
        now = datetime.now().isoformat()
        self.conn.execute(
            """INSERT OR REPLACE INTO sessions VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, now, now, goal, state.to_json(), json.dumps(events, ensure_ascii=False)),
        )
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
# 7. Agent Loop（核心）
# ========================================

class AgentLoop:
    """
    核心 Agent Loop
    
    流程：plan → act → observe → reflect → finish/fail
    """
    
    def __init__(
        self,
        llm: LLMProvider,
        registry: ToolRegistry,
        max_steps: int = 30,
        permission: str = "workspace-write",
        session_dir: str = "sessions",
    ):
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
        """执行 Agent Loop"""
        self.state.goal = goal
        self.state.status = "planning"
        self.logger.log("run_started", {"goal": goal})
        
        system_msg = self._build_system_msg()
        
        for step in range(self.max_steps):
            self.state.step_count = step
            
            # 1. PLAN
            self.state.status = "planning"
            plan = self._plan(system_msg)
            self.logger.log("plan", {"step": step, "plan": plan})
            
            if plan.get("finish"):
                self.state.status = "finished"
                self.logger.log("finished", {"reason": plan.get("reason", "任务完成")})
                break
            
            # 2. ACT
            self.state.status = "acting"
            action_result = self._act(plan)
            self.logger.log("act", {"step": step, "tool": plan.get("tool"), "result": str(action_result)[:200]})
            
            # 3. OBSERVE
            self.state.status = "observing"
            obs = self._observe(action_result)
            self.state.observations.append(obs)
            
            # 4. REFLECT
            self.state.status = "reflecting"
            self._reflect(plan, action_result)
            
            # 检查是否失败
            if self.state.status == "failed":
                break
        
        else:
            self.state.status = "finished"
            self.logger.log("finished", {"reason": "达到最大步数"})
        
        # 保存会话
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
        """LLM 决策"""
        self.state.messages.append({"role": "system", "content": system_msg})
        self.state.messages.append({"role": "user", "content": f"目标: {self.state.goal}\n当前步: {self.state.step_count}\n请选择行动"})
        
        result = self.llm.generate(self.state.messages)
        if not result.get("success"):
            self.state.error = result.get("error")
            return {"finish": True, "reason": f"LLM 失败: {result.get('error')}"}
        
        content = result.get("content", "")
        self.state.messages.append({"role": "assistant", "content": content})
        
        # 尝试解析 JSON
        try:
            # 尝试找到 JSON 块
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            return json.loads(content.strip())
        except:
            # 回退：从文本推断
            return {"finish": True, "reason": content[:200]}
    
    def _act(self, plan: Dict) -> Any:
        """执行工具"""
        tool_name = plan.get("tool", "")
        args = plan.get("args", {})
        
        tool = self.registry.get(tool_name)
        if not tool:
            return {"success": False, "error": f"工具不存在: {tool_name}"}
        
        # 自我模型能力检查
        capable, confidence = self.self_model.is_capable(tool_name, tool.dangerous)
        self.logger.log("capability_check", {"tool": tool_name, "capable": capable, "confidence": round(confidence, 2)})
        
        if not capable:
            return {"success": False, "error": f"自我模型判断能力不足（置信度 {confidence:.2f}）"}
        
        # 执行
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
        """观察结果"""
        return {"step": self.state.step_count, "success": bool(result.get("success")), "summary": str(result)[:300]}
    
    def _reflect(self, plan: Dict, result: Any):
        """反思"""
        success = bool(result.get("success"))
        self.logger.log("reflect", {"success": success, "tool": plan.get("tool")})
        
        if not success:
            # 连续失败检查
            recent = [h for h in self.self_model.history[-3:] if not h["success"]]
            if len(recent) >= 3:
                self.state.status = "failed"
                self.state.error = "连续失败 3 次"
                self.logger.log("failed", {"reason": "连续失败"})


# ========================================
# 8. 便捷接口
# ========================================

def run_agent(goal: str, provider: str = "mock", llm_model: str = "gpt-5.6-sol", max_steps: int = 30) -> AgentState:
    """便捷函数：创建并运行 Agent"""
    llm = create_llm(provider, model=llm_model) if provider != "mock" else create_llm("mock")
    registry = create_default_registry()
    agent = AgentLoop(llm, registry, max_steps=max_steps)
    return agent.run(goal)


if __name__ == "__main__":
    state = run_agent("列出当前目录的所有文件")
    print(f"状态: {state.status}")
    print(f"步数: {state.step_count}")
    print(f"观察数: {len(state.observations)}")
