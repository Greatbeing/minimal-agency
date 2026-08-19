# -*- coding: utf-8 -*-
"""
BreakShell Phase 3 — 认知进化
==================================
- 反思系统（Reflection）
- 多层记忆（Working + Episodic + Semantic）
- 多角色协作（Planner + Executor + Reviewer）
- 记忆检索 + 经验复用
"""

from __future__ import annotations

import json
import os
import uuid
import time
import sqlite3
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path


# ========================================
# 1. Reflection System
# ========================================

@dataclass
class ReflectionResult:
    """反思结果"""
    goal: str
    success: bool
    steps_taken: int
    tools_used: List[str]
    failed_steps: List[Dict[str, Any]]
    useful_knowledge: List[str]
    reusable_procedure: List[str]
    uncertainties: List[str]
    next_time_changes: List[str]
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def summary(self) -> str:
        return f"目标: {self.goal}\n成功: {self.success}\n步数: {self.steps_taken}\n有用知识: {len(self.useful_knowledge)}\n可复用程序: {len(self.reusable_procedure)}"


class ReflectionEngine:
    """反思引擎"""
    
    def reflect(self, goal: str, steps: List[Dict], tool_calls: List[Dict], success: bool) -> ReflectionResult:
        """生成反思"""
        failed_steps = [s for s in steps if not s.get("success", True)]
        tools_used = list(set(tc.get("tool", "") for tc in tool_calls))
        
        # 提取有用知识
        useful_knowledge = self._extract_knowledge(steps, tool_calls)
        
        # 提取可复用程序
        reusable_procedure = self._extract_procedure(steps, tool_calls)
        
        # 识别不确定性
        uncertainties = self._identify_uncertainties(steps, failed_steps)
        
        # 下次改进
        next_time_changes = self._suggest_improvements(steps, failed_steps, success)
        
        return ReflectionResult(
            goal=goal,
            success=success,
            steps_taken=len(steps),
            tools_used=tools_used,
            failed_steps=failed_steps,
            useful_knowledge=useful_knowledge,
            reusable_procedure=reusable_procedure,
            uncertainties=uncertainties,
            next_time_changes=next_time_changes,
        )
    
    def _extract_knowledge(self, steps: List[Dict], tool_calls: List[Dict]) -> List[str]:
        knowledge = []
        for tc in tool_calls:
            if tc.get("result", {}).get("success"):
                knowledge.append(f"{tc['tool']} 调用成功: {str(tc.get('args', {}))[:100]}")
        return knowledge[:10]
    
    def _extract_procedure(self, steps: List[Dict], tool_calls: List[Dict]) -> List[str]:
        procedure = []
        for i, tc in enumerate(tool_calls):
            if tc.get("result", {}).get("success"):
                procedure.append(f"步骤 {i+1}: 使用 {tc['tool']}")
        return procedure[:10]
    
    def _identify_uncertainties(self, steps: List[Dict], failed_steps: List[Dict]) -> List[str]:
        uncertainties = []
        for fs in failed_steps:
            uncertainties.append(f"失败: {fs.get('tool', 'unknown')} - {fs.get('error', 'unknown error')}")
        return uncertainties[:5]
    
    def _suggest_improvements(self, steps: List[Dict], failed_steps: List[Dict], success: bool) -> List[str]:
        improvements = []
        if not success:
            improvements.append("任务未完成，需要更多步骤或不同工具")
        if len(failed_steps) > len(steps) * 0.3:
            improvements.append("失败率过高，需要更好的规划")
        if not improvements:
            improvements.append("任务成功，保持当前策略")
        return improvements


# ========================================
# 2. Multi-Layer Memory
# ========================================

class WorkingMemory:
    """工作记忆 — 当前任务上下文"""
    
    def __init__(self, max_items: int = 20):
        self.items: List[Dict[str, Any]] = []
        self.max_items = max_items
    
    def add(self, item: Dict[str, Any]):
        self.items.append({"timestamp": datetime.now().isoformat(), **item})
        if len(self.items) > self.max_items:
            self.items = self.items[-self.max_items:]
    
    def get_context(self, n: int = 10) -> List[Dict]:
        return self.items[-n:]
    
    def clear(self):
        self.items = []


class EpisodicMemory:
    """情景记忆 — 历史任务经验"""
    
    def __init__(self, db_path: str = "memory.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""CREATE TABLE IF NOT EXISTS episodes (
            id TEXT PRIMARY KEY, timestamp TEXT, goal TEXT, success INTEGER,
            steps INTEGER, tools TEXT, reflection TEXT)""")
        self.conn.commit()
    
    def store(self, reflection: ReflectionResult):
        self.conn.execute(
            """INSERT INTO episodes VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (str(uuid.uuid4())[:8], reflection.timestamp, reflection.goal,
             int(reflection.success), reflection.steps_taken,
             json.dumps(reflection.tools_used), json.dumps(reflection.to_dict())),
        )
        self.conn.commit()
    
    def recall(self, goal: str, limit: int = 5) -> List[Dict]:
        """检索相关经验"""
        rows = self.conn.execute(
            """SELECT * FROM episodes WHERE goal LIKE ? ORDER BY timestamp DESC LIMIT ?""",
            (f"%{goal}%", limit)
        ).fetchall()
        return [{"id": r[0], "timestamp": r[1], "goal": r[2], "success": bool(r[3]),
                 "steps": r[4], "tools": json.loads(r[5]), "reflection": json.loads(r[6])} for r in rows]
    
    def get_success_rate(self, goal_type: str = "") -> float:
        if goal_type:
            rows = self.conn.execute("SELECT success FROM episodes WHERE goal LIKE ?", (f"%{goal_type}%",)).fetchall()
        else:
            rows = self.conn.execute("SELECT success FROM episodes").fetchall()
        if not rows:
            return 0.5
        return sum(r[0] for r in rows) / len(rows)


class SemanticMemory:
    """语义记忆 — 事实知识和概念关系"""
    
    def __init__(self, db_path: str = "memory.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""CREATE TABLE IF NOT EXISTS knowledge (
            id TEXT PRIMARY KEY, key TEXT, value TEXT, source TEXT,
            confidence REAL, timestamp TEXT)""")
        self.conn.commit()
    
    def store(self, key: str, value: str, source: str = "experience", confidence: float = 0.8):
        self.conn.execute(
            """INSERT INTO knowledge VALUES (?, ?, ?, ?, ?, ?)""",
            (str(uuid.uuid4())[:8], key, value, source, confidence, datetime.now().isoformat()),
        )
        self.conn.commit()
    
    def recall(self, key: str, limit: int = 5) -> List[Dict]:
        rows = self.conn.execute(
            """SELECT * FROM knowledge WHERE key LIKE ? ORDER BY confidence DESC LIMIT ?""",
            (f"%{key}%", limit)
        ).fetchall()
        return [{"id": r[0], "key": r[1], "value": r[2], "source": r[3], "confidence": r[4]} for r in rows]
    
    def get_all(self) -> List[Dict]:
        rows = self.conn.execute("SELECT * FROM knowledge ORDER BY confidence DESC LIMIT 50").fetchall()
        return [{"key": r[1], "value": r[2], "confidence": r[4]} for r in rows]


# ========================================
# 3. Multi-Role Collaboration
# ========================================

class Role:
    """角色基类"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def act(self, context: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class PlannerRole(Role):
    """规划者"""
    
    def __init__(self):
        super().__init__("Planner", "任务规划与分解")
    
    def act(self, context: Dict[str, Any]) -> Dict[str, Any]:
        goal = context.get("goal", "")
        return {
            "role": self.name,
            "plan": f"分解目标: {goal}",
            "steps": [f"分析 {goal}", "选择工具", "执行操作", "验证结果"],
        }


class ExecutorRole(Role):
    """执行者"""
    
    def __init__(self):
        super().__init__("Executor", "工具调用与执行")
    
    def act(self, context: Dict[str, Any]) -> Dict[str, Any]:
        tool = context.get("tool", "")
        args = context.get("args", {})
        return {
            "role": self.name,
            "action": f"执行 {tool}",
            "args": args,
        }


class ReviewerRole(Role):
    """审查者"""
    
    def __init__(self):
        super().__init__("Reviewer", "结果检查与质量评估")
    
    def act(self, context: Dict[str, Any]) -> Dict[str, Any]:
        result = context.get("result", {})
        success = result.get("success", False)
        return {
            "role": self.name,
            "approved": success,
            "feedback": "结果符合预期" if success else "需要修正",
        }


class MultiRoleOrchestrator:
    """多角色编排器"""
    
    def __init__(self):
        self.planner = PlannerRole()
        self.executor = ExecutorRole()
        self.reviewer = ReviewerRole()
        self.roles = {
            "planner": self.planner,
            "executor": self.executor,
            "reviewer": self.reviewer,
        }
    
    def execute(self, goal: str, tool_call: Dict, tool_result: Dict) -> Dict[str, Any]:
        """执行多角色流程"""
        # 1. 规划
        plan = self.planner.act({"goal": goal})
        
        # 2. 执行
        exec_result = self.executor.act(tool_call)
        
        # 3. 审查
        review = self.reviewer.act({"result": tool_result})
        
        return {
            "plan": plan,
            "execution": exec_result,
            "review": review,
            "approved": review.get("approved", False),
        }


# ========================================
# 4. Memory Retrieval + Experience Reuse
# ========================================

class MemoryRetriever:
    """记忆检索器"""
    
    def __init__(self, episodic: EpisodicMemory, semantic: SemanticMemory):
        self.episodic = episodic
        self.semantic = semantic
    
    def retrieve_experience(self, goal: str) -> Dict[str, Any]:
        """检索相关经验"""
        # 情景记忆
        episodes = self.episodic.recall(goal)
        
        # 语义记忆
        knowledge = self.semantic.recall(goal)
        
        # 计算成功率
        success_rate = self.episodic.get_success_rate(goal)
        
        return {
            "episodes": episodes,
            "knowledge": knowledge,
            "success_rate": success_rate,
            "has_experience": len(episodes) > 0,
        }
    
    def get_advice(self, goal: str) -> List[str]:
        """基于经验给出建议"""
        exp = self.retrieve_experience(goal)
        advice = []
        
        if exp["has_experience"]:
            advice.append(f"历史成功率: {exp['success_rate']:.0%}")
            if exp["episodes"]:
                best = max(exp["episodes"], key=lambda x: x["success"])
                advice.append(f"最佳经验: {best['goal']}")
        else:
            advice.append("无相关经验，需要探索")
        
        return advice


# ========================================
# 5. Cognitive Agent（整合所有组件）
# ========================================

class CognitiveAgent:
    """
    认知 Agent — Phase 3 整合
    
    整合：
    - 反思引擎
    - 多层记忆
    - 多角色协作
    - 经验检索
    """
    
    def __init__(self, llm_provider=None):
        self.reflection_engine = ReflectionEngine()
        self.working_memory = WorkingMemory()
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory()
        self.role_orchestrator = MultiRoleOrchestrator()
        self.memory_retriever = MemoryRetriever(self.episodic_memory, self.semantic_memory)
        self.llm = llm_provider
    
    def process(self, goal: str, steps: List[Dict], tool_calls: List[Dict], success: bool) -> Dict[str, Any]:
        """处理完整任务周期"""
        # 1. 反思
        reflection = self.reflection_engine.reflect(goal, steps, tool_calls, success)
        
        # 2. 存储情景记忆
        self.episodic_memory.store(reflection)
        
        # 3. 存储语义记忆
        for knowledge in reflection.useful_knowledge:
            self.semantic_memory.store(goal, knowledge, "reflection", 0.7)
        
        # 4. 更新工作记忆
        self.working_memory.add({"goal": goal, "success": success, "steps": len(steps)})
        
        return {
            "reflection": reflection.to_dict(),
            "memory_updated": True,
        }
    
    def get_context_for_new_task(self, goal: str) -> Dict[str, Any]:
        """为新任务获取上下文"""
        return {
            "experience": self.memory_retriever.retrieve_experience(goal),
            "advice": self.memory_retriever.get_advice(goal),
            "recent_tasks": self.working_memory.get_context(5),
        }
    
    def multi_role_step(self, goal: str, tool_call: Dict, tool_result: Dict) -> Dict[str, Any]:
        """多角色步骤"""
        return self.role_orchestrator.execute(goal, tool_call, tool_result)


# ========================================
# 6. 便捷接口
# ========================================

def create_cognitive_agent() -> CognitiveAgent:
    """创建认知 Agent"""
    return CognitiveAgent()


if __name__ == "__main__":
    agent = create_cognitive_agent()
    
    # 模拟任务
    goal = "分析项目结构"
    steps = [{"step": 0, "success": True}, {"step": 1, "success": True}]
    tool_calls = [
        {"tool": "list_dir", "args": {"path": "."}, "result": {"success": True}},
        {"tool": "read_file", "args": {"path": "README.md"}, "result": {"success": True}},
    ]
    success = True
    
    # 处理
    result = agent.process(goal, steps, tool_calls, success)
    print("反思结果:")
    print(json.dumps(result["reflection"], ensure_ascii=False, indent=2))
    
    # 获取上下文
    context = agent.get_context_for_new_task("分析代码")
    print(f"\n新任务上下文:")
    print(f"  历史成功率: {context['experience']['success_rate']:.0%}")
    print(f"  建议: {context['advice']}")
    
    # 多角色
    role_result = agent.multi_role_step("test", {"tool": "list_dir"}, {"success": True})
    print(f"\n多角色结果: 审查 {'通过' if role_result['approved'] else '未通过'}")
