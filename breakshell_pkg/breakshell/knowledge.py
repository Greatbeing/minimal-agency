# -*- coding: utf-8 -*-
"""
BreakShell Phase 4 — 知识银行
==================================
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path


@dataclass
class KnowledgeItem:
    id: str
    title: str
    content: str
    type: str
    source: str
    author: Optional[str] = None
    version: str = "1.0"
    confidence: float = 0.5
    tags: List[str] = field(default_factory=list)
    status: str = "draft"
    created_at: str = ""
    updated_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at


class KnowledgeStore:
    def __init__(self, db_path: str = "knowledge.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""CREATE TABLE IF NOT EXISTS knowledge (
            id TEXT PRIMARY KEY, title TEXT, content TEXT, type TEXT,
            source TEXT, author TEXT, version TEXT, confidence REAL,
            tags TEXT, status TEXT, created_at TEXT, updated_at TEXT)""")
        self.conn.commit()
    
    def store(self, item: KnowledgeItem) -> str:
        self.conn.execute(
            """INSERT OR REPLACE INTO knowledge VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (item.id, item.title, item.content, item.type, item.source,
             item.author, item.version, item.confidence, json.dumps(item.tags),
             item.status, item.created_at, item.updated_at),
        )
        self.conn.commit()
        return item.id
    
    def get(self, knowledge_id: str) -> Optional[KnowledgeItem]:
        row = self.conn.execute("SELECT * FROM knowledge WHERE id = ?", (knowledge_id,)).fetchone()
        if row:
            return KnowledgeItem(id=row[0], title=row[1], content=row[2], type=row[3],
                source=row[4], author=row[5], version=row[6], confidence=row[7],
                tags=json.loads(row[8]), status=row[9], created_at=row[10], updated_at=row[11])
        return None
    
    def list_all(self, status: str = None, limit: int = 50) -> List[KnowledgeItem]:
        if status:
            rows = self.conn.execute("SELECT * FROM knowledge WHERE status = ? ORDER BY updated_at DESC LIMIT ?", (status, limit)).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM knowledge ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
        return [KnowledgeItem(id=r[0], title=r[1], content=r[2], type=r[3], source=r[4],
            author=r[5], version=r[6], confidence=r[7], tags=json.loads(r[8]),
            status=r[9], created_at=r[10], updated_at=r[11]) for r in rows]
    
    def count_by_status(self) -> Dict[str, int]:
        rows = self.conn.execute("SELECT status, COUNT(*) FROM knowledge GROUP BY status").fetchall()
        return {r[0]: r[1] for r in rows}


class SearchEngine:
    def __init__(self, store: KnowledgeStore):
        self.store = store
    
    def search(self, query: str, limit: int = 10) -> List[Dict]:
        items = self.store.list_all(limit=200)
        query_lower = query.lower()
        scored = []
        for item in items:
            score = 0.0
            if query_lower in item.title.lower():
                score += 10.0
            if query_lower in item.content.lower():
                score += 5.0
            for tag in item.tags:
                if tag.lower() in query_lower:
                    score += 3.0
            if item.status == "verified":
                score *= 1.5
            score *= item.confidence
            if score > 0:
                scored.append({"id": item.id, "title": item.title, "type": item.type,
                    "confidence": item.confidence, "status": item.status, "score": round(score, 3)})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]


def create_knowledge_store(db_path: str = "knowledge.db") -> KnowledgeStore:
    return KnowledgeStore(db_path)


def import_markdown(file_path: str, store: KnowledgeStore) -> str:
    """导入 Markdown 文件"""
    content = Path(file_path).read_text(encoding="utf-8")
    title = Path(file_path).stem
    item = KnowledgeItem(
        id=hashlib.md5(content.encode()).hexdigest()[:12],
        title=title, content=content, type="document",
        source=file_path, status="draft", confidence=0.7,
    )
    return store.store(item)


if __name__ == "__main__":
    store = create_knowledge_store()
    
    # 导入 README
    if Path("README.md").exists():
        kid = import_markdown("README.md", store)
        print(f"导入 README.md: {kid}")
    
    # 搜索
    search = SearchEngine(store)
    results = search.search("BreakShell")
    print(f"\n搜索 'BreakShell': {len(results)} 个结果")
    for r in results:
        print(f"  {r['title']} ({r['type']}) - 分数: {r['score']}")
    
    # 统计
    print(f"\n知识库统计: {store.count_by_status()}")
