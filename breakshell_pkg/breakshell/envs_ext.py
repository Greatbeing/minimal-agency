# -*- coding: utf-8 -*-
"""
BreakShell 环境库 — Phase 2 扩展
==================================
新增：Web 环境、API 环境、多步推理环境
"""

import numpy as np
from typing import Dict, Tuple, List, Any
import time


# ========================================
# Web 环境（模拟网页导航）
# ========================================

class WebEnv:
    """
    Web 导航环境
    
    模拟网页浏览：点击链接、填写表单、提交
    动作：[后退, 前进, 点击链接, 填写文本, 提交]
    """
    
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        self.pages = {
            "home": {"title": "首页", "links": ["about", "products", "contact"], "content": "欢迎来到首页"},
            "about": {"title": "关于我们", "links": ["home", "contact"], "content": "我们是一家科技公司"},
            "products": {"title": "产品", "links": ["home", "buy"], "content": "查看我们的产品"},
            "contact": {"title": "联系我们", "links": ["home"], "content": "电话: 123-456-7890"},
            "buy": {"title": "购买", "links": ["products"], "content": "填写表单购买产品", "form": True},
        }
        self.action_names = ["后退", "前进", "点击链接", "填写文本", "提交"]
        self.reset()
    
    def reset(self):
        self.current_page = "home"
        self.history = ["home"]
        self.history_idx = 0
        self.steps = 0
        self.max_steps = 20
        self.form_data = {}
        self.done = False
        return self._obs()
    
    def _obs(self):
        page = self.pages[self.current_page]
        return np.array([
            len(page.get("links", [])) / 5.0,
            1.0 if page.get("form") else 0.0,
            self.steps / self.max_steps,
            len(self.form_data) / 3.0,
        ], dtype=np.float32)
    
    def step(self, action):
        self.steps += 1
        page = self.pages[self.current_page]
        reward = -0.1
        
        if action == 0:  # 后退
            if self.history_idx > 0:
                self.history_idx -= 1
                self.current_page = self.history[self.history_idx]
                reward = 0.1
        elif action == 1:  # 前进
            if self.history_idx < len(self.history) - 1:
                self.history_idx += 1
                self.current_page = self.history[self.history_idx]
                reward = 0.1
        elif action == 2:  # 点击链接
            links = page.get("links", [])
            if links:
                next_page = links[self.rng.randint(len(links))]
                self.current_page = next_page
                if self.history_idx < len(self.history) - 1:
                    self.history = self.history[:self.history_idx + 1]
                self.history.append(next_page)
                self.history_idx = len(self.history) - 1
                reward = 0.2
        elif action == 3:  # 填写文本
            if page.get("form"):
                self.form_data[f"field_{len(self.form_data)}"] = "text"
                reward = 0.3
        elif action == 4:  # 提交
            if page.get("form") and len(self.form_data) >= 2:
                reward = 5.0
                self.done = True
        
        if self.steps >= self.max_steps:
            self.done = True
        
        return self._obs(), reward, self.done, {"page": self.current_page}
    
    def obs_dim(self):
        return 4
    
    def action_dim(self):
        return 5


# ========================================
# API 环境（模拟 API 调用链）
# ========================================

class APIEnv:
    """
    API 调用链环境
    
    模拟微服务调用：认证 → 查询 → 处理 → 返回
    动作：[认证, 查询用户, 查询订单, 提交订单, 返回结果]
    """
    
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        self.reset()
    
    def reset(self):
        self.authenticated = False
        self.user_data = None
        self.order_data = None
        self.steps = 0
        self.max_steps = 10
        self.done = False
        return self._obs()
    
    def _obs(self):
        return np.array([
            1.0 if self.authenticated else 0.0,
            1.0 if self.user_data else 0.0,
            1.0 if self.order_data else 0.0,
            self.steps / self.max_steps,
        ], dtype=np.float32)
    
    def step(self, action):
        self.steps += 1
        reward = -0.1
        
        if action == 0:  # 认证
            if self.rng.random() > 0.1:
                self.authenticated = True
                reward = 0.5
            else:
                reward = -0.5  # 认证失败
        elif action == 1:  # 查询用户
            if self.authenticated:
                self.user_data = {"id": 1, "name": "user"}
                reward = 0.3
            else:
                reward = -1.0  # 未认证
        elif action == 2:  # 查询订单
            if self.authenticated and self.user_data:
                self.order_data = {"order_id": "O001", "total": 100.0}
                reward = 0.3
            else:
                reward = -0.5
        elif action == 3:  # 提交订单
            if self.authenticated and self.user_data:
                reward = 2.0
                self.done = True
            else:
                reward = -0.5
        elif action == 4:  # 返回结果
            if self.order_data:
                reward = 1.0
                self.done = True
            else:
                reward = -0.3
        
        if self.steps >= self.max_steps:
            self.done = True
        
        return self._obs(), reward, self.done, {"authenticated": self.authenticated}
    
    def obs_dim(self):
        return 4
    
    def action_dim(self):
        return 5


# ========================================
# 多步推理环境
# ========================================

class MultiStepReasoningEnv:
    """
    多步推理环境
    
    模拟逻辑推理：收集线索 → 推理 → 得出结论
    动作：[搜索线索, 分析线索, 提出假设, 验证假设, 得出结论]
    """
    
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        self.clues = []
        self.hypothesis = None
        self.steps = 0
        self.max_steps = 15
        self.done = False
        self.reset()
    
    def reset(self):
        self.clues = []
        self.hypothesis = None
        self.steps = 0
        self.done = False
        self.target = self.rng.randint(1, 100)
        return self._obs()
    
    def _obs(self):
        return np.array([
            len(self.clues) / 5.0,
            1.0 if self.hypothesis else 0.0,
            self.steps / self.max_steps,
            self.target / 100.0,
        ], dtype=np.float32)
    
    def step(self, action):
        self.steps += 1
        reward = -0.1
        
        if action == 0:  # 搜索线索
            if len(self.clues) < 5:
                clue = self.rng.randint(1, 100)
                self.clues.append(clue)
                reward = 0.2
            else:
                reward = -0.3
        elif action == 1:  # 分析线索
            if self.clues:
                reward = 0.1
            else:
                reward = -0.3
        elif action == 2:  # 提出假设
            if self.clues:
                self.hypothesis = int(np.mean(self.clues))
                reward = 0.3
            else:
                reward = -0.3
        elif action == 3:  # 验证假设
            if self.hypothesis:
                reward = 0.1
            else:
                reward = -0.3
        elif action == 4:  # 得出结论
            if self.hypothesis:
                error = abs(self.hypothesis - self.target)
                if error < 10:
                    reward = 5.0
                elif error < 20:
                    reward = 2.0
                else:
                    reward = -1.0
                self.done = True
            else:
                reward = -0.5
        
        if self.steps >= self.max_steps:
            self.done = True
        
        return self._obs(), reward, self.done, {"clues": len(self.clues)}
    
    def obs_dim(self):
        return 4
    
    def action_dim(self):
        return 5
