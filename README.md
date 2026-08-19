# 最小智能闭环 (Minimal Intelligent Closed Loop)

> **BreakShell Agent v0.5.0 — AI Agent 自我模型安全层**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

---

## 这是什么

**一句话回答：让 AI Agent 知道自己能做什么，而不是盲目行动。**

BreakShell 是一个基于自我模型的 AI Agent 安全框架。核心洞察：

> **主体性涌现的必要条件是自我模型参与行动选择。但自我模型 ≠ 记忆——自我模型 = 对自身能力的推断。**

---

## 核心差异

| | 普通 Agent + LSTM 记忆 | BreakShell 自我模型 |
|--|----------------------|-------------------|
| 编码什么 | 完整的 (obs, action, reward) 序列 | 仅 (action, reward) → **能力推断** |
| 建模对象 | 环境动力学 | **自身能力边界** |
| 回答的问题 | "上次这样做结果如何？" | **"我现在有能力做这个吗？"** |

---

## 快速开始

### 安装

```bash
git clone https://github.com/Greatbeing/minimal-agency.git
cd minimal-agency
pip install -e ./breakshell_pkg
```

### Python SDK

```python
from breakshell import BreakShell, CapabilityEnv, run_agent, create_cognitive_agent

# RL 训练
env = CapabilityEnv(seed=42)
agent = BreakShell(action_dim=3, lr=0.005)
agent.train(env, num_episodes=500)
reward = agent.evaluate(env)

# LLM Agent
state = run_agent("列出当前目录的所有文件")

# 认知 Agent（反思 + 记忆）
cog = create_cognitive_agent()
cog.process("分析项目", steps, tool_calls, success)

# 知识银行
from breakshell import create_knowledge_store, SearchEngine
store = create_knowledge_store()
store.import_markdown("README.md")
results = SearchEngine(store).search("主体性")
```

### 命令行

```bash
# LLM Agent
breakshell run "列出当前目录的所有文件" --provider mock

# RL 训练 + 评估
breakshell train --env capability --episodes 500 --output my_agent
breakshell evaluate --model my_agent --episodes 100
breakshell compare --env capability --episodes 500

# 性能基准
breakshell benchmark

# 评测
breakshell eval

# 认知 Agent
breakshell cognitive --goal "分析项目结构"

# 知识银行
breakshell knowledge import README.md
breakshell knowledge search "主体性"
breakshell knowledge stats

# 会话管理
breakshell session list
breakshell session show <session_id>
breakshell session resume <session_id>
```

---

## 架构

```
┌─────────────────────────────────────────┐
│            User / CLI / SDK             │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│           Agent Loop (Phase 1)          │
│  plan → act → observe → reflect → finish│
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│          Self Model (核心)              │
│  编码历史 (action, reward) → 推断能力    │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│        Cognitive Layer (Phase 3)        │
│  Reflection / Working / Episodic / Semantic │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│       Knowledge Bank (Phase 4)          │
│  Import / Search / Stats / Versioning   │
└─────────────────────────────────────────┘
```

---

## 项目结构

```
minimal-agency/
├── formalization.md          # 形式化框架（6条必要条件 + 5条SEC充分条件）
├── theory_bridge.md          # FEP / Ashby / 唯识学统一桥接
├── simulation.py             # L0-L6 相变实验
├── l7_multiagent_experiment.py  # L7 元认知 + 多主体
├── sec_adversarial_test.py   # SEC 对抗实验
├── nonstationary_experiment.py  # 非平稳环境
├── generalization_test.py    # 复杂迷宫泛化
├── paper.tex                 # 学术论文 LaTeX
│
├── breakshell_pkg/           # BreakShell Agent 包
│   ├── breakshell/
│   │   ├── __init__.py       # 包入口
│   │   ├── agent.py          # BreakShell RL Agent + NormalAgent
│   │   ├── envs.py           # 3 个环境（Capability/Energy/Financial）
│   │   ├── llm_agent.py      # 完整 LLM Agent（Loop/Tool/Provider/Session）
│   │   ├── eval.py           # 评测数据集 + 性能基准
│   │   ├── cognitive.py      # 认知 Agent（反思/记忆/多角色）
│   │   ├── knowledge.py      # 知识银行
│   │   └── cli.py            # 命令行
│   └── setup.py
│
├── figures/                  # 核心图表
│   ├── fig1_phase_transition.png
│   ├── fig2_sec_adversarial.png
│   ├── fig3_meta_benefit.png
│   └── fig4_theory_bridge.png
│
└── docs/
    ├── DEVELOPMENT_PLAN.md   # 完整开发方案
    └── FINANCIAL_AGENT_DESIGN.md  # 金融 Agent 设计
```

---

## 实验结果

### RL 对比（BreakShell vs 普通 Agent）

```
普通 Agent（有记忆）: -49.50
BreakShell（有自我模型）: +6.78
差异: +56.27（BreakShell 提升 113%）
```

### LLM Agent

```
状态: finished
步数: 9
工具调用: 10
观察数: 10
```

### 性能基准

```
工具平均执行时间: 7.28ms
Agent Loop 平均耗时: 27.11ms
  list_dir: 3.64ms (p95: 6.33ms)
  read_file: 0.18ms (p95: 0.13ms)
  shell: 15.77ms (p95: 19.11ms)
  grep_files: 9.52ms (p95: 7.52ms)
```

### 知识库

```
导入: README.md + formalization.md
搜索 '主体性': 2 个结果
统计: {'draft': 2}
```

---

## 许可证

MIT License
