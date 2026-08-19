# 最小智能闭环 (Minimal Intelligent Closed Loop)

> **BreakShell Agent — AI Agent 自我模型安全层**

[![CI](https://github.com/Greatbeing/minimal-agency/actions/workflows/ci.yml/badge.svg)](https://github.com/Greatbeing/minimal-agency/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

---

## 这是什么

**一句话回答：让 AI Agent 知道自己能做什么，而不是盲目行动。**

BreakShell 是一个基于自我模型的 AI Agent 安全层。它的核心洞察是：

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
from breakshell import BreakShell, CapabilityEnv

# 创建环境
env = CapabilityEnv(seed=42)

# 创建 Agent
agent = BreakShell(action_dim=3, lr=0.005)

# 训练
agent.train(env, num_episodes=500)

# 评估
reward = agent.evaluate(env, num_episodes=100)
print(f"平均奖励: {reward:+.4f}")

# 保存/加载
agent.save("my_agent")
agent.load("my_agent")
```

### 命令行

```bash
# 训练
breakshell train --env capability --episodes 500 --output my_agent

# 评估
breakshell evaluate --model my_agent --episodes 100

# 对比实验（BreakShell vs 普通 Agent）
breakshell compare --env capability --episodes 500
```

---

## 环境

| 环境 | 描述 | 动作 |
|------|------|------|
| `CapabilityEnv` | 能力匹配环境（能力隐藏） | 保守/适中/激进 |
| `EnergyEnv` | 能量管理环境 | 保守/适中/激进 |
| `FinancialEnv` | 金融市场环境 | 空仓/半仓/满仓 |

---

## 实验结果

```
普通 Agent（有记忆）: -49.50
BreakShell（有自我模型）: +6.78
差异: +56.27（BreakShell 提升 113%）
```

---

## 项目结构

```
最小智能闭环 (Minimal Intelligent Closed Loop)
├── formalization.md          # 形式化框架（6条必要条件 + 5条SEC充分条件）
├── theory_bridge.md          # FEP / Ashby / 唯识学统一桥接
├── simulation.py             # L0-L6 相变实验
├── l7_multiagent_experiment.py  # L7 元认知 + 多主体
├── sec_adversarial_test.py   # SEC 对抗实验（MirrorSelfModel 反例）
├── nonstationary_experiment.py  # 非平稳环境 L7 超越 L6
├── generalization_test.py    # 复杂迷宫泛化
├── paper.tex                 # 学术论文 LaTeX
│
├── breakshell_pkg/           # BreakShell Agent 包
│   ├── breakshell/
│   │   ├── __init__.py       # 包入口
│   │   ├── agent.py          # BreakShell + NormalAgent
│   │   ├── envs.py           # 环境库
│   │   └── cli.py            # 命令行
│   └── setup.py              # pip install
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

## 关键发现

1. **自我模型 ≠ 记忆** — 自我模型的价值取决于环境复杂度
2. **形式耦合 ≠ 功能耦合** — 消融实验揭示拥有自我模型不等于使用自我模型
3. **主体性是相变** — L0-L5 SI≈0，L6 SI=0.1828（simulation.py 验证）
4. **SEC-4/5 是当前瓶颈** — 所有前沿 LLM 在这两条上不稳定

---

## 许可证

MIT License
