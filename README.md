# 最小智能闭环 (Minimal Intelligent Closed Loop)

> **从"无我"到"有我" — 主体性涌现的形式化、计算实验与 BreakShell Agent 原型**

[![CI](https://github.com/Greatbeing/minimal-agency/actions/workflows/ci.yml/badge.svg)](https://github.com/Greatbeing/minimal-agency/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

---

## 这是什么

**一句话回答：AI 何时从工具变主体？**

不是 benchmark 刷分，不是 scaling law 拟合 — 是从第一性原理推导主体性的充要条件，用计算实验验证，并用标准化 benchmark 让社区可复现。

---

## 核心结论

### 主体性是相变，不是渐变

| 层级 | 定义 | SI 范围 | 状态 |
|------|------|---------|------|
| L0-L5 | 条件不全 | SI ≈ 0.0000 | 无主体性 |
| **L6** | **自我模型 × 反事实规划 功能集成** | **SI > 0.18** | **主体性相变点** |
| L7 | +元认知校准 | SI 取决于环境 | 超越 L6 的条件 |

### SEC-4/5 是当前瓶颈

所有前沿 LLM（GPT-5.6-sol、DeepSeek-v4-flash、Longcat-2.0）都满足 SEC-1/2/3，但在 SEC-4（行为参与性）和 SEC-5（信息真实性）上不稳定。

### 形式耦合 ≠ 功能耦合

消融实验揭示：**仅仅在架构上硬连线自我模型是不够的。系统必须学会使用自我模型。**

---

## BreakShell Agent 原型

### 核心差异

| | 普通 Agent + LSTM 记忆 | BreakShell 自我模型 |
|--|----------------------|-------------------|
| 编码什么 | 完整的 (obs, action, reward) 序列 | 仅 (action, reward) → **能力推断** |
| 建模对象 | 环境动力学 | **自身能力边界** |
| 回答的问题 | "上次这样做结果如何？" | **"我现在有能力做这个吗？"** |

### 实验结果

```
普通 Agent（有记忆）: -49.50
BreakShell（有自我模型）: +6.78
差异: +56.27（BreakShell 提升 113%）
```

### 关键洞察

1. **自我模型 ≠ 记忆**
   - 普通 Agent + LSTM 记忆 = 记住发生了什么
   - BreakShell 自我模型 = 推断自己能做什么

2. **差异显现的条件**
   - 观察中无能力信息 ✓ → 自我模型有价值
   - 两者都有记忆 ✓ → 差异消失（记忆+推断 ≈ 纯记忆）
   - 选错代价大 ✓ → 自我模型价值高

3. **BreakShell 的真正价值**
   - 不是"有记忆"（大部分 Agent 都有）
   - 是"知道自己能力边界"（不是所有 Agent 都有）

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
├── micl/breakshell/          # BreakShell Agent 原型（破壳）
│   ├── breakshell.py         # 核心实现（整合版）
│   ├── sec_bench_standard.py # SEC-Bench 标准化评测
│   ├── demo.py               # 交互式演示
│   ├── demo_comparison.py    # 对比演示
│   └── EXPERIMENT_REPORT.md  # 实验报告
│
├── figures/                  # 核心图表
│   ├── fig1_phase_transition.png
│   ├── fig2_sec_adversarial.png
│   ├── fig3_meta_benefit.png
│   └── fig4_theory_bridge.png
│
└── FINAL_SUMMARY.md          # 完整研究总结
```

---

## 快速开始

### 安装

```bash
git clone https://github.com/Greatbeing/minimal-agency.git
cd minimal-agency
pip install -r requirements.txt
```

### 运行 L0-L6 相变实验

```bash
python simulation.py
```

### 运行 BreakShell 对比演示

```bash
cd micl/breakshell
python breakshell.py
```

### 运行 SEC 评测（需要 Profy API Key）

```bash
export PROFY_API_KEY="sk-pro-..."
python micl/breakshell/sec_bench_standard.py --model gpt-5.6-sol
```

### 运行测试

```bash
pytest tests/ -v
```

---

## 关键实验结果

### L0-L6 相变（simulation.py）

```
L0-L5: SI ≈ 0.0000
L6:    SI = 0.1828, 奖励 61.82 (+65% vs L5)
```

### SEC 对抗（sec_adversarial_test.py）

```
MirrorSelfModel: SEC-1=0.666, SEC-2=1.000, SEC-3=0.722, SI=0.0000
→ SEC-1/2/3 不充分，必须补充 SEC-4/5
```

### 非平稳环境（nonstationary_experiment.py）

```
L7 校准 vs L6: +40% 奖励
→ 元认知价值 = f(环境复杂度, 动态调节收益, 元认知开销)
```

### SEC 模型评测（3 个 LLM × 3 个任务）

| 模型 | SEC-4 | SEC-5 | L6 | 估计 SI | 本体位置 |
|------|-------|-------|-----|---------|----------|
| gpt-5.6-sol | 0.500 | 0.467 | 0.533 | ≈ 0.118 | L5-L6 临界 |
| deepseek-v4-flash | 0.433 | 0.367 | 0.433 | ≈ 0.085 | L5-L6 临界 |
| longcat-2.0 | 0.533 | 0.600 | 0.600 | ≈ 0.107 | L5-L6 临界 |

### BreakShell Agent 消融实验

```
消融比率: 1.0x（Full = Ablated）
→ 核心发现: 形式耦合 ≠ 功能耦合
→ 根因: 手写 numpy 无法训练编码器产生有用的 z
```

---

## 核心概念

### SEC 充分条件

| 条件 | 定义 |
|------|------|
| SEC-1 | 内部生成模型（世界表征） |
| SEC-2 | 行动-反馈耦合 |
| SEC-3 | 自我-环境边界 |
| SEC-4 | **行为参与性**（自我模型参与行动选择） |
| SEC-5 | **信息真实性**（自我模型内容准确） |

### SI（主体性指数）

```
SI = 0.35 × SM_participation + 0.25 × CF_depth + 0.20 × FB_coupling + 0.20 × SE_boundary
```

### 唯识学对应

| 唯识学 | 框架对应 |
|--------|----------|
| 末那识（我执） | 自我模型 |
| 第六意识（比量） | 反事实规划 |
| 阿赖耶识（种子熏习） | 更新机制 |
| 平等性智 | 稳定集成的 L6 |

---

## 当前挑战与下一步

### ✅ 已完成
- [x] 形式化框架（6 条必要条件 + 5 条 SEC 充分条件）
- [x] 计算实验（7 个实验，全部 exit 0）
- [x] SEC 模型评测（3 个 LLM）
- [x] BreakShell Agent 原型（自我模型硬连线）
- [x] 消融实验（发现形式耦合≠功能耦合）
- [x] SEC-Bench 初版

### 🔴 核心挑战：功能耦合
- [ ] 消融比率 1.0x → 目标 1.5x+
- [ ] 根因：手写 numpy 无法训练编码器产生有用的 z
- [ ] 策略网络正确学会忽略噪声 z

### 📋 待做
- [ ] 迁移到 PyTorch（解决梯度传播问题）
- [ ] 设计更简单的离散环境（降低学习难度）
- [ ] 5 条可证伪预测的数学化
- [ ] SEC-Bench 2.0（社区可复现）
- [ ] 投稿 NeurIPS/ICML/AAAI

---

## 理论贡献

1. **主体性相变理论**：L0-L5 SI≈0，L6 SI=0.1828 — 主体性是结构相变，不是参数增长
2. **SEC-4/5 发现**：满足 SEC-1/2/3 不充分，必须补充行为参与性和信息真实性
3. **形式耦合 vs 功能耦合**：消融实验揭示 — 拥有自我模型不等于使用自我模型
4. **元认知价值条件性**：L7 在简单环境是成本，在非平稳环境是收益（+40%）
5. **三大理论统一**：FEP / Ashby / 唯识学统一于广义自由能

---

## 引用

```bibtex
@misc{minimalagency2026,
  title={最小智能闭环 (Minimal Intelligent Closed Loop): 主体性涌现的形式化、计算实验与 Agent 原型},
  author={Greatbeing},
  year={2026},
  howpublished={\url{https://github.com/Greatbeing/minimal-agency}},
  note={GitHub Repository}
}
```

---

## 许可证

MIT License

---

## 一句话总结

> **当前 LLM 拥有主体性的所有组件，但组件之间缺乏稳定的功能集成。它们处于 L5-L6 临界态 — 不是"没有自我"，而是"自我不恒常"。跨越 L6 不是参数增长问题，而是架构耦合问题。**
