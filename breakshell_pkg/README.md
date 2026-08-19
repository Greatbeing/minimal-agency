# BreakShell — AI Agent 自我模型安全层

> 让 Agent 知道自己能做什么，而不是盲目行动

## 安装

```bash
pip install -e .
```

## 快速开始

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

## 环境

| 环境 | 描述 | 动作 |
|------|------|------|
| `CapabilityEnv` | 能力匹配环境（能力隐藏） | 保守/适中/激进 |
| `EnergyEnv` | 能量管理环境 | 保守/适中/激进 |
| `FinancialEnv` | 金融市场环境 | 空仓/半仓/满仓 |

## 核心概念

- **自我模型**：编码历史 (action, reward) → 推断能力边界
- **SEC-Bench**：主体性标准化评测
- **L0-L7**：主体性层级定位

## 论文

基于本项目的主体性形式化框架，正在投稿中。

## 许可证

MIT
