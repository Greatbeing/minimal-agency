# 三项目开发方案（GPT-5.6-sol 制定）

> 基于 minimal-agency / yogacara-agent / AI Knowledge Bank 三个仓库的完整开发方案

---

## 一、项目定位与依赖关系

| 项目 | 定位 | 首要目标 | 依赖关系 |
|---|---|---|---|
| `minimal-agency / BreakShell` | 极简 Agent 原型与交互接口 | 验证 Agent Loop、工具调用和可观测性 | 基础项目，可独立开发 |
| `yogacara-agent` | 基于唯识理论的 Agent 框架 | 建立感知、记忆、认知、行动、自省模型 | 可复用 BreakShell 的 Agent 能力 |
| `AI Knowledge Bank` | AI 时代知识协作网络 | 沉淀 Agent 运行数据、知识、经验和协作结果 | 依赖前两个项目提供 Agent 与知识处理能力 |

推荐顺序：
```
BreakShell → yogacara-agent → AI Knowledge Bank
```

---

## 二、总体阶段划分

| 阶段 | 时间 | 主要目标 |
|---|---:|---|
| 阶段 0：仓库治理与需求收敛 | 第 1-2 周 | 统一架构、文档、开发规范和验收标准 |
| 阶段 1：BreakShell MVP | 第 3-6 周 | 完成可运行的最小 Agent |
| 阶段 2：BreakShell 工程化 | 第 7-9 周 | 增加工具系统、记忆、评测和可观测性 |
| 阶段 3：yogacara-agent 核心框架 | 第 10-14 周 | 建立唯识概念映射下的 Agent Runtime |
| 阶段 4：知识银行 MVP | 第 15-19 周 | 完成知识采集、检索、溯源和协作 |
| 阶段 5：三项目集成与开放测试 | 第 20-24 周 | 联调、评测、部署、文档和社区发布 |

---

## 三、阶段 0 详细计划

### 时间：第 1-2 周

### 任务 1：统一仓库结构

每个仓库至少包含：
```
/
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── docs/
├── examples/
├── tests/
├── src/
├── scripts/
└── .github/
    ├── workflows/
    ├── ISSUE_TEMPLATE/
    └── PULL_REQUEST_TEMPLATE.md
```

### 任务 2：明确项目边界

需要分别写清楚：
- 项目解决什么问题
- 不解决什么问题
- 目标用户是谁
- 当前阶段不承诺哪些能力
- MVP 的可验证指标
- 三个项目之间的接口边界

### 任务 3：统一基础技术决策

- 语言：Python 3.11+
- 包管理：uv 或 Poetry
- API：FastAPI
- 数据校验：Pydantic
- 测试：pytest
- 异步任务：先 asyncio
- 数据库：PostgreSQL
- 向量检索：pgvector
- 缓存：Redis
- 本地开发：Docker Compose
- CI：GitHub Actions
- 代码质量：Ruff、MyPy、pre-commit

### 阶段产出

- 三个项目的 README 初版
- 项目关系图
- 技术选型文档
- MVP 验收标准
- Issue、PR、版本发布规范
- 第一个公共开发路线图

---

## 四、阶段 1 详细计划（BreakShell MVP）

### 时间：第 3-6 周

### Agent Loop 最小结构：
```python
class AgentState:
    goal: str
    messages: list
    observations: list
    tool_calls: list
    status: str
```

### MVP 工具：
- 文件读取
- 文件写入
- Shell 命令执行（带白名单）
- HTTP 请求
- 简单 Python 函数调用

### CLI 命令：
```bash
breakshell run "分析当前目录中的项目结构"
breakshell tool list
breakshell session list
breakshell config show
```

### 验收标准：
- 能完成至少 5 类简单任务
- Agent Loop 能处理工具调用和错误重试
- 支持至少一个云端模型和一个本地模型
- 所有工具调用有日志
- 具备单元测试和 3 个端到端测试
- Shell 工具不能默认执行高风险命令

---

## 五、阶段 3 详细计划（yogacara-agent）

### 时间：第 10-14 周

### 概念映射：

| 唯识概念 | 工程模块 | 职责 |
|---|---|---|
| 前五识 | Perception | 接收文本、文件、图像、工具结果等输入 |
| 第六识 | Cognition | 任务理解、推理、规划和决策 |
| 第七识 | Self Model | 维护当前 Agent 的身份、目标、偏好和边界 |
| 第八识 | Long-term Memory | 保存长期知识、经验、记忆和潜在模式 |
| 种子 | Memory Item / Capability | 可被未来任务激活的知识、技能或经验 |
| 现行 | Execution | 将内部状态转化为工具调用和外部行动 |
| 熏习 | Learning / Consolidation | 从任务结果中提炼可复用经验 |
| 反省 | Reflection | 评估过程、错误、偏差和结果质量 |

---

## 六、阶段 4 详细计划（AI Knowledge Bank）

### 时间：第 15-19 周

### 核心功能：
- 知识对象（KnowledgeItem）统一 schema
- Markdown/PDF/网页/GitHub/Agent 运行记录导入
- 混合检索（关键词 + 向量 + 元数据）
- 知识溯源（原始来源/提取时间/修改记录）
- 知识质量机制（draft/reviewing/verified/deprecated）

---

## 七、推荐时间表

| 周期 | BreakShell | yogacara-agent | AI Knowledge Bank |
|---|---|---|---|
| 第 1-2 周 | 仓库整理 | 概念和边界设计 | 数据模型设计 |
| 第 3-4 周 | Agent Loop | 技术调研 | 原型数据结构 |
| 第 5-6 周 | 工具和 CLI | 接口草案 | 导入原型 |
| 第 7-9 周 | 记忆、权限、评测 | 复用 Runtime | 检索技术验证 |
| 第 10-12 周 | 维护和修复 | Cognitive State、Memory | API 设计 |
| 第 13-14 周 | 集成支持 | Reflection、多 Agent | Agent 接入 |
| 第 15-17 周 | 集成测试 | 知识调用 | 导入和搜索 |
| 第 18-19 周 | 端到端示例 | 反思沉淀 | 溯源、版本、审核 |
| 第 20-22 周 | 稳定性优化 | 评测优化 | 协作界面 |
| 第 23-24 周 | 发布准备 | 发布准备 | 发布准备 |

---

## 八、关键里程碑

| 里程碑 | 时间 | 产出 |
|--------|------|------|
| M1 | 第 2 周 | 职责清晰、技术栈确定 |
| M2 | 第 6 周 | BreakShell 完整 Agent Loop |
| M3 | 第 9 周 | BreakShell v0.1 |
| M4 | 第 14 周 | yogacara-agent Alpha |
| M5 | 第 19 周 | Knowledge Bank MVP |
| M6 | 第 24 周 | 三项目端到端联通 + Beta 发布 |

---

## 九、设计原则

1. **先事件化，再智能化** — 所有 Agent 行为先转化为标准事件
2. **先结构化状态，再增加自主性** — 状态必须可观察、可恢复、可重放
3. **先混合检索，再建设知识图谱** — PostgreSQL + pgvector 起步
4. **哲学概念转化为可测试接口** — 每个概念都要有量化评测
5. **用评测数据驱动迭代** — 每个版本都有固定基线

---

## 十、结论

> **最稳妥的推进路径：先完成 BreakShell 的可运行 Agent Loop → 抽象为 yogacara-agent 的认知运行时 → 将运行过程和经验沉淀到 AI Knowledge Bank → 通过真实任务验证知识是否能改善 Agent 表现。**

> **第一阶段不要追求完整的自主智能或复杂哲学建模，优先确保每个模块都有清晰接口、可重复运行、可观测日志和可量化评测。**
