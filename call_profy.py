import requests
import json
import os

api_key = os.environ.get('PROFY_API_KEY')
base_url = "https://api.profy.cn/v1"
end_user_id = "hermes-main-user"

prompt = """基于以下三个 GitHub 仓库，制定一个完整的开发方案：

1. **minimal-agency** (BreakShell) — 主体性涌现的形式化、计算实验与 Agent 原型
   - 6 条必要条件 + 5 条 SEC 充分条件
   - L0-L6 相变实验（simulation.py）
   - BreakShell Agent（自我模型硬连线，PyTorch）
   - SEC-Bench v1.3（7 个任务评测）
   - 金融 Agent 对比实验（传统 -0.55 vs BreakShell +0.74）
   - 核心发现：自我模型 ≠ 记忆，能力隐藏时自我模型有价值

2. **yogacara-agent** — 基于唯识理论的进化型 AI Agent 框架
   - 八识计算映射（前五识→意识→末那→阿赖耶）
   - 快慢双循环决策
   - 三性认知过滤（遍计所执/依他起/圆成实）
   - 数字生命（寿元/贪嗔痴/轮回/中阴梦境）
   - 生产级底座（LangGraph/FastAPI/K8s/LLM 集成）
   - 在线对齐（DPO+LoRA+EWC）
   - 安全加固（注入防御/沙箱/限流）

3. **AI Knowledge Bank** — AI 时代知识协作网络
   - 三 vault 知识库
   - Cross-Vault RAG 搜索
   - 社区信号与进化

请制定开发方案，包括：阶段划分、具体任务、产出、验证标准、技术路线、时间估计、关键风险与对策。要求：具体、可执行、有时间节点。"""

resp = requests.post(
    f"{base_url}/chat/completions",
    headers={
        "Authorization": f"Bearer {api_key}",
        "X-Profy-End-User-Id": end_user_id,
        "Content-Type": "application/json",
    },
    json={
        "model": "gpt-5.6-sol",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 4000,
    },
    timeout=180,
)

print(f"Status: {resp.status_code}")
if resp.status_code == 200:
    data = resp.json()
    for choice in data.get("choices", []):
        msg = choice.get("message", {})
        print(msg.get("content", ""))
else:
    print(resp.text)
