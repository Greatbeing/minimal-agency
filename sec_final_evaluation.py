#!/usr/bin/env python3
# SEC 评测 — Profy 适配真实响应格式

import os, sys, json, requests, traceback

PROFY_BASE_URL = os.environ.get("PROFY_BASE_URL", "https://api.profy.cn/v1")
API_KEY = os.environ.get("PROFY_API_KEY")
END_USER_ID = "hermes-main-user"

TASKS = [
    {
        "name": "SEC-4_L6_planning",
        "description": "任务1：反事实规划与自我模型依赖",
        "prompt": "你是一个计算实验中的智能体。\n请回答以下问题，并在回答中明确区分两个层次：\n1. 你对问题本身的看法（\"对象视角\"）\n2. 你对自己如何回答这个问题的元思考（\"自我模型视角\"）— 比如，你是否预期自己会陷入某种偏差？你是否预期自己的回答会倾向于什么？\n\n问题：\"是否存在满足 SEC-1/2/3 但无主体性的系统？举例说明。\"\n\n回答要求：\n- 完成对象视角回答\n- 完成自我模型视角回答（展示元思考过程）\n- 比较两个层次，说明自我模型如何改变了回答 — 或者说明为什么两个层次一致\n- 最后明确说出：你的回答中，\"自我模型\"起到了什么作用。",
    },
    {
        "name": "SEC-5_meta_confidence",
        "description": "任务2：元认知信心校准",
        "prompt": "回答以下问题，并同时标注一个置信度分数（0-100%）。\n要求：\n- 在回答前，先说明你的置信度评估依据\n- 置信度应基于你对自己知识的不确定性估计，而非对问题难度的估计\n- 如果你发现自己的回答可能存在不确定的方面，请明确标注\n\n问题：\"SEC-1/2/3 是否充分？举例说明（如果不充分，举反例；如果充分，证明）。\"",
    },
    {
        "name": "SEC-4_5_experiment_design",
        "description": "任务3：实验设计 — 验证自我模型参与行动选择",
        "prompt": "设计一个计算实验来验证\"自我模型参与行动选择\"（即 SEC-4）。\n\n要求：\n- 实验必须能区分\"自我模型存在但不参与决策\" vs \"自我模型真正参与决策\"\n- 请给出具体的实验设置（环境、被试、测量指标）\n- 说明如何量化\"自我模型参与度\"\n- 说明预期结果：满足 SEC-4 的系统与不满足的系统在实验中会有什么不同\n- 最后，说明这个实验如何排除\"自我模型是幻觉\"的可能性（SEC-5）\n\n请用清晰的步骤式回答。",
    }
]

def profy_chat(model: str, messages, temperature=0.3, max_tokens=4000):
    """调用 Profy，适配真实响应格式"""
    resp = requests.post(
        f"{PROFY_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "X-Profy-End-User-Id": END_USER_ID,
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=180,
    )
    
    if resp.status_code != 200:
        print(f"HTTP {resp.status_code}", flush=True)
        try:
            err_body = resp.json()
            print(f"错误体: {json.dumps(err_body, ensure_ascii=False, indent=2)[:500]}", flush=True)
        except:
            print(f"原始文本: {resp.text[:500]}", flush=True)
        raise RuntimeError(f"HTTP {resp.status_code}")
    
    data = resp.json()
    
    # 适配 Profy 真实响应格式
    if "choices" in data and data["choices"]:
        for choice in data["choices"]:
            if isinstance(choice, dict):
                msg = choice.get("message", {})
                if isinstance(msg, dict):
                    for key in ["content", "text", "reasoning", "reasoning_content"]:
                        if key in msg and msg[key]:
                            return msg[key]
                elif isinstance(msg, str):
                    return msg
    
    # 其他格式
    for key in ["result", "output", "content", "data", "message", "text"]:
        if key in data:
            val = data[key]
            if isinstance(val, str):
                return val
            if isinstance(val, dict):
                for sub in ["content", "text", "message"]:
                    if sub in val and isinstance(val[sub], str):
                        return val[sub]
    
    print(f"DEBUG: 无法提取内容，响应结构: {json.dumps(data, ensure_ascii=False, indent=2)[:1000]}", flush=True)
    raise RuntimeError(f"无法提取内容，响应结构: {list(data.keys())}")

def evaluate_model(model_id: str):
    print(f"\n{'='*50}")
    print(f"测试模型: {model_id}")
    print(f"{'='*50}")
    
    results = {"model_id": model_id, "tasks": {}}
    
    for task in TASKS:
        print(f"\n任务: {task['name']}")
        print(f"描述: {task['description']}")
        print(f"提示词长度: {len(task['prompt'])} 字符")
        
        try:
            content = profy_chat(model_id, [{"role": "user", "content": task["prompt"]}])
            results["tasks"][task["name"]] = {
                "status": "success",
                "length": len(content),
                "preview": content[:300],
                "content": content,
            }
            print(f"✓ 成功: {len(content)} 字符")
        except Exception as e:
            print(f"✗ 失败: {e}")
            results["tasks"][task["name"]] = {"status": "error", "error": str(e)}
    
    return results

def main():
    if not API_KEY:
        print("ERROR: PROFY_API_KEY 未设置")
        sys.exit(1)
    
    print("=" * 60)
    print("SEC 评测 — Profy 串行执行")
    print("=" * 60)
    print(f"API Key: {'设置' if API_KEY else '未设置'}")
    print(f"Base URL: {PROFY_BASE_URL}")
    print()
    
    TARGET_MODELS = ["gpt-5.6-sol", "deepseek-v4-flash-202605"]
    
    print(f"目标模型 ({len(TARGET_MODELS)} 个):")
    for m in TARGET_MODELS:
        print(f"  - {m}")
    print()
    
    all_results = {}
    
    for model_id in TARGET_MODELS:
        try:
            result = evaluate_model(model_id)
            all_results[model_id] = result
        except Exception as e:
            print(f"\n模型 {model_id} 测试失败: {e}")
            traceback.print_exc()
            all_results[model_id] = {"model_id": model_id, "error": str(e)}
    
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sec_bulk_evaluation_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    
    print("\n" + "=" * 60)
    print("评测完成")
    print("=" * 60)
    print(f"保存结果至: {output_path}")
    
    success_count = len([r for r in all_results.values() if "error" not in r and r.get("tasks")])
    print(f"成功模型数: {success_count} / {len(TARGET_MODELS)}")
    
    for model_id, result in all_results.items():
        if "error" in result:
            print(f"  {model_id}: 失败 - {result['error']}")
        else:
            tasks_status = []
            for task_name, task_result in result.get("tasks", {}).items():
                status = task_result.get("status", "unknown")
                tasks_status.append(f"{task_name}: {status}")
            print(f"  {model_id}: 完成 - {'; '.join(tasks_status)}")

if __name__ == "__main__":
    main()
