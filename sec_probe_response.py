#!/usr/bin/env python3
"""
Profy 响应结构探测 + SEC 任务串行执行
"""

import os, sys, json, requests, traceback

PROFY_BASE_URL = os.environ.get("PROFY_BASE_URL", "https://api.profy.cn/v1")
API_KEY = os.environ.get("PROFY_API_KEY")
END_USER_ID = "hermes-main-user"

def profy_chat(model: str, messages, temperature=0.3, max_tokens=4000):
    """调用 Profy，打印全响应结构，提取命中文本"""
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
        timeout=120,
    )
    try:
        data = resp.json()
    except Exception as e:
        print(f"JSON 解析失败: {e}")
        print(f"响应文本 (前 500 字): {resp.text[:500]}")
        raise

    print(f"完整响应结构: {json.dumps(data, ensure_ascii=False, indent=2)[:1200]}")
    print("-" * 60)

    # 提取文本的通用尝试
    def try_extract(d):
        for key in ["choices", "data", "result", "output", "content"]:
            if key in d and isinstance(d[key], (list, dict, str)):
                if isinstance(d[key], list) and d[key]:
                    first = d[key][0]
                    if isinstance(first, dict):
                        for sub in ["message", "text", "content", "reasoning"]:
                            if sub in first:
                                val = first[sub]
                                if isinstance(val, str):
                                    return val
                                if isinstance(val, dict):
                                    for s2 in ["content", "text"]:
                                        if s2 in val:
                                            return val[s2]
                    elif isinstance(first, str):
                        return first
                elif isinstance(d[key], str):
                    return d[key]
                elif isinstance(d[key], dict):
                    for sub in ["text", "content", "message"]:
                        if sub in d[key]:
                            return d[key][sub]
        return None

    text = try_extract(data)
    if not text:
        print("无法提取文本，列出顶级键:", list(data.keys()))
        # 把 choices 里的第一个元素打印出来看看
        if "choices" in data and data["choices"]:
            print("choices[0]:", json.dumps(data["choices"][0], ensure_ascii=False, indent=2)[:800])
        raise RuntimeError("未发现可提取的文本字段")
    return text

def run_task(model, task):
    print(f"\n{'='*50}\n任务: {task['name']}\n描述: {task['description']}\n提示词长度: {len(task['prompt'])} 字符")
    try:
        content = profy_chat(model, [{"role": "user", "content": task["prompt"]}])
        print(f"✓ 回答获得: {len(content)} 字符")
        print("预览 (前 300 字):")
        print(content[:300])
        return {"status": "success", "length": len(content), "content": content}
    except Exception as e:
        print(f"✗ 失败: {e}")
        traceback.print_exc()
        return {"status": "error", "error": str(e)}

def main():
    if not API_KEY:
        print("PROFY_API_KEY 未设置"); sys.exit(1)
    print("="*60 + "\nSEC 评测 — Profy 串行 (探测版)\n" + "="*60)
    print(f"API Key 前缀: {API_KEY[:8]}...\nBase URL: {PROFY_BASE_URL}\n")

    TARGET_MODELS = ["gpt-5.6-sol", "deepseek-v4-flash-202605"]
    all_results = {}

    for model in TARGET_MODELS:
        print(f"\n{'#'*50}\n# 模型: {model}\n{'#'*50}")
        tasks_out = {}
        for task in TASKS:
            tasks_out[task["name"]] = run_task(model, task)
        all_results[model] = {"model_id": model, "tasks": tasks_out}

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sec_response_probe_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存至 {out_path}")

if __name__ == "__main__":
    main()
