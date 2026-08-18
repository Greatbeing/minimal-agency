#!/usr/bin/env python3
"""Profy 401 诊断：打印原始错误体"""

import os, sys, requests, json

PROFY_BASE_URL = os.environ.get("PROFY_BASE_URL", "https://api.profy.cn/v1")
API_KEY = os.environ.get("PROFY_API_KEY")
END_USER_ID = "hermes-main-user"

def main():
    if not API_KEY:
        print("PROFY_API_KEY 未设置")
        sys.exit(1)
    print(f"API Key 长度: {len(API_KEY)}", flush=True)
    print(f"API Key 前缀: {API_KEY[:10]}...", flush=True)
    print(f"Base URL: {PROFY_BASE_URL}", flush=True)

    # 1) 测试 GET /v1/models（无需模型特定参数）
    print("\n[1] GET /v1/models", flush=True)
    r = requests.get(f"{PROFY_BASE_URL}/models", headers={
        "Authorization": f"Bearer {API_KEY}",
        "X-Profy-End-User-Id": END_USER_ID,
    }, timeout=30)
    print(f"Status: {r.status_code}", flush=True)
    try:
        body = r.json()
        print(f"Body keys: {list(body.keys())}", flush=True)
        print(f"Body: {json.dumps(body, ensure_ascii=False, indent=2)[:500]}", flush=True)
    except Exception as e:
        print(f"非 JSON 响应: {e}", flush=True)
        print(f"原始文本: {r.text[:500]}", flush=True)

    # 2) 测试 POST /v1/chat/completions（文本模型，极短提示）
    print("\n[2] POST /v1/chat/completions", flush=True)
    payload = {
        "model": "gpt-5.6-sol",
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 10,
    }
    r = requests.post(f"{PROFY_BASE_URL}/chat/completions", headers={
        "Authorization": f"Bearer {API_KEY}",
        "X-Profy-End-User-Id": END_USER_ID,
        "Content-Type": "application/json",
    }, json=payload, timeout=60)
    print(f"Status: {r.status_code}", flush=True)
    try:
        body = r.json()
        print(f"Body keys: {list(body.keys())}", flush=True)
        print(f"Body: {json.dumps(body, ensure_ascii=False, indent=2)[:1000]}", flush=True)
    except Exception as e:
        print(f"非 JSON 响应: {e}", flush=True)
        print(f"原始文本: {r.text[:500]}", flush=True)

if __name__ == "__main__":
    main()
