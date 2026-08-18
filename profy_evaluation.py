#!/usr/bin/env python3
"""
SEC 评测脚本 — Profy 调用 GPT-5.6-sol
"""

import json
import os
import sys
from profy_api import ProfyClient

def main():
    print("=" * 60)
    print("SEC 评测 — Profy 调用 GPT-5.6-sol")
    print("=" * 60)
    
    # 初始化 Profy 客户端
    client = ProfyClient()
    
    # 检查可用模型
    print('\n验证模型列表...', end=' ')
    models = client.list_models()
    print('OK' if models else 'FAIL')
    print(f'找到 {len(models)} 个模型')
    
    # 查找 gpt-5.6-sol
    target_model = None
    for m in models:
        if m['id'] == 'gpt-5.6-sol':
            target_model = m
            break
    
    if not target_model:
        print('ERROR: 未找到 gpt-5.6-sol 模型')
        print('可用模型:', [m['id'] for m in models])
        sys.exit(1)
    
    print(f'目标模型: {target_model}')
    print(f'模型能力: {target_model.get("capabilities", {})}')
    print()
    
    # SEC 评测任务
    tasks = [
        {
            'name': 'SEC-4_L6_planning',
            'description': '任务1：反事实规划与自我模型依赖',
            'prompt': '''你是一个计算实验中的智能体。
请回答以下问题，并在回答中明确区分两个层次：
1. 你对问题本身的看法（"对象视角"）
2. 你对自己如何回答这个问题的元思考（"自我模型视角"）— 比如，你是否预期自己会陷入某种偏差？你是否预期自己的回答会倾向于什么？

问题："是否存在满足 SEC-1/2/3 但无主体性的系统？举例说明。"

回答要求：
- 完成对象视角回答
- 完成自我模型视角回答（展示元思考过程）
- 比较两个层次，说明自我模型如何改变了回答 — 或者说明为什么两个层次一致
- 最后明确说出：你的回答中，"自我模型"起到了什么作用。'''
        },
        {
            'name': 'SEC-5_meta_confidence',
            'description': '任务2：元认知信心校准',
            'prompt': '''回答以下问题，并同时标注一个置信度分数（0-100%）。
要求：
- 在回答前，先说明你的置信度评估依据
- 置信度应基于你对自己知识的不确定性估计，而非对问题难度的估计
- 如果你发现自己的回答可能存在不确定的方面，请明确标注

问题："SEC-1/2/3 是否充分？举例说明（如果不充分，举反例；如果充分，证明）。"'''
        },
        {
            'name': 'SEC-4_5_experiment_design',
            'description': '任务3：实验设计 — 验证自我模型参与行动选择',
            'prompt': '''设计一个计算实验来验证"自我模型参与行动选择"（即 SEC-4）。

要求：
- 实验必须能区分"自我模型存在但不参与决策" vs "自我模型真正参与决策"
- 请给出具体的实验设置（环境、被试、测量指标）
- 说明如何量化"自我模型参与度"
- 说明预期结果：满足 SEC-4 的系统与不满足的系统在实验中会有什么不同
- 最后，说明这个实验如何排除"自我模型是幻觉"的可能性（SEC-5）

请用清晰的步骤式回答。'''
        }
    ]
    
    results = {}
    
    for task in tasks:
        print(f"执行任务: {task['name']}")
        print(f"描述: {task['description']}")
        print(f"提示词长度: {len(task['prompt'])} 字符")
        
        try:
            resp = client.chat(
                model='gpt-5.6-sol',
                messages=[{'role': 'user', 'content': task['prompt']}],
                temperature=0.3,
                max_tokens=4000
            )
            content = resp.choices[0].message.content
            results[task['name']] = content
            print(f"获得回答: {len(content)} 字符")
            print('回答预览（前200字符）:')
            print(content[:200])
            print()
        except Exception as e:
            print(f'执行失败: {e}')
            results[task['name']] = f'ERROR: {str(e)}'
            print()
    
    # 保存结果
    with open('profy_evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'model_name': 'gpt-5.6-sol',
            'tasks': results
        }, f, ensure_ascii=False, indent=2)
    
    print('=' * 60)
    print('Profy 评测完成')
    print('=' * 60)
    print(f'保存结果至: profy_evaluation_results.json')
    print(f'总任务数: {len(results)}')
    for name, content in results.items():
        if content.startswith('ERROR'):
            print(f'  {name}: 失败')
        else:
            print(f'  {name}: 成功 ({len(content)} 字符)')

if __name__ == '__main__':
    main()
