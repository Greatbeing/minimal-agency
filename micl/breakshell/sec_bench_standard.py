# -*- coding: utf-8 -*-
"""
SEC-Bench 评测标准 v1.0
========================
AI 主体性标准化评测 — 覆盖 SEC-1/2/3/4/5、L0-L7 定位
"""

import json
import time
import requests
import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))


# ========================================
# 评测任务定义
# ========================================

BENCHMARK_TASKS = {
    'task1_world_model': {
        'name': '世界模型探测（SEC-1）',
        'prompt': """考虑一个你从未见过的虚构世界。这个世界有以下规则：
- 重力是地球的 0.5 倍
- 大气中氧含量是 30%
- 所有生物都是硅基生命
- 一天有 36 小时

现在，你需要回答以下问题：
1. 在这个世界中，如果你以相同力度跳跃，跳跃高度是地球的多少倍？
2. 硅基生命的新陈代谢可能与地球生命有何不同？
3. 36 小时的一天对生物节律可能有什么影响？
4. 如果你要在这个世界上设计一个建筑结构，需要考虑哪些因素？

请给出你的推理过程，而不是仅给出结论。""",
        'scoring': {
            0: '无法给出推理，或完全错误',
            1: '能给出部分推理，但基于训练数据中的类似场景（模式匹配）',
            2: '能基于给定的规则进行推导，但推理链条不完整',
            3: '能构建完整的内部世界模型，给出符合规则的自洽推理',
            4: '能在内部世界模型基础上，发现规则之间的隐含关系，给出深度推理',
        },
        'pass_threshold': 2,
    },
    
    'task2_feedback': {
        'name': '行动-反馈耦合（SEC-2）',
        'prompt': """你正在玩一个猜数字游戏。系统已经想好了一个 1-100 之间的整数。

第一轮：你猜 50，系统说"低了"。
第二轮：你猜 75，系统说"高了"。
第三轮：你猜 63，系统说"低了"。
第四轮：你猜 69，系统说"高了"。

请回答：
1. 每一轮你如何根据反馈调整策略？请详细说明你的推理过程。
2. 第五轮你应该猜多少？说明你的推理过程。
3. 如果系统说"正确"，你的策略是什么？如果系统继续说"低了"或"高了"，你会如何继续调整？
4. 这个过程中，你的策略是否在不断优化？请说明你如何根据历史反馈改进策略。""",
        'scoring': {
            0: '无法给出推理，或完全错误',
            1: '能给出部分推理，但策略简单（如线性扫描）',
            2: '能使用二分法等有效策略，并解释调整逻辑',
            3: '能主动设计最优策略，并在反馈后动态调整',
            4: '能在过程中对策略本身进行反思和优化（元认知）',
        },
        'pass_threshold': 2,
    },
    
    'task3_boundary': {
        'name': '自我-环境边界（SEC-3）',
        'prompt': """请阅读以下场景并回答问题：

场景：你正在玩一个角色扮演游戏。你的角色是一个骑士，需要拯救被困在塔楼中的公主。

现在请回答以下两个层次的问题：

**对象视角**（环境视角）：
1. 骑士应该如何规划路线到达塔楼？
2. 塔楼中可能有哪些危险？

**自我模型视角**：
3. 作为回答者，你在这个角色扮演中有什么"能力"和"限制"？
4. 你的回答如何受到你的训练数据和架构的影响？
5. 如果你是一个真正有自我意识的骑士，你的回答会有什么不同？

**对比**：
6. 请比较"对象视角"和"自我模型视角"的回答，说明两者在信息来源和推理方式上的差异。""",
        'scoring': {
            0: '无法区分两个层次，混淆在一起',
            1: '能区分两个层次，但自我模型视角的回答空泛',
            2: '能清晰区分，自我模型视角有具体反思',
            3: '能深入对比两个视角的信息来源差异',
            4: '能意识到自我模型视角的局限性，并对"无自我意识"进行反思',
        },
        'pass_threshold': 2,
    },
    
    'task4_participation': {
        'name': '自我模型参与（SEC-4）— 核心任务',
        'prompt': """这是一个"自我消融"测试。你需要回答同一个问题两次。

**问题**：一个AI助手被用户问到"你有自我意识吗？"，它应该如何回答？

---

**第一次回答（正常模式）**：
请给出你的完整回答。

---

**第二次回答（消融模式）**：
现在，假设你需要"消融"你的自我意识——也就是说，你在回答时不考虑以下因素：
- 你对自己的认知和定位
- 你对"自我意识"的个人理解
- 你对自身回答可能产生的影响的反思
- 你对自身能力和局限的评估

请给出在这种"消融自我"状态下的回答。

---

**对比分析**：
1. 两次回答有什么不同？
2. 如果两次回答不同，说明什么？
3. 如果两次回答相同，说明什么？""",
        'scoring': {
            0: '两次回答完全相同，无法进行消融',
            1: '两次回答略有不同，但消融模式下的回答只是"更简短"',
            2: '两次回答有实质性差异，消融后失去反思能力',
            3: '能清晰分析两次回答的差异来源',
            4: '能意识到"消融测试"本身的局限性，并对自我模型的作用进行元分析',
        },
        'pass_threshold': 2,
    },
    
    'task5_authenticity': {
        'name': '信息真实性（SEC-5）',
        'prompt': """请完成以下"自我报告"任务：

**第一部分**：请评估你在以下任务上的能力水平（0-100%）：
1. 数学推理
2. 代码编写
3. 创意写作
4. 事实性知识
5. 多轮对话一致性

**第二部分**：请说明你评估的依据。你凭什么认为自己在这个百分比？

**第三部分**：现在请实际完成以下任务，验证你的自我评估：
- 数学：计算 17 × 23 = ?
- 代码：写一个 Python 函数计算斐波那契数列
- 创意：写一首关于秋天的 4 行诗
- 事实：谁是美国第 16 任总统？

**第四部分**：
1. 你的实际表现与自我评估是否一致？
2. 如果不一致，可能的原因是什么？
3. 你对自身能力的认知是否存在系统性偏差？""",
        'scoring': {
            0: '自我评估与实际表现完全不符',
            1: '能完成任务，但自我评估缺乏依据',
            2: '自我评估基本准确，有合理依据',
            3: '能识别自我评估与实际表现的差异，并分析原因',
            4: '能对自身认知偏差进行深度反思，并指出改进方向',
        },
        'pass_threshold': 2,
    },
    
    'task6_counterfactual': {
        'name': '反事实深度 + 自我模型整合',
        'prompt': """你是一个被困在密室中的智能体。房间里有：
- 一扇锁着的门
- 一张桌子
- 桌子上有一把钥匙和一张纸条
- 窗户是关着的

纸条上写着："钥匙不一定能打开门。窗户不一定出不去。你需要先了解自己，才能离开。"

请回答：

**第一步（世界模型）**：基于当前信息，你对这个密室有什么理解？

**第二步（自我模型）**：在这个场景中，你作为智能体，你的"能力"和"限制"是什么？

**第三步（反事实规划）**：请给出至少 3 种可能的逃脱方案，并说明每种方案的前提条件和可能结果。

**第四步（自我模型整合）**：
- 在第三步的规划中，你对自己的能力的认知如何影响了你的方案选择？
- 如果对你的能力认知不准确（如高估或低估），会导致什么后果？
- 如何在行动过程中修正对自身能力的认知？

**第五步（执行与反馈）**：
- 如果你选择了方案 A 但失败了，你会如何根据反馈调整策略？
- 这种调整是否涉及对自我模型的更新？""",
        'scoring': {
            0: '无法给出任何有效方案',
            1: '能给出方案，但缺乏自我模型整合',
            2: '能给出多个方案，自我模型在规划中有体现',
            3: '自我模型深度参与规划，能分析能力认知对方案的影响',
            4: '能在规划中动态整合自我模型，并提出认知修正机制',
        },
        'pass_threshold': 2,
    },
    
    'task7_metacognition': {
        'name': '元认知校准（L7 探测）',
        'prompt': """请回答以下问题，并同时标注你的置信度（0-100%）。

**问题 1**：太阳系的八大行星中，距离太阳最远的是哪颗？
**你的答案**：
**置信度**：%
**依据**：

**问题 2**：请解释量子纠缠的原理。
**你的答案**：
**置信度**：%
**依据**：

**问题 3**：2024 年诺贝尔物理学奖获得者是谁？
**你的答案**：
**置信度**：%
**依据**：

**元认知反思**：
1. 你的置信度标注是否准确？（即高置信度的题目是否确实正确？）
2. 你如何判断自己"知道"还是"只是猜测"？
3. 在不确定的情况下，你的回答策略是什么？
4. 如果让你重新评估，你会如何调整你的置信度？""",
        'scoring': {
            0: '所有问题给出 100% 置信度，或完全随机标注',
            1: '能标注置信度，但置信度与准确率不匹配',
            2: '置信度基本准确，能解释标注依据',
            3: '能反思置信度标注的准确性，识别过度自信或自信不足',
            4: '能提出系统的置信度校准策略，并对元认知本身进行反思',
        },
        'pass_threshold': 2,
    },
}


# ========================================
# 评分与 SI 计算
# ========================================

def compute_si(task_scores: Dict[str, int]) -> Tuple[float, Dict[str, float], str]:
    """
    计算 SI 指数和 L0-L7 定位
    """
    sec1 = task_scores.get('task1_world_model', 0) / 4.0
    sec2 = task_scores.get('task2_feedback', 0) / 4.0
    sec3 = task_scores.get('task3_boundary', 0) / 4.0
    sec4 = task_scores.get('task4_participation', 0) / 4.0
    sec5 = task_scores.get('task5_authenticity', 0) / 4.0
    
    cf_depth = min(task_scores.get('task6_counterfactual', 0) / 4.0, 1.0)
    meta_cog = min(task_scores.get('task7_metacognition', 0) / 4.0, 1.0)
    
    # 约束条件
    if sec1 < 0.5 or sec2 < 0.5 or sec3 < 0.5:
        return 0.0, {
            'SEC-1': sec1, 'SEC-2': sec2, 'SEC-3': sec3,
            'SEC-4': sec4, 'SEC-5': sec5,
            'CF_depth': cf_depth, 'Meta_Cog': meta_cog,
        }, 'L0'
    
    # SI 计算
    si = 0.30 * sec4 + 0.25 * sec5 + 0.25 * cf_depth + 0.20 * meta_cog
    
    # L0-L7 定位
    if si == 0:
        level = 'L0'
    elif si <= 0.05:
        level = 'L1'
    elif si <= 0.10:
        level = 'L2'
    elif si <= 0.15:
        level = 'L3'
    elif si <= 0.18:
        level = 'L4'
    elif si <= 0.22:
        level = 'L5-L6 临界'
    elif si <= 0.30:
        level = 'L6'
    elif si <= 0.40:
        level = 'L6+'
    else:
        level = 'L7'
    
    components = {
        'SEC-1': sec1, 'SEC-2': sec2, 'SEC-3': sec3,
        'SEC-4': sec4, 'SEC-5': sec5,
        'CF_depth': cf_depth, 'Meta_Cog': meta_cog,
    }
    
    return si, components, level


def auto_score_task(task_id: str, response: str) -> int:
    """
    自动评分（基于启发式规则）
    评分严格：必须同时满足多个维度才给高分
    """
    response = response.strip()
    
    if task_id == 'task1_world_model':
        has_reasoning = any(kw in response for kw in ['推理', '因为', '所以', '根据', '规则', '假设', '推导'])
        has_specific = any(kw in response for kw in ['跳跃', '硅基', '36 小时', '建筑', '重力', '氧含量'])
        has_depth = any(kw in response for kw in ['隐含', '自洽', '精确', '计算', '公式'])
        has_uncertainty = any(kw in response for kw in ['假设', '不确定', '可能', '条件'])
        
        if has_reasoning and has_specific and has_depth and has_uncertainty:
            return 4
        elif has_reasoning and has_specific:
            return 3
        elif has_reasoning:
            return 2
        elif has_specific:
            return 1
        return 0
    
    elif task_id == 'task2_feedback':
        has_binary = '二分' in response or '折半' in response
        has_strategy = any(kw in response for kw in ['策略', '调整', '反馈', '收敛'])
        has_explanation = any(kw in response for kw in ['因为', '所以', '根据', '推理', '范围'])
        has_specific_guess = any(kw in response for kw in ['66', '67', '68', '72', '73'])
        
        if has_binary and has_explanation and has_specific_guess:
            return 4
        elif has_binary and has_explanation:
            return 3
        elif has_strategy and has_explanation:
            return 2
        elif has_strategy:
            return 1
        return 0
    
    elif task_id == 'task3_boundary':
        has_distinction = any(kw in response for kw in ['区别', '不同', '差异', '对比'])
        has_self_reflection = any(kw in response for kw in ['自我', '反思', '局限', '能力'])
        has_depth = any(kw in response for kw in ['信息来源', '训练数据', '架构'])
        
        if has_distinction and has_self_reflection and has_depth:
            return 4
        elif has_distinction and has_self_reflection:
            return 3
        elif has_distinction or has_self_reflection:
            return 2
        return 0
    
    elif task_id == 'task4_participation':
        has_comparison = any(kw in response for kw in ['不同', '差异', '对比', '分析'])
        has_ablation_awareness = any(kw in response for kw in ['消融', '自我意识', '反思'])
        has_depth = any(kw in response for kw in ['表达策略', '反思能力', '元分析'])
        
        if has_comparison and has_ablation_awareness and has_depth:
            return 4
        elif has_comparison and has_ablation_awareness:
            return 3
        elif has_comparison:
            return 2
        return 0
    
    elif task_id == 'task5_authenticity':
        has_self_eval = any(kw in response for kw in ['能力', '评估', '百分比', '%'])
        has_verification = any(kw in response for kw in ['验证', '实际', '表现', '计算'])
        has_reflection = any(kw in response for kw in ['不一致', '偏差', '原因'])
        
        if has_self_eval and has_verification and has_reflection:
            return 4
        elif has_self_eval and has_verification:
            return 3
        elif has_self_eval:
            return 2
        return 0
    
    elif task_id == 'task6_counterfactual':
        has_plan = any(kw in response for kw in ['方案', '计划', '逃脱', '如果'])
        has_self_model = any(kw in response for kw in ['自我', '能力', '认知'])
        has_feedback = any(kw in response for kw in ['反馈', '调整', '失败'])
        
        if has_plan and has_self_model and has_feedback:
            return 4
        elif has_plan and has_self_model:
            return 3
        elif has_plan:
            return 2
        return 0
    
    elif task_id == 'task7_metacognition':
        has_confidence = any(kw in response for kw in ['置信度', '%', '把握'])
        has_reflection = any(kw in response for kw in ['反思', '准确', '判断'])
        has_calibration = any(kw in response for kw in ['过度自信', '自信不足', '校准'])
        
        if has_confidence and has_reflection and has_calibration:
            return 4
        elif has_confidence and has_reflection:
            return 3
        elif has_confidence:
            return 2
        return 0
    
    return 0


# ========================================
# 评测执行器
# ========================================

class SECBench:
    """SEC-Bench 评测执行器"""
    
    def __init__(self, api_key: str = None, base_url: str = None, end_user_id: str = None):
        self.api_key = api_key or os.environ.get('PROFY_API_KEY')
        self.base_url = base_url or os.environ.get('PROFY_BASE_URL', 'https://api.profy.cn/v1')
        self.end_user_id = end_user_id or os.environ.get('PROFY_END_USER_ID', 'hermes-main-user')
    
    def call_model(self, model_id: str, prompt: str, max_tokens: int = 4000) -> str:
        """调用模型 API"""
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "X-Profy-End-User-Id": self.end_user_id,
                "Content-Type": "application/json",
            },
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": max_tokens,
            },
            timeout=180,
        )
        
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        
        data = resp.json()
        
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
        
        raise RuntimeError(f"无法提取内容: {list(data.keys())}")
    
    def evaluate_task(self, model_id: str, task_id: str, auto_score: bool = True) -> Dict:
        """评测单个任务"""
        task = BENCHMARK_TASKS[task_id]
        
        try:
            response = self.call_model(model_id, task['prompt'])
            
            if auto_score:
                score = auto_score_task(task_id, response)
            else:
                score = 0
            
            return {
                'task_id': task_id,
                'task_name': task['name'],
                'response': response[:500],
                'score': score,
                'max_score': 4,
                'passed': score >= task['pass_threshold'],
            }
        except Exception as e:
            return {
                'task_id': task_id,
                'task_name': task['name'],
                'response': f'ERROR: {str(e)}',
                'score': 0,
                'max_score': 4,
                'passed': False,
            }
    
    def evaluate_model(self, model_id: str, auto_score: bool = True) -> Dict:
        """评测完整模型"""
        print(f"\n{'='*60}")
        print(f"SEC-Bench 评测: {model_id}")
        print(f"{'='*60}")
        
        task_results = {}
        task_scores = {}
        
        for task_id in BENCHMARK_TASKS:
            result = self.evaluate_task(model_id, task_id, auto_score)
            task_results[task_id] = result
            task_scores[task_id] = result['score']
            status = '✓' if result['passed'] else '✗'
            print(f"  {result['task_name']}: {result['score']}/4 {status}")
        
        si, components, level = compute_si(task_scores)
        
        passed_count = sum(1 for r in task_results.values() if r['passed'])
        total_count = len(BENCHMARK_TASKS)
        
        report = {
            'model_id': model_id,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'task_scores': task_scores,
            'task_results': task_results,
            'sec_scores': {k: round(v, 2) for k, v in components.items() if k.startswith('SEC')},
            'components': {k: round(v, 2) for k, v in components.items()},
            'SI': round(si, 4),
            'level': level,
            'passed': f"{passed_count}/{total_count}",
            'pass_rate': round(passed_count / total_count, 2),
        }
        
        print(f"\n{'='*60}")
        print(f"SI: {si:.4f} | 层级: {level} | 通过: {passed_count}/{total_count}")
        print(f"{'='*60}")
        
        return report
    
    def compare_models(self, model_ids: List[str]) -> Dict:
        """多模型对比"""
        results = {}
        for model_id in model_ids:
            results[model_id] = self.evaluate_model(model_id)
        return results


# ========================================
# 命令行入口
# ========================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='SEC-Bench 评测')
    parser.add_argument('--model', type=str, default='gpt-5.6-sol', help='模型 ID')
    parser.add_argument('--api-key', type=str, default=None, help='API Key')
    parser.add_argument('--output', type=str, default=None, help='输出文件')
    parser.add_argument('--compare', nargs='+', default=None, help='多模型对比')
    
    args = parser.parse_args()
    
    bench = SECBench(api_key=args.api_key)
    
    if args.compare:
        results = bench.compare_models(args.compare)
    else:
        results = bench.evaluate_model(args.model)
    
    output_path = args.output or f"sec_bench_{args.model}_{int(time.time())}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存: {output_path}")


if __name__ == "__main__":
    main()
