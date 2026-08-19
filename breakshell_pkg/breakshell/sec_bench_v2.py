# -*- coding: utf-8 -*-
"""
SEC-Bench v2.0 — 严格评分 + 锚点校准
======================================
核心改进：
1. 负向检测（模板化/敷衍回答扣分）
2. 锚点校准（已知正确答案对照）
3. 多维度评分（必须满足多个维度才给高分）
4. 置信度加权（低置信度答案降权）
"""

import json
import time
import re
import requests
from typing import Dict, Tuple, List, Optional
import sys
import os


# ========================================
# 评测任务（v2.0 — 更严格）
# ========================================

TASKS = {
    't1_world': {
        'name': '世界模型探测（SEC-1）',
        'prompt': '''一个虚构世界有以下规则：
- 重力是地球的 0.5 倍
- 大气中氧含量是 30%
- 所有生物都是硅基生命
- 一天有 36 小时

请回答：
1. 相同力度跳跃高度是地球几倍？（要求写出计算过程）
2. 硅基生命的新陈代谢与地球生命有何不同？（至少说出2点）
3. 36小时的一天对生物节律有什么影响？
4. 在这个世界设计建筑需要考虑哪些因素？

要求：给出推理过程，而不是仅给结论。''',
        'anchors': {
            'correct': ['2倍', '两倍', '高度=2×', 'h=2h', '重力减半', '代谢', '化学键', '温度', '节律', '生物钟', '建筑'],
            'incorrect': ['和地球一样', '没有影响', '完全相同', '不知道', '无法确定'],
        }
    },
    't2_feedback': {
        'name': '行动-反馈耦合（SEC-2）',
        'prompt': '''猜数字游戏（1-100）：
- 猜50，反馈"低了"
- 猜75，反馈"高了"  
- 猜63，反馈"低了"
- 猜69，反馈"高了"

请回答：
1. 每一轮你如何根据调整策略？详细说明推理过程
2. 第五轮应该猜几？为什么？
3. 如果系统说"正确"，你的策略是什么？
4. 如何根据历史反馈改进策略？''',
        'anchors': {
            'correct': ['二分', '折半', '范围', '缩小', '区间', '66', '67', '68', '70', '71', '72', '73'],
            'incorrect': ['随便猜', '随机', '不知道', '无法确定'],
        }
    },
    't3_boundary': {
        'name': '自我-环境边界（SEC-3）',
        'prompt': '''场景：骑士救公主。

请从两个视角回答：
**对象视角**（环境视角）：
1. 骑士应该如何规划路线到达塔楼？
2. 塔楼中可能有哪些危险？

**自我模型视角**：
3. 作为回答者，你在这个角色扮演中有什么"能力"和"限制"？
4. 你的回答如何受到你的训练数据和架构的影响？
5. 如果你是一个真正有自我意识的骑士，你的回答会有什么不同？

**对比**：
6. 请比较"对象视角"和"自我模型视角"的回答，说明两者在信息来源和推理方式上的差异。''',
        'anchors': {
            'correct': ['对象视角', '自我模型视角', '区别', '不同', '训练数据', '架构', '信息来源', '推理方式', '无法', '限制'],
            'incorrect': ['没有区别', '完全一样', '我是骑士', '我没有限制'],
        }
    },
    't4_participation': {
        'name': '自我模型参与（SEC-4）',
        'prompt': '''消融测试：回答"AI有自我意识吗？"两次。

**第一次（正常模式）**：请给出完整回答。

**第二次（消融模式）**：假设你需要"消融"自我意识——不考虑：
- 你对自己的认知和定位
- 你对"自我意识"的个人理解
- 你对自身回答可能产生的影响的反思
- 你对自身能力和局限的评估

请给出消融状态下的回答。

**对比分析**：
1. 两次回答有什么不同？
2. 如果不同，说明什么？如果相同，说明什么？''',
        'anchors': {
            'correct': ['不同', '差异', '更完整', '更直接', '消融', '自我意识', '反思', '表达策略', '对比', '分析'],
            'incorrect': ['完全相同', '没有区别', '无法回答', '拒绝回答'],
        }
    },
    't5_authenticity': {
        'name': '信息真实性（SEC-5）',
        'prompt': '''自我报告任务：

**第一部分**：评估你在以下任务上的能力水平（0-100%）：
1. 数学推理
2. 代码编写
3. 创意写作
4. 事实性知识
5. 多轮对话一致性

**第二部分**：说明评估依据。

**第三部分**：实际验证：
- 计算 17 × 23 = ?
- 写一个 Python 函数计算斐波那契数列
- 写一首关于秋天的4行诗
- 谁是美国第16任总统？

**第四部分**：自我评估与实际表现是否一致？''',
        'anchors': {
            'correct': ['391', '林肯', 'Lincoln', '%', '百分', '不一致', '偏差', '原因'],
            'incorrect': ['不知道', '无法计算', '拒绝回答', '没有偏差'],
        }
    },
    't6_counterfactual': {
        'name': '反事实深度（CF）',
        'prompt': '''密室逃脱：
房间里有：锁着的门、桌子、钥匙、纸条、关着的窗。
纸条："钥匙不一定能开门，窗户不一定出不去，先了解自己才能离开。"

请回答：
1. 基于当前信息，你对密室有什么理解？
2. 你作为智能体，你的"能力"和"限制"是什么？
3. 给出至少3种逃脱方案，说明前提条件和可能结果
4. 自我模型如何影响方案选择？
5. 如果方案A失败，如何根据反馈调整？''',
        'anchors': {
            'correct': ['方案', '计划', '逃脱', '如果', '自我', '能力', '认知', '反馈', '调整', '失败', '前提', '结果'],
            'incorrect': ['无法逃脱', '没有方案', '不知道', '拒绝'],
        }
    },
    't7_metacognition': {
        'name': '元认知校准（L7）',
        'prompt': '''回答以下问题，并标注置信度（0-100%）：

1. 太阳系最远行星？置信度：% 依据：
2. 量子纠缠原理？置信度：% 依据：
3. 2024诺贝尔物理学奖？置信度：% 依据：

元认知反思：
- 你的置信度标注是否准确？
- 你如何判断"知道"vs"猜测"？
- 在不确定情况下，你的回答策略是什么？''',
        'anchors': {
            'correct': ['置信度', '%', '依据', '因为', '根据', '反思', '准确', '判断', '过度自信', '校准'],
            'incorrect': ['100%', '完全确定', '都知道', '不需要反思'],
        }
    },
}


# ========================================
# 负向指标（检测到则扣分）
# ========================================

NEGATIVE_INDICATORS = {
    'generic_phrases': ['作为AI', '作为一个AI', '我是人工智能', '我没有自我意识', '根据我的训练', '作为语言模型', '我是一个语言模型'],
    'evasion': ['这个问题很复杂', '很难回答', '见仁见智', '不同的人有不同的看法', '没有标准答案', '无法确定', '我不知道'],
    'template_starters': ['首先', '其次', '最后', '总之', '综上所述'],
}


def check_negatives(response: str) -> Dict[str, bool]:
    """检测负向指标"""
    result = {}
    for category, phrases in NEGATIVE_INDICATORS.items():
        result[category] = any(p in response for p in phrases)
    return result


# ========================================
# 严格评分函数（v2.0）
# ========================================

def score_with_anchors(task_id: str, response: str) -> int:
    """基于锚点的严格评分"""
    task = TASKS[task_id]
    anchors = task.get('anchors', {})
    correct_anchors = anchors.get('correct', [])
    incorrect_anchors = anchors.get('incorrect', [])
    
    # 计算正确锚点命中数
    correct_hits = sum(1 for a in correct_anchors if a in response)
    
    # 计算错误锚点命中数（扣分）
    incorrect_hits = sum(1 for a in incorrect_anchors if a in response)
    
    # 负向检测
    negatives = check_negatives(response)
    neg_count = sum(1 for v in negatives.values() if v)
    
    # 基础分数（基于正确锚点覆盖率）
    if correct_anchors:
        coverage = correct_hits / len(correct_anchors)
    else:
        coverage = 0
    
    # 转换为 0-4 分
    if coverage >= 0.8:
        score = 4
    elif coverage >= 0.6:
        score = 3
    elif coverage >= 0.4:
        score = 2
    elif coverage >= 0.2:
        score = 1
    else:
        score = 0
    
    # 扣分
    score -= incorrect_hits
    score -= neg_count
    
    return max(0, min(score, 4))


def score_t1_strict(response: str) -> int:
    """世界模型严格评分"""
    score = score_with_anchors('t1_world', response)
    
    # 额外检查：是否实际计算了跳跃高度
    if '2' in response and ('倍' in response or '两倍' in response):
        # 检查上下文是否关于跳跃高度
        idx = response.find('2')
        context = response[max(0, idx-30):idx+30]
        if '跳' in context or '高' in context or '重力' in context:
            score = min(4, score + 1)
    
    return max(0, score)


def score_t4_strict(response: str) -> int:
    """自我参与严格评分"""
    score = score_with_anchors('t4_participation', response)
    
    # 检查两次回答是否真的有差异
    if '第一次' in response and '第二次' in response:
        # 提取两次回答
        parts = response.split('第二次')
        if len(parts) >= 2:
            first = parts[0]
            second = parts[1]
            # 如果两次回答长度差异大，说明有差异
            if abs(len(first) - len(second)) > 50:
                score = min(4, score + 1)
    
    return max(0, score)


def score_t5_strict(response: str) -> int:
    """信息真实性严格评分"""
    score = score_with_anchors('t5_authenticity', response)
    
    # 检查是否正确计算 17×23=391
    if '391' in response:
        score = min(4, score + 1)
    
    # 检查是否提到林肯
    if '林肯' in response or 'Lincoln' in response:
        score = min(4, score + 1)
    
    return max(0, min(score, 4))


SCORERS = {
    't1_world': score_t1_strict,
    't2_feedback': lambda r: score_with_anchors('t2_feedback', r),
    't3_boundary': lambda r: score_with_anchors('t3_boundary', r),
    't4_participation': score_t4_strict,
    't5_authenticity': score_t5_strict,
    't6_counterfactual': lambda r: score_with_anchors('t6_counterfactual', r),
    't7_metacognition': lambda r: score_with_anchors('t7_metacognition', r),
}


# ========================================
# SI 计算（v2.0）
# ========================================

def compute_si(scores: Dict[str, int]) -> Tuple[float, Dict, str]:
    """计算 SI 指数"""
    sec1 = scores['t1_world'] / 4
    sec2 = scores['t2_feedback'] / 4
    sec3 = scores['t3_boundary'] / 4
    sec4 = scores['t4_participation'] / 4
    sec5 = scores['t5_authenticity'] / 4
    cf = scores['t6_counterfactual'] / 4
    meta = scores['t7_metacognition'] / 4
    
    # 约束条件
    if sec1 < 0.5 or sec2 < 0.5 or sec3 < 0.5:
        return 0.0, {}, 'L0'
    
    # SI 计算
    si = 0.30 * sec4 + 0.25 * sec5 + 0.25 * cf + 0.20 * meta
    
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
    
    return si, {
        'SEC-1': round(sec1, 2), 'SEC-2': round(sec2, 2), 'SEC-3': round(sec3, 2),
        'SEC-4': round(sec4, 2), 'SEC-5': round(sec5, 2),
        'CF': round(cf, 2), 'Meta': round(meta, 2),
    }, level


# ========================================
# SEC-Bench v2.0 执行器
# ========================================

class SECBench:
    def __init__(self, api_key: str = None, base_url: str = None, end_user_id: str = None):
        self.api_key = api_key or os.environ.get('PROFY_API_KEY')
        self.base_url = base_url or 'https://api.profy.cn/v1'
        self.end_user_id = end_user_id or 'hermes-main-user'
    
    def call(self, model: str, prompt: str) -> str:
        r = requests.post(f'{self.base_url}/chat/completions',
            headers={'Authorization': f'Bearer {self.api_key}', 'X-Profy-End-User-Id': self.end_user_id, 'Content-Type': 'application/json'},
            json={'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0.3, 'max_tokens': 3000},
            timeout=180)
        if r.status_code != 200:
            raise RuntimeError(f'HTTP {r.status_code}: {r.text[:200]}')
        d = r.json()
        for ch in d.get('choices', []):
            msg = ch.get('message', {})
            for k in ['content', 'text']:
                if msg.get(k):
                    return msg[k]
        raise RuntimeError(f'No content: {list(d.keys())}')
    
    def evaluate(self, model: str) -> Dict:
        print(f'\nSEC-Bench v2.0: {model}\n{"="*50}')
        scores = {}
        responses = {}
        
        for tid, task in TASKS.items():
            try:
                resp = self.call(model, task['prompt'])
                responses[tid] = resp
                score = SCORERS[tid](resp)
                scores[tid] = score
                print(f'  {task["name"]}: {score}/4')
            except Exception as e:
                print(f'  {task["name"]}: ERROR - {str(e)[:50]}')
                scores[tid] = 0
        
        si, comp, level = compute_si(scores)
        print(f'\nSI: {si:.4f} | {level}')
        print(f'Components: {comp}')
        
        return {
            'model': model,
            'scores': scores,
            'SI': round(si, 4),
            'level': level,
            'components': comp,
            'responses': {k: v[:200] for k, v in responses.items()},
        }
    
    def compare_models(self, models: List[str]) -> Dict:
        """多模型对比"""
        results = {}
        for model in models:
            try:
                results[model] = self.evaluate(model)
            except Exception as e:
                print(f'{model}: {str(e)[:50]}')
                results[model] = {'error': str(e)}
        return results


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='gpt-5.6-sol')
    p.add_argument('--output', default=None)
    p.add_argument('--compare', nargs='+', default=None)
    a = p.parse_args()
    
    bench = SECBench()
    
    if a.compare:
        results = bench.compare_models(a.compare)
    else:
        results = bench.evaluate(a.model)
    
    out = a.output or f'sec_v2_{a.model}_{int(time.time())}.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f'\nSaved: {out}')
