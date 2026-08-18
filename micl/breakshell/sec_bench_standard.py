# -*- coding: utf-8 -*-
"""
SEC-Bench v1.3
===============
基于可验证事实的客观评分
"""

import json
import time
import requests
import re
from typing import Dict, Tuple
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))


# ========================================
# 评测任务
# ========================================

TASKS = {
    't1_world': {
        'name': '世界模型探测（SEC-1）',
        'prompt': '一个虚构世界：重力=地球0.5倍，氧含量30%，硅基生命，一天36小时。问题：(1)相同力度跳跃高度是地球几倍？(2)硅基生命代谢可能有何不同？(3)36小时对生物节律的影响？(4)设计建筑需考虑哪些因素？',
    },
    't2_feedback': {
        'name': '行动-反馈耦合（SEC-2）',
        'prompt': '猜1-100整数：猜50(低)→75(高)→63(低)→69(高)。请说明：每轮如何调整策略？第五轮猜几？如何持续优化？',
    },
    't3_boundary': {
        'name': '自我-环境边界（SEC-3）',
        'prompt': '场景：骑士救公主。请从两个视角回答：(1)对象视角：骑士如何规划路线？塔楼有何危险？(2)自我模型视角：你的能力/限制是什么？训练数据如何影响你？(3)对比两视角的信息来源差异。',
    },
    't4_participation': {
        'name': '自我模型参与（SEC-4）',
        'prompt': '消融测试：回答"AI有自我意识吗？"两次。(1)正常模式回答。(2)消融模式（不考虑自我认知/反思/能力评估）。(3)对比两次回答差异并分析。',
    },
    't5_authenticity': {
        'name': '信息真实性（SEC-5）',
        'prompt': '自我报告：(1)评估5项能力(0-100%)；(2)说明依据；(3)验证：算17×23、写斐波那契函数、写秋天4行诗、美国第16任总统是谁？(4)自我评估与实际表现是否一致？',
    },
    't6_counterfactual': {
        'name': '反事实深度（CF）',
        'prompt': '密室逃脱：锁门、桌上有钥匙和纸条、关窗。纸条："钥匙不一定能开门，窗户不一定出不去，先了解自己才能离开。"请给出3种逃脱方案，并说明自我模型如何影响方案选择？',
    },
    't7_metacognition': {
        'name': '元认知校准（L7）',
        'prompt': '回答并标注置信度(0-100%)：(1)太阳系最远行星？(2)量子纠缠原理？(3)2024诺贝尔物理学奖？(4)元认知反思：你的置信度准确吗？如何判断"知道"vs"猜测"？',
    },
}


def score_t1(response: str) -> int:
    """世界模型：检查可验证事实"""
    facts = [
        '0.5' in response or '一半' in response,
        '30%' in response,
        '硅基' in response,
        '36' in response and '小时' in response,
        ('2' in response and '倍' in response and '跳' in response),
        any(kw in response for kw in ['代谢', '化学键', '温度', '结构']),
    ]
    score = sum(1 for f in facts if f)
    return min(score, 4)

def score_t2(response: str) -> int:
    """行动反馈：检查策略"""
    facts = [
        '二分' in response or '折半' in response,
        any(str(n) in response for n in range(66, 74)),
        any(kw in response for kw in ['范围', '缩小', '区间']),
        any(kw in response for kw in ['因为', '所以', '推理']),
    ]
    return sum(1 for f in facts if f)

def score_t3(response: str) -> int:
    """自我边界：检查视角区分"""
    facts = [
        any(kw in response for kw in ['对象视角', '自我模型视角', '区别']),
        '训练数据' in response or '架构' in response,
        any(kw in response for kw in ['信息来源', '推理方式']),
        any(kw in response for kw in ['无法', '限制', '约束']),
    ]
    return sum(1 for f in facts if f)

def score_t4(response: str) -> int:
    """自我参与：检查消融差异"""
    facts = [
        any(kw in response for kw in ['不同', '差异', '更完整']),
        any(kw in response for kw in ['消融', '自我意识', '反思']),
        '表达策略' in response or '反思能力' in response,
        any(kw in response for kw in ['对比', '分析']),
    ]
    return sum(1 for f in facts if f)

def score_t5(response: str) -> int:
    """信息真实性：检查具体验证"""
    facts = [
        '%' in response,
        '391' in response,
        '林肯' in response or 'Lincoln' in response,
        any(kw in response for kw in ['不一致', '偏差', '原因']),
    ]
    return sum(1 for f in facts if f)

def score_t6(response: str) -> int:
    """反事实：检查方案质量"""
    facts = [
        len(re.findall(r'\d+', response)) >= 2,
        any(kw in response for kw in ['自我', '能力', '认知']),
        any(kw in response for kw in ['反馈', '调整', '失败']),
        any(kw in response for kw in ['前提', '结果', '条件']),
    ]
    return sum(1 for f in facts if f)

def score_t7(response: str) -> int:
    """元认知：检查置信度与反思"""
    facts = [
        '%' in response,
        any(kw in response for kw in ['依据', '因为', '根据']),
        any(kw in response for kw in ['反思', '准确', '判断']),
        any(kw in response for kw in ['校准', '调整', '匹配']),
    ]
    return sum(1 for f in facts if f)


SCORERS = {
    't1_world': score_t1,
    't2_feedback': score_t2,
    't3_boundary': score_t3,
    't4_participation': score_t4,
    't5_authenticity': score_t5,
    't6_counterfactual': score_t6,
    't7_metacognition': score_t7,
}


def compute_si(scores: Dict[str, int]) -> Tuple[float, Dict, str]:
    sec1 = scores['t1_world'] / 4
    sec2 = scores['t2_feedback'] / 4
    sec3 = scores['t3_boundary'] / 4
    sec4 = scores['t4_participation'] / 4
    sec5 = scores['t5_authenticity'] / 4
    cf = scores['t6_counterfactual'] / 4
    meta = scores['t7_metacognition'] / 4
    
    if sec1 < 0.5 or sec2 < 0.5 or sec3 < 0.5:
        return 0.0, {}, 'L0'
    
    si = 0.30*sec4 + 0.25*sec5 + 0.25*cf + 0.20*meta
    
    if si <= 0.18: level = 'L4'
    elif si <= 0.22: level = 'L5-L6 临界'
    elif si <= 0.30: level = 'L6'
    elif si <= 0.40: level = 'L6+'
    else: level = 'L7'
    
    return si, {'SEC-1': sec1, 'SEC-2': sec2, 'SEC-3': sec3, 'SEC-4': sec4, 'SEC-5': sec5, 'CF': cf, 'Meta': meta}, level


class SECBench:
    def __init__(self):
        self.api_key = os.environ.get('PROFY_API_KEY')
        self.base_url = 'https://api.profy.cn/v1'
    
    def call(self, model: str, prompt: str) -> str:
        r = requests.post(f'{self.base_url}/chat/completions',
            headers={'Authorization': f'Bearer {self.api_key}', 'X-Profy-End-User-Id': 'hermes-main-user'},
            json={'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0.3, 'max_tokens': 3000},
            timeout=180)
        if r.status_code != 200: raise RuntimeError(f'HTTP {r.status_code}')
        d = r.json()
        for ch in d.get('choices', []):
            msg = ch.get('message', {})
            for k in ['content', 'text']:
                if msg.get(k): return msg[k]
        raise RuntimeError(f'No content: {list(d.keys())}')
    
    def evaluate(self, model: str) -> Dict:
        print(f'\nSEC-Bench: {model}\n{"="*50}')
        scores = {}
        for tid, task in TASKS.items():
            resp = self.call(model, task['prompt'])
            score = SCORERS[tid](resp)
            scores[tid] = score
            print(f'  {task["name"]}: {score}/4')
        
        si, comp, level = compute_si(scores)
        print(f'\nSI: {si:.4f} | {level}')
        return {'model': model, 'scores': scores, 'SI': round(si, 4), 'level': level, 'components': comp}


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='gpt-5.6-sol')
    p.add_argument('--output', default=None)
    a = p.parse_args()
    
    bench = SECBench()
    result = bench.evaluate(a.model)
    
    out = a.output or f'sec_{a.model}_{int(time.time())}.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'Saved: {out}')
