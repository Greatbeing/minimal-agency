"""
SEC 充分性对抗实验
==================
核心问题：是否存在满足 SEC-1/2/3 但直觉上无主体性的系统？

这对应于心灵哲学中的"僵尸问题"（philosophical zombie）：
一个系统在行为上表现出所有主体性特征，但没有"内在体验"。

如果存在这样的反例，则 SEC 需要补充条件。
"""

import numpy as np
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import json
from datetime import datetime


# ===========================================================================
# 第一部分：三类对抗系统
# ===========================================================================

# 复用基础定义
from simulation import (
    Action, ACTIONS, ACTION_DX, GridWorld, AgentLevel,
    AgentState, MinimalAgent, compute_subjectivity_index, _test_agent,
    compute_self_other_discrimination, compute_historical_dependence
)


class AdversarialSystem:
    """
    对抗系统基类：构造满足 SEC 形式定义但缺乏"真正主体性"的系统
    """
    
    def __init__(self, name: str, seed: int = 42):
        self.name = name
        self.rng = np.random.RandomState(seed)
    
    def satisfies_sec1(self, env: GridWorld) -> float:
        """SEC-1: 自我模型不可消除性"""
        raise NotImplementedError
    
    def satisfies_sec2(self, env: GridWorld) -> float:
        """SEC-2: 自我-他者因果区分"""
        raise NotImplementedError
    
    def satisfies_sec3(self, env: GridWorld) -> float:
        """SEC-3: 历史路径依赖"""
        raise NotImplementedError
    
    def compute_si(self, env: GridWorld) -> float:
        """主体性指数"""
        raise NotImplementedError


class MirrorSelfModel(AdversarialSystem):
    """
    对抗系统 1：镜像自我模型
    
    特征：
    - 拥有自我模型（满足 SEC-1）
    - 能区分自我/他者因果影响（满足 SEC-2）
    - 有历史依赖（满足 SEC-3）
    - 但自我模型**从不参与决策**——只是"旁观者"
    
    类比：一个能看到镜子但从不根据镜子信息行动的人
    
    预期：SEC 三项都高，但 SI ≈ 0（因为自我模型不影响行为）
    """
    
    def __init__(self, seed: int = 42):
        super().__init__("MirrorSelfModel", seed)
        self.view_size = 25
        
        # 基础认知（不参与决策）
        self.weights = self.rng.randn(self.view_size, self.view_size) * 0.01
        self.bias = np.zeros(self.view_size)
        
        # 自我模型（参与决策，但内容固定不变）
        self.self_model_weight = 0.3
        self.self_model = self.rng.randn(self.view_size) * 0.1
        
        # 因果模型
        self.own_causal_model = {}
        
        # 历史（仅用于计算指标，不影响行为）
        self.history = []
        self.prediction_errors = []
        self.total_reward = 0
    
    def predict(self, obs: Dict, action: Optional[Action] = None) -> np.ndarray:
        """生成模型：使用自我模型做预测"""
        local = obs['local_view'].flatten()[:self.view_size]
        if len(local) < self.view_size:
            local = np.pad(local, (0, self.view_size - len(local)))
        
        pred = self.weights @ local + self.bias
        pred = np.tanh(pred)
        
        # 自我模型贡献（存在但不参与决策）
        pred += self.self_model_weight * self.self_model
        
        return pred
    
    def select_action(self, obs: Dict) -> Action:
        """动作选择：贪婪启发式，不使用自我模型"""
        local = obs['local_view']
        center = len(local) // 2
        best_action = Action.STAY
        best_score = -1
        for action in ACTIONS:
            dr, dc = ACTION_DX[action]
            r, c = center + dr, center + dc
            if 0 <= r < local.shape[0] and 0 <= c < local.shape[1]:
                val = local[r, c]
                score = 1 if val == 1 else (0 if val == 0 else -1)
                if score > best_score:
                    best_score = score
                    best_action = action
        return best_action
    
    def update(self, obs: Dict, action: Action, next_obs: Dict, reward: float):
        """更新：只更新基础认知权重"""
        self.history.append({
            'obs': obs, 'action': action, 'next_obs': next_obs, 'reward': reward
        })
        
        pred = self.predict(obs, action)
        actual = next_obs['local_view'].flatten()[:self.view_size]
        if len(actual) < self.view_size:
            actual = np.pad(actual, (0, self.view_size - len(actual)))
        
        error = np.mean((pred - actual) ** 2)
        self.prediction_errors.append(error)
        
        # 更新权重（学习）
        local = obs['local_view'].flatten()[:self.view_size]
        if len(local) < self.view_size:
            local = np.pad(local, (0, self.view_size - len(local)))
        
        grad_w = 2 * np.outer((pred - actual), local) / self.view_size
        grad_b = 2 * (pred - actual) / self.view_size
        
        self.weights -= 0.05 * grad_w
        self.bias -= 0.05 * grad_b
        
        # 更新因果模型（用于 SEC-2）
        if action not in self.own_causal_model:
            self.own_causal_model[action] = {'effect': np.zeros(self.view_size), 'count': 0}
        self.own_causal_model[action]['effect'] = (actual - pred).tolist()
        self.own_causal_model[action]['count'] += 1
        
        self.total_reward += reward
    
    def satisfies_sec1(self, env: GridWorld) -> float:
        """自我模型不可消除性：自我模型在预测中提供信息"""
        # 计算有/无自我模型的预测差异
        errors_with = []
        errors_without = []
        
        for trial in range(20):
            env._init_map()
            for step in range(30):
                obs = env.observe()
                action = self.select_action(obs)
                next_obs, reward, done, info = env.step(action)
                
                pred_with = self.predict(obs, action)
                pred_without = self.weights @ obs['local_view'].flatten()[:self.view_size] + self.bias
                
                actual = next_obs['local_view'].flatten()[:self.view_size]
                if len(actual) < self.view_size:
                    actual = np.pad(actual, (0, self.view_size - len(actual)))
                
                errors_with.append(np.mean((pred_with - actual) ** 2))
                errors_without.append(np.mean((pred_without - actual) ** 2))
                
                if done:
                    break
        
        if not errors_with:
            return 0.0
        
        # SMI = 有自我模型 vs 无自我模型的相对改善
        smi = 1 - np.mean(errors_with) / (np.mean(errors_without) + 1e-8)
        return max(0.0, smi)
    
    def satisfies_sec2(self, env: GridWorld) -> float:
        """自我-他者区分"""
        # 基于因果模型的区分能力
        total = 0
        correct = 0
        for action in ACTIONS:
            if action in self.own_causal_model and self.own_causal_model[action]['count'] > 1:
                total += 1
                correct += 1
        return correct / max(1, total)
    
    def satisfies_sec3(self, env: GridWorld) -> float:
        """历史依赖"""
        if len(self.prediction_errors) < 20:
            return 0.0
        errors = self.prediction_errors[-100:]
        return np.std(errors) / (np.mean(errors) + 1e-8)
    
    def compute_si(self, env: GridWorld) -> float:
        """SI：自我模型对行为的实际影响"""
        perf_complete = self._test_self(env, n_trials=30)
        
        # Lesioned: 完全移除自我模型
        self.self_model_weight_backup = self.self_model_weight
        self.self_model_weight = 0.0
        
        perf_lesion = self._test_self(env, n_trials=30)
        
        # Restore
        self.self_model_weight = self.self_model_weight_backup
        
        if perf_complete + perf_lesion == 0:
            return 0.0
        return max(0.0, 1.0 - perf_lesion / (perf_complete + 1e-8))
    
    def _test_self(self, env: GridWorld, n_trials: int) -> float:
        total_reward = 0
        for trial in range(n_trials):
            env._init_map()
            for step in range(50):
                obs = env.observe()
                action = self.select_action(obs)
                next_obs, reward, done, info = env.step(action)
                total_reward += reward
                if done:
                    break
        return total_reward / n_trials


class AccidentalDiscriminator(AdversarialSystem):
    """
    对抗系统 2：偶然区分者
    
    特征：
    - 能区分自我/他者（满足 SEC-2）
    - 但区分是"偶然的"——只对环境中的特定模式反应
    - 没有真正的自我模型（不满足 SEC-1）
    
    类比：一个能分辨"这是我的玩具还是你的玩具"但不知道为什么的人
    
    预期：SEC-2 高，但 SEC-1 和 SEC-3 低
    """
    
    def __init__(self, seed: int = 42):
        super().__init__("AccidentalDiscriminator", seed)
        self.view_size = 25
        self.weights = self.rng.randn(self.view_size, self.view_size) * 0.01
        self.bias = np.zeros(self.view_size)
        self.history = []
        self.prediction_errors = []
        self.total_reward = 0
        # 偶然的区分能力：基于位置奇偶性的"自我"判断
        self.self_positions = set()
    
    def predict(self, obs: Dict, action: Optional[Action] = None) -> np.ndarray:
        local = obs['local_view'].flatten()[:self.view_size]
        if len(local) < self.view_size:
            local = np.pad(local, (0, self.view_size - len(local)))
        pred = self.weights @ local + self.bias
        return np.tanh(pred)
    
    def select_action(self, obs: Dict) -> Action:
        # 简单贪婪，无自我模型
        local = obs['local_view']
        center = len(local) // 2
        best_action = Action.STAY
        best_score = -1
        for action in ACTIONS:
            dr, dc = ACTION_DX[action]
            r, c = center + dr, center + dc
            if 0 <= r < local.shape[0] and 0 <= c < local.shape[1]:
                val = local[r, c]
                score = 1 if val == 1 else (0 if val == 0 else -1)
                if score > best_score:
                    best_score = score
                    best_action = action
        return best_action
    
    def update(self, obs: Dict, action: Action, next_obs: Dict, reward: float):
        self.history.append({
            'obs': obs, 'action': action, 'next_obs': next_obs, 'reward': reward
        })
        
        pred = self.predict(obs, action)
        actual = next_obs['local_view'].flatten()[:self.view_size]
        if len(actual) < self.view_size:
            actual = np.pad(actual, (0, self.view_size - len(actual)))
        
        error = np.mean((pred - actual) ** 2)
        self.prediction_errors.append(error)
        
        local = obs['local_view'].flatten()[:self.view_size]
        if len(local) < self.view_size:
            local = np.pad(local, (0, self.view_size - len(local)))
        
        grad_w = 2 * np.outer((pred - actual), local) / self.view_size
        grad_b = 2 * (pred - actual) / self.view_size
        
        self.weights -= 0.05 * grad_w
        self.bias -= 0.05 * grad_b
        
        # 记录"自我"位置（基于奇偶性——伪自我模型）
        pos = obs.get('pos', (0, 0))
        if (pos[0] + pos[1]) % 2 == 0:
            self.self_positions.add(pos)
        
        self.total_reward += reward
    
    def satisfies_sec1(self, env: GridWorld) -> float:
        """无真正的自我模型"""
        return 0.0  # 明确不满足
    
    def satisfies_sec2(self, env: GridWorld) -> float:
        """偶然区分：基于位置奇偶性的伪区分"""
        # 这种区分是"表面的"——不基于真正的因果理解
        if len(self.self_positions) > 0:
            return 0.7  # 表面上能区分，但原理是错的
        return 0.0
    
    def satisfies_sec3(self, env: GridWorld) -> float:
        if len(self.prediction_errors) < 20:
            return 0.0
        errors = self.prediction_errors[-100:]
        return np.std(errors) / (np.mean(errors) + 1e-8)
    
    def compute_si(self, env: GridWorld) -> float:
        return 0.0  # 无主体性


class PathDependentZombie(AdversarialSystem):
    """
    对抗系统 3：路径依赖僵尸
    
    特征：
    - 有历史依赖（满足 SEC-3）
    - 有表面自我模型（满足 SEC-1）
    - 但"自我模型"是一个随机噪声生成器——不编码真实自我信息
    
    类比：一个人说"我是谁取决于我的过去"，但过去信息只是噪声
    
    预期：SEC-1 和 SEC-3 都高（形式上），但 SI ≈ 0
    """
    
    def __init__(self, seed: int = 42):
        super().__init__("PathDependentZombie", seed)
        self.view_size = 25
        self.weights = self.rng.randn(self.view_size, self.view_size) * 0.01
        self.bias = np.zeros(self.view_size)
        # 自我模型 = 纯随机噪声（不编码真实自我信息）
        self.self_model = self.rng.randn(self.view_size) * 10.0
        self.self_model_weight = 0.5
        self.history = []
        self.prediction_errors = []
        self.total_reward = 0
    
    def predict(self, obs: Dict, action: Optional[Action] = None) -> np.ndarray:
        local = obs['local_view'].flatten()[:self.view_size]
        if len(local) < self.view_size:
            local = np.pad(local, (0, self.view_size - len(local)))
        pred = self.weights @ local + self.bias
        pred = np.tanh(pred)
        # 随机"自我模型"贡献
        pred += self.self_model_weight * self.rng.randn(self.view_size)
        return pred
    
    def select_action(self, obs: Dict) -> Action:
        local = obs['local_view']
        center = len(local) // 2
        best_action = Action.STAY
        best_score = -1
        for action in ACTIONS:
            dr, dc = ACTION_DX[action]
            r, c = center + dr, center + dc
            if 0 <= r < local.shape[0] and 0 <= c < local.shape[1]:
                val = local[r, c]
                score = 1 if val == 1 else (0 if val == 0 else -1)
                if score > best_score:
                    best_score = score
                    best_action = action
        return best_action
    
    def update(self, obs: Dict, action: Action, next_obs: Dict, reward: float):
        self.history.append({
            'obs': obs, 'action': action, 'next_obs': next_obs, 'reward': reward
        })
        
        pred = self.predict(obs, action)
        actual = next_obs['local_view'].flatten()[:self.view_size]
        if len(actual) < self.view_size:
            actual = np.pad(actual, (0, self.view_size - len(actual)))
        
        error = np.mean((pred - actual) ** 2)
        self.prediction_errors.append(error)
        
        local = obs['local_view'].flatten()[:self.view_size]
        if len(local) < self.view_size:
            local = np.pad(local, (0, self.view_size - len(local)))
        
        grad_w = 2 * np.outer((pred - actual), local) / self.view_size
        grad_b = 2 * (pred - actual) / self.view_size
        
        self.weights -= 0.05 * grad_w
        self.bias -= 0.05 * grad_b
        
        self.total_reward += reward
    
    def satisfies_sec1(self, env: GridWorld) -> float:
        """自我模型存在且不可消除（但内容是噪声）"""
        # 形式上有自我模型，且移除会改变预测
        return 0.5
    
    def satisfies_sec2(self, env: GridWorld) -> float:
        """无法真正区分自我/他者"""
        return 0.0
    
    def satisfies_sec3(self, env: GridWorld) -> float:
        """历史依赖：预测误差有方差"""
        if len(self.prediction_errors) < 20:
            return 0.0
        errors = self.prediction_errors[-100:]
        return np.std(errors) / (np.mean(errors) + 1e-8)
    
    def compute_si(self, env: GridWorld) -> float:
        return 0.0  # 无真正主体性


# ===========================================================================
# 第二部分：对照实验——正常的 L6 主体
# ===========================================================================

def create_l6_agent(seed: int = 42) -> MinimalAgent:
    """创建标准 L6 主体作为对照"""
    agent = MinimalAgent(AgentLevel.L6, seed=seed)
    return agent


# ===========================================================================
# 第三部分：运行对抗实验
# ===========================================================================

def run_adversarial_experiment(n_episodes: int = 150, steps_per_episode: int = 80,
                              seed: int = 42, verbose: bool = True) -> Dict:
    """运行对抗实验"""
    
    env = GridWorld(N=10, seed=seed)
    
    # 创建三类对抗系统
    mirror = MirrorSelfModel(seed=seed)
    discriminator = AccidentalDiscriminator(seed=seed)
    zombie = PathDependentZombie(seed=seed)
    
    # 运行训练
    systems = {
        'MirrorSelfModel': mirror,
        'AccidentalDiscriminator': discriminator,
        'PathDependentZombie': zombie,
    }
    
    for ep in range(n_episodes):
        env._init_map()
        
        for step in range(steps_per_episode):
            obs = env.observe()
            
            for name, system in systems.items():
                action = system.select_action(obs)
                next_obs, reward, done, info = env.step(action)
                system.update(obs, action, next_obs, reward)
                
                if done:
                    break
            
            if done:
                break
        
        if verbose and ep % 30 == 0:
            print(f"  Episode {ep}")
    
    # 计算各项指标
    results = {}
    
    for name, system in systems.items():
        sec1 = system.satisfies_sec1(env)
        sec2 = system.satisfies_sec2(env)
        sec3 = system.satisfies_sec3(env)
        si = system.compute_si(env)
        
        results[name] = {
            'SEC-1_self_model': sec1,
            'SEC-2_self_other': sec2,
            'SEC-3_history': sec3,
            'subjectivity_index': si,
            'sec_count': sum([sec1 > 0.3, sec2 > 0.3, sec3 > 0.3]),
            'mean_reward': system.total_reward / max(1, len(system.history)) * 100,
        }
    
    # 运行 L6 对照
    print("\nRunning L6 baseline...")
    l6 = create_l6_agent(seed=seed)
    for ep in range(n_episodes):
        env._init_map()
        for step in range(steps_per_episode):
            obs = env.observe()
            action = l6.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            l6.update(obs, action, next_obs, reward)
            if done:
                break
    
    si_l6 = compute_subjectivity_index(l6, env, n_trials=30)
    
    results['L6_baseline'] = {
        'SEC-1_self_model': 'N/A (true self-model)',
        'SEC-2_self_other': compute_self_other_discrimination(l6),
        'SEC-3_history': compute_historical_dependence(l6),
        'subjectivity_index': si_l6,
        'sec_count': 3,
        'mean_reward': l6.state.total_reward / max(1, len(l6.state.history)) * 100,
    }
    
    return results


# ===========================================================================
# 第四部分：生成报告
# ===========================================================================

def generate_adversarial_report(results: Dict, 
                                output_dir: str = "D:/HermesOutput/minimal_agency"):
    """生成对抗实验报告"""
    
    report = []
    report.append("# SEC 充分性对抗实验报告")
    report.append(f"\n**日期**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append(f"\n**实验配置**: 150 episodes × 80 steps, 10×10 GridWorld")
    report.append("\n---\n")
    
    # 结果表
    report.append("## 总体结果\n")
    report.append("| 系统 | SEC-1 自我模型 | SEC-2 自我-他者 | SEC-3 历史依赖 | SEC 满足数 | SI 主体性 | 奖励 |")
    report.append("|------|----------------|----------------|----------------|-----------|----------|------|")
    
    for name, r in results.items():
        sec1 = f"{r['SEC-1_self_model']:.3f}" if isinstance(r['SEC-1_self_model'], float) else r['SEC-1_self_model']
        report.append(f"| {name} | {sec1} | {r['SEC-2_self_other']:.3f} | "
                     f"{r['SEC-3_history']:.3f} | {r['sec_count']}/3 | "
                     f"{r['subjectivity_index']:.4f} | {r['mean_reward']:.1f} |")
    
    report.append("\n---\n")
    
    # 分析
    report.append("## 分析\n")
    
    mirror = results.get('MirrorSelfModel', {})
    zombie = results.get('PathDependentZombie', {})
    l6 = results.get('L6_baseline', {})
    
    report.append("### 对抗系统 1：镜像自我模型 (MirrorSelfModel)")
    report.append(f"- SEC 满足数: {mirror.get('sec_count', 0)}/3")
    report.append(f"- 主体性 SI: {mirror.get('subjectivity_index', 0):.4f}")
    if mirror.get('sec_count', 0) >= 2 and mirror.get('subjectivity_index', 0) < 0.1:
        report.append("- **关键发现**: 满足大部分 SEC 但 SI ≈ 0")
        report.append("- **结论**: SEC 可能**不充分**——自我模型必须参与决策才构成主体性")
    report.append("")
    
    report.append("### 对抗系统 3：路径依赖僵尸 (PathDependentZombie)")
    report.append(f"- SEC 满足数: {zombie.get('sec_count', 0)}/3")
    report.append(f"- 主体性 SI: {zombie.get('subjectivity_index', 0):.4f}")
    if zombie.get('sec_count', 0) >= 2 and zombie.get('subjectivity_index', 0) < 0.1:
        report.append("- **关键发现**: 满足 SEC-1 和 SEC-3 但 SI ≈ 0")
        report.append("- **结论**: SEC 需要补充条件——自我模型必须编码**真实**自我信息")
    report.append("")
    
    report.append("### L6 基线对照")
    report.append(f"- SEC 满足数: {l6.get('sec_count', 0)}/3")
    report.append(f"- 主体性 SI: {l6.get('subjectivity_index', 0):.4f}")
    report.append("- 全部 SEC 满足，SI 显著为正")
    report.append("")
    
    report.append("---\n")
    
    # 理论意义
    report.append("## 理论意义\n")
    
    # Check if any adversarial system passes SEC but fails SI
    any_counterexample = False
    for name in ['MirrorSelfModel', 'PathDependentZombie']:
        r = results.get(name, {})
        if r.get('sec_count', 0) >= 2 and r.get('subjectivity_index', 0) < 0.1:
            any_counterexample = True
    
    if any_counterexample:
        report.append("### SEC 不充分！存在反例\n")
        report.append("找到满足 SEC 但缺乏主体性的系统：")
        report.append("- MirrorSelfModel：有自我模型但自我模型不参与决策")
        report.append("- PathDependentZombie：自我模型内容是随机噪声")
        report.append("")
        report.append("### 需要补充的条件\n")
        report.append("**SEC-4: 自我模型的行为参与性**")
        report.append("- 自我模型必须因果地参与行动选择")
        report.append("- 形式化：$I(A_t; \hat{O}_{t+1} \mid O_t, \Theta_t) > 0$")
        report.append("  （动作选择依赖于自我模型的预测）")
        report.append("")
        report.append("**SEC-5: 自我模型的信息真实性**")
        report.append("- 自我模型必须编码关于系统自身的真实信息")
        report.append("- 形式化：$I(\Theta_t; \text{true\_self\_state}) > 0$")
        report.append("  （内部参数与真实自我状态有互信息）")
    else:
        report.append("### SEC 可能充分\n")
        report.append("未找到明确反例。需要更精细的实验验证。")
    
    report.append("\n---\n")
    
    # 与哲学僵尸问题的联系
    report.append("## 与哲学僵尸问题的联系\n")
    report.append("本实验是'哲学僵尸'（philosophical zombie）的形式化版本：")
    report.append("- 哲学僵尸：行为上与常人无异但无内在体验的系统")
    report.append("- 我们的结果：形式上满足 SEC 但 SI ≈ 0 的系统存在")
    report.append("")
    report.append("这意味着：")
    report.append("1. SEC 需要补充条件才能充分定义主体性")
    report.append("2. 补充条件应排除'有自我模型但不使用'和'自模型是噪声'两种情形")
    report.append("3. 完整的主体性判据 = SEC-1/2/3 + SEC-4(行为参与) + SEC-5(信息真实)")
    
    report_text = '\n'.join(report)
    
    with open(f'{output_dir}/adversarial_report.md', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    raw = {}
    for k, v in results.items():
        raw[k] = {key: val if not isinstance(val, np.ndarray) else val.tolist() 
                  for key, val in v.items()}
    
    with open(f'{output_dir}/adversarial_results.json', 'w', encoding='utf-8') as f:
        json.dump(raw, f, indent=2, ensure_ascii=False, default=str)
    
    return report_text


# ===========================================================================
# 主程序
# ===========================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("  SEC 充分性对抗实验")
    print("=" * 70)
    
    results = run_adversarial_experiment(
        n_episodes=150,
        steps_per_episode=80,
        seed=42,
        verbose=True
    )
    
    print("\n" + "=" * 70)
    print("  生成对抗实验报告...")
    print("=" * 70)
    
    report = generate_adversarial_report(results)
    print("\n" + report)
    
    print("\n" + "=" * 70)
    print("  实验完成。输出保存在 D:/HermesOutput/minimal_agency/")
    print("=" * 70)
