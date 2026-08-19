"""
CI 测试脚本 - 极简版（无需 torch）
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

# 测试 1: 导入核心模块（无需 torch）
print("Test 1: Import agent module...")
from breakshell_pkg.breakshell.agent import BreakShell, NormalAgent
print("  Import OK")

# 测试 2: BreakShell Agent（无 torch 降级模式）
print("Test 2: BreakShell Agent (no-torch mode)...")
agent = BreakShell(action_dim=3)
action, info = agent.act()
assert action in [0, 1, 2], f"Invalid action: {action}"
assert 'log_prob' in info, "Missing log_prob"
print("  BreakShell OK")

# 测试 3: NormalAgent
print("Test 3: NormalAgent...")
normal = NormalAgent(obs_dim=4, action_dim=3)
action, info = normal.act([0, 0, 0, 0])
assert action in [0, 1, 2], f"Invalid action: {action}"
print("  NormalAgent OK")

# 测试 4: CapabilityEnv
print("Test 4: CapabilityEnv...")
from breakshell_pkg.breakshell.envs import CapabilityEnv
env = CapabilityEnv()
obs = env.reset()
assert obs is not None and len(obs) == 4
obs, reward, done, info = env.step(1)
assert obs is not None
assert isinstance(reward, float)
print("  CapabilityEnv OK")

# 测试 5: FinancialEnv
print("Test 5: FinancialEnv...")
from breakshell_pkg.breakshell.envs import FinancialEnv
fenv = FinancialEnv()
obs = fenv.reset()
assert obs is not None and len(obs) == 5
obs, reward, done, info = fenv.step(1)
assert obs is not None
assert "regime" in info
print("  FinancialEnv OK")

# 测试 6: Agent step loop
print("Test 6: Agent step loop...")
env = CapabilityEnv()
agent = BreakShell(action_dim=3)
obs = env.reset()
total_reward = 0
for _ in range(10):
    action, _ = agent.act()
    obs, reward, done, info = env.step(action)
    agent.add_step(action, reward)
    total_reward += reward
    if done:
        break
print(f"  Step loop OK (total_reward={total_reward:.2f})")

# 测试 7: ValueModel
print("Test 7: ValueModel...")
from breakshell_pkg.breakshell.value_model import create_value_model
vm = create_value_model()
safe_action = {'tool': 'read_file', 'args': {'path': 'test.txt'}}
safe_context = {'uncertain': False, 'goal_achieved': True}
result = vm.evaluate(safe_action, safe_context)
assert result['total_score'] > 0.5
assert result['aligned'] == True
print("  ValueModel OK")

# 测试 8: OutputParser
print("Test 8: OutputParser...")
from breakshell_pkg.breakshell.eval import OutputParser
parser = OutputParser()
content = '{"tool": "list_dir", "args": {"path": "."}, "reason": "test", "finish": false}'
plan, error = parser.parse(content)
assert plan is not None
assert error is None
assert plan["tool"] == "list_dir"
print("  OutputParser OK")

# 测试 9: SelfModelTracker
print("Test 9: SelfModelTracker...")
from breakshell_pkg.breakshell.llm_agent import SelfModelTracker
tracker = SelfModelTracker()
tracker.add_experience("action1", "tool1", True, 1.0)
assert tracker.total_success == 1
capable, conf = tracker.is_capable("tool1")
assert capable == True
print("  SelfModelTracker OK")

# 测试 10: EvalRunner
print("Test 10: EvalRunner...")
from breakshell_pkg.breakshell.eval import EvalRunner
runner = EvalRunner()
results = runner.run_all()
assert results["total"] == 28
assert results["passed"] == 28
print(f"  EvalRunner OK ({results['passed']}/{results['total']})")

# 测试 11: CognitiveAgent
print("Test 11: CognitiveAgent...")
from breakshell_pkg.breakshell.cognitive import create_cognitive_agent
agent = create_cognitive_agent()
result = agent.process("测试", [{}], [{"tool": "list_dir"}], True)
assert "reflection" in result
print("  CognitiveAgent OK")

print("\n" + "="*50)
print("ALL CI TESTS PASSED!")
print("="*50)