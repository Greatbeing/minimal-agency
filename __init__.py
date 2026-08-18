"""最小智能闭环 (Minimal Intelligent Closed Loop) — 主体性涌现的形式化、计算实验与 BreakShell Agent"""

__version__ = "0.1.0"

from micl.breakshell.agent import BreakShellAgent
from micl.breakshell.self_model import SelfModel
from micl.breakshell.planner import CounterfactualPlanner
from micl.breakshell.si_measurement import SIMeasurement

__all__ = [
    "BreakShellAgent",
    "SelfModel",
    "CounterfactualPlanner",
    "SIMeasurement",
]
