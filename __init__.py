"""有我 (Self-Presence) — 主体性涌现的形式化、计算实验与 BreakShell Agent"""

__version__ = "0.1.0"

from youwo.breakshell.agent import BreakShellAgent
from youwo.breakshell.self_model import SelfModel
from youwo.breakshell.planner import CounterfactualPlanner
from youwo.breakshell.si_measurement import SIMeasurement

__all__ = [
    "BreakShellAgent",
    "SelfModel",
    "CounterfactualPlanner",
    "SIMeasurement",
]
