# -*- coding: utf-8 -*-
"""
BreakShell — AI Agent 自我模型安全层
=====================================

安装：
    pip install breakshell

使用：
    from breakshell import BreakShell, CapabilityEnv
    
    env = CapabilityEnv(seed=42)
    agent = BreakShell()
    agent.train(env)
    agent.save("my_agent")

命令行：
    breakshell train --env capability --episodes 500
    breakshell evaluate --model my_agent --episodes 100
    breakshell compare --env capability
"""

from .agent import BreakShell, NormalAgent
from .envs import CapabilityEnv, EnergyEnv, FinancialEnv

__version__ = "0.2.0"
__all__ = ["BreakShell", "NormalAgent", "CapabilityEnv", "EnergyEnv", "FinancialEnv"]
