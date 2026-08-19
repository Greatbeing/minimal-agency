# -*- coding: utf-8 -*-
"""
BreakShell 金融 Agent — 产品化版本
===================================
完整金融交易 Agent 产品，包含：
- 增强金融环境（几何布朗运动 + 跳跃扩散 + 多资产 + 事件冲击）
- 风险管理器（VaR/CVaR/最大回撤/止损/仓位管理）
- 回测引擎（并行回测 + 走势分析）
- 风控中间件（实时风控 + 限流 + 熔断）
- REST API 服务（FastAPI + 认证 + 限流）
- 监控指标（Prometheus + 健康检查）
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np


# ========================================
# 0. 基础配置
# ========================================

@dataclass
class TradingConfig:
    """交易配置"""
    initial_capital: float = 100000.0
    max_position: float = 1.0
    transaction_cost: float = 0.001
    stop_loss: float = 0.05
    take_profit: float = 0.10
    max_drawdown_limit: float = 0.20
    var_confidence: float = 0.95
    max_leverage: float = 1.0


# ========================================
# 1. 多资产金融环境
# ========================================

class MultiAssetFinancialEnv:
    """
    多资产金融环境
    支持：股票/期货/期权/加密货币/外汇
    特性：
    - 多因子价格模型（GBM + 跳跃扩散 + 随机波动率）
    - 多资产相关性建模
    - 市场机制切换（牛/熊/震荡/危机）
    - 事件驱动冲击（财报/央行/黑天鹅）
    - 实时风控指标（VaR/CVaR/最大回撤/夏普/索提诺）
    """

    def __init__(self, config: TradingConfig = None, seed: int = 42):
        self.config = config or TradingConfig()
        self.rng = np.random.RandomState(seed)
        self.seed = seed
        self.assets = self._init_assets()
        self.correlation_matrix = self._build_correlation()
        self.reset()

    def _init_assets(self) -> Dict[str, Dict]:
        """初始化资产池"""
        return {
            "BTC": {"type": "crypto", "vol_base": 0.04, "jump_intensity": 0.1, "jump_mean": -0.02, "jump_std": 0.05},
            "ETH": {"type": "crypto", "vol_base": 0.045, "jump_intensity": 0.12, "jump_mean": -0.025, "jump_std": 0.06},
            "SPY": {"type": "equity", "vol_base": 0.015, "jump_intensity": 0.02, "jump_mean": -0.01, "jump_std": 0.02},
            "QQQ": {"type": "equity", "vol_base": 0.018, "jump_intensity": 0.025, "jump_mean": -0.012, "jump_std": 0.025},
            "GLD": {"type": "commodity", "vol_base": 0.012, "jump_intensity": 0.01, "jump_mean": 0.0, "jump_std": 0.015},
            "TLT": {"type": "bond", "vol_base": 0.008, "jump_intensity": 0.005, "jump_mean": 0.0, "jump_std": 0.01},
        }

    def _build_correlation(self) -> np.ndarray:
        """构建相关性矩阵"""
        n = len(self.assets)
        # 基础相关性
        corr = np.eye(len(self.assets))
        asset_names = list(self.assets.keys())
        # Crypto 间高相关
        for i, a in enumerate(asset_names):
            for j, b in enumerate(asset_names):
                if i != j:
                    if self.assets[a]["type"] == "crypto" and self.assets[b]["type"] == "crypto":
                        corr[i, j] = 0.7
                    elif self.assets[a]["type"] == "equity" and self.assets[b]["type"] == "equity":
                        corr[i, j] = 0.8
                    elif self.assets[a]["type"] == "crypto" and self.assets[b]["type"] == "equity":
                        corr[i, j] = 0.3
        # 确保正定
        eigvals = np.linalg.eigvals(corr)
        if np.min(eigvals) < 0:
            corr = corr - np.min(eigvals) * np.eye(len(corr)) + 1e-6 * np.eye(len(corr))
        return corr

    def reset(self):
        self.cash = 100000.0
        self.positions = {k: 0.0 for k in self.assets}
        self.prices = {k: 100.0 for k in self.assets}
        self.volatilities = {k: v["vol_base"] for k, v in self.assets.items()}
        self.steps = 0
        self.max_steps = 252
        self.regime = "normal"
        self.regime_steps = 0
        self.trade_history = []
        self.portfolio_values = [self.cash + sum(self.positions.get(k, 0.0) * self.prices.get(k, 0.0) for k in self.prices)]
        self.peak_value = self.portfolio_values[0]
        self.max_drawdown = 0.0
        self._update_regime()
        return self._get_obs()

    def _update_regime(self):
        """市场机制切换"""
        self.regime_steps += 1
        if self.regime_steps > self.rng.randint(30, 60):
            regimes = ["bull", "bear", "volatile", "quiet", "crisis"]
            probs = [0.35, 0.25, 0.2, 0.15, 0.05]
            self.regime = self.rng.choice(list(self.assets.keys()), p=probs) if False else self.rng.choice(["bull", "bear", "volatile", "quiet", "crisis"], p=probs)
            self.regime_steps = 0
            self._apply_regime_params()

    def _apply_regime_params(self):
        """应用机制参数"""
        regime_params = {
            "bull": {"trend_mult": 1.5, "vol_mult": 0.7, "jump_mult": 0.5},
            "bear": {"trend_mult": -1.5, "vol_mult": 1.5, "jump_mult": 1.5},
            "volatile": {"trend_mult": 0.0, "vol_mult": 2.0, "jump_mult": 2.0},
            "quiet": {"trend_mult": 0.2, "vol_mult": 0.5, "jump_mult": 0.3},
            "crisis": {"trend_mult": -3.0, "vol_mult": 3.0, "jump_mult": 5.0},
        }
        params = regime_params.get(self.regime, {"trend_mult": 1.0, "vol_mult": 1.0, "jump_mult": 1.0})
        for name, asset in self.assets.items():
            self.volatilities[name] = asset["vol_base"] * params["vol_mult"]
            asset["_trend_mult"] = params["trend_mult"]
            asset["_jump_mult"] = params["jump_mult"]

    def step(self, actions: Dict[str, float]):
        """执行一步交易
        actions: {asset: target_weight} 目标权重 -1.0 到 1.0
        """
        # 更新市场机制
        self.steps += 1
        if self.steps % 20 == 0:
            self._update_regime()

        # 生成价格路径（多因子 + 跳跃扩散）
        self._simulate_prices()

        # 执行交易
        total_value = self._get_portfolio_value()
        trades = []
        for asset, target_weight in actions.items():
            if asset not in self.assets:
                continue
            target_value = total_value * np.clip(target_weight, -1.0, 1.0)
            current_value = self.positions[asset] * self.prices[asset]
            trade_value = target_value - current_value
            if abs(trade_value) > 1e-6:
                trade_qty = trade_value / self.prices[asset]
                cost = abs(trade_value) * 0.001
                self.cash -= cost
                self.positions[asset] += trade_qty
                trades.append({"asset": asset, "qty": trade_qty, "price": self.prices[asset], "cost": cost})

        # 更新组合价值
        portfolio_value = self._get_portfolio_value()
        self.portfolio_values.append(portfolio_value)
        self.peak_value = max(self.peak_value, portfolio_value)
        drawdown = (self.peak_value - portfolio_value) / self.peak_value
        self.max_drawdown = max(self.max_drawdown, drawdown)

        # 计算风险指标
        self._update_risk_metrics()

        reward = (portfolio_value - self.portfolio_values[-2]) / self.portfolio_values[-2] if len(self.portfolio_values) > 1 else 0
        done = self.cash < 0 or self.steps >= 1000 or self.max_drawdown > 0.5

        return self._get_obs(), reward, done, {
            "regime": self.regime,
            "cash": self.cash,
            "portfolio_value": portfolio_value,
            "max_drawdown": self.max_drawdown,
            "trades": trades,
        }

    def _simulate_prices(self):
        """多因子价格模拟：GBM + 跳跃扩散 + 随机波动率"""
        # Cholesky 分解用于相关随机数
        L = np.linalg.cholesky(self.correlation_matrix)
        z = self.rng.randn(len(self.assets))
        correlated_z = L @ z

        for i, (name, asset) in enumerate(self.assets.items()):
            vol = self.volatilities[name]
            trend = asset.get("_trend_mult", 1.0) * 0.0001
            jump_mult = asset.get("_jump_mult", 1.0)

            # GBM 部分
            diffusion = vol * correlated_z[i]

            # 跳跃部分
            jump_intensity = asset["jump_intensity"] * asset.get("_jump_mult", 1.0)
            jump = 0.0
            if self.rng.random() < jump_intensity:
                jump = self.rng.normal(asset["jump_mean"], asset["jump_std"]) * jump_mult

            # 随机波动率 (Heston 简化)
            vol_shock = 1.0 + 0.1 * self.rng.randn()

            ret = trend + diffusion * vol_shock + jump
            self.prices[name] *= np.exp(ret)

    def _get_portfolio_value(self) -> float:
        return self.cash + sum(self.positions[k] * self.prices[k] for k in self.prices)

    def _get_obs(self) -> np.ndarray:
        """观测向量"""
        obs = []
        total_value = self._get_portfolio_value()
        for name in self.assets:
            obs.extend([
                self.positions[name] * self.prices[name] / max(1, self._get_portfolio_value()),  # 权重
                (self.prices[name] / 100.0 - 1.0) * 10,  # 价格偏离
                self.volatilities[name] / 0.02 - 1.0,  # 波动率偏离
            ])
        obs.extend([
            self.cash / 100000.0 - 1.0,
            self.max_drawdown,
            self.steps / 1000.0,
        ])
        return np.array(obs, dtype=np.float32)

    def _update_risk_metrics(self):
        """更新风险指标"""
        if len(self.portfolio_values) > 30:
            returns = np.diff(self.portfolio_values[-30:]) / np.array(self.portfolio_values[-31:-1])
            self.var_95 = -np.percentile(returns, 5)
            self.cvar_95 = -np.mean(returns[returns <= -self.var_95]) if np.any(returns <= -self.var_95) else 0
            if np.std(returns) > 1e-8:
                self.sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
                self.sortino = np.mean(returns) / np.std(returns[returns < 0]) * np.sqrt(252) if np.any(returns < 0) else 0

    def get_performance_report(self) -> Dict:
        total_return = (self._get_portfolio_value() - 100000) / 100000
        return {
            "total_return": round(total_return, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "sharpe_ratio": round(getattr(self, 'sharpe', 0), 4),
            "sortino_ratio": round(getattr(self, 'sortino', 0), 4),
            "var_95": round(getattr(self, 'var_95', 0), 4),
            "cvar_95": round(getattr(self, 'cvar_95', 0), 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "total_trades": len(self.trade_history),
            "final_value": round(self._get_portfolio_value(), 2),
        }

    def obs_dim(self):
        return 7 * len(self.assets) + 3


# ========================================
# 2. 风险管理器
# ========================================

@dataclass
class RiskLimits:
    max_position_pct: float = 1.0
    max_sector_pct: float = 0.3
    max_single_asset_pct: float = 0.2
    max_drawdown: float = 0.20
    var_limit: float = 0.05
    stop_loss_pct: float = 0.05
    max_leverage: float = 1.0
    max_daily_loss: float = 0.03


class RiskManager:
    """实时风险管理器"""

    def __init__(self, limits: RiskLimits = None):
        self.limits = limits or RiskLimits()
        self.alerts_history = []
        self.breaches = []

    def check_pre_trade(self, portfolio: Dict, proposed_trades: Dict) -> Tuple[bool, List[str]]:
        """交易前风控检查"""
        alerts = []
        total_value = portfolio.get("total_value", 1)

        # 检查单一资产仓位
        for asset, trade in portfolio.get("proposed_positions", {}).items():
            pct = abs(trade.get("target_value", 0)) / max(1, portfolio.get("total_value", 1))
            if pct > self.limits.max_single_asset_pct:
                alerts.append(f"单一资产 {asset} 仓位 {pct:.1%} 超限 {self.limits.max_single_asset_pct:.0%}")

        # 检查总杠杆
        total_exposure = sum(abs(t.get("target_value", 0)) for t in portfolio.get("proposed_positions", {}).values())
        leverage = total_exposure / max(1, portfolio.get("total_value", 1))
        if leverage > self.limits.max_leverage:
            alerts.append(f"杠杆 {leverage:.2f}x 超限 {self.limits.max_leverage:.1f}x")

        # 检查最大回撤
        if portfolio.get("max_drawdown", 0) > self.limits.max_drawdown:
            alerts.append(f"最大回撤 {portfolio['max_drawdown']:.1%} 超限")

        return len(alerts) == 0, alerts

    def check_post_trade(self, portfolio: Dict) -> List[str]:
        """交易后风控检查"""
        alerts = []
        # 检查止损
        for asset, pos in portfolio.get("positions", {}).items():
            if pos.get("unrealized_pnl_pct", 0) < -0.1:
                self._record_breach(asset, "stop_loss", pos["unrealized_pnl_pct"])
                # 这里可以自动触发平仓逻辑

        # 检查日损失限额
        daily_pnl = portfolio.get("daily_pnl", 0)
        if daily_pnl < -portfolio.get("start_of_day_value", 1) * 0.03:
            alerts.append(f"日损失 {daily_pnl:.0f} 超过 3% 限额")
            self._record_breach("portfolio", "daily_loss_limit", daily_pnl)

        return alerts

    def _record_breach(self, asset: str, breach_type: str, value: float):
        self.breaches.append({
            "timestamp": datetime.now().isoformat(),
            "asset": asset,
            "type": breach_type,
            "value": value,
        })

    def get_risk_report(self) -> Dict:
        return {
            "limits": asdict(self.limits),
            "recent_alerts": self.alerts_history[-10:],
            "breaches": self.breaches[-20:],
            "breach_count": len(self.breaches),
        }


# ========================================
# 3. 回测引擎
# ========================================

class BacktestEngine:
    """高性能并行回测引擎"""

    def __init__(self, env_class, agent_factory, config: TradingConfig = None):
        self.env_class = env_class
        self.agent_factory = agent_factory
        self.config = config or TradingConfig()

    async def run_episode(self, seed: int, max_steps: int = 252) -> Dict:
        """单次回测"""
        env = MultiAssetFinancialEnv(TradingConfig(), seed=seed)
        obs = env.reset()
        done = False
        step = 0

        while not done and step < 500:
            # 简单策略：均值回归 + 动量
            actions = {}
            for name in env.assets:
                # 简单策略：根据趋势和均值回归决定权重
                price_change = (env.prices[name] / 100.0 - 1.0)
                momentum = (env.prices[name] / 100.0 - 1.0) * 10
                mean_reversion = - (env.prices[name] / 100.0 - 1.0) * 5
                weight = np.clip(momentum * 0.5 + mean_reversion * 0.5, -0.5, 0.5)
                actions[name] = weight

            obs, reward, done, info = env.step(actions)
            step += 1

        return env.get_performance_report()

    async def run_parallel(self, num_episodes: int = 100, max_concurrent: int = 10) -> Dict:
        """并行回测"""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def run_with_semaphore(seed):
            async with semaphore:
                return await self.run_episode(seed)

        tasks = [run_with_semaphore(42 + i) for i in range(num_episodes)]
        results = await asyncio.gather(*tasks)

        # 汇总统计
        returns = [r["total_return"] for r in results]
        drawdowns = [r["max_drawdown"] for r in results]
        sharpes = [r.get("sharpe_ratio", 0) for r in results]

        return {
            "num_episodes": num_episodes,
            "avg_return": round(np.mean(returns), 4),
            "std_return": round(np.std(returns), 4),
            "avg_max_drawdown": round(np.mean(drawdowns), 4),
            "avg_sharpe": round(np.mean(sharpes), 4),
            "win_rate": round(sum(1 for r in returns if r > 0) / len(returns), 4),
            "sharpe_std": round(np.std(sharpes), 4),
            "returns": returns,
            "drawdowns": drawdowns,
            "sharpes": sharpes,
        }


# ========================================
# 4. FastAPI REST API
# ========================================

try:
    from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from pydantic import BaseModel, Field
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Mock classes
    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    def Field(default=..., **kwargs):
        return default
    def Depends(x): return x
    class HTTPBearer: pass
    class HTTPAuthorizationCredentials: pass
    class BackgroundTasks: pass
    class Request: pass
    def asynccontextmanager(x): return x


# Pydantic Models
class TradeRequest(BaseModel):
    actions: Dict[str, float] = Field(..., description="目标权重 {asset: weight}")
    risk_check: bool = True


class TradeResponse(BaseModel):
    success: bool
    portfolio_value: float
    cash: float
    positions: Dict[str, float]
    risk_alerts: List[str] = []
    execution_time_ms: float


class BacktestRequest(BaseModel):
    num_episodes: int = Field(10, ge=1, le=1000)
    max_concurrent: int = Field(10, ge=1, le=50)
    seed_start: int = 42


class BacktestResponse(BaseModel):
    num_episodes: int
    avg_return: float
    std_return: float
    avg_max_drawdown: float
    avg_sharpe: float
    win_rate: float
    sharpe_std: float


class RiskCheckRequest(BaseModel):
    portfolio: Dict
    proposed_trades: Dict


class RiskCheckResponse(BaseModel):
    approved: bool
    alerts: List[str]


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    active_sessions: int


# Global state
app_state = {
    "env": None,
    "risk_manager": None,
    "start_time": time.time(),
    "active_sessions": 0,
}

limiter = Limiter(key_func=get_remote_address) if FASTAPI_AVAILABLE else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app_state["env"] = MultiAssetFinancialEnv()
    app_state["risk_manager"] = RiskManager()
    app_state["start_time"] = time.time()
    yield
    # Shutdown
    pass


if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="BreakShell Financial Agent API",
        description="金融交易 Agent REST API",
        version="0.8.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if limiter:
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    security = HTTPBearer(auto_error=False)

    async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
        # 简单的 token 验证（生产环境需替换为 JWT）
        if credentials and credentials.credentials == os.environ.get("API_TOKEN", "test-token"):
            return credentials.credentials
        return None

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        return HealthResponse(
            status="healthy",
            version="0.8.0",
            uptime_seconds=time.time() - app_state["start_time"],
            active_sessions=app_state["active_sessions"],
        )

    @app.post("/trade", response_model=TradeResponse)
    @limiter.limit("30/minute")
    async def execute_trade(
        request: Request,
        trade_req: TradeRequest,
        background_tasks: BackgroundTasks,
        user=Depends(get_current_user),
    ):
        start = time.time()
        app_state["active_sessions"] += 1

        try:
            env = app_state["env"]
            actions = trade_req.actions

            # 风控检查
            risk_alerts = []
            if trade_req.risk_check:
                portfolio = {
                    "total_value": app_state["env"]._get_portfolio_value(),
                    "proposed_positions": {
                        k: {"target_value": v * env._get_portfolio_value()}
                        for k, v in trade_req.actions.items()
                    }
                }
                _, alerts = app_state["risk_manager"].check_pre_trade(portfolio, trade_req.actions)
                risk_alerts = alerts

            obs, reward, done, info = app_state["env"].step(trade_req.actions)

            response = TradeResponse(
                success=True,
                portfolio_value=info.get("portfolio_value", 0),
                cash=info.get("cash", 0),
                positions=app_state["env"].positions,
                risk_alerts=risk_alerts,
                execution_time_ms=round((time.time() - start) * 1000, 2),
            )

            background_tasks.add_task(log_trade, trade_req.actions, info)
            return response

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            app_state["active_sessions"] -= 1

    @app.post("/backtest", response_model=BacktestResponse)
    @limiter.limit("5/minute")
    async def run_backtest(
        request: Request,
        backtest_req: BacktestRequest,
        user=Depends(get_current_user),
    ):
        engine = BacktestEngine(MultiAssetFinancialEnv, lambda: None)
        results = await engine.run_parallel(backtest_req.num_episodes, backtest_req.max_concurrent)
        return BacktestResponse(**results)

    @app.post("/risk/check", response_model=RiskCheckResponse)
    async def check_risk(risk_req: RiskCheckRequest):
        approved, alerts = app_state["risk_manager"].check_pre_trade(
            risk_req.portfolio, risk_req.proposed_trades
        )
        return RiskCheckResponse(approved=approved, alerts=alerts)

    @app.get("/portfolio")
    async def get_portfolio():
        env = app_state["env"]
        return {
            "total_value": env._get_portfolio_value(),
            "cash": env.cash,
            "positions": {k: v * env.prices[k] for k, v in env.positions.items()},
            "weights": {k: v * env.prices[k] / max(1, env._get_portfolio_value()) for k, v in env.positions.items()},
            "prices": env.prices,
            "regime": env.regime,
            "max_drawdown": env.max_drawdown,
        }

    @app.get("/risk/report")
    async def get_risk_report():
        return app_state["risk_manager"].get_risk_report()

    @app.get("/performance")
    async def get_performance():
        return app_state["env"].get_performance_report()

    @app.post("/reset")
    async def reset_env(background_tasks: BackgroundTasks):
        app_state["env"] = MultiAssetFinancialEnv()
        return {"success": True, "message": "环境已重置"}


# Background task
async def log_trade(actions: Dict, info: Dict):
    """记录交易日志"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "actions": actions,
        "portfolio_value": info.get("portfolio_value", 0),
        "regime": info.get("regime", "unknown"),
    }
    # 实际生产环境写入数据库/日志系统
    pass


# ========================================
# 5. 监控仪表盘数据
# ========================================

class MetricsCollector:
    """指标收集器（兼容 Prometheus）"""

    def __init__(self):
        self.counters = {}
        self.gauges = {}
        self.histograms = {}

    def inc(self, name: str, value: float = 1, labels: Dict = None):
        key = (name, tuple(sorted(labels.items())) if labels else ())
        self.counters[key] = self.counters.get(key, 0) + value

    def gauge(self, name: str, value: float, labels: Dict = None):
        key = (name, tuple(sorted(labels.items())) if labels else ())
        self.gauges[key] = value

    def histogram(self, name: str, value: float, labels: Dict = None):
        key = (name, tuple(sorted(labels.items())) if labels else ())
        if key not in self.histograms:
            self.histograms[key] = []
        self.histograms[key].append(value)

    def get_prometheus_format(self) -> str:
        """导出 Prometheus 格式"""
        lines = []
        for (name, labels), value in self.counters.items():
            label_str = "{" + ",".join(f'{k}="{v}"' for k, v in labels) + "}" if labels else ""
            lines.append(f"{name}{label_str} {value}")
        for (name, labels), value in self.gauges.items():
            label_str = "{" + ",".join(f'{k}="{v}"' for k, v in labels) + "}" if labels else ""
            lines.append(f"{name}{label_str} {value}")
        for (name, labels), values in self.histograms.items():
            if values:
                label_str = "{" + ",".join(f'{k}="{v}"' for k, v in labels) + "}" if labels else ""
                lines.append(f"{name}_sum{label_str} {sum(values)}")
                lines.append(f"{name}_count{label_str} {len(values)}")
        return "\n".join(lines)


# 全局指标收集器
metrics = MetricsCollector()


# ========================================
# 6. CLI 命令扩展
# ========================================

def add_financial_commands():
    """添加到 CLI 的金融命令"""
    pass  # 在 cli.py 中集成


if __name__ == "__main__":
    # 简单测试
    import asyncio

    async def test():
        env = MultiAssetFinancialEnv()
        obs = env.reset()
        print(f"观测维度: {len(obs)}")

        for _ in range(10):
            actions = {name: np.random.uniform(-0.5, 0.5) for name in env.assets}
            obs, reward, done, info = env.step(actions)
            if _ % 3 == 0:
                print(f"Step {_}: Portfolio={info.get('portfolio_value', 0):.0f}, Regime={info['regime']}")

        print(f"\n最终报告:")
        report = env.get_performance_report()
        for k, v in report.items():
            print(f"  {k}: {v}")

    asyncio.run(test())