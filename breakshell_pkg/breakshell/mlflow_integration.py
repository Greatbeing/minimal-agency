# -*- coding: utf-8 -*-
"""
BreakShell MLflow 模型版本管理集成
====================================
提供模型训练、版本控制、实验跟踪、模型注册和部署
"""

from __future__ import annotations

import os
import json
import pickle
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

try:
    import mlflow
    import mlflow.pytorch
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    from mlflow.entities import ViewType
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

import torch
import torch.nn as nn


# ========================================
# 1. 配置与常量
# ========================================

class MLflowConfig:
    """MLflow 配置"""
    
    def __init__(self):
        self.tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        self.experiment_name = os.environ.get("MLFLOW_EXPERIMENT", "breakshell-experiments")
        self.model_registry_name = os.environ.get("MLFLOW_MODEL_REGISTRY", "breakshell-models")
        self.artifact_location = os.environ.get("MLFLOW_ARTIFACT_LOCATION", None)
        
    def setup(self):
        """初始化 MLflow"""
        if not MLFLOW_AVAILABLE:
            raise RuntimeError("MLflow 未安装，请运行: pip install mlflow")
        
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # 创建或获取实验
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                self.experiment_name,
                artifact_location=self.artifact_location
            )
        else:
            experiment_id = experiment.experiment_id
        
        mlflow.set_experiment(self.experiment_name)
        return experiment_id


MLFLOW_CONFIG = MLflowConfig()


# ========================================
# 2. 模型包装器
# ========================================

if MLFLOW_AVAILABLE:
    class BreakShellMLflowModel(mlflow.pyfunc.PythonModel):
        """BreakShell 模型的 MLflow 包装器"""
        
        def load_context(self, context):
            """加载模型上下文"""
            import torch
            from breakshell import BreakShell
            
            # 加载 PyTorch 模型
            model_path = context.artifacts["model"]
            self.model = BreakShell(action_dim=3)
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
            self.model.eval()
            
            # 加载配置
            config_path = context.artifacts.get("config")
            if config_path:
                with open(config_path, "r") as f:
                    self.config = json.load(f)
            else:
                self.config = {}
        
        def predict(self, context, model_input):
            """预测接口"""
            import torch
            
            # 支持多种输入格式
            if isinstance(model_input, dict):
                obs = model_input.get("observation")
                if obs is None:
                    obs = model_input.get("obs")
            else:
                obs = model_input
            
            # 转换为 tensor
            if isinstance(obs, list):
                obs = np.array(obs)
            if isinstance(obs, np.ndarray):
                obs = torch.FloatTensor(obs)
            
            # 确保批次维度
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            
            with torch.no_grad():
                action, info = self.model.act(obs)
            
            return {
                "action": int(action) if isinstance(action, (int, np.integer)) else action.tolist(),
                "log_prob": info.get("log_prob", 0.0),
                "value": info.get("value", 0.0)
            }
else:
    BreakShellMLflowModel = None


# ========================================
# 3. 实验管理器
# ========================================

@dataclass
class ExperimentConfig:
    """实验配置"""
    name: str
    description: str = ""
    tags: Dict[str, str] = None
    params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
        if self.params is None:
            self.params = {}


class ExperimentManager:
    """MLflow 实验管理器"""
    
    def __init__(self, config: MLflowConfig = None):
        self.config = config or MLFLOW_CONFIG
        if MLFLOW_AVAILABLE:
            self.config.setup()
        self.client = MlflowClient() if MLFLOW_AVAILABLE else None
    
    def start_run(self, run_name: str = None, tags: Dict = None, params: Dict = None) -> str:
        """开始新实验运行"""
        if not MLFLOW_AVAILABLE:
            raise RuntimeError("MLflow 未安装")
        
        run = mlflow.start_run(run_name=run_name)
        run_id = run.info.run_id
        
        if tags:
            mlflow.set_tags(tags)
        if params:
            mlflow.log_params(params)
        
        return run_id
    
    def end_run(self, status: str = "FINISHED"):
        """结束实验运行"""
        if MLFLOW_AVAILABLE:
            mlflow.end_run(status=status)
    
    def log_metric(self, key: str, value: float, step: int = None):
        """记录指标"""
        if MLFLOW_AVAILABLE:
            mlflow.log_metric(key, value, step=step)
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """批量记录指标"""
        if MLFLOW_AVAILABLE:
            for k, v in metrics.items():
                mlflow.log_metric(k, v, step=step)
    
    def log_param(self, key: str, value: Any):
        """记录参数"""
        if MLFLOW_AVAILABLE:
            mlflow.log_param(key, value)
    
    def log_params(self, params: Dict[str, Any]):
        """批量记录参数"""
        if MLFLOW_AVAILABLE:
            mlflow.log_params(params)
    
    def log_artifact(self, local_path: str, artifact_path: str = None):
        """记录工件"""
        if MLFLOW_AVAILABLE:
            mlflow.log_artifact(local_path, artifact_path)
    
    def log_artifacts(self, local_dir: str, artifact_path: str = None):
        """批量记录工件"""
        if MLFLOW_AVAILABLE:
            mlflow.log_artifacts(local_dir, artifact_path)
    
    def log_model(self, model, artifact_path: str = "model", **kwargs):
        """记录模型"""
        if MLFLOW_AVAILABLE:
            mlflow.pyfunc.log_model(
                artifact_path=artifact_path,
                python_model=BreakShellMLflowModel(),
                artifacts=kwargs.get("artifacts", {}),
                **{k: v for k, v in kwargs.items() if k != "artifacts"}
            )
    
    def get_run(self, run_id: str):
        """获取运行信息"""
        if MLFLOW_AVAILABLE:
            return self.client.get_run(run_id)
        return None
    
    def search_runs(self, experiment_ids: List[str] = None, filter_string: str = "", 
                    max_results: int = 100, order_by: List[str] = None):
        """搜索运行"""
        if MLFLOW_AVAILABLE:
            return self.client.search_runs(
                experiment_ids=experiment_ids or [mlflow.get_experiment_by_name(self.config.experiment_name).experiment_id],
                filter_string=filter_string,
                max_results=max_results,
                order_by=order_by or ["attributes.start_time DESC"]
            )
        return []
    
    def get_best_run(self, metric: str = "eval_reward", ascending: bool = False) -> Optional[str]:
        """获取最佳运行"""
        runs = self.search_runs(
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
            max_results=1
        )
        return runs[0].info.run_id if runs else None


# ========================================
# 4. 模型注册管理器
# ========================================

class ModelRegistryManager:
    """模型注册管理器"""
    
    def __init__(self, config: MLflowConfig = None):
        self.config = config or MLFLOW_CONFIG
        self.client = MlflowClient() if MLFLOW_AVAILABLE else None
        self.model_name = self.config.model_registry_name
    
    def register_model(self, run_id: str, model_path: str = "model", 
                       version_description: str = None) -> int:
        """注册模型到模型注册表"""
        if not MLFLOW_AVAILABLE:
            raise RuntimeError("MLflow 未安装")
        
        model_uri = f"runs:/{run_id}/{model_path}"
        
        # 注册模型
        mv = mlflow.register_model(model_uri, self.model_name)
        
        # 更新版本描述
        if version_description:
            self.client.update_model_version(
                name=self.model_name,
                version=mv.version,
                description=version_description
            )
        
        return mv.version
    
    def transition_model_version(self, version: int, stage: str, 
                                  archive_existing: bool = True):
        """转换模型版本阶段"""
        if not MLFLOW_AVAILABLE:
            raise RuntimeError("MLflow 未安装")
        
        self.client.transition_model_version_stage(
            name=self.model_name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing
        )
    
    def get_latest_version(self, stage: str = "Production") -> Optional[int]:
        """获取指定阶段的最新版本"""
        if not MLFLOW_AVAILABLE:
            return None
        
        versions = self.client.get_latest_versions(self.model_name, stages=[stage])
        return int(versions[0].version) if versions else None
    
    def get_model_version(self, version: int):
        """获取模型版本详情"""
        if not MLFLOW_AVAILABLE:
            return None
        return self.client.get_model_version(self.model_name, version)
    
    def list_model_versions(self, stages: List[str] = None) -> List[Dict]:
        """列出所有模型版本"""
        if not MLFLOW_AVAILABLE:
            return []
        
        versions = self.client.get_latest_versions(self.model_name, stages=stages or ["None", "Staging", "Production", "Archived"])
        return [
            {
                "version": int(v.version),
                "stage": v.current_stage,
                "description": v.description,
                "creation_time": v.creation_timestamp,
                "run_id": v.run_id,
            }
            for v in versions
        ]
    
    def set_model_tag(self, key: str, value: str):
        """设置模型标签"""
        if not MLFLOW_AVAILABLE:
            return
        self.client.set_registered_model_tag(self.model_name, key, value)
    
    def load_model(self, version: int = None, stage: str = "Production"):
        """加载模型"""
        if not MLFLOW_AVAILABLE:
            raise RuntimeError("MLflow 未安装")
        
        if version:
            model_uri = f"models:/{self.model_name}/{version}"
        else:
            model_uri = f"models:/{self.model_name}/{stage}"
        
        return mlflow.pyfunc.load_model(model_uri)


# ========================================
# 5. 训练集成
# ========================================

class BreakShellTrainer:
    """BreakShell 训练器 with MLflow 集成"""
    
    def __init__(self, experiment_manager: ExperimentManager = None,
                 registry_manager: ModelRegistryManager = None):
        self.experiment_manager = experiment_manager or ExperimentManager()
        self.registry_manager = registry_manager or ModelRegistryManager()
    
    def train_with_tracking(self, 
                           env_name: str = "capability",
                           episodes: int = 500,
                           lr: float = 0.005,
                           run_name: str = None,
                           tags: Dict = None,
                           register_model: bool = True,
                           model_stage: str = "Staging") -> Dict[str, Any]:
        """带跟踪的训练"""
        from breakshell import BreakShell, CapabilityEnv, EnergyEnv, FinancialEnv
        
        # 环境映射
        env_map = {
            "capability": CapabilityEnv,
            "energy": EnergyEnv,
            "financial": FinancialEnv,
        }
        
        EnvClass = env_map.get(env_name, CapabilityEnv)
        env = EnvClass()
        
        # 创建 Agent
        obs_dim = env.obs_dim()
        action_dim = env.action_dim()
        agent = BreakShell(action_dim=action_dim, lr=lr)
        
        # 实验配置
        exp_tags = {
            "env": env_name,
            "algorithm": "REINFORCE",
            "framework": "pytorch",
            **(tags or {})
        }
        exp_params = {
            "episodes": episodes,
            "lr": lr,
            "obs_dim": env.obs_dim(),
            "action_dim": action_dim,
        }
        
        # 开始实验
        run_id = self.experiment_manager.start_run(
            run_name=run_name or f"breakshell-{env_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            tags=exp_tags,
            params=exp_params
        )
        
        try:
            # 训练循环
            episode_rewards = []
            for episode in range(episodes):
                obs = env.reset()
                episode_reward = 0
                done = False
                
                while not done:
                    action, info = agent.act()
                    obs, reward, done, _ = env.step(action)
                    agent.add_step(action, reward)
                    episode_reward += reward
                
                episode_rewards.append(episode_reward)
                
                # 记录每轮指标
                if episode % 10 == 0:
                    self.experiment_manager.log_metric("episode_reward", episode_reward, step=episode)
                    self.experiment_manager.log_metric("avg_reward_10", np.mean(episode_rewards[-10:]), step=episode)
                
                # 策略更新
                if len(agent.history) >= 32:
                    agent.update_policy()
            
            # 最终评估
            eval_reward = agent.evaluate(env, num_episodes=50)
            self.experiment_manager.log_metric("final_eval_reward", eval_reward)
            self.experiment_manager.log_metric("avg_train_reward", np.mean(episode_rewards))
            
            # 保存模型
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = os.path.join(tmpdir, "model.pt")
                agent.save(model_path)
                
                config_path = os.path.join(tmpdir, "config.json")
                with open(config_path, "w") as f:
                    json.dump({
                        "obs_dim": obs_dim,
                        "action_dim": action_dim,
                        "lr": lr,
                        "episodes": episodes,
                        "env": env_name,
                    }, f)
                
                # 记录模型
                self.experiment_manager.log_model(
                    agent,
                    artifact_path="model",
                    artifacts={
                        "model": model_path,
                        "config": config_path,
                    }
                )
            
            result = {
                "run_id": run_id,
                "episodes": episodes,
                "final_eval_reward": eval_reward,
                "avg_train_reward": np.mean(episode_rewards),
                "episode_rewards": episode_rewards,
            }
            
            # 注册模型
            if register_model and eval_reward > 0:
                version = self.registry_manager.register_model(
                    run_id=run_id,
                    model_path="model",
                    version_description=f"{env_name} env, {episodes} episodes, eval_reward={eval_reward:.2f}"
                )
                
                # 推广到 Staging
                self.registry_manager.transition_model_version(version, "Staging")
                
                result["model_version"] = version
            
            return result
            
        except Exception as e:
            self.experiment_manager.end_run(status="FAILED")
            raise
        finally:
            self.experiment_manager.end_run()


# ========================================
# 6. 便捷函数
# ========================================

def setup_mlflow(tracking_uri: str = None, experiment_name: str = None) -> ExperimentManager:
    """快速设置 MLflow"""
    config = MLflowConfig()
    if tracking_uri:
        config.tracking_uri = tracking_uri
    if experiment_name:
        config.experiment_name = experiment_name
    return ExperimentManager(config)


def train_with_mlflow(env_name: str = "capability", episodes: int = 500, **kwargs) -> Dict:
    """一键训练 + MLflow 跟踪"""
    trainer = BreakShellTrainer()
    return trainer.train_with_tracking(env_name=env_name, episodes=episodes, **kwargs)


def load_production_model() -> Any:
    """加载生产环境模型"""
    registry = ModelRegistryManager()
    return registry.load_model(stage="Production")


def get_model_versions() -> List[Dict]:
    """获取所有模型版本"""
    registry = ModelRegistryManager()
    return registry.list_model_versions()


# ========================================
# 7. 自动化实验运行器
# ========================================

class AutoExperimentRunner:
    """自动化实验运行器：超参数搜索、多环境对比"""
    
    def __init__(self, experiment_manager: ExperimentManager = None):
        self.experiment_manager = experiment_manager or ExperimentManager()
    
    def run_hyperparameter_search(self, 
                                  param_grid: Dict[str, List],
                                  env_name: str = "capability",
                                  episodes: int = 200,
                                  max_runs: int = 20) -> List[Dict]:
        """超参数网格搜索"""
        from itertools import product
        
        # 生成参数组合
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(product(*values))[:max_runs]
        
        results = []
        for i, combo in enumerate(combinations):
            params = dict(zip(keys, combo))
            print(f"Running {i+1}/{len(combinations)}: {params}")
            
            try:
                trainer = BreakShellTrainer()
                result = trainer.train_with_tracking(
                    env_name=env_name,
                    episodes=episodes,
                    run_name=f"hp-search-{env_name}-{i}",
                    tags={"type": "hyperparameter_search", "trial": str(i)},
                    register_model=False,
                    **params
                )
                result["params"] = params
                results.append(result)
            except Exception as e:
                print(f"Trial {i} failed: {e}")
                results.append({"params": params, "error": str(e)})
        
        # 按评估奖励排序
        results.sort(key=lambda x: x.get("final_eval_reward", -float("inf")), reverse=True)
        return results
    
    def run_multi_env_comparison(self, 
                                 environments: List[str] = None,
                                 episodes: int = 500) -> Dict[str, Dict]:
        """多环境对比实验"""
        environments = environments or ["capability", "energy", "financial"]
        results = {}
        
        for env in environments:
            print(f"Training on {env}...")
            trainer = BreakShellTrainer()
            result = trainer.train_with_tracking(
                env_name=env,
                episodes=episodes,
                run_name=f"multi-env-{env}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                tags={"type": "multi_env_comparison", "env": env},
                register_model=True
            )
            results[env] = result
        
        return results


# ========================================
# 7. 导出
# ========================================

__all__ = [
    "MLflowConfig",
    "MLFLOW_CONFIG",
    "MLFLOW_AVAILABLE",
    "BreakShellMLflowModel",
    "ExperimentConfig",
    "ExperimentManager",
    "ModelRegistryManager",
    "BreakShellTrainer",
    "AutoExperimentRunner",
    "setup_mlflow",
    "train_with_mlflow",
    "load_production_model",
    "get_model_versions",
]

if __name__ == "__main__":
    # 测试 MLflow 集成
    if MLFLOW_AVAILABLE:
        print("MLflow 可用")
        
        # 设置
        exp_manager = setup_mlflow(
            tracking_uri="http://localhost:5000",
            experiment_name="breakshell-test"
        )
        
        # 运行实验
        run_id = exp_manager.start_run(
            run_name="test-run",
            tags={"test": "true"},
            params={"lr": 0.001}
        )
        
        exp_manager.log_metric("test_metric", 0.95)
        exp_manager.log_param("test_param", "value")
        
        print(f"Run ID: {run_id}")
        
        # 注册模型
        registry = ModelRegistryManager()
        print(f"Model versions: {registry.list_model_versions()}")
        
        exp_manager.end_run()
    else:
        print("MLflow 未安装")