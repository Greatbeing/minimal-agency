# BreakShell Financial Agent - 部署指南

## 📋 目录
1. [快速开始](#快速开始)
2. [环境要求](#环境要求)
3. [本地开发](#本地开发)
4. [Docker 部署](#docker-部署)
5. [生产环境部署](#生产环境部署)
6. [监控与告警](#监控与告警)
7. [常见问题](#常见问题)

---

## 🚀 快速开始

### 1. 克隆仓库
```bash
git clone https://github.com/Greatbeing/minimal-agency.git
cd minimal-agency
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
pip install -e ./breakshell_pkg
```

### 3. 运行测试
```bash
breakshell eval
# 应该显示: 28/28 通过 (100%)
```

### 4. 启动 API 服务
```bash
python -m uvicorn breakshell.financial_product:app --host 0.0.0.0 --port 8000 --reload
```

### 5. 验证服务
```bash
curl http://localhost:8000/health
# {"status":"healthy","version":"0.9.0","uptime_seconds":...}
```

---

## 🔧 环境要求

### 必需
- Python 3.11+
- pip 23.0+
- Git

### 可选（生产环境）
- Docker 24.0+ / Docker Compose 2.20+
- Redis 7.0+
- Prometheus 2.47+
- Grafana 10.1+

### API Keys（可选）
```bash
export PROFY_API_KEY="your-profy-key"
export DEEPSEEK_API_KEY="your-deepseek-key"
export API_TOKEN="your-secure-token"
```

---

## 💻 本地开发

### 运行 CLI
```bash
# 列出文件
breakshell run "列出当前目录的所有文件"

# 训练 RL Agent
breakshell train --env capability --episodes 500 --output my_agent

# 评估 Agent
breakshell evaluate --model my_agent --episodes 100

# 对比实验
breakshell compare --env capability --episodes 500

# 性能基准
breakshell benchmark

# 评测
breakshell eval

# 认知 Agent
breakshell cognitive --goal "分析项目结构"

# 知识银行
breakshell knowledge import README.md
breakshell knowledge search "主体性"
```

### 运行金融环境
```python
from breakshell import MultiAssetFinancialEnv

env = MultiAssetFinancialEnv()
obs = env.reset()

for _ in range(100):
    actions = {name: 0.1 for name in env.assets}
    obs, reward, done, info = env.step(actions)
    
    if _ % 10 == 0:
        print(f"Step {_}: Portfolio={info['portfolio_value']:.0f}")

print(env.get_performance_report())
```

---

## 🐳 Docker 部署

### 构建镜像
```bash
docker build -t breakshell:latest .
```

### 运行容器
```bash
docker run -d \
  --name breakshell-api \
  -p 8000:8000 \
  -e API_TOKEN=your-token \
  -e PROFY_API_KEY=$PROFY_API_KEY \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/data:/app/data \
  breakshell:latest
```

### 验证
```bash
curl -H "Authorization: Bearer your-token" http://localhost:8000/health
```

---

## 🏭 生产环境部署

### 1. 准备环境变量
```bash
# 创建 .env 文件
cat > .env << EOF
API_TOKEN=your-secure-random-token
PROFY_API_KEY=your-profy-key
DEEPSEEK_API_KEY=your-deepseek-key
GRAFANA_PASSWORD=secure-password
GRAFANA_PASSWORD=secure-password
EOF
```

### 2. 启动完整栈
```bash
docker-compose up -d
```

### 3. 验证服务
```bash
# API 健康检查
curl -H "Authorization: Bearer your-token" http://localhost:8000/health

# 访问 Grafana
open http://localhost:3000
# 用户名: admin, 密码: 你的 GRAFANA_PASSWORD

# 访问 Prometheus
open http://localhost:9090
```

### 4. 访问服务
| 服务 | 地址 | 说明 |
|------|------|------|
| API | http://localhost:8000 | REST API |
| API 文档 | http://localhost:8000/docs | Swagger UI |
| Grafana | http://localhost:3000 | 监控仪表盘 |
| Prometheus | http://localhost:9090 | 指标查询 |
| Grafana 仪表盘 | http://localhost:3000/d/breakshell-financial | 交易监控 |

---

## 📊 监控与告警

### 关键指标
| 指标 | 描述 | 告警阈值 |
|------|------|----------|
| `breakshell_portfolio_drawdown` | 最大回撤 | > 20% |
| `breakshell_sharpe_ratio` | 夏普比率 | < 0.5 |
| `breakshell_var_95` | VaR 95% | > 5% |
| `breakshell_max_drawdown` | 最大回撤 | > 20% |
| `breakshell_api_latency_p99` | API P99 延迟 | > 1s |
| `breakshell_error_rate` | 错误率 | > 1% |

### Grafana 仪表盘
访问 http://localhost:3000/d/breakshell-financial

预置面板：
- 交易频率 (每秒)
- 组合价值 vs 现金
- 最大回撤
- 夏普比率 / 索提诺比率
- VaR 95% / CVaR 95%
- API 请求频率 / 延迟
- CPU / 内存使用率
- 风控告警频率

### Prometheus 告警规则
```yaml
# config/prometheus/rules/alerts.yml
groups:
  - name: breakshell-alerts
    rules:
      - alert: HighDrawdown
        expr: breakshell_portfolio_drawdown > 0.2
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "最大回撤超过 20%"
          
      - alert: LowSharpeRatio
        expr: breakshell_sharpe_ratio < 0.5
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "夏普比率过低"
          
      - alert: HighErrorRate
        expr: rate(breakshell_api_errors_total[5m]) > 0.01
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "API 错误率过高"
```

---

## ❓ 常见问题

### Q: 评测失败怎么办？
```bash
# 重新安装包
pip install -e ./breakshell_pkg --force-reinstall --no-deps
breakshell eval
```

### Q: API 返回 401 Unauthorized
```bash
# 检查 Token
curl -H "Authorization: Bearer YOUR_TOKEN" http://localhost:8000/health

# 检查环境变量
echo $API_TOKEN
```

### Q: Docker 构建失败
```bash
# 清理缓存重试
docker system prune -f
docker build --no-cache -t breakshell:latest .
```

### Q: 端口冲突
```bash
# 修改端口
# docker-compose.yml 中修改 ports 映射
# 或环境变量 API_PORT=8001
```

### Q: 评测分数低
```bash
# 查看详细报告
cat eval_report.md

# 运行单个测试调试
python -m pytest tests/test_e2e.py::TestLLMAgent::test_run_agent -v
```

---

## 📚 相关文档
- [API 文档](http://localhost:8000/docs) - Swagger UI
- [SEC-Bench 评测标准](breakshell_pkg/breakshell/SEC_BENCH_STANDARD.md)
- [金融 Agent 设计](docs/FINANCIAL_AGENT_DESIGN.md)
- [开发计划](docs/DEVELOPMENT_PLAN.md)
- [论文草稿](paper.tex)

---

## 🤝 贡献指南
详见 [CONTRIBUTING.md](CONTRIBUTING.md)

---

## 📄 许可证
MIT License - 详见 [LICENSE](LICENSE)

---

**BreakShell** - 让 AI Agent 知道自己能做什么，而不是盲目行动。