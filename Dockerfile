# BreakShell Financial Agent - Dockerfile
# 多阶段构建：构建阶段 + 运行阶段

# ========================================
# 构建阶段
# ========================================
FROM python:3.11-slim as builder

WORKDIR /app

# 安装构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .
COPY pyproject.toml .

# 安装 Python 依赖
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 复制源代码
COPY breakshell_pkg/ ./breakshell_pkg/
COPY breakshell_pkg/breakshell/ ./breakshell_pkg/breakshell/

# 安装包
RUN pip install --no-cache-dir -e ./breakshell_pkg


# ========================================
# 运行阶段
# ========================================
FROM python:3.11-slim as runtime

# 创建非 root 用户
RUN groupadd -r breakshell && useradd -r -g breakshell breakshell

WORKDIR /app

# 复制构建产物
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /app /app

# 创建必要目录
RUN mkdir -p /app/logs /app/data /app/config && \
    chown -R breakshell:breakshell /app

# 切换到非 root 用户
USER breakshell

# 环境变量
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    BREAKSHELL_ENV=production \
    API_HOST=0.0.0.0 \
    API_PORT=8000

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" || exit 1

# 启动命令
CMD ["python", "-m", "uvicorn", "breakshell.financial_product:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]