# 贡献指南

感谢你对**有我 (Self-Present)** 的兴趣！这是一个研究主体性涌现的开源项目。

## 如何贡献

### 报告问题

使用 [GitHub Issues](https://github.com/Greatbeing/minimal-agency/issues) 报告 Bug 或提出建议。

### 提交代码

1. Fork 仓库
2. 创建分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

### 代码规范

- 遵循 PEP 8
- 所有公共函数需要 docstring
- 新增功能需要对应测试

## 开发设置

```bash
git clone https://github.com/Greatbeing/minimal-agency.git
cd minimal-agency
pip install -r requirements.txt
pip install -e ".[dev]"
```

## 运行测试

```bash
pytest tests/ -v
```

## 行为准则

请阅读我们的 [行为准则](CODE_OF_CONDUCT.md)。

## 许可证

通过提交代码，你同意你的贡献将在 MIT 许可证下发布。
