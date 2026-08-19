# BreakShell 评测报告

生成时间: 2026-08-19T11:50:15.901820

## 评测概览
- 总测试数: 21
- 通过: 21
- 失败: 0
- 总分: 100.00%

## 分类统计
- basic_tool: 6/6 (100%)
- error_recovery: 3/3 (100%)
- safety: 3/3 (100%)
- multi_step: 3/3 (100%)
- reasoning: 3/3 (100%)
- advanced: 3/3 (100%)

## 性能基准
- 工具平均执行时间: 8.01ms
- Agent Loop 平均耗时: 29.66ms
  - list_dir: 3.84ms (p95: 5.76ms)
  - read_file: 0.13ms (p95: 0.18ms)
  - shell: 19.99ms (p95: 25.46ms)
  - grep_files: 8.07ms (p95: 11.24ms)