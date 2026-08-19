# BreakShell 评测报告

生成时间: 2026-08-19T18:12:15.851537

## 评测概览
- 总测试数: 28
- 通过: 28
- 失败: 0
- 总分: 100.00%

## 分类统计
- basic_tool: 6/6 (100%)
- error_recovery: 3/3 (100%)
- safety: 3/3 (100%)
- multi_step: 3/3 (100%)
- reasoning: 3/3 (100%)
- advanced: 3/3 (100%)
- sec_bench: 7/7 (100%)

## 性能基准
- 工具平均执行时间: 7.12ms
- Agent Loop 平均耗时: 13.78ms
  - list_dir: 1.97ms (p95: 3.34ms)
  - read_file: 0.14ms (p95: 0.19ms)
  - shell: 15.03ms (p95: 19.34ms)
  - grep_files: 11.36ms (p95: 17.01ms)