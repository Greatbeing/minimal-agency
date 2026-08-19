# BreakShell 评测报告

生成时间: 2026-08-19T12:03:10.985417

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
- 工具平均执行时间: 8.78ms
- Agent Loop 平均耗时: 30.01ms
  - list_dir: 5.37ms (p95: 8.01ms)
  - read_file: 0.14ms (p95: 0.17ms)
  - shell: 20.98ms (p95: 25.82ms)
  - grep_files: 8.63ms (p95: 11.92ms)