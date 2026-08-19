# BreakShell 评测报告

生成时间: 2026-08-19T17:40:02.872392

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
- 工具平均执行时间: 6.57ms
- Agent Loop 平均耗时: 11.14ms
  - list_dir: 1.55ms (p95: 2.35ms)
  - read_file: 0.15ms (p95: 0.23ms)
  - shell: 15.12ms (p95: 17.73ms)
  - grep_files: 9.45ms (p95: 11.9ms)