# BreakShell 评测报告

生成时间: 2026-08-19T15:45:33.972109

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
- 工具平均执行时间: 5.56ms
- Agent Loop 平均耗时: 9.12ms
  - list_dir: 1.15ms (p95: 1.33ms)
  - read_file: 0.12ms (p95: 0.15ms)
  - shell: 13.76ms (p95: 16.23ms)
  - grep_files: 7.23ms (p95: 9.26ms)