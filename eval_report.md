# BreakShell 评测报告

生成时间: 2026-08-19T16:08:55.517608

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
- 工具平均执行时间: 9.83ms
- Agent Loop 平均耗时: 12.54ms
  - list_dir: 1.52ms (p95: 2.12ms)
  - read_file: 0.29ms (p95: 0.23ms)
  - shell: 19.16ms (p95: 22.73ms)
  - grep_files: 18.35ms (p95: 15.14ms)