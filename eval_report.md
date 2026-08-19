# BreakShell 评测报告

生成时间: 2026-08-19T11:51:11.716980

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
- 工具平均执行时间: 6.91ms
- Agent Loop 平均耗时: 33.97ms
  - list_dir: 3.33ms (p95: 4.44ms)
  - read_file: 0.12ms (p95: 0.18ms)
  - shell: 17.06ms (p95: 20.77ms)
  - grep_files: 7.15ms (p95: 10.04ms)