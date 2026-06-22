# ETH LF Portfolio V9D Same-Side Continuation Probe

独立测试版，不修改 V8/V9C 原文件。

V9D 目的：测试“已经持有同方向仓位时，不因为引擎切换而平掉再重开同方向仓位”的逻辑。

具体规则：

- 止损仍然优先执行。
- 反向信号仍然允许退出。
- 只有 channel exit 触发时，如果当前持仓方向仍被另一个 raw engine 支持，则 suppress channel exit。
- 这避免 Momentum long exit 时，Bull Reclaim long 仍有效，却把多仓平掉的 churn。

## Reclaim first + same-side continuation

```bat
python backtest/lf/eth_lf_portfolio_v9d_same_side_continuation_backtest.py ^
  --start-date 2023-01-01 ^
  --end-date 2026-06-15 ^
  --warmup-start-date 2022-01-01 ^
  --preset turbo ^
  --bear-preset high ^
  --bull-preset high ^
  --bull-execution-mode inherit ^
  --micro-filter-mode soft ^
  --micro-not-aligned-risk-scale 0.5 ^
  --micro-contra-risk-scale 0.5 ^
  --global-risk-scale 1.3 ^
  --priority-mode reclaim_first ^
  --same-side-continuation-mode suppress_exit ^
  --range-pct 0.002 ^
  --price-step 1 ^
  --out-dir data/reports/lf/eth_lf_portfolio_v9d_same_side_continuation/reclaim_first_suppress_gs_1p3
```

## Reclaim first control: same-side continuation off

```bat
python backtest/lf/eth_lf_portfolio_v9d_same_side_continuation_backtest.py ^
  --start-date 2023-01-01 ^
  --end-date 2026-06-15 ^
  --warmup-start-date 2022-01-01 ^
  --preset turbo ^
  --bear-preset high ^
  --bull-preset high ^
  --bull-execution-mode inherit ^
  --micro-filter-mode soft ^
  --micro-not-aligned-risk-scale 0.5 ^
  --micro-contra-risk-scale 0.5 ^
  --global-risk-scale 1.3 ^
  --priority-mode reclaim_first ^
  --same-side-continuation-mode off ^
  --range-pct 0.002 ^
  --price-step 1 ^
  --out-dir data/reports/lf/eth_lf_portfolio_v9d_same_side_continuation/reclaim_first_off_gs_1p3
```

## 重点看

summary 中新增：

```text
same_side_continuation_mode
same_side_exit_suppressed_count
```

如果 suppress count 很低，但收益变化很大，需要检查是否是少数大单造成的路径依赖。
