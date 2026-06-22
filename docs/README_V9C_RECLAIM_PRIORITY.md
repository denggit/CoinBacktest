# ETH LF Portfolio V9C Reclaim Priority Probe

独立测试版，不修改 V8/V6/V7B 原文件。

目的：在 V8 MicroConfirmScaled 的基础上，只测试引擎冲突优先级是否影响组合表现。默认把 `BULL_RECLAIM_V2` 放到第一优先级。

## 默认测试：Reclaim 第一优先级

```bat
python backtest/lf/eth_lf_portfolio_v9c_reclaim_priority_backtest.py ^
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
  --range-pct 0.002 ^
  --price-step 1 ^
  --out-dir data/reports/lf/eth_lf_portfolio_v9c_reclaim_priority/reclaim_first_gs_1p3
```

## 对照：V8 原优先级

```bat
python backtest/lf/eth_lf_portfolio_v9c_reclaim_priority_backtest.py ^
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
  --priority-mode v8 ^
  --range-pct 0.002 ^
  --price-step 1 ^
  --out-dir data/reports/lf/eth_lf_portfolio_v9c_reclaim_priority/v8_priority_gs_1p3
```

## 备用测试：Reclaim 第一，Bear 第二

```bat
python backtest/lf/eth_lf_portfolio_v9c_reclaim_priority_backtest.py ^
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
  --priority-mode reclaim_bear_second ^
  --range-pct 0.002 ^
  --price-step 1 ^
  --out-dir data/reports/lf/eth_lf_portfolio_v9c_reclaim_priority/reclaim_bear_second_gs_1p3
```

## 判断标准

重点对比 V8 champion：

```text
V8 soft_05 gs_1.3:
return +4726.53%
maxDD -27.18%
PF 4.49
```

如果 Reclaim 第一优先级没有明显超过这个基准，直接放弃，不要继续调参数。

## 时序说明

- 当前 4H close 确认信号，下一根 4H open 执行。
- Momentum/Bear/Bull 三个引擎仍然只使用已确认 4H/1D 特征。
- V9C 只改同一 4H bar 多引擎同时发信号时的路由优先级。
- micro confirmation 和 global risk scale 沿用 V8 逻辑。
