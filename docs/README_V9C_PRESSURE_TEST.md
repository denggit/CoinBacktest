# V9C Reclaim Priority Pressure Test

用于测试 V9C `reclaim_first + global_risk_scale=1.3` 是否真的能取代 V8。

包含：

- V9C strategy 文件：`backtest/lf/eth_lf_portfolio_v9c_reclaim_priority_backtest.py`
- 压测工具：`research/v9c_reclaim_priority_pressure_test.py`

## 一键压测

```bat
python research/v9c_reclaim_priority_pressure_test.py ^
  --start-date 2023-01-01 ^
  --end-date 2026-06-15 ^
  --warmup-start-date 2022-01-01 ^
  --priority-mode reclaim_first ^
  --global-risk-scale 1.3 ^
  --out-dir data/reports/research/v9c_reclaim_priority_pressure_test/reclaim_first_gs_1p3
```

## 会跑哪些 real backtest

```text
base
fee_2x
slippage_2x
no_2026
```

## 会做哪些 post-process removal stress

基于 base trades：

```text
remove_top1_pnl
remove_top3_pnl
remove_2025_07_07_long
```

## 输出总表

```text
data/reports/research/v9c_reclaim_priority_pressure_test/reclaim_first_gs_1p3/v9c_reclaim_priority_pressure_summary.csv
```

## 判断标准

V8 confirmed champion：

```text
V8 gs_1.3:
return +4726.53%
maxDD -27.18%
PF 4.49
```

V9C base：

```text
V9C reclaim_first gs_1.3:
return +6138.14%
maxDD -27.33%
PF 4.72
```

重点不是 base 是否更高，而是：

```text
remove_2025_07_07_long 后是否仍然优于 V8
remove_top3 后是否仍然健康
fee/slippage/no2026 是否还能接受
```
