# ETH Range Anchored VP Absorption Breakout V1

## 文件

```text
backtest/mf/eth_range_anchored_vp_absorption_v1_backtest.py
```

## 依赖数据

请先用本地工具聚合数据，例如：

```bash
python tools/prebuild_okx_range_all.py \
  --symbol ETH-USDT-SWAP \
  --start-date 2022-01-01 \
  --end-date 2026-06-15 \
  --range-pcts 0.0015 0.002 0.0025 \
  --price-step 1 \
  --chunksize 1000000 \
  --flush-rows 1000000
```

策略默认读取：

```text
data/okx_range_bars.db
数据表: ETH_USDT_SWAP_range_bars_r0020

data/okx_range_footprints.db
数据表: ETH_USDT_SWAP_range_footprint_r0020_step1
```

## 推荐先跑 2023 单年

```bash
python backtest/mf/eth_range_anchored_vp_absorption_v1_backtest.py \
  --start-date 2023-01-01 \
  --end-date 2023-12-31 \
  --warmup-start-date 2022-01-01 \
  --preset high \
  --range-pct 0.002 \
  --price-step 1 \
  --out-dir data/reports/mf/range_avp_abs_v1_2023_high
```

## 再跑全样本

```bash
python backtest/mf/eth_range_anchored_vp_absorption_v1_backtest.py \
  --start-date 2023-01-01 \
  --end-date 2026-06-15 \
  --warmup-start-date 2022-01-01 \
  --preset high \
  --range-pct 0.002 \
  --price-step 1 \
  --out-dir data/reports/mf/range_avp_abs_v1_full_high
```

## 需要发回来的文件

请把输出目录里的这些文件发回来：

```text
eth_range_anchored_vp_absorption_v1_summary.json
eth_range_anchored_vp_absorption_v1_trades.csv
eth_range_anchored_vp_absorption_v1_equity.csv
eth_range_anchored_vp_absorption_v1_signal_audit.csv
eth_range_anchored_vp_absorption_v1_setup_events.csv
ETH_Range_AnchoredVP_AbsorptionBreakout_V1_*.txt
```

如果文件太大，优先发：

```text
summary.json
setup_events.csv
signal_audit.csv
trades.csv
完整报告 txt
```

## 当前版本定位

这是 long-only V1，只做：

```text
下跌结构 -> anchored profile -> VAL/lower POC 附近反复 sell bubble absorption -> buy stop 突破入场 -> lower POC 下方止损 -> VAH 止盈
```

它不是优化版，目标是先验证 setup 是否像 Fabio 图形，以及无手续费/正常手续费下是否有基础正期望。
