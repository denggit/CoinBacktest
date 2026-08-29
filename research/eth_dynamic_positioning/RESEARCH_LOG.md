# ETH Dynamic Positioning — Research Log

## RDPOS-01 — Trend + Location + Volatility Position State

### 背景

前序研究暴露两个问题：

1. 一笔一笔的 setup/entry/TP/SL 策略反复缺乏跨期经济优势；
2. 连续收益预测曾出现“Rank IC 正但经济 spread 不足以覆盖成本”的问题。

同时，仓库现有 clean causal portfolio 已经测试过 price-only 7/30/90d trend + volatility scaling，结果不足以实盘。因此 RDPOS-01 不重复“趋势 + vol”本身，而是只测试 **current location 是否能改善仓位管理**。

### 冻结设计

- 1H 状态数据；
- 4H 决策时钟；
- medium sleeve: 24/72/168h；
- slow sleeve: 168/336/720h；
- trend 为连续 score；
- location 只调整仓位强弱，不独立翻多翻空；
- volatility scaling；
- no-trade band = 0.20x；
- max step = 0.50x / decision；
- gross cap = 2.0x；net cap = 1.5x；
- 2022 warmup，2023-01-01 开始正式收益；
- 每侧手续费 0.055%，另计 slippage；
- funding 优先真实 OKX，允许明确标记 Binance proxy，不允许把缺失 funding 当成已验证实盘成本。

### 工程审计

- future mutation causality test；
- current-bar signal cannot execute before next hourly open；
- 2022 warmup isolation；
- positive funding: long pays / short receives；
- medium/slow opposite-side gross turnover accounting；
- no-trade band test；
- 4H decision cadence test：禁止大 gap 在决策间隔内演化成逐小时连续追单。

### 当前状态

代码交付，等待用户本地真实 ETH 数据结果。禁止在看到结果前改参数。
