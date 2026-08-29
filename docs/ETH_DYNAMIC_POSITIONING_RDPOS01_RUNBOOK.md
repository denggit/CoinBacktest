# ETH Dynamic Positioning RDPOS-01 Runbook

## 目的

RDPOS-01 不把研究对象定义成一笔一笔的交易，而是定义成两个持续存在、独立记账的仓位 sleeve：

- `medium`：1d / 3d / 7d 趋势状态；
- `slow`：7d / 14d / 30d 趋势状态。

每个 sleeve 的目标仓位由三层信息决定：

1. **Trend**：多周期标准化趋势强度，连续值 `[-1, +1]`；
2. **Location**：当前价格相对对应趋势锚点的标准化延伸程度，只调整仓位大小，不独立翻转方向；
3. **Volatility**：按过去 7 天实现波动率做风险缩放。

最终研究的是 `target exposure -> current exposure -> 是否值得调整`，不是 `setup -> entry -> TP/SL`。

## 为什么重新做

仓库已有 `research/eth_market_process_portfolio/portfolio/clean_causal.py`，其 frozen 7/30/90 日趋势 + volatility scaling 历史基线已经证明：**单纯 trend + vol 不足以达到实盘资本效率要求**。

RDPOS-01 的增量假设仅有：

- 用当前 price location 控制趋势仓位大小；
- medium/slow sleeve 独立保留，允许一多一空；
- 4H 固定决策时钟；
- no-trade band；
- 单次最大调仓步长；
- 手续费按两个 sleeve 的 gross turnover 计算，而不是提前净额化。

如果 `base_location` 不能在经济指标上优于 `trend_only_no_location`，直接判定 Location 假设失败，不继续微调。

## 因果时序

1H K 线 timestamp 视为 bar start。

```text
03:00 bar 覆盖 03:00 -> 03:59:59
04:00 才可看到 03:00 bar 最终 close/high/low
04:00 open 才允许执行基于 03:00 closed bar 的新目标
```

实现字段：

```text
available_time = bar_timestamp + 1h
```

只有 `available_time.hour % 4 == 0` 的状态允许成为 4H 决策。

额外执行延迟压力是在 mandatory next-open 之后继续推迟，不会把执行提前到信号 bar 内。

2022-01-01 起只用于 warmup；正式账户收益从 2023-01-01 开始，2022 不允许进入正式收益曲线。

## 交易成本

默认：

```text
fee per side = 0.055%
round-trip fee reference = 0.11%
slippage per side = 0.01%
```

调仓手续费按：

```text
abs(delta_medium) + abs(delta_slow)
```

计算。

例如：

```text
medium +0.5x
slow   -0.5x
net     0.0x
gross   1.0x
```

不能因为 net=0 就把手续费算成 0。

## Funding

`--funding-source auto`：

1. 优先使用本地 `OKXDerivativesLoader` 的完整历史 funding；
2. 若不足，尝试仓库现有 Binance ETHUSDT funding archive 作为明确标记的 proxy；
3. 两者都不完整时，不伪造 funding，base 会标记 `FUNDING_UNAVAILABLE`；
4. 同时输出 5% / 10% 年化 gross carry drag 压力场景。

Funding 不完整时，即使 CAGR 很高，也不能标记为 live-ready。

## 运行

Windows 一行：

```text
python research\eth_dynamic_positioning\01_trend_location_vol_positioning.py
```

若本地 1H K 线未预构建，先使用项目既有 prebuild；只有明确需要联网补数据时才加：

```text
python research\eth_dynamic_positioning\01_trend_location_vol_positioning.py --allow-fetch
```

## 输出

```text
data/reports/research/eth_dynamic_positioning/01_trend_location_vol_positioning/
```

主要文件：

- `summary.json`
- `scenario_summary.csv`
- `yearly.csv`
- `monthly.csv`
- `equity_hourly.csv`
- `decision_audit.csv`
- `sleeve_episodes.csv`
- `funding_coverage.json`
- `verdict.json`
- `REPORT.md`

## Frozen scenarios

不允许从场景中挑冠军：

- `base_location`
- `trend_only_no_location`
- `no_trade_band_off`
- `cost_2x`
- `cost_3x`
- `delay_plus_4h`
- `carry_stress_5pct`
- `carry_stress_10pct`

这些场景只用于回答机制是否成立。

## 晋级条件

RDPOS-01 只有同时满足以下条件才允许进入下一阶段：

1. `CAGR > abs(MDD)`；
2. `Calmar >= 1`；
3. 至少 3 个自然年度为正；
4. 2x 成本后总收益仍为正；
5. `base_location` 的 Calmar 高于 `trend_only_no_location`；
6. 交易成本不吞掉大部分正向 gross contribution；
7. funding 历史覆盖完整。

否则停止或重新定义假设，禁止围绕某个年份/某段亏损微调阈值。
