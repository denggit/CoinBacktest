# ETH Trend Pullback V1

## 目的

做一个适合 ETH 永续的中短周期趋势回调策略 baseline：目标持仓从数小时到 2–3 天，避免 Turtle 那种十几天到数周的长期资金占用。

本版本不是参数优化结果，也不是从亏损样本里救出来的版本。V1 规则先冻结，跑完以后按统一验收门槛决定保留/淘汰。

## 完整规则

1. **4H 趋势方向**
   - Long: EMA50 > EMA200、4H close > EMA50、EMA50 的过去 24h slope > 0。
   - Short 对称。
2. **1H 健康回调 + reclaim**
   - Long: EMA20 > EMA50；前一根 1H close 回到/跌破 EMA20；当前已完成 1H close 重新站上 EMA20；最近 3h 低点没有明显跌穿 EMA50。
   - Short 对称。
   - reclaim 后 setup 最多保持 3 小时。
3. **15m 重新加速入场**
   - Long: 在有效 setup 内，15m close 突破前 4 根已完成 15m close 的最高值，close > EMA20 且阳线；距离 1H EMA20 不超过 1.5×1H ATR。
   - Short 对称。
   - 15m closed-bar 生成信号，下一根 15m open 入场。
4. **止损**
   - Long: 前 3h causal low - 0.25×15m ATR；Short 对称。
   - 最小止损 0.4%，最大止损 2.5%；过宽直接跳过。
5. **退出**
   - 没有固定 TP。
   - MFE >= 0.8R 后，下一根 bar 起保护到覆盖 round-trip fee+slippage 的 breakeven。
   - MFE >= 1.25R 后，用已完成 1H EMA20 ± 0.5×1H ATR 做结构 trailing，下一根 15m bar 起生效。
   - 4H regime 消失：下一根 15m open 出场。
   - 1H close 反穿 EMA20：下一根 15m open 出场。
   - 12h 仍 MFE < 0.4R：下一根 open 离场。
   - 最大持仓 72h。
6. **风险与交易成本**
   - 默认单笔风险 1% equity。
   - 最大 notional 3× equity。
   - fee 0.055%/side，即 round-trip 0.11%。
   - slippage 0.02%/side。
   - funding 优先本地 OKX；覆盖不足则可明确使用 Binance ETHUSDT archive proxy。

## 因果对齐

15m bar 的 timestamp 视为 bar start。该 bar 最早在 `timestamp + 15m` 可用。

1H/4H context 都显式生成：

```text
available_time = bar_start_time + timeframe
```

随后只按 `available_time <= signal_available_time` 做 `merge_asof(direction="backward")`。任何 signal row 不满足这一约束都会 fail closed，不允许交易。

## 正式回测窗口

```text
warmup:   2022-01-01
backtest: 2023-01-01 -> 2026-06-30
```

2022 只参与 EMA/ATR warmup，永远不进入交易和正式绩效，专门避免上一版 Turtle 的 warmup 误交易问题。

## Windows 一行命令

```text
python backtest\mf\eth_trend_pullback_v1_backtest.py --no-build-missing
```

默认会跑：

- BASE
- long / short 诊断
- 1x / 2x / 3x 交易成本压力
- funding 1.5x 压力
- 4 个预先固定的参数邻域 robustness（只诊断，不挑最好参数）

## 输出

```text
data/reports/research/trend/eth_trend_pullback_v1/
```

核心文件：

- `summary.json`
- `report.md`
- `trades.csv`
- `equity.csv`
- `signal_audit.csv`
- `funding_ledger.csv`
- `side_breakdown.csv`
- `cost_stress.csv`
- `robustness.csv`
- 项目原生 `print_full_report` 报告

## V1 验收原则

### 必须满足才允许继续实盘化

1. 正式 2023–2026 仍为正收益；
2. **CAGR > MDD**，否则资本效率不够；
3. 目标优先 `Calmar >= 1.2`，更希望 >= 1.5；
4. 2x 成本仍盈利，3x 成本不能直接崩溃；
5. funding 后仍盈利；
6. 交易数不能只剩几十笔到无法验证；
7. 参数邻域不能只有 BASE 一个点赚钱；
8. 收益不能只由单一年份或少数 Top trades 支撑；
9. 平均持仓应主要落在数小时到 72h，若明显重新漂到一周以上则偏离本策略目标；
10. 因果审计必须 0 failure。

若不通过，不继续 V2/V3 无限救策略，直接归档并换下一个完整策略。

## Portfolio 说明

本 backtest 是单 sleeve，因此自身只有一个活动仓位。未来 Portfolio 层可以同时持有：

- 长周期 Turtle sleeve；
- 中短周期 Trend Pullback sleeve；
- 其他 long/short sleeves。

这些 sleeve 可以逻辑上同时一多一空，最终由 Portfolio/position management 决定净额、gross exposure、相关性折扣和交易所实际持仓表达。本 V1 不把 Portfolio 仓位管理塞进单策略回测。
