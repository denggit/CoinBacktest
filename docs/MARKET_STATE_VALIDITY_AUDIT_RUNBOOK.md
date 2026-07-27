# Market State Validity Audit V1

## 目的

验证 Market State Map 中的静态状态和状态转移是否真的具有前向路径差异，而不是继续凭肉眼调整绿色、红色或阈值。

本研究不是可实盘策略，不会修改现有策略、回测或 `analyze_tool`。

## 当前数据是否足够

第一轮足够。默认仅使用现有本地 `1m Trade Bar`：

- OHLCV、成交笔数、成交额；
- 主动买入/卖出金额与 delta；
- CVD 来源字段；
- 大单买卖金额；
- 当前 Market State 引擎生成的趋势、波动、订单流、冲击/吸收与结构位置。

暂时不需要 Order Book、OI、Funding、Basis 或真实爆仓数据。

只有以下情况才需要先补数据：

1. `orderflow_coverage_ratio` 明显低于 0.95；
2. 本地 Trade Bar 缺少 `buy_notional / sell_notional / delta_notional`；
3. 审计显示现有状态无区分度，但需要进一步判断“新仓建立、平仓或挤仓”时，再优先增加 OI、Funding、Basis；
4. 需要研究盘口流动性真空、撤单和恢复速度时，再增加标准化历史 Order Book。

## 默认研究范围

- Warmup：2022-01-01
- 正式审计：2023-01-01 至 2026-06-30
- Holdout：2025-07-01 至 2026-06-30
- 前向周期：5 / 15 / 30 / 60 / 180 bars
- 完整买卖成本：0.11%
- 参数邻域：fast / base / slow 三套窗口

## 因果时序

```text
左标签 1m Trade Bar 收盘
-> available_time 生效
-> 下一根 bar open 作为诊断入场价
-> 未来 high/low/close 只作为结果标签
```

禁止同一根 bar 的 open 使用其最终 high、low、close、volume 或订单流结果。

## 运行命令

```bash
python research/market_state/01_market_state_validity_audit.py --symbol ETH-USDT-SWAP --timeframe 1m --warmup-start-date 2022-01-01 --start-date 2023-01-01 --end-date 2026-06-30 --holdout-start 2025-07-01 --local-only
```

如果本地 Trade Bar 有缺失，并且允许脚本补建缺失日期：

```bash
python research/market_state/01_market_state_validity_audit.py --symbol ETH-USDT-SWAP --timeframe 1m --warmup-start-date 2022-01-01 --start-date 2023-01-01 --end-date 2026-06-30 --holdout-start 2025-07-01 --no-local-only
```

## 主要输出

默认目录：

```text
data/reports/research/market_state/01_market_state_validity_audit
```

重点文件：

- `00_EXECUTIVE_SUMMARY.md`：结论摘要；
- `02_event_path_summary.csv`：每类状态/转移在各前向周期的收益、MFE、MAE、成本后表现和 matched-baseline uplift；
- `03_yearly_breakdown.csv`：年度稳定性；
- `04_pre_holdout_vs_holdout.csv`：样本内与 holdout；
- `05_profile_stability_and_verdict.csv`：快/基准/慢参数邻域与最终 robust flag；
- `06_causal_audit.csv`：available time 和 next-open 时序检查；
- `07_top_bottom_trap_examples.csv`：状态启动后立即位于未来顶部/底部附近的失败样本；
- `08_verdict.json`：机器可读结论；
- `diagnostic_reports/`：趋势启动和交易观察状态的非重叠固定周期成本诊断报告；
- `gpt_review_pack.zip`：完成后发给 GPT 复核。

## 判断规则

静态趋势绿色/红色只有在以下条件同时满足时，才保留方向许可价值：

- 成本后平均收益为正；
- 优于同年份、同波动环境基线；
- 年度多数为正；
- Holdout 仍为正；
- fast/base/slow 参数邻域至少 2/3 同方向成立；
- 样本量足够。

如果静态趋势失败，但吸收、扫单收回、突破接受或状态转移通过，则停止依赖绿色/红色入场，转向状态转移的顺序回测。

如果状态转移也没有稳定区分度，应停止继续堆特征，再决定是否引入 OI 或 Order Book。
