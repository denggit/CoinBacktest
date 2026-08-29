# ETH Turtle Baseline V1

这是“第一个个人量化 baseline”，目标不是找神参数，而是先验证一个公开、经典、机械化的趋势跟踪体系在 ETH 永续上能不能真实活下来。

## 固定策略族

- `fast`: 10 日突破 / 5 日退出（币圈快速适配版）
- `s1`: 20 日突破 / 10 日退出（Turtle System 1 结构）
- `s2`: 55 日突破 / 20 日退出（Turtle System 2 结构）

三套规则在运行前就固定，不做参数网格，不从结果里挑隐藏最优参数。

## V1 规则

- 数据：`src.data_feed.okx_loader.OKXDataLoader`，ETH-USDT-SWAP 1H。
- 通道：只由已经完成的 UTC 日线构造，并 `shift(1)`，禁止偷看当天未完成信息。
- 入场：1H 内触发前序日线 Donchian 突破价即视为挂单成交；跳空时按更差的 1H open 成交。
- N：20 日 daily True Range 的 Wilder-style EMA，整体再滞后 1 日。
- 初始止损：2N。
- 出场：反向 Donchian exit channel 或 2N protective stop。
- 仓位：默认每笔风险 1% equity；名义仓位上限 3x equity。
- 成本：默认每边手续费 0.055%，每次成交滑点 0.02%。
- V1 不加仓、不做 EMA/成交量/时间过滤、不做机器学习。
- Funding 不伪造；V1 明确不计 funding，后续必须接真实 funding 数据再做最终实盘确认。
- MDD 使用逐 1H mark-to-market equity，不只看平仓资金曲线。

## 运行

Windows 一行：

`python backtest\lf\eth_turtle_baseline_v1_backtest.py --start-date 2022-01-01 --end-date 2026-06-30 --variant all --sides both`

如果只跑多头：

`python backtest\lf\eth_turtle_baseline_v1_backtest.py --start-date 2022-01-01 --end-date 2026-06-30 --variant all --sides long`

## 输出

默认输出到：

`data/reports/research/trend/eth_turtle_baseline_v1/`

每个 variant 会输出：

- `summary.csv`
- `trades.csv`
- `equity.csv`
- `yearly.csv`
- `monthly.csv`
- `cost_stress.csv`
- `run_config.json`
- `report.md`

根目录还会输出 `comparison_<sides>.csv`。

## 判定方式

不要只看总收益。优先检查：

1. 最大无入场天数；
2. 最大连续亏损日 / 连续亏损交易；
3. mark-to-market MDD；
4. 不同年份是否都能活；
5. 2x 成本下是否仍盈利；
6. CAGR / 总收益；
7. 交易数是否足够。

如果三套都不行，直接判定“经典 Turtle baseline 不适合当前 ETH 单品种约束”，不继续通过参数网格救活。
