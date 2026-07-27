# ETH 做多马丁格尔限价单回测

入口：

```text
backtest/mf/eth_martingale_limit_long_backtest.py
```

## 已固化的三个方案

| variant | 方案 | 首单:首次加仓 | 每档跌幅 | 后续加仓金额倍数 | 止盈 | 最大加仓次数 | 杠杆 |
|---|---|---:|---:|---:|---:|---:|---:|
| `midterm` | 中期趋势加仓 | 1:0.94 | 1.00% | 1.05 | 4.10% | 8 | 10x |
| `aggressive` | 短期进取型投资 | 1:0.54 | 0.53% | 1.10 | 4.10% | 12 | 13x |
| `longterm` | 长期稳健投资 | 1:1.11 | 1.37% | 1.05 | 5.00% | 7 | 9x |

`最大加仓次数`不包含首单。加仓价差倍数均为 1.00。

## 执行定义

- 每个新周期以首个可见市场价格或上一个周期退出价为锚点。
- 首单不是市价追入，而是在锚点下方一个档位预挂买入限价单。
- 后续每跌一个档位继续挂买入限价单。
- `1:0.94` 等比例解释为“首单名义金额 : 第一次加仓名义金额”。
- 第二次及后续加仓金额在第一次加仓基础上按金额倍数递增。
- 完整阶梯的总名义金额 = 周期开始资金 × 杠杆 × `capital_utilization`。
- 止盈价 = 当前加权平均持仓价 × `(1 + 止盈目标)`，每次加仓后重算。
- 默认每次成交单边手续费为 0.055%，基础一买一卖合计 0.11%。
- 默认回测区间为 `2023-01-01` 到 `2026-06-30`。

## 数据源与时序

支持：

- `trade_bar`：项目 `OKXTradeBarLoader`。
- `range_bar`：项目 `OKXRangeBarLoader`。
- `raw_trade`：项目 `OKXTickLoader`，按成交时间顺序流式回放。

raw trade 使用真实逐笔先后顺序。trade bar / range bar 无法知道 bar 内 high、low 的先后，因此采用保守路径：先向下处理挂单与爆仓；只要本 bar 发生任何加仓，该 bar 不允许同时止盈。这样不会利用同一根 bar 的未知路径制造乐观收益。

爆仓价格是包含入场手续费、预计退出手续费和维护保证金率的近似全仓模型，不等同于 OKX 标记价格、风险档位和保险基金的完整实盘清算引擎。

## 运行命令

三个方案一起跑 trade bar：

```text
python backtest/mf/eth_martingale_limit_long_backtest.py --data-source trade_bar --trade-bar-timeframe 1m
```

三个方案一起跑 0.20% range bar：

```text
python backtest/mf/eth_martingale_limit_long_backtest.py --data-source range_bar --range-pct 0.002
```

三个方案一起跑 raw trade：

```text
python backtest/mf/eth_martingale_limit_long_backtest.py --data-source raw_trade --chunksize 300000
```

只跑短期进取方案并且只读本地缓存：

```text
python backtest/mf/eth_martingale_limit_long_backtest.py --data-source raw_trade --variant aggressive --cache-only
```

## 输出

默认写到：

```text
data/reports/backtest/mf/eth_martingale_limit_long/<data_source>/
```

核心文件：

- `00_variant_comparison.csv`：三个方案对比。
- `01_trades.csv`：已结束周期。
- `02_order_fills.csv`：每一档限价成交。
- `03_equity_daily.csv`：含持仓浮盈亏的日度权益。
- `04_summary.json`：真实 MTM 最大回撤、爆仓次数、费用等核心结果。
- `05_config.json`：完整参数和执行假设。
- `06_open_position.json`：期末持仓审计。

项目公共 `print_full_report` 仍会生成报告，但它的最大回撤是“已平仓周期资金曲线回撤”。马丁策略最重要的是持仓期间浮亏，因此风险判断应以 `04_summary.json` 的 `max_mtm_drawdown_pct` 和 `03_equity_daily.csv` 为准。

## 风险边界

这是参数复现回测，不是已验证的正期望策略。马丁格尔没有止损，且这里使用 9x～13x 杠杆，极端单边下跌时可能在完整阶梯成交前后迅速爆仓。正式评价必须至少比较 raw trade、trade bar、range bar，并做手续费、维护保证金、区间起点和年度稳定性压力测试。
