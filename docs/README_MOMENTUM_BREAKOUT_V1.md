# ETH 1D+4H Momentum Breakout V1

## 定位

这是一个新的独立候选引擎：

```text
backtest/lf/eth_1d_4h_momentum_breakout_v1_backtest.py
```

它不是 Pullback，也不是原本的 V4B 趋势延续，而是：

```text
1D 判断趋势环境
4H 只做强动量突破
成交量确认
下一根 4H open 执行
```

这个引擎有成为主引擎候选的潜力，但 V1 还不能直接替代 V4B，因为 2023 年是亏损的。

## 反未来函数约束

- 1D regime 全部 `shift(1)` 后映射到 4H。
- 4H `entry_high` / `entry_low` / `exit_high` / `exit_low` 全部 `rolling(...).shift(1)`。
- 当前 4H 收盘确认信号，下一根 4H open 入场。
- 执行复用 V8 SAFE 引擎：当前 bar close 更新的新 stop 下一根 bar 才生效。
- 默认单边手续费 `fee_rate=0.00055`，完整开平约 0.11%。

## 运行

```bash
python backtest/lf/eth_1d_4h_momentum_breakout_v1_backtest.py --start-date 2023-01-01 --end-date 2026-06-15 --preset high
```

保守版：

```bash
python backtest/lf/eth_1d_4h_momentum_breakout_v1_backtest.py --start-date 2023-01-01 --end-date 2026-06-15 --preset stable --out-dir data/reports/lf/eth_1d_4h_momentum_breakout_v1_stable
```

激进版：

```bash
python backtest/lf/eth_1d_4h_momentum_breakout_v1_backtest.py --start-date 2023-01-01 --end-date 2026-06-15 --preset turbo --out-dir data/reports/lf/eth_1d_4h_momentum_breakout_v1_turbo
```

## 本地回测结果

区间：2023-01-01 到 2026-06-15。

### stable

```text
总收益      331.91%
最大回撤    12.70%
PF          3.31
胜率        24.69%
交易次数    81

2023   -6.19%
2024  +93.84%
2025  +93.32%
2026  +22.85%
```

### high

```text
总收益      755.83%
最大回撤    20.37%
PF          3.09
胜率        24.69%
交易次数    81

2023   -11.10%
2024  +159.70%
2025  +171.98%
2026   +36.30%
```

### turbo

```text
总收益      1195.66%
最大回撤    25.73%
PF          2.93
胜率        24.69%
交易次数    81

2023   -15.06%
2024  +209.30%
2025  +239.45%
2026   +45.29%
```

### ultra

```text
总收益      1737.15%
最大回撤    30.77%
PF          2.79
胜率        24.69%
交易次数    81

2023   -19.18%
2024  +258.21%
2025  +313.97%
2026   +53.29%
```

## 结论

V1 的收益弹性非常强，甚至有主引擎候选潜力；但 2023 年亏损明显，说明它还没有通过年度稳定性要求。

下一步建议做 V2：

```text
1. 保留 high/turbo 的收益弹性
2. 专门降低 2023 假突破亏损
3. 不使用年份/月分过滤
4. 用市场状态分桶：bull/bear quality、Bear confirmed short、低质量 quick breakout 降仓
5. 再判断是否能替代 V4B 或进入 Portfolio
```
