# ETH 1D+4H Pullback Continuation V1

新增文件：

```text
backtest/lf/eth_1d_4h_pullback_continuation_v1_backtest.py
```

## 定位

第三个独立引擎的第一版。目标不是替代 Portfolio V4B，而是尝试做一个：

```text
更高胜率
更短持仓
顺趋势回踩延续
未来接入组合后改善持仓体验
```

## 逻辑

多头：

```text
1D bull regime
4H EMA20 > EMA50
价格最近回踩 EMA20 / EMA50
4H 收盘重新站回 EMA20
RSI 从中性偏弱区间回升
下一根 4H open 做多
```

空头反向，但空头更严格。

## 反未来函数约束

```text
1. 1D regime shift(1) 后映射到 4H。
2. 当前 4H 收盘确认信号。
3. 下一根 4H open 执行。
4. 执行复用 V8 SAFE 引擎，当前 close 更新的新 stop 下一根 bar 才生效。
5. 不允许同一根 K 线内新 stop 立刻触发。
```

## 默认手续费

```text
fee_rate = 0.00055
```

完整开平约 0.11%。

## 运行

```bash
python backtest/lf/eth_1d_4h_pullback_continuation_v1_backtest.py \
  --start-date 2023-01-01 \
  --end-date 2026-06-15 \
  --preset high
```

只做多：

```bash
python backtest/lf/eth_1d_4h_pullback_continuation_v1_backtest.py \
  --start-date 2023-01-01 \
  --end-date 2026-06-15 \
  --preset high \
  --disable-short \
  --out-dir data/reports/lf/eth_1d_4h_pullback_continuation_v1_longonly
```

## 回测结果

### high 默认多空

```text
总收益      17.03%
最大回撤    21.83%
PF          1.27
胜率        30.71%
交易次数    127
平均持仓    48.54 小时

2023   -6.80%
2024   -10.43%
2025   +19.64%
2026   +17.17%
```

### high long-only

```text
总收益      17.99%
最大回撤    12.96%
PF          1.53
胜率        33.85%

2023   +0.34%
2024   -5.71%
2025   +23.70%
2026   +0.83%
```

## 结论

V1 还不能接入组合。

它确实比 V4B 胜率高，但问题很明显：

```text
收益太低
2024 表现差
PF 不够高
默认多空回撤不低
```

所以这个版本只能作为研究起点，不能作为合格补充引擎。

下一步应该做 V2：

```text
1. 先 long-only
2. 专门过滤 2024 的弱回踩结构，但不能按年份调参
3. 加强 1D bull 质量过滤
4. 减少 EMA20 假回踩，只保留 EMA50 / 深回踩 reclaim
5. 尝试 1H / MF 执行，但必须单独验证
```
