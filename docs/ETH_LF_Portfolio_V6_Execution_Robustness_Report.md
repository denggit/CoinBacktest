# ETH LF Portfolio V6 Execution Robustness Report

## 结论

V6 inherit turbo 的主要脆弱点不是入场信号本身，而是执行时效。

我测试了两类策略层面的鲁棒性修复：

1. 下一根 open 价格跳得太远时跳过入场；
2. 如果信号延迟 1 根 4H，再用“原信号 close 到迟到入场价格”的 adverse move 做过滤。

结果都没有改善。说明 V6 的延迟敏感性不是由 next-open gap 造成，而是 4H 级别信号衰减很快：错过第一根可执行 open 后，入场位置和行情状态已经变了。

## Baseline

```text
V6 inherit turbo:
总收益      2617.57%
最大回撤    27.28%
PF          3.95
胜率        23.73%
交易次数    118
2023       +11.72%
2024      +224.77%
2025      +168.35%
2026      +179.10%
```

## 普通入场 gap guard 测试

逻辑：如果信号确认后，下一根 open 已经朝不利方向跳得太远，则跳过交易。

测试阈值：0.25%、0.50%、0.75%、1.00%、1.50%，以及 ATR 过滤 0.2/0.35/0.5/0.75 ATR。

结果：对 baseline 基本没有影响。

原因：历史中 V6 的主要交易并不是因为 next-open gap 太大而受损。普通 gap guard 不是关键修复点。

## 延迟 1 根 4H + late-entry guard 测试

逻辑：如果信号迟到 1 根 4H，只有当迟到入场价相对原信号 close 没有明显不利移动时才允许进场。

结果：比单纯延迟更差或没有改善。

```text
1 bar 延迟，无过滤:
总收益      1240.76%
最大回撤    37.84%
PF          3.23
2023       -23.26%

1 bar 延迟 + 0.50% late-entry guard:
总收益       972.30%
最大回撤     43.63%
PF           3.25
2023        -29.96%

1 bar 延迟 + 0.75% late-entry guard:
总收益       992.73%
最大回撤     36.25%
PF           3.06
2023        -20.80%
```

结论：迟到后再挑价格，不能修复 alpha 衰减。错过第一根可执行 open 后，很多交易本来就不应该再做。

## 实盘建议

### 1. 不要做“迟到补单”

4H 信号如果没有在下一根 4H open 附近执行，就不要追。

推荐规则：

```text
当前 4H close 确认信号
只允许在下一根 4H open 后的短窗口内执行
超过窗口未成交，取消信号
不等下一根 4H 再补单
```

### 2. 实盘应该做 execution daemon

建议实盘执行层独立出来：

```text
4H close 前后监听 websocket kline close
REST 拉取最后一根 4H K 线做确认
在 close 后 5-30 秒内生成信号
下一根 open 附近执行
订单失败自动重试，但最多重试 1-2 次
超过 1-3 分钟仍未成交，取消该信号
```

### 3. 用限价保护，不要无限追价

建议：

```text
long 入场价不得高于触发时参考价 + 0.05%~0.10%
short 入场价不得低于触发时参考价 - 0.05%~0.10%
超出则取消，不追单
```

这个规则不一定提升回测收益，但能防实盘异常滑点。

### 4. 监控实际成交质量

每笔实盘记录：

```text
signal_time
expected_next_open
actual_order_time
actual_fill_price
fill_slippage_pct
decision_delay_seconds
```

后续用真实成交数据反推是否要扩大/缩小执行窗口。

## 下一步

不建议现在做 V7 信号版本。

更值得做的是新增一个实盘执行模拟层：

```text
--entry-window-minutes
--max-adverse-entry-pct
--stale-signal-policy skip
--entry-delay-seconds
```

这不是找新 alpha，而是把 V6 从“研究回测”推进到“可实盘执行的策略”。
