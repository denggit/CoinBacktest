# ETH LF Portfolio V6 Momentum + Bear + Bull Reclaim

## 定位

V6 在 V5 基础上加入 Bull Range Reclaim V2 作为低优先级 long 补充。

优先级：

```text
1. Momentum V3
2. Bear V3 only short
3. Bull Range Reclaim V2 long
```

规则：

```text
Momentum V3 有信号：最高优先级。
Momentum 静默且 Bear V3 short 出现：Bear V3 only short。
Momentum/Bear 都静默且 Bull V2 long 出现：Bull Reclaim long。
不双开、不对冲、不简单叠加收益。
```

## 反未来函数约束

- Momentum/Bear/Bull 的 1D/1W 高周期数据都先 `shift(1)` 后映射到 4H。
- 当前 4H close 确认，下一根 4H open 执行。
- 当前 close 更新的新 stop 下一根 bar 才生效。
- 支持 `--warmup-start-date` / `--warmup-days`，warmup 只用于指标，不允许开仓、不计入报告。
- 不使用年份、月份、日期过滤。

## Bull 执行模式

V6 支持两种 Bull 执行模式：

```text
--bull-execution-mode inherit
    Bull 只提供入场信号，执行/保护参数继承主引擎 Momentum。
    收益更高，但 Bull 的单独高胜率特征会被弱化。

--bull-execution-mode own
    Bull 使用自己的短持仓保护参数。
    胜率更高，回撤更低，但总收益低于 V5/V6 inherit。
```

默认是 `inherit`。

## 推荐运行

```bash
python backtest/lf/eth_lf_portfolio_v6_momentum_bear_bull_reclaim_backtest.py \
  --start-date 2023-01-01 \
  --end-date 2026-06-15 \
  --warmup-start-date 2022-01-01 \
  --preset turbo \
  --bull-preset high \
  --bull-execution-mode inherit
```

## V6 inherit / turbo 结果

```text
总收益      2617.5741%
最大回撤    27.2762%
PF          3.9486
胜率        23.7288%
交易次数    118

engine_counts:
MOMENTUM_V3      49
BULL_RECLAIM_V2  46
BEAR_V3_ONLY     23

2023   +11.72%
2024  +224.77%
2025  +168.35%
2026  +179.10%
```

对比 V5 turbo（同样 warmup 起点）：

```text
V5 turbo:
总收益      1741.5734%
最大回撤    25.9304%
PF          4.0110
胜率        21.6495%
交易次数    97
2023        -7.09%左右

V6 inherit turbo:
总收益      2617.5741%
最大回撤    27.2762%
PF          3.9486
胜率        23.7288%
交易次数    118
2023       +11.72%
```

## V6 own / turbo 结果

```text
总收益      1432.4306%
最大回撤    24.6677%
PF          3.9212
胜率        34.1667%
交易次数    120

engine_counts:
MOMENTUM_V3      49
BULL_RECLAIM_V2  48
BEAR_V3_ONLY     23

2023   +14.66%
2024  +100.55%
2025  +138.77%
2026  +179.10%
```

## 判断

- `inherit` 是收益增强版：总收益显著高于 V5，但回撤小幅上升，胜率改善有限。
- `own` 是体验增强版：胜率明显提高、回撤低一点，但总收益低于 V5 turbo。
- 作为主研究版本，优先看 `inherit`。
- 如果用户更重视心理体验和胜率，可以看 `own`。
