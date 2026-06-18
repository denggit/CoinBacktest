# ETH LF Portfolio V2 Confirm-Only

## 定位

Portfolio V2 是在 Portfolio V1 报告复盘后的修正版。

V1 发现：

- Turtle 单独交易不是增量，反而是负贡献；
- Turtle 做不做空，对组合整体区别不大；
- 真正有价值的是 V8 Trend Rider 主引擎；
- Turtle 更适合做“超级趋势确认器”，而不是单独抢仓位。

所以 V2 改成：

```text
V8 Trend Rider：唯一可以单独开仓的主引擎
Turtle V2：只做确认，不允许单独开仓
V8 + Turtle 同方向：共振，适度提高 quality_mult
V8 与 Turtle 冲突：V8 优先，记录 conflict
只有 Turtle 信号：忽略，不开仓
```

## 反未来函数约束

V2 继续沿用 SAFE 执行约束：

```text
1. 日线 regime 必须 shift 后映射到 4H
2. Donchian entry / exit 使用 rolling(...).shift(1)
3. 当前 4H 收盘确认信号
4. 下一根 4H open 执行入场/退出
5. 当前 bar close 更新的新 stop 下一根 bar 才生效
6. 不允许同一根 K 线内新 stop 立刻触发
7. 同一时间只允许一个 active position
```

## 默认运行

```bash
python backtest/lf/eth_lf_portfolio_v2_confirm_only_backtest.py \
  --start-date 2023-01-01 \
  --end-date 2026-06-15 \
  --preset turbo
```

## 更激进压力测试

```bash
python backtest/lf/eth_lf_portfolio_v2_confirm_only_backtest.py \
  --start-date 2023-01-01 \
  --end-date 2026-06-15 \
  --preset ultra \
  --out-dir data/reports/lf/eth_lf_portfolio_v2_confirm_only_ultra
```

## 本地回测结果

测试区间：2023-01-01 至 2026-06-15。

### turbo 默认

```text
总收益      603.43%
CAGR        79.24%
最大回撤    26.58%
PF          2.45
交易次数    139

2023   +12.05%
2024   +76.92%
2025   +76.90%
2026   +100.60%
```

### ultra

```text
总收益      944.13%
CAGR        101.72%
最大回撤    35.35%
PF          2.38
交易次数    139

2023   +12.19%
2024   +92.98%
2025   +104.87%
2026   +135.40%
```

## 重要提醒

`confluence_quality_boost` 默认是 1.30。虽然把它拉到 1.50 会让回测收益更高，但目前共振闭合交易样本很少，强行拉高容易过拟合，所以默认不使用过高 boost。

V2 是结构性优化：去掉 Turtle 单独负贡献，只保留 Turtle 确认价值。它不是简单加杠杆，也没有放宽 SAFE 执行假设。
