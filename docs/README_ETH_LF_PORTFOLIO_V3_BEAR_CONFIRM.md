# ETH LF Portfolio V3 Bear Confirm

新增文件：

```text
backtest/lf/eth_lf_portfolio_v3_bear_confirm_backtest.py
```

## 定位

Portfolio V3 是低频组合调度器，不是简单把多个策略收益相加。

组合引擎：

1. **V8 Trend Rider**：主引擎，负责绝大多数多空趋势交易。
2. **Turtle V2**：只做 long super-trend 确认器，不允许单独开仓。
3. **Bear Short Engine V3**：做空确认器，也允许在 V8 没信号时作为少量独立 short 候选。

组合约束：

```text
同一时间只允许一个 ETH active position
不双开
不对冲
收盘确认，下一根 4H open 执行
当前 bar close 更新的新 stop 只能下一根 bar 生效
```

## 调度规则

```text
V8 long:
    正常做多。
    如果 Turtle 同向 long，则标记 V8_TURTLE_CONFLUENCE，并提高 quality_mult。
    如果 Bear 同时 short，不对冲，V8 long 优先，记录冲突。

V8 short:
    默认正常做空。
    如果 Bear V3 同向 short，则标记 V8_BEAR_CONFLUENCE，并提高 quality_mult。
    默认不强制 Bear gate，因为测试中严格 gate 会明显削弱 2026 大跌收益。

Bear V3 short only:
    如果 V8 无信号，Bear V3 可以作为独立 short 候选。
    默认开启，可用 --disable-bear-standalone 关闭。

Turtle only:
    忽略，不开仓。V1 复盘显示 Turtle 单独交易没有正贡献。
```

## 反未来函数约束

```text
1. V8 / Turtle / Bear 的高周期 regime 均在各自 build_features 内 shift 后再映射到 4H。
2. Donchian entry / exit 使用 rolling(...).shift(1)。
3. 当前 4H 收盘确认 signal。
4. 下一根 4H open 入场、加仓、退出。
5. 当前 bar close 更新的新 trailing stop 下一根 bar 才生效。
6. 不允许同一根 K 线内用收盘后才知道的新 stop 触发。
```

## 默认手续费

默认单边手续费：

```text
fee_rate = 0.00055
```

即完整开平仓约 0.11%。

## 运行

默认 turbo：

```bash
python backtest/lf/eth_lf_portfolio_v3_bear_confirm_backtest.py \
  --start-date 2023-01-01 \
  --end-date 2026-06-15 \
  --preset turbo
```

ultra 压力测试：

```bash
python backtest/lf/eth_lf_portfolio_v3_bear_confirm_backtest.py \
  --start-date 2023-01-01 \
  --end-date 2026-06-15 \
  --preset ultra \
  --out-dir data/reports/lf/eth_lf_portfolio_v3_bear_confirm_ultra
```

防守版，降低 Bear 未确认的 V8 short：

```bash
python backtest/lf/eth_lf_portfolio_v3_bear_confirm_backtest.py \
  --start-date 2023-01-01 \
  --end-date 2026-06-15 \
  --preset turbo \
  --reduce-nonbear-short-quality 0.85 \
  --out-dir data/reports/lf/eth_lf_portfolio_v3_bear_confirm_defensive
```

## 本地测试结果

区间：2023-01-01 到 2026-06-15

### turbo 默认

```text
总收益      645.2661%
CAGR        82.37%
最大回撤    26.3403%
PF          2.5081
交易次数    140

2023   +11.54%
2024   +79.72%
2025   +79.97%
2026   +106.58%

engine_counts:
V8_LONG                 70
V8_SHORT                63
V8_BEAR_CONFLUENCE       4
BEAR_V3_ONLY             2
V8_TURTLE_CONFLUENCE     1
```

### ultra 压力测试

```text
总收益      1022.3053%
CAGR        106.12%
最大回撤    35.0921%
PF          2.4390
交易次数    140

2023   +11.50%
2024   +97.08%
2025   +109.60%
2026   +143.65%
```

### turbo 防守版

```text
参数：--reduce-nonbear-short-quality 0.85

总收益      616.9690%
CAGR        80.27%
最大回撤    24.7623%
PF          2.5000

2023   +11.91%
2024   +81.80%
2025   +83.34%
2026   +92.21%
```

## 结论

V3 对比 V2 是小幅但结构更干净的升级。Bear V3 不是大幅增加交易次数，而是：

```text
1. 给 V8 short 做确认加权；
2. 在 V8 无信号时补充少量高质量熊市 short；
3. 保持单持仓、无对冲、无收益简单相加。
```

注意：`V8_BEAR_CONFLUENCE` 和 `BEAR_V3_ONLY` 的闭合交易样本仍然不多，所以默认 `bear_confluence_quality_boost=1.30`，不建议为了收益继续无脑拉到很高，容易过拟合。
