# R03.2 长上下文 3%–5% Swing 机会模型

## 目标

本阶段只回答一个问题：扩展到数月级高周期上下文并加入连续市场过程后，能否让同一套多空模型在 2024 与 2025 稳定识别目标先于风险线的 3%–5% 机会。

本阶段不开发持仓管理模型、强化学习、短线 Sleeve、执行模型或 AetherEdge 接入。

## 与 R03.1 的唯一研究差异

保持不变：

- 3% / 5% 目标和 1.25% / 1.75% 风险线。
- 真实 1m 路径上的 target-before-adverse 标签。
- 同一分钟冲突按 adverse-first。
- LightGBM 参数、信号分位、费用、滑点、延迟、退出和验收门槛。
- 多空独立训练，2024 支持、2025 选择、2026 只复核冠军。

只改变特征上下文：

| 周期 | R03.2 最长窗口 |
|---|---:|
| 1D | 365 日 |
| 4H | 720 根，约 120 日 |
| 1H | 720 根，约 30 日 |
| 30m / 15m / 5m / 1m | 沿用 R03.1，用于战术入场位置 |

新增特征包括：

- 长期高低点位置、距长期高低点时间。
- 从长期高点回撤和从长期低点反弹的幅度。
- 最近结构高点、低点相对上一结构段的抬升或降低。
- 价格位于 EMA50 上下及 EMA20/EMA50 排列的持续时间。
- EMA100 / EMA200 关系和 EMA200 斜率。
- 推动、回调、恢复和区间占比。
- 波动率与 ATR 的启动、扩张、衰退生命周期。
- 主动成交压力的持续性。
- 日线、4H、1H 的方向一致性和大趋势中的战术回调关系。

这些都是在 bar 完成后计算并按 `available_time` 对齐。模型看到的是当前时刻的长周期因果快照，不是单根 K 线，也暂时不是直接读取完整序列的 TCN / Transformer。

## 缓存

R03.2 使用独立缓存，不覆盖 R03 或 R03.1：

```text
data/cache/eth_ai_trading/r03_2_long_context
data/cache/eth_ai_trading/r03_2_exact_outcomes
```

第一次运行必须构建新的长上下文特征缓存。后续运行会复用；不要无故添加 `--force-rebuild-long-context-cache`。

## 运行

```bat
python research\eth_ai_trading\03_2_swing_long_context.py
```

报告：

```text
data/reports/research/eth_ai_trading/03_2_swing_long_context
```

重点文件：

```text
99_decision.md
07_feature_importance.csv
05_trade_stress_matrix.csv
06_trades.csv
```

## 决策纪律

- `PASS_SWING_LONG_CONTEXT_MVP`：进入模型导出、离线/实时特征一致性和 AetherEdge 影子推理。
- `FAIL_VALIDATION`：长上下文仍无法让 2024 与 2025 使用同一逻辑，不通过调退出或参数网格救模型。
- `FAIL_LOCKED_HOLDOUT`：2024/2025 候选未能在 2026 复核，不能实盘。
