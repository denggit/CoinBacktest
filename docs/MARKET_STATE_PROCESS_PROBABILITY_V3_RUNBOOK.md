# Market State Process & Conditional Probability Map V3

## 目标

V3 研究市场状态的多阶段过程、阶段有效期、阶段增量与因果条件概率。它不是独立交易策略，也不以单个状态扣费后是否赚钱作为唯一判定标准。

## 当前四类固定过程

1. 多头反转：卖压 → 卖压吸收 → 下扫收回 → 买盘恢复。
2. 空头反转：买压 → 买压吸收 → 上扫拒绝 → 卖盘恢复。
3. 向上突破：波动压缩 → 买方有效冲击 → 突破接受。
4. 向下突破：波动压缩 → 卖方有效冲击 → 跌破接受。

各阶段必须出现在后续已关闭 Bar，不能在同一 Bar 内连跳。每阶段超过固定 TTL 后失效。

## 数据

现有 `1m Trade Bar` 足够运行。需要真实的 `notional / buy_notional / sell_notional / delta_notional`。本轮不要求 Order Book、OI、Funding 或真实 liquidation。

默认数据窗口：

- Warmup：2022-01-01
- 正式研究：2023-01-01 至 2026-06-30
- Holdout：2025-07-01 起

## Windows 运行命令

```bash
python research\market_state\03_market_state_process_probability_v3.py --symbol ETH-USDT-SWAP --timeframe 1m --warmup-start-date 2022-01-01 --start-date 2023-01-01 --end-date 2026-06-30 --holdout-start 2025-07-01 --local-only
```

本地缺段且允许构建时：

```bash
python research\market_state\03_market_state_process_probability_v3.py --symbol ETH-USDT-SWAP --timeframe 1m --warmup-start-date 2022-01-01 --start-date 2023-01-01 --end-date 2026-06-30 --holdout-start 2025-07-01 --no-local-only
```

## 输出目录

`data/reports/research/market_state/03_market_state_process_probability_v3`

重点文件：

- `01_stage_information_summary.csv`：各阶段方向、MFE/MAE 与同年同波动基线增量。
- `02_stage_progression.csv`：每一步进入下一步的概率和阶段间隔。
- `03_stage_incremental_information.csv`：新增一步是否比父阶段更有信息。
- `05_pre_holdout_vs_holdout.csv`：前期与 Holdout 稳定性。
- `06_probability_calibration.csv`：在线概率的 Brier score 与偏差。
- `08_episode_duration_and_expiry.csv`：过程持续、超时与失效位置。
- `09_process_registry.csv`：KEEP_PROCESS_CANDIDATE / KEEP_STAGE_ONLY / REVISE_PROCESS / DROP_PROCESS。
- `10_causal_audit.csv`：可用时间与阶段顺序审计。
- `gpt_review_pack.zip`：交给 GPT 复核。

## 判定原则

- 单阶段可以作为上下文保留，不要求独立成为策略。
- 后一阶段必须相对父阶段增加稳定信息，不能只减少样本。
- 最终过程候选需要跨参数、跨年份和 Holdout 同方向。
- 当前过程不能训练自己的概率。
- 单独的固定持有诊断报告只用于观察，不参与状态有效性判定。
