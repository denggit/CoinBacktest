# Market State Process Semantics V3.1

## 目标

V3.1 不增加新的市场状态轴，只纠正 V3 已验证过宽的两个过程语义：

1. Sweep/Reclaim 后的“恢复”不能由轻微订单流翻正自动完成。
2. 波动压缩后的“突破接受”不能由一次晚期冲击或单根收盘越界自动完成。

本研究仍然是市场状态过程审计，不是独立交易策略。

## V3.1 固定过程

### 多头反转

`卖压 → 卖压吸收 → 下扫收回 → 严格买盘恢复`

最后阶段必须在 Sweep 后的后续已关闭 Bar 出现，并同时满足：

- 新的反向短周期订单流超过阈值；
- 中周期订单流已经翻向；
- 订单流加速度为正；
- 订单流对价格产生正向响应；
- 价格继续收复 Sweep 收盘价以上的 ATR 缓冲。

空头反转完全对称。

### 向上突破

`压缩成熟 → 新买方突破冲击 → 回踩或停留接受`

- 压缩必须连续达到最少 Bars，只在首次成熟时启动一次过程。
- 冲击必须发生在压缩结束后的短窗口内，同时伴随波动退出、订单流强度、冲击有效性和真实价格越过已知阻力。
- 最终接受必须发生在后续 Bar；价格需要回踩守住或连续停留在突破位之上，且不能在期间明显跌回失败区。

向下跌破完全对称。

## 数据

继续使用现有 `1m Trade Bar`：

- `notional / buy_notional / sell_notional / delta_notional`
- OHLCV
- 本地 Market State 衍生的订单流、冲击、吸收和结构位置

本轮不需要 Order Book、OI、Funding 或 liquidation 数据。

默认窗口：

- Warmup：2022-01-01
- 正式研究：2023-01-01 至 2026-06-30
- Holdout：2025-07-01 起

## Windows 运行命令

```bash
python research\market_state\04_market_state_process_semantics_v3_1.py --symbol ETH-USDT-SWAP --timeframe 1m --warmup-start-date 2022-01-01 --start-date 2023-01-01 --end-date 2026-06-30 --holdout-start 2025-07-01 --local-only
```

本地缺段且允许构建时：

```bash
python research\market_state\04_market_state_process_semantics_v3_1.py --symbol ETH-USDT-SWAP --timeframe 1m --warmup-start-date 2022-01-01 --start-date 2023-01-01 --end-date 2026-06-30 --holdout-start 2025-07-01 --no-local-only
```

不需要直接比较旧 V3 进度率时，可追加：

```text
--skip-legacy-comparison
```

## 输出目录

`data/reports/research/market_state/04_market_state_process_semantics_v3_1`

重点文件：

- `01_stage_information_summary.csv`：各严格阶段的方向、MFE/MAE及基线增量。
- `02_stage_progression.csv`：V3.1 每一步推进率和阶段间隔。
- `03_stage_incremental_information.csv`：严格新阶段是否增加信息。
- `05_pre_holdout_vs_holdout.csv`：样本内与 Holdout 稳定性。
- `09_process_registry.csv`：过程保留、只保留阶段、重构或删除结论。
- `10_causal_audit.csv`：可用时间与阶段顺序审计。
- `13_v3_vs_v3_1_progression.csv`：旧 V3 与 V3.1 推进率、样本选择性直接比较。
- `gpt_review_pack.zip`：用于下一轮复核。

## 判定原则

- 旧 V3 接近自动推进的阶段，V3.1 必须显著降低推进率。
- 降低推进率本身不是成功；严格阶段还必须相对父阶段提供稳定的信息增量。
- 不能通过把样本压到极少来制造漂亮结果。
- Fast/Base/Slow、年度与 Holdout 必须方向一致。
- 所有阶段只能使用当时已关闭且已可用的数据。
