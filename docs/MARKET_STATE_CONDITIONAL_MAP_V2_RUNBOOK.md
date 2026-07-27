# Market State Conditional Map V2

## 目的

V2 不把任何单个状态当成开仓策略，也不要求每个状态独立覆盖手续费。
它验证每个状态维度是否对自己负责的问题提供稳定的信息增量：

- 方向、订单流、冲击、吸收、位置：后续路径方向、延续/反转概率、MFE/MAE；
- 波动和市场活动：后续路径宽度与极端波动风险；
- 趋势阶段、年龄和质量：已确认路径是在增强还是衰减；
- 条件阶梯：每增加一层状态后，是否在父条件基础上继续增加信息，而不是只减少样本。

V2 不是策略回测。手续费不参与状态有效性的判定；以后形成可执行策略时仍必须按完整成本回测。

## 数据

第一轮只使用项目已有的 1m rich Trade Bar：

- OHLCV；
- trades_count、notional；
- buy/sell/delta notional；
- large buy/sell/delta notional；
- 当前 Market State V0.3 的趋势、波动、订单流、冲击/吸收和结构位置。

暂时不需要 Order Book、OI、Funding、Basis 或真实 liquidation。

## 运行

Windows 单行命令：

```bash
python research\market_state\02_market_state_conditional_map_v2.py --symbol ETH-USDT-SWAP --timeframe 1m --warmup-start-date 2022-01-01 --start-date 2023-01-01 --end-date 2026-06-30 --holdout-start 2025-07-01 --local-only
```

如果本地数据缺失，才使用：

```bash
python research\market_state\02_market_state_conditional_map_v2.py --symbol ETH-USDT-SWAP --timeframe 1m --warmup-start-date 2022-01-01 --start-date 2023-01-01 --end-date 2026-06-30 --holdout-start 2025-07-01 --no-local-only
```

报告目录：

```text
data\reports\research\market_state\02_market_state_conditional_map_v2
```

完成后优先检查：

```text
00_EXECUTIVE_SUMMARY.md
06_state_information_registry.csv
07_ladder_incremental.csv
11_axis_verdict_summary.csv
gpt_review_pack.zip
```

## 状态结果解释

### KEEP

状态在至少两个参数 Profile、至少两个前向周期、年度与 Holdout 中，对预设职责提供一致的正向信息。
这仍不代表可以单独交易，只表示值得保留在完整状态地图中。

### KEEP_CONTEXT_ONLY

有一定正向区分能力，但跨参数、跨周期或 Holdout 还不够稳定。保留为上下文，不能冻结语义。

### REVISE_SEMANTICS

状态有稳定信息，但方向与当前命名/预期相反。例如“上涨结构”可能更像成熟后回撤风险，而不是做多许可。
应修改名称和用途，不能直接删除信息。

### DROP

对职责目标没有稳定区分能力，继续加入组合只会增加复杂度和过拟合风险。

## 条件阶梯

V2 固定研究以下多阶段条件，不根据结果临时调参：

- 卖压吸收 → 低位 → 下扫收回 → 买盘恢复；
- 买压吸收 → 高位 → 上扫拒绝 → 卖盘恢复；
- 上涨结构 → 持续买压 → 买盘有效 → 突破接受；
- 下跌结构 → 持续卖压 → 卖盘有效 → 跌破接受；
- 波动压缩 → 单向订单流 → 有效冲击 → 突破/跌破接受。

`07_ladder_incremental.csv` 关注：

- `incremental_primary_uplift`：子条件相对父条件增加多少信息；
- `holdout_incremental_uplift`：Holdout 是否仍增加；
- `retention_ratio`：加入条件后保留多少样本；
- `positive_increment_year_ratio`：不同年份是否方向一致。

如果信息增量为负，说明新增条件没有帮助；如果增量为正但样本骤降，也不能直接继续堆条件。

## 因果规则

- 状态只使用已关闭数据；
- 条件在附加未来路径标签之前生成；
- 所有前向结果只作为研究标签；
- 信号时刻之后的下一根 open 作为统一执行参考；
- 高周期数据若以后加入，必须按 `available_time` 对齐；
- 不允许用全样本未来结果修改状态阈值；
- 不允许围绕单笔亏损增加条件。
