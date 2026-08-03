# ETH AI Trading R03.3.3.1 — 市场状态连续性小修正与审计

## 研究定位

本阶段仍然不直接开仓。它修复并审计 R03.3.3 的四层辅助市场状态：

- 战略层：1D / 4H；
- 战术层：4H / 1H；
- 入场层：30m / 15m / 5m / 1m；
- 活跃度层：低波动压缩、正常、高波动扩张。

每层保留连续分数与三态离散状态。四层各有 `-1 / 0 / +1` 三种状态，因此共有 12 个“层内状态名称”，理论联合组合为 `3^4 = 81`。系统不会把 81 种组合训练成 81 个固定类别，而是让四个维度同时存在，并继续保留连续分数、状态年龄、边界距离和翻转率。

## 修正一：战略状态因果阈值

旧版战略分数长期无法达到固定 `±0.30`，导致战略状态全为中性。

新版按过去 365 个完整日的战略分数分布计算阈值：

- 多头进入：过去分布 85% 分位；
- 空头进入：过去分布 15% 分位；
- 多头退出：过去分布 60% 分位；
- 空头退出：过去分布 40% 分位。

阈值严格使用前一日及更早数据，当前日和未来 OOS 数据均不能参与阈值计算。每个年度缓存包含 420 天 warmup，保证年初也有完整校准历史。

## 修正二：严格持续标签

旧版只比较当前状态与预测窗口终点状态。若状态中间离开、之后又回到原状态，会被误判为“持续”。

新版定义：

```text
状态持续 = 整个预测窗口内一次状态切换都没有
```

未来状态仍然只用于监督标签，不进入当前特征。

## 审计一：机械连续性基准

完整多周期 LightGBM 必须和三个简单基准比较：

1. 只使用状态年龄；
2. 只使用距迟滞切换边界的距离；
3. 状态年龄 + 边界距离 + 当前状态。

若完整模型不能在 2024 和 2025 同时明显超过最佳机械基准，只能得到：

```text
PASS_MECHANICAL_CONTINUITY_ONLY
```

不能宣称模型理解了更复杂的市场过程。

## 审计二：独立转换预警

完整 Universal 模型在训练期冻结最低持续概率 10% 分位阈值。测试期连续低分信号在 1 小时内合并为一个独立预警段。

报告统计：

- 每月独立预警次数；
- 预警后目标窗口内真实转换率；
- 假预警率；
- 转换前中位领先时间；
- 独立转换事件覆盖率；
- 每个预警段持续时间。

转换预警仍然只是风险上下文，禁止直接反向开仓或强制平仓。

## 数据合同

- 2020—2021：`src.data_feed.okx_loader.OKXDataLoader` 普通 1m OHLCV；
- 2022—2025：`src.data_feed.okx_trade_bar_loader.OKXTradeBarLoader`；
- Universal 模型只使用全历史共有 OHLCV 特征；
- Trade-enhanced 模型只在真实 Trade 特征存在时做增量对照；
- 2026H1 继续封存。

## 运行

```bat
python research\eth_ai_trading\03_3_3_1_market_state_continuity_audit.py
```

首次运行会建立独立新版缓存：

```text
data/cache/eth_ai_trading/r03_3_3_1_universal_state
```

报告目录：

```text
data/reports/research/eth_ai_trading/03_3_3_1_market_state_continuity_audit
```

重点文件：

- `03_state_duration_atlas.csv`
- `06_model_metrics.csv`
- `08_stable_candidates.csv`
- `16_strategic_threshold_audit.csv`
- `17_mechanical_baseline_metrics.csv`
- `18_mechanical_increment_audit.csv`
- `19_transition_alert_metrics.csv`
- `20_transition_alert_episodes.csv`
- `99_decision.md`
- `gpt_review_pack.zip`

## 决策含义

- `PASS_STATE_CONTINUITY_INCREMENT`：完整多周期模型跨年稳定，并在两年都超过机械基准；
- `PASS_MECHANICAL_CONTINUITY_ONLY`：状态持续可预测，但主要来自年龄和边界距离；
- `FAIL_NO_STABLE_CONTINUITY_MODEL`：连机械连续性也不能稳定利用；
- `BLOCKED_*`：数据、依赖或管线未真正运行。
