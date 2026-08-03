# ETH AI Trading R03.3.3 — 多周期市场状态连续性与转换研究

## 研究定位

R03.3.3 不直接产生开仓信号。它建立可供后续方向、入场、持仓管理和风险模型消费的市场上下文：

- 战略层：1D / 4H，持续数天至数月；
- 战术层：4H / 1H，持续数小时至数天；
- 入场层：30m / 15m / 5m / 1m，持续数分钟至数小时；
- 活跃度层：当前波动处于压缩、正常还是扩张。

每一层输出连续分数、因果迟滞状态、状态年龄、翻转率和跨层对齐关系。大周期和小周期允许同时处于不同状态。

## 数据合同

### 2020—2021

使用：

```python
src.data_feed.okx_loader.OKXDataLoader
```

读取普通 `ETH-USDT-SWAP` 1m OHLCV。

### 2022—2025

使用：

```python
src.data_feed.okx_trade_bar_loader.OKXTradeBarLoader
```

读取现有1m Trade Bar。

Universal模型仅使用所有年份共有的OHLCV派生特征。Trade增强模型仅在2022年以后加入真实主动买卖、大单和订单流特征，不把2020—2021缺失字段填成0。

2026年上半年继续封存。

## 因果状态定义

状态特征只使用决策时刻已经完成并可见的bar：

- 1D、4H、1H、30m、15m、5m、1m特征统一移动到bar完成时间；
- 状态年龄和翻转率只使用过去及当前状态；
- 未来状态仅用于监督训练标签；
- 训练与测试边界加入最大目标周期加24小时的purge/embargo。

## 监督目标

- 战略状态未来24小时是否保持；
- 战略状态未来72小时是否保持；
- 战术状态未来3小时是否保持；
- 战术状态未来6小时是否保持；
- 入场状态未来1小时是否保持；
- 活跃度状态未来3小时是否保持。

低持续概率可以被后续模型解释为状态转换风险，但本研究不直接触发订单。

## 迟滞状态

方向状态采用双门槛：

- 进入上涨/下跌需要超过较高门槛；
- 已进入后，只有跌破较低退出门槛才回到中性；
- 明确穿越反方向进入门槛时允许直接转换。

报告同时展示原始硬阈值状态和迟滞状态的持续时间与每日翻转次数。

## 训练切分

- 2020：365天长期特征warmup；
- 2021—2023：训练；
- 2024：纯OOS；
- 2021—2024：重新训练；
- 2025：纯OOS；
- 2026H1：封存。

冠军任务还会运行训练年份归因矩阵，比较仅2023、仅2024、近期组合和2021以后全历史训练。

## 运行

```bat
python research\eth_ai_trading\03_3_3_market_state_continuity.py
```

强制重建本阶段独立缓存：

```bat
python research\eth_ai_trading\03_3_3_market_state_continuity.py --force-rebuild-state-cache
```

## 缓存与报告

缓存：

```text
data/cache/eth_ai_trading/r03_3_3_universal_state
```

报告：

```text
data/reports/research/eth_ai_trading/03_3_3_market_state_continuity
```

重点文件：

- `03_state_duration_atlas.csv`
- `04_continuity_target_distribution.csv`
- `05_state_opportunity_link.csv`
- `06_model_metrics.csv`
- `08_stable_candidates.csv`
- `09_training_year_attribution.csv`
- `10_trade_feature_increment.csv`
- `11_feature_importance.csv`
- `99_decision.md`
- `gpt_review_pack.zip`

## 通过标准

同一任务必须同时在2024和2025满足：

- AUC不低于0.60；
- Brier Skill不差于常数概率基准；
- 最低持续概率十分位的状态转换Lift不低于1.25。

通过只表示状态连续性可以作为辅助上下文。是否能赚钱仍需后续方向与入场模型在真实成本下证明。
