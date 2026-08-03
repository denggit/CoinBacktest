# ETH AI Trading R03.3 — Future Market Process Forecast

## 目的

R03.3不识别已经发生的“当前状态”，而是预测未来6/12/24小时内是否出现新的市场过程启动点：

1. 上涨单边扩张；
2. 下跌单边扩张；
3. 高波动双向震荡；
4. 低机会状态。

正样本必须严格位于启动点之前。启动后才出现的高分信号会进入tail-car统计，不会计为预测成功。

## 数据

- 复用R03.2长期缓存：1D最长365天、4H约120天、1H约30天，以及30m/15m/5m/1m。
- 新增公共5s Trade Bar微观特征：主动流不平衡、压力持续、方向翻转、冲击一致性、吸收代理、大单占比、成交活跃度。
- 所有数据通过`src.data_feed.OKXTradeBarLoader`读取。
- `build_missing=False`，不会访问或自动重建Raw Trades。
- 5s数据覆盖不足时默认阻塞，不会静默降级成纯1m模型。

10s与5s来自同一批Trades的确定性聚合，第一版不重复加入，避免增加高度共线特征与一倍读取成本。

## 时间划分

- 2022：仅供R03.2长期特征warmup。
- 2023：训练/校准来源。
- 2024：第一次纯OOS。
- 2023–2024：扩展训练/校准来源。
- 2025：第二次纯OOS。
- 2026-01-01至2026-06-30：完全封存，本轮不展示结果。

## 运行

```text
python research\eth_ai_trading\03_3_future_process_forecast.py
```

正常情况下不要追加任何force参数。只有对应缓存损坏或事件定义代码改变时才使用：

```text
python research\eth_ai_trading\03_3_future_process_forecast.py --force-rebuild-events
```

```text
python research\eth_ai_trading\03_3_future_process_forecast.py --force-rebuild-micro
```

## 缓存

```text
data\cache\eth_ai_trading\r03_2_long_context
data\cache\eth_ai_trading\r03_3_process_events
data\cache\eth_ai_trading\r03_3_micro_5s
```

## 报告

```text
data\reports\research\eth_ai_trading\03_3_future_process_forecast
```

重点查看：

- `03_event_atlas.csv`：独立过程启动点与未来路径；
- `04_event_yearly_summary.csv`：每年独立事件数量；
- `07_probability_metrics.csv`：AUC、AP、Brier；
- `08_top_quantile_forecast_metrics.csv`：Lift、领先小时、ongoing与tail-car；
- `09_stable_candidates.csv`：同配置2024/2025稳定审核；
- `10_micro_increment.csv`：5s微观特征的真实增量；
- `12_pre_event_feature_uplift.csv`：启动前24/12/6/3/1小时特征差异；
- `99_decision.md`：是否允许进入R03.4入场模型。

## 通过纪律

R03.3不是交易策略，不看PF。至少一个上涨扩张、下跌扩张或高波动震荡头必须使用同一配置同时通过2024和2025：

- AUC不低于0.56；
- Top5% Lift不低于1.5；
- 信号量足够；
- 12/24小时模型真实正样本中位领先不少于3小时；
- 已走30%以上的尾班车率不超过35%；
- 校准后Brier不差于常数基准。

通过后只进入R03.4低MAE入场研究，不直接写入回测或AetherEdge。
