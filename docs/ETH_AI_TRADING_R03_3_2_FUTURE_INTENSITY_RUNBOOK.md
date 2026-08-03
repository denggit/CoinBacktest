# ETH AI Trading R03.3.2 — Continuous Future Opportunity Intensity

## 研究目的

R03.3和R03.3.1证明，离散“状态启动事件”可以找到一定提前量，但事件阈值造成大量看似误报的样本，也无法确认这些样本是否仍有可交易波动。

R03.3.2不再预测某个稀疏事件标签，改为预测未来6/12小时的连续机会强度。

## 当下市场状态的定义

当下市场状态不使用未来数据，也不强制压成一个“趋势/震荡”标签。模型在每个15分钟决策点看到一个因果状态向量：

1. 长期结构：1D/4H长期方向、区间位置、趋势年龄、回撤和反弹阶段。
2. 中周期过程：4H/1H推动、回调、恢复、趋势效率和结构延续。
3. 短周期结构：30m/15m/5m/1m突破距离、区间位置和当前回调状态。
4. 波动阶段：ATR、实现波动、压缩/扩张和波动率生命周期。
5. 订单流与冲击：主动买卖、大单压力、吸收代理和价格冲击效率。
6. 活跃度与位置：成交量异常、成交强度、距离长期高低点和边界的位置。

这些维度全部来自决策时刻已经完成并可用的bar。未来行情只用于生成训练目标。

## 连续目标

每个6小时和12小时窗口生成：

- `future_range_pct`：未来最高价/最低价形成的完整区间。
- `future_max_directional_pct`：向上MFE和向下MFE中较大的一侧。
- `future_two_sided_pct`：向上MFE和向下MFE中较小的一侧，代表双向短线机会。
- `future_range_atr_multiple`：未来区间相对当前因果4H ATR尺度的倍数。

未来窗口严格从当前15分钟bar之后开始。

## 模型对比

- `macro_lightgbm`
- `multiframe_lightgbm`
- `multiframe_micro_lightgbm`

通过对比后两者，审核公共5s Trade Bar在连续强度预测中是否提供增量。

## Walk-forward

- WF_2024：2023训练，2023Q4校准，2024纯OOS。
- WF_2025：2023至2024Q3训练，2024Q4校准，2025纯OOS。
- 2026-01-01至2026-06-30封存，不读取结果。

## 核心验收

不是要求准确预测具体涨跌幅，而是要求机会强度排序稳定：

- 两年Rank IC均不低于0.10；
- 两年Top Decile实际强度均至少为全样本1.20倍；
- 两年十分位预测与实际均值的单调性均不低于0.70；
- 微观数据增量单独报告，不因特征更多而默认保留。

若仍失败，停止继续堆普通价格和Trade Bar特征，转向OI、Books、清算或具体交易机制。

## 运行

```bat
python research\eth_ai_trading\03_3_2_future_opportunity_intensity.py
```

正常运行不要使用任何`--force`参数。

新目标缓存：

```text
data\cache\eth_ai_trading\r03_3_2_future_intensity
```

报告：

```text
data\reports\research\eth_ai_trading\03_3_2_future_intensity
```

重点文件：

- `02_current_state_definition.json`
- `03_target_distribution.csv`
- `04_regression_metrics.csv`
- `05_prediction_decile_curve.csv`
- `06_calibration_threshold_metrics.csv`
- `07_stable_candidates.csv`
- `08_micro_increment.csv`
- `09_feature_importance.csv`
- `99_decision.md`
- `gpt_review_pack.zip`
