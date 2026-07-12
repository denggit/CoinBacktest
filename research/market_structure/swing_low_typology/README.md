# Swing Low Typology

该目录只做 **历史 Swing Low 类型研究**，不写策略、不做收益回测。

## 01：Causal Swing Low Typology

定义与 Analyze Tool 的 `Swing Extreme 后续涨跌幅` 插件一致：

- 先通过未来路径确认某个历史 extreme 后，在限定 bars 内上涨达到目标幅度；
- 未来数据只负责生成 `swing_low` 历史标签；
- 聚类特征严格截止 extreme 当根已关闭 trade bar；左标签时间会换算为 bar close available_time；
- 不允许 confirmation、completion、future return、MFE/MAE 等字段进入特征；
- 2023-2024 默认作为开发段，拟合清洗、缩放、PCA、聚类数量和中心；
- 2025-2026H1 只使用冻结模型分组，检查类型是否仍然存在；
- 使用浅层决策树生成可解释的组别规则，但它只解释聚类，不预测涨跌。

主要特征覆盖：

- extreme 当根 K 线结构；
- 5/15/30/60/120 bars 的价格路径、回撤、位置、波动和低点测试；
- 主动买卖 Delta、大单 Delta、卖压持续性与前后半段变化；
- 成交额、成交笔数、最大单、大单成交占比；
- 价格与订单流错位、吸收类特征；
- 只使用 trade bar，字段不足时直接失败，不退化成 OHLCV。

运行：

```bash
python research/market_structure/swing_low_typology/01_causal_swing_low_typology_research.py --symbol ETH-USDT-SWAP --timeframe 1m --start-date 2023-01-01 --end-date 2026-06-30 --warmup-start-date 2022-01-01 --target-move-pct 1.0 --max-completion-bars 60
```
