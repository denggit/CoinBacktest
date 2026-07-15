# Swing Low Typology

该目录只做 **历史 Swing Low 类型研究**，不写策略、不做收益回测。

## 01：Causal Swing Low Typology

历史 Swing Low 的结构低点与可交易收益分开定义：

- 结构锚点仍使用历史 extreme 当根的 `low`；
- 假设在 extreme 后下一根 K 线的 `open` 入场；
- 后续最多观察指定 bars，只在某根未来 K 线**收盘后**，用该根 `close` 判断是否达到目标幅度；
- 未来 `high/low` 不参与 TP、MFE、MAE 或完成速度计算；
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


> 标签口径在 1.1.0 起统一为 `low` 结构锚点、`next bar open` 入场、`future closed-bar close` 判断收益。旧版 01/02/03 报告不能与新版 04 混用，必须从 01 开始重新运行。

## 02：C3 Hierarchical Sequence Typology

对 01 中占比最大的 C3 使用约 315 个因果特征做第二阶段冻结分类：

- 价格结构、回撤、反弹、测试次数与间隔；
- CVD、大单 CVD、主动买卖流和成交活动；
- 价格对订单流的响应效率与吸收特征；
- 低点前 240 根 bar 的 12 段顺序路径；
- 2023-2024 拟合，2025-2026 使用冻结模型分类。

输出 C3-A 至 C3-E，并为 03 提供冻结的来源类型和完整因果特征矩阵。

## 03：Mechanism Hierarchical Typology

03 不再增加 KMeans 数量，而是用**弱监督机制分数 + 训练期稳健归一化 + 留出期冻结映射**做分层研究。

第一层：

- `shock`：成交/波动冲击与末段集中下杀；
- `trend`：长周期方向性下跌、价格与 CVD 持续走弱；
- `base`：反复测试、底部停留、吸收与压缩。

第二层只深入冻结的来源集合：

- C3-C：均匀下跌、阶段加速、同步持续卖压、价格-CVD 背离、趋势中短暂停顿；
- C3-E：吸收、压缩、spring/假跌破、反复测试支撑、缓慢积累。

新增事件级顺序特征包括测试深度变化、测试后反弹变化、卖压及价格冲击衰减、成交先缩后放、大单卖压持续/衰减，以及完整的价格、CVD、成交额、成交笔数、大单流和支撑测试路径。

因果保障：

- 所有类型特征截止 Swing Low 当根已关闭 bar；
- 未来路径只保留在历史标签与独立诊断中；
- 训练期拟合归一化、分数校准、模糊阈值和浅层解释规则；
- 留出期只应用冻结模型；
- 同时扰动未来元数据和 Swing Low 后原始 OHLC/订单流，类型特征必须完全不变；
- 随机种子稳定性通过训练期 bootstrap 重拟合并在训练/留出分别报告。

运行：

```bash
python research/market_structure/swing_low_typology/03_mechanism_hierarchical_typology_research.py --symbol ETH-USDT-SWAP --timeframe 1m --start-date 2023-01-01 --end-date 2026-06-30 --warmup-start-date 2022-01-01 --target-move-pct 1.0 --max-completion-bars 60
```

完整运行会在 03 报告目录生成 `gpt_review_pack.zip`。本研究仍是历史类型发现，不是实时 Swing Low 识别模型，也不包含交易或收益回测。

## 04：Online Mechanism Recognizability

04 从历史类型发现转向**当前已关闭 K 线的实时可识别性研究**，但仍然不是策略或收益回测。

研究范围只保留 03 中相对清晰的六类：

- `T2_staged_acceleration`
- `T3_sync_persistent_selling`
- `T4_price_cvd_divergence`
- `B2_compression`
- `B3_spring_false_breakdown`
- `B4_repeated_support_test`

`B1_absorption` 和 `B5_slow_accumulation` 暂时排除。

时序定义：

```text
当前 1m bar 已关闭
    ↓
当前 bar 及此前数据生成特征和候选分数
    ↓
下一根 1m bar open 作为可执行参考价格
    ↓
最多观察后续 60 根 bar；某根已关闭 bar 的 close 达到 +1% 才视为 TP 成功
```

04 的主标签是：

- `joint_swing_tp_success`：当前 bar 必须精确对应 03 中 T2/T3/T4/B2/B3/B4 的历史 Swing Low，并且从下一根开盘计算，60 根内至少有一根已关闭 K 线的 close 达到 +1%。

同时输出辅助标签，防止只依赖单一分数定义：

- `historical_clear_swing_low`：是否精确命中 03 清晰类型 Swing Low；
- `tp_hit_1pct`：60 根内是否有未来 close 达到 next open 的 +1%；
- `mfe_only_score`：未来 close 的 MFE/1%，例如最高收盘上涨 0.6% 对应 60 分；
- `tp_priority_score`：未来 close 达到 +1% 直接为 100，否则使用 close 路径的 `(MFE-MAE)/1%`；
- `first_touch_score`：比较未来 close 先达到 +1% 还是 -1%，识别先深跌再反弹的危险路径。

模型验证严格按时间分段：

- 2023：拟合候选模型；
- 2024：选择 Logistic 或浅层 HistGradientBoosting；
- 2025–2026H1：冻结模型留出验证。

2023 年末和 2024 年末会剔除未来 60 根标签跨越下一数据分段边界的候选，避免训练标签读取验证期或留出期路径。

同时比较：

- 一个覆盖六类机制的统一模型；
- 每种机制各自的 specialist 模型；specialist 的正样本必须同时满足“在线识别机制与 03 历史机制一致”以及“下一根开盘后触及 +1%”。

报告会给出 PR-AUC、概率分桶、Top 1%/5%/10% 精度与提升倍数、逐机制结果、随机种子稳定性、代表性误报/漏报，以及未来原始 bar 扰动因果审计。

### 04 的大样本实现

在线候选通常有数十万行，不能把所有候选的 315+ 个特征先放进一个巨大 `list[dict]`。04 使用以下等价优化：

- 候选门控结果按列一次性构建，避免几十万个 Python 字典；
- 特征按 `--candidate-feature-chunk-size` 分块处理，默认每块 20,000 行；
- 第一遍只计算冻结机制模型实际使用的紧凑序列特征；这些字段与 02 完整特征中的同名字段逐值一致；
- 通过冻结机制清晰度门槛后，再为保留候选补齐原 04 使用的完整序列特征，因此最终模型特征和旧实现保持一致；
- 机制路径使用 `--feature-workers` 多进程计算，默认 4 个 worker；
- 每块立即应用冻结机制清晰度阈值，原流程最终必然丢弃的模糊候选不会长期驻留内存；
- MFE、MAE、TP 和首触标签使用 future close 的零拷贝滑动窗口分块计算；
- 特征仍然严格截止当前已关闭 bar，没有改变任何时序或未来标签定义。

内存较小的机器可以降低 `--candidate-feature-chunk-size`；CPU 核数较少时可以设置 `--feature-workers 1`。这两个参数只影响资源使用和速度，不改变研究结果。

运行：

```bash
python research/market_structure/swing_low_typology/04_online_mechanism_recognizability_research.py --symbol ETH-USDT-SWAP --timeframe 1m --start-date 2023-01-01 --end-date 2026-06-30 --warmup-start-date 2022-01-01 --target-move-pct 1.0 --forward-horizon-bars 60
```

完整运行会生成 `gpt_review_pack.zip`。只有 04 证明某些类型能够在当时稳定识别，才进入后续事件研究；04 本身不设置止损、仓位、手续费或交易退出。
