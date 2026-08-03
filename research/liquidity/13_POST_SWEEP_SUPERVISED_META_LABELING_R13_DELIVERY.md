# 13 Post-Sweep Supervised Meta-Labeling R13

> Fix1 / v1.0.2：修复 OI 可选字段缺失时 `pd.to_numeric(None)` 退化为标量并触发 `.notna()` 异常；所有可选 OI 输入统一保持与事件表对齐的 Series 语义。与此同时明确 R13 v1 只研究最简单的开仓筛选：2R 结构目标与自然结构止损均未在 180 分钟观察窗内触发的 TIME/INVALID 路径会被视为未解析并退出训练，不把 180 分钟收盘当作策略时间平仓。

## 1. 研究目标

R13 不再继续手工叠加 Sweep-Reclaim 条件，而是检验：在 R09 已确认的真实 Swing-Low Liquidity Sweep 事件中，监督学习能否利用结构、扫盘后路径、1 秒 Trade、r0020 Range、Range Footprint 与 OI 的非线性交互，筛选出扣除真实成本后稳定盈利的 Long / Short 子集；其余事件必须 Skip。

R13 是研究脚本，不是可直接实盘的策略插件。只有冻结 Holdout、2 倍成本和稳健性门禁全部通过，候选才允许进入下一阶段的标准回测。

## 2. 独立样本与数据主键

- 独立事件主键：R09 `zone_event_id`。
- 全量预期：约 18,292 个真实 Zone Sweep，而不是把 R09/R12 展开的十几万行当作独立样本。
- 同一 Sweep 的 M0/M3/M5/M10 记录必须处于同一时间分区，禁止跨训练、验证和 Holdout。
- M0、M3、M5、M10 分别训练独立模型，不把四个检查点随机混成一个训练集。

## 3. 决策时点

| 模型 | 决策时点 | 可使用信息 |
|---|---:|---|
| M0 | Sweep 首个可执行时点 | R09 结构、事件 Bar 与完整 1m 释放信息 |
| M3 | Sweep 后 3 分钟 | M0 信息 + R12 前 3 分钟因果路径 |
| M5 | Sweep 后 5 分钟 | M3 信息 + 完整 5m 释放与 R12 前 5 分钟路径 |
| M10 | Sweep 后 10 分钟 | M5 信息 + R12 前 10 分钟路径；仅作延迟对照 |

重要因果修正：R09 的 `stop_release_score`、`high_stop_release_label` 与 5m 释放字段并非 Sweep 瞬间全部可见。R13 只允许它们进入 M5/M10；15m 释放字段在 R13 v1 全部禁用。绝对 ETH 价格、全局 Bar 位置、事件 ID、未来路径和最终结果均不进入模型。

## 4. 标签与成本

### M3/M5/M10 主标签

- 路径：R12 自然结构止损、2R 目标、最长 180 分钟。
- Long 正标签：在 180 分钟观察窗内 **2R先于自然结构止损**，且 `long_net_1x_r >= +0.25R`。
- Short 正标签：在 180 分钟观察窗内 **2R先于自然结构止损**，且 `short_net_1x_r >= +0.25R`。
- 自然结构止损先触发：负标签。
- `TIME` / `INVALID`：未解析标签，退出训练、校准和交易选择；180 分钟只是标签观察窗，不是最终策略的强制时间平仓。
- 1x 成本：开平合计约 13bp（11bp 手续费 + 2bp 滑点）。
- 2x 成本压力：26bp。

### M0 参考标签

M0 沿用 R09 的对称 TP15/SL15 路径，仅作为 Sweep 瞬间可分性的参考，不与 M3/M5/M10 的自然止损 2R 标签混为同一种交易。若同一 1m Bar 同时触及 TP 与 SL，因先后顺序不可知，该事件 Long/Short 标签都置为缺失并退出训练与选交易。

## 5. 冻结时间切分

- TRAIN：2023-01-01 至 2024-12-31。
- VALIDATION：2025-01-01 至 2025-09-30。
- HOLDOUT：2025-10-01 至 2026-06-30。

禁止随机切分。预处理器、缺失值填充、类别编码和模型仅在 TRAIN 拟合；交易分数阈值仅在 VALIDATION 冻结；HOLDOUT 只做最终评价，不用于模型、特征、阈值或超参数选择。

## 6. 固定模型

- 正则化 Logistic Regression：线性可分性基准。
- 浅层 HistGradientBoosting：有限非线性交互的主要模型。

R13 v1 不运行模型网格，不使用 LightGBM/XGBoost/CatBoost，不使用 LSTM、Transformer 或强化学习。HistGradientBoosting 的深度、叶子数、最小叶节点样本和迭代次数已冻结。

## 7. 模块消融

消融顺序固定，只有前一模块在 TRAIN / VALIDATION / HOLDOUT 三段覆盖率均达到 80%，才允许继续加入下一模块：

| 编号 | 特征集 |
|---|---|
| A | R09 因果结构与当时可见释放信息 |
| B | A + R12 扫后动态状态 |
| C | B + 1 秒 Trade Bar |
| D | C + r0020 Range Bar |
| E | D + r0020 step1 Range Footprint |
| F | E + Binance 5m OI 上下文 |

如果某一模块覆盖不足，后续累计消融全部阻断，避免“缺失数据发生在哪个时期”成为模型信号。

### 1 秒 Trade

按事件窗口提取聚合统计，不把逐秒序列直接展开成数百维：成交强度、Buy/Sell Notional、Delta、极端 1 秒成交、单位卖量价格冲击、冲击效率变化以及扫前/扫后阶段对比。只使用 `bar_time < decision_time` 的已完成 1 秒 Bar。

### r0020 Range

固定使用 r0020，不从 r0015/r0025 中挑最好结果。提取最近 1/3/5 个已完成 Range Bar 的方向、完成时间、成交量、Delta、上下行效率，以及最近两次下行 Range 的冲击/持续时间变化。

### Range Footprint

复用公共 Footprint 因果上下文，提取价格档位上的主动买卖、低点附近大单卖出集中度、Delta、POC/价值重心和单位卖量价格冲击。仅保留 `fp_` 因果特征。

### OI

使用 Binance 5m OI 代理，并强制 1 分钟发布延迟。OI只作为去杠杆、新增空头和仓位流状态的上下文，不作为硬条件。

### Books

Books 历史仅覆盖约 2025-11 至 2026-06，不能进入长期主模型。R13 v1 明确排除 Books；未来只能作为冻结主模型分数之上的独立短覆盖增量研究。

## 8. 交易选择

模型分别输出 Long 与 Short 盈利概率。阈值由 VALIDATION 的 90% / 95% / 98% 分位冻结，95% 为主要门槛：

- Long 超阈值且相对 Short 更强：LONG。
- Short 超阈值且相对 Long 更强：SHORT。
- 两者都不够，或没有可执行标签：SKIP。

模型不允许强迫每个 Sweep 交易。

## 9. 晋级门禁

主要 HGB + 95% 阈值必须同时满足：

- Holdout 交易数不少于 150；
- Holdout 1x 成本后平均净 R > 0；
- Holdout PF >= 1.30；
- Holdout 2x 成本后平均净 R > 0；
- Holdout 正收益月份比例 >= 70%；
- 去掉前 10 大盈利后累计净 R 仍为正；
- Validation 1x 成本后平均净 R > 0。

任何一项失败均 `rejected`，不允许从多个检查点、模型或消融中事后挑最好看的结果。

## 10. 运行命令

### 自检

```bat
python research\liquidity\13_post_sweep_supervised_meta_labeling_study.py --self-test --no-progress
```

### 小样本端到端门禁

```bat
python research\liquidity\13_post_sweep_supervised_meta_labeling_study.py --symbol ETH-USDT-SWAP --start-date 2023-01-01 --end-date "2026-06-30 23:59:59" --max-events 1000 --skip-review-pack
```

`--max-events` 会在 TRAIN / VALIDATION / HOLDOUT 中分层抽样，不会只取最早的 1000 个事件。

### 全量研究

```bat
python research\liquidity\13_post_sweep_supervised_meta_labeling_study.py --symbol ETH-USDT-SWAP --start-date 2023-01-01 --end-date "2026-06-30 23:59:59"
```

首次全量运行会分块构建外部数据模块并写入：

```text
data\reports\research\liquidity\13_post_sweep_supervised_meta_labeling_r13\cache\full
```

后续重复运行会复用缓存。数据或提取逻辑变化后使用：

```bat
python research\liquidity\13_post_sweep_supervised_meta_labeling_study.py --symbol ETH-USDT-SWAP --start-date 2023-01-01 --end-date "2026-06-30 23:59:59" --rebuild-feature-cache
```

### 只跑基础模型

用于快速验证 R09/R12 数据链路，不代表最终模型：

```bat
python research\liquidity\13_post_sweep_supervised_meta_labeling_study.py --symbol ETH-USDT-SWAP --start-date 2023-01-01 --end-date "2026-06-30 23:59:59" --disable-trade-1s --disable-range --disable-footprint --disable-oi
```

## 11. 输出目录与重点报告

```text
data\reports\research\liquidity\13_post_sweep_supervised_meta_labeling_r13
```

- `01_data_quality.csv`：独立事件、检查点与模块基础质量。
- `02_frozen_design.csv`：冻结的切分、标签、模型与消融设计。
- `03_source_dataset_audit.csv`：M0/M3/M5/M10样本量与因果可用性。
- `04_module_coverage.csv`：各模块在每个检查点和分区的覆盖率。
- `05_module_build_audit.csv`：分块读取和模块构建状态。
- `06_model_classification_summary.csv`：AUC、PR-AUC、Brier等分类指标。
- `07_high_score_trade_selection.csv`：验证与Holdout高分交易的真实成本表现。
- `08_ablation_incremental_value.csv`：每个新增数据模块的财务增量。
- `09_score_decile_monotonicity.csv`：模型分数与实际净收益是否单调。
- `10_candidate_scorecard.csv`：最终 `promote_to_backtest` / `rejected` 与失败门禁。
- `11_feature_contract.csv`：每个模型实际保留、丢弃的特征与原因。
- `12_causal_audit.csv`：来源时间、释放字段可用时间、分组切分和阈值冻结门禁。
- `13_prediction_sample.csv.gz`：验证/Holdout预测样本。
- `14_model_dataset_sample.csv.gz`：小型安全数据样本，不包含完整大表。
- `15_research_brief.md`：自动结论。
- `gpt_review_pack.zip`：供后续审阅。

## 12. 解释结果时的优先顺序

1. 先看 `01_data_quality.csv` 与 `12_causal_audit.csv` 是否全部通过。
2. 再看模块三段覆盖率，覆盖不足的消融不得解释为失败或成功。
3. 看 `08_ablation_incremental_value.csv`，判断 Trade、Range、Footprint、OI 是否在严格 Holdout 中产生真实成本后的增量。
4. 看 `09_score_decile_monotonicity.csv`，高分组净收益必须总体更高，而不能只提升 AUC。
5. 最后看 `10_candidate_scorecard.csv`。只有通过全部门禁才进入正式策略回测。

如果全部数据模块加入后，最高分 Holdout 交易仍无法覆盖 13bp/26bp 成本，应结束这条 Liquidity Sweep 模型主线，而不是继续更换更复杂模型。
