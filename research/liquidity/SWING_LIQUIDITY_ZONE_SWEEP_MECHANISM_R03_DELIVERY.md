# Swing Liquidity Zone Sweep Mechanism R03

## 研究定位

R03 不是最终策略回测，也不在本阶段使用 Books、CVD、Footprint 或冰山订单。
本版先回答三个问题：

1. 哪类尚未消费的 Swing Low Zone 被首次扫破后更容易形成反转？
2. Sweep 深度是否应按扫前波动率归一化，而不是只看绝对 bp？
3. Sweep 后的 MFE、MAE、结构低点存活时间和多日路径，是否支持非纯时间出场？

## 核心实现

- 沿用 R02 的 15m、30m、1H、4H、1D 因果 Swing Low 活跃池。
- 仅使用事件时已经确认的 Swing 信息；高阶最终确认结果不进入特征。
- 将同一根已关闭 1m bar 内、价格接近的多个活跃 Swing Low 聚合成一个 Zone。
- 主口径 Zone 合并距离为 10bp，并同时输出 5/10/25/50bp 敏感性。
- 使用只依赖当前及过去信息的在线 impulse 去重，避免同一轮下跌重复发出大量信号。
- 入场路径统一从 Sweep bar 关闭后的下一根 1m open 开始。
- Sweep 深度同时输出绝对 bp 和相对扫前 1H/4H/1D ATR 的比例。
- 路径覆盖 5m 至 3 天，输出 MFE、MAE、close return、结构低点首次再破时间、结构存活率、TP 在结构再破前是否达成。
- 构造同月份、同波动状态、同前置跌幅、但未扫到任何活跃 Swing Zone 的匹配对照组。
- 未来结果与当下特征分离为 `model_feature_table` 和 `model_label_table`，防止未来标签污染模型输入。

## 本阶段明确不做

- 不将未来 reclaim 作为入场条件。
- 不使用 Books、CVD、Range Footprint、冰山订单或吸收识别。
- 不训练最终模型。
- 不优化止盈止损。
- 不输出可实盘收益结论。

## 主要报告

- `02_zone_construction_sensitivity.csv`：Zone 合并距离敏感性。
- `03_same_bar_zone_event_table.csv`：同 bar Zone 全事件。
- `04_online_first_zone_feature_table.csv`：在线去重后的首事件特征。
- `05_matched_control_feature_table.csv`：匹配的非 Swing Sweep 对照。
- `06_path_horizon_summary.csv`：5m 至 3 天路径汇总。
- `07_structural_exit_summary.csv`：结构低点存活及 TP-before-lower-low。
- `08_control_comparison.csv`：Zone Sweep 与普通下跌对照。
- `09_control_match_balance.csv`：匹配前置状态平衡检查。
- `10_zone_attribute_path_bins.csv`：低点质量、年龄、周期、Sweep 深度、ATR 归一化深度等分层。
- `12_model_feature_table.csv`：仅事件时可知特征。
- `13_model_label_table.csv`：未来路径标签。
- `14_causal_audit.csv`：因果与表关联审计。
- `gpt_review_pack.zip`：用于后续分析。

## Windows 运行命令

```bat
python research\liquidity\03_swing_liquidity_zone_sweep_mechanism.py --symbol ETH-USDT-SWAP --warmup-start-date 2022-01-01 --start-date 2023-01-01 --end-date "2026-06-30 23:59:59" --swing-timeframes 15m,30m,1H,4H,1D --confirmation-orders 1,2,3,5 --data-source trade_bar --no-build-missing
```

默认报告目录：

```text
data\reports\research\liquidity\swing_liquidity_zone_sweep_mechanism_r03
```

## 验证结果

- R03 self-test：通过。
- R02 + R03 + Review Pack + Liquidity Loader 相关测试：34 passed。
- 新模块 compileall：通过。
- 新脚本只 import `src.*`，没有 import 其他研究脚本。
- 真实项目数据工程冒烟：159,839 根 1m bar，4,606 个 Level Sweep 聚合为 2,056 个 10bp Zone，在线去重后 1,539 个首事件，匹配对照 1,333 个；因果审计 0 违规。

冒烟数据仅用于工程验证，不能作为正式研究结论。正式结论必须以完整 2023-01-01 至 2026-06-30 报告为准。
