# Follower-Friendly Strategy Factory V1.3

这个目录是 **CoinBacktest 里的独立研究引擎**，用于寻找更适合带单展示的 ETH 永续合约子策略。

边界：

- 不加入 V10B portfolio。
- 不修改任何现有 V10B / V10A 文件。
- 不接 AetherEdge，不做实盘交易。
- 只输出候选策略、逐笔 MFE/MAE、因果审计、压力测试、排行榜。

核心目标不是找“单独收益最高”的策略，而是找：

- 交易次数更多；
- 胜率更舒服；
- 持仓更短；
- 连续亏损更短；
- MFE/MAE 结构可优化；
- fee/slippage/delay 压测后不死；
- 后续可以作为独立 engine 再考虑加入 portfolio。

## V1.3 重点变化

- 数据加载改为直接使用 `src.data_feed.OKXDataLoader`，对齐 `research/ohlcv_breakout_event_study_lab.py` 的数据入口习惯。
- 高周期 context 对齐改为复用 `src.research_common.event_study.causal_align_context`。
- 进度条默认开启，不再提供关闭/间隔参数。
- `--max-specs` 改为 family 间 round-robin 均衡采样，避免再次出现 3000 个全是某一个 family。
- 新增均值回归方向：failed breakout、range boundary、Bollinger/VWAP extreme、trend pullback reversion。

## 运行命令

快速 sanity：

```bash
python research/follower_friendly_strategy_factory/factory_v1.py --fast
```

跑 3000 个候选并写出 Top trades：

```bash
python research/follower_friendly_strategy_factory/factory_v1.py --max-specs 3000 --write-top-trades
```

更偏带单胜率，可以先只跑均值回归和趋势回踩：

```bash
python research/follower_friendly_strategy_factory/factory_v1.py --families trend_pullback_continuation,vwap_reversion_regime --max-specs 3000 --write-top-trades
```

Windows / Unix 都可以直接运行，不需要 `cd` 以外的 shell 特性。

进度条默认开启，不需要额外参数；批量研究时会自动显示 factory / stress 进度。


## 速度设计

V1.3 已经做了第一轮提速，并修复了 max_specs 截断导致 family 覆盖不足的问题，适合先跑 5m/15m 独立策略工厂：

- 所有 OHLCV 只加载一次；
- primary/context 指标只预计算一次；
- 高周期 context 只做一次 causal merge；
- 每个 spec 只生成一个 vectorized signal；
- 回测核心使用 numpy 数组事件循环，不再 per-spec 复制完整 DataFrame；
- equity 只在资金变化时记录，不再每根 bar 写一行；
- 默认只抽样写 replay audit，避免全量 trades 输出爆炸；
- 压测默认只跑 scoreboard Top 1，避免第一次大批量研究被 stress 阶段拖慢；需要扩大复核时再显式设置 `--stress-top-n 10` 或更高。

仍然没有做的重优化：

- 没有把多 spec 合并成矩阵批量执行；
- 没有强依赖 numba/C++，保持本地环境兼容；
- 暂时不直接扫 tick/trades 原始逐笔数据，建议先使用已聚合好的 1m/5m/15m/range/footprint 特征。

如果要跑 1 万以上 spec 或 trade/range-footprint 重数据，下一版应该做 batch runner / numba optional / 分片并行 / 特征缓存。


## 当前数据源范围

V1.3 当前只使用普通 OKX OHLCV K 线加载器：

```text
--data-source ohlcv
```

也就是说，它现在不是 trade-bar / range-bar / footprint 策略工厂。

原因是 trade bar、range bar、footprint 属于更重的数据源，不能只是把列硬塞进现有 K 线策略里；必须做单独 data adapter、feature schema、available_time 审计和重数据缓存，否则很容易又慢又有隐蔽时序问题。

下一版建议做：

```text
V2 data adapters:
1. trade_bar context：1m/5m trade 聚合 K + CVD/delta/taker ratio/large trade
2. range_bar confirmation：range bar 只做确认/过滤/add trigger，不做 primary 开仓主轴
3. footprint context：只作为增强确认，先验证是否真的提升 MFE/MAE
```

## 输出文件

默认输出目录：

```text
data/reports/research/follower_friendly_strategy_factory_v1_3
```

主要文件：

- `00_factory_meta.json`：参数、commit、因果对齐说明。
- `01_spec_manifest.csv`：所有策略规格。
- `01_family_counts.csv`：本轮 family 覆盖数量，防止 max_specs 只跑到单一策略族。
- `02_base_summary.csv`：基础回测汇总。
- `03_base_yearly.csv`：年度拆分。
- `04_stress_summary.csv`：fee_2x、slippage_2x、delay_1bar 压测。
- `05_stress_yearly.csv`：压力测试年度拆分。
- `06_scoreboard.csv`：带单友好分数排行榜。
- `07_replay_audit_sample.csv`：逐笔审计样本。
- `08_top_candidate_trades_with_mfe_mae.csv`：Top 候选逐笔 MFE/MAE，需要 `--write-top-trades`。

## 重要时序规则

V1 默认：

```text
closed primary bar 生成信号
next primary bar open 执行
higher timeframe context 使用 available_time 对齐
```

高周期 context 不按 bar start 直接 ffill，而是：

```text
context_available_time = context_bar_start_time + timeframe_delta
merge_asof(direction="backward")
```

输出的 replay audit 会检查：

- `context_available_time_flag`
- `entry_not_next_open_flag`
- `entry_price_mismatch_flag`
- `same_bar_stop_tp_both_hit_flag`

## 已排除的已知坑

V1 不重复生成这些旧方向：

- 1m 裸 EMA/VWAP/momentum/突破；
- range bar 作为 primary 开仓轴；
- 非 causal 5m 左标签高周期 context；
- 只看单点收益、不过压力测试的候选。

## 下一步研究方式

先看 `06_scoreboard.csv`，再看 Top 候选的 `08_top_candidate_trades_with_mfe_mae.csv`。

判断逻辑：

- 如果 `avg_mfe_r` 高但最终亏，优先改退出结构，而不是否定入场信号。
- 如果 `avg_mae_r` 绝对值很大，说明入场位置差或止损太窄。
- 如果 base 好但 `fee_2x/slippage_2x/delay_1bar` 死，不能进实盘候选。
- 如果只有一年赚钱，不能进实盘候选。
- 如果 `same_bar_stop_tp_both_hit_flag` 很多，说明 OHLC 路径不确定，必须更保守复核。
