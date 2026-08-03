# 10 Structured Pullback Entry Study R10

## 1. 研究目标

R10 不再等待 Higher Low 被扫后猜测最终最低点，而是验证另一类结构交易：

```text
前置下跌 / 底部形成
→ 市场反弹
→ Higher Low 因果确认
→ 从确认时刻开始，在该 Higher Low 价格挂限价多单
→ 若以后回踩成交，止损放在更早的结构低点下方
```

本研究首先回答：

1. Higher Low 确认后，未来是否经常回踩并成交；
2. 未回踩而直接上涨的机会损失有多大；
3. 用更早结构低点止损后，H0、1R、2R、3R是否具有真实成本后的正期望；
4. 哪些可解释结构家族跨周期、跨时期更稳定。

R10 是事件级研究，不是最终资本约束组合回测。

## 2. 预声明假设

| ID | 假设 |
|---|---|
| B0 | 所有因果确认的 Higher Low 回踩基准 |
| P1 | 明显下跌后的第一个 Higher Low |
| P2 | 突破下降结构（BOS）后的第一个 Higher Low 回踩 |
| P3 | 等低点/双底层级上方形成的 Higher Low |
| P4 | 启动过强势上行位移的 Higher Low |
| P5 | 低位盘整突破后的 Higher Low 回踩 |
| P6 | 挂单生效时已经与其他周期活跃低点汇聚的 Higher Low |
| P7 | 成熟上涨趋势中的中继 Higher Low |
| P8 | 假跌破旧低点并收回后形成的 Higher Low |

家族允许重叠，但分别报告。R10 不搜索家族组合，也不做参数网格。

## 3. 因果交易时序

- `structure_available_time` 之前不得挂单；
- Higher Low 只有右侧确认完成后才成为候选；
- 限价单从 `structure_available_time` 对应的第一根可执行 1m Bar 开始有效；
- 挂单价格为已确认 Higher Low 的价格；
- 下一次同周期 Swing Low 因果可用时，旧挂单撤销；
- 不把已经发生过的 Higher Low 最低价事后回填成成交；
- 若 Bar 开盘低于限价，按更优的开盘价成交；
- 若限价在 Bar 内成交，该 Bar 在成交前可能出现的 High 不得用于止盈或 MFE；
- 同一 Bar 同时触发止盈和止损时，保守按止损处理。

## 4. 止损与目标

### 止损

- P3、P5：止损锚点为前两个结构低点中的更低者；
- 其他家族：止损锚点为前一个 Swing Low；
- 最终止损价为锚点下方 5bp；
- 研究使用固定风险 R 归一化，不使用固定张数。

### 目标

- `H0`：Higher Low 形成前的反弹高点；
- `R1`：1R；
- `R2`：2R；
- `R3`：3R。

## 5. 成本

- 手续费基准：开平合计 0.11%；
- 现实成本：0.11%手续费 + 开平合计2bp滑点；
- 压力成本：现实成本的2倍。

## 6. 性能设计

- R09结构特征表直接复用，不逐事件重算多周期结构；
- 同一个 Level 的限价成交只搜索一次，再映射到重叠家族；
- 使用预构建区间阈值索引寻找首次触价、首次止盈和首次止损；
- 使用区间最值索引计算 MFE/MAE；
- 多周期汇聚在挂单生效时因果计算，不使用未来 Sweep 时的汇聚状态；
- 不加载 Raw Trades、Footprint或Books。

## 7. 运行命令

### 小样本门禁

```bat
python research\liquidity\10_structured_pullback_entry_study.py --symbol ETH-USDT-SWAP --start-date 2023-01-01 --end-date "2026-06-30 23:59:59" --max-candidates 1000 --skip-review-pack
```

### 全量研究

```bat
python research\liquidity\10_structured_pullback_entry_study.py --symbol ETH-USDT-SWAP --start-date 2023-01-01 --end-date "2026-06-30 23:59:59"
```

若 R02/R09缓存缺失：

```bat
python research\liquidity\10_structured_pullback_entry_study.py --symbol ETH-USDT-SWAP --start-date 2023-01-01 --end-date "2026-06-30 23:59:59" --rebuild-r02-if-missing --rebuild-r09-if-missing
```

## 8. 输出目录

```text
data\reports\research\liquidity\10_structured_pullback_entry_r10
```

主要文件：

```text
00_manifest.json
01_data_quality.csv
02_hypothesis_definitions.csv
03_candidate_funnel_summary.csv
04_family_geometry_summary.csv
05_family_target_outcome_summary.csv
06_family_timeframe_summary.csv
07_period_stability.csv
08_fill_age_summary.csv
09_family_overlap.csv
10_family_target_scorecard.csv
11_causal_audit.csv
13_event_sample.csv
14_candidate_feature_table.csv.gz
15_family_candidate_execution_plan.csv.gz
16_trade_outcome_table.csv.gz
17_research_brief.md
gpt_review_pack.zip
```

## 9. 重点查看指标

- `candidate_rows`：结构候选数量；
- `fill_rate`：确认后实际回踩成交率；
- `h0_before_fill_rate`：还未回踩就先上涨到H0的错过率；
- `median_order_age_minutes_to_fill`：挂单到成交等待时间；
- `median_actual_risk_bp`：结构止损宽度；
- `tp_before_sl_rate`；
- `mean_net_r_realistic`；
- `profit_factor_net_r_realistic`；
- `mean_net_r_2x_cost`；
- 去除前10大盈利后的期望；
- 三个时期和15m/30m/1H/4H/1D分层稳定性。

## 10. 自动筛选边界

自动标记 `promote_to_backtest` 只代表值得进入下一阶段候选回测，仍不能声明可实盘。候选还需要：

- 资本并发和重叠信号处理；
- 完整权益曲线、MDD和连续亏损；
- 限价成交与滑点压力；
- 延迟、撤单和订单生命周期；
- Portfolio overlay；
- 实盘Replay审计。

## 11. 已完成测试

- R10专项测试；
- R02–R10相关回归；
- Self-test；
- FutureWarning / DeprecationWarning作为错误；
- 编译检查；
- 微秒/纳秒时间精度回归；
- 新增文件 import boundary 检查。

仓库全量 import-boundary 工具仍会报告基线中既有的 Swing Low research 互相导入问题；R10新增违规为0。


## v1.0.2 Fix

- Fixed `attach_trade_outcomes` under pandas Copy-on-Write, where `Series.to_numpy()` can return a read-only mask and an in-place `&=` raised `ValueError: output array is read-only`.
- The validity mask now owns writable memory and is refined without mutating a pandas-backed view.
- Added a Copy-on-Write regression test.
