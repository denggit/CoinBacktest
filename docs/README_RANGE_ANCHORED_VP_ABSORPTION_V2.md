# ETH Range Anchored VP Absorption Breakout V2

## 关键修正

V2 不是修 V1，而是按新的逻辑链路重写：

```text
lower high broken
-> 创建 anchored profile
-> 回看/继续观察 VAL 到 lower POC 区间内的 sell bubble absorption
-> absorption zone 上方 buy stop
-> lower POC 下方止损
-> VAH 止盈
```

V2 不做动态止损，只验证基础入场/止损/止盈逻辑。

## 运行命令

```bash
python backtest/mf/eth_range_anchored_vp_absorption_v2_backtest.py \
  --start-date 2023-01-01 \
  --end-date 2026-06-15 \
  --warmup-start-date 2022-01-01 \
  --preset high \
  --range-pct 0.002 \
  --price-step 1 \
  --out-dir data/reports/mf/range_avp_abs_v2_full_high
```

## 请发回

```text
eth_range_anchored_vp_absorption_v2_summary.json
eth_range_anchored_vp_absorption_v2_event_counts.csv
eth_range_anchored_vp_absorption_v2_setup_events.csv
eth_range_anchored_vp_absorption_v2_signal_audit.csv
eth_range_anchored_vp_absorption_v2_trades.csv
ETH_Range_AnchoredVP_AbsorptionBreakout_V2_*.txt
```

先看 `event_counts.csv`，确认漏斗是否合理：

```text
lower_high_broken_profile_created
active_waiting_for_absorption
buy_stop_placed
entry_triggered
reject_order_low_rr
cancel_pending_break_lower_poc
```

## 与 V1 的区别

- V1：先创建 profile，再等 lower high break，逻辑顺序有问题。
- V2：先识别 lower high broken，再冻结 anchored profile。
- V2 的 profile 从上涨起涨 swing low / base 开始，到 lower high broken 当前 bar 结束。
- V2 的 absorption 可以在 lower high broken 前已经发生；策略会在 break 后回看过去已经发生的数据，不算未来函数。
- V2 的大红气泡不要求价格完全不再下跌，而是看 sell_notional / sell impact 很高，同时不跌破 lower POC。
- V2 不做动态止损，目标先验证基础逻辑。

## 2026-06-20 性能优化版说明

本包里的回测逻辑没有改变，只做了等价提速：

1. footprint 表只排序一次，并用 `np.searchsorted` 按 `bar_id` 切 anchored profile 窗口，避免每次 profile 都全表扫描 400 万+ 行 footprint。
2. absorption bubble 扫描由逐行 `itertuples -> Series` 改成向量化布尔筛选，规则保持一致。
3. 保留原有 profile cache、事件漏斗和输出文件名。

建议默认不要开 `--write-full-audit`，除非只跑小样本调试；完整 audit 在 50 万+ range bars 上会非常大。

## 进度日志版说明

回测会在 signal generation 和 execution/backtest 两个阶段分别每 3 个月打印一次进度，例如：

```text
[ETH_Range_AnchoredVP_AbsorptionBreakout_V2][signals] completed_to=2023-04-01 bar=... events=... signals=... active=... pending=...
[ETH_Range_AnchoredVP_AbsorptionBreakout_V2][backtest] completed_to=2023-04-01 bar=... trades=... capital=... open_position=...
```

这只增加日志，不改变任何信号、成交、止损、止盈或参数逻辑。

## fast2 版说明

在 progress/optimized 版基础上，又把 `add_features` 里的 max-sell footprint bucket 计算从 pandas `groupby().idxmax()` 改成按已排序 `bar_id` 的 numpy 分段扫描，并增加特征阶段日志：

```text
[features] start add_features
[features] rolling sell quantiles...
[features] max sell footprint bucket per range bar...
[features] max sell bucket progress groups=...
[features] finished add_features
```

这个改动只改变计算实现，不改变任意策略条件、窗口、时序、profile、bubble、入场、止损或止盈规则。

## fast3 / start-gated monthly progress 版说明

本版解决两个问题：

1. 进度日志从 `start_date` 开始按 1 个月打印，不再从 `warmup_start_date` 的 2022 年开始按 3 个月打印。
2. signals 主循环不再从 warmup 数据最开头全量扫描，而是从 `start_date` 前的必要缓冲区开始扫描。

缓冲区长度为：

```text
max(max_profile_bars + absorption_scan_back_bars + bubble_quantile_window + 50, 2000 bars)
```

这仍然只使用过去数据，不会引入未来函数；作用是保留 `start_date` 附近所需的 pivot/profile/absorption 上下文，同时避免把完整 2022 年都当成可交易状态机去跑。策略核心条件、入场、止损、止盈和参数不变。

## fast4 版说明

本版继续只做等价算法优化，不改变策略逻辑、参数、时序或信号定义。

优化点：

1. `_find_latest_lh_break()` 不再每根 bar 都用多次全列表推导扫描全部历史 pivot。
   - 先做 `close > lower_high` 的等价短路。
   - 对前一个更高 pivot high 使用倒序扫描，找到即停止，等价于原逻辑的 `prev_highs[-1]`。
   - 对 pivot lows 使用 `bisect` 只扫描 `major_high_i < pivot_low < i` 的相关区间。
2. signals 主循环额外每 25,000 根 bar 打一次 `bar_progress`，月度进度日志仍保留。

这不会引入未来函数：pivot 仍然需要右侧确认；lower-high break 仍然只在当前 bar close 后确认；anchored profile 仍然冻结在 break bar；buy stop 仍然只能后续 bar 生效。

## fast5 DB-backed profile / low-memory 版说明

本版仍然不改变策略条件、参数、时序、入场、止损、止盈或手续费。优化点集中在数据访问架构：

1. 不再把 400 万+ footprint 全量加载进内存。
2. 新增 `FootprintStore`，用 SQLite 根据明确的历史 `bar_id` 区间计算 anchored profile：
   - `WHERE bar_id >= start_bar_id AND bar_id <= end_bar_id`
   - `GROUP BY price_bucket`
3. 每根 range bar 的 max-sell bubble bucket 从 SQLite 读取，不再用 pandas 持有完整 footprint 表。
4. 保留 fast4 的 lower-high-break 等价优化、月度 progress、25,000 bar progress。

反未来函数说明：

- SQLite 查询窗口仍由状态机传入的历史 `start_bar_id/end_bar_id` 决定。
- `end_bar_id` 仍然是 lower-high-break 当前 bar，不会超过当前时刻。
- absorption/buy-stop/stop/target 逻辑不变。
- 该版本是 DB-backed replay，不是参数优化，也不是信号逻辑优化。

注意：第一次运行时如 DB 没有索引，会自动执行 `CREATE INDEX IF NOT EXISTS`，这是 SQLite 元数据优化，不改变任何行情数据。
