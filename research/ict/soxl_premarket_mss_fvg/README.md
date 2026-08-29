# SOXL ICT Premarket Sweep → MSS → FVG Research

## Goal

研究 `SOXL-USDT-SWAP` 是否存在以下可交易路径：

`04:00-08:30 NY 盘前流动性 -> 08:30-16:30 扫流动性 -> MSS -> displacement FVG -> FVG 回踩限价入场 -> 对侧盘前极值止盈`

本目录只做 CoinBacktest 研究，不包含实盘执行代码。

## Data boundary

- 唯一原始行情入口：`src.data_feed.okx_loader.OKXDataLoader`。
- 只请求/读取 1m K；2m / 5m / 15m 全部由研究内从 1m 聚合。
- research 目录禁止直接 HTTP、SQLite、ccxt 或交易所 endpoint。
- Loader 的项目本地时间戳按固定 `UTC+8` 还原，再转换为 DST-aware `America/New_York`。

## Frozen R01 semantics

- 盘前：纽约时间 `[04:00, 08:30)`。
- 交易：纽约时间 `[08:30, 16:30)`。
- 周末不交易；默认排除美国股票市场全天休市日。
- 流动性：盘前绝对 high/low；另比较加入单个最明显、因果确认的内部 15m swing high/low。
- 15m major swing：pivot left/right = 2/2，只允许使用 08:30 前已经完成右侧确认的 pivot；以局部 prominence 排名，不看收益挑选。
- sweep：盘中 1m completed bar 首次穿过冻结流动性价位。
- MSS：1m/2m/5m 各自用 causal 1/1 short-term pivot；R01 使用 sweep 时已经确认的最近反向 pivot 作为冻结结构线。
- displacement：MSS break bar body >= 前 20 根（当前 bar 排除）body median 的 1.5x，且 close 位于反转方向外侧 25%。
- FVG：严格三根 K；R01 基线要求 MSS break bar 本身就是 FVG 第三根。
- long limit = bullish FVG 第三根 low；short limit = bearish FVG 第三根 high。
- stop = sweep 后到 signal 完成时为止的最不利 1m 极值。
- target = 对侧盘前绝对 extreme。
- 未成交撤单：target 先到、stop extreme 先失效、或 16:30。
- 持仓最晚 16:30 平仓，不隔夜。
- 同一 1m 内顺序不可知时使用保守路径，不给策略乐观收益。

## Predeclared robustness

不是参数寻优：

- execution TF: `1m,2m,5m`
- liquidity: `extremes_only` vs `extremes_plus_major_swing`
- displacement: `1.25x / 1.50x / 1.75x`
- round-trip cost: `0.11% × 1 / 2 / 3`
- order activation delay: `0 / 1 / 2 min`

SOXL 永续历史很短，因此禁止针对这段样本继续网格调 pivot、FVG、session 或 RR。

## Windows command

```text
python research\ict\soxl_premarket_mss_fvg\01_premarket_sweep_mss_fvg_research.py --symbol SOXL-USDT-SWAP --start-date 2026-05-20 --end-date 2026-06-30
```

只用本地已有数据：

```text
python research\ict\soxl_premarket_mss_fvg\01_premarket_sweep_mss_fvg_research.py --symbol SOXL-USDT-SWAP --start-date 2026-05-20 --end-date 2026-06-30 --local-only
```

自测：

```text
python research\ict\soxl_premarket_mss_fvg\01_premarket_sweep_mss_fvg_research.py --self-test
```

## Output

默认报告目录：

`data/reports/research/ict/soxl_premarket_mss_fvg_r01/`

核心结果：

- `02_premarket_liquidity_levels.csv`
- `03_sweep_events.csv`
- `04_signal_attempts.csv`
- `05_base_trade_lifecycle.csv`
- `06_base_variant_summary.csv`
- `07_cost_stress.csv`
- `08_displacement_sensitivity.csv`
- `09_order_delay_stress.csv`
- `10_execution_timeframe_compare.csv`
- `11_liquidity_level_type_compare.csv`
- `12_weekday_compare.csv`
- `13_sweep_time_bucket_compare.csv`
- `15_causal_audit.csv`
- `16_findings.md`
- `gpt_review_pack.zip`

---

## R02 — corrected sweep-episode MSS model

Run:

```text
python research\ict\soxl_premarket_mss_fvg\02_sweep_episode_state_machine_research.py --symbol SOXL-USDT-SWAP --start-date 2026-05-20 --end-date 2026-06-30
```

R02 report root:

`data/reports/research/ict/soxl/mss/r02_state_machine/`

Key changes versus R01:

- sweep opens a stateful episode instead of freezing MSS structure immediately;
- terminal sweep extreme is continuously updated from completed 1m data;
- the relevant 1m/2m/5m short-term pivot dynamically follows that terminal structure;
- opposite premarket liquidity must still be fresh;
- weak 15m pivots are reported but not force-traded as "major" liquidity;
- diagnostic tables split liquidity modes, plus explicit Long/Short and dynamic-reference-source reports.

### Longer SOXL history

`src/data_feed/alpaca_stock_loader.py` provides a reusable historical US-equity minute-bar adapter for Alpaca.  Do not mix SOXL spot and OKX SOXL perpetual blindly.  First validate the overlapping period for price-path, premarket levels, sweep timing and R02 MSS/FVG signal agreement; only then use spot history as a research proxy.

---

## R03 — Alpaca SOXL spot ↔ OKX SOXL perpetual overlap audit

Before interpreting the 2019-2026 Alpaca spot history as evidence for the OKX
perpetual, run the structural proxy audit on the overlapping 2026 period:

```text
python research\ict\soxl_premarket_mss_fvg\03_spot_perp_overlap_audit.py --start-date 2026-05-20 --end-date 2026-06-30
```

Report root:

`data/reports/research/ict/soxl/mss/r03_spot_perp_overlap_audit/`

The audit compares, after clipping both sources to New York `04:00-16:30`:

- aligned 1m return correlation;
- daily rebased intraday path correlation;
- premarket extreme timing;
- external-liquidity sweep agreement;
- base R02 MSS/FVG setup agreement.

The PASS / CAUTION / FAIL proxy gates are fixed engineering gates and are not
optimized on strategy PnL.

### Long-history R02 on Alpaca split-adjusted SOXL

R02 now supports either data source while keeping the same state machine and
execution semantics. After the overlap audit, run the long-history proxy study:

```text
python research\ict\soxl_premarket_mss_fvg\02_sweep_episode_state_machine_research.py --data-source alpaca --alpaca-symbol SOXL --alpaca-feed sip --alpaca-adjustment split --start-date 2019-01-02 --end-date 2026-06-30 --local-only --out-dir data\reports\research\ict\soxl\mss\r02_state_machine_alpaca_2019_2026
```

The long-history report remains a **spot-proxy research result**. Even after an
overlap PASS, final promotion still requires validation on OKX perpetual overlap
and future/forward data.

### Alpaca no-trade minute handling

Alpaca stock aggregates may omit a minute when no eligible trade occurs. The long-history path therefore densifies only **internal** same-day gaps by carrying the last already-observed close forward with zero volume. It never backfills before the first trade or after the final observed trade of the day. This preserves causal 2m/5m/15m aggregation while keeping sparse/no-trade minutes auditable.

---

## R04 — ICT-faithful path MSS / displacement / FVG

R02's strict break-bar implementation is archived as an **incorrect operationalization**.
Do not interpret its zero long-history attempts as evidence that ICT has no edge.

R04 separates the concepts:

`fresh liquidity sweep -> terminal extreme -> short-term structural MSS -> strong reversal displacement leg containing FVG -> FVG retrace entry`

Key rule: the MSS candle does **not** have to be the displacement candle and does
**not** have to be FVG candle 3. FVG only needs to be inside the reversal leg
that delivers price through the MSS structure.

Long-history Alpaca command:

```text
python research\ict\soxl_premarket_mss_fvg\04_ict_path_mss_displacement_research.py --data-source alpaca --alpaca-symbol SOXL --alpaca-feed sip --alpaca-adjustment split --start-date 2019-01-02 --end-date 2026-06-30 --local-only --out-dir data\reports\research\ict\soxl\mss\r04_ict_path_alpaca_2019_2026
```

R04 reports remain under:

`data/reports/research/ict/soxl/mss/`

---

## R06 — remote unconsumed 1H / 4H / 1D liquidity

R06 keeps R05's ungated displacement-discovery semantics and expands the liquidity source.  Premarket liquidity and each higher-timeframe family are evaluated independently:

- `premarket_extreme`
- `major_15m_swing`
- `remote_1h_swing`
- `remote_4h_swing`
- `remote_1d_swing`

A HTF swing must be causally confirmed and still unconsumed at New York 08:30.  Every active level is retained; the research does not force a nearest-only assumption.  HTF age, distance/rank and exact-minute cross-timeframe confluence are diagnostic dimensions.

Long-history Alpaca command:

```text
python research\ict\soxl_premarket_mss_fvg\06_htf_unconsumed_liquidity_discovery.py --data-source alpaca --alpaca-symbol SOXL --alpaca-feed sip --alpaca-adjustment split --start-date 2019-01-02 --end-date 2026-06-30 --local-only --out-dir data\reports\research\ict\soxl\mss\r06_htf_unconsumed_liquidity_alpaca_2019_2026
```

Reports remain under `data/reports/research/ict/soxl/mss/`.

---

## R07 — ICT Semantic Gap Atlas

R07 deliberately does **not** add stricter entry filters.  It keeps the R06 broad causal universe and measures what may distinguish setups a discretionary ICT trader visually accepts from those a mechanical translation currently over-trades.

Primary new report tables:

- `24_semantic_feature_frozen_edges.csv` — discovery-through-2024 bin edges;
- `25_semantic_feature_atlas.csv` — frozen feature-response curves for discovery / 2025 / 2026;
- `26_semantic_category_atlas.csv` — V/direct vs post-terminal structure, FVG timing, Long/Short and other categorical semantics;
- `27_mfe_transition_atlas.csv` — what happens after trades first reach +0.5R/+1R/+2R/+3R;
- `28_entry_failure_vs_favorable_failure.csv` — immediate bad entries vs good initial moves that later fail.

Long-history Alpaca command:

```text
python research\ict\soxl_premarket_mss_fvg\07_semantic_gap_atlas.py --data-source alpaca --alpaca-symbol SOXL --alpaca-feed sip --alpaca-adjustment split --start-date 2019-01-02 --end-date 2026-06-30 --local-only --out-dir data\reports\research\ict\soxl\mss\r07_semantic_gap_atlas_alpaca_2019_2026
```

R07 is a semantic-discovery study, not a promoted trading strategy.  A semantic feature is eligible for later strategy work only if its response shape is interpretable and remains directionally consistent in the frozen 2025 forward and 2026 late holdout periods.

### R08 liquidity-consumption maturity

`08_liquidity_consumption_maturity_atlas.py` studies *how* liquidity is consumed before a valid MSS/FVG setup. It intentionally keeps fast spike-and-reclaim, shallow/equal-high-low probes, progressive sweeps and slower acceptance paths in the same broad universe. Default research window is 2023-07-01 through 2026-06-30; no maturity feature is an entry gate.

### R09 mechanism archetype validation + true EQH/EQL pools

`09_mechanism_archetype_validation.py` keeps the R08 broad universe and adds two research layers without making them entry gates:

- actual causal equal-high/equal-low swing pools from 1m/5m/15m pivots, replacing the coarse `near_touch_count` interpretation;
- overlapping mechanism tags learned from discovery-period feature distributions only (not from PnL), so fast rejection, sustained consumption, deep/progressive flush and EQH/EQL stop-run paths can coexist.

Default recent-liquidity run:

```text
python research\ict\soxl_premarket_mss_fvg\09_mechanism_archetype_validation.py --data-source alpaca --alpaca-symbol SOXL --alpaca-feed sip --alpaca-adjustment split --start-date 2023-07-01 --end-date 2026-06-30 --local-only --out-dir data\reports\research\ict\soxl\mss\r09_mechanism_archetype_validation_alpaca_2023_2026
```

Key new outputs:

- `05_equal_high_low_pool_catalog.csv`
- `34_mechanism_distribution_edges.csv`
- `35_mechanism_archetype_atlas.csv`
- `36_mechanism_combination_atlas.csv`
- `37_equal_pool_performance_atlas.csv`
- `38_opportunity_preservation_audit.csv`

R09 does not promote any R08 bucket into a strict strategy rule.  The goal is to identify mechanism shapes that remain directionally stable in 2025 and 2026 while preserving useful trade frequency.

### R10 — Multi-Timeframe Structural Trade Management Atlas
`10_trade_management_atlas.py` freezes the R09 entry universe and studies post-fill management only. It builds causal 1m/2m/5m/15m ST and IT structure, compares internal structural partials, a cost-cover ITH/ITL partial, and 2m/5m/15m structure runners after the original opposite-liquidity target. Reports remain under `data/reports/research/ict/soxl/mss/`.

### R11 — Entry Opportunity Expansion Atlas

`11_entry_opportunity_expansion_atlas.py` expands the entry research without tightening the existing Sweep -> MSS -> FVG gate.  It adds causally confirmed intraday 15m swing liquidity, including the specific case where both original premarket sides have already been consumed, then compares a local 50% equilibrium target with the opposite fresh 15m swing.

On each frozen MSS setup R11 also compares FVG near-edge, FVG 50% CE and two explicitly-labelled quantitative Order Block proxy retracements.  EQH/EQL remains an ordinary liquidity/context family rather than a special required condition.

Default recent-liquidity run:

```text
python research\ict\soxl_premarket_mss_fvg\11_entry_opportunity_expansion_atlas.py --data-source alpaca --alpaca-symbol SOXL --alpaca-feed sip --alpaca-adjustment split --start-date 2023-07-01 --end-date 2026-06-30 --local-only --out-dir data\reports\research\ict\soxl\mss\r11_entry_opportunity_expansion_alpaca_2023_2026
```

Key outputs include the intraday 15m swing/sweep catalogs, entry-model atlas, local target atlas, premarket-consumption-state atlas, swing-strength atlas and base-universe preservation audit.

### R12 — Structure Hierarchy + FVG Train Semantic Alignment

`12_structure_hierarchy_fvg_train_atlas.py` is a manual-semantics alignment study built around the 2026-08-05 replay examples. It separates early 04:00-08:30 and late 08:30-09:30 session liquidity, preserves all causal low-timeframe swing candidates with continuous visibility scores, allows multiple MSS attempts per sweep, and records the complete FVG train associated with the actual break impulse.

It compares uncapped FVG entries, broken-swing +/-0.10 entry caps, break-middle-FVG +/-0.10 caps, and close-break next-open market entry without promoting any one model to a final rule. The default golden-date export is `2026-08-05` so OKX and Alpaca semantic replay can be checked bar-by-bar before the next full strategy iteration.

Recent Alpaca run:

```text
python research\ict\soxl_premarket_mss_fvg\12_structure_hierarchy_fvg_train_atlas.py --data-source alpaca --alpaca-symbol SOXL --alpaca-feed sip --alpaca-adjustment split --start-date 2023-07-01 --end-date 2026-08-14 --local-only --out-dir data\reports\research\ict\soxl\mss\r12_structure_hierarchy_fvg_train_alpaca_2023_2026_08
```

Golden OKX semantic replay (recommended first):

```text
python research\ict\soxl_premarket_mss_fvg\12_structure_hierarchy_fvg_train_atlas.py --data-source okx --symbol SOXL-USDT-SWAP --start-date 2026-08-05 --end-date 2026-08-05 --local-only --golden-date 2026-08-05 --out-dir data\reports\research\ict\soxl\mss\r12_golden_okx_2026_08_05
```

### R14 — Executable Profitability Freeze

`14_executable_profitability_freeze.py` converts the R13 semantic atlas into a realistic one-account strategy study. It permits at most one setup per physical liquidity sweep, excludes micro pivots from independent execution, uses 1m/2m only for entries, and removes the old Swing +/- $0.10 gate. The first profit core does **not** require EQL/equal-like liquidity: it combines a 1m shallow/equal-like target leg with a separate 2m partial-consumed target leg, while deferring fresh/deep-target management to later expansion.

Recommended fast run reusing the already-generated R13 causal intermediates:

```text
python research\ict\soxl_premarket_mss_fvg\14_executable_profitability_freeze.py --data-source alpaca --alpaca-symbol SOXL --alpaca-feed sip --alpaca-adjustment split --start-date 2023-07-01 --end-date 2026-08-14 --local-only --r13-cache-dir data\reports\research\ict\soxl\mss\r13_semantic_consolidation_alpaca_2023_2026_08 --out-dir data\reports\research\ict\soxl\mss\r14_executable_profitability_alpaca_2023_2026_08
```

R14 reports actual setup/trade frequency, one-account overlap handling, PF/win rate/expectancy/payoff, compounded return/MDD, monthly/yearly results, cost/delay stress, top-winner removal and 1m/2m contribution. R14 is a candidate-freeze study; because 2026 has already been inspected, it is not described as untouched OOS validation.

### R15 — Daily Liquidity Traversal Path Atlas

`15_daily_liquidity_traversal_path_atlas.py` deliberately steps back from R14's narrow profit-core filters after the executable frequency collapsed from 8,354 physical sweeps to only ~0.28 selected setups/session.  R15 studies the **daily path first**, then asks where a causal entry could have been taken.

For every valid session R15 freezes four predeclared range interpretations without using PnL:

- 04:00-08:30 ET high/low, available from 08:30;
- 04:00-09:30 ET high/low, available from 09:30;
- the most prominent causally confirmed 15m swing-high/swing-low pair known by 08:30;
- the most prominent causally confirmed 15m swing-high/swing-low pair known by 09:30.

For each range it records the first raid side, penetration, reclaim, repeated raids, progress through 25/50/75/100% of the range, and whether price later reaches the opposite boundary.  The later path is an outcome label only; it is never fed back into MSS/FVG generation.

After the first raid, R15 attaches causal 1m/2m swing/MSS/displacement/FVG candidates with no equal-liquidity requirement, no target-consumption-state profitability gate, and no Swing +/- $0.10 cap.  Entry models include first-train FVG, break-middle FVG, closest-to-broken-swing FVG and close-break next-open market execution.  It reports how often each model actually captures completed daily traversals and what distinguishes successful from failed paths.

Recommended run:

```text
python research\ict\soxl_premarket_mss_fvg\15_daily_liquidity_traversal_path_atlas.py --data-source alpaca --alpaca-symbol SOXL --alpaca-feed sip --alpaca-adjustment split --start-date 2023-07-01 --end-date 2026-08-14 --local-only --out-dir data\reports\research\ict\soxl\mss\r15_daily_liquidity_traversal_path_atlas_alpaca_2023_2026_08
```

R15 is not a final strategy freeze.  Its purpose is to measure the real daily opportunity ceiling and recover the entry paths that R14 filtered away before another executable strategy is attempted.

### R16 — Entry Archetype Survival Atlas

`16_entry_archetype_survival_atlas.py` keeps R15's daily liquidity-path universe and changes the question from “does a sweep/MSS/FVG exist?” to **“where can a real order enter with the lowest immediate-stop probability while still capturing the daily path?”**

It compares reclaim market/retest entries, MSS next-open market, MSS FVG near/CE, a clearly-labelled quantitative Order Block proxy, OB-FVG overlap, and 2m-structure + 1m-FVG hybrid execution.  The old Swing +/- $0.10 cap is not used.  Pre-raid compression / three-wave approach is recorded only as a causal feature, never as a gate.

The lifecycle report explicitly measures stop within 1/3/5/10 minutes and whether 25/50/75/100% of the frozen dealing range is reached before stop.  This is the bridge to the later 50% cost-cover -> 75% partial -> opposite-liquidity 90% exit -> structural runner management study.

Recommended fast run reusing R15 causal intermediates:

```text
python research\ict\soxl_premarket_mss_fvg\16_entry_archetype_survival_atlas.py --data-source alpaca --alpaca-symbol SOXL --alpaca-feed sip --alpaca-adjustment split --start-date 2023-07-01 --end-date 2026-08-14 --local-only --r15-cache-dir data\reports\research\ict\soxl\mss\r15_daily_liquidity_traversal_path_atlas_alpaca_2023_2026_08 --out-dir data\reports\research\ict\soxl\mss\r16_entry_archetype_survival_atlas_alpaca_2023_2026_08
```

Key outputs:
- `06_entry_archetype_scorecard.csv`: fill rate, immediate-stop rates, 25/50/75/100 first-hit rates, PF/expectancy if fully exiting at each milestone;
- `07_entry_archetype_period_stability.csv`: same metrics across 2023H2-2024 / 2025 / 2026;
- `08_immediate_stop_vs_survivor_features.csv`: causal feature contrast for immediate stops versus 50%-survivors;
- `09_fixed_feature_stop_risk_atlas.csv`: predeclared feature bins, not PnL-fitted thresholds;
- `10_approach_compression_atlas.csv`: whether multi-push/three-wave compression changes stop risk or path capture;
- `11_daily_entry_frequency.csv`: how frequently each archetype actually produces/fills an entry.

#### R16 path-metadata hotfix
The first full R16 run revealed that ordinary MSS/FVG rows lost `range_model` and dealing-range metadata after concatenation with other entry families.  This affected only their 25/50/75/100 milestone and milestone-exit performance summaries; fill and immediate-stop labels were still valid.  The hotfix preserves path metadata in the R15 selective cache load and row-wise fills canonical path fields after heterogeneous concatenation, with conflict checks.  R16 should be rerun before comparing MSS/FVG PF at 25/50/75/100.

### R17 — No-Stop Opposite-Liquidity Diagnostic

`17_no_stop_opposite_liquidity_diagnostic.py` is a deliberately narrow counterfactual built on the frozen R16 1m entry:

- `prominent_15m_pair_0830` liquidity range;
- first visible 1m MSS;
- break-associated FVG near-edge limit entry;
- R16 signal, order price and actual fill time frozen unchanged;
- no intraday stop after fill;
- opposite frozen external-liquidity boundary is the only TP;
- if the opposite boundary is not reached, exit at the final 1m close before 16:30 ET.

The original terminal-extreme stop is retained only as a diagnostic.  R17 reports how many eventual opposite-liquidity winners were washed out by that stop, and how large the EOD/MAE tail becomes when the stop is removed.  This is not a recommendation to trade without a stop.

Recommended run using the completed R16 report:

```text
python research\ict\soxl_premarket_mss_fvg\17_no_stop_opposite_liquidity_diagnostic.py --data-source alpaca --alpaca-symbol SOXL --alpaca-feed sip --alpaca-adjustment split --start-date 2023-07-01 --end-date 2026-08-14 --local-only --r16-cache-dir data\reports\research\ict\soxl\mss\r16_entry_archetype_survival_atlas_alpaca_2023_2026_08 --out-dir data\reports\research\ict\soxl\mss\r17_no_stop_opposite_liquidity_alpaca_2023_2026_08
```

### R18 — Opposite-Liquidity Probability Hypotheses

`18_opposite_liquidity_probability_hypotheses.py` freezes the research universe back to ICT liquidity delivery: the 08:30 causally confirmed prominent 15m High/Low pair, a raid of one side, and the opposite external liquidity as the only directional target.  It does not use 25/50/75 milestones.

R18 fits simple discovery-only probability models for H1 liquidity context, H2 terminal maturity, H3 meaningful MSS, H4 displacement, H5 mitigation entry and H6 as-of cross-timeframe confirmation.  The model is fitted on 2023H2-2024 and evaluated unchanged on 2025 and 2026.  Entry archetypes are additionally compared on the same physical sweeps using `TP before terminal SL` rather than unrelated average-PF buckets.

Recommended fast run using the completed R15/R16 reports:

```text
python research\ict\soxl_premarket_mss_fvg\18_opposite_liquidity_probability_hypotheses.py --r15-cache-dir data\reports\research\ict\soxl\mss\r15_daily_liquidity_traversal_path_atlas_alpaca_2023_2026_08 --r16-cache-dir data\reports\research\ict\soxl\mss\r16_entry_archetype_survival_atlas_alpaca_2023_2026_08 --start-date 2023-07-01 --end-date 2026-08-14 --range-model prominent_15m_pair_0830 --out-dir data\reports\research\ict\soxl\mss\r18_opposite_liquidity_probability_hypotheses_alpaca_2023_2026_08
```

Key outputs:
- `01_sweep_mss_opposite_baseline.csv`: base probability of reaching opposite liquidity;
- `03_event_hypothesis_incremental_metrics.csv`: H1/H2/H3/H4/H6 incremental OOS probability quality;
- `04_event_probability_calibration.csv`: predicted vs realized opposite-liquidity probability;
- `06_entry_fill_and_tp_first_summary.csv`: fill and TP-before-SL rate by frozen entry archetype;
- `08_entry_paired_same_sweep_comparison.csv`: same-sweep paired entry comparison;
- `09_entry_probability_incremental_metrics.csv`: probability quality for TP-before-terminal-SL after fill;
- `12_golden_replay_2026-08-05.csv`: probability snapshots for the golden date.

## R19 — Event-Conditioned Entry Study
R19 freezes the R18 2m H1-H4 opposite-liquidity probability layer, then asks whether entry performance improves when that probability state is **already causally available before order placement**.  It explicitly rejects retroactive use of later 2m confirmation to filter an earlier 1m order.

## R20 — Broad Position-Management Backtest

`20_broad_position_management_backtest.py` changes the research objective from setup filtering to executable lifecycle management.

Frozen broad entry:
- range: `prominent_15m_pair_0830`;
- entry: `mss_first_visible_close_break_next_open_market`, 1m;
- no event-probability/session/HTF/CVD/profitability bucket may remove base trades;
- hard frequency gate: at least `0.5 filled trades / valid session` before management research is allowed to continue.

Predeclared management policies keep the same entry universe and compare: full opposite-liquidity hold, +1R breakeven protection, 25% partial at +1R plus breakeven, the same with +0.5R lock after +2R, and the same with causal 2m structure trailing. Stop changes triggered by an OHLC bar become active only on the next 1m bar; same-minute stop versus partial/TP ambiguity resolves to stop. Policy selection uses 2023H2-2024 only; 2025/2026 are evaluation-only. Baseline cost remains 0.11%, with 1.5x/2x cost and 1m/2m entry-delay stress.

Recommended Windows run:

```text
python research\ict\soxl_premarket_mss_fvg\20_broad_position_management_backtest.py --alpaca-symbol SOXL --alpaca-feed sip --alpaca-adjustment split --start-date 2023-07-01 --end-date 2026-08-14 --local-only --r16-cache-dir data\reports\research\ict\soxl\mss\r16_entry_archetype_survival_atlas_alpaca_2023_2026_08 --out-dir data\reports\research\ict\soxl\mss\r20_broad_position_management_backtest_alpaca_2023_2026_08
```


## R20 v2 broad-universe correction
- Corrected R20 from 1m close-confirmed-only entries to the earliest causal first-visible 1m/2m structure break per physical liquidity path.
- Wick-only vs close-confirmed is diagnostic/feature information, not a profitability filter.
- The >=0.5 trades/session target remains hard; R20 no longer aborts before management just because a mistakenly narrowed input subset fails it.
