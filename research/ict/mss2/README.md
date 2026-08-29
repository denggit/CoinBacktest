# ICT MSS2

Causal ETH ICT/MSS research branch. The current focus is no longer “every swing sweep + MSS + FVG”, but **which multi-layer liquidity pools are actually meaningful, how a sweep consumes a liquidity stack, and how to exit structurally without a forced short time stop**.

Read `00_RESEARCH_LOG.md` before changing semantics.

## R01 - Liquidity taxonomy + 1m/2m MSS/FVG atlas

Windows one-line command:

```text
python research\ict\mss2\01_liquidity_mss_fvg_atlas.py --symbol ETH-USDT-SWAP --warmup-start-date 2022-01-01 --start-date 2023-01-01 --end-date "2026-08-15 23:59:59"
```

R01 result is frozen as `research_continue`: universal MSS/FVG entry failed after costs, but multi-liquidity consumption became the next hypothesis.

## R02 - Liquidity pool / stack exhaustion + structural exit

Default Windows one-line command:

```text
python research\ict\mss2\02_liquidity_pool_stack_structural_exit.py --symbol ETH-USDT-SWAP --warmup-start-date 2022-01-01 --start-date 2023-01-01 --end-date "2026-08-15 23:59:59"
```

Default R02 report directory:

```text
data\reports\research\ict\mss2\r02_liquidity_pool_stack_structural_exit
```

R02 defaults:

- liquidity: 15m / 30m / 1H / 4H / 1D;
- execution: 1m / 2m / 5m;
- pool sensitivity: 5 / 10 / 20bp;
- sweep episode gap sensitivity: 5 / 15 / 30m;
- no forced time TP;
- structural stop;
- opposing active liquidity targets: nearest level / pool / multi-TF pool / 1H+ / 4H+ / 1D+;
- 7-day observation limit is censoring only, not a time exit;
- baseline 0.11% cost plus 2x / 3x stress.

## R03 - Order-flow uplift + FVG execution overlay

R03 fixes the legacy R02 trade-ID collision, freezes the long-side multi-liquidity-stack hypothesis, and tests whether trade-bar / Range Footprint absorption improves quality or recovers frequency from a broader >=3-pool cohort. It also compares first-FVG market, proximal limit, and 50/50 hybrid execution without introducing an NY-open gate.

Default Windows one-line command:

```text
python research\ict\mss2\03_liquidity_stack_orderflow_execution.py --symbol ETH-USDT-SWAP --warmup-start-date 2022-01-01 --start-date 2023-01-01 --end-date "2026-08-15 23:59:59"
```

R03 reuses the completed R02 report in:

```text
data\reports\research\ict\mss2\r02_liquidity_pool_stack_structural_exit
```

and reads all new market data only through `src.data_feed` loaders.

Default R03 report directory:

```text
data\reports\research\ict\mss2\r03_liquidity_stack_orderflow_execution
```

Important: footprint coverage is optional and may be much shorter than the trade-bar history. R03 evaluates it only on matched covered rows; missing footprint is never interpreted as a negative feature.

## R03.2 correction

Run the corrected R03.2 study with:

```text
python research\ict\mss2\03_2_liquidity_stack_orderflow_execution_fix.py --symbol ETH-USDT-SWAP --warmup-start-date 2022-01-01 --start-date 2023-01-01 --end-date "2026-08-15 23:59:59"
```

R03.2 keeps old R03 artifacts for history but writes to `data/reports/research/ict/mss2/r03_2_liquidity_stack_orderflow_execution_fix`. See `R03_2_CORRECTION_NOTES.md` and `00_RESEARCH_LOG.md` for the corrected checkpoint and execution semantics.

## R03.3 - ICT hierarchy + key liquidity + entry/exit + CVD

Run:

```text
python research\ict\mss2\03_3_liquidity_hierarchy_entry_exit_research.py --symbol ETH-USDT-SWAP --warmup-start-date 2022-01-01 --start-date 2023-01-01 --end-date "2026-08-15 23:59:59"
```

Default report directory:

```text
data\reports\research\ict\mss2\r03_3_liquidity_hierarchy_entry_exit
```

R03.3 separates raw pool count from liquidity quality. It adds a causal ICT ST/IT/LT swing-on-swing hierarchy, fixed-N pool-quality decomposition, entry-method and structural-target comparisons, causal trade-bar CVD diagnostics, and a corrected frozen-core FVG market/limit/50:50 execution overlay. See `R03_3_DEEP_RESEARCH_NOTES.md` before interpreting or modifying the study.

## R03.3.1 amendment

R03.3 now also refreshes MSS directly from naked 1m K on the hierarchy-defined research stages. In addition to pre-sweep `recent` and `structural` references it includes `post_sweep_st`: a new small execution-timeframe ST swing may form after the liquidity sweep, but becomes usable only after its right confirmation closes. A later close through it is then a valid MSS candidate.

Displacement is studied as a continuous feature family rather than an admission formula. The report freezes 2023-2024 quartiles for distance/speed/efficiency/body/FVG/attack-relative features and evaluates them on 2025-2026, including direct buckets where reversal displacement is weaker than the attack into the extreme.

## R04 - Multi-horizon liquidity opportunity atlas

Run:

```text
python research\ict\mss2\04_multi_horizon_liquidity_opportunity_atlas.py --symbol ETH-USDT-SWAP --warmup-start-date 2022-01-01 --start-date 2023-01-01 --end-date "2026-08-15 23:59:59"
```

Default report directory:

```text
data\reports\research\ict\mss2\r04_multi_horizon_liquidity_opportunity_atlas
```

R04 reuses the completed R02 + R03.3 reports and asks whether the same causal 5m liquidity-reclaim opportunity is a short rebound, medium move, or multi-day reversal. It follows +0.3/+0.5/+0.75/+1/+1.5/+2/+3/+5% targets and 1h through 14d paths with the original structural stop, **without introducing a time stop**. Future labels are physically separated from causal features; incomplete right-edge horizons are censored rather than counted as failures.

It also measures continuation after first opposing 4H liquidity and reports an algebraic short-profit partial fraction required to cover original stop risk + costs. That partial diagnostic is **not** a split-position backtest. Actual partial+runner/trailing management is deferred until R04 establishes whether the path structure supports it.

By default R04 recomputes causal 1m trade-bar context for the full opportunity universe. Add `--skip-tradebar` for a faster path-only diagnostic run.


## R05 — Entry Timing + Structural Stop / Runner

`05_entry_timing_structural_stop_runner_atlas.py` separates 1m/2m/5m reclaim timing from structural invalidation and runner management. It compares entry-time structural stops, measures MAE before short/medium/swing targets, and tests monotone 2m/5m/15m ITL/LTL or strong bullish displacement/FVG anchors as trailing stops. 1m trailing is intentionally forbidden. 3%/5% are diagnostics, not promoted fixed TPs.

## R06 — Adaptive Risk + Protected Position Lifecycle

Run:

```text
python research\ict\mss2\06_adaptive_risk_position_lifecycle.py --symbol ETH-USDT-SWAP --warmup-start-date 2022-01-01 --start-date 2023-01-01 --end-date "2026-08-15 23:59:59"
```

Default report directory:

```text
data\reports\research\ict\mss2\r06_adaptive_risk_position_lifecycle
```

R06 freezes the broad R05 `N>=3 + (4H OR LT)` Long family instead of adding another strict entry filter. It studies causal risk tiers, delayed promotion of 5m/15m ITL/LTL into protected stops, an optional risk-recycled add-on, a slower 15m major-runner state after +3%, single-position overlap, and full account-equity behavior under 1x/2x/3x costs. There is no fixed TP and no time stop. The primary acceptance metrics are equity smoothness and robustness: daily MTM MDD, positive-month/quarter rate, drawdown duration, rolling-90d positivity, trend R², Ulcer index, per-year stability, overlap, and top-winner concentration.

## R07 — ICT Family Expansion Atlas

Run:

```text
python research\ict\mss2\07_ict_family_expansion_atlas.py --symbol ETH-USDT-SWAP --warmup-start-date 2022-01-01 --start-date 2023-01-01 --end-date "2026-08-15 23:59:59"
```

Default report directory:

```text
data\reports\research\ict\mss2\r07_ict_family_expansion_atlas
```

R07 stops trying to rescue the R06 equity curve by further hard-filtering the same Long reversal family. It broadens into three ICT-style causal families: properly-confirmed BSL/SSL reversals, close-through-liquidity continuation with FVG resting-limit retracement, and small confirmed-reversal FVG corridor trades that use **limit entries only** and compare original structural versus local FVG-invalidation stops. The small corridor family separately freezes a causally-existing opposite FVG objective so it is not judged only by distant 4H targets.

Read `R07_RESEARCH_NOTES.md` and the generated `R07_ICT_SOURCE_BASIS.md` before interpreting results. R07 is a breadth/complementarity atlas, not an automatic portfolio promotion.

## Current default horizon / prebuild / chart review

New/future runs default to `2026-08-15 23:59:59`. See `PREBUILD_2026_08_15.md` before rerunning the current chain. R07 writes `manual_review/` with recent resolved examples for direct K-line verification.

## R08 — Full-Trend ICT Structure Atlas

Run:

```text
python research\ict\mss2\08_full_trend_ict_structure_atlas.py --symbol ETH-USDT-SWAP --warmup-start-date 2022-01-01 --start-date 2023-01-01 --end-date "2026-08-15 23:59:59"
```

Default report directory:

```text
data\reports\research\ict\mss2\r08_full_trend_ict_structure_atlas
```

R08 is a **structure-validation gate**, not a strategy/PF study.  It rebuilds classical ICT ST->IT->LT hierarchy on 15m/30m/1H/4H, forms complete opposite-LT trend legs, audits monotonic ITH/ITL progression, requires a post-terminal IT-BOS, and labels >=3/5/7% whole-trend scales.  ST-only swings never enter the trend-qualified liquidity universe.  Do not build R09 trading logic until the generated `manual_review/` structure files visually match ICT chart interpretation.

## R08.1 note
R08.1 separates canonical native-timeframe full-trend IT/LT liquidity from nested lower-timeframe IT/LT context. Do not add these counts together as independent physical liquidity. Use `06c_projection_impact_summary.csv` to compare win rate, net expectancy and PF rather than assuming the stricter taxonomy is automatically more profitable.

## R09 — ICT Liquidity Quality × Execution Atlas

Run:

```text
python research\ict\mss2\09_liquidity_quality_execution_atlas.py --symbol ETH-USDT-SWAP --warmup-start-date 2022-01-01 --start-date 2023-01-01 --end-date "2026-08-15 23:59:59"
```

Default report directory:

```text
data\reports\research\ict\mss2\r09_liquidity_quality_execution_atlas
```

R09 consumes the corrected R08.1 native/nested IT/LT liquidity files and **does not** collapse the strategy to the narrow high-quality 1H+nested cohort. It deduplicates physical levels, creates causal root sweep opportunities, preserves 15m/30m/1H/4H context as a C/B/A/A+ structural ladder, and compares sweep-immediate, reclaim, MSS and FVG resting-limit execution on the same opportunity universe. Later 15-minute multi-level cascade is a future diagnostic only and cannot influence initial tier or entry. Read `R09_RESEARCH_NOTES.md` before interpreting PF.

## R10 — Unified ICT Liquidity Trading Engine

Run:

```text
python research\ict\mss2\10_unified_ict_liquidity_trading_engine.py --symbol ETH-USDT-SWAP --warmup-start-date 2022-01-01 --start-date 2023-01-01 --end-date "2026-08-15 23:59:59"
```

Default report directory:

```text
data\reports\research\ict\mss2\r10_unified_ict_liquidity_trading_engine
```

R10 is the consolidation phase.  It consumes the completed R09 report and freezes one base trade per SSL episode: 2m episode reclaim -> next-open Long with the causal structural stop.  MSS is a later state upgrade, not a new trade.  The main Base+Runner architecture realizes a Base at 2R, gives the Runner break-even protection only from the next 1m bar, follows 5m LTL, and slows to later 15m LTL only after structural MSS + 3R.  Add-ons are disabled.

R10 evaluates only three pre-declared lifecycle variants and three pre-declared risk schedules under 1x/2x/3x costs.  The decision surface is the capital curve (`07_portfolio_equity_scorecard.csv`), not a best-PF signal grid.  Use `manual_review/01_recent_20_unified_positions.csv` for chart-by-chart verification.

### R11.1 — Continuous Visible Liquidity Path Atlas
`11_daily_liquidity_path_atlas.py` is a path-first reset for 24/7 ETH. It continuously tracks every causally known, unconsumed 15m/30m/1H/4H IT/LT level; 00:00 is reporting-only and never freezes or resets liquidity. Each root sweep freezes the opposite liquidity visible at that exact moment and follows the path continuously across midnight before any new strategy is promoted.

## R12 — Completed-Trend Swing Sweep -> Opposite Liquidity Path Atlas

Run:

```text
python research\ict\mss2\12_completed_trend_swing_opposite_liquidity_path_atlas.py --symbol ETH-USDT-SWAP --warmup-start-date 2022-01-01 --start-date 2023-01-01 --end-date "2026-08-15 23:59:59"
```

Default report directory:

```text
data\reports\research\ict\mss2\r12_completed_trend_swing_opposite_liquidity_path_atlas
```

R12 supersedes the broad R11/R11.1 map as the active research direction. It reads R08.1 completed-trend native + nested-lower-TF liquidity, deduplicates to physical ITH/ITL/LTH/LTL, waits for the physical first sweep, and freezes the opposite and deeper same-side completed-trend liquidity at that exact causal moment. The primary outcome is which side is reached first, with separate labels for direct opposite delivery, same-side continuation, cascade-then-opposite delivery, partial reversal, censoring, and same-bar ambiguity. Reclaim/MSS/FVG are diagnostics only; no entry/SL/TP is promoted in R12.

For chart review start with:

```text
manual_review\01_recent_20_direct_opposite_delivery.csv
manual_review\02_recent_20_same_side_continuation_failures.csv
manual_review\03_recent_20_cascade_then_opposite_delivery.csv
manual_review\04_recent_60_compact_chart_check.csv
```

## R13 — Reversal Quality & Causal Entry Discovery

Run:

```text
python research\ict\mss2\13_reversal_quality_entry_discovery.py --symbol ETH-USDT-SWAP --warmup-start-date 2022-01-01 --start-date 2023-01-01 --end-date "2026-08-15 23:59:59"
```

Default report directory:

```text
data\reports\research\ict\mss2\r13_reversal_quality_entry_discovery
```

R13 compares direct opposite delivery with any path that first reaches deeper same-side completed-trend liquidity. It measures liquidity age, sweep morphology, expected response, reclaim retention, MSS quality and FVG timing, then replays a small causal entry family against frozen opposing-liquidity TP and deeper-same-side SL.

The report was corrected to attribute each feature family only to an entry available after that feature exists: 15-minute response features use `response_15m_market`, reclaim features use `reclaim_market`, MSS features use their matching MSS entry, and FVG features use `fvg_market`. The old root-next-open attribution for post-root features is invalid and must not be cited.

Corrected result: no unfiltered reversal entry survives discovery to validation at 2x cost, and no feature-bin rule is promoted. The R13-specific holdout beginning 2025-08-01 remains sealed. Read `R13_RESEARCH_NOTES.md` and `STRATEGIC_RESET_R13.md` before starting another reversal filter study.

## R14 — Liquidity Acceptance / Continuation

Run:

```text
python research\ict\mss2\14_liquidity_acceptance_continuation.py --symbol ETH-USDT-SWAP --warmup-start-date 2022-01-01 --start-date 2023-01-01 --end-date "2026-08-15 23:59:59"
```

R14 tests the dominant deeper-same-side-first path as a distinct continuation mechanism. BSL sweeps imply long continuation and SSL sweeps imply short continuation. Entry requires a root close outside the swept region or persistent outside closes for 5/15 minutes; target is frozen deeper same-side completed-trend liquidity and full region reclaim plus 2bps is the stop.

BSL continuation is negative in every split/year. SSL 5/15-minute persistence is positive at 2x/3x costs but has only 4–28 resolved trades per split/model and collapses after top-five removal. Root-close acceptance preserves frequency but loses in validation on both sides. No strategy is promoted; see `R14_RESEARCH_NOTES.md`.

## R15 — Acceptance Fixed-R First Passage

`15_acceptance_fixed_r_first_passage.py` freezes the R14 SSL root-close-outside entry and reclaim stop, then tests exact 0.5R/1R/2R/3R target-first passage with stop-first ambiguity. All targets lose after costs in discovery and validation. Median stop distance is only ~0.17–0.18%, below the 0.22% round trip at 2x costs, and median resolution is the entry bar. Fixed-R exit rescue is stopped; read `R15_RESEARCH_NOTES.md`.

## R16 — Acceptance Structural Stop Atlas

`16_acceptance_structural_stop_atlas.py` keeps the SSL root-close entry and deeper same-side target fixed, then compares region-edge touch, root sweep-bar extreme, and close-reclaim behavioral invalidation. The wider stop improves discovery but all models fail validation and top-winner resilience. This closes the completed-trend acceptance branch. Read `R16_RESEARCH_NOTES.md` and `STRATEGIC_RESET_R16.md` before starting the next independent sleeve.

## R17 — Trend Pullback Re-acceleration Path Atlas

Run:

```text
python research\ict\mss2\17_trend_pullback_reacceleration_atlas.py
```

R17 is independent of the archived sweep, q70, raw-breakout, and broad Higher-Low limit-entry branches. It requires aligned causal 1D/4H structural trend, a confirmed 30m counter-trend pivot, a 15m reclaim of the pivot-bar range, a later 5m re-acceleration close, and next-observable 1m execution. Stops use the local 30m extreme plus 0.25× causal ATR and skip distances above 1.50%.

The exact sequence is rejected. All Long targets have 2× PF below 0.64 in both discovery and validation. All Short discovery targets are below 0.72 PF; the isolated 16-trade validation Short 1R cell collapses after five-winner removal. Holdout stays sealed and no strategy is promoted. Read `R17_PRECOMMITMENT.md` and `R17_RESEARCH_NOTES.md` before any further trend-pullback work.

## R18 — Independent Positioning-Unwind Path Atlas

Run:

```text
python research\ict\mss2\18_positioning_unwind_path_atlas.py
python research\ict\mss2\18_validate_positioning_unwind_atlas.py
```

R18 uses Binance USD-M `ETHUSDT` base OI only as a cross-exchange positioning
proxy and keeps price/execution on OKX `ETH-USDT-SWAP`. It freezes a prior 1h
directional price/OI build, the first causal 5m OI-release sign transition, an
opposite OKX 5m close through the prior bar extreme, and next-observable 1m
entry. Invalid/gapped OI observations are excluded without interpolation.

The exact reversal transition is rejected. Across 8,592 fully observed setups
and 34,368 target paths, gross PF is only 0.96–1.07. Discovery 2× PF is
0.19–0.45 and validation is 0.28–0.56; every primary monthly sum is negative.
Independent raw-bar replay finds zero path-ordering differences. Holdout remains
sealed and no strategy is promoted. Read `R18_PRECOMMITMENT.md` and
`R18_RESEARCH_NOTES.md` before any further positioning study.

## R19 — Positioning Rebuild Continuation-Resumption Atlas

Run:

```text
python research\ict\mss2\19_positioning_rebuild_continuation_atlas.py
python research\ict\mss2\18_validate_positioning_unwind_atlas.py --report-dir data/reports/research/ict/mss2/r19_positioning_rebuild_continuation_atlas
```

R19 freezes the opposite positioning sequence: directional 1h price/base-OI
build, first 5m OI release, uninterrupted negative-OI episode, first nonnegative
OI rebuild within 60 minutes, original-direction release-bar break, and
next-observable 1m entry. It produces 8,584 fully observed setups and 34,336
paths. Discovery 2× PF is at most 0.50 and validation at most 0.71; every
primary monthly 2× sum is negative. Nineteen causal and eleven independent
replay checks pass. The 2,661 holdout candidates have zero outcomes.

R18/R19 close the simple positioning-transition branch. Read
`R19_RESEARCH_NOTES.md`, `V10B_CANDIDATE_AUDIT.md`, and
`STRATEGIC_RESET_R19.md` before R20. The historical LF V10B headline is not a
promoted shortcut: it was selected on its full 2023–2026 window, fails top-ten,
coverage, monthly, MDD, parity, and untouched-holdout gates.

## R20 — Frozen LF V10B Component Falsification

Run:

```text
python research\ict\mss2\20_frozen_v10b_component_falsification.py
```

R20 freezes the existing V10B components and tests simple unlevered signed
entry-to-exit returns through visible 2025H1. It does not tune the historical
rules and loads no July or holdout data. Bear V3 Short and Bull Reclaim Long
both have negative gross expectancy in discovery and validation; their 2× PFs
are 0.72/0.72 and 0.65/0.22. Momentum cells contain only 2–6 trades per split
and fail sample/top-winner gates. All four components fail the forward-
incubation gate, while fourteen causal/arithmetic checks pass.

This closes the V10B reuse shortcut. Read `R20_PRECOMMITMENT.md` and
`R20_RESEARCH_NOTES.md`; do not rescue it with sizing, add-ons, component
selection, micro filters, or another structural-stop variant.

## R21 — Canonical Daily Channel Trend Following

Run:

```text
python research\ict\mss2\21_canonical_daily_channel_trend.py
python research\ict\mss2\21_validate_canonical_daily_channel_trend.py
```

R21 precommits a price-only daily 20/10 Donchian model and one canonical 55/20
sensitivity, both with completed daily signals, next-day 00:00 entries, fixed
2×ATR(20) stops, and next-open channel exits. Long/Short and discovery/2025H1
validation simulations are independent; July and holdout data are absent.

The primary Long cell has discovery 2× PF 1.76 on 13 trades, but only one
validation trade and a -54.96% discovery return after removing its top five
winners. Primary Short is negative at 2× costs in both splits (PF 0.34/0.64).
The 55/20 sensitivity has the same sparse, 2024-winner-concentrated Long shape
and no Short discovery edge. All directions fail the frozen gate.

Eleven causal/cost checks and eight independent raw-replay checks pass across
all 42 emitted paths. R21 is rejected; do not rescue it with channel grids,
filters, trailing stops, pyramiding, leverage, or holdout selection. Read
`R21_PRECOMMITMENT.md` and `R21_RESEARCH_NOTES.md` before selecting R22.

## R22 — BTC-Led ETH Catch-Up First Passage

Run:

```text
python research\ict\mss2\22_btc_led_eth_catchup_first_passage.py
python research\ict\mss2\22_validate_btc_led_eth_catchup.py
```

R22 uses a completed 1h BTC impulse and prior-only rolling ETH beta/residual
volatility to identify same-direction ETH under-reaction. It produces useful
frequency, but not edge: primary R1 Long discovery/validation 2× PF is
0.85/1.13 on 207/36 trades; Short is 0.72/0.92 on 138/25. R2 diagnostics also
fail. Both discovery years lose and top-ten removal is negative.

Fifteen causal checks and five independent raw-array replay checks pass across
all 781 emitted paths. R22 is rejected and cannot be rescued with beta/sigma,
threshold, session, target, stop, or trend variants. Read
`R22_RESEARCH_NOTES.md` and the mandatory `STRATEGIC_RESET_R22.md`.

## R23 — Frozen Panic-Wick Structural Long

Run:

```text
python research\ict\mss2\23_frozen_panic_wick_structural_long.py
python research\ict\mss2\23_validate_frozen_panic_wick.py
```

R23 freezes the strongest historical shadow-wick executable prior after
documenting its full-window selection across at least 81 V1 combinations and a
later exit ladder. Discovery 2× PF is 1.67 on 119 trades, but validation PF is
0.96 on 111 trades; removing the top ten discovery winners changes the net sum
to -4.52%. The 2025 frequency surge carries no stressed-cost expectancy.

Twelve causal checks and eight independent structural-state replay checks pass.
The rule is rejected and cannot be rescued with session, flow, prior-move,
wick, volatility, target, delay, or exit filters. Read
`PANIC_WICK_CANDIDATE_AUDIT.md` and `R23_RESEARCH_NOTES.md`.

## R24 — Scheduled Funding-Window Unwind

Run:

```text
python research\ict\mss2\24_scheduled_funding_window_unwind.py
```

R24 tests reversal after >=1.5-sigma completed hourly moves into the canonical
00:00/08:00/16:00 funding clock. It has no gross edge: primary R1 Long/Short
discovery 2× PF is 0.70/0.70 and validation is 0.72/0.37. Every visible year
loses and timeout rates are high. Fourteen causal checks and two focused tests
pass. No clock, threshold, target, stop, or hold-period rescue is allowed; read
`R24_PRECOMMITMENT.md` and `R24_RESEARCH_NOTES.md`.

## R25 — r0020 Directional-Run Exhaustion Reversal

Run:

```text
python research\ict\mss2\25_r0020_directional_run_exhaustion.py
python research\ict\mss2\25_validate_r0020_directional_run_exhaustion.py
```

R25 freezes fixed r0020 only: a maximal run of at least four completed bars,
the first opposite completed bar, strictly-later 1m entry, run-origin target,
and run-plus-confirmation extreme stop. It does not repeat the rejected
Range-Bar activity-continuation rule and uses no scale/run-length/filter family.

The event has no gross edge despite extreme frequency. Primary Long discovery/
validation gross PF is 1.05/1.08 and 2x PF is 0.41/0.42 on 4,567/2,228 trades.
Short gross PF is 1.01/0.98 and 2x PF is 0.39/0.38 on 4,707/2,182 trades. Every
visible month, quarter, and year loses at 2x cost; top-winner removal and a
one-minute delay remain negative.

Six focused tests, sixteen internal causal/cost checks, and eighteen independent
raw-source replay checks pass. Two invalid r0020 timestamp rows reset sequence
state; July and holdout remain absent. Reject the mechanism and do not rescue it
with alternate scales, run lengths, duration, flow, footprint, session, target,
stop, confirmation, or ML filters. Read `R25_RESEARCH_NOTES.md` and the mandatory
`STRATEGIC_RESET_R25.md` before R26.

## R26 — Relative Positioning Leadership Repricing

Run:

```text
python research\ict\mss2\26_relative_positioning_leadership_repricing.py
python research\ict\mss2\26_validate_relative_positioning_leadership.py
```

After the OKX spot/perpetual source gate failed, R26 tests the one complete and
independent remaining positioning state. A Binance top-trader position-share
cross relative to global-account share arms direction; the relative spread
must retain its sign until the first same-direction completed OKX 5m price
confirmation within one hour. Entry is the next eligible 1m open, the stop is
the two-bar confirmation extreme plus 0.25x ATR, and the primary target is the
direction-side extreme of the one-hour range frozen at the cross.

The study physically loads nothing at or after 2025-07-01. It produces 160/59
Long and 169/69 Short non-overlapping structural trades in discovery/
validation. Long primary gross PF is 1.22/1.20 but 2x PF is only 0.39/0.45;
Short gross PF is 0.93/0.95 and 2x PF 0.29/0.40. Every visible year loses at 2x,
and all fixed-R diagnostics fail after 1x cost. Seventeen independent raw-source
replay checks pass. Reject both directions and do not rescue them with ratio
thresholds, OI/taker filters, confirmation/exit changes, or ML. Read
`R26_PRECOMMITMENT.md` and `R26_RESEARCH_NOTES.md`.

## Pre-R27 — Reproducible Local Source Readiness Gate

Run:

```text
python research/ict/mss2/00_pre_r27_source_readiness_audit.py
```

This is not R27 and does not load strategy outcomes. It discovers every actual
local market series through `src.data_feed`, physically stops before
2025-07-01, profiles fixed-cadence and dated-archive coverage, and joins source
readiness to the mechanism families frozen through R26. The current result is
`UNASSIGNED_NO_ELIGIBLE_MECHANISM`: the only complete 1m price instruments are
ETH-USDT-SWAP and BTC-USDT-SWAP; the novel spot, carry/basis, liquidation, and
book-response lanes are unavailable; complete trade/Range-Bar/Binance lanes map
to rejected families. Read `POST_R26_SOURCE_MECHANISM_AUDIT.md` and the generated
`pre_r27_source_readiness_audit/SOURCE_READINESS_AUDIT.md` before proposing R27.

## R27 — Sequential ICT Reversal Path Study

Run in order:

```text
python research\ict\mss2\27_sequential_ict_reversal_path_discovery.py
python research\ict\mss2\27_validate_frozen_sequential_ict_reversal.py
python research\ict\mss2\27_finalize_sequential_ict_reversal_report.py
```

R27 corrects the earlier overgeneralization from R13. It tests one frozen causal
sequence from sweep through reclaim, new post-sweep structure, meaningful MSS,
strong displacement, actual FVG retracement fill, and protected swing. Every
state uses the same sweep-invalidating stop and opposite completed-trend
liquidity target; sweep entry remains a baseline only.

No state passes the preregistered discovery gate. SSL S2/S3 appears positive in
discovery (2× PF 1.44/1.55 on 52/42 fills), but direct-delivery uplift reaches
only +5.18 percentage points, top-five removal is negative, and 2025H1 2× PF
falls to 0.62/0.56. BSL loses throughout. S5/S6 shrink to zero or one executable
fill per split/side and cannot support inference. Internal and independent
replay audits have zero violations; the 2025-08-01 holdout remains sealed.

Reject the frozen sequential reversal mechanism and do not rescue it with
threshold relaxation or feature stacking. Read `R27_PRECOMMITMENT.md` and
`R27_RESEARCH_NOTES.md`; review the figures and `manual_review/` pack in the R27
report directory.
