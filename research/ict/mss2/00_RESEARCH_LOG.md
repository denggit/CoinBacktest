# ICT MSS2 Research Log

## Goal

Research whether ETH-USDT-SWAP has a robust, causal edge in the sequence:

`HTF liquidity -> first true sweep -> LTF displacement -> close-confirmed MSS -> displacement FVG -> FVG proximal limit pullback -> structural stop -> opposing liquidity / fixed-R exit`

This branch is intentionally `research/ict/mss2` because `research/ict/mss` is already occupied by another line of work.

## Non-negotiable constraints

- Source data: official naked 1m OHLC(V) through `src.data_feed.OKXDataLoader`; no research-local market-data interface.
- Warmup: 2022-01-01; formal research: 2023-01-01 through 2026-08-15 23:59:59 by default for new/future runs.
- No lookahead. Left-labelled bars are usable only after close.
- No "future eventual swing order" may become a current feature.
- No `recent N swings only` expiry for HTF liquidity. Old/remote unconsumed levels remain alive until first true sweep.
- 1m and 2m signal timeframes are compared on the same underlying 1m liquidity-consumption lifecycle.
- Resting-order fills, stop and target paths are always resolved on original 1m naked K, including for 2m signals.
- Same-bar ambiguity is pessimistic. On the fill 1m bar, stop may count but target cannot count because target-before-fill ordering is unknowable from OHLC.
- Session / weekday fields are research stratifications, not hard filters.
- Long and short sides are always reported separately.

## R01 - Liquidity taxonomy + 1m/2m MSS/FVG atlas

File: `01_liquidity_mss_fvg_atlas.py`

### 1. Swing candidate != liquidity

HTF candidates are built symmetrically on:

- 15m
- 30m
- 1H
- 4H

Every order-1 pivot enters the broad candidate universe. Higher confirmation orders 2/3/5 are attached only at the exact time they become causally knowable. The research never retroactively treats an order-1 pivot as order-5 before that confirmation exists.

### 2. Liquidity attributes available by sweep time

Each candidate carries/receives:

- side: buy-side high / sell-side low;
- source timeframe;
- confirmed pivot order at the first sweep;
- external-20 / external-50 flags based only on prior HTF bars;
- pivot rejection / reaction features known by order-1 availability;
- active same-side levels near the same price;
- number of active source timeframes near the same price;
- same-price/equal-high/equal-low style pool evidence;
- age from original pivot to sweep;
- active-age from causal availability to sweep;
- old/remote flags: >=6h, >=24h, >=72h;
- first approach to the level;
- first near-touch to the level;
- clean first sweep vs pre-tested level;
- sweep depth.

No level is removed merely because it is old.

### 3. Structural liquidity taxonomy

The taxonomy is intentionally interpretable and not outcome-trained:

- `minor_swing_candidate`
- `structural_swing`
- `structural_external`
- `major_swing`
- `major_external`
- `same_price_pool`
- `multi_tf_pool`

A structural score and A/B/C/D bin are also emitted. These are research groupings, not claims that A is automatically tradable. R01 is designed to test whether higher structural quality actually predicts better post-sweep behavior.

### 4. Sweep-only control

Before testing MSS, R01 measures the post-sweep path from the next 1m open. This answers the key incremental-edge question:

`Does MSS/displacement/FVG add information beyond the sweep itself?`

Sweep-only forward labels include 5/15/30/60/120/180 minute directional close return, MFE and MAE.

### 5. MSS references

Two causal reference modes are compared:

- `recent`: latest pre-sweep opposite execution-TF pivot already confirmed to order >=1;
- `structural`: latest pre-sweep opposite execution-TF pivot already confirmed to order >=2.

The reference must be known **before the execution bar containing the sweep starts**. A reference that only becomes confirmed because of the sweep bar is rejected.

MSS requires a close through the reference, not a wick-only break.

### 6. Displacement

Displacement is measured over the complete sweep-to-MSS leg, not reduced to a single large candle. R01 records:

- displacement / pre-sweep ATR;
- break distance / pre-sweep ATR;
- path efficiency;
- MSS body / pre-sweep ATR;
- MSS body/range ratio;
- bars/minutes from sweep to MSS.

### 7. FVG and entry

Bullish FVG: `low[t] > high[t-2]`.

Bearish FVG: `high[t] < low[t-2]`.

The first FVG inside the displacement leg is recorded. Proximal entry is:

- long: upper FVG boundary;
- short: lower FVG boundary.

The limit order is not active until the MSS close. Fill search then uses original 1m K data, even for a 2m signal.

Structural stop:

- long: below the complete sweep-to-MSS low plus small buffer;
- short: above the complete sweep-to-MSS high plus small buffer.

### 8. Targets and diagnostics

R01 measures:

- fixed 1R / 2R / 3R;
- nearest active opposing 15m liquidity;
- nearest active opposing liquidity across configured HTFs;
- baseline 0.11% round-trip cost;
- 2x and 3x cost stress in summary tables.

This remains event research, not yet a capital-constrained final portfolio backtest.

### 9. Date/session stratification

R01 reports by:

- year / quarter / month;
- UTC weekend and New-York-calendar weekend;
- New York weekday;
- every New York hour;
- every London hour;
- every Shanghai hour;
- broad Asia day window;
- London AM window;
- New York 09:00-10:30 window;
- New York cash-open first 30 minutes.

These time windows are explicit research bins. They are not pre-assumed to be profitable ICT killzones for a 24/7 ETH market.

### 10. 1m vs 2m comparison

R01 runs the full event construction for:

- 1m recent reference;
- 1m structural reference;
- 2m recent reference;
- 2m structural reference.

It also writes a one-to-one level overlap file to show which underlying liquidity sweeps generate an MSS/FVG/fill on 1m, 2m, both, or neither.

## Causal engineering completed

Synthetic tests cover:

1. 2m left-label aggregation and complete-bar policy.
2. Pivot availability only after the right confirmation bar closes.
3. A sweep before level availability is ignored.
4. MSS reference is pre-sweep and close-confirmed.
5. Entry can only occur after MSS close.
6. 2m cannot react before its sweep-containing bar closes.
7. Old/remote liquidity is not expired.
8. Future labels are physically removed from the causal feature table.
9. Same-bar stop/target is pessimistic stop-first.
10. Sweep-only baseline starts from the next 1m open.

Current targeted result: `9 passed`.

Project-wide `test_import_boundaries.py` currently reports 155 pre-existing unexpected violations in the uploaded snapshot. None belongs to `research/ict/mss2` or `src/research_common/ict_mss2`.

## What R01 is expected to decide

Do not promote a final strategy merely because aggregate MSS returns are positive. We need to see a stable hierarchy such as:

1. sweep-only has weak/no edge;
2. MSS improves it;
3. displacement quality improves it further;
4. FVG pullback improves executable expectancy after costs;
5. the improvement is not concentrated in one year/session/side;
6. meaningful liquidity classes outperform weak swing candidates;
7. old/remote and multi-TF liquidity behavior is empirically distinguishable;
8. 1m or 2m shows a clear robustness/execution advantage rather than a single lucky slice.

If that hierarchy is absent, stop or redesign this MSS hypothesis instead of parameter-mining it.

## Next stage after R01 results

Only after reviewing the R01 report:

- freeze the liquidity classes that demonstrably carry edge;
- freeze a displacement plateau rather than a single best threshold;
- decide 1m vs 2m, or whether both are complementary;
- test session filters only if R01 shows stable cross-year uplift;
- build R02 capital backtest with one-position/portfolio conflict rules, latency/slippage, realistic fees, walk-forward and holdout;
- reject the direction if the edge disappears under realistic costs or is driven by a narrow year/session/parameter cell.

---

# R01 actual result review (2026-08-15)

User ran the full 2023-01-01 -> 2026-06-30 report and returned `r01_liquidity_mss_fvg_atlas.zip`.

## Frozen R01 conclusions

1. **Universal sweep -> MSS -> FVG limit is rejected as a finished strategy.**
   - Every 1m/2m recent/structural execution branch remained negative after the default 0.11% round-trip cost.
   - 2m was consistently cleaner than 1m and structural references were cleaner than recent references, but the best branch was still not tradable.
2. **MSS is informative ex post but confirmation consumes much of the move.**
   - Sweeps that later produced structural MSS had strong subsequent directional paths.
   - Waiting through MSS and then an FVG pullback did not preserve enough universal expectancy.
3. **Liquidity taxonomy carries information.**
   - Sell-side sweeps into long reversals strengthened with higher source timeframe / older liquidity.
   - The short-side mirror was much weaker, consistent with ETH long/short asymmetry seen in other project research.
4. **NY cash open is not a privileged ETH gate.** Session/weekday effects exist but are secondary and must remain diagnostics unless cross-year evidence supports a filter.
5. **Displacement size, path efficiency and FVG width did not show a robust monotone payoff relationship.** Do not optimize those thresholds in R02.

## Critical statistical correction from R01 review

The R01 output contained 104,417 level-level sweep rows, but only 44,723 unique 1m sweep bars. A single liquidation impulse can consume several HTF swing levels simultaneously, so level rows are not independent observations.

Re-aggregating R01 by unique sweep bar and merging nearby swept prices into 10bp pools produced a materially stronger long-side pattern:

- 1 independent price pool consumed: ~0bp mean 60m directional return;
- 2 pools: ~+6bp;
- 3 pools: ~+16bp;
- >=4 independent pools: much stronger raw reversal response;
- after imposing a 180-minute event separation, 112 long-side >=4-pool events remained with roughly +28bp mean 60m return, +48bp mean 180m return and ~66% positive 60m outcomes.

The >=4-pool effect weakened in 2026 H1, so it is a research hypothesis rather than a promoted edge. The key discovery is that **liquidity consumption density/stack structure is more informative than an isolated swing label**.

This changes the research question from:

> Which single swing is real liquidity?

into:

> How much independent, causally available liquidity is concentrated in a zone, and how much of that stack did the current impulse consume before reversal evidence appears?

# R02 - Liquidity Pool / Stack Exhaustion + Structural Exit

## Why R02 exists

The user noted that many profitable ETH discretionary trades are held for more than one day and that current ETH volatility can be too low for fixed short time exits. They proposed using a structural stop and opposing liquidity pool as the profit objective. This is also more consistent with the ICT concept of a draw on opposing liquidity than forcing a 60/180-minute close.

R02 therefore **does not optimize the failed R01 MSS/FVG parameters**. It changes the statistical unit, timeframe comparison and exit model.

## R02 statistical units

### Sweep stage

One 1m bar/direction that consumes one or more already-active liquidity levels. It is one statistical event regardless of how many swing rows are swept.

At each stage, consumed prices are clustered with fixed 5/10/20bp sensitivities. These are robustness views, not fitted thresholds.

### Causal sweep episode

A same-direction sequence continues only when the next stage:

1. occurs inside a fixed gap (primary 15m; descriptive 5/15/30m sensitivity), and
2. extends the current sweep extreme.

Every episode-stage row contains only current and prior stage information. The eventual future episode length, pool count or final extreme is never backfilled.

Important causal features include:

- levels consumed stage/cumulative;
- independent price pools stage/cumulative at 5/10/20bp;
- pool consumption rate per minute;
- consumption depth;
- number of distinct source timeframes;
- highest source timeframe reached;
- order-2/order-3 confirmed levels consumed;
- >=1H / >=4H / >=1D levels consumed;
- old/remote, clean and pretested composition;
- timeframe signature.

There is deliberately no arbitrary weighted master score yet. R02 should first learn whether the components have stable monotone value before assigning weights.

## Timeframe research

Default liquidity universe is:

- 15m
- 30m
- 1H
- 4H
- 1D

5m is supported as an optional liquidity-source sensitivity, but it is not mixed into the default R01-comparable universe because it could mechanically inflate stack counts.

Execution is compared on the exact same 1m lifecycle at:

- 1m
- 2m
- 5m

This explicitly tests whether ETH needs slower confirmation than index-style 15m context -> 1m trigger. Session clocks remain diagnostics, not admission rules.

## Entry triggers compared

For each causal episode stage:

1. `stage_reclaim`
   - close reclaims the currently swept stage's consumed-price boundary;
   - may confirm on the execution bar containing the sweep;
   - entry is never before that execution bar has closed.
2. `episode_reclaim`
   - reclaims the cumulative episode liquidity boundary.
3. `mss_structural_market`
   - close breaks a structural execution-TF swing already confirmed before the sweep-containing execution bar began;
   - enter next available 1m open.
4. `mss_structural_fvg_limit`
   - same structural MSS semantics;
   - first displacement-leg FVG; resting limit after MSS close;
   - cancel if structural stop is breached on a completed 1m bar before fill.

`recent` MSS remains optional but is not the R02 default because R01 showed structural > recent.

## Structural stop

For all entries, stop is frozen beyond the full causal episode-start -> signal/entry extreme plus the small fixed buffer.

The stop is not tightened by future structure. R02 is specifically trying to learn the target geometry first.

## No time-profit exit

R02 has **no forced time TP**.

Default maximum observation is 7 days. At 7 days:

- unresolved trade = `censored`;
- it is not force-closed;
- it is not assigned zero return;
- it is excluded from resolved-return PF/expectancy and separately reported through censored rate.

For path interpretation, R02 stores 1h / 6h / 12h / 1d / 2d / 3d / 7d mark, MFE and MAE labels.

## Opposing-liquidity targets frozen at entry

At entry, the active opposing book is queried causally and the target is frozen. R02 compares:

1. nearest active opposing level;
2. nearest opposing pool containing >=2 active levels within 10bp;
3. nearest opposing pool containing >=2 levels from >=2 source timeframes;
4. nearest opposing >=1H liquidity;
5. nearest opposing >=4H liquidity;
6. nearest opposing >=1D liquidity;
7. fixed 1R/2R/3R/5R only as geometry diagnostics.

R02 first compares full-exit target definitions. Partial exits / runner logic should be tested only after a target hierarchy is identified; otherwise too many management degrees of freedom are introduced at once.

## Important causal correction versus the R01 target book

During R02 design we found that R01's opposing-liquidity target index removed a level at `sweep_pos`. At the **start** of that 1m bar, however, the future sweep was not known yet. This was conservative rather than profit-inflating, but technically non-causal target-book timing.

R02 removes the level at `sweep_pos + 1` instead. The level is active at the start of its eventual sweep bar; same-bar competing outcomes are then handled pessimistically. This correction must remain frozen in later versions.

## Independence and threshold statistics

For each 5/10/20bp pool tolerance and threshold 1/2/3/4, R02 selects the **first causal stage where the episode reaches that threshold**. This avoids counting the same episode repeatedly when testing “>=4 pools consumed”.

Default trade summaries similarly select the first eligible entry per episode / execution timeframe / trigger after each stack threshold is known.

## R02 reporting requirements

Reports must include:

- episode-gap 5/15/30m sensitivity;
- pool tolerance 5/10/20bp sensitivity;
- year stability, with 2026 decay highlighted rather than optimized away;
- long and short separately;
- source-timeframe / multi-timeframe context;
- 1m/2m/5m execution comparison;
- reclaim vs structural MSS market vs MSS/FVG limit;
- nearest level vs pool2 vs pool2 multi-TF vs 1H+/4H+/1D+ target;
- target availability, target hit, stop, censored rates;
- holding-duration distribution, including >1 day;
- baseline 0.11%, 2x and 3x cost PF/expectancy on resolved exits;
- 1h/6h/12h/1d/2d/3d/7d path atlas;
- explicit causal audit.

Do not interpret a high PF with a high censored rate as a finished strategy. A target must be reachable often enough to define a practical exit.

## R02 promotion criteria

`promote_to_backtest` requires more than one lucky cell. Prefer:

1. same directional effect across neighboring 5/10/20bp pool definitions;
2. stable 2023/2024/2025 and non-catastrophic 2026 behavior;
3. independent episodes rather than repeated stages;
4. a clear entry-trigger hierarchy that persists across 1m/2m/5m;
5. an opposing-liquidity target with practical target availability and censor rate;
6. positive resolved expectancy after 0.11% costs and meaningful resilience at 2x costs;
7. no dependence on one session/hour;
8. zero causal-audit violations.

If stack exhaustion has forward-path edge but every causal confirmation gives it away, R03 should focus on earlier exhaustion/reclaim execution rather than further MSS/FVG tuning.

## R02 engineering validation

Targeted tests after implementation: `16 passed` for R01+R02 helpers.

A 14-day synthetic end-to-end run generated:

- 20,160 1m bars;
- 585 liquidity candidates;
- 300 unique sweep stages;
- 248 sweep episodes;
- 1,838 1m/2m/5m trigger/outcome rows;
- all five R02 causal-audit counters = 0.

The synthetic run is only an engineering/causality test. It is not evidence of ETH edge.

---

# R02 actual result review (2026-08-15)

User ran the full 2023-01-01 -> 2026-06-30 R02 report and returned `r02_liquidity_pool_stack_structural_exit.zip`.

## Frozen R02 conclusions

1. **Removing the short time-stop did not rescue the universal model by itself.** Full-sample reclaim/MSS/FVG branches remained weak unless conditioned on meaningful liquidity-stack consumption.
2. **The long-side >=4 independent-pool stack is the primary candidate.** At 10bp pool clustering and first causal >=4-pool crossing, 5m episode-reclaim + structural stop + opposing 4H liquidity target produced 269 trades and remained positive in each year under 2x cost, though the edge is still modest and not yet a production strategy.
3. **4H+ involvement is strongly informative but was discovered post-hoc.** Within the core >=4-pool 5m cohort, stacks containing >=4H source liquidity were much stronger than stacks capped below 4H. This must be treated as a frozen R03 hypothesis, not as an already-validated rule.
4. **Speed of consumption matters.** Rapid multi-pool clearing was stronger than slow clearing, supporting a continuous `liquidity consumption velocity` concept rather than a stock-market session gate.
5. **Opposing 4H liquidity was the best structural target family.** Nearest liquidity/pools were too close to pay for failures and costs, while 1D liquidity was often too far. The 4H target had a median geometry around 2.4R and a meaningful fraction of winners required >12h / >1d holding.
6. **ETH execution is not index-style 15m -> 1m by default.** 1m/2m/5m episode-reclaim all showed similar core behavior, with 5m slightly strongest in R02. NY cash open did not improve the candidate and is removed from strategy admission thinking.
7. **Short-side mirror is not supported.** Multi-buy-side-liquidity clearing behaved more like bullish continuation than short exhaustion. R03 focuses on long-side sell-side-liquidity exhaustion instead of forcing symmetry.
8. **Waiting for an MSS/FVG pullback did not obviously improve the R02 candidate.** This motivates a separate execution overlay study after the core stack event is frozen.

## R02 engineering issue found and fixed for R03

Old R02 `trade_event_id` restarted from `R02_TRADE_000000001` independently for 1m / 2m / 5m trigger builds. R02's own row-wise/grouped summaries remained correct because they did not join different execution timeframes by ID, but any later feature/label join could become many-to-many.

Fix:

- new R02 IDs are `R02_1M_TRADE_...`, `R02_2M_TRADE_...`, `R02_5M_TRADE_...`;
- R03 includes a guarded legacy-report repair that only uses positional repair after verifying the R02 feature/label ID sequence is exactly identical;
- targeted regression test ensures IDs are globally unique across execution timeframes.

# R03 - Liquidity Stack Order-Flow Uplift + Execution Overlay

## Research objective

R03 deliberately separates three layers:

1. **Core edge**: the already-defined long-side multi-pool sell-side-liquidity exhaustion event.
2. **Quality / frequency layer**: causal trade-bar and Range Footprint evidence that may distinguish true exhaustion from continued liquidation.
3. **Execution layer**: market vs FVG proximal limit vs 50/50 market+limit after the same frozen stack event.

The research must not let microstructure data retroactively redefine swing availability, pool membership or episode history.

## Frozen cohorts

### Core

- Long only.
- 10bp independent price pools.
- First causal episode stage reaching >=4 independent pools.
- 1m / 2m / 5m episode-reclaim comparison, with 5m the primary R02 baseline.
- Structural stop beyond causal episode-to-signal extreme.
- Opposing active 4H+ liquidity as the primary target.
- No NY-open gate.
- No time-profit exit; 7d remains censoring only.

### Frequency expansion

The only predeclared relaxation is first causal >=3-pool crossing. R03 does **not** open a broad threshold grid. The question is whether causal microstructure can recover a useful subset of >=3-pool events with materially more frequency than the >=4 baseline.

## Trade-bar features

Primary broad-coverage module uses cached OKX 1m trade bars through `src.data_feed.OKXTradeBarLoader` only. No new market-data interface is created.

At each decision timestamp, only left-labelled 1m trade bars whose full interval completed before the decision are used. Features include:

- episode and last 1/3/5/15m notional, buy/sell notional, Delta, trades count and large-trade flow;
- sell/notional intensity versus the fixed pre-episode 60m baseline;
- downside bp per USD 1m sell notional;
- episode downside-impact ratio versus the pre-episode baseline;
- close-off-low / reclaim strength;
- last-5m Delta improvement versus the whole episode;
- large-sell share and maximum trade notional.

A fixed mechanism flag is predeclared rather than tuned:

`more sell activity than pre-episode baseline + lower downside impact per sell million + improving last-5m Delta`.

The report also freezes 2023-2024 feature quartiles and reuses them on 2025-2026 as **diagnostic strata only**. The best quartile is not automatically a strategy rule.

## Footprint module

R03 reuses the existing causal r0020 / step1 Range Footprint pipeline. Missing historical footprint is never encoded as a negative signal. Footprint uplift is judged only on the matched subset where the causal footprint context is actually present.

Fixed mechanism semantics:

- low-3-bin sell flow at least as large as the previous down Range bar;
- lower downside impact per sell flow than the previous down bar;
- Delta improvement;
- close-off-low improvement.

This is intended to describe absorption: more/aggressive selling is producing less additional downside and better reclaim behavior.

## Execution overlay

Execution is explicitly secondary to the core edge.

After the frozen >=4-pool stack threshold is known, R03 finds the first same-direction FVG on 1m / 2m / 5m and compares:

1. `stack_first_fvg_market`: market at the next eligible 1m open after FVG close;
2. `stack_first_fvg_limit`: proximal FVG resting limit;
3. `stack_first_fvg_hybrid_50_50`: 50% market + 50% proximal limit.

Fair-comparison rule:

- the opposing 4H target is frozen at the FVG signal/first market-executable time for **all** three execution variants;
- the limit leg is cancelled if the frozen target or structural stop is hit before the limit fills;
- same-bar limit ambiguity stays pessimistic through the shared outcome resolver;
- unfilled hybrid half contributes zero PnL and zero cost rather than pretending full deployment.

Execution cost convention:

- market round trip default 0.11% for project comparability;
- limit-entry + taker-exit default 0.09% (2bp entry-fee saving versus all-market convention);
- 2x / 3x stresses multiply each execution variant's own cost.

## R03 interpretation rules

- Do not promote a microstructure feature because one historical quartile is best.
- Prefer monotone or mechanistically consistent feature behavior in both train (2023-2024) and forward (2025-2026) periods.
- The >=4 / 4H observations were seen in R02 on this historical corpus; R03 is a replication/ablation pass, **not** a pristine external holdout for that core hypothesis.
- A useful frequency-recovery result should produce more trades than core >=4 while retaining positive 2x-cost expectancy across multiple years.
- Footprint results must be interpreted on matched coverage only.
- Execution overlays cannot redefine the liquidity edge. If market beats limit, that is an execution conclusion, not evidence that FVG created the edge.

## Engineering validation before user full-data run

- R01/R02/R03 targeted unit tests: 21 passed.
- Legacy R02 global-ID repair tested.
- Trade-bar left-labelled decision-time exclusion tested with an unavailable decision-start bar containing an extreme synthetic value.
- FVG market/limit/hybrid target freeze and shared-target semantics tested.
- Main R03 script smoke-tested against the real user R02 report with empty external microstructure caches; report creation and review-pack generation completed.
- R03 smoke candidate extraction reproduced R02 core counts/metrics (5m core >=4: 269 trades; 2x-cost PF ~1.146) before any new microstructure data was attached.
- No external-data smoke can establish edge; user must run against local trade-bar/footprint databases.

## R03.1 execution-position alignment bugfix — 2026-08-15

### Symptom
The real full-history R03 run reached the FVG overlay and failed inside `_structural_stop_before_entry` with `ValueError: zero-size array to reduction operation fmin which has no identity`.

### Root cause
R02 persisted `sweep_pos_1m`, `episode_start_pos_1m`, `active_pos_1m`, and liquidity `sweep_pos_1m` relative to the naked 1m frame loaded from the R02 warmup start (`2022-01-01`). R03 originally reloaded naked 1m only from the research start (`2023-01-01`) before using those persisted positions. This shifted every positional reference by roughly one year. The zero-length stop slice exposed the problem; silently continuing would also have corrupted the dynamic opposing-liquidity target book.

### Fix
- R03 execution overlay now reads the R02 `00_manifest.json` and reloads naked 1m from the exact R02 `warmup_start_date`.
- R02/R03 symbol mismatch is rejected.
- Before any overlay trade is built, persisted stage/lifecycle integer positions are cross-checked against their redundant timestamps. Any mismatch hard-fails rather than producing a misleading backtest.
- The alignment audit is written to `15a_execution_position_alignment_audit.csv`.
- `_structural_stop_before_entry` now bounds-checks the slice and safely returns NaN for invalid/all-NaN intervals instead of raising or allowing negative-index aliasing.

### Causality note
This fix does not loosen causality. It restores the exact historical coordinate system used by R02 and adds a hard audit preventing future positional drift.

# R03.2 - Corrected Microstructure Grain + Frozen-Core Execution — 2026-08-15

## Why R03.2 exists

Review of the user's real R03 report exposed two interpretation-breaking problems. R03.2 is a correction release; it does not add a new liquidity threshold search.

### Problem A - >=3 and >=4 are not the same concrete checkpoint rows

R03 assumed `core_ge4` could reuse features extracted only for `expand_ge3`. This is false at trade-row grain. The first episode stage reaching >=3 pools and the later first stage reaching >=4 pools often have different signal times and different `trade_event_id`s.

Observed on the real R02 report:

- `expand_ge3`: 2,626 concrete trade IDs;
- `core_ge4`: 832 concrete trade IDs;
- overlap: only 383 IDs;
- exact union: 3,075 concrete checkpoints.

Therefore R03.2 extracts Trade Bar and Footprint context for the exact union `expand_ge3 U core_ge4`, once per concrete ID. A row-attachment audit hard-fails if any requested checkpoint disappears, duplicates, or is replaced by an unexpected ID. Footprint *validity* can still be partial via `fp_causal_valid`; missing footprint history remains missing coverage, never a zero/negative signal.

### Problem B - old FVG overlay changed the signal

R03's old FVG overlay started from the first >=4 pool stage and searched for an FVG. The profitable R02 baseline, however, was the later/independent `5m episode_reclaim` opportunity. The old overlay therefore compared different signals and could not answer whether market/limit/hybrid execution improves the 269-trade core.

R03.2 freezes the exact R02 5m core:

- Long only;
- 10bp independent pools;
- first causal >=4-pool episode-reclaim trade;
- R02 stored structural stop;
- R02 stored opposing 4H liquidity target;
- no NY-open gate;
- no fixed time-profit exit;
- same absolute 7-day censor horizon from the original reclaim entry.

Only execution is varied after that same reclaim signal:

1. `reclaim_market` - original R02 next-open market entry;
2. `post_reclaim_fvg_market` - wait for first same-direction 1m/2m/5m FVG after reclaim, then market at the first executable 1m open;
3. `post_reclaim_fvg_limit` - same FVG, proximal resting limit;
4. `hybrid_reclaim_market_fvg_limit` - 50% original reclaim market + 50% FVG proximal limit.

If no FVG appears within the fixed 180-minute wait, the pure-FVG alternatives remain explicit unfilled opportunities; they are never dropped from the denominator. The hybrid retains only the 50% reclaim-market half. The hybrid limit half is cancelled once the open market half resolves the setup; a limit fill on the same 1m bar as the market leg exit is conservatively ignored because intrabar ordering is unknown.

## Hard tie-out rule

Before any corrected execution result is trusted, R03.2 recomputes every frozen core reclaim-market path from naked 1m K using the exact stored R02 stop and exact stored `target_htf240_price`. Outcome and gross return must tie back to R02. Any mismatch hard-fails the script.

This prevents a later execution study from silently changing signal, target book, stop geometry, bar origin, or same-bar semantics.

## Real-report checkpoint replication before delivery

Using the supplied R02 report only (no external market DB required for this replication):

- R02 trade rows: 350,149;
- candidate cohort rows: 3,458;
- exact microstructure checkpoint union: 3,075;
- `core_ge4`: 832;
- `expand_ge3`: 2,626;
- concrete-ID overlap: 383;
- frozen 5m core opportunities: 269;
- frozen 5m core with >=4H source involvement: 149.

These counts reproduce the report review that motivated the correction.

## Validation before delivery

- R01/R02/R03/R03.2 targeted tests: 26 passed.
- New test proves first >=3 and first >=4 checkpoints from the same episode are both retained when IDs/times differ.
- New hard join audit detects missing microstructure checkpoint rows.
- New execution test proves original reclaim baseline ties out and all variants share the exact same stop/4H target.
- New no-FVG test proves every frozen core opportunity remains in every execution denominator instead of disappearing when an FVG never forms.
- R03.2 CLI `--help`: passed.
- Repository-wide import-boundary test still reports 155 pre-existing legacy violations; `research/ict/mss2` + `src/research_common/ict_mss2` add 0.

## Interpretation discipline

R03.2 is still research. The 269-trade >=4/5m reclaim core and the 149-trade >=4H-involvement subgroup were observed on the historical corpus before this correction. R03.2 can correct microstructure evidence and execution comparisons, but it does not magically turn these observations into an untouched holdout. Do not optimize new thresholds from R03.2 output. Prefer stable mechanism uplift across years and cost stress, and accept lower frequency if >=3 expansion still fails.

# R03.3 - ICT Swing Hierarchy + Key Liquidity + Entry/Exit + CVD + Execution Fix — 2026-08-15

## Motivation

The user challenged the remaining assumption that raw `N pools` is the correct definition of important liquidity. They specifically requested differentiation of obvious/key pools, ICT short-term/intermediate-term/long-term swings, entry/exit direction, CVD/order flow, non-MSS entries, other ICT-style entry logic, and repair of the broken R03.2 execution overlay.

R03.3 does not discard the R02 multi-pool exhaustion result. It decomposes it so we can learn whether the real edge is quantity, hierarchy quality, HTF involvement, speed, or their interaction.

## New causal ICT hierarchy

R03.3 introduces a separate swing-on-swing taxonomy:

- ST = already-confirmed base swing;
- IT = ST extreme relative to its immediate left/right ST peers;
- LT = IT extreme relative to its immediate left/right IT peers.

IT/LT classifications carry explicit earliest availability timestamps and are never backfilled. A sweep can use the classification only if it was known by the **start of the sweep 1m bar**. This is deliberately stricter than using sweep-bar close.

The existing order-1/2/3/5 pivot hierarchy is preserved as a different feature family and is not relabeled as ICT ST/IT/LT.

## Key-pool decomposition

Cumulative 10bp pools now record ST-only, IT+, LT, 4H+, multi-TF, external50, clean and structural-key counts. The report explicitly compares quality **at fixed N=1/2/3/4 crossings**, preventing a confound where a higher-quality level merely appears later in a larger episode.

No learned or hand-tuned weighted `liquidity_score` is introduced in this version.

## Entry / exit comparison

MSS is no longer treated as mandatory. Existing causal R02 entries are compared on hierarchy-defined stages:

- stage reclaim;
- episode reclaim;
- structural MSS market;
- structural MSS+FVG limit.

Targets are compared independently: nearest / pool2 / multi-TF pool / 1H+ / 4H+ / 1D+ / 2R / 3R / 5R. The 7-day boundary remains censoring only; no short fixed time-profit exit is added.

## CVD

CVD is explicitly a non-ICT ETH microstructure extension. It is rebuilt causally from completed 1m trade-bar delta within the current episode and includes episode recovery plus 3m/5m/15m bullish price-vs-CVD divergence diagnostics.

## R03.2 execution bug fix

The R03.2 real report showed 798 rows, all `reclaim_market`, and zero FVG signals. Root cause: datetime integer-unit conversion could compare microsecond integer arrays with nanosecond `Timestamp.value`. R03.3 uses direct `DatetimeIndex.searchsorted(Timestamp)` instead. It also preserves no-target opportunities rather than dropping them.

Hard audit now requires, for every 1m/2m/5m FVG timeframe:

- every frozen core ID is present;
- exactly four execution variants per core ID;
- at least one non-vacuous FVG signal;
- exact expected row count;
- original reclaim-market outcome/gross return ties to R02 before comparing variants.

## Performance engineering

The first hierarchy-pool implementation repeatedly concatenated DataFrames inside the episode loop and was rejected as too slow. It was replaced with pre-compressed level tuples plus incremental per-episode in-memory clustering. On the supplied R02 report, hierarchy construction itself runs in seconds for the full 44,723 stages. Wide duplicate review tables were also reduced to essential audit columns; the primary raw trade extract is limited to Long / 5m / episode-reclaim rows while all entry/target summaries still use the complete in-memory population.

## Preliminary sanity check on the supplied R02 corpus (not an untouched holdout)

Before the user's R03.3 run, the new hierarchy decomposition was sanity-checked against the existing R02 report only. At the 5m Long episode-reclaim fixed-N crossing, quality matters materially:

- N=3 + no 4H+ pool: 555 opportunities, 2x PF ~0.757;
- N=3 + 4H+ pool: 297 opportunities, 2x PF ~1.192;
- N=4 + no 4H+ pool: 120 opportunities, 2x PF ~0.760;
- N=4 + 4H+ pool: 149 opportunities, 2x PF ~1.502;
- N=4 + LT pool: 151 opportunities, 2x PF ~1.394;
- N=4 + external50 pool: 167 opportunities, 2x PF ~1.306.

The 4H+ N=4 subgroup remains >1 in each year on this already-observed corpus. These are **diagnostic replication results**, not permission to promote or tune. R03.3 is designed to expose the full hierarchy/entry/exit/CVD relationships for disciplined review.

## Validation state before delivery

- R01/R02/R03/R03.2/R03.3 targeted tests: 31 passed.
- R03.3 CLI help/compile: passed.
- Full real R02 hierarchy functions were exercised on 135,725 lifecycle levels / 44,723 sweep stages.
- Full hierarchy pool enrichment: 44,723 stages and 65,893 cumulative pool snapshots in a few seconds in the delivery environment.
- Real naked-K execution overlay cannot be run in the delivery container because the user's local market DB is not included in the report ZIP; synthetic execution tests cover non-vacuous FVG, no-FVG, missing-target and same-stop/target semantics.

# R03.3.1 amendment — post-sweep local MSS + displacement atlas — 2026-08-15

User correction: a valid bullish MSS may reference a small STH that only forms **after** the sell-side sweep; the earlier implementation only allowed pre-sweep references. Added `post_sweep_st` as a separate causal reference mode. The new ST must have pivot position after the sweep, its right confirmation must already be closed, and the MSS break occurs only on a later causally eligible bar. Existing `recent` and `structural` pre-sweep modes are preserved for direct comparison.

Also removed any implicit notion that strong displacement needs one mechanical cutoff or must exceed the attack into the extreme. Added continuous displacement/attack/path/body/FVG features and frozen 2023-2024 quartile payoff tables, plus direct reversal-vs-attack buckets. These features are research descriptors only and do not gate trades in R03.3.

Validation after amendment: 35/35 targeted ICT MSS2 tests passed. New regression tests explicitly construct sweep -> newly formed post-sweep STH -> causal confirmation -> later bullish MSS, verify pre-sweep modes do not steal that reference, audit post-sweep reference timing, and verify displacement research retains non-monotonic quartiles rather than assuming Q4 is best.

# R04 - Multi-Horizon Liquidity Opportunity Atlas — 2026-08-15

## Motivation

The user clarified that the research should not force ETH liquidity reversals into a single holding period. A setup that reliably captures ~0.5% within a few hours can be valuable if frequent, while a rarer major reversal may deserve 1–5+ days if it keeps delivering toward higher-timeframe liquidity. The key research question is therefore not “short-term or swing?” but **which causal liquidity event properties predict each opportunity horizon**.

The user also described a discretionary management habit: realize a short-term portion once it can cover the original stop loss plus transaction costs, then let the remainder run with a trailing structural stop. R04 does not optimize split ratios yet. It only adds an algebraic feasibility diagnostic for how large the partial would need to be at +0.5%, +0.75%, or +1.0% to cover original structural risk plus costs if the remaining size later lost at the original stop.

## Frozen R04 research grain

R04 starts from the completed R03.3 causal hierarchy report and the R02 causal trade/structural-target report. The concrete opportunity grain is:

- Long only;
- 5m `episode_reclaim` entry;
- one concrete R02 `trade_event_id` per causal stage;
- hierarchy cohort memberships are collapsed into causal flags on that same concrete trade;
- rule scoreboards take only the **first qualifying stage per episode**, so one liquidation episode cannot inflate frequency by being counted again at N=2/N=3/N=4.

R04 does **not** assume `N>=4` or `4H+` is the final strategy rule. It retains all hierarchy crossing opportunities so frequency and horizon quality can be compared at fixed causal definitions.

## Future-label separation / no leakage

The R03.3 convenience extract contains realized target return columns. R04 explicitly strips every realized `target_*` result from its causal feature table. Only the frozen opposing 4H target **price** is retained because that price was already known causally at entry in R02.

Outputs are physically separated:

- `03_opportunity_features_causal.csv.gz`: entry-time-only liquidity hierarchy, consumption, risk, session and optional completed trade-bar context;
- `04_opportunity_future_labels.csv.gz`: all future MFE/MAE/target/continuation labels.

A hard audit fails if any MFE/MAE/close-return/TP-before-stop/post-target future label leaks into the feature table.

## Multi-horizon target atlas

Every causal entry is followed on naked 1m K with the original structural stop. Fixed price targets are diagnostics rather than forced time exits:

- +0.3%
- +0.5%
- +0.75%
- +1.0%
- +1.5%
- +2.0%
- +3.0%
- +5.0%

Path windows:

- 1h / 3h / 6h / 12h;
- 1d / 2d / 3d / 5d / 7d / 14d.

No fixed time stop is added. A horizon is only an observation/censoring window. Same-bar fixed target and structural stop are pessimistically resolved as stop-first.

Nested opportunity labels are intentionally non-exclusive:

- short rebound: +0.5% before structural SL within 6h;
- stronger short rebound: +0.75% within 12h;
- medium: +1.5% within 1d;
- medium/swing: +2% within 2d;
- swing: +3% within 3d;
- major reversal: +5% within 7d.

A trade can satisfy several nested labels. R04 therefore studies `P(1% | 0.5%)`, `P(2% | 1%)`, `P(3% | 2%)`, and `P(5% | 3%)` rather than assigning one permanent horizon class at entry.

## Right-edge censoring

R04 never treats an incomplete future window as a failure. If the research period ends before a full horizon is observable, the label remains censored unless the target or stop has already resolved the question. This is particularly important for 3d/5d/7d/14d labels near 2026-06-30.

## Opposing 4H liquidity as a decision point

R03.3 repeatedly showed opposing 4H liquidity as a strong first target, but R04 no longer assumes it must be a full exit. For trades that reach the frozen opposing 4H target before structural stop, R04 starts a new continuation observation from the **next 1m bar** and records:

- post-4H additional MFE / MAE over 6h, 12h, 1d, 2d, 3d, 5d, 7d;
- total MFE from original entry after the 4H touch;
- how much of later MFE would have been captured by fully exiting at the first 4H target.

Starting on the next 1m bar avoids claiming an intrabar path after the same bar first touched the 4H target.

## Partial-profit feasibility diagnostic

For a short target `t`, original structural risk `r`, and assumed total round-trip cost `c`, R04 reports the fraction `f` needed so that taking profit on fraction `f` at `t` can cover the loss of the remaining fraction at the original stop plus total costs:

`f >= (r + c) / (t + r)`

This is an algebraic diagnostic only. R04 does **not** simulate a split-position strategy, move-to-breakeven rule, runner target, or trailing stop. Those remain downstream execution/position-management research only if the path atlas supports them.

## Frozen descriptive rule scoreboard

To make frequency-vs-horizon tradeoffs visible without parameter sweeping, R04 reports first qualifying stage per episode for a small frozen set:

- any reclaim;
- IT+ / LT / 4H+ involvement;
- N>=2/3/4 plus structural-key liquidity;
- N>=3/4 plus 4H+;
- N>=3/4 plus LT.

Each rule reports trades/month, short/medium/swing/major hit rates and 2x-cost resolved PF for representative fixed targets. These are descriptive comparisons, not optimized admission rules.

## Trade-bar context

By default R04 recomputes causal 1m trade-bar context for the full opportunity universe rather than reusing the sparse R03.3 checkpoint subset. It only summarizes the pre-frozen `absorption_mechanism` and `flow_recovery` flags against short/medium/swing labels. No new trade-bar threshold search is introduced in R04. Use `--skip-tradebar` only for a faster path-only run.

## Performance / engineering

The future label table has >100 columns. An initial implementation produced Pandas `DataFrame highly fragmented` warnings due to repeated column insertion. It was rejected and replaced with NumPy arrays accumulated in a dictionary and one final DataFrame materialization. The full-history path queries use segment-tree first-threshold searches plus range min/max indexes instead of scanning each 14-day path bar-by-bar.

## Validation before delivery

- R01/R02/R03/R03.2/R03.3/R04 targeted tests: 42 passed.
- R04-specific tests: 7 passed.
- Tests cover same-bar target/stop pessimism, right-edge censoring, post-4H next-bar continuation, partial-risk algebra, first-stage-per-episode dedupe, cohort flag collapse, and hard future-label exclusion.
- Full R04 main-entry synthetic smoke passed end-to-end: report files + causal audit + GPT review pack generated.
- R04 CLI `--help` passed.
- Repository-wide import-boundary scan still has 155 pre-existing legacy violations; `research/ict/mss2` + `src/research_common/ict_mss2` add 0.

## Next decision after the real R04 run

Review R04 in this order:

1. Is there a high-frequency short-rebound cohort with ~0.5–1.0% target thickness that survives 2x costs?
2. Which entry-time hierarchy/consumption features materially increase the transition from 0.5% -> 1% -> 2% -> 3% -> 5%?
3. For winners that reach opposing 4H liquidity, how much additional MFE remains over the following 1–7 days, and how much MFE is currently surrendered by a full 4H exit?
4. Is short partial-profit risk coverage mathematically feasible for a meaningful share of opportunities without requiring nearly 100% of the position to be closed?
5. Only if these are stable across years should R05 test actual partial+runner/trailing execution or a multi-horizon opportunity classifier.

# R05 — Entry Timing × Structural Stop × Runner Atlas — 2026-08-15

R04 showed that the major-reversal edge can survive a 5% diagnostic target across years, while the same setup's median full-episode SL is ~1.52%. User feedback correctly identified that this stop is probably a final thesis invalidation, not an appropriate stop for every short/medium horizon. The user also emphasized a practical split: 1m/2m may be useful for earlier entry after a deep sweep, but trailing should avoid 1m noise and should instead migrate on 2m/5m/15m intermediate/long-term structure or unusually strong completed bullish displacement/FVG anchors.

R05 therefore freezes an apples-to-apples same-stage 1m/2m/5m reclaim comparison, a structural initial-stop atlas (episode / qualifying-stage / reclaim-leg / signal-bar / entry-time ITL), MAE-before-target diagnostics, and structural runner trails based only on 2m/5m/15m ITL/LTL or causal displacement shock anchors. No 1m trailing is implemented. New stops activate only after the confirming bar has closed and only move upward. 3%/5% targets remain right-tail diagnostics; 14d remains censoring rather than a time exit.

Existing R02/R03.3 report sanity check (not new holdout): for N>=4+4H, 1m reclaim enters ~8.5m earlier than 5m at the median, but episode-extreme risk only improves from ~1.52% to ~1.48%; therefore earlier entry alone does not solve wide risk if the invalidation remains the full episode extreme.

R04's reported Pandas fragmentation warning was also fixed in the cumulative patch by column-projecting the future-label table before repeated scoreboard slicing. Replaying the real R04 report produces no PerformanceWarning.

# R05.1 — Exclusive Return-Range Reporting — 2026-08-15

User requested that short-return research not mix in trades that later become long-tail winners. Audit confirmed R05 originally retained nested targets only, so a >=5% winner also appeared in +0.5% winner diagnostics. R05.1 preserves that nested view for opportunity-upgrade probabilities but adds mutually exclusive future MFE buckets before the frozen episode-extreme thesis stop: <0.3%, 0.3-1%, 1-3%, 3-5%, >=5%. Same-bar stop/high is pessimistically stop-first; incomplete right-edge paths are not force-classified. Bucket labels are reporting-only future outcomes and are hard-separated from causal features. Dedicated per-bucket subreports now cover entry timing, initial structural stops, MAE-before-target, yearly stability, and structural runner behavior.

# R05.2 — Initial-stop performance repair — 2026-08-15

Real full-history execution appeared stuck at `structural initial-stop atlas`. Root cause was algorithmic: latest causal 2m/5m/15m ITL/LTL lookup rebuilt/copied/concatenated/sorted hierarchy DataFrames for every opportunity. Replaced with once-per-timeframe NumPy/searchsorted lookup preserving exact causal IT->LT availability semantics. Added stage progress reporting and cached TP/MAE path queries across stop variants. Synthetic lookup benchmark showed ~1,100x hot-path speedup (500 queries over 3,000 hierarchy rows); 52 targeted tests pass. No trading logic changed.

# R06 — Adaptive Risk × Protected Structure × Position Lifecycle — 2026-08-15

R05 changed the project from a fixed-target event study into a viable candidate trading family. The important frozen result is the broad Long `N>=3 + (4H OR LT)` liquidity-exhaustion family: roughly 9–10 nominal opportunities/month, with 1m/2m/5m reclaim distributions very similar and 5m LTL structural trailing producing ~1.24–1.27 2x-cost PF across 2023/2024/2025/2026. R05 also showed that tightening the initial SL to signal-bar structure destroys too many eventual winners, while trailing too quickly on 2m/5m ITL also destroys the right tail. Major >=5% winners have multi-day right tails, so fixed 5% remains a benchmark rather than a promoted exit.

User clarified the system objective: do **not** optimize toward an ultra-strict low-frequency sample with a cosmetically high PF. A desirable live family should preserve a useful opportunity stream and produce a steadily rising account curve. Different setup qualities may use different risk, and one causal add-on is acceptable when the trade proves itself. Small 0.5–1% winners and multi-day / multi-week runners may coexist in the same family.

R06 therefore freezes the broad R05 family and stops adding hard entry filters. Initial setup quality changes risk budget instead of trade admission. Critically, a later N=4 episode stage may **not** retroactively upgrade the risk of an earlier N=3 entry. Higher initial tier is only allowed when the first causal N>=3 stage itself already jumps to >=4 pools; A+ additionally requires both 4H and LT liquidity already present at that same stage.

R06 also separates `anchor formation` from `stop promotion`. A causal 5m/15m ITL/LTL is only a candidate low when first known. To become a protected stop anchor it must later receive additional bullish proof: a later close above a frozen confirmation high that was already knowable when the low became causal. A 15m q95 bullish displacement + FVG remains a separate immediate protected anchor candidate. 1m is still prohibited for trailing structure.

Frozen R06 management comparison set:

- R05 immediate 5m LTL baseline;
- protected 5m LTL only after later HH close;
- protected 5m LTL OR 15m q95 bullish displacement+FVG;
- protected 5m LTL until +3% causal milestone, then protected 15m ITL/LTL;
- protected 5m LTL until +3%, then protected 15m LTL only.

No fixed TP and no time stop are introduced. +3/+5/+10% are state/right-tail diagnostics only. Paths continue until a structural stop or the end of available market data.

R06 optionally allows **one** add-on, but only after a protected 5m LTL has been promoted by a later HH close. It is risk-recycling, not averaging down: the add-on uses only risk capacity freed by the common promoted stop, and total open risk to that stop cannot exceed the setup's configured risk budget. Total notional is also capped.

Capital research uses one-ETH-position semantics. Independent overlapping base episodes are skipped and reported; the internal add-on belongs to the same setup. Risk schedules are a small frozen diagnostic set rather than optimized parameters: equal 1%, conservative tiered 0.75/1.0/1.25%, and full tiered 1.0/1.5/2.0% for B/A/A+ respectively. All stay <=2% setup risk.

The primary R06 acceptance criterion is equity quality rather than PF alone. Reports include daily mark-to-market MDD, drawdown duration, Ulcer index, log-equity trend R², positive-month/quarter rate, rolling-90d positive rate, longest gap between entries, loss streak, exposure, 1x/2x/3x cost stress, overlap suppression, and equity after zeroing the top 5/10 winners. Per-year equity summaries are mandatory so a smooth full-sample curve cannot hide a bad year.

Engineering validation before delivery:

- R01→R06 targeted tests: 58 passed.
- Full synthetic R06 main-entry smoke passed end-to-end: protected anchors → adaptive paths → risk sizing → overlap allocator → daily MTM curves → yearly/monthly scorecards → causal audit → GPT review pack.
- Smoke rerun with `RuntimeWarning` promoted to error passed; no all-NaN aggregation warning remains.
- Portfolio daily MTM generation is O(days + trades) per scenario via a monotone trade pointer rather than per-day DataFrame rescans.
- Repository-wide import-boundary scan still reports the same 155 pre-existing legacy violations; R06 adds 0 in `research/ict/mss2` / `src/research_common/ict_mss2`.

R06 is still research-only. No risk schedule, add-on, or management variant is promoted before the real 2023–2026 run is reviewed for equity smoothness, yearly stability, 2x/3x costs, overlap, and winner concentration.

# R06 real-run review — 2026-08-16

The user-supplied 2023-01-01 -> 2026-06-30 R06 report confirmed that the broad family remains executable at roughly 9.3 opportunities/month, but it **did not** meet the desired "steadily rising" standalone account-curve standard.

Representative 2m reclaim + R05 immediate 5m-LTL management + conservative tiered risk:

- 1x cost: about +83.5% total return, daily MTM MDD about -14.5%, ~61% positive months, all calendar-year slices positive, log-equity trend R2 ~0.72.
- 2x cost: about +30.7%, MDD about -22.7%, ~54% positive months, 2024 negative, log-equity trend R2 ~0.17, longest drawdown roughly 795 days.
- 3x cost: negative total return / PF around 1, with materially larger MDD.
- Removing the top 5 winners under 2x costs drives ending equity below 1.0, showing strong right-tail dependence.

Risk-tier decomposition on the 2m / immediate-5m-LTL baseline at 2x cost also invalidated the original ordinal A tier assumption:

- B: ~277 trades, PF ~1.13;
- A = fast N4 without the full key-liquidity context: ~26 trades, PF ~0.83;
- A+ = fast N4 plus both 4H and LT key liquidity already present at the causal qualifying stage: ~90 trades, PF ~1.73.

Therefore **fast N4 alone is not promoted to higher risk**. Setup-quality risk scaling remains a useful concept, but the tier semantics need redesign rather than more hard filtering.

R06 delayed protected-LTL promotion (wait for a later HH after LTL formation) was too slow and underperformed the immediate R05 LTL baseline. Slow 15m runner management preserved more 5%/10% right tail but worsened whole-position drawdown. This supports future base+runner research rather than using one trailing speed for the entire position. R06 add-on V1 is frozen as rejected: the risk-recycled add-on materially increased drawdown and did not reliably improve return.

The 795-day underwater period is considered unacceptable for a standalone first live strategy. The interpretation is not that the SSL-exhaustion Long edge disappeared; rather, this family is right-tail / regime-clustered and should not be expected to smooth the whole account alone. The next research step therefore broadens into complementary ICT-style families instead of tightening the existing Long setup further.

# R07 — ICT family expansion atlas — 2026-08-16

User correction: an earlier conversational shorthand, "BSL sweep -> Short is bad", was too coarse. The actual R02/R03 research never treated sweep-only as the final short entry; sweep-only was an event-study control and real entries required reclaim / MSS / FVG confirmation. R07 therefore re-audits BSL reversal only with actual confirmation and separately studies new limit-entry reversal mechanics.

Pre-R07 audit using the already-completed user R02/R03.3 reports (not a new holdout): confirmed BSL-reversal Shorts still fail across the existing entry/target set. Across episode reclaim, structural MSS market, MSS+FVG limit, and refreshed post-sweep-ST MSS, no >=30-sample BSL subgroup produced 2x-cost PF >1 across nearest liquidity / pools / 1H / 4H / 1D / 1R / 2R / 3R / 5R target diagnostics. The best observed existing subgroup was still below 0.9 PF. This **does not reject every possible BSL reversal**, because R07 introduces a different confirmed-state -> FVG resting-limit corridor architecture with local FVG invalidation and a frozen opposite-FVG objective.

R07 intentionally broadens the research into complementary mechanisms rather than adding more filters to R06:

1. **Proper BSL reversal / SSL reversal audit** — only actual reclaim/MSS/FVG-confirmed entries; sweep-only is never a trade. Results are reported across nearby structural targets as well as HTF/fixed-R targets so a short scalp is not unfairly judged only by a 4H objective.
2. **Liquidity expansion continuation, both directions** — after key liquidity is consumed, price must close through the consumed boundary, produce a same-direction FVG, then a resting limit waits at FVG proximal or CE. No market chase is allowed. A pre-FVG opposite candle overlapping the imbalance is recorded as an order-block context flag, not a gate.
3. **Confirmed reversal -> FVG corridor scalp** — after an already-causal episode reclaim, wait for the first same-direction FVG and use proximal/CE resting limits. Compare the original reclaim structural stop with a local FVG-invalidation structural stop. The small-range objective is a causally-existing, still-unrebalanced opposite FVG frozen when the order is placed. The objective must still be ahead of the market at that moment; if it is delivered on a completed bar before the limit fills, the setup is stale and cancelled.
4. **Complementarity diagnostics** — monthly opportunity activity and same-hour overlap versus the R06 Long family. Sensitivity variants are deduplicated to one earliest family/episode opportunity before counts, so CE/proximal, quality-cohort, and stop variants cannot inflate frequency.

R07 does not use an NY-open gate or other equity-index session prior. It uses ICT source material only to define causal candidate mechanics; ETH data must validate them.

Engineering / causality notes:

- Legacy R02 `R02_TRADE_...` IDs repeat across 1m/2m/5m in older reports. R07 uses the existing safe positional global-ID repair after verifying feature/label row-order identity; unsafe many-to-many outcome joins are forbidden.
- 1m/2m/5m FVG candidate detection is vectorized; only actual gaps enter the Python lifecycle loop. Full-rebalance lifecycle remains causal on the 1m clock.
- Small FVG families are limit-only and model maker-entry + taker-exit baseline cost as 0.08% round trip, with 2x/3x stress reported.
- FVGs become active only after the FVG bar closes. Limit entry begins on the next eligible 1m bar. Stop may trigger on the fill bar; target cannot, preserving pessimistic same-bar semantics.
- Synthetic full-main smoke produced real continuation and corridor limit rows with zero causal-audit violations and zero market entries.
- R01->R07 targeted MSS2 regression tests: 68 passed before final packaging.
- Repository import-boundary scan still has 155 pre-existing legacy violations; R07 adds 0 under `research/ict/mss2` / `src/research_common/ict_mss2`.

R07 remains discovery-only. No complementary family is added to capital before the real user run shows cross-year 2x-cost stability, sufficient opportunity count, acceptable target thickness after maker/taker costs, and genuine low overlap / underwater-period complementarity with R06.


## 2026-08-16 — Default research horizon and manual trade review policy

- New/future ICT MSS2 strategy research defaults to `end_date=2026-08-15 23:59:59`. Historical R01–R06 findings remain labeled with the exact earlier windows actually used; never rewrite past results as if they included July/August 2026.
- Any result used for a current/full-history conclusion must ensure its upstream report dependencies cover the same requested end date. A downstream script must not silently claim 2026-08-15 coverage when its R02/R03.3/R05/R06 source report still ends at 2026-06-30.
- Every strategy-like research version should emit a `manual_review/` directory with recent executable examples. The assistant should point the user to those files for chart-by-chart validation instead of pasting dates into chat.
- Manual-review rows should preserve the causal chain: context/sweep time, confirmation time, FVG/order activation when applicable, fill time/price, frozen stop/target, exit/outcome time, and short human-readable trigger/exit logic.

# R08 — Full-Trend ICT Structure Foundation — 2026-08-16

User chart review identified a foundational problem more important than another strategy parameter pass: local pivots can be mechanically valid yet still be too small to represent the ICT liquidity a discretionary trader actually watches.  The new requirement is to see the complete large structural leg from origin to terminal before qualifying historical liquidity.  STH/STL are construction inputs only; future trading liquidity should come from IT/LT structure.

R08 therefore pauses P&L research and rebuilds the structure foundation according to classical ICT recursive hierarchy: ST -> IT -> LT.  The Episode-12 imbalance-rebalance intermediate-swing extension is explicitly kept separate rather than silently approximated.

A completed leg is built between opposite classical LT anchors, not by taking a rolling-window maximum/minimum.  Internal ITH/ITL sequences are audited for monotonic directional integrity: bullish legs require higher ITHs and higher ITLs; bearish legs are the mirror.  A post-terminal close through the latest opposing IT level is required as an IT-level BOS before the whole prior trend is considered causally complete for future liquidity qualification.  Research scales >=3%, >=5%, and >=7% are reported independently; these thresholds are CoinBacktest sensitivities, not ICT definitions.

For future historical liquidity, a completed bearish leg contributes only its origin LTH plus internal ITH retracement highs (BSL); a completed bullish leg contributes only its origin LTL plus internal ITL retracement lows (SSL).  ST-only swings are excluded.  Because the reversal BOS itself consumes the last opposing IT level, every candidate is checked for consumption before its activation timestamp; consumed levels are not kept active.

R08 is intentionally not a trading-strategy report.  Manual chart validation is a mandatory gate before these levels replace the existing liquidity universe.  Review `manual_review/01_recent_30_completed_clean_trend_legs.csv` first, then `manual_review/02_recent_60_active_key_liquidity_levels.csv`.

Engineering validation before packaging: 72/72 targeted MSS2 tests pass; a 120-day / 172,800-row 1m synthetic multi-timeframe smoke produced hierarchy -> complete legs -> BOS -> trend-qualified liquidity in ~2 seconds with zero causal-audit violations.  Repository import-boundary scan remains at the same 155 pre-existing violations; R08 adds 0.

### R08.1 - Native vs nested full-trend liquidity projection correction
- Fixed R08 cross-timeframe projection contamination.
- Canonical full-trend liquidity is now native-timeframe IT/LT only.
- Lower-timeframe IT/LT nested inside a higher-timeframe completed trend is retained in a separate taxonomy because preliminary real-data review suggests it may be especially relevant for SSL->Long.
- Higher-timeframe swings projected into lower-timeframe trends are rejected.
- Added direct bare-1m projection impact atlas with win rate, mean gross/net return, PF, and positive-expectancy flag at 1x/2x/3x costs.
- No claim that stricter native-only structure automatically improves profitability; this is explicitly tested.
- MSS2 targeted regression after correction: 76/76 passed.

# R09 — ICT Liquidity Quality × Execution Atlas — 2026-08-16

R08.1 corrected the ICT structure foundation and showed that full-trend context materially changes the meaning of lower-timeframe intermediate liquidity.  A narrow diagnostic cohort — completed 1H trend context with nested 15m/30m ITL SSL — showed strong next-day Long drift after physical-level/15m-episode deduplication, but only ~99 events.  That result is **not** converted into a hard admission filter.  R09 deliberately re-expands the entire corrected R08.1 native + nested IT/LT liquidity universe so frequency is preserved.

R09 defines the independent statistical unit as a *causal root sweep opportunity*: the first same-side physical liquidity sweep after a 15-minute inactivity gap.  Multiple trend-context rows for the same physical swing are collapsed before counting.  Initial setup quality uses only levels swept on that root minute and context already activated at that minute.  Later sweeps during the next 15 minutes are stored as `future_cascade_*` diagnostics only; they never upgrade/downgrade the initial tier or entry admission.

The initial context ladder is structural rather than outcome-fitted: C=15m completed-trend context, B=30m, A=1H, A+=4H.  Native/nested, IT/LT, trend magnitude, age, simultaneous root-level count and context confluence remain separate causal attributes.  Risk mapping is deferred until the real R09 run shows whether these structural tiers have stable cross-year expectancy.

Execution is compared on the same root universe: sweep-immediate next-open, episode reclaim market, structural MSS market, post-sweep-ST MSS market, reclaim->FVG resting limit, and MSS+FVG resting limit on 1m/2m/5m.  FVG entries never chase market.  Structural stop uses the causal sweep/confirmation extreme plus the frozen 2bps execution buffer.  Fixed-R and fixed-percent targets are research labels, not promoted final exits; 7d is censoring, not a time stop.  Outcomes report raw net-percent and risk-normalized R expectancy/PF so variable structural-stop width cannot inflate strategy quality.

A frequency-only preview using the real user-supplied R08.1 liquidity report (2023-01-01 -> 2026-08-15) produced 2,331 unique physical first-swept levels and 937 independent root opportunities (~21.6/month): 438 SSL (~10.1/month) and 499 BSL.  SSL context counts were C=167, B=140, A=86, A+=45.  This preview uses no execution outcome and only demonstrates that the corrected structure foundation does not force the research into the prior ~99-event narrow cohort.

Engineering validation before delivery: 81/81 targeted MSS2 tests pass; synthetic R09 execution smoke generates immediate/reclaim/MSS/MSS+FVG/reclaim+FVG trades with zero causal-audit violations.  Reclaim->FVG execution bars are cached once per timeframe rather than re-aggregated per event.

# R10 — Unified ICT Liquidity Trading Engine — 2026-08-16

R10 begins the consolidation phase.  The project stops expanding Sweep/Reclaim/MSS/FVG as independent strategies and instead tests one coherent SSL-Long lifecycle on the corrected R08/R09 liquidity foundation.

Frozen before seeing R10 results:

- Universe: broad R09 SSL root events, not the narrow high-quality ~99-event cohort.
- Unified base entry: 2m episode reclaim, next-open market execution.  On the real R09 report this preserves 400 independent SSL episodes (~9.2/month): C=155, B=129, A(1H)=74, A+(4H)=42.
- Structural MSS is a later causal trade-state upgrade, not a second entry.  The real R09 report has 163 SSL episodes with a 2m structural MSS candidate.
- FVG remains useful execution research, but R10 v1 deliberately does not spawn a second trade after the unified position exists.
- Initial SL remains the causal sweep/reclaim structural extreme + 2bps buffer.  Do not tighten it only to improve R:R.
- Add-on is disabled.  R06 add-on V1 remains rejected.

R10 lifecycle comparison is intentionally small:

1. `full_5m_ltl`: full position follows causal 5m LTL trailing; comparator.
2. `base75_2r_runner25`: no early trailing; 75% Base realizes at 2R.  From the next 1m bar the 25% Runner cannot lose below entry and follows 5m LTL.  If a causal 2m structural MSS exists and price subsequently reaches 3R, later Runner anchors slow to 15m LTL.
3. `base50_2r_runner50`: same state machine with a larger 50% Runner to measure curve smoothness vs right-tail capture.

Risk schedules are frozen diagnostic controls, not optimized parameters:

- `equal_low`: C/B/A/A+ = 0.50% risk.
- `quality_scaled`: C=0.35%, B=0.10%, A=0.75%, A+=0.75%.
- `quality_scaled_no_B`: C=0.35%, B=0, A=0.75%, A+=0.75%.

A+ deliberately does not exceed A risk because R09 showed 4H context was sparse and less year-stable than 1H context.  All schedules are <=0.75% per setup in R10 v1 and notional is capped at 3x.

Portfolio semantics: one ETH net position; overlapping root episodes are skipped while the current position is open.  No time stop.  Right-edge open positions remain mark-to-market rather than being force-closed.  Default market roundtrip cost remains 0.11%, stressed at 1x/2x/3x.

Primary acceptance criterion is the account curve, especially 2x cost: executed trades/month, daily MTM MDD, longest drawdown, positive month/quarter rate, rolling-90d positivity, equity trend R2, per-year stability, and dependence on top-5/top-10 winners.  R10 does not auto-promote the highest-return scenario.

Engineering validation before packaging:

- R01->R10 targeted MSS2 regression: 88/88 passed.
- R10 synthetic lifecycle/risk smoke passed with `RuntimeWarning` promoted to error: Base partial, MSS state upgrade, major-state transition, risk sizing, and causal audit all completed without RuntimeWarning.
- Repository-wide `tests/test_import_boundaries.py` still fails because the current CoinBacktest baseline contains 164 unexpected historical research-import violations not covered by its allowlist; differential scan shows R10 / MSS2 adds 0 new violations.  This is recorded explicitly rather than reported as a pass.

Full repository `PYTHONPATH=. pytest -q` was also attempted on the supplied `CoinBacktest(10).zip` baseline.  Collection stops on five pre-existing missing historical modules/files outside MSS2 (`research.liquidity.liquidity_touch_rebound_v1`, `research.liquidity.panic_selloff_rejection_recovery_long`, `analyze_tool.plugins.panic_low_excursion_rejection`, and related script paths).  Because collection cannot complete, R10 does **not** claim full-suite green.  The MSS2-targeted suite is green and the missing baseline modules are not modified or recreated by this patch.

## R11 — Daily Visible Liquidity Path Atlas
- Reverted from R10 strategy consolidation to path-first discovery after recognizing that admission filters were hiding too much of the daily liquidity process.
- Broad universe now admits every causally confirmed 15m/30m/1H/4H classical IT/LT physical swing while continuing to exclude ST swings.
- Completed-trend/native/nested/3/5/7% labels are descriptive only, not filters.
- At every UTC+8 project-day open, snapshot all still-unconsumed IT/LT liquidity and cluster nearby physical levels into regions.
- Study complete intraday region-sweep sequences and causal first-sweep landmarks (reclaim, 1m/2m/5m post-sweep ST MSS, directional FVG).
- R11 intentionally has no promoted trading rule, fixed TP, risk schedule or capital backtest. The next strategy must be derived from path archetypes rather than pre-filtered setups.


## R11.1 — Continuous-path correction (ETH 24/7)
- R11 day-open framing was corrected before running the research: ETH has no meaningful daily open/reset for this liquidity study.
- Removed 00:00 liquidity freeze and same-day path termination. Calendar date is reporting-only.
- IT/LT levels now activate continuously at real causal `available_time` and remain active until first consumption. Newly confirmed intraday liquidity can enter the map immediately.
- Each root sweep freezes the nearest opposite active/unconsumed liquidity at the exact sweep time; target/path evaluation runs continuously for 24h/48h and may cross midnight.
- Tightened activation/consumption boundary: a bar ending at the moment a swing becomes available cannot retroactively consume that liquidity; first eligible 1m bar starts at or after `it_available_time`.
- R11.1 remains path discovery only: no promoted entry/SL/TP, risk schedule or capital strategy.
- Validation: 95 MSS2 targeted tests passed; 7 R11.1 focused tests passed; warning-as-error cross-midnight smoke passed with causal-audit violations=0.

# R12 — Completed-Trend Swing Sweep -> Opposite Liquidity Path Atlas — 2026-08-16

R11/R11.1 is superseded as the active research framing. The user clarified that the desired study is not a day/session map and not the broad universe of every IT/LT swing. R12 returns to the original path hypothesis: first identify historical ITH/ITL/LTH/LTL that belong to a **completed, causally confirmed trend**, then wait for those still-unconsumed swings to be swept in the future and study whether price truly delivers to still-unconsumed liquidity on the opposite side.

R12 uses R08.1 native + nested-lower-TF completed-trend liquidity contexts, rejects invalid higher-TF projection, and collapses repeated context rows to one physical `swing_id`. A physical swing becomes eligible from the earliest completed-trend context that was causally available while the swing was still unconsumed. Later trend contexts may enrich that swing only after their own activation time; they cannot retroactively upgrade an earlier root sweep.

Each physical first sweep is a root event. Same-bar same-side levels are grouped for the root description, while same-bar SSL+BSL sweeps are retained as ambiguous because 1m OHLC cannot establish the intra-bar order. At root-bar close, R12 freezes the nearest three still-unconsumed completed-trend regions on the opposite side and the nearest deeper same-side completed-trend region. The primary classification starts on the next 1m bar and uses a first-passage race:

- `direct_opposite_delivery`: opposite liquidity is reached before deeper same-side liquidity;
- `cascade_then_opposite_delivery`: deeper same-side liquidity breaks first, but the frozen opposite target is still reached within the path horizon;
- `same_side_continuation_no_opposite_hit`: deeper same-side liquidity breaks and the opposite target is not reached;
- `partial_reversal_ge25/50/75_no_barrier`: neither barrier is reached but a substantial fraction of the target distance is covered;
- `censored_no_barrier_hit`, `no_visible_opposite_liquidity`, and same-bar ambiguity are kept explicitly.

Reclaim, post-sweep 1m/2m/5m ST-MSS and the first directional FVG are recorded only as causal landmarks. R12 does **not** declare Sweep/Reclaim/MSS/FVG to be the entry. Its purpose is to compare successful opposite-delivery paths against failures and identify which pre-sweep, sweep, and early post-sweep characteristics separate them before building another trading strategy.

Primary outputs include physical liquidity lifecycle, root sweeps, full path rows, outcome counts, root-taxonomy success rates, success-vs-failure feature differences, confirmation-landmark uplift, monthly path counts, causal audit, and manual chart-review samples for direct successes / same-side failures / cascade-then-reversal cases.

Engineering validation before packaging: 101/101 MSS2 targeted tests pass. A 120-day / 172,800-row synthetic stress with 1,200 completed-trend contexts produced 858 root/path rows with physical lifecycle ~1.5s, root construction ~4.6s, and full path classification ~14.3s after optimizing the nearest-region lookup to stop once the required nearby clusters are found. Warning-as-error smoke passes with zero causal-audit violations. Repository boundary test still reports 164 pre-existing violations and R12/mss2 adds 0. Full `pytest -q` cannot collect because the current CoinBacktest(10) baseline is missing five historical liquidity/panic modules; this is unchanged by R12.

# R13 — Reversal Quality & Causal Entry Discovery — 2026-08-16

R13 used the completed-trend R12 first-passage roots to compare direct opposite delivery with any path that reached deeper same-side liquidity first. Cascade-then-opposite is correctly a failure for the direct-reversal thesis. Discovery ends 2024-12-31, validation ends 2025-06-30, July 2025 is a 30-day embargo, and the holdout begins 2025-08-01. All 305 available holdout rows remain excluded.

The source coverage defect inherited from the old R12 report was corrected before R13: bare OKX 1m data now covers 2022-01-01 00:00 through 2026-08-15 23:59 with 2,430,720 rows and zero internal gaps. R08.1 and R12 were rebuilt through the corrected end date.

R13 measures liquidity age, sweep morphology, 5/15/30/60-minute expected response, reclaim retention, 1m/2m/5m MSS quality, and FVG timing/width. Entries are next-eligible-bar market or causal resting FVG-limit candidates with frozen opposing-liquidity TP and deeper-same-side structural SL. Same-bar TP/SL is stop-first and a target cannot be credited on an FVG fill bar.

A material reporting bug was found after the initial run: early-response features were causal, but their bin economics were incorrectly credited from root next open. That is an oracle attribution. Script 13.0.1 adds response/FVG market entries and maps root, response, reclaim, MSS and FVG features to their own availability-time entry. The invalid pre-correction early-response PFs are withdrawn. Eight R13 tests pass and the regenerated causal audit has zero violations.

Corrected result: no unfiltered reversal entry survives validation at 2x costs. SSL root/response/reclaim/reclaim-FVG/2m-MSS-FVG PF changes from roughly 1.10/1.12/1.32/1.45/1.22 in discovery to 0.80/0.77/0.63/0.65/0.78 in validation. BSL remains negative or near flat. Profitable SSL behavior is concentrated in 2023 and deteriorates through 2024 into 2025.

Some broad bins are hypothesis-generating only. Youngest-half age, middle-half structural RR, and middle-half 15-minute body/ATR have 2x discovery/validation PF near 1.24/1.41, 1.52/1.41, and 1.87/1.20. Top-five winner removal reduces validation PF to about 0.57, 0.36, and 0.46. Isolated cells such as top-quartile pre-sweep return or Q3 target progress fail neighboring-bin robustness. No filter combination is allowed and no strategy is promoted.

Frozen R13 conclusion: completed-trend reversal contains descriptive path information but none of R13's broad entries establishes a robust edge. Universal reversal, Boolean confirmation, required BSL symmetry, and filter-stacking rescue are stopped. This does not test or reject the later R27 ordered quality-aware state sequence.

Next hypothesis: the dominant event class is deeper-same-side-first, representing roughly 73–84% of discovery and 74–82% of validation races. R14 pivots to a separate acceptance/continuation sleeve toward frozen deeper same-side liquidity, with swept-region reclaim as structural invalidation. It will begin with a tiny market-entry family and keep the holdout sealed.

# R14 — Liquidity Acceptance / Continuation — 2026-08-16

R14 executed the post-reset pivot without adding reversal filters. It uses all non-ambiguous R12 roots with valid deeper same-side completed-trend liquidity, maps BSL sweeps to long continuation and SSL sweeps to short continuation, and treats full reclaim of the root region plus 2bps as structural failure. Root-close, 5m and 15m outside acceptance enter only after their required bars close. Delayed signals are rejected if target or stop was already touched.

The R14 universe contains 753 discovery/validation events and 5,271 model rows. The separate holdout has 322 eligible rows and remains excluded. Six focused tests pass; the full report has zero causal-audit violations.

BSL continuation is definitively rejected: root, 5m and 15m 2x PF is roughly 0.45–0.66 in discovery and 0.49–0.65 in validation, with every year below one. SSL root-close acceptance has useful frequency (4.29 discovery and 6.50 validation trades/month) but PF deteriorates from 1.11 to 0.56.

SSL 5/15m persistence produces high headline PF across splits and years, but only 4–28 trades survive per split/model. Top-five removal reduces discovery PF to 0.17–0.22 and validation to zero/undefined; longest discovery gaps exceed 120 days. The 60/80/100% variants often have identical filled rows because prior structural invalidation removes the weaker extra signals. Threshold optimization and execution-overlay rescue are stopped.

Frozen result: no R14 continuation sleeve is promoted and the holdout remains sealed. R15 will test a small exact fixed-R first-passage ladder for the higher-frequency SSL root-close entry. This is a target/path diagnostic only: no filters, time exits, allocation, runner or portfolio construction.

# R15 — Acceptance Fixed-R First Passage — 2026-08-16

R15 froze the 142 pre-holdout R14 SSL root-close-outside entries and the exact R14 reclaim stop, then replayed 0.5R/1R/2R/3R targets with stop-first same-bar semantics. It added no filter, time exit, runner, risk schedule or holdout access. Four focused tests pass and all causal checks are zero.

Every target fails. Discovery/validation 2x PF is 0.10/0.12 at 0.5R, 0.16/0.35 at 1R, 0.36/0.53 at 2R, and 0.45/0.75 at 3R. Every yearly 2x PF is below one. Median risk is only 0.169%/0.181%, smaller than the 0.22% 2x round trip, and median resolution occurs on the entry bar.

This invalidates the earlier MFE/risk screening intuition: an OHLC bar can show favorable MFE while also touching the stop, and the pessimistic first-passage rule must count the stop. Fixed-R target rescue and Base+Runner construction are stopped.

R16 is the last bounded audit before the next strategic reset. It keeps the same SSL root acceptance and deeper same-side target while comparing region touch, root sweep-bar extreme, and causal close-reclaim-plus-disaster-stop invalidation. If none is robust, the acceptance-continuation branch is archived.

# R16 — Acceptance Structural / Behavioral Stop Atlas — 2026-08-16

R16 changed only the stop for the 142 frozen SSL root-close acceptance entries and deeper same-side target. It compared the R14 region edge, the root sweep-bar extreme plus 2bps, and a causal close-reclaim next-open exit protected by the root-bar extreme. Same-bar target/stop or target/reclaim ties are failures. Six focused tests pass and the report has zero causal violations.

The initial full launch stopped before replay on a duplicate-column merge error because real R14 entries already carried zone/target fields. The merge now adds only absent feature columns and a regression test covers the production schema.

All stop models fail validation and winner resilience. Region-edge discovery/validation 2x PF is 1.12/0.57; root-bar extreme is 1.20/0.36; behavioral reclaim is 0.98/0.50. The root-bar model is profitable in 2024 only (PF 1.56) and loses in 2023/2025. Top-five removal leaves no model near one.

The R14–R16 acceptance-continuation branch is archived. No further filters, persistence thresholds, stops, fixed-R exits, FVG or order-flow overlays may be added to rescue it. Holdout remains sealed and no capital strategy is promoted.

`STRATEGIC_RESET_R16.md` records the mandatory reset after three post-R13 studies. The next independent direction is trend pullback continuation Long/Short: causal 1D/4H trend state, 1H/30m orderly pullback, 15m/5m reclaim/re-acceleration, and next-open 1m execution. It must not inherit the archived q70 score, raw impulse/breakout entry, R13 filters or R14 acceptance rules.

# R13–R16 validation closeout — 2026-08-16

- Full MSS2 targeted suite: 125 passed.
- Pandas 3 timestamp-unit regression in R05 was corrected by normalizing pivot/availability indexes to nanoseconds before integer comparison; causal semantics are unchanged.
- Repository-wide research-import boundary scan still exits nonzero on the large historical baseline. R13–R16 add zero `research/ict/mss2` research-to-research imports; unrelated legacy violations are intentionally untouched.

# R17 — Trend Pullback Re-acceleration Path Atlas — 2026-08-16

R17 froze a new price-only continuation mechanism before outcomes: aligned causal order-1 1D+4H HH/HL or LH/LL state, confirmed 30m counter-trend pivot, 15m close beyond the pivot-bar range, later 5m close beyond the reclaim bar, and next-observable 1m market entry. It does not inherit q70, completed-trend sweeps, R13/R14 filters, raw breakout entry, or the already-rejected Higher-Low resting-limit design.

Risk was predeclared as the local 30m pivot extreme plus 0.25× causal 30m ATR, with a 1.50% maximum distance. Setup expiry was 12 hours; the frozen 4H pivot extreme and 1R/2R/3R were 72-hour diagnostic first-passage targets. Same-bar ambiguity was stop-first. Discovery was 2023–2024, validation 2025H1, July embargoed, and the 2025-08-01 holdout remained sealed.

The visible atlas contained 1,668 aligned pullbacks and 422 executable setups. Long had 227 discovery / 35 validation entries; Short 144 / 16. Long 2× PF ranges only 0.48–0.60 discovery and 0.39–0.64 validation. Short discovery ranges 0.43–0.71. The lone validation Short 1R PF 1.09 has 16 trades and falls to PF 0.45 after top-five removal. No target survives direction, split, cost, or winner-concentration gates.

Independent brute-force replay of all 1,688 setup-target paths against bare 1m K found zero ordering discrepancies; saved return, cost, grouped PF, trade-count, duplicate, and holdout checks reconcile exactly. Thirteen causal checks have zero violations. R17 is rejected and must not be rescued by threshold, ATR-buffer, stop-ceiling, or expiry tuning. No strategy is promoted.

The initial next-path proposal, failed-auction balance re-entry, was stopped during repository audit before R18 implementation. `eth_market_process_portfolio/integration/R02` already tested 452 base-family trades at PF 0.34, 2×-fee PF 0.12, negative returns in every year, and negative top-ten-removed return; loose and strict neighbors also failed. MSS2 will not repeat it.

Proposed R18 boundary: a genuinely independent positioning-unwind atlas using causal Binance 5m OI through `src.data_feed`, aligned to OKX price. It should test price/OI expansion → OI release + price stabilization paths without sweep admission or future turning-point features, with Long and Short separate. This boundary still requires a small precommitment and source-coverage audit before any outcome run.

# R18 — Independent Positioning-Unwind Path Atlas — 2026-08-17

R18 froze an all-market cross-exchange positioning mechanism before outcomes:
the immediately prior completed 1h OKX price move and Binance base OI must build
together, base OI must make its first causal 5m transition from nonnegative to
negative, and the completed OKX 5m close must reacquire the prior bar extreme in
the opposite direction. Binance metrics use an explicit one-minute publication
lag; execution is the first eligible OKX 1m open. OI USD, ratios, future OI, and
oracle turning points are outside admission.

The pre-outcome quality audit found 262,341 pre-embargo Binance rows, zero
duplicates/base-OI nulls, 416 irregular intervals, 188 partial days, 81
nonpositive base-OI rows, and one 10.5-hour gap. R18 rejects invalid/gapped rows
without interpolation. The OKX execution series through validation has
1,838,880 consecutive valid 1m rows. The Binance cache ends at 2026-07-01 07:55
project time, short of the requested 2026-08-15 end and therefore not live-ready.

There are 9,350 visible transitions, 8,604 executable setups, and 8,592 setups
with a full non-embargoed 24h path, producing 34,368 target rows. Gross PF is
only 0.96–1.07. Long/Short discovery 2× PF ranges 0.19–0.45 and validation
0.28–0.56; every primary monthly sum is negative and every 2023/2024/2025 cell
loses after 2× cost. Median risk is only 0.24% discovery / 0.38% validation, so
the high-frequency sign transition has no room for 0.22% stressed round-trip
costs.

Twenty causal checks and eleven independent reconciliation checks pass. A
separate raw-array replay found zero outcome, exit-time, or exit-price differences
across all 34,368 paths; grouped PF reconciles within 3.2e-15. Six focused tests
pass. The 3,376 aggregate holdout candidates have zero outcome rows.

Frozen result: reject the exact build → release → reversal-reacquisition branch.
Do not rescue it with OI magnitude, ratio, funding, volatility, target, stop, or
horizon filters. No strategy is promoted.

R19 may test the economically opposite state already separated before R18:
directional price/OI build → temporary OI release without price reversal → OI
rebuild plus break of the release range in the original direction. This is a
continuation-resumption mechanism, not a filter on failed R18 reversal. It must
be minimally precommitted and stopped if unfiltered 2× discovery/validation
economics fail.

# R19 — Positioning Rebuild Continuation-Resumption Atlas — 2026-08-17

R19 froze the original-direction continuation path before outcomes: completed
1h directional OKX price plus rising Binance base OI, first causal 5m OI release
transition, uninterrupted negative-OI episode for at most 60 minutes, first
nonnegative rebuild, completed OKX 5m close beyond the frozen release bar, and
next-observable OKX 1m entry. Stops use the release-through-rebuild episode
extreme plus 0.25× causal ATR and skip distances above 1.50%. The structural
comparator is one rebuild-time 1h volatility range; 1R/2R/3R remain diagnostic.

The state machine finds 61,779 releases, 14,946 aggregate successful rebuild
breaks, 8,799 visible candidates, 8,591 executable setups, and 8,584 setups with
fully visible paths. Seven late-June paths are embargo-censored. Discovery 2× PF
is 0.26–0.50 and validation 0.41–0.71; every primary monthly 2× sum is negative,
and every year/direction/target cell loses after 2× cost. The 2,661 aggregate
holdout candidates have zero outcome rows.

Nineteen causal checks and eleven independent raw-replay checks pass exactly.
Event/path floats use `%.17g` and round-trip parsing. A final engineering fix
separates 83 genuine time expiries from one right-edge censor; the previous
post-hoc decrement could alter an unrelated episode's audit count. Admission
also now requires finite nonzero build return. Regression tests cover both
issues, and neither changes visible events or economics. Thirteen combined
R18/R19 focused tests pass.

Frozen result: reject R19 and archive the simple Binance base-OI transition
branch. No magnitude, ratio, funding, volatility, stop, target, or horizon
filter may rescue R18/R19.

# Repository candidate audit — LF V10B — 2026-08-17

The saved LF V10B artifact reports 105 trades, dollar PF 8.29, 17,231.75%
compounded return, and 21.13% realized-capital MDD. Git provenance shows the
promoted 21-bar structural stop came from a large full-window 2023–2026 grid of
sources, scopes, lookbacks, buffers, MFE triggers, and hold variants. The same
window supplied later verification; there is no untouched split.

The current implementation is causally ordered and charges 0.055% fee plus
0.020% slippage per side. Independent exact reruns retain large 1×/2×/3×
historical PF, but the candidate fails the master portfolio gates: 2.53
trades/month, 62-day longest flat, 45.2% positive months, 21.13% base MDD and
25.20% 2× MDD. Top-five removal survives, but top-ten removal gives PF below one
and negative compounded return. The saved headline also fails exact parity with
the current data/code path despite the same 105 trades.

`V10B_CANDIDATE_AUDIT.md` therefore classifies V10B as a promising contaminated
prior, not an independent sleeve. `STRATEGIC_RESET_R19.md` records the mandatory
reset after R17–R19. R20, if executed, is limited to one frozen visible-window
component falsification with no grid, no holdout outcomes, and no parameter
selection from the V10B headline.

# R20 — Frozen LF V10B Component Falsification — 2026-08-17

R20 froze the current V10B source path but stripped leverage, capital
compounding, and dynamic quantity from its primary score. The unit is the signed
zero-cost move from quantity-weighted average entry to exit, with conservative
0.15%/0.30%/0.45% deductions at 1×/2×/3× costs. The add-on path remains frozen
inside average entry; no threshold, component, stop, or filter was selected.

Loading stops at 2025-06-30 19:59:59 so the 20:00-labelled 4H bar that closes at
the July boundary is unavailable. Discovery and validation paths must exit
before 2025-01-01 and 2025-07-01 respectively. The final corpus has 82 complete
visible trades, no boundary censor, and no embargo or holdout row. Fourteen
signal, next-open, split, price, uniqueness, signed-return, and cost checks pass.

The result rejects the historical shortcut. Bear V3 Short has discovery/
validation 2× PF 0.72/0.72 on 15/4 trades; Bull Reclaim V2 Long is 0.65/0.22 on
44/3. Both have negative gross expectancy in both aggregate split cells.
Momentum Long has 5/3 trades and no discovery winner. Momentum Short has only
6/2 trades; its apparent validation PF is a two-trade result and discovery
top-five removal leaves a loss. No component passes the precommitted forward-
incubation gate.

Frozen conclusion: V10B's full-window account curve depends on dynamic sizing,
add-ons, compounding, and a few convex winners rather than stable unlevered
component-level expectancy. Do not import its engines, quality multipliers,
component selection, micro filters, or optimized structural stop into MSS2.
R20 promotes no strategy and leaves portfolio construction premature.

# R21 — Canonical Daily Channel Trend Following — 2026-08-17

R21 changed both horizon and mechanism. Before outcomes it froze a primary
daily 20-day breakout / 10-day exit channel and one canonical 55/20 sensitivity,
with Wilder ATR(20), a fixed 2×ATR initial stop, next-calendar-day 00:00 entry,
and next-open channel exit. Long/Short and 2023–2024 discovery / 2025H1
validation simulations reset independently. There are no filters, add-ons,
targets, time-profit exits, leverage, compounding, July rows, or holdout rows.

The raw OKX 1m source has exactly 1,838,880 expected rows through 2025H1, 100%
coverage, and no internal gap. The run emits 41 closed paths and one split-
boundary censor. Eleven signal, execution, boundary, fixed-stop, and cost checks
pass. A separate validator rebuilds channels and ATR from raw bars without the
R21 simulator and passes eight checks on every one of the 42 paths. Three
focused regression tests cover same-day stop-first behavior, later gap-through
pricing, and channel-exit precedence at the next open.

D20 Long has discovery/validation 2× PF 1.76/inf on only 13/1 trades. Removing
the top five discovery winners leaves -54.96%, while 2023 loses -18.85% and
2024 makes +60.73%. D55 Long similarly loses in 2023, wins from three trades in
2024, and loses its lone 2025H1 trade. D20 Short has discovery/validation 2× PF
0.34/0.64; D55 Short has only 3/1 trades and no discovery edge. Every direction
fails the precommitted gate.

The Long variants also have 125–198 day discovery entry gaps and 107–149 day
longest flat intervals, so they cannot address the portfolio coverage gap.
Reject and archive D20/X10 and D55/X20. No channel grid, trend or volatility
filter, trailing stop, pyramiding, sizing, leverage, or holdout result may
rescue R21. No strategy is promoted and portfolio construction remains
premature.

# R22 — BTC-Led ETH Catch-Up First Passage — 2026-08-17

After rejecting funding/basis for having only June 2026 local coverage, R22
froze a new cross-market mechanism. A completed 1h BTC move must exceed two
prior 168h sigmas; ETH must move the same direction but lag its prior-only 720h
beta expectation by 0.75 prior residual sigma. Entry is the next-hour ETH open.
R1/R2 targets and a fixed 1.5× hourly ATR stop use exact stop-first 1m passage
with a 24h boundary-open safety exit.

ETH and BTC each have 1,838,880 complete visible 1m rows and exact timestamp
parity. The feature set contains 30,648 complete hours and 576 signal events.
The run emits 780 closed target paths and one boundary censor. Fifteen causal,
boundary, formula, and cost checks pass; an independent raw-array replay matches
entry open, reason, time, price, and gross return on all 781 paths.

The hypothesis fails despite adequate frequency. R1 Long discovery/validation
2× PF is 0.85/1.13 on 207/36 trades; Short is 0.72/0.92 on 138/25. R2 Long is
0.92/1.14 and Short 0.66/1.08. R1 Long loses at 2× costs in both 2023 and 2024;
removing its top ten discovery winners leaves -46.96%. Short is negative in
every visible year. Low timeout rates show the failure is directional edge,
not path resolution.

Reject the exact BTC-led ETH catch-up rule. No beta/sigma window, impulse/lag
threshold, session, target/stop, or 2025-based rescue is allowed. July and
holdout remain sealed; no sleeve is promoted.

`STRATEGIC_RESET_R22.md` records the mandatory reset after R20–R22. The next
bounded audit may examine the historical panic-wick Long prior, but only as a
full-window-selection-contaminated candidate requiring frozen visible-split
falsification, not as existing proof.

# R23 — Frozen Panic-Wick Structural Long Falsification — 2026-08-17

R23 audited the historical shadow-wick branch's saved `priority_union +
multi_sweep_deeper_higher_low_trail + delay2` candidate. The source claim was
selected on the full 2023–2026 window after many feature exclusions, three
entry policies, nine V1 exits, three delays (81 combinations), and a later
seven-mode exit ladder. R23 froze only the saved rule and treated it as a
contaminated prior.

Trade-derived 1m data contain 1,837,343 of 1,838,880 requested minutes, with
1,537 missing minutes in 72 gaps. Calendar placeholders preserve clock time but
cannot signal, enter, or resolve paths; events require 240 observed prior
minutes. The rule produces 713 events and 230 closed split-reset trades, none
crossing a gap. Twelve causal checks and eight independent state-machine replay
checks pass exactly.

Discovery has 119 trades, 2× PF 1.67, +0.163% mean, and +20.79% compounded
return, but top-ten removal gives PF 0.84 and -4.52%. Validation has 111 trades,
2× PF 0.96, -0.013% mean, -1.76% compounded return, and -11.83% MDD. The year
cells are PF 1.24/1.76/0.96 for 2023/2024/2025H1. Validation frequency rises to
18.5 trades/month while the edge disappears.

Reject and archive the panic-wick shortcut. No Asia/session, prior-move,
delta/taker, wick, volatility, fixed target, delay, or structural-exit rescue is
allowed. July and holdout remain sealed and no sleeve is promoted.

# R24 — Scheduled Funding-Window Unwind — 2026-08-17

R24 tested a distinct scheduled positioning-flow hypothesis without fabricating
missing funding rates. At canonical 00:00/08:00/16:00 funding times, a completed
pre-settlement hourly return above 1.5 prior 720h sigmas triggers an opposite-
direction ETH entry. R1/R2 paths use a fixed 1.5× completed hourly ATR stop,
exact stop-first 1m passage, and the next eight-hour clock as safety exit.

All 1,838,880 requested 1m rows are present. The run contains 376 events, 550
closed paths, no boundary censor, and no July/holdout row. Fourteen causal,
schedule, barrier, split, and cost checks pass; two focused tests cover
same-minute ambiguity and timeout pricing.

The rule has no gross edge. R1 Long discovery/validation gross PF is 1.02/0.92
and 2× PF 0.70/0.72. Short gross PF is 0.98/0.49 and 2× PF 0.70/0.37. Every
visible year/direction loses, R2 fails, top-ten removal is negative, and timeout
rates are 20–32% in most cells.

Reject the scheduled-clock-only unwind. No schedule, z-score, ATR, target, hold,
or direction variant may rescue it. Actual funding/mark/basis history remains
too short for the required pre-holdout design. No strategy is promoted.

# R25 — r0020 Directional-Run Exhaustion Reversal — 2026-08-17

R25 first audited all prior Range-Bar research. R10/R11 had already shown that
high formation activity is two-sided volatility expansion rather than
directional continuation. Other branches had used direction balance, speed,
duration change, flow, footprint, and nonlinear models. R25 therefore froze one
distinct event only: a maximal four-plus same-direction r0020 run, first
opposite completed bar, next-observed 1m entry, run-origin target, and
run-plus-confirmation extreme stop.

The local visible r0020 read has 474,704 overlap-query rows, unique bar IDs, no
required nulls, two source-invalid `start_ts > end_ts` rows, 37,114 zero-duration
rows, and 47,691 rows sharing an end timestamp. Invalid rows reset the sequence;
equal-time rows use deterministic `(end_ts, bar_id)` order and are observable
before the strictly later entry. A monthly source-shape profile records the
temporal zero-duration variation. Bare 1m execution coverage is complete.

The primary simulation closes 13,684 paths plus one boundary censor. Long
discovery/validation gross PF is 1.05/1.08 and 2x PF is 0.41/0.42 on
4,567/2,228 trades. Short gross PF is 1.01/0.98 and 2x PF is 0.39/0.38 on
4,707/2,182 trades. Mean 2x return is approximately -0.205% to -0.224% per
trade; every month, quarter, and year loses. Top-five/top-ten removal and a
one-minute delay are also negative.

Six focused tests and sixteen internal causal/cost checks pass. A separate raw-
source validator independently reconstructs maximal runs, barriers, entry,
first passage, stop-first fill, cost, overlap, and split boundaries and passes
eighteen checks across all visible events and paths.

Frozen result: reject r0020 directional-run exhaustion. No alternate scale,
run length, duration acceleration, delta, footprint, session, target, stop,
confirmation, or model filter may rescue the gross-null family. July and
holdout remain sealed; no sleeve is promoted.

`STRATEGIC_RESET_R25.md` records the mandatory reset after R23–R25. Before R26,
audit repository novelty and pre-embargo availability for a truly distinct
spot-led perpetual-dislocation mechanism. If complete spot/swap overlap does
not exist, do not substitute short 2026 funding/basis/books data.

# R26 source feasibility gate — 2026-08-17

The spot-led perpetual-dislocation proposal is abandoned before implementation.
Local-only reads through `src.data_feed.OKXDataLoader` return zero rows for
`ETH-USDT`, `ETH-USDC`, `ETH-USD-SWAP`, and `BTC-USDT`. No remote download or
proxy substitution was attempted.

`OKXDerivativesLoader` confirms no visible funding, mark-price, or liquidation
history. The available OKX contract OI is daily and begins only on 2024-01-01,
so it cannot support both discovery years. A filename-only books inventory
finds zero pre-July-2025 days; no book archive contents were opened.

The official Binance USD-M 5-minute metrics cache is the one complete,
independent pre-embargo source lane. It has 367,365 rows from 2022-01-01 through
2025-06-30, zero duplicate timestamps, and near-complete top-trader/global
ratio fields throughout discovery and validation. R18/R19 used only base OI
and explicitly excluded these ratios. Repository search finds no standalone
relative-positioning leadership-cross strategy, so R26 may precommit that one
mechanism. Base-OI transitions, taker flow, sweep filters, funding, grids, and
holdout outcomes remain excluded.

# R26 — Relative Positioning Leadership Repricing — 2026-08-17

R26 froze a standalone Binance positioning-ratio mechanism before outcomes.
Top-trader position long share crossing global-account long share arms Long;
the downward mirror arms Short. The spread must retain its new sign until the
first same-direction completed OKX 5m close through the prior bar extreme
within one hour. Entry is the next eligible 1m open, with a two-bar-plus-ATR
stop and cross-time one-hour structural target.

The script physically loads only through 2025-06-30. Visible source quality is
adequate: 262,341 metric rows, 62 combined ratio nulls, 416 excluded irregular
intervals, and complete OKX 1m execution data. The run produces 634 visible
events, 486 executable setups, and 457 non-overlapping primary paths.

The mechanism fails. Long discovery/validation structural gross PF is
1.22/1.20 but 1x PF is 0.68/0.73 and 2x PF 0.39/0.45 on 160/59 trades. Short
gross PF is 0.93/0.95, 1x PF 0.51/0.61, and 2x PF 0.29/0.40 on 169/69 trades.
Every visible year loses at 2x cost. Positive-month rates are only 8.3–33.3%,
top-five/top-ten removal stays negative, and every fixed-R diagnostic is below
one after 1x cost.

Eighteen internal causal checks and seventeen independent raw-source replay
checks pass. Reject both directions. No spread threshold, quantile, OI/taker,
session, confirmation window, stop, target, fixed-R, or ML rescue is allowed.
July and holdout remain sealed; no sleeve is promoted.

# Post-R26 source/mechanism audit — 2026-08-17

A new repository and loader audit finds no defensible R27 under the current
local source boundary. Compression/expansion is already rejected across the
market-process, MHF, impulse, and Range-Bar branches. Absorption, failed price
progress, failed auction, CVD, and large-trade flow are already tested across
MSS2, post-sweep, order-flow, momentum, and panic-wick research. The Q70 reclaim
model is permanently archived after sealed-holdout failure.

Only BTC has a second full visible OKX swap history; its lead/lag catch-up rule
failed in R22. Local SOL/XRP/DOGE/BNB/LTC/ADA/AVAX swap checks return zero rows.
OKX spot is absent, OKX funding/mark/liquidation have no visible rows, contract
OI begins only in 2024, and books/liquidity primitives begin after the visible
cutoff. No sealed archive contents were used.

`POST_R26_SOURCE_MECHANISM_AUDIT.md` therefore leaves R27 unassigned rather than
manufacturing another adjacent threshold. The highest-value unresolved lanes
require complete pre-embargo spot/swap, funding/basis, or books history. The
master goal remains open; July and the 2025-08-01 holdout remain sealed.

# Pre-R27 reproducible local source-readiness gate — 2026-08-17

The post-R26 conclusion was re-audited from the actual local catalog rather
than a guessed symbol list. A new read-only interface in `src.data_feed`
enumerates all timestamped series in the six supported market databases and
all actual series directories in raw trades, books, liquidity primitives,
liquidity maps, and Binance futures metrics. Every aggregate is physically
bounded to `< 2025-07-01`; post-cutoff archive contents are never opened.

The catalog records 28 non-empty pre-embargo SQLite series. The fixed-cadence
price catalog has exactly eight tables: seven ETH-USDT-SWAP timeframes and one
BTC-USDT-SWAP 1m table. Both 1m series contain exactly 1,838,880 of 1,838,880
expected rows. No ETH spot or additional swap price series exists. The only
OKX derivatives series is 547 daily ETH contract-OI rows from 2024-01-01 through
2025-06-30; funding, mark, and liquidation tables each have zero visible rows.

ETH and BTC raw trades each cover all 1,277 visible days. Binance ETHUSDT raw
metrics also cover all 1,277 days, while the normalized 5m table contains
367,365 of 367,776 expected timestamps; its gap-aware positioning mechanisms
are already rejected in R18/R19/R26. Books, liquidity primitives, and liquidity
maps each cover 0 of 1,277 pre-embargo days. Range-Bar and footprint stores span
the visible window but map to activity, continuation, absorption, and run-
exhaustion families already frozen.

Three physical-seal checks pass and no mechanism has both a ready source and
novel economic logic. `00_pre_r27_source_readiness_audit.py` therefore emits
`UNASSIGNED_NO_ELIGIBLE_MECHANISM`; R27 remains unassigned. The result is a
reproducible source gate, not a completed strategy study. The master goal stays
open, no sleeve is promoted, and July plus the 2025-08-01 holdout remain sealed.

# Active-goal blocker audit — 2026-08-17

The same source/mechanism constraint has now survived three consecutive active-
goal continuations. The reproducible pre-R27 audit was rerun from current local
state: 28 pre-embargo SQLite series, six archive lanes, three passing physical-
seal checks, eleven `NO_R27` mechanism decisions, and zero eligible mechanisms.
The currently callable tool/plugin inventory also exposes no OKX, Binance,
crypto historical-market-data, funding, or order-book source.

Further local experimentation would require tuning a frozen family, substituting
a weaker proxy for a missing source, or opening sealed data. All three violate
the research contract. The master goal is therefore blocked, not complete.

Resume only after one of these state changes:

- complete ETH spot plus ETH-USDT-SWAP history is available through
  `src.data_feed` from warmup through 2025H1;
- complete funding plus mark/index history is available over the same window;
- complete historical books/liquidity primitives cover discovery and validation;
- or the user explicitly authorizes establishing a new external source and a
  genuinely new forward seal.

Synthetic data cannot establish a tradable edge. July and the 2025-08-01 MSS2
holdout remain sealed while blocked.

# R27 research correction — Sequential ICT reversal path — 2026-08-17

The source-readiness blocker is superseded by a substantive audit correction:
R13 measured broad liquidity/sweep/response quality and Boolean reclaim, MSS,
and FVG landmarks, but it did not require the full causal order `reclaim → new
post-sweep structure → meaningful MSS → strong displacement → executable FVG
retracement → protected swing`. Sweep-immediate loss and generic MSS/FVG
failure therefore cannot permanently archive this untested mechanism.

`R27_PRECOMMITMENT.md` freezes one S0–S6 sequence, continuous state-quality
fields, a common sweep-invalidating stop, opposite completed-trend liquidity
target, discovery-only selection of the earliest stable divergence, and a
single 2025H1 validation opening. No exhaustive grid, R13 bin combination,
loss-specific rule, holdout read, or external source is permitted. The prior
source audit remains valid for genuinely new data mechanisms, but the required
R27 study uses already-ready causal 1m price and the R12/R13 root universe.

# R27 — Sequential ICT Reversal Path Study — 2026-08-17

R27 completes the research correction with one preregistered S0–S6 causal state
machine. It recomputes failed acceptance, post-sweep impulse/pullback structure,
the break of that meaningful reference, joint displacement quality, a real
displacement-linked FVG proximal retracement, and a later protected swing. S0
is baseline only. All stages use the same sweep-extreme-plus-0.10-ATR stop and
frozen opposite-liquidity target; state signals enter next-open, S5 uses a
resting limit, fill-bar target credit is prohibited, and same-bar ambiguity is
stop-first.

Discovery contains 207 SSL and 306 BSL roots. SSL direct-delivery probability
rises only from 27.05% at S0 to 32.23% at S3, below the frozen +10-point gate.
S2/S3 discovery 2× PF is 1.44/1.55 on 52/42 fills, but both are negative after
top-five removal. BSL is negative at every state. S4–S6 reach and fill counts
collapse before establishing a stable divergence. The discovery freeze is
`NO_DIVERGENCE` for both directions.

The separate one-time 2025H1 validation uses 105 SSL and 101 BSL roots. SSL S2/
S3 2× PF falls to 0.62/0.56 with -0.473%/-0.614% expectancy. BSL remains
negative. The apparent SSL S5/S6 validation gains each contain one fill and are
explicitly rejected as sample noise. Validation reports
`REJECT_NO_DISCOVERY_DIVERGENCE` for both sides.

All internal causal and independent raw-bar replay checks have zero violations.
Discovery physically ends 2024-12-31; validation price reads end 2025-07-31 for
30-day label maturity; the 2025-08-01 holdout remains sealed. R27 promotes no
entry, probe/main lifecycle, risk schedule, or strategy sleeve. The completed-
trend sweep-reversal mainline is now archived at the frozen semantics; no
threshold relaxation, alternate-pivot/FVG grid, R13-bin stacking, or loss-
specific rescue is allowed. The master goal remains open.

A post-validation reporting audit found that S6 stored but had not separately
replayed its tightened protected-pivot stop. A guarded reporting-only replay
added that diagnostic after asserting all frozen entry/outcome/economics rows
and both decisions unchanged. It yields one SSL winner and one BSL stop in
validation, with no discovery S6 fills; this is explicitly non-inferential and
does not alter `NO_DIVERGENCE` or the archived conclusion.
