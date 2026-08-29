# Research Log

## R01 — 2026-08-15

### 研究目标

首次把用户描述的 SOXL ICT 盘前流动性模型量化成严格、可重放、无未来函数的规则。目标是先回答“是否有 edge”，不是先把回测收益调高。

### 已完成

1. 全量机械扫描当前 CoinBacktest 文本源码，并深读 `src/data_feed`、research common、report、progress、causal alignment、现有 swing/replay 实现。
2. 研究完全独立放入 `research/ict/soxl_premarket_mss_fvg/`，未修改旧策略。
3. 数据只通过 `OKXDataLoader`；1m 聚合 2m/5m/15m。
4. 实现 New York DST-aware session、weekday/US-equity-holiday gate、session coverage gate。
5. 实现盘前 extreme + 因果确认 major 15m swing liquidity。
6. 实现 first sweep、causal short-term pivot、MSS、displacement、FVG、limit/SL/TP/cancel replay。
7. 所有高周期聚合显式 `available_time`；当前 bar 不进入自身 displacement baseline。
8. 1m OHLC 无法判断同 bar 先后顺序时按保守路径处理。
9. 实现 1m/2m/5m、liquidity mode、cost 1/2/3x、delay 0/1/2m、displacement 1.25/1.50/1.75x 对比。
10. 实现逐笔 causal audit、完整 research CSV、findings、manifest、平台 `print_full_report` 和 review pack。
11. 合成路径 self-test 通过；新增专项 pytest。

### 工程问题与修复

- 修复 tz-aware DatetimeIndex 转 NumPy 后 object 比较造成的时间比较异常，关键比较保留在 Pandas tz-aware 轴。
- `enforce_single_lifecycle` 的初始时间不用 `Timestamp.min` 做 NY localize，避免极端时间边界溢出。
- 合成路径显式构造 sweep 前已确认 short-term high、FVG retrace fill 和 opposite target，保证生命周期测试真实覆盖。

### 当前冻结定义

R01 暂时采用严格版本：MSS break bar 必须同时是 FVG 第三根。这样会少抓信号，但定义最清楚、最不容易把事后看到的 displacement leg 倒灌进信号。

### 未完成 / 环境限制

当前执行容器无法解析 `www.okx.com` DNS，因此真实 SOXL 1m 历史无法通过 `OKXDataLoader` 拉取。本轮没有伪造真实回测结果，也没有绕开 `data_feed`。

### 下一步

在本地已有 OKX 数据环境运行 R01，先看：
- 1m/2m/5m 哪个存在稳定正向；
- extreme 与 major 15m swing 哪类 liquidity 更有效；
- fill/cancel 结构；
- cost 2x/3x 和 delay 1/2m 是否还活；
- weekday/time bucket 是否只是样本偶然。

如果严格 R01 信号极少，再做 R02：保持 MSS 因果性不变，把 FVG 从“必须发生在 break bar”扩展为“必须发生在 sweep→MSS displacement leg 内”，两者并排对照，不能直接替换 R01。

## R02 — 2026-08-15 — Sweep Episode / Dynamic MSS semantic correction

### Why R02 exists

R01 real-report review showed that the frozen-at-first-sweep MSS reference did not match the intended ICT structure.  Some sweep-to-MSS gaps were very long, because the code kept waiting for a stale pivot captured at the first liquidity cross even after price continued to extend and formed newer structure.

### Semantic corrections

1. Replaced frozen-at-sweep MSS with a causal sweep-episode state machine.
   - first cross opens an episode;
   - completed 1m bars continuously update the current terminal extreme;
   - the MSS reference is the latest causally-confirmed opposing pivot that structurally precedes the current terminal extreme;
   - a direct V can still use a valid pre-sweep pivot;
   - a later W/M leg can replace it with a newer post-sweep pivot before a more extreme terminal print.
2. Opposite premarket target must still be fresh at the sweep.  If already consumed, the setup is rejected rather than pretending old liquidity remains a valid target.
3. Reworked 15m swing significance.  R02 ranks by two-sided excursion, normalized by the median completed premarket 15m range.  The best pivot is tradable only when both-sided excursion is at least one typical 15m range.  Weak candidates remain in reports but are not force-labelled as major liquidity.
4. Aggregate diagnostics no longer pool duplicate extreme trades from both liquidity-mode variants.  Timeframe, level, weekday and time-bucket comparisons are split by `liquidity_mode`.
5. Added an explicit Long/Short comparison and dynamic reference-source comparison.
6. Report root moved under `data/reports/research/ict/soxl/mss/` as the permanent SOXL ICT/MSS location.

### Deliberately unchanged in R02

- strict MSS break bar must still be the third candle of the FVG;
- displacement threshold logic;
- FVG near-edge entry;
- terminal-extreme stop;
- opposite premarket extreme target;
- 0.11% base round-trip cost and conservative intrabar replay.

Keeping these fixed isolates the MSS/liquidity semantic correction instead of changing several strategy dimensions at once.

### Longer-history data path

Added `src/data_feed/alpaca_stock_loader.py` as a reusable Alpaca historical US-equity adapter.  It is not yet treated as equivalent to OKX SOXL perpetual.  SOXL spot history must first pass an overlap proxy audit versus OKX perpetual on the common 2026 period before any long-history spot result can be interpreted as evidence for the perpetual strategy.

### Verification

- R02 script self-test: PASS.
- R01 + R02 + Alpaca loader + causal alignment + review-pack targeted tests: 15 passed.

## R03 — 2026-08-15 — Spot/perpetual structural proxy audit + long-history wiring

### Data now available

User completed the Alpaca SIP split-adjusted SOXL 1m prebuild:

- rows: 1,419,524
- table: `ALPACA_SOXL_1Min_sip_split`
- UTC range: `2019-01-02 09:00:00+00:00 -> 2026-06-30 23:59:00+00:00`

The first timestamp corresponds to New York 04:00, so the required premarket
window is present from the start of the local history.

### Engineering changes

1. R02 accepts `--data-source okx|alpaca` without changing state-machine rules.
2. Both sources are clipped to New York `04:00-16:30` before liquidity/swing/MSS
   construction. OKX's extra 24h synthetic session therefore cannot create
   strategy structure.
3. Alpaca long-history mode uses the already prebuilt SIP / split-adjusted table.
4. Added an efficient data-feed-level local range query so the 2026 overlap audit
   does not materialize the full 1.4M-row table in Python. A persistent timestamp
   SQLite index is created lazily on first range query.
5. Added R03 overlap audit comparing 1m returns, daily rebased price paths,
   premarket extreme timing, external sweep keys, and base R02 setup keys.
6. Proxy gates are declared independently of strategy PnL. They are not tuned on
   the 2019-2026 strategy result.

### Research policy

- `PASS`: Alpaca is acceptable as a long-history **structure proxy**; still not a
  substitute for final OKX perpetual validation.
- `CAUTION`: use long history for hypothesis screening only; do not promote from
  spot PnL alone.
- `FAIL`: stop proxy transfer and investigate structural mismatch before using
  the Alpaca history as evidence for the perpetual.

### Verification

- R02 self-test: PASS.
- R03 synthetic constant-basis overlap self-test: PASS.
- Targeted Alpaca / R01 / R02 / R03 / causal alignment / review-pack tests: 21 passed before final packaging.

### Long-history performance/data-semantics hardening

- Optimized `slice_ny_day` to use monotonic DatetimeIndex search boundaries instead of rebuilding a full-table boolean date mask for every session. This preserves identical half-open session semantics while removing the dominant O(days × full_rows) scan for the 1.4M-row history.
- Alpaca minute bars can omit minutes with no eligible stock trade. Added forward-only, same-day internal densification using only the last already-observed close; no backfill before the first print and no fill after the last print. Synthetic no-trade bars are explicitly flagged and use zero volume/trade_count.
- Added an Alpaca-specific quality gate so legitimate sparse premarket minutes are not mistaken for data corruption, while large raw gaps after 08:30 and truncated/early-close sessions remain rejectable.

## R04 — 2026-08-15 — ICT semantic correction: separate MSS, displacement leg, and FVG

### Why R04 exists

The 2019-2026 Alpaca R02 run produced 2,301 fresh eligible sweeps but **zero**
MSS/FVG attempts. Review showed this was not evidence against ICT. R02 had
incorrectly collapsed three separate concepts into one break-bar requirement:

- MSS break bar had to be a large-body candle;
- that same bar had to close near its directional extreme;
- that same bar also had to be the FVG third candle;
- the complete FVG sequence had to start strictly after the terminal extreme.

That implementation was too strict and did not match the intended ICT process.
R02 remains archived as a failed operationalization; it must not be used as the
strategy conclusion.

### R04 frozen semantics

1. Liquidity / fresh-target / sweep-episode / dynamic terminal extreme remain from R02.
2. **MSS is only the structural event**: after liquidity is taken, a completed
   1m/2m/5m bar closes through the latest causally valid opposing short-term
   pivot tied to the current terminal extreme.
3. **Displacement is the reversal leg**, from terminal extreme through MSS; no
   single candle is required to carry all displacement properties.
4. Base displacement qualification is relative and non-PnL-tuned: outbound
   terminal->MSS directional speed must be at least the inbound
   reference->terminal directional speed. This directly rejects a fast move
   into the extreme followed by a slow grinding reversal.
5. **FVG is separate from MSS**: at least one directional three-candle FVG must
   occur anywhere inside that reversal leg and be fully known by MSS
   confirmation. The MSS candle does not need to be FVG candle 3.
6. The first FVG candle is allowed to contain the terminal extreme; R02's
   `FVG sequence starts strictly after terminal` restriction is removed.
7. If several FVGs exist inside the displacement leg, R04 deterministically uses
   the latest FVG known when MSS confirms. Entry remains the third-candle near
   edge; stop remains terminal extreme; target remains fresh opposite premarket
   absolute extreme.
8. Reversal path efficiency and relative speed are written as diagnostics. They
   are not silently optimized from PnL.

### Verification

- R04 synthetic end-to-end self-test: PASS.
- Explicit unit test proves an FVG may precede the MSS break bar and still be
  selected from the same displacement leg.
- Explicit unit test proves the first FVG candle may contain the terminal extreme.
- R04 + R02 + R03 overlap + causal alignment + review-pack targeted suite: 11 passed.

### Next step

Rerun the 2019-2026 Alpaca long-history study with R04. Only after R04 produces a
real trade sample should profitability, Long/Short asymmetry, timeframe,
liquidity type, annual stability, cost stress, and OKX transferability be judged.


### R04 semantic hotfix — post-terminal MSS reference

- Audit found R04 still inherited one R02 restriction: every MSS reference pivot had to occur before the current terminal extreme.
- This incorrectly rejected the common path `low raid -> final low -> rally -> new small STH -> higher low/pullback -> break that STH`. The short mirror had the same problem.
- R04 now prioritizes the latest causally confirmed post-sweep opposing pivot, including pivots formed after the terminal extreme. If none exists, it falls back to the latest pre-terminal pivot for a direct/V reversal.
- MSS reference and displacement inbound anchor are now separate concepts. A post-terminal MSS pivot cannot define the move *into* the terminal extreme, so relative displacement uses the latest pre-terminal opposing pivot (or the swept liquidity level as a causal fallback).
- Added explicit tests for post-terminal STH MSS selection and independent inbound-anchor selection.

## R05 — 2026-08-15 — displacement discovery + Pandas 3 timestamp hotfix

### Research correction

- Removed the pre-imposed `reversal_speed >= inbound_speed` entry gate.  Displacement strength is now a research variable rather than an assumed formula.
- MSS, displacement and FVG remain separate concepts.
- A post-terminal STH/STL can be the MSS reference.
- A directional FVG may complete before/on/after MSS while the same terminal extreme remains intact; the order activates only once both MSS and FVG are known.
- Retained speed, efficiency, duration, body share, FVG size/location/timing and related path metrics for discovery-period quartile analysis, with 2025/2026 forward periods frozen.

### Critical engineering fix

The first long-history R05 run still produced `2301 fresh sweeps -> 0 attempts`.  Root cause was a Pandas 3 timestamp-unit mismatch introduced by the performance path: `DatetimeIndex.asi8` can follow a microsecond-resolution index while `Timestamp.value` is always nanoseconds.  The resulting 1000x integer mismatch put every sweep beyond the execution frame.  R05 now explicitly converts all search axes to ns before `np.searchsorted` and has a regression test that forces us-resolution indexes.

The pre-hotfix R02/R04/R05 zero-attempt reports are invalid as strategy evidence.

## R06 — 2026-08-15 — remote unconsumed 1H/4H/1D liquidity families

### Hypothesis

Premarket liquidity may not be the only meaningful external pool.  Old, still-unconsumed higher-timeframe swing highs/lows can remain relevant, and farther/older levels may have different edge from the nearest swing.  R06 therefore adds them without pooling the families in the primary result.

### Causal HTF liquidity model

1. Build 1H / 4H / session-1D bars from the same New York stock-session tape used by the SOXL research.
2. A swing becomes known only after its right-side pivot bars are fully closed; `level_available_time` is the causal confirmation timestamp.
3. From confirmation onward, scan completed 1m bars chronologically.  The level is consumed only by the first **strict trade-through** (`high > swing_high` or `low < swing_low`).
4. At each trading day's 08:30 anchor, keep every swing with `level_available_time <= 08:30` and no consumption at/before 08:30.
5. Do **not** keep only the nearest swing.  Every active 1H/4H/1D level is retained with age, distance from 08:30 premarket close and nearest-rank diagnostics.
6. If one 1m bar sweeps several levels from the same timeframe, count one physical family sweep and record the number/prices swept.  Do not multiply one market move into several trades.
7. If different families are swept in the same minute (for example 1H + 4H), retain an exact-minute confluence tag.  Family PnLs remain separate and are not additive.
8. Entry/MSS/FVG/displacement semantics remain R05.  The target remains the opposite fresh absolute premarket extreme so liquidity-source edge is isolated before experimenting with alternative targets.

### Primary family comparison

- `premarket_extreme`
- `major_15m_swing`
- `remote_1h_swing`
- `remote_4h_swing`
- `remote_1d_swing`

R06 reports yearly, Long/Short, timeframe, cost/delay, HTF age/distance/rank, and exact-minute HTF confluence separately.

### Verification

- R06 HTF liquidity tests cover causal consumption, all-active-not-nearest selection, pre-08:30 consumption rejection, same-family same-minute deduplication and causal daily pivot confirmation.
- R05 microsecond-resolution regression remains green.
- R06 self-test: PASS.
- R04/R05/R06 + causal alignment + Alpaca loader + review-pack targeted suite: 21 passed.

## R07 — 2026-08-15 — ICT Semantic Gap Atlas

### Why this revision exists

R06 confirmed that the broad mechanical `sweep -> MSS -> FVG` universe contains many trades a discretionary ICT trader would likely skip, while a large fraction of final stop-outs still achieved meaningful MFE first.  R07 therefore stops tightening entry gates and studies the semantic gap directly.

### Frozen behavior

- R06 liquidity families and HTF unconsumed-liquidity logic are unchanged.
- Post-terminal STH/STL remains a valid MSS reference.
- Displacement remains ungated; no speed/body/overshoot/FVG-size threshold filters a trade.
- FVG may exist before/on MSS or complete after MSS while the same terminal extreme remains valid.
- Stop, target, fee, order-delay and lifecycle rules are unchanged so entry semantics are isolated from exit optimization.

### New causal semantic descriptors

R07 adds signal-time-only features describing:

- terminal extension beyond the swept level and extension relative to the first sweep print;
- how long price closes remain accepted outside the swept liquidity and how quickly it reclaims the level;
- terminal retest count (10bp diagnostic tolerance only, never a gate);
- timing of the MSS reference relative to terminal formation and MSS confirmation;
- existing displacement speed/efficiency/body-delivery variables;
- MSS overshoot;
- number of directional FVGs known at MSS and by signal;
- selected FVG sequence rank, size, timing and entry depth;
- entry progress from terminal extreme toward the opposite target.

Every causal semantic feature carries `semantic_feature_available_time = signal_time`.  The causal audit explicitly checks that no outcome columns exist in the signal-attempt frame.

### Outcome-path labels (post-replay only)

After replay, R07 labels whether a filled trade:

- reached +0.5R / +1R / +2R / +3R;
- hit the opposite-liquidity target;
- stopped within 15 minutes without reaching +0.5R;
- reached +0.5R / +1R / +2R but ultimately finished negative.

These labels are analysis-only and are never available to signal generation.

### Anti-overfit discovery protocol

Continuous semantic-response curves use frozen quantile edges learned only from data through 2024.  2025 and 2026 use those unchanged edges.  R07 reports the entire bucket response rather than auto-promoting the best discovery bucket into a strategy rule.

### Verification

- R07 semantic features preserve the exact attempt universe (no candidate filtering).
- Outcome labels appear only after replay.
- Discovery-only bin-edge test confirms 2025/2026 extreme values cannot move the frozen thresholds.
- FVG-count diagnostics do not change signal semantics.
- R02-R07 ICT + Alpaca loader + causal alignment + review-pack targeted suite: 32 passed.
- R07 end-to-end synthetic self-test: PASS.

## R08 — Liquidity Consumption Maturity Atlas

- Default long-history window narrowed to 2023-07-01 through 2026-06-30 because this setup depends on liquid US-equity price discovery; older SOXL years remain available but are no longer the default discovery sample.
- Kept the entire causal Sweep -> MSS -> FVG candidate universe unchanged. No maturity or displacement feature filters entries.
- Added causal consumption-path diagnostics for multiple mechanisms rather than one strength score: shallow sweep, fast spike/reclaim, progressive extension, long acceptance outside liquidity, pre-sweep near/equal touches, final-terminal reclaim, penetration area (depth x time), progressive extrema and terminal retests.
- Fixed the R07 reclaim semantic ambiguity by measuring `maturity_first_reclaim_after_final_terminal_minutes` strictly from the final terminal onward; it cannot be negative.
- Added discovery-frozen 1D response curves and 2D response surfaces. Discovery is 2023H2-2024; 2025 and 2026 remain forward/late holdout.
- Added explicit opportunity-frequency reporting so profitable sub-buckets are not confused with the size of the broad mechanical ICT universe.
- R08 does not convert any discovered bucket into a strategy rule; the goal is to find stable mechanism shapes first.

## R09 — 2026-08-15 — Mechanism Archetype Validation + causal EQH/EQL pools

### Why R09 exists

R08 showed that several profitable-looking mechanisms can coexist: premarket liquidity often benefits from a real consumption/acceptance phase, some remote HTF sweeps behave more like fast rejection, and major 15m swings can tolerate deeper flushes.  The goal is therefore **not** to AND profitable R08 buckets into a strict filter.  R09 validates overlapping mechanism families while preserving the broad causal trade universe.

### True EQH/EQL liquidity

R08's `near_touch_count` is archived as a coarse proxy only.  R09 adds real equal-high/equal-low pools:

1. Aggregate the same causal 1m tape to 1m/5m/15m source bars.
2. Detect same-side causal swing pivots; every pivot is usable only after right-side confirmation.
3. Chronologically cluster same-side pivots with a volatility-scaled price tolerance (`max(5bp, 0.25 * source-TF median bar-range bp)` by default).  This tolerance is structural and not fitted from PnL.
4. Require at least two confirmed swing members.
5. Pool liquidity sits beyond the outer member boundary (max high for EQH, min low for EQL).
6. From pool availability through 08:30, a strict trade-through consumes the pool.  Only still-unconsumed pools become active liquidity for the trading window.
7. All active pools are retained; no nearest-only selection.
8. The same pool catalog also annotates existing premarket/15m/HTF levels when they sit inside an already-known EQH/EQL structure, allowing equal-liquidity context to be studied without filtering the original setup.

### Mechanism archetypes

R09 mechanism tags overlap and never gate entry.  Distribution landmarks use Q25/Q50/Q75 from **2023H2-2024 attempts only**, without looking at PnL; 2025 and 2026 reuse those frozen landmarks.

Tags include:

- fast rejection;
- sustained consumption;
- deep flush;
- progressive flush;
- equal-pool stop run;
- moderate MSS delivery;
- extended MSS delivery;
- clean reversal path;
- mature MSS reference.

A setup can carry several tags simultaneously.  R09 outputs both single-tag performance and observed tag combinations instead of collapsing them into one black-box score.

### Frozen execution

MSS/FVG/entry/terminal stop/opposite premarket target/cost/delay replay remain unchanged from R08.  R09 is still a mechanism-validation study, not a promoted strategy and not an exit optimization.

### Verification

- R09 synthetic end-to-end self-test: PASS.
- Explicit tests verify causal EQH/EQL pools require >=2 confirmed swing members and are available by 08:30.
- Existing attempts are preserved when equal-pool context and mechanism tags are attached.
- Mechanism distribution landmarks ignore forward/holdout values and ignore PnL.
- R02-R09 + overlap + semantic/maturity + review-pack + causal-alignment targeted suite: 32 passed.
- Repository-wide import-boundary test still fails on pre-existing legacy research->research violations; R09 adds no research->research imports and does not edit the allowlist.

## R10 — Multi-Timeframe Structural Trade Management Atlas

### Why
R07-R09 repeatedly showed that many entries travel +1R/+2R and later finish at the initial terminal-extreme stop. The entry universe also remains broad enough for practical frequency, so R10 freezes entry discovery and studies realization of edge instead of adding stricter entry filters.

### Frozen entry
- Same causal liquidity sweep -> MSS -> FVG attempts as R09.
- Same limit fill semantics and initial terminal-extreme stop.
- EQH/EQL remains an ordinary liquidity/context family; it is not required for R10.
- Management features/scenarios are post-fill only and must preserve the exact R09 filled attempt IDs.

### New causal structure
- 1m/2m/5m/15m short-term pivots are available only after right-side confirmation.
- Intermediate-term highs/lows are built from three same-side ST pivots; the center pivot becomes IT only after the right neighbouring ST pivot itself is confirmed.
- Known internal targets must already be available at the fill time and lie between entry and the original opposite-liquidity target.
- Runner trail pivots can only tighten after their causal available time.

### Pre-declared management comparisons
- Baseline full exit at original opposite liquidity / initial stop.
- First known 2m/5m/15m internal ST target: 50% partial (research anchor, not final parameter).
- Nearest causal ITH/ITL: dynamic cost-cover partial. Fraction is computed from the target R and current full-position round-trip cost so a later initial-stop outcome is approximately breakeven; this is mechanism-defined, not PnL fitted.
- Original opposite-liquidity target: realize 80% of the remaining position and leave a runner managed by 2m/5m/15m causal structure.
- Combined ITH/ITL cost-cover partial -> original target 80% -> 5m structure runner.

### What R10 must decide
Compare PF, expectancy, MDD, positive-month rate and cost stress while keeping trade count fixed. R10 is not allowed to select a final 50%/80% parameter merely because one backtest is best; it is an atlas to identify whether partial realization and multi-timeframe structural runners are robust mechanisms worth a later frozen strategy test.

## R11 — Entry Opportunity Expansion Atlas

### Why R11 exists

R10 showed that post-fill management can improve drawdown/giveback, but no management overlay can rescue a universe that still contains too many weak entries.  R11 therefore moves back to **entry opportunity coverage** without solving the problem by stacking stricter filters.

The working hypothesis is that the premarket high/low is not the day's only source of liquidity.  After one or both premarket sides are consumed, the regular session can form new, visually meaningful 15m highs/lows.  Those newly confirmed intraday swings can themselves become liquidity and later support a new Sweep -> MSS -> retracement cycle.

### New causal intraday 15m liquidity

- Every 15m swing is available only after right-side confirmation.
- The swing-quality scale uses only 15m bar ranges already known at confirmation time.
- R11 records whether the premarket high/low were both fresh, one side consumed, or both consumed when the intraday swing became available and again when it was swept.
- No swing-strength threshold gates R11 entry; obviousness is descriptive.
- Same-minute sweeps of multiple nearby intraday levels are deduped into one physical event.

### Local target models

A swept intraday 15m level searches for the latest **fresh, causally confirmed opposite 15m swing**.  Two target variants are compared independently:

1. local equilibrium / 50% midpoint of the local 15m range;
2. full opposite 15m swing.

This explicitly tests the user's idea that after the original premarket liquidity story is spent, a new intraday range may justify a closer target rather than forcing every trade to reach the old premarket opposite extreme.

### Entry execution expansion

MSS remains frozen.  The exact same signal is replayed with:

- current FVG near-edge limit;
- FVG 50% consequent-encroachment limit;
- latest opposite-close displacement candle open as a quantitative Order Block proxy;
- midpoint of that proxy candle.

The OB variants are deliberately labelled **proxy**.  They are research formulas for testing whether Order-Block-like retracements add execution edge; they are not presented as the one canonical discretionary ICT definition.

### Validation

- Default sample: 2023-07-01 -> 2026-06-30.
- Discovery: 2023H2-2024; forward: 2025; late holdout: 2026.
- Existing base MSS/FVG attempts must be preserved inside the FVG-near-edge entry variant.
- 1x/2x/3x round-trip cost and 1m/2m order-delay stress remain reported.
- R11 does not force protected-low/high exits; entry opportunity expansion is isolated first.
- Targeted R04-R11 + data-feed + causal-alignment tests: 45 passed; review-pack test: 1 passed; R11 self-test PASS with RuntimeWarning/FutureWarning promoted to errors.

## R12 — Structure Hierarchy + FVG Train Semantic Alignment

### Why R12 exists
Manual replay of 2026-08-05 showed that the remaining gap is not simply "more/less strict MSS". The previous engine still treated the latest 1/1 pivot as the structural reference too often and selected one FVG too mechanically. It also stopped too early after a micro break even though the same liquidity episode can later produce a more meaningful MSS.

### New semantic research
- Keeps 04:00-08:30 ET and 08:30-09:30 ET extremes as separate liquidity families. The late range is only frozen at 09:30; running late-session extremes are exported with causal available times.
- Keeps every causal low-timeframe pivot but adds continuous excursion/prominence/visibility features. `latest pivot` is no longer assumed to mean `important swing`.
- One sweep may emit multiple structure-break candidates; a weak micro break does not terminate the episode.
- For every break, R12 labels the latest newly broken pivot, highest-visibility newly broken pivot and outermost newly broken barrier. These are research interpretations, not frozen rules.
- FVGs are associated as a train only when the FVG middle candle is before/on the actual structure-break candle. A break candle may itself be the middle candle, with the FVG confirmed on the following bar.
- Entry comparison: uncapped FVG train, broken-swing +/-0.10 cap, break-middle-FVG +/-0.10 cap, and close-break next-open market entry.
- If the old external target is already consumed by signal time, the sweep is retained for research and a nearer active internal-structure target can still be tested.
- 2026-08-05 is exported as a golden semantic replay date by default.

### Validation
- R12 unit tests: 4 passed.
- Full SOXL ICT R02-R12 targeted suite: 40 passed.
- Causal alignment + review pack + Alpaca loader targeted tests: 8 passed.
- R12 end-to-end self-test: PASS with warnings promoted to errors.
- No new `research -> research` imports.

## R13 — Semantic Consolidation Atlas

### Frozen findings carried into R14
- Liquidity consumption cannot be represented as a binary touched/untouched flag.  R13 exports fresh, shallow/equal-like, partial and accepted/deep states while preserving continuous penetration/reclaim/acceptance features.
- Shallow/equal-like and partial-raided opposite liquidity can remain a valid draw on liquidity later in the same session; accepted/deep-consumed old external liquidity should not be mechanically reused as the original TP.
- Swing +/- $0.10 is diagnostic only.  It removes a material set of valid candidates and is not promoted to an entry gate.
- 1m is the strongest execution candidate in the current atlas; 2m has selected promising mechanisms and is retained as a secondary execution study.  5m does not have a sufficiently stable standalone entry result to justify independent R14 entries.
- 2026 has been inspected repeatedly; future work must treat R14 as robustness/candidate freeze rather than an untouched holdout claim.

## R14 — Executable Profitability Freeze

### Why R14 exists
R12/R13 were intentionally wide semantic atlases.  Their trade/variant counts are not executable strategy counts because one physical sweep can generate many MSS/FVG/target interpretations.  R14 is the first study that forces the system to answer the live question: *what single setup would the account actually take now?*

### Frozen executable rules
- One physical liquidity sweep can create at most one R14 setup per policy.  Later MSS/re-entry from the same sweep is suppressed even if the earlier pending order never fills.
- Only causal visible/strong structure tiers (P50-P80 / >=P80 visibility) can independently open a trade.  Micro pivots remain research context.
- Cross-timeframe arbitration is causal: the earliest eligible signal wins.  A later 1m setup is never chosen retroactively over an earlier 2m setup.
- 1m is primary and 2m is secondary; 5m cannot independently enter in R14.
- No fixed-dollar Swing +/- $0.10 entry cap.
- Entry routing is predeclared before R14 PnL as separate executable policies rather than a future-aware preference chain: the main combined policy uses 1m break-middle near for shallow/equal-like targets and 2m break-middle CE for partial-consumed targets; a second combined policy changes only the 1m leg to first-train near.
- Target does **not** have to be equal-like.  The first executable profit core deliberately combines two already-positive mechanisms:
  - 1m shallow/equal-like external target + break-middle FVG near-edge;
  - 2m partial-consumed external target + break-middle FVG CE.
  A second combined policy swaps the 1m leg to first-train near-edge.  Fresh/deep external targets are not declared invalid liquidity; they are deferred from this narrow freeze because their current full-target management is not yet stable.
- Source-liquidity consumption state does not gate entry in R14.
- After per-sweep setup selection, only one pending/position lifecycle can exist in the account at a time.

### Robustness protocol
- Baseline round-trip cost 0.11%.
- 1.5x and 2x cost stress.
- 1m and 2m order-delay stress.
- Discovery 2023H2-2024 / forward 2025 / late 2026 reporting, but late 2026 is not called untouched OOS.
- Top-5/top-10 winner removal.
- Monthly/yearly/account curve, actual filled trades per session, active-day rate, longest no-trade session gap, 1m/2m and target-state contribution.

### Performance architecture
R14 can reuse R13's causal intermediate reports (`01/04/07/08`) with `--r13-cache-dir`.  This skips the expensive semantic rebuild but never reads R13 performance/PnL ranking tables.  Without a cache, R14 rebuilds its own causal intermediates through `src.research_common` only; there is no research->research import.

### R14 pre-replay opportunity audit on frozen R13 causal intermediates
Before account replay, the R14 selector was run only on R13 causal semantic intermediates (no R13 PnL/ranking tables) to verify the one-sweep-one-setup frequency and prevent variant-count inflation:
- `core_break_middle`: 222 selected setups / 783 valid sessions = 0.2835 setup/session; 164 active setup days. Composition: 79 x 1m shallow/equal-like + 143 x 2m partial-consumed.
- `core_first_train_1m`: 244 / 783 = 0.3116 setup/session; 176 active setup days. Composition: 105 x 1m shallow/equal-like + 139 x 2m partial-consumed.
- `shallow_1m_break_middle`: 80 / 783 = 0.1022 setup/session.
- `partial_2m_break_middle_ce`: 145 / 783 = 0.1852 setup/session.
These are selected setups before limit-fill probability and before one-account overlap suppression, so final filled trades/session must be lower. `0.2835/session` is about 1.42 selected setups per five-session week, not a claim of one filled trade every day.

## R15 — Daily Liquidity Traversal Path Atlas

### Why R15 replaces the immediate R14->execution-router plan
R14 proved that the selected 1m profit kernel can have attractive PF, but its one-sweep/one-setup executable frequency is far below discretionary observation.  The selection funnel is the warning: 8,354 physical sweep events were reduced to 222 core setups before fill, only 2.7% of the broad physical sweep universe.  The combination of target-state routing, visibility tiers and fixed FVG models therefore risks answering the wrong question: it may be measuring a narrow survivor subset instead of the actual daily ICT opportunity process.

R15 intentionally backs up one level.  It does not begin with profitable R13 buckets.  It begins with a range that was actually visible before/around the cash open and labels what every trading day subsequently did.

### Daily range models
Four models are compared without PnL-based selection:
1. early 04:00-08:30 ET absolute high/low, frozen at 08:30;
2. full 04:00-09:30 ET absolute high/low, frozen at 09:30;
3. most prominent causally-confirmed 15m high/low pair known by 08:30;
4. most prominent causally-confirmed 15m high/low pair known by 09:30.

The 15m pair uses only pivots whose right-side confirmation is already available at the anchor time.  Prominence is geometric/two-sided excursion relative to completed 15m range, not a forward label and not a PnL threshold.

### Path labels
After the range is frozen, R15 studies every first boundary raid and records:
- first side/time;
- shallow/deep penetration as continuous range-normalised values;
- reclaim time and outside closes;
- repeated raids of the same side;
- 25/50/75/100% progress toward the opposite boundary;
- full first-side -> opposite-side traversal;
- partial reversal and same-side acceptance/continuation archetypes.

Future information is allowed only in these outcome labels.  Causal candidate generation does not see `traversal_complete` or later milestone times.

### Entry-path discovery
Each first raid becomes a single path event.  R15 then runs the causal 1m/2m swing hierarchy / MSS / displacement / FVG machinery against that event and keeps micro, visible and strong structural tiers for study rather than filtering on R14's profitability states.  No EQL target is required.  No partial-consumed target is required.  No Swing +/- $0.10 cap is used.

Execution variants include first-train near/CE, last pre/on-break near, break-middle near/CE, closest-to-broken-swing near, and close-break next-open market.  Replay uses the opposite frozen range boundary as the path-study target and the causal terminal sweep extreme as stop.  The objective is to measure traversal capture rate and determine where the manual ~daily opportunities are being lost.

### Required decision from R15
Before any R16 executable strategy, answer:
- how many valid sessions have a boundary raid;
- how many first raids later traverse to the opposite boundary;
- how many have a causal 1m/2m MSS/FVG opportunity before that traversal;
- which range definition best represents the recurring daily process;
- which entry style captures the most traversals without destroying PF/expectancy;
- what separates successful traversals from failures using only features available at signal time.

### Validation
- R15 synthetic end-to-end path self-test: PASS.
- R15 unit tests: 3 passed.
- Full SOXL ICT targeted test glob after R15: 60 passed.
- R15 adds no research->research import.

## R16 — Entry Archetype Survival Atlas

### Why R16 exists
R15 rejected the opportunity-scarcity hypothesis: almost every valid session has a first liquidity raid, and a large fraction later traverses 50/75/100% of the frozen dealing range.  The broad `sweep -> any MSS/FVG -> opposite boundary` replay still had little edge, so the next bottleneck is entry timing rather than path availability.

R16 therefore asks two concrete questions before any new executable freeze:
1. Which causal entry archetype minimizes entries that are stopped almost immediately?
2. Once an order is actually filled, how often does price reach 25/50/75/100% of the dealing range before the terminal-extreme stop?

### Entry archetypes
R16 compares parallel counterfactual entry styles on the same R15 path universe:
- liquidity raid -> confirmed reclaim -> next 1m open market;
- liquidity raid -> confirmed reclaim -> retest of the swept level by limit order;
- first causal MSS -> close-break next-open market;
- first causal MSS -> break-middle / closest-to-swing FVG near-edge and CE limit entries;
- first *visible* MSS (causal visibility >= P50) versions of the same MSS/FVG routes;
- quantitative Order Block proxy: last opposing closed candle in the selected displacement leg, tested at candle open and midpoint;
- Order-Block-proxy x displacement-FVG overlap midpoint;
- visible 2m structure confirmation -> causally-known 1m FVG near-edge / CE execution.

The Order Block definition is explicitly a quantitative proxy, not a claim that one mechanical candle rule is the canonical discretionary ICT definition.

### Stop-survival research
- Static initial stop remains the terminal sweep extreme known by the entry signal.
- R16 labels stop within 1/3/5/10 minutes after fill.
- It separately measures 25/50/75/100% dealing-range milestones *before stop*.
- If a milestone is already behind the actual fill price it is marked `already_passed_at_fill` rather than counted as a successful capture.
- Limit orders are cancelled if the terminal stop is invalidated or the opposite boundary is reached before fill.
- Same-minute stop + target ambiguity is resolved conservatively to stop.

### Causal pre-entry features
To identify obvious bad entries without future leakage, R16 records only signal-time state:
- raid count so far;
- penetration so far / dealing-range width;
- reclaim state;
- signal delay from raid;
- entry location and initial risk / dealing-range width;
- causal swing visibility and MSS displacement features;
- pre-raid approach efficiency / volatility contraction;
- monotonic micro-swing count and a non-gating three-swing contraction flag.

`three waves` is never a hard requirement.  It is an observable path descriptor for later validation.

### Anti-overfit rules
- No Swing +/- $0.10 gate; the fixed-dollar chase cap is retired.
- Same physical path contributes at most one candidate per entry archetype, so repeated MSS attempts cannot inflate one archetype's frequency.
- R16 does not select only R13/R14 historically profitable target states.
- Fixed semantic feature bins are used for immediate-stop diagnostics; no PnL-optimized quantile threshold is promoted in R16.
- Any exclusion rule discovered from R16 must be frozen and checked unchanged across discovery / 2025 / 2026 before executable use.

### Performance architecture
R16 can reuse R15's `03_daily_path_outcomes.csv`, `06_causal_mss_narratives.csv`, and a selective column load from `07_entry_candidate_atlas.csv`.  This avoids rerunning R15's ~3 minute daily MSS/FVG scan.  The R15 performance summaries are not used to choose entry archetypes.

### Validation
- R16 warnings-as-errors self-test: PASS.
- R16 unit tests: 4 passed.
- Full SOXL ICT targeted test glob including R16: 64 passed.
- Repository-wide import-boundary test still fails on pre-existing `research/ict/mss/* -> research.ict.mss.common.*` legacy violations; R16 adds no research->research import.

### R16 hotfix — heterogeneous path-metadata preservation

The first full R16 report exposed an engineering bug in ordinary MSS/FVG archetypes.  R16 concatenates heterogeneous entry-family DataFrames before attaching canonical R15 path metadata.  Because reclaim/market/OB families already carried columns such as `range_model`, Pandas created those columns across the union schema and left MSS/FVG rows as NaN.  The old metadata helper only attached columns that were globally absent, so FVG rows never recovered their `range_model`, dealing-range boundaries, width or target metadata.  Their fill/immediate-stop statistics remained usable, but milestone 25/50/75/100 and milestone-exit PF summaries were invalid/missing.

Hotfix:
- R15 cache loading now preserves `range_model`, `path_event_id` and core path/range fields directly in `07_entry_candidate_atlas.csv` selective loads.
- `attach_path_metadata()` now fills row-level missing values after heterogeneous concatenation rather than checking only whether a column exists globally.
- Existing non-null entry metadata is preserved and checked against canonical path metadata; conflicting values fail loudly rather than silently drifting.
- Added regression tests reproducing the exact heterogeneous-concat NaN failure and a conflict-detection test.

The hotfix changes report metadata propagation only.  It does not change liquidity paths, swing/MSS selection, displacement/FVG choice, entry price, stop, fill logic, future-path labels, fee assumptions or causal timing.

Validation:
- R16 targeted tests after hotfix: 6 passed.
- Real R15 cache audit: 191,140 FVG entry rows loaded with 0 missing `range_model`; `path_event_id == event_id` for all rows.

## R17 — No-Stop Opposite-Liquidity Diagnostic

### Why this diagnostic exists
R16 showed a large gap between the probability that a sweep/MSS path eventually reaches the opposite external liquidity and the probability that the first 1m break-FVG entry reaches that target before the terminal-extreme stop.  A material subset of stopped trades later reaches the original opposite target.  R17 isolates whether the initial stop is therefore washing out a real liquidity-delivery edge or merely suppressing unacceptable adverse excursions.

### Frozen experiment
R17 intentionally does **not** add a new entry filter or target:
- range = `prominent_15m_pair_0830`;
- entry = `mss_first_visible_break_fvg_near`, 1m;
- R16 lifecycle rows and actual fill times are reused exactly;
- post-fill terminal stop is removed;
- opposite external liquidity remains the sole TP;
- no TP by session end -> final 1m close before 16:30 ET.

No 25/50/75 milestone enters the strategy logic.  Old stop time is carried only as a counterfactual label (`rescued_after_old_terminal_stop`).

### Required decision
The diagnostic must answer both sides of the tradeoff:
1. How much does opposite-liquidity TP rate / PF / expectancy improve without the terminal stop?
2. What happens to MAE, EOD loss tails, and period stability when wrong reversals are allowed to run until the close?

Only if the first effect materially dominates the second should later research consider replacing the initial terminal stop with a more structure-aware invalidation rule.  R17 itself does not promote no-stop execution to live trading.

### Validation
- Exact R16 cache selection audit: 347 candidates / 305 frozen fills for the default range/archetype/TF.
- R17 unit tests: 4 passed.
- R17 warnings-as-errors self-test: PASS.
- Existing SOXL ICT R01-R13 targeted tests plus R17 in the packaged baseline: 14 passed.
- R17 adds no research->research import.

## R18 — Opposite-Liquidity Probability Hypotheses (2026-08-17)

### Why R18 exists
R15/R16/R17 established that the `prominent_15m_pair_0830` liquidity universe offers roughly one sweep/MSS opportunity per session, but the mechanical first-MSS/FVG entry reaches the opposite external liquidity before the terminal stop too infrequently.  R18 stops expanding generic Price Action targets and reframes the discretionary ICT question as a causal probability problem:

`P(opposite external liquidity by session end | causal sweep/MSS state)`

and, after a real fill:

`P(opposite TP before terminal-extreme SL | causal entry state)`.

No 25/50/75 dealing-range milestone is used as a model target, predictor or trading filter.

### Frozen hypotheses
- H1 liquidity context: source/target prominence and age plus pre-raid approach geometry;
- H2 terminal maturity: initial/terminal penetration, terminal-version evolution, reclaim and timing;
- H3 meaningful MSS: causal swing visibility/prominence and which barrier was broken;
- H4 displacement: directional dominance, efficiency and normalized overshoot;
- H5 mitigation entry: next-open market, break-FVG near/CE, OB, OBxFVG and 2m-structure->1m-FVG geometry;
- H6 cross-timeframe confirmation: only confirmation already available by the snapshot is allowed.

### Modeling discipline
- 2023H2-2024 is the only fit period.
- The fitted Logistic Regression is frozen and evaluated unchanged on 2025 and 2026.
- Primary diagnostics are Brier score, log loss, AUC and probability calibration, not an optimized PnL threshold.
- Predictor columns are an explicit whitelist; full-day path outcome, MFE/MAE, stop result and 25/50/75 milestones are forbidden.
- 1m snapshots cannot see a later 2m confirmation; cross-TF features are strict as-of state.
- Entry methods are also compared pairwise on the same physical sweeps to avoid comparing unrelated samples.

### Author-side cache trial on current R15/R16 reports
The packaged implementation was run against the current 2023-07-01 -> 2026-08-14 R15/R16 reports before delivery:
- `prominent_15m_pair_0830`: 775 path rows, 11,374 causal MSS narratives;
- 758 first-visible 1m MSS snapshots and 758 first-visible 2m MSS snapshots;
- opposite-liquidity base rate stayed stable: 46.99% discovery, 45.68% in 2025, 44.30% in 2026;
- 1m H1-H4 probability ranking reached AUC ~0.673 in 2025 and ~0.700 in 2026;
- 2m H1-H4 was materially stronger: AUC ~0.718 in 2025 and ~0.796 in 2026;
- H6 cross-TF, as currently encoded, did not improve the frozen model and slightly degraded later-period metrics;
- the filled-entry model showed ranking information (AUC ~0.644 in 2025 / ~0.719 in 2026) but calibration/Brier did not consistently beat the constant baseline, so it is **not** ready to route live entries.

Interpretation: there is evidence that causal ICT state can rank which sweep/MSS events are more likely to deliver to the opposite liquidity, especially once 2m structure is known.  There is not yet enough evidence that the current entry-state probability is well calibrated or economically tradeable.  R18 is a probability-discovery stage, not a strategy freeze.

### Validation
- R18 warnings-as-errors self-test: PASS.
- R18 unit tests: 4 passed.
- Full `tests/research/ict/test_soxl_ict_*.py`: 58 passed.
- Full R18 run on the uploaded R15/R16 caches completed in seconds with bounded memory and no model-time future feature.

## R19 — Event-Conditioned Entry Study
- R18 result carried forward: 2m liquidity/terminal/MSS/displacement state can rank opposite-liquidity delivery better than the unstable direct entry probability model.
- New causal gate: an event probability may condition an entry only when its 2m snapshot was available no later than the entry order time.
- Research goal: measure TP-before-terminal-SL, RR and net expectancy of market/FVG/OB-hybrid entries inside frozen event-probability bands.
- Fixed probability thresholds are diagnostic only; no threshold is frozen as a strategy rule in R19.
- 25%/50%/75% dealing-range targets remain outside the research question.

## R20 — Broad Position-Management Backtest (2026-08-17)

### Strategic change
R20 is not permitted to improve results by shrinking the setup universe. The broad `prominent_15m_pair_0830 -> first-visible 1m MSS -> next-open market` stream is frozen first. If its filled frequency is below 0.5 trades per valid session, the run fails immediately. Event probability and other causal features may be used in later sizing/management work, but not as an R20 hard entry gate.

### Trading question
Given the broad roughly-daily MSS opportunity stream, can causal position management make the account profitable and stable after 0.11% round-trip cost?

### Frozen lifecycle policies
1. full position to opposite external liquidity / initial terminal stop / session close;
2. full position, move stop to entry after +1R (active next bar);
3. take 25% at +1R, move remainder stop to entry next bar;
4. same, then after +2R lock +0.5R on the remainder from the next bar;
5. same +1R partial/protection, then trail the remainder with causally-confirmed 2m ST structure.

All five policies preserve the same base trade universe. Same-minute stop versus partial/main target is resolved to stop. Structural trail can only use pivots whose confirmation `available_time` is already known at the current bar start. No 25/50/75 dealing-range target is used.

### Selection / validation
- Management policy selection: Discovery 2023H2-2024 only.
- 2025 and 2026: evaluation only, never used to select a policy.
- Stress: cost 1x/1.5x/2x and market-entry delay 0/1/2 minutes.
- Account sizing: 1% risk/trade, default max notional 1.0x, configurable.
- Priority inside Discovery: no-trade gap -> consecutive losses -> MDD -> CAGR -> total return.

### Engineering validation before delivery
- R20 warnings-as-errors self-test: PASS.
- R20 targeted tests: 5 passed.
- Full SOXL ICT targeted suite including R20: 82 passed.
- Repository-wide pytest is not green in the supplied baseline because five pre-existing liquidity/panic tests import modules/files absent from the archive; R20 does not touch those paths.


## R20 v2 broad-universe correction
- Corrected R20 from 1m close-confirmed-only entries to the earliest causal first-visible 1m/2m structure break per physical liquidity path.
- Wick-only vs close-confirmed is diagnostic/feature information, not a profitability filter.
- The >=0.5 trades/session target remains hard; R20 no longer aborts before management just because a mistakenly narrowed input subset fails it.
