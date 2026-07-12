# ETH Directional Impulse Continuation — Research Log

- Portfolio plan: `ETH_NOVA_PORTFOLIO`
- Edge ID: `ETH_MOM_DIRECTIONAL_IMPULSE_CONTINUATION`
- Research directory: `research/momentum/directional_impulse_continuation/`
- Status: `research`
- Current round: `03_impulse_notional_expansion_event_study.py`

## Scope boundary

This directory studies only continuation after a short-window directional ETH price impulse.

It does not use or inherit any existing Portfolio composition, weights, frequency targets, risk allocation, return sources, or prior Portfolio conclusions. If evidence points to reversal or pullback-only behavior, that mechanism must be moved to a separate Edge rather than silently changing this Edge.

## Round 01 — Basic impulse event study

### Research question

After ETH produces an unusually strong directional move over 1, 3, 5, 10, or 15 closed one-minute bars, does price subsequently continue, reverse, or show no stable directional displacement over the next 1–240 minutes?

### Research hypothesis

An abnormal directional impulse may reflect active capital, information arrival, stop cascades, or a volatility-regime transition. If the mechanism persists after the signal bar closes, direction-adjusted forward returns from the next bar open should remain positive after realistic round-trip costs across a stable neighborhood of impulse windows, thresholds, horizons, and years.

### Changed from previous round

This is the first round. It deliberately starts from a high-recall price-only event and adds no trend, volume, session, order-flow, footprint, funding, OI, liquidation, or regime filters.

Data-source correction during Round 01: the initial draft used `OKXDataLoader`. Before accepting the production result, it was corrected in place to use the already-built local `OKXTradeBarLoader` 1m cache. This is an infrastructure correction only; event logic, thresholds, timing, costs, and statistics are unchanged.

Trade-bar gap correction during Round 01: `OKXTradeBarLoader` intentionally omits empty/no-trade resample buckets, so the local 1m frame is not a perfectly continuous calendar-minute index. The first trade-bar draft incorrectly rejected all such gaps. The corrected script now follows the project's shared trade-aggregation convention by regularizing missing minutes as flat previous-close, zero-volume rows, while retaining `source_bar_observed_flag`. Synthetic rows are excluded from impulse windows, signal bars, next-minute entries, and the complete 240-minute forward path. This is a data-axis correctness fix, not a strategy/filter change.

### Data

- Symbol: `ETH-USDT-SWAP`
- Timeframe: `1m`
- Data source: local `OKXTradeBarLoader` cache (`data/okx_trade_bars.db`)
- Cache policy: `build_missing=False`; the research script will not download ordinary K-lines or missing trade files
- Missing-minute policy: regularize to a calendar 1m axis using flat previous-close/zero-volume rows, but exclude every event whose impulse, signal, entry, or full 240m forward path touches a synthetic row
- Warmup start: `2022-01-01`
- Research start: `2023-01-01`
- Research end: `2026-06-30` inclusive
- Time convention: project-native timezone-naive UTC+8
- Signal timing: fully closed bar `t`
- Simulated entry: bar `t+1` open

### Core definition

- Impulse windows: `1m, 3m, 5m, 10m, 15m`
- Normalization: window return divided by a causal historical rolling standard deviation of one-minute log returns, scaled by `sqrt(window)`
- Historical volatility lookback: `1440` closed one-minute observations, minimum `720`
- Leakage protection: volatility baseline is shifted by the complete impulse window, so it excludes every bar used in the current impulse
- Thresholds: `1.0, 1.5, 2.0, 2.5`
- Deduplication: within each direction/window/threshold stream, retain the first event and suppress later events occurring fewer than `impulse_window` bars after the last retained event
- Horizons: `1m, 3m, 5m, 10m, 15m, 30m, 60m, 120m, 240m`
- Fee-only cost: `0.11%` round trip
- Normal cost: `0.15%` round trip

### Required metrics

The production run completed on local 1m trade bars covering `2022-01-01` through `2026-06-30`. The full report is under `01_basic_impulse_event_study/`.

Representative deduplicated results:

| Direction | Window | Threshold | Horizon | Events | Events/month | Mean gross | Median gross | Mean net | Median net | Win rate | PF | Positive years |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LONG | 15m | 2.0 | 240m | 7,692 | 183.14 | +0.0712% | -0.0196% | -0.0788% | -0.1696% | 43.02% | 0.864 | 0/4 |
| LONG | 15m | 2.5 | 240m | 4,365 | 103.93 | +0.0710% | -0.0243% | -0.0790% | -0.1743% | 43.16% | 0.870 | 0/4 |
| LONG | 10m | 2.5 | 240m | 5,985 | 142.50 | +0.0700% | -0.0233% | -0.0800% | -0.1733% | 43.21% | 0.868 | 0/4 |
| SHORT | 15m | 1.0 | 240m | 28,768 | 684.95 | -0.0197% | -0.0578% | -0.1697% | -0.2078% | 39.88% | 0.708 | 0/4 |

### Result explanation

1. The unfiltered event is not a tradable continuation edge. Every normal-cost combination is negative and every combination has zero positive net years out of four.
2. LONG impulses show a smooth but weak gross continuation gradient at the 240m horizon as impulse strength increases. The best broad result is only about +0.07% gross, below the 0.15% normal round-trip cost. Median gross is negative for the strongest 10m/15m candidates, so most events do not continue even before costs.
3. SHORT impulses do not show continuation. Direction-adjusted gross returns are generally negative and worsen as threshold increases, which leans toward reversal/no-continuation. Any reversal hypothesis belongs to a separate Edge.
4. The result is not driven by five extreme winners: top-five positive-return contribution is generally low for the broad candidates. The main failure is insufficient average displacement, not one-event dependency.
5. Raw events are heavily duplicated for longer windows: overlap reaches about 84% for 15m. Deduplicated results are therefore the primary evidence.
6. Descriptive feature buckets reveal one mechanism worth a dedicated causal pass: LONG impulses following an already same-direction preceding move can survive normal cost in the 240m horizon. The strongest descriptive bucket was 1m LONG with prior 1m return above +0.50%: 647 events, mean net +0.5378%, median net +0.2661%, PF 1.710. This is not accepted evidence because it came from a many-feature exploratory table and lacks yearly/threshold/platform validation.

### Failed branches

- Immediate unfiltered continuation over 1m-60m: failed; gross displacement is near zero or negative.
- Unfiltered 120m-240m continuation: weak gross signal only; normal costs destroy it.
- Symmetric SHORT continuation: failed in the base event and may instead contain a separate reversal mechanism.

### Next-round reason

Round 02 tests one mechanism only: whether continuation requires persistence across two adjacent equal-length impulse windows. The prior-window move is normalized causally with a volatility baseline ending before both windows. Fixed persistence bands are compared for LONG and SHORT separately, while retaining all current-impulse thresholds and horizons. No volume, trend, regime, order-flow, session, or exit logic is added.

<!-- AUTO_RESULT_START -->
## Round 01 generated result

Primary decision: `research_continue`. The base event is not tradable, but the adjacent-window persistence mechanism has enough descriptive evidence and frequency to justify one dedicated round.
<!-- AUTO_RESULT_END -->

## Round 02 — Adjacent-window impulse persistence

### Research question

Does a current abnormal directional impulse continue only when the immediately preceding equal-length window already moved in the same direction?

### Research hypothesis

A one-off impulse may be noise, liquidation impact, or a temporary price jump. A shock that persists across two adjacent windows is more likely to represent continuing active capital or a durable information/volatility transition. If true, normal-cost forward returns should improve monotonically from opposite/flat prior movement to strong same-direction prior movement.

### Changed from Round 01

Only one new continuous mechanism is added: the direction-adjusted normalized return of the immediately preceding equal-length window. The base impulse definition, thresholds, closed-bar timing, next-open entry, costs, horizons, and trade-bar source remain unchanged.

### Fixed persistence bands

```text
opposite_or_flat          <= 0.0
weak_same_0_0.5           > 0.0 and < 0.5
moderate_same_0.5_1.0     >= 0.5 and < 1.0
strong_same_ge_1.0        >= 1.0
```

### Data and performance design

- Same local `OKXTradeBarLoader` 1m cache.
- Prior normalization denominator is shifted by `2 * impulse_window`, excluding both adjacent windows.
- One minimum-threshold event frame is built per direction/window. Higher threshold memberships and threshold-specific dedup flags are stored as columns, avoiding duplicate event rows for nested thresholds.
- Every persistence band, threshold, window, horizon, direction, year, and month is retained.

### Production result

Representative deduplicated results:

| Direction | Window | Threshold | Persistence | Horizon | Events | Events/month | Mean gross | Mean net | Median net | Win rate | PF | Positive years |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LONG | 15m | 2.5 | strong same >= 1.0 | 240m | 901 | 21.45 | +0.2070% | +0.0570% | -0.1401% | 44.28% | 1.101 | 2/4 |
| LONG | 15m | 2.0 | moderate same 0.5-1.0 | 240m | 725 | 17.26 | +0.1568% | +0.0068% | -0.1486% | 43.59% | 1.012 | 3/4 |
| SHORT | 15m | 2.0 | strong same >= 1.0 | 240m | 1,627 | 38.74 | +0.0655% | -0.0845% | -0.2615% | 41.36% | 0.879 | 1/4 |

For the best LONG 15m / threshold 2.5 / strong-persistence row, yearly mean net was approximately -0.0001% in 2023, +0.0784% in 2024, +0.1636% in 2025, and -0.0869% in 2026 H1. The top-five positive-return contribution was 8.84% overall, while within individual years it was about 20%-27% because each yearly sample contained only 122-265 events.

### Result explanation

1. The broad adjacent-window persistence mechanism did not validate. Among 180 deduplicated LONG window/threshold/horizon combinations, only 11 showed monotonically increasing mean net across all four persistence bands, and none showed monotonic median net. SHORT had only one monotonic-mean combination and no positive normal-cost row.
2. Only two of 900 deduplicated LONG persistence rows had positive mean net, both at the 240m horizon. No row had positive median net. The best result therefore reflects a right-skewed distribution rather than continuation experienced by the typical event.
3. The strongest LONG row is not a threshold plateau. At 15m / strong persistence / 240m, mean net progressed from -0.0637% at threshold 1.0, to -0.0475%, -0.0223%, and only then +0.0570% at threshold 2.5. The effect exists only at the most extreme tested current impulse.
4. The best row is not year-stable: only 2024 and 2025 were clearly positive, 2023 was flat, and 2026 H1 was negative. PF 1.101 and mean net +0.0570% are too marginal to justify backtest promotion.
5. The first-round exploratory 1m absolute-prior-return bucket did not generalize into a broad normalized persistence gradient. For 1m LONG / strong normalized persistence / 240m, all current-impulse thresholds remained negative after normal costs.
6. SHORT continuation remains failed. Persistence improves some gross outcomes at long horizons but never covers normal execution costs.

### Failed branches

- Broad adjacent-window persistence monotonicity: failed.
- Typical-event persistence effect measured by median net: failed in every combination.
- Stable threshold neighborhood: failed; only the most extreme LONG 15m point became marginally positive.
- Symmetric SHORT persistence continuation: failed.

### Decision

`research_continue` for the overall Edge, but the adjacent-window persistence branch is rejected and must not be stacked into the next round. The evidence is too weak and isolated for a strategy backtest.

### Next-round reason

Round 03 tests one new basic mechanism only: whether an impulse accompanied by unusually high quote notional has stronger continuation than an equally large price impulse on normal or low activity. It branches from the Round-01 base event rather than stacking Round-02 persistence. This is a core mechanism test for active-capital participation and uses only existing local trade-bar `notional` and `trades_count` fields.


## Round 03 — Impulse-window quote-notional expansion

### Research question

Does continuation improve when the current directional impulse window carries unusually high traded quote notional relative to a causal historical baseline?

### Research hypothesis

A large price move occurring on ordinary activity may be a thin-liquidity jump or temporary repricing. A similarly large move accompanied by broad quote-notional expansion is more likely to represent active capital and may persist longer. If this mechanism is real, normal-cost continuation should improve across adjacent notional-expansion bands without requiring the Round-02 persistence condition.

### Changed from Round 02

The persistence condition is removed. Only current impulse-window quote-notional expansion is added to the original Round-01 event. Current price impulse windows, thresholds, horizons, costs, closed-bar timing, next-open execution, event deduplication, and local trade-bar source remain unchanged.

### Causal activity definition

```text
current_window_notional
/
rolling median of equal-length window notionals shifted by impulse_window
```

The historical baseline uses 1,440 observations with a minimum of 720 and excludes every bar in the current impulse window.

### Fixed notional-expansion bands

```text
low_lt_0.75          < 0.75x
normal_0.75_1.25     0.75x to < 1.25x
elevated_1.25_2.0    1.25x to < 2.0x
extreme_ge_2.0       >= 2.0x
```

### Data and performance design

- Same local `OKXTradeBarLoader` 1m cache with `build_missing=False`.
- Primary activity measure is trade-bar quote `notional`, not ordinary-K-line volume.
- `trades_count` is retained as a continuous audit feature but is not a second filter.
- Synthetic gap rows are excluded from the activity window, signal, entry, and full forward path.
- Every direction, impulse window, threshold, horizon, raw/deduplicated set, activity band, year, and month is retained.

### Production result

Representative deduplicated results:

| Direction | Window | Threshold | Notional band | Horizon | Events | Events/month | Mean gross | Mean net | Median net | PF | Positive years |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LONG | 10m | 2.0 | normal 0.75-1.25x | 240m | 492 | 11.71 | +0.1169% | -0.0331% | -0.0712% | 0.934 | 1/4 |
| SHORT | 5m | 2.0 | low <0.75x | 60m | 121 | 2.88 | +0.1682% | +0.0182% | -0.1288% | 1.064 | 2/4 |
| SHORT | 1m | 2.5 | normal 0.75-1.25x | 240m | 315 | 7.50 | +0.1597% | +0.0097% | -0.1468% | 1.018 | 2/4 |

### Result explanation

1. Quote-notional expansion did not validate as a continuation mechanism. Across the 180 deduplicated LONG window/threshold/horizon groups, only 11 showed monotonically improving mean net from low to extreme activity and none showed monotonic median net. SHORT had only two monotonic-mean groups and no monotonic-median group.
2. No credible LONG activity bucket survived normal cost. The best eligible LONG row remained -0.0331% mean net with negative median and only one positive year.
3. The two positive SHORT descriptive rows were not evidence for the stated hypothesis. The best used the low-notional bucket, had only 121 events, negative median net, top-five contribution around 45%, and was mostly driven by 2025. The other used normal rather than expanded activity and had inconsistent years.
4. Extreme quote notional was not consistently better than normal activity. Median performance across parameter combinations was effectively unchanged for LONG and slightly worse for SHORT.
5. The broad event still shows large MFE and MAE relative to terminal return. For example, stronger 10m-15m impulses often travel roughly 0.5%-1.4% in both directions over 30-240m while fixed-time gross displacement stays close to zero. This suggests path order may matter even though terminal continuation fails.

### Failed branches

- Monotonic improvement from low to extreme quote notional: failed.
- Cost-after LONG continuation conditioned on activity expansion: failed.
- Cost-after SHORT continuation conditioned on activity expansion: failed.
- Stable year/threshold neighborhood: failed.

### Decision

`research_continue` for the overall Edge, but the quote-notional expansion branch is rejected and must not be stacked into the next round.

### Next-round reason

Round 04 returns to the Round-01 base event and tests one path mechanism only: whether a fixed favorable excursion is reached before an equally distant adverse excursion. This directly addresses the possibility that continuation exists briefly and is later surrendered before a fixed-time close. It uses conservative same-bar ambiguity handling and does not stack persistence or quote-notional filters.

## Round 04 — First-passage path and exit-opportunity study

### Research question

After a basic directional impulse, does price hit a fixed favorable barrier before an equally distant adverse barrier often enough to support a causal exit rule?

### Research hypothesis

Fixed-time exits can miss a short-lived continuation if price first extends in the impulse direction and later mean-reverts. If such a path exists, favorable-first probabilities and conservative barrier-plus-time-stop returns should improve across nearby impulse windows and thresholds.

### Changed from Round 03

The quote-notional condition is removed. Round 04 uses only the original price impulse and changes the outcome measurement from terminal close-only returns to first-passage path order.

### Fixed path design

```text
symmetric barriers = 25, 50, 75, 100 bps
time limits        = 5, 15, 30, 60, 120, 240 minutes
```

If neither barrier is touched, the event exits at the time-limit close.

If target and stop are both first touched in the same 1m bar:

```text
primary result = conservative stop-first
upper bound    = optimistic target-first
third view     = ambiguity-excluded
```

### Performance design

- First-passage paths are processed in bounded chunks; no full multi-year event-by-240 matrix is held in memory.
- Each minimum-threshold event pool is built once per direction/window.
- First-touch times are computed once and reused across all thresholds, time limits, raw/deduplicated sets, years, and months.
- MFE/MAE and fixed-time returns are precomputed once per time limit.
- No per-event Python path loop and no repeated full-data scan per variant.
- `05_events.csv` stores only minimum-threshold deduplicated events with higher-threshold membership flags, reducing output size without changing summaries.

### Status

Pending production run.

### Round 04 production result

The symmetric first-passage study completed on local 1m trade bars through `2026-06-30`.

Representative deduplicated results:

| Direction | Window | Threshold | Barrier | Limit | Events | Events/month | Target first | Stop first | Mean gross | Mean net | Median net | PF | Positive years |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LONG | 10m | 2.5 | 50bps | 5m | 5,985 | 142.50 | 15.44% | 8.30% | +0.0028% | -0.1472% | -0.1770% | 0.303 | 0/4 |
| LONG | 15m | 2.0 | 100bps | 240m | 7,692 | 183.14 | 36.88% | 34.54% | +0.0093% | -0.1407% | -0.1819% | 0.704 | 0/4 |
| SHORT | 10m | 2.5 | 75bps | 30m | 6,065 | 144.40 | 26.53% | 18.17% | +0.0072% | -0.1428% | -0.2044% | 0.551 | 0/4 |
| SHORT | 10m | 2.5 | 100bps | 60m | 6,065 | 144.40 | 26.07% | 17.92% | +0.0168% | -0.1332% | -0.2091% | 0.651 | 0/4 |

### Result explanation

1. Round 04 does not establish a tradable strategy. All 960 deduplicated path variants are negative after the normal 0.15% round-trip cost; no variant has PF above 1 or a positive year count.
2. There is a real but small path-order asymmetry. Stronger 10m-15m impulses, especially SHORT, reach favorable barriers before equal adverse barriers more often. The best conservative gross expectancy is only +0.0283% and remains far below execution cost.
3. MFE is materially larger than terminal return, but MFE is an ex-post upper bound. Symmetric fixed TP/SL captures only a few basis points of gross expectancy, proving that the observed excursion cannot be treated as realizable profit without a causal state-selection and exit mechanism.
4. The useful part of the signal is early. For LONG 10m / threshold 2.5 / 50bps / 5m, target-first exceeds stop-first by about 7.13 percentage points and median target touch is about 2 minutes. For SHORT 10m / threshold 2.5 / 100bps / 60m, target-first exceeds stop-first by about 8.15 percentage points and favorable touches arrive earlier than adverse touches. This supports studying whether the impulse remains active immediately after the signal rather than relying on a 240m hold.
5. Same-bar ambiguity is extremely small in the leading 75-100bps rows, so the failure is not caused by conservative ambiguity handling.
6. The overall Edge remains research-only. Evidence supports a directional path tendency, not yet cost-after PnL.

### Failed branch

- Symmetric fixed target/stop plus time exit: failed after costs for every direction/window/threshold/barrier/limit combination.
- Treating large MFE as directly capturable profit: rejected; first-passage results show only a small realizable gross uplift.

### Next-round reason

Round 05 tests post-signal persistence rather than pre-signal persistence. After the original impulse closes, it observes a fully closed 1m/3m/5m checkpoint, classifies causal direction-adjusted progress using a volatility baseline ending before the original impulse, and enters only at the next bar open. It measures the remaining return/MFE/MAE over 3m-60m. This determines whether an impulse is still active or already exhausted before dynamic exit, order-flow, range-bar, or footprint logic is introduced.

## Round 05 — Post-impulse state confirmation

### Research question

Can the first 1m/3m/5m of post-signal directional progress identify events that still have enough remaining continuation after a causal next-open confirmation entry?

### Fixed state definition

```text
post_progress = side * (checkpoint_close / original_next_open - 1)
post_progress_z = post_progress / (pre-impulse historical 1m volatility * sqrt(checkpoint))
```

Fixed state bands:

```text
reversal_or_flat_le_0
weak_continuation_0_0.5
moderate_continuation_0.5_1.0
strong_continuation_ge_1.0
```

### Strict timing

```text
original signal bar closes
→ wait until 1m / 3m / 5m checkpoint fully closes
→ classify post-impulse state
→ enter at the next 1m open
→ measure only the remaining 3m-60m path
```

### Production result

The causal post-signal confirmation study completed on local 1m trade bars through `2026-06-30`.

Representative deduplicated results:

| Direction | Window | Threshold | Checkpoint | State | Remaining horizon | Events | Events/month | Mean gross | Mean net | Median net | PF | Positive years |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LONG | 10m | 2.5 | 1m | strong continuation >=1.0 | 60m | 1,519 | 36.17 | +0.0843% | -0.0657% | -0.1963% | 0.835 | 0/4 |
| LONG | 15m | 2.5 | 1m | strong continuation >=1.0 | 60m | 1,135 | 27.02 | +0.0785% | -0.0715% | -0.1957% | 0.826 | 0/4 |
| SHORT | 15m | 2.5 | 5m | weak continuation 0-0.5 | 60m | 465 | 11.07 | +0.0869% | -0.0631% | -0.2038% | 0.828 | 1/4 |

### Result explanation

1. Delayed post-signal confirmation did not produce a tradable continuation entry. Across all 3,600 deduplicated state/horizon combinations, none had positive mean net, positive median net, or PF above 1 after the normal 0.15% cost.
2. Strong 1m continuation after a strong 10m-15m LONG impulse does improve mean gross relative to reversal/flat states. For LONG 10m / threshold 2.5 / 1m checkpoint / 60m remaining horizon, strong state mean gross was +0.0843% versus roughly -0.0057% for reversal/flat, an uplift near 9 bps. However median gross remained negative and all four years were negative after cost.
3. The state effect is not broadly monotonic. Only 8 of 360 LONG direction/window/threshold/checkpoint/horizon groups showed monotonic mean improvement from reversal to strong continuation, and none showed monotonic median improvement. SHORT had only 2 monotonic-mean groups and none for median.
4. More than half of events had already reversed or gone flat by the checkpoint. At threshold 2.5, roughly 55% of LONG and 57% of SHORT events were reversal/flat after 1m. Strong continuation accounted for about 25% of LONG and 24% of SHORT events after 1m, then declined at later checkpoints.
5. Strong state is persistent conditional on already being strong: for 10m-15m threshold-2.5 events, roughly 53%-57% of 1m-strong events remained strong at 3m, and about 66%-70% of 3m-strong events remained strong at 5m. This is a real state persistence pattern, but the remaining post-confirmation return is still too small to cover cost.
6. Waiting for confirmation consumes the early path advantage. The best remaining mean gross after confirmation is only about 8.7 bps, below the 15 bps normal round-trip cost. This supports immediate entry plus causal exit management rather than delayed entry.
7. The best LONG row was not dominated by only five winning events (top-five positive-return contribution about 7.6%), but the distribution remained unfavorable because median net was deeply negative. The best SHORT descriptive row had about 20% top-five contribution and was less reliable.

### Failed branch

- Delayed 1m/3m/5m price-state confirmation as an entry filter: failed after costs.
- Monotonic state-strength improvement in mean and median: failed.
- Waiting for stronger confirmation before entry: rejected because most of the small path advantage is already consumed.

### Decision

`research_continue` for the overall Edge. Post-signal state has descriptive information, especially for strong LONG impulses, but it should be used as a live holding/exit state rather than as a delayed entry condition.

### Next-round reason

Round 06 returns to immediate next-open entry and tests one causal exit mechanism only: hold while the same-window directional impulse remains active, and exit at the next open when live impulse strength decays below a fixed fraction of its original signal strength. This directly tests whether the transient MFE can be retained without adding environment, order-flow, range-bar, footprint, or other filters.

## Round 06 — Causal impulse-decay exit

### Research question

After immediate next-open entry, can a causal exit based only on decay of the same directional impulse preserve enough of the transient favorable path to cover normal execution cost?

### Fixed exit family

```text
live_retention = live_same_window_directional_impulse_z / original_signal_impulse_z

retention floors = 0.00, 0.25, 0.50, 0.75
maximum holds    = 5m, 15m, 30m, 60m
```

### Strict timing

```text
signal bar closes
→ enter next 1m open
→ after each subsequent 1m bar fully closes, update live impulse state
→ if retention <= floor, exit next 1m open
→ otherwise exit next open after max hold
```

### Boundary

No trend/session/notional/persistence/order-flow/range-bar/footprint filter is stacked. No hard stop, position sizing, portfolio logic, or parameter optimization is introduced in this round.

### Status

Pending production run.

### Round 06 production result

The same-window impulse-retention exit completed on local 1m trade bars through `2026-06-30`.

Representative best deduplicated result:

| Direction | Window | Threshold | Retention floor | Max hold | Events | Events/month | Mean hold | Mean gross | Mean net | Median net | PF | Positive years |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LONG | 10m | 2.5 | 0.25 | 15m | 5,990 | 142.62 | 8.19m | +0.0057% | -0.1443% | -0.2126% | 0.410 | 0/4 |

Additional path diagnostics for this row:

```text
mean MFE       = +0.3636%
mean MAE       = -0.2392%
mean giveback  =  0.3579%
decay exit rate = 92.45%
```

#### Result explanation

1. No retention-floor / maximum-hold combination produced a tradable result. The best mean gross was only about 0.57bps against 15bps normal round-trip cost, while median gross was already negative.
2. The rule surrendered almost all aggregate favorable excursion. Mean MFE was about +36.36bps but only +0.57bps remained at exit.
3. Maximum hold had little influence because the retention rule mechanically exited most events around eight minutes. At the same 10m / threshold-2.5 / floor-0.25 setting, decay exits rose from 92.45% at a 15m cap to 100% at a 60m cap while mean gross stayed near zero.
4. The state definition confounded genuine loss of directional pressure with rolling-window ageing. As the original shock bars left the same-length window, retention decayed even when price merely consolidated near the impulse extreme.
5. All four years were negative after normal cost. 2025 and 2026 H1 were also negative before cost for the representative best row.
6. SHORT dynamic-exit rows were worse; no SHORT combination had positive mean gross.

#### Failed branch

- `live_same_window_impulse_z / original_signal_impulse_z` as a holding or exit state: rejected.
- Retention-floor tuning and maximum-hold tuning: stopped; the failure is structural rather than a missing parameter point.

#### Decision

`research_continue` for the overall Edge. Dynamic profit protection remains a valid research objective, but the next round must first describe how profit forms and gives back rather than proposing another exit family.

## Round 07 — Post-entry path anatomy

### Research question

Before proposing another filter or exit rule, how does favorable excursion form, peak, and give back during the first 60 minutes after the Round-01 next-open entry, separately for LONG and SHORT?

### Changed from Round 06

Round 06's same-window impulse-retention exit is removed. Round 07 does not test a strategy rule. It decomposes the raw path so the next upgrade is evidence-led rather than another ungrounded parameter family.

### Fixed descriptive design

```text
maximum path              = 60m
close-profit milestones   = 10, 15, 25, 50 bps
close-giveback milestones = 5, 10, 15, 25 bps
post-activation lags      = 1, 2, 3, 5, 10, 15 minutes
```

Reported separately for every direction, impulse window, threshold, and raw/deduplicated event set:

- minute-by-minute close-return distribution;
- running MFE and MAE;
- MFE peak time and close-path peak time;
- first favorable/adverse close milestone order;
- retention after a causally observable close-profit milestone;
- time from activation to fixed close giveback;
- annual and monthly path stability;
- LONG versus SHORT differences;
- concentration of MFE in the largest events.

### Boundary

No new entry condition, environment filter, trailing stop, TP/SL, range bar, footprint, order-flow feature, position sizing, or portfolio rule is introduced. Ex-post MFE and peak time are descriptive only and cannot be treated as executable profit.

### Performance design

- One local trade-bar load and one impulse feature build per window.
- A bounded chunk kernel writes the 60m close/MFE/MAE paths to temporary float32 memmaps.
- All thresholds, event sets, years, months, activations, and giveback tables reuse those arrays.
- No per-event Python market-path loop and no repeated full-data scan per variant.
- Temporary path matrices are deleted after each direction/window.
- Synthetic-gap-dependent events are excluded.

### Status

Pending production run.

### Round 07 production result

The 60-minute path anatomy completed on local 1m OKX trade bars through `2026-06-30`.

Representative deduplicated strong-event results (`15m`, threshold `2.5`):

| Direction | Events | Events/month | Mean MFE 60m | Median MFE 60m | Mean MAE 60m | Mean final 60m | Median final 60m | Median MFE peak minute |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LONG | 4,370 | 104.05 | +0.736% | +0.458% | -0.662% | +0.0166% | -0.0546% | 16m |
| SHORT | 4,411 | 105.02 | +0.870% | +0.510% | -0.690% | -0.0218% | -0.0979% | 13m |

Key path findings:

1. A stronger impulse reliably predicts larger subsequent excursion, but both MFE and MAE expand. It is primarily a post-event volatility signal, not an unconditional directional-return signal.
2. Immediate next-open pursuit is poor for the full event pool. For `15m/2.5`, adverse `15bps` was reached before favorable `15bps` in about 54.5% of LONG events and 56.6% of SHORT events.
3. A real immediate-runner subgroup exists. Within five minutes, favorable `50bps` occurred in about 9.6% of LONG and 11.8% of SHORT events, versus adverse `50bps` in about 6.8% and 9.8% respectively.
4. Path types are heterogeneous. Roughly one third of strong events peak within five minutes, around half within fifteen minutes, and about one third peak after thirty minutes.
5. Giveback after meaningful profit is fast. After a favorable `50bps` close milestone, a `10bps` giveback had a median time of about three minutes; over 78% of LONG and 82% of SHORT cases gave back that amount within five minutes.
6. LONG and SHORT differ. LONG contains a smaller right-tail runner population that can continue after activation; SHORT tends to peak earlier and mean/median profit decay more consistently after activation.
7. The high-excursion pattern exists in every year, but final directional return and median return are not stable or large enough to cover normal cost.

Failed interpretation:

- `Large MFE means the base event is already a tradable edge`: rejected.
- `One universal fixed hold or one universal trailing rule should fit all events`: rejected.

Decision:

`research_continue`. The next step must decompose immediate continuation, pullback continuation, failure, two-sided expansion, and muted paths before another strategy rule is proposed.

## Round 08 — CVD Path-Regime Anatomy

### Research question

Can causal OKX trade-bar order flow explain which path type follows a directional price impulse, separately for LONG and SHORT?

### Why CVD is studied now

Round 07 showed that price-only impulse strength mixes several different mechanisms. Local trade bars already expose:

```text
buy_notional
sell_notional
delta_notional
large_buy_notional
large_sell_notional
large_delta_notional
trades_count
```

Round 08 uses window changes in CVD (`sum(delta_notional)`), not the arbitrary absolute cumulative CVD level.

### Fixed timescale design

```text
price impulse context: 1m, 3m, 5m, 10m, 15m
signal-time flow windows: 1m, 3m, 5m, 15m
post-signal checkpoints: 1m, 3m, 5m, 15m
future descriptive path: 60m
```

The broader price impulse is treated as context. Smaller flow windows are examined as potential execution-state information. The 15m post checkpoint is descriptive/late-state context, not assumed suitable for a fast exit.

### Path labels

Fixed economically interpretable milestones are used only to describe outcomes:

```text
immediate_runner:
  favorable 15bps occurs first and favorable 50bps is reached within 5m

pullback_runner:
  adverse 15bps occurs first, then favorable 50bps is reached within 60m

one_sided_continuation:
  favorable 25bps occurs and adverse 25bps does not occur within 60m

directional_failure:
  adverse 25bps occurs and favorable 25bps does not occur within 60m

two_sided_expansion:
  both favorable and adverse 25bps occur within 60m

muted:
  neither side reaches 25bps within 60m
```

These labels are ex-post outcomes, not entry or exit parameters.

### Causal features

At signal close and at post-signal checkpoints, report continuous features including:

```text
direction-adjusted delta pressure
large-trade delta pressure
large-trade notional share
CVD pressure acceleration versus the preceding equal window
notional/trade speed ratio versus the preceding equal window
price-flow alignment
aggressive-flow absorption proxy
price progress with non-supportive flow proxy
```

### Strict timing

```text
signal flow feature:
  uses only bars ending at the fully closed signal bar

post-k feature:
  uses only bars t+1 ... t+k
  becomes available after t+k closes
  earliest hypothetical execution is the next 1m open
```

Future path labels never enter a feature calculation.

### Research boundary

- No CVD threshold is selected.
- No entry filter is applied.
- No exit is simulated.
- No trend/session/range-bar/footprint condition is stacked.
- All quintiles and neighboring impulse thresholds are retained.
- Any useful separation must later be checked across years before becoming a small mechanism-based rule.

### Performance design

- One local trade-bar load with `build_missing=False`.
- Prefix sums for all order-flow windows.
- One bounded vectorized 60m path build per direction/window.
- Nested thresholds, path labels, yearly tables, and CVD summaries reuse arrays.
- No `iterrows`/per-event market-path scan.
- Pre-signal and future paths touching synthetic gap bars are excluded.

### Status

Pending production run.

### Round 08 production result

The full local event-level run completed on ETH-USDT-SWAP 1m OKX trade bars through `2026-06-30`.

Key evidence:

1. Signal-close CVD has almost no useful separation for immediate continuation versus failure (`AUC` roughly `0.48-0.52`). Extreme same-direction CVD at signal close often represents already-spent aggression or absorption rather than remaining momentum.
2. The first fully closed post-signal minute contains meaningful state information. For strong (`2.5`) 5m/10m/15m impulses, `post1 price > 0` and `post1 direction-adjusted total CVD > 0` had materially higher immediate-runner and one-sided-continuation rates than the other three price/CVD quadrants.
3. `post1 price <= 0` and `post1 CVD <= 0` had very low immediate-runner rates and high directional-failure rates across years.
4. `post1 CVD > 0` with `post1 price <= 0` behaves like an absorption / failed-price-progress state and is much weaker than price-and-CVD alignment.
5. Price is the main discriminator; total CVD adds modest incremental information after controlling for price. Large-trade CVD is weaker than total CVD.
6. The most promising architecture is a broader 5m/10m/15m price impulse as an event anchor and a smaller 1m price/CVD execution state. The 10m anchor was descriptively strongest, but no single window is promoted because residual executability has not yet been tested.
7. These findings are descriptive only because Round 08 path labels use future outcomes. They cannot be used directly as signals.

Decision: `research_continue`. The next round must remove label leakage entirely and recompute all remaining paths from the earliest executable open after the post1 state becomes available.

## Round 09 - Post-1m confirmed residual path

- 研究问题：第一根 post-signal 1m trade bar 完全关闭后，价格/CVD 四象限状态在下一根 open 是否仍有剩余顺势路径。
- 研究假设：较大周期价格冲击作为事件锚点，小周期价格与总 CVD 同向推进可能保留可交易的剩余延续空间。
- 与上一轮相比改变了什么：不使用未来 path label 定义或筛选状态；只用 post1 已关闭价格与 delta pressure 的正负号定义四象限状态，并从 `p+2 open` 重新计算收益、MFE/MAE 和 first-passage。
- 使用的数据：ETH-USDT-SWAP 本地 1m OKX trade bars，UTC+8 项目约定，`build_missing=False`。
- 固定 impulse window：1m、3m、5m、10m、15m。
- 固定 threshold：1.0、1.5、2.0、2.5，保留全部结果。
- 固定 residual horizons：1m、3m、5m、10m、15m、30m、60m。
- 固定 first-passage 距离：15bps、25bps、50bps；时间限制 3m、5m、10m、15m、30m、60m。
- 事件数、交易数、月均频率、mean/median net、胜率、PF、年度表现：等待本地生产运行。
- 因果时序：signal bar `p` closed -> bar `p+1` closed -> post1 state available -> bar `p+2` open residual entry。
- 未来函数审计：post1 feature available time 必须不晚于 confirmed entry；未来 path label 不得进入 state；任何 synthetic gap 依赖事件排除。
- 结果解释：Pending production run。
- 失败分支：尚未判定。
- 下一轮理由：只有确认后剩余路径在相邻窗口、相邻 threshold、多年份和正常成本下仍稳定，才允许升级为正式执行确认假设。
- 状态：research_pending。
- Script: `09_post1_confirmed_residual_path_study.py`

### Round 09 production result

The local production run completed through `2026-06-30`.

- Across 1,120 deduplicated direction/window/threshold/post1-state/horizon combinations, `mean_net > 0`: 0, `median_net > 0`: 0, normal-cost `PF > 1`: 0, and combinations with at least two positive years: 0.
- The best fixed-horizon residual result still had only about `+6.13bps` mean gross versus `15bps` normal round-trip cost; median net remained negative and annual consistency was absent.
- The originally promising `post1 price aligned + CVD aligned` state mainly described profit already realized inside the first post-signal minute. After the state became visible and entry moved to the next 1m open, residual advantage was too small.
- Decision: reject **full post1-close confirmation as a delayed entry rule**. Keep the broader hypothesis that macro/meso context may separate path types; do not tune more 1m confirmation horizons.
- Status: `research_continue`, but research scope must broaden from one post1 bar to macro environment and impulse-formation structure.

## Round 10 — Macro/Meso Path Atlas

- 研究问题：哪些冲击前宏观环境与冲击形成结构，会稳定改变立即延续、回踩延续、失败、双向扩张等路径概率？
- 研究假设：Directional Impulse 不是单一事件；宏观方向/压缩状态与冲击形成速度、方向一致性、订单流效率共同决定后续路径类型。
- 与上一轮相比改变了什么：不再围绕 post1 单根 K 线；引入冲击前 30m/60m/240m rolling macro context，以及 r0015/r0020/r0025 Range Bar 的形成速度、方向一致性和订单流结构。
- 使用的数据：本地 1m OKX trade bars + 本地 OKX Range Bars；禁止普通 K 线下载；Range Bar 只用 `load_local_data`，缺失覆盖直接 HOLD/skip，不自动构建。
- 固定 impulse window：5m、10m、15m。
- 固定 threshold：1.5、2.0、2.5，保留全部结果。
- 路径结果：立即延续、回踩后延续、单边延续、方向失败、双向扩张、无明显移动。
- 因果时序：宏观 context 截止于 impulse 开始前；Range Bar 必须 `end_ts <= signal_time`；未来路径标签只作为结果变量，绝不参与特征或信号。
- 研究边界：每个机制单独评估，不组合过滤、不选择最佳 bucket、不模拟 TP/SL、不做策略回测。
- 性能：一次加载 trade bars；Range Bar 前缀和；每个 direction/window 只构建一次 60m path memmap；threshold/年份/机制复用。
- 必须输出：机制决策矩阵，状态仅允许 `retain_for_causal_validation / weak_keep_for_more_anatomy / reject_expected_direction / insufficient_evidence`。
- 事件数、月均频率、路径比例、年度一致性：等待生产运行。
- 状态：`research_pending`。
- Script: `10_macro_meso_path_atlas.py`

### Round 10 engineering fix — range loader return contract

- Fixed the production-only pandas ambiguity where `OKXRangeBarLoader.load_local_data()` returns `end_ts` both as index and retained column.
- The script now follows the shared project usage pattern: reset the loader index before column operations, validate required range fields, and only sort if a vectorized `end_ts, bar_id` order check fails.
- Added a deterministic self-test reproducing the exact duplicate index/column contract.
- Research definitions, event timing, macro/meso features, path labels, thresholds, and outputs are unchanged.
- Status remains `research_pending` until the corrected production run completes.

### Round 10 production result

The corrected local production run completed through `2026-06-30` using complete local r0015/r0020/r0025 Range-Bar caches.

Key findings:

1. Range-Bar formation activity was the strongest Round-10 path-classification variable. Across adjacent 5m/10m/15m impulse windows, thresholds 1.5/2.0/2.5 and all three Range-Bar scales, higher bars-per-minute was associated with materially higher runner-path probability.
2. Representative `10m / threshold 2.5 / r0025` results: LONG runner probability rose from about 12.9% in the low-activity bucket to 45.5% above one Range Bar per minute; SHORT rose from about 14.9% to 49.1%. Directional-failure probability also declined, but the study had not yet shown executable target-first or cost-after-entry advantage.
3. The automatic Round-10 decision matrix understated the activity result because it compared structurally empty fixed edge buckets instead of the lowest and highest adequately populated predeclared buckets.
4. Continuation paths were more common when the market was already active rather than deeply compressed before the impulse.
5. Extremely straight/high-efficiency impulses were weaker than impulses with internal exchange and path activity, suggesting that a clean one-way spike can be closer to exhaustion.
6. A final small counter-direction Range Bar before signal close was descriptively associated with more subsequent runner paths, but this was not combined with activity and was not promoted as a filter.
7. Simple total CVD and large-trade CVD were weaker than Range-Bar path activity in this atlas.

Decision: `research_continue`. The single next mechanism is Range-Bar formation activity. Round 11 must distinguish true directional first-passage advantage from mere two-sided volatility expansion, use fully-contained Range Bars as the primary definition, and keep end-time-only membership as a boundary sensitivity check.

## Round 11 — Range activity directional validation

- 研究问题：Round 10 中 Range Bar 单位时间生成密度的路径提升，是真正方向性 first-passage 优势，还是仅代表双向波动扩大？
- 研究假设：如果 activity 是方向机制，高 activity 应提高 target-first 与 stop-first 的差值，并在严格完整落入冲击窗口的 Range Bar 口径下保持。
- 与上一轮相比改变了什么：只保留 Range Bar activity 一个变量；修正结构性空桶比较；加入 end-time-only 与 fully-contained 边界敏感性；从 next-open 重新计算 first-passage、固定时间收益、成本、年度和事件依赖。
- 使用的数据：本地 ETH-USDT-SWAP 1m trade bar；本地 r0015/r0020/r0025 Range Bar；UTC+8 项目时间约定。
- 固定 impulse window：5m、10m、15m。
- 固定 threshold：1.5、2.0、2.5。
- 固定 horizon：5m、15m、30m、60m。
- 固定 first-passage：15bps、25bps、50bps。
- 主判断：25bps 对称 first-passage / 15m；必须提高 `target_first_rate - stop_first_rate`，不能只提高两边触达率。
- 事件数、交易数、月均频率、mean/median net、胜率、PF、年度表现：等待本地生产运行。
- 因果时序：signal close 后 next 1m open；主 Range Bar 口径要求 `start_ts >= impulse_start_time` 且 `end_ts <= signal_time`；任何未来 path 只作为结果。
- 研究边界：不叠加宏观状态、末端反向 Range Bar、CVD、Footprint 或其他过滤器；不做参数优化和完整策略回测。
- 性能：Range Bar 每尺度只读一次；`searchsorted` 区间索引；first-passage 每 direction/window 只扫描一次并复用所有分层。
- 状态：`research_pending`。
- Script: `11_range_activity_directional_validation.py`

### Round 11 production result

The corrected local production run completed through `2026-06-30`.

1. Higher Range-Bar activity increased both favorable and adverse touch rates, but adverse-first increased more. Across LONG/SHORT and r0015/r0020/r0025, the high-minus-low 25bps directional first-passage gap was negative.
2. The mechanism is therefore classified as two-sided volatility expansion, not executable next-open directional continuation.
3. Low-activity impulses retained a small positive directional ordering bias, but mean gross advantage remained only around 1-2bps and far below normal 15bps round-trip cost.
4. A single LONG high-activity / 60m right-tail cell was cost-positive on mean only; median remained negative and neighboring settings did not form a stable platform.
5. Fully-contained and end-time-only Range-Bar definitions agreed, and causal audits were clean.

Decision: reject total Range-Bar activity as a direct continuation entry factor. Continue only with post-impulse micro path states that may distinguish reacquisition from failure.

## Round 12 — Post-impulse micro reacquisition single-factor atlas

- 研究问题：在 1s/3s/5s/15s trade bar 上，哪个单变量最早识别冲击后的原方向重新接管，并在状态可见后留下可执行空间？
- 研究边界：可以并行研究多个变量，但每个变量独立分层；禁止 cross-factor AND、阈值搜索和事后最优 checkpoint 选择。
- 固定检查点：15s、30s、60s、120s，保证四个 micro timeframe 使用相同信息时间。
- 单变量：方向价格进展、方向 delta pressure、成交额速度比、成交笔数速度比、方向路径效率、price-per-delta impact、大单 delta pressure、顺向 delta bar 比例、delta pressure 加速度。
- 固定结果：30s/60s/180s/300s/900s fixed horizon；15/25/50bps first-passage；30s/60s/180s/300s time limit。
- 因果：micro bar 左标签，只有 `timestamp + timeframe <= checkpoint_time` 的 bar 可进入特征；entry 为 checkpoint 时刻下一根 micro bar open。
- 数据完整性：每个 micro timeframe 与本地 1m trade bar 的分钟成交笔数对账；不一致的事件/timeframe 排除；不下载、不自动补建。
- 性能：秒级数据按 7 天块读取；一次生成 prefix arrays；全部变量、checkpoint、threshold 复用；事件状态批量向量化写出。
- 状态：`research_pending`。
- Script: `12_post_impulse_micro_reacquisition_single_factor_atlas.py`
