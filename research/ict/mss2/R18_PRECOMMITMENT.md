# R18 Precommitment — Independent Positioning-Unwind Path Atlas

Date frozen: 2026-08-17, before any R18 outcome calculation.

## Independent mechanism

R18 does not repair R13–R17, condition on a liquidity sweep, repeat the rejected
failed-auction study, or reuse a future-labelled OI turning point. It tests one
all-market state transition:

```text
completed 1h directional price move + rising Binance base OI
→ first completed 5m Binance base-OI change from nonnegative to negative
→ contemporaneously completed OKX 5m close reacquires the prior 5m extreme
→ next observable OKX 1m open
```

Binance USD-M `ETHUSDT` OI is a cross-exchange positioning proxy. Execution and
all price paths remain OKX `ETH-USDT-SWAP`.

## Frozen causal event contract

- Binance metrics are official 5-minute observations loaded only through
  `src.data_feed.binance_futures_metrics_loader`.
- `oi_available_time = source timestamp + 1 minute`. No metrics row may be used
  before that publication timestamp.
- OKX 1-minute bars are loaded only through `src.data_feed` and aggregated into
  complete left-labelled 5-minute bars. A price bar is usable only at its end.
- Each OI observation is aligned to the latest complete OKX 5-minute price bar
  whose end is at or before `oi_available_time`.
- The build state is measured on the immediately preceding valid metrics row:
  its one-hour OKX close change and one-hour Binance base-OI change must both be
  available then. Rising OI plus falling price arms a potential Long unwind;
  rising OI plus rising price arms a potential Short unwind.
- Release requires the current causal 5-minute base-OI change to be negative
  and the preceding causal 5-minute base-OI change to be nonnegative. This is
  the first sign transition, not every bar in a release episode.
- Long stabilization requires the current completed OKX 5-minute close to be
  strictly above the preceding completed 5-minute high. Short stabilization is
  the strict close-below-prior-low mirror.
- The immediately preceding metrics observation must be 4–6 minutes old. The
  5-minute and one-hour OI baselines must be no more than one minute stale
  relative to their nominal windows. No gap interpolation is allowed.
- Current and baseline base OI must be strictly positive. OI USD, taker ratios,
  account ratios, funding, and all future OI fields are excluded from admission.
- The signal becomes available at the later of the OI publication time and the
  stabilization price-bar end. Entry is the first OKX 1-minute open whose bar
  start is at or after that time. A signal arriving seconds after a minute
  boundary therefore waits for the following 1-minute open.
- At most one event exists per release transition. Long and Short are evaluated
  separately and are never pooled in primary evidence.

## Frozen stop, targets, and ordering

- Causal volatility is the 12-bar simple mean true range on completed OKX
  5-minute bars (one hour), with 12 observations required.
- Long stop: the lower low of the stabilization bar and its immediately prior
  5-minute bar, minus `0.25 ×` causal 5-minute ATR. Short is the exact mirror.
- Maximum stop distance is `1.50%` of entry. Wider or invalid geometries are
  skipped rather than resized or rescued.
- Structural target: the opposite extreme of the 12 completed 5-minute price
  bars ending at the build observation—one-hour high for Long, one-hour low for
  Short—frozen before the release signal. It must remain beyond entry.
- Diagnostic fixed barriers are 1R, 2R, and 3R. The structural barrier is also
  reported; no target is selected from results.
- Diagnostic path horizon is 24 hours. An unresolved path exits at the final
  1-minute close and is labelled `horizon_exit`; this is not a proposed live
  time stop.
- If stop and target touch in the same 1-minute OHLC bar, stop wins.
- Round-trip market cost is 0.11%; report 1×, 2×, and 3× cost.

## Frozen data-quality treatment

The pre-outcome audit found:

- 262,341 Binance rows from 2023-01-01 through 2025-06-30, with zero duplicate
  timestamps and zero missing base-OI fields;
- 188 partial archive days in that window, mostly 287 rather than 288 rows;
- 416 non-exact-five-minute timestamp intervals, including one 10.5-hour gap;
- 81 rows with nonpositive base OI and 91 rows with nonpositive OI USD,
  concentrated in nine months;
- 5-minute and one-hour causal base-OI features are available on 261,962 and
  261,950 rows respectively under a one-minute baseline tolerance;
- the OKX 1-minute execution series is complete from 2022-01-01 through
  2025-06-30: 1,838,880 rows, zero gaps, duplicates, null OHLCV values, or OHLC
  consistency violations.

R18 drops nonpositive base-OI observations and any transition crossing an
ineligible gap. Nonpositive OI-USD observations are recorded but do not gate an
event because OI USD was frozen outside admission. R18 does not fill,
interpolate, or infer missing positioning data. Ratio-column nulls are
immaterial because ratios are not used.

The local Binance cache ends at project time `2026-07-01 07:55:00`
(`2026-06-30 23:55:00` UTC), short of the master request through 2026-08-15.
That does not affect the frozen discovery/validation calculation but prevents a
claim of current live-data readiness.

## Splits and physical leakage boundary

- Warmup begins 2022-01-01.
- Discovery entries: 2023-01-01 through 2024-12-31.
- Validation entries: 2025-01-01 through 2025-06-30.
- July 2025 is embargoed.
- Existing holdout begins 2025-08-01 and remains sealed. Only aggregate causal
  candidate counts may be emitted; no holdout price path or economics may be
  computed.
- The causal event table must reject any column beginning `future_` or
  containing `oracle`. Future OI paths, first-passage outcomes, and manual-review
  rankings live only in physically separate outcome artifacts.

## Decision boundary

R18 is a mechanism/path atlas, not a portfolio or live strategy. The exact
unfiltered Long or Short transition can justify a later strategy version only
if the same direction has positive 2×-cost evidence in both discovery and
validation, credible sample size, yearly breadth, and survival after removing
the top five winners. Otherwise that direction is rejected immediately. No
magnitude threshold, ratio feature, regime filter, target choice, stop change,
funding feature, or ML rescue is permitted in R18.
