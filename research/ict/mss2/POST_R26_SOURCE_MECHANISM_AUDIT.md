# Post-R26 Source and Mechanism Audit

Date: 2026-08-17

## Purpose

R26 rejected the last complete but previously unused Binance positioning-ratio
state. This audit asks whether a clean R27 can be precommitted without recycling
a failed family, opening sealed periods, downloading a substitute history, or
inventing a feature threshold.

## Local source inventory

| Source lane | Visible coverage | Research consequence |
| --- | --- | --- |
| OKX ETH-USDT-SWAP 1m OHLCV | 1,838,880 / 1,838,880 expected rows, 2022 through 2025H1 | Available, but the major price-only structure, reversal, continuation, trend, impulse, compression, and Range-Bar families have been tested. |
| OKX ETH raw trades / trade bars / Range Bars / footprints | Broad visible history | Available, but activity, CVD, notional expansion, absorption, panic-wick, Range-Bar continuation, and run exhaustion have already failed as standalone entries or rescue factors. |
| OKX BTC-USDT-SWAP 1m | 1,838,880 / 1,838,880 expected rows | R22's BTC-led ETH catch-up mechanism failed both directions; changing beta/sigma or lag thresholds is frozen. |
| Other OKX swap / spot price series | The actual 1m table catalog contains only ETH-USDT-SWAP and BTC-USDT-SWAP | No alternate-leader rotation or spot-led perpetual study is source-backed. This supersedes symbol-by-symbol guessed probes. |
| OKX contract OI | Daily only, 2024-01-01 through 2025-06-30 | Missing the entire 2023 discovery year; insufficient for a two-year discovery design. |
| OKX funding / mark / liquidation | Tables exist but each has zero pre-embargo rows | Basis, funding sign/crowding, mark dislocation, and liquidation-flow studies are unavailable. |
| OKX books / liquidity primitives / liquidity map | 0 / 1,277 expected pre-embargo days in each lane | Archive contents were not opened; only filenames before the physical cutoff were counted. |
| Binance USD-M 5m metrics | 367,365 / 367,776 expected rows plus all 1,277 raw archive days | Base-OI release/rebuild failed in R18/R19; relative positioning leadership failed in R26. The small timestamp gaps are handled causally and do not justify a new ratio/filter rescue. |

## Repository candidate audit

### Volatility compression / expansion

Not novel. The R21 precommitment records that intraday compression breakout and
expansion exhaustion had already been rejected before the daily channel study.
The market-process integration lab explicitly replays `compression_breakout`
and `expansion_exhaustion`, while the MHF branch performs a broad compression /
burst / contraction screen. R10/R11 additionally show Range-Bar activity is
two-sided volatility expansion rather than directional edge. Do not create R27
by choosing another compression window or breakout threshold.

### Price-impact failure / absorption / failed auction

Not novel. MSS2 R03/R04, the post-sweep micro/footprint chain, the market-
process order-flow branch, the directional-impulse CVD chain, and R23 already
test trade-bar/footprint absorption, failed price progress, reclaim, large
trades, and taker flow. The broad first-layer order-flow study is below PF one
at every reported horizon and year. Do not run the existing sell-pressure shock
screen as an MSS2 shortcut or promote a new absorption-score cut.

### Archived Q70 reclaim model

Not a valid prior. Its immutable lifecycle is
`ARCHIVED_AFTER_SEALED_HOLDOUT_FAILURE`; the untouched 2026H1 result had PF 1.09,
only two positive months, 3x-cost failure, calibration drift, and winner
concentration. It cannot be relabelled as a current sleeve or repaired with the
already-opened period.

### Additional cross-asset leaders

No local visible histories exist for the checked SOL/XRP/DOGE/BNB/LTC/ADA/AVAX
swaps. BTC is the only full overlap and its catch-up branch is frozen after R22.
Source substitution or remote downloads are outside the current boundary.

## Decision

No clean R27 is justified from the current local source set. Writing one now
would mean one of:

- tuning a stopped family;
- changing an asset after a failed lead/lag result without a local source;
- screening another price/flow threshold after repeated gross-null evidence;
- using sealed-period books/liquidity data;
- or fabricating spot/funding/basis state from price proxies.

That would be research for research's sake and violates the strategic-reset
discipline. R27 remains unassigned.

## Source-backed next mechanism, when data exists

The highest-value unresolved economic hypothesis remains genuine cross-market
price discovery and carry state:

1. complete OKX ETH spot plus ETH-USDT-SWAP overlap from warmup through 2025H1;
2. or complete funding, mark, and index/basis history over the same window;
3. or complete historical books/liquidity primitives over both discovery years
   and validation.

The first study after such coverage exists should use one frozen causal
mechanism—spot-led swap convergence, funding/basis unwind, or persistent book
liquidity response—not a grid. Until then, preserve July and the 2025-08-01
holdout, retain all negative results, and do not open another adjacent threshold
study merely to increment the research number.

## Reproducible source-readiness gate

The final inventory is now produced by:

```text
python research/ict/mss2/00_pre_r27_source_readiness_audit.py
```

Unlike the earlier hand-picked checks, this enumerates every actual timestamped
series in the supported local market databases and every actual series directory
in the raw/derived archive lanes. All SQL has a strict `< 2025-07-01` predicate;
dated-file coverage also stops before that day. The generated three seal checks
pass, 28 visible SQLite series and six archive-series directories are recorded,
and zero mechanisms satisfy both `source_gate=READY` and `novelty=NOVEL`.

Generated evidence is stored under
`data/reports/research/ict/mss2/pre_r27_source_readiness_audit/`. R27 remains
unassigned, and this audit is a source gate rather than R27 itself.
