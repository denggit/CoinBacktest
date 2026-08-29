# ICT MSS Research

This directory studies whether a causal ICT Market Structure Shift has a tradable edge on ETH-USDT-SWAP using bare candles only.

## Non-negotiable timing rules

- All market data comes from `src.data_feed`; research code does not create a second data API.
- HTF candles are unavailable until the full source candle has closed.
- HTF and execution-timeframe swing pivots are unavailable until all right-confirmation bars have closed.
- A liquidity-quality feature may use information only through the **open of the sweep bar**. The sweep bar's high/low and all post-sweep data are forbidden inputs to the liquidity classifier.
- MSS displacement must close through a micro swing that was already confirmed before displacement began.
- A three-candle FVG becomes known only after candle three closes. The limit starts on the following execution bar.
- Bare-OHLC intrabar ambiguity is resolved conservatively: fill-candle favorable extremes cannot hit TP/MFE and same-bar TP+SL is stop-first.

## R01 — unconditional mechanism test

`01_ict_mss_edge_discovery.py` tested 15m/30m/1H/4H swing sweep -> 1m MSS -> displacement/FVG -> FVG-near-edge limit retest.

Actual 2022 warmup / 2023-2026H1 research result:

- HTF liquidity levels: 134,960
- first-sweep episodes: 57,419
- causal sweep->MSS/FVG pairs: 259,959
- causal audit violations: 0
- edge-gate passes: 0

The unconditional mechanism is materially negative after costs and is approximately flat-to-negative before costs. Therefore R01 is frozen as **no edge**. Do not loss-tune its displacement thresholds.

Useful relative clues, not edges:

- 1H/4H liquidity generally degraded less than undifferentiated 15m/30m swings.
- larger 1m micro structure (`order=5`) was less bad in later years than the shortest structure.
- forcing MSS very quickly after the sweep did not help and often hurt.

## R02 — liquidity taxonomy + 1m/2m + calendar/session study

`02_ict_mss_liquidity_taxonomy.py` directly tests the hypothesis raised by R01: not every swing is meaningful liquidity.

It keeps **all still-unconsumed HTF swings**, including old/remote levels; it never searches only the nearest swing. Before each first sweep it classifies the level/episode using fixed causal dimensions:

- source timeframe and confirmed pivot order;
- confirmed pivot prominence and pivot rejection;
- age since the level first became tradable information;
- distance from price when the level became active;
- maximum price excursion away from the level **before the sweep bar**;
- same-price active swing clustering at 5/10/25 bp;
- whether clustered liquidity comes from multiple HTFs;
- structural-major / mature / remote / stacked combinations.

Execution is tested natively on both:

- 1m MSS/FVG;
- complete 2m candles causally aggregated from the same 1m data.

Physical search/fill/outcome windows are measured in minutes, so 1m and 2m receive equal clock-time opportunity.

Calendar/session output includes:

- weekday vs weekend and each UTC weekday;
- Asia session;
- London session;
- New York session;
- ICT London kill zone;
- ICT New York kill zone;
- US cash-open 09:30-11:00 New York time;
- two-hour New York clock buckets.

London/New-York session clocks use real DST conversion from the project's configured candle timestamp offset.

## Research split / leakage note

Warmup remains 2022-01-01 onward; research runs 2023-01-01 through 2026-06-30.

Because aggregate 2026H1 R01 outcomes have already been inspected, R02 does **not** falsely call 2026H1 pristine sealed data. R02's liquidity thresholds and session windows are fixed in source before its outcomes are inspected; a promoted candidate must be independently positive in 2023, 2024, 2025 and 2026H1, survive 2x cost and removal of the top 10 winners.

## Run

Windows, repository root, one line:

```text
python research\ict\mss\02_ict_mss_liquidity_taxonomy.py
```

Upload after completion:

```text
data\reports\research\ict\mss\02_ict_mss_liquidity_taxonomy\gpt_review_pack.zip
```

Do not promote a strategy because one exploratory session or age bucket looks good. Only frozen candidate gates can trigger a separate executable strategy backtest.
