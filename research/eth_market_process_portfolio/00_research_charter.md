# ETH Market Process Portfolio Research Charter

## 1. Goal

Build an ETH-USDT-SWAP portfolio with a durable positive expectancy, a smoother equity curve, and execution assumptions suitable for a non-VIP OKX account and copy trading.

This project does not start from an existing portfolio edge. It studies market processes first and promotes only independently validated processes into the portfolio layer.

## 2. Frozen research boundaries

- Symbol: `ETH-USDT-SWAP`
- Warmup start: `2022-01-01`
- Default research window: `2023-01-01` through `2026-06-30 23:59:59`
- Default round-trip cost: `0.11%`
- Signal timing: closed data only; execution at the next executable bar/event
- High-timeframe alignment: use `available_time`, never bar start time
- No future information, optimistic same-bar path assumptions, loss-driven parameter tweaks, or single-point parameter claims
- All short-history sources must be labelled `WINDOW_ONLY` and may not be presented as full-history evidence

## 3. Research modules

### Liquidity

Studies resting liquidity, wall lifecycle, liquidity voids, sweeps, consumption, cancellation, replenishment, and price response. It does not own final entry or portfolio sizing rules.

### Order flow

Studies aggressive trading, CVD paths, large trade flow, book response, price impact efficiency, absorption, exhaustion, and continuation/failure states.

### Volatility

Studies compression, expansion, shock, path efficiency, persistence, and post-expansion exhaustion. It must not reduce market state to a single ATR threshold.

### Positioning

Studies open interest, funding, mark/perpetual basis, liquidation, crowding, leverage build-up, and deleveraging. Its evidence window is limited by actual local data coverage.

### Integration

Combines validated single-module mechanisms into causal market processes. It cannot rescue a failed module by searching a large joint parameter grid.

### Portfolio

Accepts only promoted, replay-audited process candidates. It owns conflict routing, de-duplication, risk allocation, exposure caps, walk-forward evaluation, and copy-trading suitability.

## 4. Promotion ladder

1. Data-quality gate
2. Single-factor descriptive atlas
3. Causal event study
4. Mechanism/path validation
5. Cross-period and parameter-neighbourhood stability
6. Realistic entry/exit replay with cost and delay
7. Candidate sleeve
8. Cross-module integration
9. Portfolio walk-forward and stress tests

A study may stop at any stage. Failure is recorded and archived; it is not repaired through unrestricted parameter search.

### Condition-ladder rule

New strategy discovery starts from the widest causally valid candidate universe.
Conditions are introduced one at a time, and every child condition must report
its retention, gross/net return increment, frequency and cross-period
consistency relative to its immediate parent. A research branch may not begin
by AND-ing a large set of environment, PA and order-flow conditions into a rare
finished signal.

## 5. Required evidence

Every promoted candidate must report at least:

- exact sample and data coverage
- event count and events per month
- year/quarter breakdown
- forward path, MFE and MAE
- win rate, mean/median net return, profit factor
- realistic fee, slippage and delay stress
- parameter neighbourhood
- top-trade concentration and removal test
- walk-forward/holdout result
- causal replay audit, including higher-timeframe available time
- memory/runtime profile for heavy research

## 6. Research sequencing

1. Local data coverage audit
2. Liquidity primitive and wall lifecycle audit
3. Order-flow response atlas
4. Liquidity × order-flow process research
5. Volatility state research
6. Positioning overlay research
7. Integrated process scoring
8. Candidate strategy replay
9. Portfolio construction

## 7. Code placement

- Research-specific experimental code stays under this directory.
- Only stable, independently reusable components may be promoted into `src/research_common`, `src/market_state`, or `src/portfolio_common`.
- Data access must use `src.data_feed` loaders. Research scripts do not call exchange APIs directly.
- Research scripts must not import other numbered research scripts.
- Large inputs must be streamed or chunked; multi-year raw trades/books must not be loaded into memory at once.

## 8. Stop rules

Stop or redirect a research branch when:

- net edge does not clear realistic costs
- the result is concentrated in one year, session, or a few trades
- neighbourhood stability is absent
- holdout degrades materially
- event frequency is too low for the intended sleeve
- the mechanism cannot be implemented causally in live trading
- additional complexity produces declining out-of-sample value

## Data-access boundary (frozen)

All research code must obtain market data through public interfaces under
`src.data_feed`. Research scripts must not implement exchange HTTP/WebSocket
requests, raw archive parsing, SQLite schema knowledge, or independent aggregation
pipelines.

When a required reusable dataset or aggregation does not yet exist:

1. add or extend a general-purpose loader/builder under `src.data_feed`;
2. keep exchange-specific fetching, local caching, chunking and schema normalization there;
3. expose a stable strategy-facing API;
4. add focused tests before a research script consumes it.

Research-only event definitions, labels and experimental transforms remain inside
this research domain. Generic data acquisition and generic data aggregation do not.
