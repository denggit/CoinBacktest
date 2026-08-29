# SOXL ICT MSS R02 Patch Manifest

## Purpose

Correct the R01 ICT semantics before any further optimization:

`premarket liquidity -> sweep episode -> terminal extreme -> dynamic MSS -> displacement FVG -> retrace entry`

## Added

- `src/research_common/ict/premarket_mss_fvg_v2.py`
  - sweep-episode state machine;
  - dynamic MSS reference;
  - fresh opposite-liquidity gate;
  - two-sided 15m swing significance;
  - duplicate-safe grouped diagnostics.
- `research/ict/soxl_premarket_mss_fvg/02_sweep_episode_state_machine_research.py`
- `src/data_feed/alpaca_stock_loader.py`
  - Alpaca historical US-stock bars adapter;
  - SIP/IEX/BOATS feed selection;
  - pagination and local SQLite cache;
  - UTC-aware timestamps.
- `tools/prebuild_alpaca_stock_bars.py`
- R02 and Alpaca loader tests.

## Modified

- `research/ict/soxl_premarket_mss_fvg/README.md`
- `research/ict/soxl_premarket_mss_fvg/RESEARCH_LOG.md`

## Report location

All R02 outputs default under:

`data/reports/research/ict/soxl/mss/r02_state_machine/`

## Important research boundary

Alpaca SOXL spot data is only a candidate long-history proxy.  Do not combine its PnL with OKX perpetual PnL or treat the instruments as interchangeable until an overlap audit shows that premarket extremes, sweeps and R02 MSS/FVG signals are sufficiently aligned on the common 2026 period.

## Verification

- R02 self-test: PASS.
- Targeted pytest: 15 passed.
- Research code does not call Alpaca/OKX HTTP or SQLite directly; external data access stays in `src/data_feed`.
