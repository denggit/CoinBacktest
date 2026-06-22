# ETH LF Portfolio V7 Range/Footprint Context

V7 is based on `backtest/lf/eth_lf_portfolio_v6_momentum_bear_bull_reclaim_backtest.py`.

It keeps the V6 LF engines unchanged:

- Momentum Breakout V3 as top priority.
- Bear Short Engine V3 standalone short as second priority.
- Bull Range Reclaim V2 long as low-priority supplement.
- Single active position, no hedge.

V7 adds a range-bar / footprint context risk filter.

## Core idea

LF still decides direction. Range-bar / footprint context only judges whether the just-confirmed 4H signal has hostile micro orderflow.

Default mode is `soft`:

- If micro context is not hostile, trade normally.
- If micro context contradicts the LF signal, reduce entry risk by `--micro-contra-risk-scale`.
- It does not boost aligned signals by default.
- It does not change LF signal generation.

This is designed to reduce bad entries / drawdown first. If drawdown improves, risk scaling can be studied later.

## Timing / no-future design

For each 4H LF signal row, V7 aggregates range bars whose `end_ts` falls inside that completed 4H bar:

```text
range_end_ts.floor('4h') == signal_4h_bar_open_time
```

The micro context is only used for the next 4H open entry, so it is known at the time V6 would already enter. It does not use range bars after the entry time.

## New file

```text
backtest/lf/eth_lf_portfolio_v7_range_footprint_context_backtest.py
```

## Suggested first run: soft mode

```bat
python backtest/lf/eth_lf_portfolio_v7_range_footprint_context_backtest.py ^
  --start-date 2023-01-01 ^
  --end-date 2026-06-15 ^
  --warmup-start-date 2022-01-01 ^
  --preset turbo ^
  --bear-preset high ^
  --bull-preset high ^
  --bull-execution-mode inherit ^
  --micro-filter-mode soft ^
  --range-pct 0.002 ^
  --price-step 1 ^
  --micro-contra-risk-scale 0.5 ^
  --out-dir data/reports/lf/eth_lf_portfolio_v7_range_footprint_context/soft
```

## Baseline control

This disables the range/footprint filter and should be close to V6 behavior except names/output paths.

```bat
python backtest/lf/eth_lf_portfolio_v7_range_footprint_context_backtest.py ^
  --start-date 2023-01-01 ^
  --end-date 2026-06-15 ^
  --warmup-start-date 2022-01-01 ^
  --preset turbo ^
  --bear-preset high ^
  --bull-preset high ^
  --bull-execution-mode inherit ^
  --micro-filter-mode off ^
  --out-dir data/reports/lf/eth_lf_portfolio_v7_range_footprint_context/off_control
```

## Optional strict mode

Strict mode blocks contradicted signals. This is more aggressive and should not be trusted unless it improves yearly robustness.

```bat
python backtest/lf/eth_lf_portfolio_v7_range_footprint_context_backtest.py ^
  --start-date 2023-01-01 ^
  --end-date 2026-06-15 ^
  --warmup-start-date 2022-01-01 ^
  --preset turbo ^
  --bear-preset high ^
  --bull-preset high ^
  --bull-execution-mode inherit ^
  --micro-filter-mode strict ^
  --range-pct 0.002 ^
  --price-step 1 ^
  --out-dir data/reports/lf/eth_lf_portfolio_v7_range_footprint_context/strict
```

## New audit fields

The signal audit and trade CSV include fields such as:

```text
micro_context_available
micro_aligned
micro_contra
micro_entry_risk_scale
micro_filter_action
rf_bar_count
rf_micro_return_pct
rf_close_pos
rf_delta_sum
rf_imbalance
rf_taker_buy_ratio
rf_max_sell_bucket_share
rf_max_buy_bucket_share
```

## How to evaluate

Do not judge only by total return. Compare V7 soft with V6 / off control on:

- total return
- max drawdown
- yearly returns
- trade count
- engine counts
- micro_contra trade results
- whether risk-scaled V7 has better return/drawdown than risk-scaled V6

If soft mode lowers drawdown while preserving most returns, then it creates room to increase risk multiplier later.
