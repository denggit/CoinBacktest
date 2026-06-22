# ETH LF Portfolio V7B Micro Confirmation Risk Filter

V7B is based on V7/V6. It keeps LF engines unchanged and changes only the risk filter logic.

## Why V7B

V7 soft did not change results because it only reduced risk for `micro_contra=True`, and the run produced zero contradicted signals.

V7B instead tests a more practical idea:

```text
LF decides direction.
Range/footprint must confirm the LF signal.
If not confirmed, still allow the trade, but reduce entry risk.
```

Default soft behavior:

- `micro_aligned=True` -> full entry risk.
- `micro_aligned=False` but context exists -> reduced entry risk by `--micro-not-aligned-risk-scale`.
- no boost for aligned signals.
- no signal logic changes.

This is still a risk-reduction experiment, not return chasing.

## File

```text
backtest/lf/eth_lf_portfolio_v7b_micro_confirmation_backtest.py
```

## Recommended run: soft 0.5

```bat
python backtest/lf/eth_lf_portfolio_v7b_micro_confirmation_backtest.py ^
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
  --micro-not-aligned-risk-scale 0.5 ^
  --micro-contra-risk-scale 0.5 ^
  --out-dir data/reports/lf/eth_lf_portfolio_v7b_micro_confirmation/soft_05
```

## Softer run: soft 0.7

If 0.5 cuts too much return, test 0.7 without changing other logic.

```bat
python backtest/lf/eth_lf_portfolio_v7b_micro_confirmation_backtest.py ^
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
  --micro-not-aligned-risk-scale 0.7 ^
  --micro-contra-risk-scale 0.5 ^
  --out-dir data/reports/lf/eth_lf_portfolio_v7b_micro_confirmation/soft_07
```

## Off control

```bat
python backtest/lf/eth_lf_portfolio_v7b_micro_confirmation_backtest.py ^
  --start-date 2023-01-01 ^
  --end-date 2026-06-15 ^
  --warmup-start-date 2022-01-01 ^
  --preset turbo ^
  --bear-preset high ^
  --bull-preset high ^
  --bull-execution-mode inherit ^
  --micro-filter-mode off ^
  --out-dir data/reports/lf/eth_lf_portfolio_v7b_micro_confirmation/off_control
```

## What to compare

Compare against V6/V7 off control:

- total return
- max drawdown
- yearly returns
- `micro_filter_action` distribution in trades
- whether risk-scaled V7B beats original V6 at similar drawdown

A useful result is not necessarily higher 1x return. The target is lower drawdown / smoother equity so that risk scaling can safely increase total return later.
