# Range Footprint Event Lab V1

This is a research tool, not a trading strategy.

It scans OKX ETH range bars and optional range-footprint max-bucket features, creates objective event samples, and measures future MFE/MAE and first-touch outcomes.

The purpose is to find whether any order-flow event has stable expectancy before writing another full strategy.

## Event types

V1 produces these event groups:

- `sweep_low_reclaim_sell_pressure` long
- `sweep_high_reclaim_buy_pressure` short
- `initiative_breakout_up_buy_pressure` long
- `initiative_breakout_down_sell_pressure` short
- `delta_divergence_low_reclaim` long
- `delta_divergence_high_reclaim` short

Event detection uses shifted rolling swing levels and shifted rolling quantiles, so the event features themselves do not use future data.

Future labels are only used after the event to calculate MFE/MAE and hit rates.

## Run command

```bat
python tools/research_range_footprint_events.py ^
  --start-date 2023-01-01 ^
  --end-date 2026-06-15 ^
  --warmup-start-date 2022-01-01 ^
  --range-pct 0.002 ^
  --price-step 1 ^
  --swing-window 80 ^
  --quantile-window 300 ^
  --pressure-quantile 0.88 ^
  --horizon-bars 80 ^
  --out-dir data/reports/research/range_footprint_event_lab_v1_default
```

## Outputs

- `range_footprint_event_lab_v1_events.csv`
- `range_footprint_event_lab_v1_summary.csv`
- `range_footprint_event_lab_v1_yearly.csv`
- `range_footprint_event_lab_v1_config.json`

Start with `summary.csv`. A candidate is interesting only if:

- event count is not too small,
- `net_exp_2r_stop_r` is positive,
- yearly results are not supported by only one year,
- average MFE/MAE ratio is attractive,
- the edge survives cost assumptions.

## Notes

This tool is designed to avoid the previous pattern of repeatedly hand-tuning a full strategy. Use it to discover stable event-level edge first, then turn only robust events into a backtest strategy.
