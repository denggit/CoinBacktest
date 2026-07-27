# OKX Flow–Impact State Round 01 Runbook

## Purpose

Build the long-history pressure-event atlas before any TP/SL or Liquidity enhancement work.

## Default Windows command

```bat
python research\mhf\flow_impact_state\01_pressure_event_atlas.py --symbol ETH-USDT-SWAP --timeframe 1m --warmup-start-date 2022-01-01 --start-date 2023-01-01 --end-date 2026-06-30 --min-pressure-z 1.5
```

The script is cache-only. It reads `data\okx_trade_bars.db` through `src.data_feed.okx_trade_bar_loader.OKXTradeBarLoader` and never downloads or rebuilds missing data.

## Fast validation

```bat
python research\mhf\flow_impact_state\01_pressure_event_atlas.py --self-test
```

## Output

```text
data/reports/research/mhf/flow_impact_state/01_pressure_event_atlas
```

Primary review artifact:

```text
gpt_review_pack.zip
```

## Interpretation order

1. `02_event_frequency.csv`: verify the selected broad event definition remains near 10–30 unique pressure processes/day before any trading filter.
2. `02b_threshold_frequency_calibration.csv`: compare fixed pressure-z thresholds by frequency only; never select the threshold from forward returns.
3. `02c_yearly_event_frequency.csv`: reject event definitions whose frequency disappears in individual years.
4. `04_response_state_summary.csv`: compare effective response with opposite/absorbed response for continuation and reversal.
5. `08_first_touch_by_state.csv`: reject cells where both favorable and adverse touch merely rise together.
6. `09_pressure_duration_summary.csv`: determine whether pressure persistence differs by response state.
7. `06_yearly_response_summary.csv`: require multi-year consistency.
8. `12_signal_audit.csv`: every retained event must be next-open and synthetic-gap independent.

Do not choose TP/SL from Round 01. The next round may isolate only one mechanism difference.
