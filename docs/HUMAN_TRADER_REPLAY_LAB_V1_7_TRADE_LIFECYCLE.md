# Human Trader Replay Lab V1.7 — Trade Lifecycle / Outcome Recorder

## Goal

Complete the manual-trader training sample from market judgment through order, fill, protection, exit and realized outcome.

## New lifecycle events

```text
LIMIT_ORDER (when used)
ORDER_FILLED
TRADE_OPEN
SL / TP
TAKE_PROFIT_HIT | STOP_LOSS_HIT | TRADE_EXIT_AMBIGUOUS | MANUAL_EXIT
TRADE_CLOSED
```

`TRADE_CLOSED` stores:

- trade_id / side
- entry and exit price/time
- exit_reason
- gross_return_pct
- fee_round_trip_rate = 0.0011
- net_return_pct
- r_multiple when an original SL exists
- mfe_pct / mae_pct
- holding_minutes
- trigger_bar_time / trigger_bar
- path audit flags

## Causal path policy

- MARKET entry at the cursor open can use the subsequently closed entry 1m bar.
- Resting LIMIT fills happen intrabar. Their trigger bar is excluded from automated post-entry SL/TP detection because OHLC does not identify whether an extreme occurred before or after the fill.
- If a later closed 1m bar touches both SL and TP, event type is `TRADE_EXIT_AMBIGUOUS`; resolution is explicitly `conservative_stop_assumption`.
- MFE/MAE excludes the exit-trigger bar's unknown post-exit path and includes the actual simulated exit price.

## Persistence

Events are committed to SQLite immediately. End Episode is a finalizer, not a save operation. It appends `EPISODE_SUMMARY` and changes episode status from `active` to `closed`.

## Legacy compatibility

V1.6 LONG/SHORT fills without explicit TRADE_OPEN events are reconstructed as legacy active trades. Refreshing an active Episode can therefore catch up a TP/SL that was already crossed before installing V1.7.
