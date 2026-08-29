# Human Trader Replay Lab V1.6 — Manual Order Ticket + Replay Rewind

## Purpose

This revision fixes three interaction gaps in V1.5:

1. LIMIT entry price is now manually editable rather than forced to use the chart-selected price.
2. SL and TP can be attached when LONG/SHORT is submitted.
3. Replay can move backward without contaminating the current training trajectory with decisions from the abandoned future path.

## Order semantics

The chart-selected price remains a convenience input, not an order constraint. The trader can type Entry/SL/TP directly or copy the selected chart price into any field.

Pending limit orders store the planned bracket in `LIMIT_ORDER.payload.stop_loss` and `take_profit`. When the order fills, the fill event is followed by `SL` / `TP` events at the same causally-known fill time.

## Rewind semantics

Rewind never exposes data after the new cursor. Active events after the target cursor are archived via `events.is_active=0`. They are omitted from the active episode trajectory and active-order reconstruction, but preserved under `discarded_events` in export schema version 2.

This means rewinding across a prior limit fill restores the earlier resting order state rather than leaving the future fill active.


## V1.6.1 schema migration hotfix

- Fix startup migration from V1.0-V1.5 SQLite stores that do not yet contain `events.is_active`.
- Migration order is now: create legacy-compatible tables/index -> inspect schema -> add `is_active` -> create V1.6 active-branch index.
- Existing events are preserved and receive `is_active=1`.
