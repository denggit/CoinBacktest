# Human Trader Replay Lab V1.6.3 - Annotation correction

## Goal

Allow mistaken BSL / SSL / other chart annotations to be removed without destroying the behavioral audit trail.

## UI

The Decision Timeline shows a `删除` button for active `LIQUIDITY`, `TARGET`, and shared `MARKER` events. The annotation disappears from the chart immediately after deletion.

## Persistence semantics

Deletion is soft-delete only:

1. the target annotation is changed to `is_active=0`;
2. it disappears from active `events` and therefore from chart rendering;
3. it remains in exported `discarded_events`;
4. an `ANNOTATION_DELETE` event is appended with the original event id, type/kind/label, and price.

This preserves the fact that the trader initially marked a level and later corrected that judgment. Order, fill, and position lifecycle events are intentionally not deletable through this endpoint.
