# Human Trader Replay Lab V1.13 - Chart Line Visibility

## Goal

Separate chart cleanup from decision correction.

- `删线`: hide the visual line only. The original `LIQUIDITY` / `TARGET` / `MARKER` event remains active in the Decision Timeline and training data.
- `恢复线`: show a previously hidden line again.
- `删记录`: archive the original annotation from the current decision branch. This is reserved for mistaken annotations.

## Audit model

Line visibility is append-only and stored as `ANNOTATION_LINE_VISIBILITY` events. These presentation events are not shown as normal Decision Timeline items, but remain available in JSON export for audit/replay reproducibility.

Rewind naturally restores the historical chart state because visibility events after the rewind cursor are archived with the rest of the future branch.
