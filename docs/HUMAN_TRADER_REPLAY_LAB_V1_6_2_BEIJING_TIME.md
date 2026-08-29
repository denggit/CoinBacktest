# Human Trader Replay Lab V1.6.2 - Beijing Time UI

## Change

The Replay Lab now displays chart, cursor, forming-bar, timeline, marker-anchor, and coverage timestamps in `Asia/Shanghai` (Beijing time).

SOXL session semantics remain anchored to `America/New_York` internally:

- Episode starts at 07:30 New York time.
- US market open remains 09:30 New York time.
- Replay ends at 16:00 New York time.
- US daylight-saving transitions are converted automatically.

Examples:

- US daylight saving: 07:30 ET -> 19:30 Beijing; 09:30 ET -> 21:30 Beijing.
- US standard time: 07:30 ET -> 20:30 Beijing; 09:30 ET -> 22:30 Beijing.

Persisted replay/event timestamps remain in the original internal New York wall-time representation so this UI-only change does not alter causal order, fills, rewind behavior, or existing databases.
