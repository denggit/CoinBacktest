# Human Trader Replay Lab V1.14 — Vectorized Fast Forward

## Playback speed remap

- 慢: 1m per request, 300 ms delay (roughly the old normal speed)
- 正常: 1m per request, 45 ms delay (roughly the old very-fast speed)
- 快: 2m per request, 20 ms delay
- 很快: 5m per request, 5 ms delay

Fast/very-fast modes intentionally batch several replay minutes into one request to reduce HTTP/render overhead.

## +15m / +30m / +60m

The old engine advanced one minute at a time and reconstructed order/trade state on every minute. V1.14 uses one cache-backed 1m chunk and vector masks to locate the earliest hidden lifecycle boundary:

1. resting limit fill;
2. active trade stop-loss;
3. active trade take-profit;
4. simultaneous SL/TP (still resolved conservatively by the existing lifecycle code).

The visible replay cursor never advances beyond the earliest event. If a +60m request hits TP after 8 minutes, the response stops at that TP cursor and the remaining 52 minutes are not returned to the browser. If no lifecycle event occurs, the cursor jumps directly to the requested target.

The vectorized scan is internal simulator work only; it does not expose future OHLC to chart/decision code.
