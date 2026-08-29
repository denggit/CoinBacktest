# Human Trader Replay Lab V1.7.1

Fixes Episode finalization semantics for resting LIMIT orders.

- Rewind may legitimately resurrect an order when its later cancel/fill belongs to an abandoned branch.
- While the Episode is active, such an order is pending and can fill/cancel normally.
- Ending the Episode now emits `LIMIT_EXPIRED` for every still-resting order before `EPISODE_SUMMARY`.
- Therefore a closed Episode never reports a live pending order.
- Legacy V1.7.0 closed Episodes are interpreted as `pending_orders=0` with unmatched intents reported as `unfilled_orders`, without deleting or rewriting the audit trail.
