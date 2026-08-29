# R04 Turtle Path Atlas

## Objective

Keep the R03 source-locked Turtle System 2 rules unchanged and reconstruct every episode minute-by-minute to learn **position-management paths**, not a new entry edge.

## Frozen source rules

- 55D intraday breakout.
- 20D opposite-channel exit.
- N = 20D EMA True Range.
- Unit sizing: 1N movement = 1% live equity.
- 2N stop.
- Add every +0.5N.
- Maximum four Units.

R04 does not change any of these.

## Path outputs

For every completed Turtle episode R04 records:

- Long/short, entry/exit time, duration, exit reason.
- Maximum Units and time to Unit 2/3/4.
- MFE and MAE in units of entry N.
- Time to MFE/MAE.
- Final directional move in N.
- Giveback from peak MFE to exit.
- Causal checkpoints at 5m, 15m, 30m, 1h, 4h, 12h, 1d, 3d and 7d while the episode is still alive.
- Path class: no-follow-through, partial-proof-then-fail, pyramid-then-fail, trend-capture.

## Discovery / validation discipline

- 2023-2024: path discovery.
- 2025: validation of qualitative path patterns only.
- 2026+: sealed.

Path outcomes are retrospective labels. They must never be inserted directly into a live signal. A later R05 may convert a repeated path pattern into a simple causal position rule, frozen on discovery and then validated without retuning.
