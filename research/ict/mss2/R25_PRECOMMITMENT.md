# R25 Precommitment — r0020 Directional-Run Exhaustion Reversal

Date frozen: 2026-08-17, before R25 outcome calculation.

## Prior-work boundary

R25 does not repeat the rejected rule “high Range-Bar activity implies
continuation.” Momentum R10 found that high activity increased runner labels,
and R11 showed that it was two-sided volatility expansion with worse
directional first passage. Earlier liquidity and swing-low studies also used
single-bar direction, trailing direction balance, duration, duration change,
flow, footprint, and nonlinear models.

The untested mechanism is narrower and event-based: a completed directional
run is first allowed to exhaust, an opposite r0020 bar must then complete, and
only that reversal confirmation can create an entry. No activity bucket,
model, sweep filter, flow filter, or feature rescue is permitted.

## Economic hypothesis

Four consecutive 0.20% Range Bars represent roughly a 0.8% one-sided auction.
If the first opposite Range Bar then completes before price has returned to the
run origin, the terminal auction may have exhausted. The asymmetric path is a
reversion to the known run origin before a new sequence extreme.

## Frozen data and sequence

- Market: OKX `ETH-USDT-SWAP`.
- Event source: local `OKXRangeBarLoader`, fixed `r0020` only, read through
  `src.data_feed`; missing data are never built or downloaded by R25.
- Execution/path source: bare OKX 1-minute bars through `src.data_feed`.
- Range Bars become observable only at completed `end_ts`.
- Sort key is `(end_ts, bar_id)`. Equal-end-time bars keep their deterministic
  raw sequence order and are simultaneously observable at that timestamp.
- Valid source rows require finite positive prices, direction in `{-1,+1}`,
  `start_ts <= end_ts`, and a unique `bar_id`. Invalid rows reset the run.
- Zero-duration bars are retained: multiple ordered trades may share a
  millisecond. A qualifying run must nevertheless span positive elapsed time.
- A run is a maximal consecutive sequence of at least four bars with the same
  direction. There is no neighboring run-length sensitivity.
- The first opposite completed r0020 bar is the confirmation and signal bar.
- Down run then up confirmation creates Long; up run then down confirmation
  creates Short.
- Run duration ratios, formation speed, flow, and run length beyond the fixed
  minimum are descriptive fields only and cannot filter the trade.

## Frozen entry, stop, and target

- Signal time is the confirmation bar `end_ts`.
- Entry is the first source-observed 1-minute open whose timestamp is strictly
  later than signal time. The entry minute itself is part of the path.
- Primary execution has no additional delay. A single one-minute additional
  delay is reported as execution stress using the same frozen stop and target;
  it cannot rescue the primary rule.
- Target is the first run bar's open: the known origin of the exhausted move.
- Stop is the adverse sequence extreme across the run and confirmation bar.
- The setup is skipped before simulation if the run origin was already touched
  by the confirmation bar, the next-minute entry is not strictly between stop
  and target, the run has non-positive elapsed span, or required source data
  are invalid.
- Exact 1-minute first passage is stop-first on same-minute ambiguity. A stop
  gap fills at the worse minute open; a target fills at the frozen target.
- There is no time exit, trailing rule, runner, add-on, or partial exit. An
  unresolved position is censored at its research-split boundary.
- Long and Short are simulated separately with at most one open position per
  direction; same-direction signals while open are ignored.

## Time, cost, and gate

- Warmup/source audit: 2022 onward.
- Discovery: 2023-01-01 through 2024-12-31, reset simulation.
- Validation: 2025-01-01 through 2025-06-30, reset simulation.
- `load_local_data` uses overlap semantics, so R25 explicitly retains only
  rows with `end_ts < split_end`; no confirmation may cross a split boundary.
- July 2025 is embargoed. Holdout begins 2025-08-01 and remains unloaded.
- Round-trip costs: 0.11%/0.22%/0.33% at 1x/2x/3x.

A direction is only a research candidate if discovery and validation both
have 2x-cost PF >=1.4 and positive expectancy; discovery/validation have at
least 100/20 closed trades; every visible year has positive 2x-cost sum;
discovery remains positive after removing the top ten winners; at least 80% of
visible split months are positive; the one-minute-delay stress has positive
2x-cost expectancy in both splits; and median realized reward/risk is positive.
Frequency and flat periods are reported but are portfolio-construction gaps,
not reasons to dilute a profitable sleeve. Passing is not live approval and
cannot open the holdout without a later frozen portfolio decision.
