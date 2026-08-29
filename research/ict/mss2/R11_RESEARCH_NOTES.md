# R11.1 — Continuous Visible Liquidity Path Atlas

## Correction from R11
The original R11 incorrectly used UTC+8 00:00 as a day-open liquidity freeze. That framing is inappropriate for ETH because ETH trades continuously 24/7.

R11.1 removes **all session/open dependence** from the research logic.

## Frozen design
- No promoted trading strategy.
- Broad liquidity universe = every causally confirmed 15m/30m/1H/4H classical ITH/ITL/LTH/LTL.
- STH/STL are construction-only.
- Liquidity activates exactly when causally available and remains live until consumed.
- Intraday newly confirmed liquidity is immediately eligible after `available_time`.
- No 00:00 reset, snapshot, target freeze or path termination.
- Calendar day may be used in summaries only.
- Paths can start before midnight and hit opposite liquidity after midnight.
- Completed-trend/native/nested/3%-5%-7% fields are labels only; no admission filtering.

## Primary questions
1. At each continuous root sweep, what opposite IT/LT liquidity was actually visible at that exact moment?
2. After SSL is taken, how often/how quickly does price reach the frozen BSL target over 24h/48h, and vice versa?
3. How does the continuous sweep sequence evolve (`SSL -> BSL`, repeated same-side sweeps, alternating paths, same-bar two-sided)?
4. Which causal landmarks (reclaim, 1m/2m/5m post-sweep MSS, directional FVG) precede successful opposite-side delivery?
5. Which path classes later justify entry research without pre-filtering the market first?

## Important anti-leakage rules
- A liquidity level cannot be consumed before its `it_available_time`.
- If IT status becomes available at 01:00, the 00:59 bar cannot retroactively consume it.
- Opposite target is frozen from liquidity that is active at the **root sweep time**, not from liquidity formed later.
- Newly formed liquidity after the root may participate in later root paths, but cannot become a retrospective target for the earlier path.

## Manual review
- `manual_review/01_recent_30_continuous_paths.csv`
- `manual_review/02_recent_30_active_liquidity_at_sweep.csv`
- `manual_review/03_recent_continuous_sweep_sequence.csv`

Manual review should compare the exact active map at each sweep time, not an arbitrary calendar-day open.
