# R27 Precommitment — Sequential ICT Reversal Path Study

Date frozen: 2026-08-17, before any R27 state-path outcome calculation.

## Research correction and hypothesis

R13 rejected universal sweep, Boolean reclaim/MSS, and generic FVG entries. It
did not test the complete ordered reversal path. R27 therefore tests whether a
completed-trend liquidity sweep becomes economically asymmetric only after a
causal sequence of higher-quality price-action states:

```text
S0 Sweep (baseline only)
→ S1 Rejected/Reclaimed
→ S2 New Structure Established
→ S3 Meaningful MSS
→ S4 Displacement Confirmed
→ S5 Executable FVG Retracement
→ S6 Protected Reversal
```

The question is not whether every later state mechanically improves results.
It is where successful direct delivery and same-side failure first diverge in
a stable, economically usable way. Sweep-immediate entry is a measuring stick,
not the main candidate.

## Frozen source universe and physical split

- Root geometry is the visible R13 comparison universe derived causally from
  R12 completed-trend ITH/ITL/LTH/LTL sweeps. Direct opposite delivery is a
  success; deeper same-side liquidity first is a failure, including a later
  cascade-then-opposite path.
- State geometry is recomputed from bare OKX `ETH-USDT-SWAP` 1m bars loaded
  only through `src.data_feed.okx_loader`.
- Discovery roots: 2023-01-01 through 2024-12-31. The discovery process may
  physically load bars only through 2024-12-31 23:59:59.
- Validation roots: 2025-01-01 through 2025-06-30. Validation is opened once,
  after the discovery decision is frozen. It may load bars through 2025-07-31
  only so a frozen 30-day first-passage label can mature.
- July 2025 is outcome embargo only. No July root is eligible.
- Holdout begins 2025-08-01 and remains sealed. R27 must not load or calculate
  any August-2025-or-later path, feature, candidate, or outcome.
- Research scripts must not import other research scripts. Reusable logic lives
  in `src/research_common/ict_mss2/r27.py`.

## Frozen causal primitives

- A 1m bar is available at its timestamp plus one minute.
- Closed-bar states execute at the first eligible 1m open at or after the
  state's availability time. S5 alone is a resting limit at the FVG proximal
  edge and is available only after the FVG-forming bar closes.
- ATR is the simple mean true range of the latest 60 completed 1m bars strictly
  before the sweep bar. It is frozen for that root path.
- A causal pivot uses two bars on each side and becomes available only after
  the second right-hand bar closes. Equal extrema do not qualify.
- All gates below use the frozen root ATR. Quality measures remain continuous;
  gates are not converted into a parameter grid.
- State search expires 360 minutes after the sweep. S5 has a further maximum
  180-minute causal limit wait. S6 has at most 180 minutes after S5 fill.

## Frozen ordered states

### S0 — Sweep

The root sweep bar closes. S0 entry is the next eligible 1m open. This is the
unconfirmed baseline only. Record liquidity hierarchy/age/prominence/cleanliness
where available and sweep depth, range, wick, close location, velocity, and
same-bar reclaim morphology.

### S1 — Rejected/Reclaimed

Within 30 minutes, the first completed close returns strictly through the
swept liquidity level in the reversal direction. Before that close there may
be at most two consecutive closes outside the level; three is failed
acceptance and blocks S1. Record reclaim delay, close penetration in ATR,
maximum/consecutive outside closes, outside close share, maximum outside depth,
reclaim-bar body ratio, and 15/30-minute reclaim retention.

### S2 — New Structure Established

After S1, price must form a causal two-left/two-right reversal-direction impulse
pivot followed by an opposite pullback pivot. The impulse excursion from the
sweep extreme must be at least 0.75 ATR. The pullback must retain at least 38.2%
of that impulse and remain at least 0.10 ATR inside the sweep extreme. For an
SSL long path this is a post-reclaim swing high followed by a higher protected
candidate low; BSL short is the exact mirror. Record impulse size, pullback
depth/retention, extreme clearance, formation delay, and pre-S2 MAE/MFE.

### S3 — Meaningful MSS

After S2 is available, the first completed close breaks the S2 impulse pivot in
the reversal direction within the state horizon. Close-through must be at
least 0.05 ATR. Record pre-MSS MAE/MFE, break distance, body/range ratio,
directional body ATR, time-to-MSS, and path efficiency. The broken reference is
necessarily post-sweep S2 structure; an arbitrary pre-existing micro pivot
cannot qualify.

### S4 — Displacement Confirmed

The S3 break bar or one of the next two completed bars must simultaneously
produce: displacement from the S2 pullback pivot of at least 1.00 ATR,
directional body/range ratio at least 0.60, close-through the MSS level at least
0.10 ATR, and directional path efficiency from the S2 pullback at least 0.65.
The first bar satisfying all conditions is S4. Record every component and the
maximum adverse re-threat observed before confirmation.

### S5 — Executable FVG Retracement

The first reversal-direction three-bar FVG formed from the S3 break through S4
is eligible only if width is at least 0.10 ATR and its midpoint lies beyond the
broken S2 impulse level in the reversal direction. After the forming bar has
closed, place a limit at the proximal edge: upper edge for SSL-long, lower edge
for BSL-short. The first touch within 180 minutes fills unless the opposite
target or the buffered sweep invalidation has already traded. Record FVG width,
location in the displacement, fill delay/depth, and distance from fill-path
extreme back to the sweep extreme. A target touch on the fill bar cannot be
credited; same-bar target/stop is stop-first.

### S6 — Protected Reversal

After S5 fill, a new causal two-left/two-right pullback pivot must remain inside
the sweep extreme. It becomes protected only when a later completed close
breaks the S4 displacement extreme in the reversal direction. S6 entry is the
next eligible open. Record protected-pivot clearance, confirmation delay, and
a secondary tightened stop at the protected pivot plus/minus 0.10 ATR. The
primary stage comparison continues to use the common sweep-invalidating stop.

## Frozen stop, target, outcome, and economics

- Common structural stop: sweep extreme plus a 0.10 ATR buffer away from the
  reversal direction. This represents failure of the raid/reversal thesis.
- S6 also reports, separately, the protected-pivot stop plus a 0.10 ATR buffer;
  it cannot replace the common stop in stage-uplift comparisons.
- Frozen target: the R12/R13 opposite completed-trend liquidity price available
  at the root sweep. No fixed-R target may replace it after results are known.
- Invalid entry/target/stop geometry is `invalid_geometry`, not a trade.
- First passage starts at entry. Same-bar target/stop ambiguity is pessimistic
  stop-first. A resting-limit fill bar may trigger the stop but never receives
  target credit.
- Market-entry round-trip cost is 0.0011; S5 limit round-trip cost is 0.0008.
  Report gross and 1×/2×/3× cost returns, expectancy, and PF.
- Report direct-delivery label probability separately from causal entry
  TP-before-SL because state delay can change the executable outcome.
- Unresolved paths at the physical split boundary are censored. Stages reached
  only after target/stop resolution are stale-before-entry and cannot trade.
- Executable first passage is capped at the same frozen 30-day (43,200-minute)
  root-path horizon used by the R12 comparison label; unresolved paths are
  censored rather than extended to the end of the loaded file.

## Frozen discovery analysis and divergence rule

For every S0–S6, report by side, split, and year:

- eligible roots, reached samples, conditional and root reach rates;
- direct opposite-delivery probability;
- filled/invalid/stale/censored counts;
- TP-before-SL and SL-before-TP;
- MAE/MFE, structural risk and RR;
- gross and 1×/2×/3× expectancy/PF;
- top-five and top-ten winner-removal sensitivity when at least ten trades
  exist.

Discovery may inspect continuous quality distributions and successful-vs-failed
paths, but may not create extra filters, search thresholds, or rescue a side.
The earliest stable divergence is the first ordered state, assessed separately
by side, satisfying all of:

- at least 50 discovery fills overall and at least 15 in each of 2023 and 2024;
- direct-delivery probability improves by at least 10 percentage points versus
  S0 and is not lower than the immediately prior reached state;
- gross and 2×-cost expectancy are positive overall and in both discovery years;
- 1× PF is at least 1.40 and 2× PF exceeds 1.00 overall;
- 2× expectancy remains positive after removing the top five winners;
- no causal audit violation.

If no state qualifies, freeze `NO_DIVERGENCE`. Otherwise freeze the earliest
qualifying state and its exact state semantics, not a post-hoc quality subset.

## One-time validation decision

After writing a machine-readable discovery freeze, open 2025H1 exactly once.
The frozen side/state advances only if validation has at least 15 fills, positive
gross and 2× expectancy, 1× PF at least 1.20, 2× PF above 1.00, positive 2×
expectancy after removing the top five winners, and zero causal/independent
replay violations. Long and Short are independent.

Failure rejects the sequential completed-trend reversal candidate at these
frozen semantics; it must not trigger threshold tuning or feature-combination
rescue. Success permits a later, separately precommitted lifecycle study of
probe/main/FVG/protected-swing sizing. R27 itself does not optimize position
sizing, leverage, Base/Runner allocation, or portfolio construction.
