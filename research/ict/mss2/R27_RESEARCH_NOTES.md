# R27 — Sequential ICT Reversal Path Study

Date completed: 2026-08-17

## Answer first

The research correction was valid: R13 had not tested a complete ordered
reversal sequence. R27 now has. The frozen S0→S6 study finds **no stable causal
divergence and promotes no strategy**.

SSL Long shows a tempting discovery-only improvement at S2/S3, but it is too
small in direct-delivery probability, too dependent on the largest five
winners, and reverses sharply in 2025H1. BSL Short is negative throughout. The
full displacement→FVG retracement→protected-swing path is so rare under the
precommitted quality semantics that it cannot support inference or execution.

The completed-trend sweep-reversal mainline may now be archived at these
semantics. Do not rescue it with threshold relaxation, R13-bin stacking,
alternate pivot widths, FVG-size grids, or loss-specific filters. The untouched
holdout remains sealed.

## Frozen sequence

The preregistered causal path is:

`S0 Sweep → S1 Rejected/Reclaimed → S2 New Structure → S3 Meaningful MSS → S4 Strong Displacement → S5 Executable FVG Retracement → S6 Protected Reversal`

Important distinctions from R13:

- S1 blocks after three consecutive outside closes (failed acceptance).
- S2 requires a causal two-left/two-right reversal impulse pivot followed by a
  retained pullback pivot inside the sweep extreme.
- S3 must break that exact post-sweep S2 impulse pivot.
- S4 uses one frozen joint displacement-quality gate.
- S5 activates only after both the qualifying FVG and S4 are known, then
  requires an actual proximal-limit retracement fill.
- S6 requires a new causal pullback pivot and a later close through the S4
  displacement extreme.

All market states enter at the next eligible 1m open. S5 is a resting limit.
The common stop is the sweep extreme plus 0.10 ATR, and the target is frozen
opposite completed-trend liquidity. Limit-fill target credit begins on the next
bar; same-bar stop/target ambiguity is stop-first. First passage is capped at
30 days.

## State results

Values below are reached roots / filled entries, direct-delivery probability,
and executable 2×-cost expectancy / PF.

### SSL Long

| State | Discovery n/fills | Direct | 2× exp / PF | Validation n/fills | Direct | 2× exp / PF |
|---|---:|---:|---:|---:|---:|---:|
| S0 | 207 / 207 | 27.05% | -0.051% / 0.91 | 105 / 105 | 18.10% | -0.345% / 0.43 |
| S1 | 143 / 100 | 30.77% | +0.057% / 1.07 | 72 / 60 | 16.67% | -0.448% / 0.49 |
| S2 | 130 / 52 | 31.54% | +0.441% / 1.44 | 63 / 32 | 17.46% | -0.473% / 0.62 |
| S3 | 121 / 42 | 32.23% | +0.610% / 1.55 | 61 / 31 | 18.03% | -0.614% / 0.56 |
| S4 | 32 / 12 | 25.00% | +0.255% / 1.25 | 22 / 11 | 18.18% | +0.054% / 1.04 |
| S5 | 2 / 1 | 0.00% | -0.688% / 0.00 | 4 / 1 | 50.00% | +11.085% / ∞ |
| S6 | 1 / 0 | 0.00% | n/a | 2 / 1 | 100.00% | +10.705% / ∞ |

S5/S6 validation numbers are single-winner artifacts, not evidence. The sample
labels in the figures make this explicit.

The separate S6 protected-pivot-stop replay is equally non-inferential: one
validation SSL fill wins (+10.70% at 2× cost) and one BSL fill stops (-0.44%).
There are no discovery S6 fills. A guarded reporting-only replay added these
columns after proving every frozen core row and both decisions unchanged.

### BSL Short

| State | Discovery n/fills | Direct | 2× exp / PF | Validation n/fills | Direct | 2× exp / PF |
|---|---:|---:|---:|---:|---:|---:|
| S0 | 306 / 306 | 15.69% | -0.223% / 0.45 | 101 / 101 | 25.74% | -0.185% / 0.58 |
| S1 | 189 / 147 | 12.17% | -0.247% / 0.55 | 68 / 51 | 22.06% | -0.105% / 0.82 |
| S2 | 169 / 83 | 11.83% | -0.336% / 0.62 | 58 / 28 | 25.86% | -0.243% / 0.76 |
| S3 | 142 / 59 | 11.97% | -0.684% / 0.34 | 52 / 21 | 25.00% | -0.157% / 0.86 |
| S4 | 36 / 17 | 8.33% | -0.905% / 0.28 | 14 / 8 | 21.43% | -0.511% / 0.53 |
| S5 | 5 / 1 | 0.00% | -1.424% / 0.00 | 4 / 1 | 25.00% | -0.527% / 0.00 |
| S6 | 4 / 0 | 0.00% | n/a | 4 / 1 | 25.00% | -0.782% / 0.00 |

## Earliest-divergence decision

No state passes the discovery gate on either side.

- SSL direct-delivery probability rises from 27.05% at S0 to 32.23% at S3,
  only +5.18 percentage points versus the required +10 points.
- SSL S2/S3 have 52/42 discovery fills. S3 misses the 50-fill gate, while every
  SSL state's 2× expectancy becomes negative after removing the top five
  winners. S2/S3 then lose in validation.
- S4 is not a rescue: only 12/11 SSL fills exist in discovery/validation and
  top-five removal is -1.25%/-1.80% expectancy.
- BSL never produces positive discovery 2× economics.

Therefore the machine-readable freeze is `NO_DIVERGENCE` for both sides. The
one-time validation correctly reports `REJECT_NO_DISCOVERY_DIVERGENCE` rather
than selecting a later lucky state.

## Causality and data seal

- Discovery price reads stop at 2024-12-31.
- Validation roots are 2025H1; price reads stop at 2025-07-31 solely for the
  frozen 30-day outcome maturity.
- No price or root at/after 2025-08-01 is loaded into R27 outputs.
- Closed states execute no earlier than availability.
- FVG orders activate only after S4 and FVG availability.
- Independent raw-bar replay ties entry opens/limit touches, outcome, exit
  time, stop-first ambiguity, and no-target-credit-on-fill-bar semantics.
- All internal and independent audit checks have zero violations.

## Frozen conclusion

R27 rejects the hypothesis that this preregistered sequential completed-trend
sweep reversal path provides a robust executable edge. Waiting through S2/S3
does improve in-sample SSL economics, but not path-label separation enough to
meet the gate, not top-winner resilience, and not validation. Waiting for the
full S4–S6 sequence destroys sample size before it establishes reliable edge.

No probe/main/FVG/protected-swing position lifecycle should be built from R27:
there is no validated state at which to allocate the probe or main position.
The master portfolio goal remains open because no strategy sleeve is promoted.

## Primary artifacts

- `R27_PRECOMMITMENT.md`
- `data/reports/research/ict/mss2/r27_sequential_ict_reversal_path/00_manifest.json`
- `16_full_state_progression.csv`
- `17_full_quality_divergence.csv`
- `17b_protected_stop_diagnostic.csv`
- `18_full_causal_audit.csv`
- `19_holdout_seal.csv`
- `figures/`
- `manual_review/`
- `gpt_review_pack.zip`
