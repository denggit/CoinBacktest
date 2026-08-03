# ETH AI Trading Research — Cumulative Handoff

Updated through: **R03.4.2.9 code delivery; R03.4.2.8B empirically complete**

## 1. Final objective

Build an ETH-USDT-SWAP strategy/portfolio that remains profitable after realistic OKX costs and delay, has controlled drawdown, is causal and executable, and can later be integrated into AetherEdge. The final strategy must maximize cost-adjusted positive expectancy and total portfolio profit. Win rate and model metrics are secondary, but opportunity coverage is now a hard guardrail: research must not obtain a cleaner PF by collapsing a ~35-signal/month q70 pool into a tiny subset.

## 2. Non-negotiable rules

- No future leakage. A feature or structure event is usable only after it is observable.
- 2026 remains sealed until the complete entry/exit/risk chain is frozen.
- Data access goes through public `src.data_feed` or existing shared AI research loaders.
- Research scripts do not import other numbered research scripts.
- Windows commands are one line, without `cd` or manual `PYTHONPATH`.
- Use `pd.Timestamp.now("UTC")`, not `pd.Timestamp.utcnow()`.
- No final live exit may depend on a fixed holding duration.
- Research checkpoints may be used for measurement, but not silently converted into time exits.
- A live mechanism must be one unified rule set. Selecting a different winner for 2024 and 2025 is forbidden.
- Entry or add filters must report signal retention and monthly frequency; excessive filtering is a failure, not an optimization.

## 3. Frozen opening model

The long opening model remains the R03.4.1 base model:

- Target: future 6h long MFE minus `1.25 ×` future 6h long MAE.
- LightGBM: 420 trees, learning rate 0.035, 31 leaves, min child samples 300.
- Decision cadence: 15 minutes.
- Entry: next 1-minute open.
- It predicts whether a **new long entry now** has value. It is not a holding model.

## 4. Frozen opening-pool conclusions

### q70 is the main research pool

R03.4.2.4 passed q70 cross-year expansion:

- 2024 q70: 431 trades, 2x-cost mean +0.281%, PF 1.64, win 62.4%, MDD -18.7%.
- 2025 q70: 419 trades, 2x-cost mean +0.517%, PF 2.22, win 66.1%, MDD -17.5%.
- q70 added roughly 43%–56% more events than q90 while preserving win rate and PF.
- q70-q90 events are independently positive, but q70-q80 is thinner and more concentrated.

Frozen score tiers:

- `q70_to_q80`: retained; future initial risk should be lower unless path quality confirms.
- `q80_to_q90`: stable expansion layer.
- `q90_plus`: core high-quality layer.

No tier is discarded. Position sizes remain unfinished because final stop distance and exit mechanics are not frozen.

## 5. Completed negative findings

### Market-state trading layer — abandoned

Strategic/tactical/entry/activity states may remain in Analyze Tool for interpretation only. They must not control direction, filtering, sizing, exits or live decisions.

### Opening-score persistence/upgrade — not a holding or adding signal

- A falling opening score after price rises is normal because remaining opportunity shrinks.
- A rising score in a losing position may simply mean price fell and the model sees more future room.
- Do not exit because the opening score falls.
- Do not renew or add because the opening score rises.

### Machine-learning failure Overlay — abandoned as an executable exit

R03.4.2.5 ranked failures but probability calibration and trigger rates drifted across years. Threshold exits misclassified recoverable drawdowns.

### Incremental-hold ML — stopped after R03.4.2.6

Decision: `RESEARCH_CONTINUE_RANKING_ONLY`.

- Some checkpoints showed local ordering information.
- No model/feature set passed both 2024 and 2025.
- Year-specific winners differed materially.
- Opening score added no stable holding value.
- Do not continue stacking more complex holding ML models.

## 6. Risk protection retained

A wide disaster protection layer around a -3% price move remains a research reference, not a frozen live stop. It must execute at the next observable market price in research, not an ideal stop fill.

Final live risk must combine:

- exchange-side hard safety stop;
- software structural exit;
- position size based on stop distance and account risk;
- realistic slippage/latency stress.

## 7. R03.4.2.7 empirical result

Decision: `FAIL_NO_ROBUST_NON_TIME_STRUCTURAL_EXIT`.

- No structural candidate passed every profit-retention, MDD and cross-year gate.
- `failed_reclaim` remained the best deterministic working baseline, not a final passed exit.
- At 1-minute delay and 2x cost it produced 236 trades / +0.586% mean / PF 1.80 / -20.7% diagnostic MDD in 2024, and 244 / +0.668% / PF 1.85 / -16.2% in 2025.
- Its main benefit is thicker winners and no scheduled holding-time exit.
- Its main cost is long occupancy: 45.2% of complete q70 events were skipped in 2024 and 41.8% in 2025.

## 8. R03.4.2.8A empirical result

Decision: `FAIL_NO_CROSS_YEAR_OCCUPIED_SIGNAL_ELIGIBILITY`.

- The strict healthy-trend/recovered-structure subset contained only 47 signals in 2024 and 28 in 2025 at the 1-minute main delay.
- Fixed-6h 2x-cost expectancy for that subset was +0.233% / +1.273%, but concentration and delay gates did not pass both years.
- Most occupied signals were classified dangerous-average-down: 61.5% in 2024 and 70.3% in 2025.
- However, dangerous classification did not mean the standalone q70 signal lacked Edge; it meant adding it to an already-risky root could duplicate the same market risk.
- The stage therefore rejected the narrow “only perfect healthy/recovered adds” route. It did not justify reducing the full strategy to 47/28 events.

## 9. R03.4.2.8B empirical result

Decision: `FAIL_NO_ROBUST_DUAL_SLOT_ACCOUNT_POLICY`.

- Static P1/P2 dual slots restored broad q70 coverage to 81.9%/87.1% and roughly 29–30 Tranches/month.
- They remained profitable, passed 3x cost and delay stress, and reduced MDD.
- But P2 reduced 1-minute/2x account return from P0 56.2% to 43.1% in 2024 and from 68.6% to 54.3% in 2025.
- The main loss was not a negative second-Tranche Edge. It was permanent dilution of every primary from 1R to 0.65R even when no second signal appeared.
- P3 removed dangerous second entries and lowered MDD further, but coverage fell to 66.6%/69.0% and returns remained below P0.
- Static risk reservation is therefore abandoned. The useful retained finding is that later q70 signals can add coverage and profit, but only after real primary risk has already been removed.

## 10. R03.4.2.9 empirical result

Decision: `FAIL_NO_ROBUST_STRUCTURE_PROTECTION`.

- S1 latest-confirmed Pivot protection triggered on roughly 99.8% of events and collapsed 2024 account return from 56.2% to 7.3%.
- S2 one-level-lagged protection still became the main exit on roughly 86% of events.
- S2 improved 2025 return to 88.7% and lowered MDD, but reduced 2024 return to 28.2% and failed top-ten/cost robustness.
- The failure is specific to directly executable 15m Pivot hard stops. It does not invalidate `failed_reclaim` as a soft break/reclaim state machine.
- Dynamic stop-funded Tranche policies were not tested because no hard protection rule passed the prerequisite gate.
- Do not continue a Pivot-layer or buffer grid. Retain the 3% disaster floor plus Failed-Reclaim.

## 11. Active stage: R03.4.2.10

R03.4.2.10 tests two real account actions without a Pivot hard stop:

1. **Soft-structure partial de-risking**
   - R1: reduce 25% once when an already-proven trend first enters soft `BROKEN` and the position is not losing.
   - R2: the same rule with a 50% partial close.
   - The remaining tranche keeps its frozen Failed-Reclaim exit.

2. **q70 risk migration**
   - M1/M2: on a later q70 signal, transfer at most 0.35R/0.50R from a healthy, non-losing old tranche to the new tranche.
   - H1: allow a prior 25% soft-break reduction, then reuse that physically released capacity for a later q70 signal, capped at 0.35R.
   - If free cycle capacity is insufficient, the old tranche must be reduced at the same execution open before the new tranche is opened.

Hard boundaries:

- each flat-to-position cycle begins with a full 1R primary;
- the one-R cycle budget is fixed in dollars when the primary opens, so floating profit cannot create extra risk capacity;
- maximum two simultaneous virtual tranches;
- losing, `BROKEN` or pending-Failed-Reclaim roots cannot migrate risk;
- no new signal resets the old tranche state or widens its 3% disaster floor;
- fixed six hours remains diagnostic only and 2026 remains sealed;
- every policy must report q70 coverage and monthly frequency, not only PF.

## 11.1 Planned order after R03.4.2.10

1. Run R03.4.2.10 and inspect whether partial reduction or risk migration preserves at least 95% of P0 return in both years.
2. Retain a migration policy only if combined cross-year return is at least P0, coverage is at least 70%, frequency is about 25+ tranches/month and risk remains at or below one cycle R.
3. If all policies fail, stop this capacity branch rather than adding a third tranche or allowing losing-position rotation.
4. Optimize entry timing and MAE without deleting most q70 signals.
5. Freeze q70/q80/q90 risk allocation, re-audit the final non-time exit chain, then open 2026.

## 12. Never repeat these mistakes

- Do not choose different live models or slot policies by historical year.
- Do not use opening-score persistence as holding confirmation.
- Do not add solely because score rises while price falls.
- Do not optimize many slot ratios or failed-reclaim parameters on 2024/2025.
- Do not treat fixed 6h as the final exit.
- Do not claim censored year-end marks are strategy exits.
- Do not celebrate PF improvement obtained by deleting most q70 opportunities.
- Do not force this single medium-horizon long Sleeve to supply the entire future portfolio frequency.
