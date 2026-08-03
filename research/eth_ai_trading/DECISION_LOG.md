# ETH AI Trading — Decision Log

## D01 — Market state removed from trading decisions

Decision: abandon strategic/tactical/entry/activity state as an opening, sizing or exit layer.

Reason: no stable 2024/2025 trading uplift; delayed and unstable directional semantics.

## D02 — Base long high-tail opening model retained

Decision: continue with the frozen R03.4.1 six-hour long utility model.

Reason: q90 events had strong cross-year cost-adjusted expectancy despite weak full-sample Rank IC.

## D03 — Fixed 6h is diagnostic only

Decision: retain fixed 6h only as a frozen benchmark.

Reason: it demonstrates opening Edge, but is not acceptable as the final live exit or black-swan control.

## D04 — Small structural stops and early trailing exits rejected

Decision: stop using local 60m/180m structure stops, small-R targets and early MFE trailing as the main exit.

Reason: they cut delayed-recovery and slow-grind winners, destroying profit thickness.

## D05 — q70 retained

Decision: promote q70 to the main research event pool, with q90 as a quality tier.

Reason: q70 raised trade count and total diagnostic profit across 2024/2025 while keeping PF and win rate broadly stable.

## D06 — Score tiers retained for future risk allocation

Decision: preserve q70-q80, q80-q90 and q90+ separately.

Reason: expectancy generally rises with score, but lower layers remain positive and improve opportunity count.

## D07 — Current ML failure Overlay rejected

Decision: do not use the R03.4.2.5 probability threshold as an executable stop.

Reason: probability calibration and trigger rate drifted across years; recoverable drawdowns were falsely exited.

## D08 — Score upgrade alone cannot trigger adding

Decision: do not pyramid solely because an open position moves from q70 to q90.

Reason: score may rise because price falls and the model sees more future room; upgrade groups often performed worse.

## D09 — Move to incremental holding value

Decision: model ‘exit now versus continue’ utility rather than abstract long-hold labels or failure probabilities.

Reason: this aligns the model target directly with the economic decision and supports a future recurrent non-time exit.

## D10 — Incremental holding ML stopped

Decision: do not continue adding more complex holding-value machine-learning layers after R03.4.2.6.

Reason: local ranking existed, but no model passed both 2024 and 2025; annual winners and feature relationships changed materially.

## D11 — Year-specific live model switching forbidden

Decision: historical annual winners may be reported but cannot be selected as separate live regimes without a causal, pre-observable gating mechanism that itself passes OOS.

Reason: choosing a 2024 model for one environment and a 2025 model for another after observing outcomes is overfitting, not a live strategy.

## D12 — Move to one causal non-time structural state machine

Decision: R03.4.2.7 uses the same pre-registered break/reclaim, lower-high/lower-low, profit-protection and disaster rules in both years.

Reason: the exit must be explainable, executable on every new bar and independent of scheduled holding duration.

## D13 — Research-boundary marks are censoring, not exits

Decision: positions open at OOS end or a data gap are marked to market and explicitly censored.

Reason: using the end of a 5-day or annual research window as an exit would silently reintroduce a time-based strategy rule.

## D14 — Failed-Reclaim retained as a working structural baseline, not declared final

Decision: keep deterministic `failed_reclaim` plus the wide disaster layer as the working non-time baseline.

Reason: it thickened average trade profit and lets trends run without a scheduled time exit, but no R03.4.2.7 policy passed every cross-year gate. Its long occupancy also skipped roughly 42%–45% of q70 events.

## D15 — R03.4.2.8A strict subset failure must not collapse trade frequency

Decision: do not reduce the complete strategy to the 47/28 strict healthy/recovered signals.

Reason: that gate tested whether a tiny, highly protected add subset was independently robust. It did not test a pre-allocated two-slot account, and deleting most opportunities would violate the project frequency and total-profit objective.

## D16 — Test capacity with pre-allocated risk slots, not unlimited adding

Decision: R03.4.2.8B compares P0 1R, P1 0.5R+0.5R, P2 0.65R+0.35R and P3 protected 0.65R+0.35R.

Reason: the unresolved problem is whether single-position occupancy—not absence of q70 Edge—caused the 2025 profit loss. Maximum-two-slot risk accounting can test that without a third tranche, score-driven averaging down or total planned slot risk above 1R.

## D17 — Opportunity coverage is a hard research guardrail

Decision: every future entry/add filter must report retained q70 coverage, annual count and monthly frequency.

Reason: higher PF obtained by deleting most trades is not acceptable. The current q70 opening pool produces roughly 35 independent signals/month; the research target is to preserve enough of that pool while improving executable account profit.

## D18 — Static dual-slot risk reservation rejected

Decision: do not freeze P1/P2/P3 from R03.4.2.8B as the live account policy.

Reason: they restored frequency and reduced MDD, but permanently reduced every primary before a second signal existed. The resulting 21%–23% annual return loss was too large relative to an already controlled P0 MDD.

## D19 — Full primary risk may be transferred only after enforceable protection

Decision: every standalone q70 primary starts at 1R. A second Tranche can use only risk already removed by a monotone live hard stop.

Reason: this preserves the proven primary Edge when no second signal appears and distinguishes real risk transfer from static risk dilution or score-driven averaging down.

## D20 — Structure protection must preserve return before it can release risk

Decision: compare disaster-only, latest-confirmed and one-level-lagged protection in a protection-only gate before any dynamic adding.

Reason: a tighter stop can manufacture apparent capacity by prematurely exiting winners. Risk release is valid only if the enforceable stop itself survives both years, costs, delay and return-retention requirements.


## D21 — Direct 15m Pivot hard stops abandoned for the q70 Sleeve

Decision: do not continue latest/lagged Pivot layers or buffer grids as exchange-style hard stops.

Reason: S1 stopped about 99.8% of events and S2 about 86%; S2 was strong in 2025 but destroyed too much 2024 return. The structure should remain a soft break/reclaim state, while 3% remains the disaster floor.

## D22 — Real exposure reduction may fund later q70 risk

Decision: test partial closes and same-open risk migration rather than claiming risk release from floating profit or a tightened Pivot stop.

Reason: physically closing old units creates enforceable capacity without permanently diluting every primary. A later q70 tranche may use only the fixed one-R cycle budget freed by partial closes or an immediate old-position reduction. Losing/BROKEN roots cannot fund migration, and no cycle may hold more than two Tranches.
