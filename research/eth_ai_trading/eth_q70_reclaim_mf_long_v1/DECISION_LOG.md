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

## D14 — q70 is an opening model, never a holding-renewal signal

Decision: later score changes cannot by themselves hold, exit, average down or pyramid.

Reason: score can rise merely because price falls and remaining future room increases.

## D15 — static dual-slot reservation rejected

Decision: do not permanently reduce every primary to reserve risk for a later signal.

Reason: R03.4.2.8B restored frequency but lost roughly 21%-23% annual return because many primaries were diluted without a second event.

## D16 — Pivot hard stop rejected for the complete base

Decision: do not place the complete q70 base stop directly below a 15m confirmed Pivot.

Reason: R03.4.2.9 stopped roughly 86%-100% of positions structurally and destroyed 2024 delayed-recovery winners.

## D17 — soft structure remains state, not a complete-position stop

Decision: keep break/reclaim information for `failed_reclaim`, but not as an automatic full hard stop.

Reason: ETH frequently sweeps obvious structure and reclaims.

## D18 — old-to-new risk migration rejected

Decision: do not sell part of a healthy winning base merely to fund a later q70 event.

Reason: R03.4.2.10 later tranches were profitable, but the removed old-winner profit was larger.

## D19 — use asymmetric add-on risk

Decision: R03.4.2.11 may add exposure without reducing the base only when each add-on has an independent executable stop and the full account tail remains capped.

Reason: a tight stop may be acceptable for an optional add-on even though it is unacceptable for the complete base.

## D20 — nominal exposure is an output, not a target

Decision: cap gross notional at 1.5x in R03.4.2.11; never force 1.5x by weakening stops or exceeding two account-R tail risk.

Reason: position quantity must follow account risk divided by actual stop distance. Higher notional is accepted only if it produces better cross-year account results.

## D21 — Classic split/Turtle/pyramid execution rejected for q70

Decision: do not continue split-entry, Turtle-style favorable-move add or classic two-step pyramid inside this medium-short-horizon sleeve.

Reason: R03.4.2.11 increased nominal exposure but materially reduced both-year account return, enlarged drawdown and turned roughly 40%-56% of P0 winning cycles into losses.

## D22 — F1 is a two-R attribution reference, not a passed stop policy

Decision: retain F1 only to study completed-close soft-failure behavior.

Reason: F1 sized from 1.5% while keeping a 3% disaster stop. Its approximately 0.67x notional and high return came with a two-R maximum price tail, so the headline result cannot be called one-R risk-controlled improvement.

## D23 — Real stop distance must equal sizing risk distance

Decision: every qualifying R03.4.2.12 policy sizes from the same frozen distance as its executable hard stop.

Reason: using a narrower operating threshold for quantity while retaining a wider disaster stop understates real tail risk. Adaptive distance must be causal, frozen at entry and never widened.

## D24 — C2 real one-R tail compression passed

Decision: freeze `C2_real_2p_soft1p5` as the initial stop/sizing candidate entering score-risk research.

Reason: the same real 2% executable hard stop plus 1.5% completed-close soft failure improved both 2024 and 2025 account return, raised initial nominal exposure to about 0.50x, kept MDD near 8%-9%, and remained profitable under every pre-registered cost/delay cell without a hidden two-R tail.

## D25 — Score tiers must earn sizing authority cross-year

Decision: do not assume q90 deserves more risk. R03.4.2.13 must use one fixed map in both years, execute every C2 trade, and accept tiering only if return is substantially retained while risk efficiency improves.

Reason: score is already a valid opening selector, but relative tier expectancy may drift. Post-hoc high-score leverage would be overfitting. Equal 0.75R/1R/1.25R runs are account-scaling diagnostics and must not be presented as score Edge.

## D26 — Equal q70 risk is frozen

Decision: R03.4.2.13 keeps every q70 C2 trade at equal one-R.

Reason: q70-q80, q80-q90 and q90+ expectancy ordering changed materially between 2024 and 2025; score-based leverage would be post-hoc overfitting.

## D27 — Entry timing must preserve the signal pool

Decision: R03.4.2.14 may delay entry by at most 60 minutes, but every formal candidate must retain at least 90% of frozen C2 cycles and keep the exact stop/exit chain.

Reason: MAE improvement is valuable only if it does not recreate the old failure mode of shrinking the strategy to a tiny perfect-looking subset.

## D28 — Historical metric scopes must be shown side by side

Decision: every later patch/report must carry a fixed-6h all-signal, P0 single-position and C2 account metric comparison.

Reason: their trade counts, exits, win rates and returns answer different questions and must not be described as one strategy changing suddenly.

## D29 — Immediate q70 entry is frozen

Decision: retain the next observable 1m-open entry after q70; do not wait for a fixed score confirmation window or pullback/reclaim replacement entry.

Reason: R03.4.2.14 showed positive short-horizon drift after q70. Waiting 30-60 minutes worsened entry price, return, win rate and drawdown; deep-MAE recovered winners were too rare to justify a delayed-entry policy.

## D30 — Final continuous C2 account passed

Decision: freeze C2 for one-time 2026 sealed validation.

Reason: continuously compounding WF_2024 and WF_2025 produced +271.4% with -9.4% MDD, PF 1.73, 18/24 positive months and 8/8 positive quarters. Every frozen 1/3/5-minute and 2x/3x-cost scenario remained profitable, including after removing the ten largest winners.

## D31 — Monthly audit is not monthly automatic replacement

Decision: monitor performance and drift monthly; optionally train a shadow candidate monthly; promote only through a quarterly or event-driven explicit release gate.

Reason: automatic calendar replacement creates silent model/version drift and can promote a temporarily lucky candidate. Live execution requires one immutable champion artifact, schema hash, training cutoff, calibration threshold and rollback path.

## D32 — Reserve execution costs inside one account-R

Decision: initial deployment should allocate approximately 0.83%-0.85% equity to price-stop risk and reserve 0.15%-0.17% for fees, slippage and jump risk.

Reason: a raw 1% price-risk budget produced historical net losses up to about 1.129R in the anchor and 1.193R under 3x cost. The live tail must be measured after execution costs, not only by stop distance.

## D33 — 2026 opens only under an immutable seal

Decision: create a SHA-256 seal before any 2026 data access; exact reproduction is allowed, but changed code/config/source must be rejected.

## D34 — The first 2026 result consumes the holdout

Decision: pass, caveat or failure is permanent for this sleeve. No q70, feature, threshold, entry, stop, exit or sizing repair may use the same January-June 2026 period.

## R03.4.2.16 — `FAIL_2026_SEALED_HOLDOUT`

The frozen C2 MF Long sleeve did not retain sufficient edge in untouched January-June 2026. The failure is binding. Do not launch this generation and do not tune against the opened H1 holdout.

## R03.4.2.16.1 — July extension approved as diagnosis only

The newly available July data may be scored with the unchanged artifact to test the regime-dependence hypothesis. It is not a second qualification attempt. Any July result must be interpreted alongside, not instead of, the failed H1 seal.


## D35 — July cannot reverse the consumed H1 seal

Decision: retain `FAIL_2026_SEALED_HOLDOUT` even though July C2 recovered.

Reason: July is one new month with 17 trades; fixed-6h opening expectancy remained weak, q70 exceedance drift worsened to 70.36%, and returns depended on a few structural winners.

## D36 — Separate entry Edge from exit-overlay Edge

Decision: all post-seal attribution must report fixed-6h opening expectancy separately from C2 `failed_reclaim` account return.

Reason: a positive C2 month can be produced by a small number of long-held winners even when the broad opening pool is near breakeven.

## D37 — State gates are development evidence only

Decision: R03.4.2.17 may describe a small predeclared set of causal 1D/4H gates, but none may be called validated or deployed.

Reason: H1 and July are already opened. Any V2 gate designed with these periods requires future untouched validation and a separate version identity.


## R03.4.2.17 hotfix3 reporting audit

- Final diagnostic: `DIAGNOSIS_SCORE_DRIFT_DOMINANT`.
- No simple causal 1D/4H Long gate explains or repairs the 2026 seal failure.
- Broad state-conditional q70 score drift is the dominant finding.
- July profit depends on the non-time exit overlay and concentrated winners.
- V1 remains not live-approved; no gate from opened 2026 data is validation.
- Hotfix3 corrects attribution wording, exact calendar monthly return sourcing and 2026 MAE fallback.


## D38 — Name and archive V1 instead of continuing sunk-cost optimization

Decision: archive the branch as `ETH Q70 Reclaim MF Long V1` with zero capital and no live approval.

Reason: the untouched 2026 H1 seal failed and R03.4.2.17 found dominant score drift with no simple state-gate repair. Preserving the model card and failed paths is valuable; retuning on the consumed holdout is not.

## D39 — The next independent model is not breakout-chasing execution

Decision: research trend persistence on 1D/4H, but seek entries through lower-timeframe pullback, absorption, reclaim and re-acceleration. Breakout/expansion may be state evidence only.

Reason: chasing a completed breakout usually leaves the nearest defensible structure too far away. If the actual hard stop is too wide for the account-risk cap, reduce notional or skip the trade. Exchange leverage does not solve the risk equation.
