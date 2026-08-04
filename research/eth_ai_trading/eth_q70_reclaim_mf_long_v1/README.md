# ETH AI Trading Research

## Read first in every new window

1. `RESEARCH_HANDOFF.md` — authoritative stage history and frozen mainline.
2. `COMPLETED_WORK.md` — durable positive and negative findings.
3. `OPEN_ITEMS_AND_ROADMAP.md` — the only allowed next step.
4. `DECISION_LOG.md` — decisions that must not be silently reversed.
5. `STAGE_DELIVERY.md` — latest code, report and test status.
6. Latest empirical `data/reports/research/eth_ai_trading/<stage>/99_decision.md`.

## Current status

```text
R03.4.2.16: FAIL_2026_SEALED_HOLDOUT
R03.4.2.16.1: JULY_FORWARD_SUPPORTS_FROZEN_C2 (diagnosis only)
R03.4.2.17: DIAGNOSIS_SCORE_DRIFT_DOMINANT
R03.4.2.18: ETH Q70 Reclaim MF Long V1 archived; capital allocation 0
```

Archived frozen V1 policy:

```text
q70 ML long entry
+ next observable 1m open immediately
+ equal risk for every q70 tier
+ real 2% exchange-side hard stop
+ 1.5% completed 15m-close soft failure
+ failed_reclaim deterministic non-time exit
+ no add-on / no fixed take profit
```

Primary annual C2 account results at 1-minute delay / 2x cost:

```text
2024: 236 trades, +85.1%, MDD -9.4%, PF 1.71
2025: 244 trades, +100.7%, MDD -8.4%, PF 1.74
```

Continuous 2024-2025 OOS account:

```text
480 trades
+271.4% total return
92.8% CAGR
-9.4% MDD
PF 1.73
18/24 positive months
8/8 positive quarters
+86.1% after removing the ten largest winners
```

Stress remained profitable in all frozen cells. The weakest 3x-cost cell still returned +127.3% with approximately -12.6% MDD.

## Historical metric scopes

These are different contracts and must not be compared as one strategy changing suddenly:

```text
fixed 6h all-signal: independent-signal/full-notional diagnostic
P0 failed_reclaim: single-position/full-notional path diagnostic
C2 equal-one-R: risk-sized single-position account result
```

## Consumed R03.4.2.16 sealed holdout — historical reproduction only

```text
python research\eth_ai_trading\03_4_2_16_2026_sealed_validation.py
```

Report directory:

```text
data\reports\research\eth_ai_trading\03_4_2_16_2026_sealed_validation
```

Read first:

```text
99_decision.md
05_continuous_scenario_summary.csv
08_okx_lot_size_audit.csv
09_net_risk_reserve.csv
10_model_governance.csv
11_live_state_contract.csv
12_final_gate.csv
gpt_review_pack.zip
```

## Model deployment cadence

```text
continuous: immutable champion inference
daily: data/feature health
monthly: performance, calibration and drift audit
monthly optional: shadow candidate retraining
quarterly or event-driven: explicit release gate
never: automatic monthly champion replacement
```

Deployment price-risk budget should begin around 0.83%-0.85% equity, leaving 0.15%-0.17% for fees, slippage and jump risk so the net account tail remains near 1%.

## Research invariants

- No future information.
- Closed data creates decisions; actions execute at the next observable open.
- Higher-timeframe context uses `available_time`, not bar start time.
- Data access stays in `src.data_feed` public loaders.
- Research entrypoints remain thin and never import other research entrypoints.
- 2023 is development/training history, not independent OOS account return.
- January–June 2026 has been opened and consumed as a failed sealed holdout; no parameter repair may use it.
- No year-specific policy selection.
- Fixed six hours is diagnostic only.
- Score is an opening selector, not a holding-renewal or add-on signal.
- No martingale, averaging down, split entry, Turtle or classic pyramid.
- No parameter changes after seeing 2026.

## R03.4.2.16 result and R03.4.2.16.1 forward extension

R03.4.2.16 was opened under an unchanged SHA-256 seal and returned `FAIL_2026_SEALED_HOLDOUT` for January-June 2026. The 1m/2x C2 account produced 134 trades, +4.8% return, -15.9% MDD, PF 1.09 and 2/6 positive months. The q70 exceedance rate drifted to 58.14%; 3x-cost cells were negative. This sleeve is not approved for live trading and may not be repaired on the H1 holdout.

R03.4.2.16.1 completed with `JULY_FORWARD_SUPPORTS_FROZEN_C2`: July produced 17 trades, +8.9%, -4.3% MDD and PF 2.74 at 1m/2x, while every 2x/3x delay cell remained profitable. However fixed-6h opening expectancy remained slightly negative at 2x, q70 exceedance rose further to 70.36%, and removing the largest winners left July negative. The month supports state dependence and exit-overlay concentration; it does not reverse the H1 seal or approve live trading.

## R03.4.2.17 state attribution — completed, reproduction only

```text
python research\eth_ai_trading\03_4_2_17_state_gate_diagnostic.py
```

R03.4.2.17 is diagnostic only. It aligns completed 1D/4H states by `available_time`, attributes C2 and fixed-6h outcomes by regime, recomputes the exact frozen score distribution, and reports predeclared counterfactual gates. No gate can be promoted on 2026; any V2 requires future untouched validation.


## R03.4.2.17 hotfix3 reporting audit

- Final diagnostic: `DIAGNOSIS_SCORE_DRIFT_DOMINANT`.
- No simple causal 1D/4H Long gate explains or repairs the 2026 seal failure.
- Broad state-conditional q70 score drift is the dominant finding.
- July profit depends on the non-time exit overlay and concentrated winners.
- V1 remains not live-approved; no gate from opened 2026 data is validation.
- Hotfix3 corrects attribution wording, exact calendar monthly return sourcing and 2026 MAE fallback.


## R03.4.2.18 archive closeout

The failed branch now has a stable model identity and immutable archive:

```text
ETH Q70 Reclaim MF Long V1
research/eth_ai_trading/archived_models/eth_q70_reclaim_mf_long_v1
```

V1 is not live-approved and may not be repaired on opened 2026 data. The next independent direction is `ETH Trend Pullback Continuation Long/Short V1`: trend state may use breakout/expansion evidence, but execution research must focus on pullback, reclaim and re-acceleration with a maximum stop-distance skip rule rather than breakout chasing.
