# ETH AI R03.4.2.2 — Causal Path-Health Recognition

## Purpose

Freeze the R03.4.1 long-entry model and determine whether price structure visible at T+60, T+180 and T+360 minutes can distinguish:

- persistent failures that should eventually be cut;
- underwater trades that are likely to recover;
- profitable paths at risk of giving back gains;
- trades with meaningful additional upside after six hours.

This stage does **not** produce a final exit strategy. It produces OOS evidence needed to design one.

## Key interpretation

The entry model score estimates whether a **new** long entry has attractive six-hour utility. It is not assumed to remain a valid holding score after price moves. Score-path variables are optional features and are audited against price-structure-only models.

## Run

```text
python research\eth_ai_trading\03_4_2_2_long_tail_path_recognition.py
```

Rebuild only the R03.4 outcome cache when explicitly required:

```text
python research\eth_ai_trading\03_4_2_2_long_tail_path_recognition.py --force-rebuild-outcomes
```

## Data and time boundaries

- R03.2 cached multicycle base features for 2023–2025.
- OKX 1m Trade Bar OHLC loaded through `src.data_feed.okx_trade_bar_loader` in bounded 31-day chunks.
- Discovery path pool: independent q70 events from prior calibration quarters.
- Primary OOS audit: q90 events in 2024 and 2025.
- Broader frequency audit: q70 OOS events, never treated as passed merely because more events exist.
- 2026 remains sealed.

## Reports

`data\reports\research\eth_ai_trading\03_4_2_2_long_tail_path_recognition`

Important files:

- `05_model_metrics.csv`
- `07_checkpoint_action_diagnostics.csv`
- `08_broad_q70_safe_bucket.csv`
- `09_score_path_ablation.csv`
- `10_feature_importance.csv`
- `11_prediction_samples.csv`
- `13_stable_candidates.csv`
- `99_decision.md`
- `gpt_review_pack.zip`

## Research discipline

- State-model outputs are not loaded.
- Future recovery/failure/continuation values are labels only.
- Probability thresholds come from expanding causal OOF predictions with a 48-hour embargo.
- Healthy trades may ultimately be held 24–48 hours or longer; this stage does not impose a time exit.
- Any future exit rule must preserve positive expectation in both 2024 and 2025 after cost and delay stress.
