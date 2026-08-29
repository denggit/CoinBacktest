# RL Market Agent V1 — R00.1 Cumulative Patch Manifest

Baseline inspected: `3c8dad01495c787cf95e1908f7c03587ff72952a` from the uploaded CoinBacktest archive.

## Scope

New clean-sheet R00 only. No old `eth_ai_trading` strategy/model code is modified or reused. No backtest strategy, AetherEdge code, `src.data_feed` implementation, exchange adapter, or existing research logic is changed.

## Added architecture

- `src/ai_research/rl_market_agent/`: reusable causal contracts, features, labels, data-source facade, shard storage, sealed dataset reader and R00 pipeline.
- `research/eth_ai_trading/rl_market_agent_v1/`: R00 entrypoint plus cumulative research records.
- `tests/ai_research/rl_market_agent/`: causal/label/feature/storage/sealed-holdout/end-to-end synthetic tests.

## R00.1 compatibility hotfix

- Fixed `build_range_event_features` for the public `OKXRangeBarLoader` contract where `end_ts` is intentionally both the index name and a retained column (`set_index("end_ts", drop=False)`).
- The feature layer now drops only the redundant incoming index labels before column-based event sorting; the observable `end_ts` column is preserved and becomes the causal event availability index again after deterministic de-duplication.
- Added regression coverage for both range bars and a defensive footprint timestamp/index collision case.
- No `src.data_feed` loader behavior, feature semantics, timestamps, labels, policy logic, or research periods were changed.

## Validation performed

- `PYTHONPATH=. pytest tests/ai_research/rl_market_agent -q` -> 11 passed.
- `PYTHONPATH=. pytest tests/data_feed -q` -> 13 passed.
- new files compile with `python -m py_compile`.
- new source/research directories contain no `research -> research` imports, direct SQLite/HTTP market-data access, or `pd.Timestamp.utcnow()`.
- repository-wide import-boundary checker is not green due to pre-existing research-to-research imports in ICT/liquidity/swing-low files; none originate from this patch.
- repository-wide `pytest -q` is not collectible in the uploaded baseline due to five pre-existing missing liquidity/panic research modules.

## Local full run

`python research\\eth_ai_trading\\rl_market_agent_v1\\00_causal_state_dataset_and_environment_audit.py`

Do not start R01 until R00 returns `PASS_R00` and its coverage/causal audit is reviewed.

### R00.2 cumulative hotfix
- `src/ai_research/rl_market_agent/features.py`: complete-bucket trade-bar -> fixed OHLCV fallback and merge helper.
- `src/ai_research/rl_market_agent/pipeline.py`: official-Kline-first fallback path, 60-day fixed context, coverage fallback notes.
- `tests/ai_research/rl_market_agent/test_features.py`: complete-bucket and UTC+8 daily-anchor tests.
- `tests/ai_research/rl_market_agent/test_pipeline.py`: partial official K-line fallback integration test.
- `R00_2_HOTFIX.md`: root cause and rerun instructions.


## R00.3 cumulative patch

- `src/ai_research/rl_market_agent/config.py`: research data end -> 2026-08-15; add label-safe `decision_end`; freeze dataset revision `R00.3` and new `r00_3` cache root.
- `src/ai_research/rl_market_agent/features.py`: generic complete-bucket 1m OHLCV -> HTF causal resampler.
- `src/ai_research/rl_market_agent/pipeline.py`: official 1m K-line single-base context, independent trade-bar path, label-safe final shard.
- `research/.../00_causal_state_dataset_and_environment_audit.py`: print raw-data and decision windows separately.
- `tests/.../test_pipeline.py`: enforce 1m-only K-line source and complete final labels.
- `R00_3_DATA_REFRESH.md`: current data/causality contract.

## R00.4 cumulative hotfix

- `features.py`: replace Pandas daily `offset=` resampling with explicit shift-resample-unshift for cross-version +08:00 anchoring.
- `config.py`: bump dataset revision/cache root to `R00.4` / `r00_4` so potentially mis-anchored R00.3 shards are never mixed in.
- `test_features.py`: add warning-free and boundary regression coverage.
- `R00_4_PANDAS_DAILY_ANCHOR_HOTFIX.md`: document the real-data trigger and fix.


## R01 cumulative patch

- `src/ai_research/rl_market_agent/splits.py`: horizon-aware purged windows; sealed-boundary guard.
- `dataset.py`: block unsafe whole-shard training iterator.
- `opportunity.py`: concrete template net-return targets, fixed LightGBM baseline, feature-family ablations.
- `strategy.py`: causal 1m entry/TP/SL/max-hold replay, adverse-first same-minute ambiguity, risk sizing and continuity/risk/return metrics; MDD includes intratrade 1m MAE rather than exit-only equity.
- `r01_config.py`: frozen WF_2024/WF_2025 folds, three broad trade templates, 0.11% cost and 1% risk contract.
- `r01_pipeline.py`: pre-OOS model/calibration selection, exact OOS strategy backtest, 1x/2x/3x cost stress, 1m/2m entry-delay stress, top-1/5/10 winner-removal stress, monthly/trade/model reports, sealed-holdout audit flag.
- `research/.../01_opportunity_model_strategy_backtest.py`: Windows/Unix-compatible R01 entrypoint.
- tests: purge/seal, conservative target, feature groups, same-bar path ambiguity, portfolio metrics and end-to-end R01 smoke.

No `src.data_feed`, exchange adapter, AetherEdge code, legacy strategy logic, or R00 feature/label values are modified.

R01 validation on the uploaded baseline:
- `PYTHONPATH=. pytest tests/ai_research/rl_market_agent -q` -> 26 passed.
- `PYTHONPATH=. pytest tests/data_feed -q` -> 13 passed.
- R01 entrypoint `--help` and `py_compile` passed.
- Full `pytest -q` remains blocked at collection by the same five pre-existing liquidity/panic missing modules; no R01 module appears in those errors.
- Repository import-boundary checker remains red from pre-existing backtest/research cross-imports; no `rl_market_agent_v1` violation is reported.
