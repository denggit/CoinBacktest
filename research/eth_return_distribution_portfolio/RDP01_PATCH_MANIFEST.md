# RDP-01 Patch Manifest

## New mainline

`ETH Return Distribution Portfolio V1`

## Added files

- `src/ai_research/return_distribution_portfolio/__init__.py`
- `src/ai_research/return_distribution_portfolio/config.py`
- `src/ai_research/return_distribution_portfolio/dataset.py`
- `src/ai_research/return_distribution_portfolio/modeling.py`
- `src/ai_research/return_distribution_portfolio/pipeline.py`
- `research/eth_return_distribution_portfolio/00_RESEARCH_CHARTER.md`
- `research/eth_return_distribution_portfolio/01_price_flow_distribution_baseline.py`
- `research/eth_return_distribution_portfolio/CUMULATIVE_STAGE_RESULTS.md`
- `research/eth_return_distribution_portfolio/DECISION_LOG.md`
- `research/eth_return_distribution_portfolio/OPEN_ITEMS_AND_ROADMAP.md`
- `tests/ai_research/test_return_distribution_portfolio.py`

## Frozen Stage-01 contract

- Decision cadence: 5 minutes.
- Horizons: 30m / 2h / 6h / 24h / 72h.
- Directional return quantiles: q10/q25/q50/q75/q90.
- Path labels also cached: long/short MFE/MAE and future realized volatility.
- First-pass features: full-history price + rich 1m Trade Bar flow only.
- Walk-forward: 2023->2024, 2023-24->2025, 2023-25->2026.
- 2026 is chronological OOS, not a new untouched sealed holdout.
- No q70/q90 event threshold.
- No trade-by-trade strategy/account yet.

## Causality and data-quality rules

- Left-labeled 5m bars are shifted to `available_time = bar_start + 5m`.
- Incomplete five-minute groups are dropped rather than silently treated as complete bars.
- Target execution price is the exact 1m open at the decision available time.
- Future paths are reindexed to a complete 1-minute clock; missing minutes invalidate affected forward windows instead of stretching a row-count window through gaps.
- Feature selection excludes all future target columns and the calendar `year` helper column.
- Missing feature imputation is fitted from the training period only.

## Tests

`python -m pytest tests/ai_research/test_return_distribution_portfolio.py -q`

Result: 6 passed.

`tests/test_import_boundaries.py` currently reports 188 pre-existing/unrelated unexpected boundary violations in the supplied working tree; zero are from this RDP mainline.
