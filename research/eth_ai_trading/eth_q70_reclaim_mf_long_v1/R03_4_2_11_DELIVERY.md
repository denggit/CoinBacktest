# R03.4.2.11 Delivery

## Delivered

A causal account-level comparison of:

- full 1R P0;
- 1.5% confirmed soft-failure sizing;
- 0.60R/0.40R staged entry;
- full-base Turtle add;
- full-base two-step pyramid.

## Key design choice

Tight independent stops are allowed only on optional add-on layers. The complete base keeps its wide disaster floor and `failed_reclaim`, so an add-on sweep cannot automatically destroy the original winner.

## Status

```text
CODE_READY
EMPIRICAL_DATA_RUN_PENDING
2026_SEALED
```

## Run

```text
python research\eth_ai_trading\03_4_2_11_staged_entry_pyramiding.py
```

## Review

Send the generated:

```text
data\reports\research\eth_ai_trading\03_4_2_11_staged_entry_pyramiding\gpt_review_pack.zip
```

for empirical review.

## Validation

```text
R03.4.2.11专项: 9 passed
Related R03.4.2.7-11 regression: 50 passed
AI Research + Data Feed: 189 passed
Entry smoke: BLOCKED_DATA as expected in the mounted no-data environment
```

Known repository-wide blockers are unchanged: five missing liquidity/analyze-tool modules and the pre-existing import-boundary violations outside this stage.
