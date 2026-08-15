# R01.2 Patch Manifest

## New source package

```text
src/ai_research/latent_liquidity_execution_audit/
    __init__.py
    config.py
    cache.py
    source.py
    statistics.py
    replay.py
    reports.py
    pipeline.py
```

## New research entry

```text
research/eth_ai_trading/eth_latent_liquidity_path_v1/01_2_stable_path_execution_audit.py
```

## New tests

```text
tests/ai_research/test_latent_liquidity_execution_audit.py
```

## Cumulative documentation carried by this patch

```text
00_RESEARCH_CHARTER.md
README.md
CURRENT_STATE.md
DECISION_LOG.md
OPEN_ITEMS_AND_ROADMAP.md
R01_DELIVERY.md
R01_PATCH_MANIFEST.md
R01_1_DELIVERY.md
R01_1_PATCH_MANIFEST.md
R01_1_FULL_HISTORY_MEMORY_HOTFIX.md
CUMULATIVE_STAGE_RESULTS.md
R01_2_DELIVERY.md
R01_2_PATCH_MANIFEST.md
```

## Frozen research rules

- no sub-15m Swing;
- all unswept 15m+ Swing levels remain supplementary inventory only;
- liquidity-release candidates are not admitted by Swing;
- target clusters are post-R01.1 diagnostic selections;
- confirmations are fixed before replay;
- next-second-or-later execution;
- 11bp default round-trip cost with 2x/3x stress;
- no live approval from R01.2.
