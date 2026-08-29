# R00 Delivery

## Full local run (Windows, one line)

```text
python research\eth_ai_trading\rl_market_agent_v1\00_causal_state_dataset_and_environment_audit.py
```

The default run reads only existing local caches, writes monthly resumable shards to `data/cache/eth_ai_trading/rl_market_agent_v1/r00`, and writes the report to `data/reports/research/eth_ai_trading/rl_market_agent_v1/r00_causal_state_dataset`.

Use `--overwrite` only when the frozen R00 schema changes. `--max-shards 1 --no-review-pack` is for a development smoke test, not a research result.

## Acceptance gate before R01

Review `99_decision.md`, `04_source_coverage.csv`, and `05_causal_audit.csv`.

R01 is blocked unless:

- R00 decision is PASS_R00;
- all required sources have >=99% availability at decision times;
- future-visibility violations are zero;
- shard hashes/manifests exist;
- 2026 shards remain sealed from training/tuning.


### R00.1 resume note

The first real-data run exposed a Pandas ambiguity in the range-bar loader schema (`end_ts` is both retained as a column and used as the index). R00.1 fixes this in the AI feature layer. After applying the cumulative patch, rerun the same command. Do **not** add `--overwrite`; already completed schema-compatible monthly shards may be reused.
