# ETH AI Trading R03.4.2.8A Runbook

Run from the CoinBacktest project root on Windows:

```text
python research\eth_ai_trading\03_4_2_8a_occupied_signal_atlas.py
```

The stage uses public `src.data_feed` loaders and existing R03.2 caches. It keeps 2026 sealed and writes reports to:

```text
data\reports\research\eth_ai_trading\03_4_2_8a_occupied_signal_atlas
```

Read `99_decision.md` first, then inspect `06_occupied_signal_atlas.csv`, `12_tranche_eligibility_gate.csv`, and `04_frozen_baseline_summary.csv`.

A PASS does not mean adding is profitable. It only permits the next account-risk simulation stage.
