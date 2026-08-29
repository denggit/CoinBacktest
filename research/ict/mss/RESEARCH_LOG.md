# ICT MSS Research Log

## R01 — causal HTF liquidity sweep -> 1m MSS -> displacement FVG retest

### Goal

Test the ICT mechanism itself before building a strategy: obvious 15m/30m/1H/4H swing liquidity is swept, 1m structure shifts in the opposite direction with displacement and an FVG, then a limit order retests the FVG near edge with the stop beyond the sweep extreme.

### Engineering decisions

- Data access is exclusively `src.data_feed.okx_loader.OKXDataLoader`; no research-specific loader/API was added.
- Only `open/high/low/close` are consumed after loading.
- HTF candles are aggregated from 1m and only complete bins are used.
- HTF swing availability includes both pivot confirmation and candle close latency.
- 1m micro swings are forward-filled only after right-side confirmation closes.
- Same-bar liquidity sweep + FVG completion is rejected because bare OHLC cannot prove intrabar ordering.
- FVG completion close is the signal availability point; limit order activates on the next bar.
- Stops use the most extreme price from sweep through FVG completion, not merely the first swept candle.
- Segment-tree first-touch searches and vectorized pivot/FVG construction are used to avoid bar-by-bar quadratic scans.
- Same-bar TP/SL ambiguity is resolved stop-first.

### R01 predeclared research variants

The atlas is intentionally small and fixed before outcomes are viewed. It checks:

- weak/core/strong displacement magnitude;
- a three-point core displacement neighborhood;
- long-only vs short-only symmetry;
- 1m micro swing order 2/3/5;
- pre-sweep vs rolling post-sweep confirmed structure;
- >=1H and 4H-only liquidity;
- stronger HTF pivot confirmation;
- multi-timeframe same-bar liquidity confluence;
- fast MSS, fast FVG fill, and wider timing windows.

This is mechanism sensitivity, not a parameter optimizer.

### Edge promotion discipline

- 2023-2024: development evidence.
- 2025: validation.
- Candidate freeze / 2x cost stress / top-10-winner removal: 2023-2025 only.
- 2026H1: sealed final confirmation only.
- Core displacement candidates also require a neighboring displacement definition to pass.
- Failure means `research_continue`, not loss-by-loss parameter tuning.

### Validation completed in development environment

- New ICT MSS unit tests: 9 passed.
- New tests + reused swing-liquidity atlas tests: 20 passed.
- Synthetic SQLite end-to-end smoke: passed through data loader, research pipeline, report writer and GPT review-pack creation.
- Full repository pytest cannot currently complete because the provided repository snapshot is missing several unrelated legacy liquidity/panic research modules and entrypoint files referenced by existing tests. The first full collection reports five missing legacy modules; after ignoring those, 455 tests pass before another missing legacy liquidity-wall entrypoint is encountered. These failures predate and are outside R01.

### Actual ETH historical result

Not executed in the supplied development archive because it does not contain `data/crypto_history.db` or another 2022-2026 1m candle database. No claim of edge/no-edge is made from synthetic data.

Run R01 against the local CoinBacktest database and preserve/upload the generated `gpt_review_pack.zip`. If the frozen gate finds no edge, the next version must change the market hypothesis/mechanism rather than tune R01 against individual losses.

---

## R01 actual ETH result — frozen 2026-08-15

User-supplied `01_ict_mss_edge_discovery` report was reviewed after the initial implementation.

### Actual counts

- 1m bare OHLC: 2022-01-01 -> 2026-06-30
- HTF levels: 134,960
- first-sweep episodes: 57,419
- raw directional FVGs: 686,116
- causal sweep -> MSS/FVG pairs: 259,959
- causal audit violations: 0
- edge-gate passes: 0

### Frozen conclusion

The unconditional rule "confirmed HTF swing gets swept, then causal 1m MSS + displacement + FVG retest" has **no tradeable edge** in R01.

Representative results were around -0.25R to -0.35R per trade after the default 0.11% round-trip cost, with PF well below 1.0. Doubling costs generally reduced mean R by roughly another ~0.3R, implying that gross/pre-cost expectancy was mostly around flat rather than hiding a strong edge behind fees.

Relative clues only:

- 4H-only events lost less on mean R than the broad baseline but remained negative and weakened in 2025/2026H1.
- short core looked temporarily better in 2025 but failed again in 2026H1; do not infer a stable short edge.
- micro order 5 was materially less negative in 2025/2026H1 but still failed the edge gate.
- `fast MSS <=15m` was worse, so immediacy after sweep is not a quality proxy by itself.

This result changes the hypothesis: R02 must study **which swings deserve to be called liquidity**, not further tune the R01 MSS thresholds.

## R02 — causal liquidity taxonomy / old-remote levels / sessions / 1m-vs-2m

### Hypothesis

A swing should not be considered meaningful liquidity merely because it is a local pivot. Meaningful liquidity may depend on scale, time survived, how far price moved away, whether multiple visible levels stack near the same price, and when the sweep occurs.

### Causal liquidity features

All R02 liquidity-quality features stop at sweep-bar open:

- HTF timeframe and order actually confirmed by the event time;
- prominence for that already-confirmed order;
- pivot rejection and range;
- age since initial confirmation;
- activation distance;
- maximum excursion away from the level through `sweep_pos - 1`;
- active, still-unswept equal-level clusters within 5/10/25bp;
- number of HTFs represented in the 10bp cluster.

`future_max_eventual_order_label` may remain in the raw level table for historical diagnostics but is explicitly not used to construct liquidity quality.

### Remote swing rule

R02 does not retain only the latest/nearest swing. Every causally confirmed level remains active until its first true sweep. "Remote" is studied with both age and prior excursion, so an old level that stayed next to price is not automatically treated the same as a level from which price traveled far away and later returned.

### 1m vs 2m

2m candles are built only from complete 1m bars from the existing loader. HTF liquidity is shared; MSS/FVG is native to each execution frame. For 2m, a sweep/FVG is not visible until the complete 2m candle closes. Search, fill expiry and outcome horizons are converted from minutes to bar counts separately for 1m/2m.

### Calendar/session atlas

R02 records weekday/weekend, weekday name, Asia/London/New-York sessions, ICT London/NY kill zones, US 09:30-11:00 cash-open window and NY two-hour clock buckets. New York/London use DST-aware timezone conversion.

### Candidate protocol

Exploratory atlases are separated from the strict edge gate. Fixed candidate rules combine 1m/2m, micro order 2/5, predeclared liquidity classes, and fixed session windows. A strict pass must be positive in 2023 and 2024 development evidence, 2025 validation, 2026H1 forward evidence, 2x-cost 2023-2025, top-10-winner removal, and full-sample robustness.

### Engineering validation

- R01 + R02 ICT tests: 12/12 passed.
- ICT tests + reused swing-liquidity atlas tests: 34/34 passed.
- R02 synthetic SQLite end-to-end smoke: passed for both 1m and 2m pipelines through report/review-pack generation.

Next required action: run R02 on the real 2022-2026 local database and upload its `gpt_review_pack.zip`. Only then decide whether a liquidity subtype/session has an edge or whether R03 needs a new mechanism.
