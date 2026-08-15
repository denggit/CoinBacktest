# R01.2 replay-quality and all-NaN quantile hotfix

## Why this hotfix exists

The first full R01.2 run completed only 978 of 2,842 requested replay Episodes. The exact-second replay rejected any absent physical 1-second row, while R01.1 had already defined short missing seconds as causal no-trade bars and only treated longer gaps as unsafe. This created cluster-dependent selection bias, especially for low-activity Cluster 8.

The run also emitted `RuntimeWarning: All-NaN slice encountered`. This was caused by the first point of `impact_bp_per_million`, whose one-second price difference is intentionally undefined.

## Fix

1. Replay uses the same regular-second normalization semantics as R01.1.
2. Up to five consecutive no-trade seconds are filled causally: OHLC uses the previous close and flow is zero.
3. Longer gaps remain marked unsafe and the Episode is excluded.
4. Replay loading adds a five-second edge pad.
5. Completion rates are reported for every cluster × side × period stratum.
6. Columnwise quantiles skip all-NaN columns and preserve their output as NaN without warning.

## Frozen behavior

No cluster, candidate, confirmation, entry, stop, cost, delay, horizon or label definition changes. The hotfix only aligns replay data-quality handling with the already frozen R01.1 causal data semantics.

## Required rerun

Run the same R01.2 command. The large R01.1 source scan and replay caches may be reused where compatible, but the micro-replay cache must be invalidated by the changed configuration/source fingerprint. Verify that overall and per-stratum replay completion is no longer materially biased before interpreting execution expectancy.
