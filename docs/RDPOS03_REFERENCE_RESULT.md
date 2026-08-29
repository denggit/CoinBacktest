# RDPOS-03 Reference Result on the supplied RDPOS-01 report

Input window: 2023-01-01 through 2026-06-16 (2022 remains history/warmup only).

Frozen base:

- total return: +123.69%
- CAGR: 26.24%
- MDD: -35.27%
- Calmar: 0.744
- mean gross exposure: 0.539x

Mature-expansion no-penalty counterfactual:

- total return: +137.13%
- CAGR: 28.39%
- MDD: -37.28%
- Calmar: 0.761
- mean gross exposure: 0.555x

Exposure-matched static base control:

- CAGR: 26.34%
- MDD: -36.74%
- Calmar: 0.717
- mean gross exposure: 0.555x

Exploratory mature-expansion reward:

- total return: +141.37%
- CAGR: 29.05%
- MDD: -38.31%
- Calmar: 0.758

Central mature-expansion future aligned market return is positive at 24h and 72h in every official year (2023/2024/2025/2026). Across the fixed 3x3x3x3 neighborhood, 79/81 combinations are positive at 24h in every year and 81/81 are positive at 72h in every year.

Interpretation:

1. Persistent mature trend expansion appears to contain real directional information.
2. Removing the old extension penalty improves Calmar versus an exposure-matched static-risk control, so the effect is not explained only by higher average gross exposure.
3. The improvement is still insufficient: CAGR remains below |MDD| and 2023 becomes slightly worse. Increasing leverage further is therefore the wrong next step.
4. The next research target is false-expansion / deterioration detection: distinguish the profitable mature expansions from the 2023 drawdown-period expansions using only causal path history.
5. This is post-discovery on the same historical sample and is not an independent live-validation result.
