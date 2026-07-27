# Liquidity Map V2.3.1 — CoinGlass Light Theme

## Scope

This patch changes only Analyze Tool presentation. It does not change:

- liquidity-map NPZ data;
- causal 24-hour amount normalization;
- wall detection thresholds or lifecycle logic;
- chart timestamps or execution causality;
- backtest or live-trading behavior.

## Changes

1. Analyze Tool uses a light gray shell and warm ivory chart canvas, matching the visual environment of the supplied CoinGlass screenshots.
2. Liquidity heat cells use an ivory → salmon → rose → burgundy-purple palette sampled from those screenshots.
3. Heat-cell opacity is nearly constant. Depth is represented mainly by RGB palette position, so changing the upper saturation threshold no longer fades all cells together.
4. The color-range sliders now display the same gradient used by the heatmap.
5. Canvas grid, axes, crosshair, tooltip, controls and detail cards were adapted for the light theme.
6. Static CSS and JavaScript URLs are cache-busted.

## Color semantics

The data semantics remain unchanged:

- lower slider: amount ratio below which cells are hidden;
- upper slider: amount ratio at which cells reach the darkest color;
- cells above the upper threshold stay at the darkest color.

## Restart

```text
python analyze_tool\server.py --host 127.0.0.1 --port 8765
```

A normal reload should fetch the cache-busted assets. `Ctrl + F5` can still be used if a service worker or proxy keeps an old file.
