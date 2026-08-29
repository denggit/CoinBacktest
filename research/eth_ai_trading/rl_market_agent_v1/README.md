# ETH RL Market Agent V1

Clean-sheet ETH AI/RL research line. Historical `eth_ai_trading` strategies, Q70 rules, and latent-liquidity gates are not inherited.

## Programme gate

1. **R00** — build a causal multi-source market-state dataset and forward-path opportunity labels. No model, no trades.
2. **R01** — supervised opportunity model **immediately converted into an executable strategy** with entry/TP/SL/max-hold, sizing, costs and walk-forward backtest. Model metrics are secondary.
3. **R02** — only after R01 produces a tradable OOS strategy candidate: sequence challenger and/or offline policy learning for flat/long/short/size/hold/reduce/exit.
4. **R03** — portfolio hardening, realistic execution pressure, then explicit sealed-2026 final audit.

The programme is not an academic edge-search project. Every modelling stage must move toward a strategy that can be backtested, hardened, migrated to AetherEdge and traded live. RL is not allowed to manufacture alpha from an uninformative state.

## Frozen champion priority

Lexicographic, not a weighted score:

1. max flat days ↓
2. max consecutive losing days ↓
3. max drawdown ↓
4. CAGR ↑
5. total return ↑

Continuity metrics are selection criteria, not an instruction to force low-quality trades.
