# R03.4.2 Patch Manifest

## Added

- `src/ai_research/long_tail_exit_audit/`
- `research/eth_ai_trading/03_4_2_long_tail_exit_audit.py`
- `docs/ETH_AI_TRADING_R03_4_2_LONG_TAIL_EXIT_AUDIT_RUNBOOK.md`
- `tests/ai_research/test_long_tail_exit_audit.py`

## Contract

- Freeze R03.4.1 base q90 long model.
- Remove all market-state inputs from the trading chain.
- Use causal one-minute paths from `src.data_feed.okx_trade_bar_loader.OKXTradeBarLoader`.
- Compare structural stops, fixed-R exits, trailing profit protection, rolling score renewal, and confirmed model invalidation.
- Positive expectancy is the primary gate; no single metric may be improved by making either OOS year negative.
- Keep 2026 sealed.
