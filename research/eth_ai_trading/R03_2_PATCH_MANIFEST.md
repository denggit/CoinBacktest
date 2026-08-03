# R03.2 Patch Manifest

## 目标

扩展高周期因果上下文与连续市场过程特征，重新验证 3%–5% Swing 开仓机会；不改标签、退出、模型参数和实盘代码。

## 新增

- `src/ai_research/swing_long_context/`
- `research/eth_ai_trading/03_2_swing_long_context.py`
- `docs/ETH_AI_TRADING_R03_2_LONG_CONTEXT_RUNBOOK.md`
- `tests/ai_research/test_swing_long_context.py`

## 修改

- `src/ai_research/swing_baseline/features.py`
- `src/ai_research/swing_baseline/dataset.py`
- `src/ai_research/swing_entry_mvp/pipeline.py`
- `src/ai_research/plan.py`
- `docs/ETH_AI_TRADING_RESEARCH_PLAN.md`
- `research/eth_ai_trading/README.md`

## 边界

- 只通过 `src.data_feed.OKXTradeBarLoader` 读取数据。
- 不直接读取 Raw Trades、ZIP 或 SQLite。
- 不修改其他策略、公共回测执行、AetherEdge 或交易所接口。
- 不执行 Git commit。
