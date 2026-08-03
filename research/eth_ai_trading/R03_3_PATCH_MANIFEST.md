# R03.3 Patch Manifest

新增未来市场过程预测MVP，不修改既有R03/R03.1/R03.2行为。

- `research/eth_ai_trading/03_3_future_process_forecast.py`
- `src/ai_research/future_process_forecast/`
- `tests/ai_research/test_future_process_forecast.py`
- `docs/ETH_AI_TRADING_R03_3_FUTURE_PROCESS_FORECAST_RUNBOOK.md`

本补丁只通过公共Trade Bar Loader读取1m和5s缓存，不读取Raw Trades，不修改`src.data_feed`，不触碰AetherEdge。
