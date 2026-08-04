# R03.3.1 Patch Manifest

## 新增

- `research/eth_ai_trading/03_3_1_process_alert_value_audit.py`
- `src/ai_research/future_process_forecast/alert_audit_config.py`
- `src/ai_research/future_process_forecast/alert_audit.py`
- `src/ai_research/future_process_forecast/alert_audit_pipeline.py`
- `src/ai_research/future_process_forecast/alert_audit_reports.py`
- `tests/ai_research/test_future_process_alert_audit.py`
- `docs/ETH_AI_TRADING_R03_3_1_ALERT_VALUE_AUDIT_RUNBOOK.md`

## 更新

- `research/eth_ai_trading/README.md`
- `docs/ETH_AI_TRADING_RESEARCH_PLAN.md`

## 边界

- 不修改R03.3模型、标签或既有缓存。
- 不读取Raw Trades，不重建缺失微观数据。
- 不打开2026H1。
- 默认复用R03.2长期特征、R03.3事件与5s微观缓存。
