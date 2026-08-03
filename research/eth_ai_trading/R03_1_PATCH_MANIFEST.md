# R03.1 Patch Manifest

## 新增

- `research/eth_ai_trading/03_1_swing_entry_mvp.py`
- `src/ai_research/swing_entry_mvp/`
- `docs/ETH_AI_TRADING_R03_1_SWING_ENTRY_MVP_RUNBOOK.md`
- `tests/ai_research/test_swing_entry_mvp.py`

## 复用与小幅重构

- `src/ai_research/swing_baseline/modeling.py`
  - 提取从已组装 `PeriodData` 训练模型的公共函数。
  - 原 R03 行为保持不变。
- `src/ai_research/sleeves/contracts.py`
- `src/ai_research/sleeves/registry.py`
- `src/ai_research/sleeves/artifacts.py`
  - Swing 明确为无最低持仓时间；目标是捕捉3%–5%，最长窗口仅作安全上限。
- `src/ai_research/plan.py`
- `docs/ETH_AI_TRADING_RESEARCH_PLAN.md`
- `docs/ETH_AI_TRADING_R03_SWING_BASELINE_RUNBOOK.md`
- `research/eth_ai_trading/README.md`
  - 补充 R03.1 阶段、运行入口和纠偏说明。
- `tests/ai_research/test_sleeves.py`
  - 固化 Swing 无最低持仓时间的公共合同。

## 明确未修改

- `src.data_feed`
- 现有1m/1s数据和数据库
- 其他 Research 的策略逻辑
- 现有策略
- AetherEdge
