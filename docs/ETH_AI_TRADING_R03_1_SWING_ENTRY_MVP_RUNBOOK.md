# ETH AI Trading R03.1：3%–5% Swing 开仓 MVP

## 目标

R03.1 不要求交易至少持有十几个小时。唯一目标是验证：

> 在未来72–120小时的最大研究窗口内，模型能否找到先达到3%或5%、且先不触发风险线的开仓机会。

目标可能在几十分钟、数小时或数天内完成。持仓时长是结果，不是入场条件。

## 相对 R03 的关键修复

1. **精确路径标签**：使用现有 R03 1m 缓存，按每根1m路径判断目标与风险线谁先触发。
2. **保守同 Bar 规则**：同一分钟同时触发目标与风险线，风险线优先。
3. **取消噪声退出**：不再使用15m趋势失效或模型反向信号强制平仓。
4. **多空独立验证**：允许最终只有 long-only 或 short-only MVP。
5. **无最低持仓时间**：目标触发即可退出。

## 出场规则

研究比较两套简单规则：

- `fixed_adverse_target`：固定风险线 + 3%/5%目标 + 最长窗口。
- `structural_protected_target`：4H结构风险（不得超过标签风险上限）+ 达到一半目标后锁定部分利润 + 最终目标。

这两套规则用于区分：模型是否能找到开仓机会，以及结构止损/利润保护是否破坏目标捕捉。

## 数据与缓存

- 历史数据只通过 `src.data_feed.OKXTradeBarLoader(timeframe="1m")` 获取。
- 复用 `data/cache/eth_ai_trading/r03_swing` 的多周期特征缓存。
- 新增的小型精确结果缓存位于：
  `data/cache/eth_ai_trading/r03_1_exact_outcomes`
- 不读取 Raw Trades、ZIP 或 SQLite，不重建1m数据。

## 运行

Windows 一行命令：

```text
python research\eth_ai_trading\03_1_swing_entry_mvp.py
```

默认报告目录：

```text
data\reports\research\eth_ai_trading\03_1_swing_entry_mvp
```

不要使用 `--force-rebuild-base-cache`，除非 R03 基础缓存本身发生了 schema 变化。

## 决策

- `PASS_SWING_ENTRY_MVP`：2024、2025和2026锁定样本外均通过，进入模型导出与AetherEdge影子推理。
- `FAIL_VALIDATION`：现有特征无法稳定识别3%–5%开仓机会，不再靠继续修改退出规则救模型。
- `FAIL_LOCKED_HOLDOUT`：2024/2025有候选但2026失效，不能实盘。
- `BLOCKED_PUBLIC_LOADER`：公共数据接口轻量预检失败。
