# Liquidity Hunt Momentum R01 Fix 1

## 修复内容

1. **真实 Range Loader 索引兼容**
   - `OKXRangeBarLoader.load_local_data()` 返回的 `end_ts` 同时是 index 和显式列。
   - 所有外部 Range / Books / Footprint DataFrame 在特征入口统一 `reset_index(drop=True)`。
   - 修复 `ValueError: 'end_ts' is both an index level and a column label`。

2. **严格 Range-Bar 入场时序**
   - 原实现直接使用下一根 Range Bar open。
   - 真实 trades 可能共享同一毫秒，导致下一根 `start_ts == signal_time`，无法证明该 open 在信号之后。
   - 现在只使用第一根 `start_ts > signal_time` 的 Range Bar open；延迟 2/3 bars 从该位置继续向后计算。

3. **Books datetime dtype**
   - Books 缺失或过期时，`book_available_time` 始终保持 `datetime64[ns]`，不再向 datetime 列写入 object NaN。
   - 在 `FutureWarning` 作为错误的模式下通过。

4. **因果审计语义**
   - Books/Footprint 缺失属于数据质量问题，不再误报为未来函数。
   - `data_missing_flag` 与 `causal_fail_flag` 分离。

5. **下一开盘退出时间**
   - 动量衰减/时间退出若在下一根 open 成交，`exit_time` 记录该 bar 的 `start_ts`，不再错误记录为 `end_ts`。

## 验证

- 内置 self-test：通过。
- 新增研究单测：15 passed。
- 公共 loader / liquidity store / review-pack 回归：13 passed。
- 使用 CoinBacktest 压缩包内真实 `okx_range_bars.db`，按 2025-10-01 至 2026-06-30、r0015/r0020/r0025 跑完整 Range 入口：通过。
- 上述真实数据库验证在 `PYTHONWARNINGS=error::FutureWarning` 下通过。
- 因压缩包不含离线 Books 分区和 Range-Footprint DB，该验证只确认真实 Range Loader、完整窗口、事件/模拟/审计路径，不代表真实策略收益结果。
